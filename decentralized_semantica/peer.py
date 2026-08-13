from __future__ import annotations

import asyncio
import json
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .algorithms import (
    assign_unique_candidates,
    child_node_id,
    choose_child,
    custodian_for,
    fit_peer_profile,
    local_topk,
    merge_topk,
    normalize_rows,
    peer_centroid_score,
    replace_centroid_contacts,
    root_node_id,
    split_entries,
    stable_int,
)
from .dataset import load_shard
from .models import (
    ChildDescriptor,
    Contact,
    CentroidEntry,
    Endpoint,
    InternalNode,
    LeafNode,
    MembershipManifest,
    PeerProfile,
    QueryState,
    TreeNode,
    contacts_to_wire,
    node_from_wire,
)
from .protocol import (
    ProtocolError,
    RpcClient,
    envelope,
    pack_message,
    read_frame,
    validate_envelope,
    write_frame,
)


RESPONSE_TYPES = {
    "CONFIGURE": "CONFIGURED",
    "PROFILE_READY": "PROFILE_READY_ACK",
    "BUILD_TURN": "BUILD_DONE",
    "TREE_NODE_GET": "TREE_NODE_REPLY",
    "CENTROID_INSERT": "CENTROID_INSERT_ACK",
    "CREATE_LEAF": "CREATE_LEAF_ACK",
    "TREE_FROZEN": "TREE_FROZEN_ACK",
    "ROUND0_START": "ROUND0_DONE",
    "LEAF_SCAN": "LEAF_SCAN_REPLY",
    "TREE_ADJACENCY_GET": "TREE_ADJACENCY_REPLY",
    "PROFILE_GET": "PROFILE_REPLY",
    "EXPANSION_START": "EXPANSION_DONE",
    "EXPANSION_REQUEST": "EXPANSION_REPLY",
    "QUERY_START": "QUERY_COMPLETE",
    "QUERY_HOP": "QUERY_HOP_ACK",
    "QUERY_RESULT": "QUERY_RESULT_ACK",
    "QUERY_DONE": "QUERY_DONE_ACK",
    "EXPORT_STATE": "EXPORT_STATE_ACK",
    "SHUTDOWN": "SHUTDOWN_ACK",
}


class PeerRuntime:
    def __init__(
        self,
        peer_id: int,
        shard_path: Path,
        run_dir: Path,
        endpoint_queue: Any,
        *,
        host: str = "127.0.0.1",
    ) -> None:
        self.peer_id = int(peer_id)
        self.shard_path = Path(shard_path)
        self.run_dir = Path(run_dir)
        self.endpoint_queue = endpoint_queue
        self.host = str(host)
        self.endpoint = Endpoint(self.host, 0)
        self.manifest: Optional[MembershipManifest] = None
        self.epoch = "unconfigured"
        self.parameters: Dict[str, Any] = {}
        self.shard = load_shard(self.shard_path)
        self.profile: Optional[PeerProfile] = None
        self.profile_cache: Dict[int, PeerProfile] = {}
        self.nodes: Dict[str, TreeNode] = {}
        self.node_locks: Dict[str, asyncio.Lock] = {}
        self.contacts_by_round: Dict[int, List[List[Contact]]] = {}
        self.initial_contact_sets: List[set[int]] = []
        self.asked_referrers: List[set[int]] = []
        self.query_states: Dict[str, QueryState] = {}
        self.query_events: Dict[str, asyncio.Event] = {}
        self.profile_ready: set[int] = set()
        self.phase = "STARTING"
        self.frozen = False
        self.stop_event = asyncio.Event()
        self.coordinator_task: Optional[asyncio.Task[None]] = None
        self.phase_timings: Dict[str, float] = {}
        self.server: Optional[asyncio.AbstractServer] = None
        self.profile_task: Optional[asyncio.Task[None]] = None
        log_dir = self.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = (log_dir / f"peer_{self.peer_id:05d}.jsonl").open(
            "a", encoding="utf-8"
        )

    @property
    def root_id(self) -> str:
        if self.manifest is None:
            raise RuntimeError("Peer has not been configured")
        return root_node_id(self.manifest.epoch)

    @property
    def root_peer_id(self) -> int:
        if self.manifest is None:
            raise RuntimeError("Peer has not been configured")
        return custodian_for(self.root_id, self.manifest)

    def _log(self, direction: str, message: Mapping[str, Any], target: Endpoint, size: int) -> None:
        payload = message.get("payload", {})
        source_endpoint = Endpoint.from_wire(message["source_endpoint"])
        if direction == "send":
            destination_endpoint = target
            destination_peer_id = self._peer_id_for_endpoint(target)
        else:
            destination_endpoint = self.endpoint
            destination_peer_id = self.peer_id
        row = {
            "time": time.time(),
            "monotonic": time.monotonic(),
            "peer_id": self.peer_id,
            "direction": direction,
            "remote": target.to_wire(),
            "source_peer_id": int(message["source_peer_id"]),
            "source_endpoint": source_endpoint.to_wire(),
            "destination_peer_id": destination_peer_id,
            "destination_endpoint": destination_endpoint.to_wire(),
            "message_type": message.get("message_type"),
            "message_id": message.get("message_id"),
            "request_id": message.get("request_id"),
            "bytes": int(size),
            "payload_bytes": len(pack_message(payload)),
            "phase": self.phase,
            "round": payload.get("round"),
            "query_id": payload.get("query_id"),
            "hop": payload.get("hop"),
        }
        self.log_handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.log_handle.flush()

    def _peer_id_for_endpoint(self, endpoint: Endpoint) -> Optional[int]:
        if endpoint == self.endpoint:
            return self.peer_id
        if self.manifest is None:
            return None
        return next(
            (
                peer_id
                for peer_id, member_endpoint in self.manifest.peers.items()
                if member_endpoint == endpoint
            ),
            None,
        )

    def _rpc_client(self) -> RpcClient:
        return RpcClient(
            source_peer_id=self.peer_id,
            source_endpoint=self.endpoint,
            epoch=self.epoch,
            timeout=float(self.parameters.get("rpc_timeout_seconds", 30.0)),
            log=self._log,
        )

    async def rpc(self, peer_id: int, message_type: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if self.manifest is None:
            raise RuntimeError("Peer has not been configured")
        return await self._rpc_client().call(
            self.manifest.peers[int(peer_id)], message_type, payload
        )

    async def rpc_endpoint(
        self, endpoint: Endpoint, message_type: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return await self._rpc_client().call(endpoint, message_type, payload)

    async def serve(self) -> None:
        self.server = await asyncio.start_server(self._handle_client, self.host, 0)
        socket = self.server.sockets[0]
        self.endpoint = Endpoint(str(socket.getsockname()[0]), int(socket.getsockname()[1]))
        self.endpoint_queue.put((self.peer_id, self.endpoint.host, self.endpoint.port))
        async with self.server:
            await self.stop_event.wait()
        self.server.close()
        await self.server.wait_closed()
        self.log_handle.close()
        # Await retained background tasks so fatal setup/coordinator failures
        # become non-zero process exits that the launcher can detect promptly.
        if self.profile_task is not None:
            await self.profile_task
        if self.coordinator_task is not None:
            await self.coordinator_task

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        remote = writer.get_extra_info("peername")
        remote_endpoint = Endpoint(str(remote[0]), int(remote[1])) if remote else Endpoint("unknown", 0)
        request: Optional[Dict[str, Any]] = None
        try:
            request, size = await read_frame(reader)
            expected_epoch = None if request.get("message_type") == "CONFIGURE" else self.epoch
            validate_envelope(request, expected_epoch)
            remote_endpoint = Endpoint.from_wire(request["source_endpoint"])
            self._log("receive", request, remote_endpoint, size)
            response_type, response_payload = await self._dispatch(request)
            response = envelope(
                response_type,
                epoch=str(request["epoch"]),
                source_peer_id=self.peer_id,
                source_endpoint=self.endpoint,
                payload=response_payload,
                request_id=str(request["message_id"]),
            )
        except Exception as exc:
            error = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            response = envelope(
                "ERROR",
                epoch=str(request.get("epoch", self.epoch) if request else self.epoch),
                source_peer_id=self.peer_id,
                source_endpoint=self.endpoint,
                payload=error,
                request_id=str(request.get("message_id")) if request else None,
            )
            (self.run_dir / "errors").mkdir(parents=True, exist_ok=True)
            with (self.run_dir / "errors" / f"peer_{self.peer_id:05d}.log").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(error["traceback"] + "\n")
        try:
            sent = await write_frame(writer, response)
            self._log("send", response, remote_endpoint, sent)
        finally:
            writer.close()
            await writer.wait_closed()

    async def _dispatch(self, request: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
        message_type = str(request["message_type"])
        if message_type not in RESPONSE_TYPES:
            raise ProtocolError(f"Unknown message type: {message_type}")
        handler = getattr(self, f"_on_{message_type.lower()}")
        payload = await handler(dict(request["payload"]), request)
        if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], str):
            return payload
        return RESPONSE_TYPES[message_type], dict(payload or {})

    async def _on_configure(self, payload: Dict[str, Any], _request: Mapping[str, Any]) -> Dict[str, Any]:
        if self.manifest is not None:
            return {"peer_id": self.peer_id, "already_configured": True}
        self.manifest = MembershipManifest.from_wire(payload["membership"])
        self.epoch = self.manifest.epoch
        self.parameters = dict(self.manifest.parameters)
        if self.peer_id not in self.manifest.peers or self.manifest.peers[self.peer_id] != self.endpoint:
            raise ProtocolError("Membership endpoint does not match listening endpoint")
        if self.peer_id == self.root_peer_id:
            self.nodes[self.root_id] = LeafNode(
                node_id=self.root_id, version=1, depth=0, parent=None
            )
        self.phase = "PROFILE"
        self.profile_task = asyncio.create_task(self._initialize_profile())
        self.profile_task.add_done_callback(self._profile_task_finished)
        return {"peer_id": self.peer_id, "root_peer_id": self.root_peer_id}

    def _profile_task_finished(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exception = task.exception()
        if exception is None:
            return
        (self.run_dir / "errors").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "errors" / f"peer_{self.peer_id:05d}_fatal.log").write_text(
            "".join(traceback.format_exception(exception)), encoding="utf-8"
        )
        self.stop_event.set()

    async def _initialize_profile(self) -> None:
        start = time.monotonic()
        self.profile = await asyncio.to_thread(
            fit_peer_profile,
            self.peer_id,
            self.endpoint,
            np.asarray(self.shard["training_embeddings"], dtype=np.float32),
            degree=int(self.parameters["degree"]),
            seed=int(self.parameters["seed"]),
            max_k=int(self.parameters["max_centroids"]),
        )
        self.profile_cache[self.peer_id] = self.profile
        self.phase_timings["profile_seconds"] = time.monotonic() - start
        await self.rpc(
            self.root_peer_id,
            "PROFILE_READY",
            {"peer_id": self.peer_id, "centroid_count": len(self.profile.centroids)},
        )

    async def _on_profile_ready(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        if self.peer_id != self.root_peer_id:
            raise ProtocolError("PROFILE_READY was not sent to the root custodian")
        self.profile_ready.add(int(payload["peer_id"]))
        if len(self.profile_ready) == len(self.manifest.peers) and self.coordinator_task is None:
            self.coordinator_task = asyncio.create_task(self._coordinate_protocol())
        return {"ready_count": len(self.profile_ready)}

    async def _coordinate_protocol(self) -> None:
        assert self.manifest is not None
        try:
            self.phase = "TREE_BUILD"
            started = time.monotonic()
            for peer_id in sorted(self.manifest.peers):
                await self.rpc(peer_id, "BUILD_TURN", {"peer_id": peer_id})
            self.phase_timings["tree_build_seconds"] = time.monotonic() - started
            frozen = await asyncio.gather(
                *(self.rpc(peer_id, "TREE_FROZEN", {}) for peer_id in sorted(self.manifest.peers))
            )
            maximum_tree_depth = max(
                int(response["payload"]["maximum_local_depth"]) for response in frozen
            )
            self.phase = "ROUND0"
            started = time.monotonic()
            await asyncio.gather(
                *(
                    self.rpc(
                        peer_id,
                        "ROUND0_START",
                        {"maximum_tree_depth": maximum_tree_depth},
                    )
                    for peer_id in sorted(self.manifest.peers)
                )
            )
            self.phase_timings["round0_seconds"] = time.monotonic() - started
            for round_index in range(1, int(self.parameters["expansion_rounds"]) + 1):
                self.phase = f"EXPANSION_{round_index}"
                started = time.monotonic()
                await asyncio.gather(
                    *(
                        self.rpc(peer_id, "EXPANSION_START", {"round": round_index})
                        for peer_id in sorted(self.manifest.peers)
                    )
                )
                self.phase_timings[f"expansion_round_{round_index}_seconds"] = (
                    time.monotonic() - started
                )
            self.phase = "QUERY"
            started = time.monotonic()
            await asyncio.gather(
                *(self.rpc(peer_id, "QUERY_START", {}) for peer_id in sorted(self.manifest.peers))
            )
            self.phase_timings["query_seconds"] = time.monotonic() - started
            self.phase = "EXPORT"
            await asyncio.gather(
                *(self.rpc(peer_id, "EXPORT_STATE", {}) for peer_id in sorted(self.manifest.peers))
            )
            (self.run_dir / "run_complete.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "root_peer_id": self.peer_id,
                        "epoch": self.epoch,
                        "phase_timings": self.phase_timings,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            for peer_id in sorted(self.manifest.peers):
                if peer_id != self.peer_id:
                    await self.rpc(peer_id, "SHUTDOWN", {})
            self.stop_event.set()
        except Exception:
            (self.run_dir / "root_failure.log").write_text(traceback.format_exc(), encoding="utf-8")
            self.stop_event.set()
            raise

    async def _on_build_turn(self, payload: Dict[str, Any], _request: Mapping[str, Any]) -> Dict[str, Any]:
        if int(payload["peer_id"]) != self.peer_id or self.profile is None:
            raise ProtocolError("Invalid build turn")
        self.phase = "TREE_BUILD"
        for centroid_index, vector in enumerate(self.profile.centroids):
            entry = CentroidEntry(
                peer_id=self.peer_id,
                centroid_index=centroid_index,
                vector=vector,
                profile_version=self.profile.version,
                endpoint=self.endpoint,
            )
            await self._insert_entry(entry)
        return {"peer_id": self.peer_id, "inserted": len(self.profile.centroids)}

    def _descriptor(self, node_id: str, center: Optional[np.ndarray] = None) -> ChildDescriptor:
        assert self.manifest is not None
        custodian = custodian_for(node_id, self.manifest)
        return ChildDescriptor(node_id, custodian, self.manifest.peers[custodian], center=center)

    async def _get_node(self, descriptor: ChildDescriptor) -> TreeNode:
        response = await self.rpc_endpoint(
            descriptor.custodian_endpoint,
            "TREE_NODE_GET",
            {"node_id": descriptor.node_id},
        )
        return node_from_wire(response["payload"]["node"])

    async def _locate_leaf(self, vector: np.ndarray) -> tuple[ChildDescriptor, LeafNode]:
        descriptor = self._descriptor(self.root_id)
        while True:
            node = await self._get_node(descriptor)
            if isinstance(node, LeafNode):
                return descriptor, node
            side = choose_child(vector, node.left.center, node.right.center)
            descriptor = node.left if side == "left" else node.right

    async def _insert_entry(self, entry: CentroidEntry) -> None:
        descriptor, _leaf = await self._locate_leaf(entry.vector)
        while True:
            response = await self.rpc_endpoint(
                descriptor.custodian_endpoint,
                "CENTROID_INSERT",
                {"node_id": descriptor.node_id, "entry": entry.to_wire()},
            )
            if response["message_type"] != "TREE_REDIRECT":
                return
            node = InternalNode.from_wire(response["payload"]["node"])
            side = choose_child(entry.vector, node.left.center, node.right.center)
            descriptor = node.left if side == "left" else node.right

    async def _on_tree_node_get(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        node = self.nodes.get(str(payload["node_id"]))
        if node is None:
            raise ProtocolError("Tree node is not held by this custodian")
        return {"node": node.to_wire(include_entries=False) if isinstance(node, LeafNode) else node.to_wire()}

    async def _on_centroid_insert(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Mapping[str, Any] | tuple[str, Mapping[str, Any]]:
        if self.frozen:
            raise ProtocolError("Tree is frozen")
        node_id = str(payload["node_id"])
        lock = self.node_locks.setdefault(node_id, asyncio.Lock())
        async with lock:
            node = self.nodes.get(node_id)
            if isinstance(node, InternalNode):
                return "TREE_REDIRECT", {"node": node.to_wire()}
            if not isinstance(node, LeafNode):
                raise ProtocolError("Insertion target is not a known leaf")
            entry = CentroidEntry.from_wire(payload["entry"])
            if all(existing.entry_id != entry.entry_id for existing in node.entries):
                node.entries.append(entry)
                node.version += 1
            if self._should_split(node):
                await self._split_leaf(node)
            return {"node_id": node_id, "entry_id": entry.entry_id}

    def _should_split(self, node: LeafNode) -> bool:
        return (
            len(node.entries) > int(self.parameters["leaf_capacity"])
            and node.failed_split_size != len(node.entries)
        )

    async def _split_leaf(self, node: LeafNode) -> None:
        result = split_entries(
            node.entries,
            seed=int(self.parameters["seed"]),
            delta=float(self.parameters["delta"]),
        )
        if result is None:
            node.failed_split_size = len(node.entries)
            return
        left_center, right_center, left_entries, right_entries = result
        left_id = child_node_id(self.epoch, node.node_id, "left")
        right_id = child_node_id(self.epoch, node.node_id, "right")
        parent_descriptor = self._descriptor(node.node_id)
        left = self._descriptor(left_id, left_center)
        right = self._descriptor(right_id, right_center)
        await asyncio.gather(
            self.rpc_endpoint(
                left.custodian_endpoint,
                "CREATE_LEAF",
                {
                    "node_id": left_id,
                    "depth": node.depth + 1,
                    "parent": parent_descriptor.to_wire(),
                    "parent_entry_count": len(node.entries),
                    "entries": [entry.to_wire() for entry in left_entries],
                },
            ),
            self.rpc_endpoint(
                right.custodian_endpoint,
                "CREATE_LEAF",
                {
                    "node_id": right_id,
                    "depth": node.depth + 1,
                    "parent": parent_descriptor.to_wire(),
                    "parent_entry_count": len(node.entries),
                    "entries": [entry.to_wire() for entry in right_entries],
                },
            ),
        )
        self.nodes[node.node_id] = InternalNode(
            node_id=node.node_id,
            version=node.version + 1,
            depth=node.depth,
            parent=node.parent,
            left=left,
            right=right,
        )

    async def _on_create_leaf(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        node_id = str(payload["node_id"])
        lock = self.node_locks.setdefault(node_id, asyncio.Lock())
        async with lock:
            if node_id in self.nodes:
                return {"node_id": node_id, "already_exists": True}
            node = LeafNode(
                node_id=node_id,
                version=1,
                depth=int(payload["depth"]),
                parent=ChildDescriptor.from_wire(payload["parent"]),
                entries=[CentroidEntry.from_wire(value) for value in payload["entries"]],
            )
            self.nodes[node_id] = node
            if (
                len(node.entries) > int(self.parameters["leaf_capacity"])
                and len(node.entries) >= int(payload["parent_entry_count"])
            ):
                # Delta cloning can produce a child with no capacity
                # reduction.  Match the centralized eager-split guard and
                # diagnose the stable overflow instead of recursing forever.
                node.failed_split_size = len(node.entries)
            elif self._should_split(node):
                await self._split_leaf(node)
            return {"node_id": node_id, "entry_count": len(node.entries)}

    async def _on_tree_frozen(
        self, _payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self.frozen = True
        self.phase = "TREE_FROZEN"
        return {
            "peer_id": self.peer_id,
            "node_count": len(self.nodes),
            "maximum_local_depth": max(
                (int(node.depth) for node in self.nodes.values()), default=0
            ),
        }

    async def _on_leaf_scan(self, payload: Dict[str, Any], _request: Mapping[str, Any]) -> Dict[str, Any]:
        node = self.nodes.get(str(payload["node_id"]))
        if not isinstance(node, LeafNode):
            raise ProtocolError("LEAF_SCAN target is not a leaf")
        return {"node_id": node.node_id, "entries": [entry.to_wire() for entry in node.entries]}

    async def _on_tree_adjacency_get(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        node = self.nodes.get(str(payload["node_id"]))
        if node is None:
            raise ProtocolError("Unknown adjacency node")
        neighbors: List[ChildDescriptor] = []
        if node.parent is not None:
            neighbors.append(node.parent)
        if isinstance(node, InternalNode):
            neighbors.extend([node.left, node.right])
        return {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "neighbors": [neighbor.to_wire() for neighbor in neighbors],
        }

    async def _on_profile_get(self, _payload: Dict[str, Any], _request: Mapping[str, Any]) -> Dict[str, Any]:
        if self.profile is None:
            raise ProtocolError("Profile is not ready")
        return {"profile": self.profile.to_wire()}

    async def _profile_for(self, peer_id: int) -> PeerProfile:
        peer_id = int(peer_id)
        if peer_id in self.profile_cache:
            return self.profile_cache[peer_id]
        response = await self.rpc(peer_id, "PROFILE_GET", {})
        profile = PeerProfile.from_wire(response["payload"]["profile"])
        self.profile_cache[peer_id] = profile
        return profile

    async def _on_round0_start(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self.phase = "ROUND0"
        await self._build_round0(int(payload["maximum_tree_depth"]))
        return {"peer_id": self.peer_id, "degree": sum(len(v) for v in self.contacts_by_round[0])}

    async def _advance_frontier(
        self,
        centroid_index: int,
        frontier: List[ChildDescriptor],
        visited: set[str],
        candidate_scores: List[Dict[int, float]],
    ) -> List[ChildDescriptor]:
        owners: set[int] = set()
        next_frontier: List[ChildDescriptor] = []
        for descriptor in frontier:
            node = await self._get_node(descriptor)
            if isinstance(node, LeafNode):
                response = await self.rpc_endpoint(
                    descriptor.custodian_endpoint, "LEAF_SCAN", {"node_id": descriptor.node_id}
                )
                owners.update(
                    int(value["peer_id"])
                    for value in response["payload"]["entries"]
                    if int(value["peer_id"]) != self.peer_id
                )
            response = await self.rpc_endpoint(
                descriptor.custodian_endpoint,
                "TREE_ADJACENCY_GET",
                {"node_id": descriptor.node_id},
            )
            for raw in response["payload"]["neighbors"]:
                neighbor = ChildDescriptor.from_wire(raw)
                if neighbor.node_id not in visited:
                    visited.add(neighbor.node_id)
                    next_frontier.append(neighbor)
        source = self.profile.centroids[centroid_index]
        for owner in sorted(owners.difference(candidate_scores[centroid_index])):
            profile = await self._profile_for(owner)
            candidate_scores[centroid_index][owner] = peer_centroid_score(source, profile)
        return next_frontier

    async def _build_round0(self, maximum_tree_depth: int) -> None:
        assert self.profile is not None
        start_descriptors: List[ChildDescriptor] = []
        for vector in self.profile.centroids:
            descriptor, _leaf = await self._locate_leaf(vector)
            start_descriptors.append(descriptor)
        frontiers = [[descriptor] for descriptor in start_descriptors]
        visited = [{descriptor.node_id} for descriptor in start_descriptors]
        candidate_scores: List[Dict[int, float]] = [dict() for _ in start_descriptors]
        caps = self.profile.contact_caps.tolist()
        active = list(range(len(frontiers)))
        assigned: List[List[Tuple[int, float]]] = [[] for _ in frontiers]
        searched_distance = np.full(len(frontiers), -1, dtype=np.int32)
        maximum_distance = max(0, 4 * int(maximum_tree_depth))
        while active:
            for index in active:
                searched_distance[index] += 1
                frontiers[index] = await self._advance_frontier(
                    index, frontiers[index], visited[index], candidate_scores
                )
            assigned, underfilled = assign_unique_candidates(candidate_scores, caps)
            if not underfilled:
                break
            active = [
                index
                for index in underfilled
                if frontiers[index] and int(searched_distance[index]) < maximum_distance
            ]
            if not active:
                missing = {index: int(caps[index] - len(assigned[index])) for index in underfilled}
                raise RuntimeError(f"Unable to fill round-0 degree for peer {self.peer_id}: {missing}")
        contacts: List[List[Contact]] = []
        for values in assigned:
            centroid_contacts = []
            for candidate, score in values:
                profile = await self._profile_for(candidate)
                centroid_contacts.append(
                    Contact(candidate, score, profile.endpoint, profile.version, "round0")
                )
            contacts.append(centroid_contacts)
        if sum(len(values) for values in contacts) != int(self.parameters["degree"]):
            raise RuntimeError("Round-0 contact degree is incomplete")
        self.contacts_by_round[0] = contacts
        self.initial_contact_sets = [
            {contact.peer_id for contact in values} for values in contacts
        ]
        self.asked_referrers = [set() for _ in contacts]

    async def _on_expansion_request(
        self, payload: Dict[str, Any], request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        round_index = int(payload["round"])
        if round_index not in self.contacts_by_round:
            raise ProtocolError(f"Contact snapshot {round_index} is unavailable")
        source_peer_id = int(request["source_peer_id"])
        peers = sorted(
            {
                contact.peer_id
                for values in self.contacts_by_round[round_index]
                for contact in values
                if contact.peer_id != source_peer_id
            }
        )
        # The referrer advertises only its frozen round-start outgoing IDs.
        # Candidate endpoint/version discovery belongs to the requester and is
        # performed directly through PROFILE_GET when its cache misses.
        return {"round": round_index, "peer_ids": peers}

    async def _on_expansion_start(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        round_index = int(payload["round"])
        self.phase = f"EXPANSION_{round_index}"
        await self._expand_round(round_index)
        return {"peer_id": self.peer_id, "round": round_index}

    async def _expand_round(self, round_index: int) -> None:
        previous = self.contacts_by_round[round_index - 1]
        current = [[deepcopy(contact) for contact in values] for values in previous]
        owner_existing = {contact.peer_id for values in previous for contact in values}
        proposals: List[List[Tuple[int, float]]] = [[] for _ in current]
        for centroid_index, values in enumerate(previous):
            ids = [contact.peer_id for contact in values]
            new_tier = [
                candidate
                for candidate in ids
                if candidate not in self.initial_contact_sets[centroid_index]
                and candidate not in self.asked_referrers[centroid_index]
            ]
            initial_tier = [
                candidate
                for candidate in ids
                if candidate in self.initial_contact_sets[centroid_index]
                and candidate not in self.asked_referrers[centroid_index]
            ]
            tier = new_tier if new_tier else initial_tier
            if not tier:
                continue
            rng = np.random.default_rng(
                stable_int(
                    f"semantic-referrer:{self.parameters['seed']}:{round_index}:"
                    f"{self.peer_id}:{centroid_index}"
                )
            )
            referrer = int(tier[int(rng.integers(0, len(tier)))])
            self.asked_referrers[centroid_index].add(referrer)
            response = await self.rpc(
                referrer,
                "EXPANSION_REQUEST",
                {"round": round_index - 1},
            )
            candidates = [
                int(candidate)
                for candidate in response["payload"]["peer_ids"]
                if int(candidate) != self.peer_id
                and int(candidate) not in owner_existing
            ]
            for candidate in candidates:
                profile = await self._profile_for(candidate)
                score = peer_centroid_score(self.profile.centroids[centroid_index], profile)
                if np.isfinite(score):
                    proposals[centroid_index].append((candidate, score))
        for centroid_index, values in enumerate(proposals):
            previous_contacts = {
                contact.peer_id: contact for contact in current[centroid_index]
            }
            excluded = {self.peer_id}
            excluded.update(
                contact.peer_id
                for other_index, contacts in enumerate(current)
                if other_index != centroid_index
                for contact in contacts
            )
            pairs = replace_centroid_contacts(
                [(contact.peer_id, contact.score) for contact in current[centroid_index]],
                values,
                int(self.profile.contact_caps[centroid_index]),
                excluded,
            )
            current[centroid_index] = [
                Contact(
                    candidate,
                    score,
                    (await self._profile_for(candidate)).endpoint,
                    (await self._profile_for(candidate)).version,
                    (
                        previous_contacts[candidate].learned_at
                        if candidate in previous_contacts
                        else f"expansion_{round_index}"
                    ),
                )
                for candidate, score in pairs
            ]
        self.contacts_by_round[round_index] = current

    def _best_query_neighbor(
        self, query: np.ndarray, visited: Iterable[int]
    ) -> Optional[int]:
        final_round = int(self.parameters["expansion_rounds"])
        visited_set = {int(value) for value in visited}
        candidates = sorted(
            {
                contact.peer_id
                for values in self.contacts_by_round[final_round]
                for contact in values
                if contact.peer_id not in visited_set
            }
        )
        if not candidates:
            return None
        # QUERY_START normalizes once with the centralized benchmark's 2-D
        # row kernel. Preserve those exact float32 bits across every hop.
        normalized_query = np.asarray(query, dtype=np.float32)
        ranked = []
        for candidate in candidates:
            profile = self.profile_cache[candidate]
            ranked.append((-peer_centroid_score(normalized_query, profile), candidate))
        ranked.sort()
        return int(ranked[0][1])

    async def _on_query_start(
        self, _payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self.phase = "QUERY"
        query_id = str(self.shard["query_id"].item())
        state = QueryState(
            query_id=query_id,
            query_vector=normalize_rows(
                np.asarray(self.shard["query_embedding"], dtype=np.float32)[None, :]
            )[0],
            target_doc_id=str(self.shard["target_doc_id"].item()),
        )
        self.query_states[query_id] = state
        event = asyncio.Event()
        self.query_events[query_id] = event
        next_peer = self._best_query_neighbor(state.query_vector, {self.peer_id})
        if next_peer is None:
            state.terminal_hop = 0
            state.terminal_reason = "dead_end"
            event.set()
        else:
            await self.rpc(
                next_peer,
                "QUERY_HOP",
                {
                    "query_id": query_id,
                    "source_peer_id": self.peer_id,
                    "source_endpoint": self.endpoint.to_wire(),
                    "query_embedding": state.query_vector,
                    "visited": [self.peer_id],
                    "hop": 1,
                    "hop_limit": int(self.parameters["query_hop_limit"]),
                },
            )
        await event.wait()
        for hop in sorted(state.results_by_hop):
            state.final_results = merge_topk(
                state.final_results,
                state.results_by_hop[hop],
                int(self.parameters["local_top_k"]),
            )
        return state.to_wire()

    async def _on_query_hop(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        query_id = str(payload["query_id"])
        query = np.asarray(payload["query_embedding"], dtype=np.float32)
        source_peer = int(payload["source_peer_id"])
        hop = int(payload["hop"])
        visited = [int(value) for value in payload["visited"]]
        local = local_topk(
            query,
            self.shard["training_doc_ids"].tolist(),
            np.asarray(self.shard["training_embeddings"], dtype=np.float32),
            int(self.parameters["local_top_k"]),
            query_is_normalized=True,
        )
        await self.rpc(
            source_peer,
            "QUERY_RESULT",
            {
                "query_id": query_id,
                "hop": hop,
                "peer_id": self.peer_id,
                "results": [[doc_id, score] for doc_id, score in local],
            },
        )
        visited.append(self.peer_id)
        if hop >= int(payload["hop_limit"]):
            await self.rpc(
                source_peer,
                "QUERY_DONE",
                {"query_id": query_id, "hop": hop, "reason": "hop_limit"},
            )
            return {"query_id": query_id, "hop": hop, "terminal": True}
        next_peer = self._best_query_neighbor(query, visited)
        if next_peer is None:
            await self.rpc(
                source_peer,
                "QUERY_DONE",
                {"query_id": query_id, "hop": hop, "reason": "exhausted_neighbors"},
            )
            return {"query_id": query_id, "hop": hop, "terminal": True}
        await self.rpc(
            next_peer,
            "QUERY_HOP",
            {**payload, "visited": visited, "hop": hop + 1},
        )
        return {"query_id": query_id, "hop": hop, "terminal": False}

    async def _on_query_result(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        state = self.query_states[str(payload["query_id"])]
        hop = int(payload["hop"])
        state.peer_by_hop[hop] = int(payload["peer_id"])
        state.results_by_hop[hop] = [
            (str(doc_id), float(score)) for doc_id, score in payload["results"]
        ]
        return {"query_id": state.query_id, "hop": hop}

    async def _on_query_done(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        query_id = str(payload["query_id"])
        state = self.query_states[query_id]
        state.terminal_hop = int(payload["hop"])
        state.terminal_reason = str(payload["reason"])
        self.query_events[query_id].set()
        return {"query_id": query_id}

    async def _on_export_state(
        self, _payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        state_dir = self.run_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "peer_id": self.peer_id,
            "endpoint": self.endpoint.to_wire(),
            "profile": self.profile.to_wire() if self.profile else None,
            "nodes": {
                node_id: (
                    node.to_wire(include_entries=True)
                    if isinstance(node, LeafNode)
                    else node.to_wire()
                )
                for node_id, node in self.nodes.items()
            },
            "contacts_by_round": {
                str(round_index): contacts_to_wire(contacts)
                for round_index, contacts in sorted(self.contacts_by_round.items())
            },
            "queries": {query_id: state.to_wire() for query_id, state in self.query_states.items()},
            "phase_timings": self.phase_timings,
        }
        path = state_dir / f"peer_{self.peer_id:05d}.msgpack"
        path.write_bytes(pack_message(payload))
        return {"peer_id": self.peer_id, "state_path": str(path)}

    async def _on_shutdown(
        self, _payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        asyncio.get_running_loop().call_later(0.05, self.stop_event.set)
        return {"peer_id": self.peer_id}


def run_peer_process(
    peer_id: int,
    shard_path: str,
    run_dir: str,
    endpoint_queue: Any,
) -> None:
    runtime = PeerRuntime(peer_id, Path(shard_path), Path(run_dir), endpoint_queue)
    asyncio.run(runtime.serve())

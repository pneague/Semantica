from __future__ import annotations

import asyncio
import contextlib
import json
import signal
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
    custody_proof_for_entries,
    fit_peer_profile,
    genesis_custody_proof,
    local_topk,
    merge_topk,
    normalize_rows,
    peer_centroid_score,
    replace_centroid_contacts,
    root_node_id,
    split_entries,
    stable_int,
    verify_custody_proof,
)
from .dataset import load_shard
from .models import (
    BootstrapConfig,
    ChildDescriptor,
    Contact,
    CentroidEntry,
    CustodyProof,
    Endpoint,
    InternalNode,
    LeafNode,
    PeerDescriptor,
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
    "SYNC_LINK": "SYNC_LINK_ACK",
    "SYNC_TOKEN": "SYNC_TOKEN_ACK",
    "SYNC_RELEASE": "SYNC_RELEASE_ACK",
    "TREE_STATS": "TREE_STATS_REPLY",
    "TREE_NODE_GET": "TREE_NODE_REPLY",
    "CENTROID_INSERT": "CENTROID_INSERT_ACK",
    "CREATE_LEAF": "CREATE_LEAF_ACK",
    "LEAF_SCAN": "LEAF_SCAN_REPLY",
    "TREE_ADJACENCY_GET": "TREE_ADJACENCY_REPLY",
    "PROFILE_GET": "PROFILE_REPLY",
    "EXPANSION_REQUEST": "EXPANSION_REPLY",
    "QUERY_HOP": "QUERY_HOP_ACK",
    "QUERY_RESULT": "QUERY_RESULT_ACK",
    "QUERY_DONE": "QUERY_DONE_ACK",
}


class ContactsPending(RuntimeError):
    """A discovered referrer has not published the required frozen round yet."""


class PeerRuntime:
    """One peer whose only overlay authority is custody of its own tree nodes."""

    def __init__(
        self,
        peer_id: int,
        shard_path: Path,
        run_dir: Path,
        endpoint_queue: Any,
        *,
        host: str = "127.0.0.1",
        bootstrap: Optional[BootstrapConfig] = None,
        root_initialization: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.peer_id = int(peer_id)
        self.shard_path = Path(shard_path)
        self.run_dir = Path(run_dir)
        self.endpoint_queue = endpoint_queue
        self.host = str(host)
        self.endpoint = Endpoint(self.host, 0)
        self.bootstrap = bootstrap
        self.root_initialization = dict(root_initialization or {})
        if bootstrap is not None:
            expected_root_id = root_node_id(bootstrap.epoch)
            expected_root_proof = genesis_custody_proof(
                bootstrap.epoch, expected_root_id, bootstrap.root.custodian
            )
            if (
                bootstrap.root.node_id != expected_root_id
                or bootstrap.root.custody_seed != 0
                or bootstrap.root.custody_proof != expected_root_proof
            ):
                raise ValueError("Bootstrap contains an invalid genesis descriptor")
        self.epoch = bootstrap.epoch if bootstrap else str(self.root_initialization.get("epoch", "unconfigured"))
        self.parameters: Dict[str, Any] = dict(bootstrap.parameters) if bootstrap else dict(self.root_initialization.get("parameters", {}))
        self.shard = load_shard(self.shard_path)
        if "peer_id" in self.shard and int(np.asarray(self.shard["peer_id"]).item()) != self.peer_id:
            raise ValueError(
                f"Private shard peer ID does not match runtime peer {self.peer_id}"
            )
        self.profile: Optional[PeerProfile] = None
        self.profile_cache: Dict[int, PeerProfile] = {}
        self.descriptor_cache: Dict[int, PeerDescriptor] = {}
        if bootstrap is not None:
            self._remember_descriptor(bootstrap.root.custodian)
        self.nodes: Dict[str, TreeNode] = {}
        self.node_locks: Dict[str, asyncio.Lock] = {}
        self.created_leaf_fingerprints: Dict[str, bytes] = {}
        self.inserted_profile = False
        self.contacts_by_round: Dict[int, List[List[Contact]]] = {}
        self.initial_contact_sets: List[set[int]] = []
        self.asked_referrers: List[set[int]] = []
        self.expansion_referrers: Dict[int, List[Optional[int]]] = {}
        self.contact_pending = False
        self.contact_retry_attempt = 0
        self.query_states: Dict[str, QueryState] = {}
        self.query_events: Dict[str, asyncio.Event] = {}
        self.compatibility_enabled = bool(self.parameters.get("compatibility_mode", False))
        self.sync_next: Optional[PeerDescriptor] = None
        self.sync_coordinator_endpoint: Optional[Endpoint] = None
        self.sync_position: Optional[int] = None
        self.sync_size = 0
        self.seen_sync_tokens: set[str] = set()
        self.phase = "STARTING"
        self.phase_timings: Dict[str, float] = {}
        self.stop_event = asyncio.Event()
        self.server: Optional[asyncio.AbstractServer] = None
        self.background_tasks: set[asyncio.Task[Any]] = set()
        self.profile_task: Optional[asyncio.Task[Any]] = None
        self.fatal_exception: Optional[BaseException] = None
        log_dir = self.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = (log_dir / f"peer_{self.peer_id:05d}.jsonl").open("a", encoding="utf-8")

    @property
    def root_id(self) -> str:
        if self.bootstrap is None:
            raise RuntimeError("Peer has not been configured")
        return root_node_id(self.bootstrap.epoch)

    @property
    def root_descriptor(self) -> ChildDescriptor:
        if self.bootstrap is None:
            raise RuntimeError("Peer has not been configured")
        return self.bootstrap.root

    @property
    def self_descriptor(self) -> PeerDescriptor:
        version = self.profile.version if self.profile is not None else 1
        return PeerDescriptor(self.peer_id, self.endpoint.host, self.endpoint.port, version)

    def _remember_descriptor(self, descriptor: PeerDescriptor) -> None:
        if descriptor.peer_id < 0:
            return
        existing = self.descriptor_cache.get(descriptor.peer_id)
        if existing is not None and (existing.host != descriptor.host or existing.port != descriptor.port):
            raise ProtocolError(f"Peer ID {descriptor.peer_id} advertised conflicting endpoints")
        if existing is None or descriptor.profile_version >= existing.profile_version:
            self.descriptor_cache[descriptor.peer_id] = descriptor

    def _write_status(self, status: str, **details: Any) -> None:
        status_dir = self.run_dir / "status"
        status_dir.mkdir(parents=True, exist_ok=True)
        path = status_dir / f"peer_{self.peer_id:05d}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "implementation": "port_based_fully_decentralized_semantica_v3",
                    "peer_id": self.peer_id,
                    "status": str(status),
                    "phase": self.phase,
                    "endpoint": self.endpoint.to_wire(),
                    "time": time.time(),
                    **details,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _remember_envelope_source(self, request: Mapping[str, Any]) -> None:
        peer_id = int(request["source_peer_id"])
        if peer_id < 0:
            return
        endpoint = Endpoint.from_wire(request["source_endpoint"])
        existing = self.descriptor_cache.get(peer_id)
        version = existing.profile_version if existing is not None else 1
        self._remember_descriptor(PeerDescriptor(peer_id, endpoint.host, endpoint.port, version))

    def _peer_id_for_endpoint(self, endpoint: Endpoint) -> Optional[int]:
        if endpoint == self.endpoint:
            return self.peer_id
        return next((peer_id for peer_id, descriptor in self.descriptor_cache.items() if descriptor.endpoint == endpoint), None)

    def _log(self, direction: str, message: Mapping[str, Any], target: Endpoint, size: int) -> None:
        payload = message.get("payload", {})
        source_endpoint = Endpoint.from_wire(message["source_endpoint"])
        row = {
            "time": time.time(),
            "monotonic": time.monotonic(),
            "peer_id": self.peer_id,
            "direction": direction,
            "remote": target.to_wire(),
            "source_peer_id": int(message["source_peer_id"]),
            "source_endpoint": source_endpoint.to_wire(),
            "destination_peer_id": self._peer_id_for_endpoint(target) if direction == "send" else self.peer_id,
            "destination_endpoint": target.to_wire() if direction == "send" else self.endpoint.to_wire(),
            "message_type": message.get("message_type"),
            "message_id": message.get("message_id"),
            "request_id": message.get("request_id"),
            "bytes": int(size),
            "payload_bytes": len(pack_message(payload)),
            "phase": self.phase,
            "round": payload.get("round"),
            "query_id": payload.get("query_id"),
            "hop": payload.get("hop"),
            "sync_phase_key": payload.get("phase_key"),
            "sync_position": payload.get("position"),
        }
        self.log_handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.log_handle.flush()

    def _rpc_client(self) -> RpcClient:
        return RpcClient(self.peer_id, self.endpoint, self.epoch, float(self.parameters.get("rpc_timeout_seconds", 30.0)), self._log)

    async def rpc_descriptor(self, descriptor: PeerDescriptor, message_type: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        self._remember_descriptor(descriptor)
        return await self._rpc_client().call(descriptor.endpoint, message_type, payload)

    async def rpc(self, peer_id: int, message_type: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        peer_id = int(peer_id)
        descriptor = self.self_descriptor if peer_id == self.peer_id else self.descriptor_cache.get(peer_id)
        if descriptor is None:
            raise ProtocolError(f"Peer {peer_id} has not been discovered through protocol data")
        return await self.rpc_descriptor(descriptor, message_type, payload)

    async def rpc_endpoint(self, endpoint: Endpoint, message_type: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return await self._rpc_client().call(endpoint, message_type, payload)

    async def serve(self) -> None:
        self.server = await asyncio.start_server(self._handle_client, self.host, 0)
        socket = self.server.sockets[0]
        self.endpoint = Endpoint(str(socket.getsockname()[0]), int(socket.getsockname()[1]))
        if self.endpoint_queue is not None:
            self.endpoint_queue.put(("LISTENING", self.peer_id, self.endpoint.host, self.endpoint.port))
        self._write_status("listening")
        if self.root_initialization:
            self._initialize_genesis()
        if self.bootstrap is not None:
            self.profile_task = self._spawn_background(self._initialize_peer())
        self._install_signal_handlers()
        async with self.server:
            await self.stop_event.wait()
        self.server.close()
        await self.server.wait_closed()
        for task in list(self.background_tasks):
            if not task.done():
                task.cancel()
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.log_handle.close()
        if self.fatal_exception is not None:
            raise self.fatal_exception

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, lambda: self._spawn_background(self._local_stop()))

    def _initialize_genesis(self) -> None:
        epoch = str(self.root_initialization["epoch"])
        seed = int(self.root_initialization["seed"])
        parameters = dict(self.root_initialization["parameters"])
        root_id = root_node_id(epoch)
        descriptor = self.self_descriptor
        proof = genesis_custody_proof(epoch, root_id, descriptor)
        root = ChildDescriptor(root_id, descriptor, 0, proof)
        self.bootstrap = BootstrapConfig(epoch, seed, root, parameters)
        self.epoch = epoch
        self.parameters = parameters
        self.compatibility_enabled = bool(parameters.get("compatibility_mode", False))
        self._remember_descriptor(descriptor)
        self.nodes[root_id] = LeafNode(root_id, 1, 0, None, 0, proof)
        bootstrap_path = Path(self.root_initialization.get("bootstrap_path", self.run_dir / "bootstrap.json"))
        bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = bootstrap_path.with_suffix(bootstrap_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.bootstrap.to_wire(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(bootstrap_path)
        self.phase = "BOOTSTRAP"
        self._write_status("bootstrap_published")

    def _spawn_background(self, awaitable: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(awaitable)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        task.add_done_callback(self._background_task_finished)
        return task

    def _background_task_finished(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        exception = task.exception()
        if exception is None:
            return
        self.fatal_exception = exception
        errors = self.run_dir / "errors"
        errors.mkdir(parents=True, exist_ok=True)
        (errors / f"peer_{self.peer_id:05d}_fatal.log").write_text("".join(traceback.format_exception(exception)), encoding="utf-8")
        self.stop_event.set()

    async def _local_stop(self) -> None:
        if self.profile is not None:
            with contextlib.suppress(Exception):
                await self._export_state()
        self.stop_event.set()

    async def _initialize_peer(self) -> None:
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
        self._remember_descriptor(self.profile.descriptor)
        self.phase_timings["profile_seconds"] = time.monotonic() - start
        if self.endpoint_queue is not None:
            self.endpoint_queue.put(("READY", self.peer_id, self.endpoint.host, self.endpoint.port))
        self._write_status("profile_ready", centroid_count=len(self.profile.centroids))
        if self.compatibility_enabled:
            self.phase = "COMPATIBILITY_READY"
            return
        await self._perform_build()
        self._write_status("tree_joined")
        if self.endpoint_queue is not None:
            self.endpoint_queue.put(("TREE_JOINED", self.peer_id, self.endpoint.host, self.endpoint.port))
        self._spawn_background(self._organize_active())

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        remote = writer.get_extra_info("peername")
        remote_endpoint = Endpoint(str(remote[0]), int(remote[1])) if remote else Endpoint("unknown", 0)
        request: Optional[Dict[str, Any]] = None
        try:
            request, size = await read_frame(reader)
            expected_epoch = None if request.get("message_type") == "CONFIGURE" else self.epoch
            validate_envelope(request, expected_epoch)
            remote_endpoint = Endpoint.from_wire(request["source_endpoint"])
            self._remember_envelope_source(request)
            self._log("receive", request, remote_endpoint, size)
            response_type, response_payload = await self._dispatch(request)
            response = envelope(response_type, epoch=str(request["epoch"]), source_peer_id=self.peer_id, source_endpoint=self.endpoint, payload=response_payload, request_id=str(request["message_id"]))
        except Exception as exc:
            response = envelope("ERROR", epoch=str(request.get("epoch", self.epoch) if request else self.epoch), source_peer_id=self.peer_id, source_endpoint=self.endpoint, payload={"error": str(exc), "traceback": traceback.format_exc()}, request_id=str(request.get("message_id")) if request else None)
            errors = self.run_dir / "errors"
            errors.mkdir(parents=True, exist_ok=True)
            with (errors / f"peer_{self.peer_id:05d}.log").open("a", encoding="utf-8") as stream:
                stream.write(str(response["payload"]["traceback"]) + "\n")
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
        payload = await getattr(self, f"_on_{message_type.lower()}")(dict(request["payload"]), request)
        if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], str):
            return payload
        return RESPONSE_TYPES[message_type], dict(payload or {})

    async def _on_configure(self, payload: Dict[str, Any], request: Mapping[str, Any]) -> Dict[str, Any]:
        if int(request["source_peer_id"]) != -1:
            raise ProtocolError("Only an external launcher may configure the genesis peer")
        if self.bootstrap is not None:
            return {"peer_id": self.peer_id, "already_configured": True}
        value = BootstrapConfig.from_wire(payload["bootstrap"])
        if value.root.custodian.peer_id != self.peer_id or value.root.custodian.endpoint != self.endpoint:
            raise ProtocolError("CONFIGURE does not describe this genesis peer")
        expected = genesis_custody_proof(value.epoch, value.root.node_id, value.root.custodian)
        if value.root.custody_seed != 0 or value.root.custody_proof != expected:
            raise ProtocolError("Invalid genesis custody proof")
        self.bootstrap = value
        self.epoch = value.epoch
        self.parameters = dict(value.parameters)
        self.compatibility_enabled = bool(self.parameters.get("compatibility_mode", False))
        self._remember_descriptor(value.root.custodian)
        self.nodes[self.root_id] = LeafNode(self.root_id, 1, 0, None, 0, value.root.custody_proof)
        self.phase = "BOOTSTRAP"
        self.profile_task = self._spawn_background(self._initialize_peer())
        return {"peer_id": self.peer_id}

    def _require_compatibility_coordinator(self, request: Mapping[str, Any], *, allow_link: bool = False) -> None:
        if not self.compatibility_enabled or int(request["source_peer_id"]) != -1:
            raise ProtocolError("Compatibility control is disabled")
        endpoint = Endpoint.from_wire(request["source_endpoint"])
        if self.sync_coordinator_endpoint is None:
            if not allow_link:
                raise ProtocolError("Compatibility coordinator is not linked")
        elif endpoint != self.sync_coordinator_endpoint:
            raise ProtocolError("Compatibility message came from another endpoint")

    async def _on_sync_link(self, payload: Dict[str, Any], request: Mapping[str, Any]) -> Dict[str, Any]:
        self._require_compatibility_coordinator(request, allow_link=True)
        coordinator = Endpoint.from_wire(payload["coordinator_endpoint"])
        if coordinator != Endpoint.from_wire(request["source_endpoint"]):
            raise ProtocolError("Coordinator endpoint differs from the envelope")
        self.sync_coordinator_endpoint = coordinator
        self.sync_position = int(payload["position"])
        self.sync_size = int(payload["cohort_size"])
        self.sync_next = PeerDescriptor.from_wire(payload["next"]) if payload.get("next") is not None else None
        if self.sync_next is not None:
            self._remember_descriptor(self.sync_next)
        return {"peer_id": self.peer_id, "position": self.sync_position}

    async def _on_sync_token(self, payload: Dict[str, Any], request: Mapping[str, Any]) -> Dict[str, Any]:
        source_peer = int(request["source_peer_id"])
        if source_peer == -1:
            self._require_compatibility_coordinator(request)
        elif self.sync_position is None or source_peer not in self.descriptor_cache:
            raise ProtocolError("Synchronization token source is not linked")
        token_id = str(payload["token_id"])
        if token_id in self.seen_sync_tokens:
            return {"token_id": token_id, "duplicate": True}
        if self.sync_position is None or int(payload["position"]) != self.sync_position:
            raise ProtocolError("Synchronization token arrived at the wrong position")
        self.seen_sync_tokens.add(token_id)
        self._spawn_background(self._process_sync_token(dict(payload)))
        return {"token_id": token_id, "accepted": True}

    async def _process_sync_token(self, token: Dict[str, Any]) -> None:
        try:
            phase = str(token["phase"])
            if phase == "BUILD":
                self.phase = "COMPATIBILITY_BUILD"
                await self._perform_build()
            elif phase == "STATS":
                token["maximum_tree_depth"] = max(int(token.get("maximum_tree_depth", 0)), self._maximum_local_depth())
            elif phase == "ROUND0":
                self.phase = "COMPATIBILITY_ORGANIZE"
                await self._build_round0(int(token["maximum_tree_depth"]), require_complete=True)
            elif phase == "EXPANSION":
                self.phase = "COMPATIBILITY_ORGANIZE"
                await self._expand_round(int(token["round"]))
            elif phase == "QUERY":
                self.phase = "COMPATIBILITY_ORGANIZE"
                await self._run_local_query()
            elif phase == "EXPORT":
                await self._export_state()
            else:
                raise ProtocolError(f"Unknown synchronization phase: {phase}")
            next_token = dict(token)
            next_token["position"] = int(token["position"]) + 1
            if self.sync_next is not None:
                await self.rpc_descriptor(self.sync_next, "SYNC_TOKEN", next_token)
            else:
                assert self.sync_coordinator_endpoint is not None
                await self.rpc_endpoint(self.sync_coordinator_endpoint, "SYNC_PHASE_COMPLETE", {"phase_key": str(token["phase_key"]), "token_id": str(token["token_id"]), "maximum_tree_depth": int(token.get("maximum_tree_depth", 0))})
        except Exception as exc:
            if self.sync_coordinator_endpoint is not None:
                with contextlib.suppress(Exception):
                    await self.rpc_endpoint(self.sync_coordinator_endpoint, "SYNC_PHASE_FAILED", {"phase_key": str(token["phase_key"]), "peer_id": self.peer_id, "error": str(exc)})
            raise

    async def _on_sync_release(self, _payload: Dict[str, Any], request: Mapping[str, Any]) -> Dict[str, Any]:
        self._require_compatibility_coordinator(request)
        self.sync_next = None
        self.sync_coordinator_endpoint = None
        self.sync_position = None
        self.sync_size = 0
        self.seen_sync_tokens.clear()
        self.compatibility_enabled = False
        self.parameters["compatibility_mode"] = False
        self.phase = "ACTIVE"
        self._write_status("active", contact_pending=False)
        return {"peer_id": self.peer_id, "released": True}

    async def _perform_build(self) -> None:
        if self.profile is None:
            raise ProtocolError("Profile is not ready for tree insertion")
        if self.inserted_profile:
            return
        started = time.monotonic()
        for centroid_index, vector in enumerate(self.profile.centroids):
            await self._insert_entry(CentroidEntry(self.peer_id, centroid_index, vector, self.profile.version, self.endpoint))
        self.inserted_profile = True
        self.phase_timings["tree_insert_seconds"] = time.monotonic() - started

    async def _get_node(self, descriptor: ChildDescriptor) -> TreeNode:
        self._remember_descriptor(descriptor.custodian)
        response = await self.rpc_descriptor(descriptor.custodian, "TREE_NODE_GET", {"node_id": descriptor.node_id})
        return node_from_wire(response["payload"]["node"])

    async def _locate_leaf(self, vector: np.ndarray) -> tuple[ChildDescriptor, LeafNode]:
        descriptor = self.root_descriptor
        while True:
            node = await self._get_node(descriptor)
            if isinstance(node, LeafNode):
                return descriptor, node
            descriptor = node.left if choose_child(vector, node.left.center, node.right.center) == "left" else node.right

    async def _insert_entry(self, entry: CentroidEntry) -> None:
        self._remember_descriptor(entry.owner)
        descriptor, _ = await self._locate_leaf(entry.vector)
        while True:
            response = await self.rpc_descriptor(descriptor.custodian, "CENTROID_INSERT", {"node_id": descriptor.node_id, "entry": entry.to_wire()})
            if response["message_type"] != "TREE_REDIRECT":
                return
            node = InternalNode.from_wire(response["payload"]["node"])
            descriptor = node.left if choose_child(entry.vector, node.left.center, node.right.center) == "left" else node.right

    def _node_descriptor(self, node: TreeNode, center: Optional[np.ndarray] = None) -> ChildDescriptor:
        return ChildDescriptor(node.node_id, node.custody_proof.custodian, node.custody_seed, node.custody_proof, center=center)

    async def _on_tree_node_get(self, payload: Dict[str, Any], _request: Mapping[str, Any]) -> Dict[str, Any]:
        node = self.nodes.get(str(payload["node_id"]))
        if node is None:
            raise ProtocolError("Tree node is not held by this custodian")
        return {"node": node.to_wire(include_entries=False) if isinstance(node, LeafNode) else node.to_wire()}

    async def _on_centroid_insert(self, payload: Dict[str, Any], _request: Mapping[str, Any]) -> Mapping[str, Any] | tuple[str, Mapping[str, Any]]:
        node_id = str(payload["node_id"])
        lock = self.node_locks.setdefault(node_id, asyncio.Lock())
        async with lock:
            node = self.nodes.get(node_id)
            if isinstance(node, InternalNode):
                return "TREE_REDIRECT", {"node": node.to_wire()}
            if not isinstance(node, LeafNode):
                raise ProtocolError("Insertion target is not a known leaf")
            entry = CentroidEntry.from_wire(payload["entry"])
            self._remember_descriptor(entry.owner)
            if all(existing.entry_id != entry.entry_id for existing in node.entries):
                node.entries.append(entry)
                node.version += 1
            if self._should_split(node):
                await self._split_leaf(node)
            return {"node_id": node_id, "entry_id": entry.entry_id, "version": node.version}

    def _should_split(self, node: LeafNode) -> bool:
        return len(node.entries) > int(self.parameters["leaf_capacity"]) and node.failed_split_size != len(node.entries)

    async def _split_leaf(self, node: LeafNode) -> None:
        result = split_entries(node.entries, seed=int(self.parameters["seed"]), delta=float(self.parameters["delta"]))
        if result is None:
            node.failed_split_size = len(node.entries)
            return
        left_center, right_center, left_entries, right_entries = result
        child_seed = node.custody_seed + 1
        left_id = child_node_id(self.epoch, node.node_id, "left")
        right_id = child_node_id(self.epoch, node.node_id, "right")
        left_proof = custody_proof_for_entries(self.epoch, left_id, child_seed, left_entries)
        right_proof = custody_proof_for_entries(self.epoch, right_id, child_seed, right_entries)
        left = ChildDescriptor(left_id, left_proof.custodian, child_seed, left_proof, left_center)
        right = ChildDescriptor(right_id, right_proof.custodian, child_seed, right_proof, right_center)
        parent = self._node_descriptor(node)
        common = {"depth": node.depth + 1, "parent": parent.to_wire(), "parent_entry_count": len(node.entries), "custody_seed": child_seed}
        await asyncio.gather(
            self.rpc_descriptor(left.custodian, "CREATE_LEAF", {**common, "side": "left", "node_id": left_id, "custody_proof": left_proof.to_wire(), "entries": [entry.to_wire() for entry in left_entries]}),
            self.rpc_descriptor(right.custodian, "CREATE_LEAF", {**common, "side": "right", "node_id": right_id, "custody_proof": right_proof.to_wire(), "entries": [entry.to_wire() for entry in right_entries]}),
        )
        self.nodes[node.node_id] = InternalNode(node.node_id, node.version + 1, node.depth, node.parent, node.custody_seed, node.custody_proof, left, right)

    async def _on_create_leaf(self, payload: Dict[str, Any], request: Mapping[str, Any]) -> Dict[str, Any]:
        node_id = str(payload["node_id"])
        side = str(payload["side"])
        parent = ChildDescriptor.from_wire(payload["parent"])
        if int(request["source_peer_id"]) != parent.custodian.peer_id or Endpoint.from_wire(request["source_endpoint"]) != parent.custodian.endpoint:
            raise ProtocolError("CREATE_LEAF did not come from the parent custodian")
        if node_id != child_node_id(self.epoch, parent.node_id, side):
            raise ProtocolError("CREATE_LEAF node ID is not derived from its parent")
        custody_seed = int(payload["custody_seed"])
        if custody_seed != parent.custody_seed + 1:
            raise ProtocolError("CREATE_LEAF custody seed is not parent seed plus one")
        entries = [CentroidEntry.from_wire(value) for value in payload["entries"]]
        proof = CustodyProof.from_wire(payload["custody_proof"])
        if not verify_custody_proof(self.epoch, node_id, custody_seed, entries, proof):
            raise ProtocolError("CREATE_LEAF custody proof is invalid")
        if proof.custodian != self.self_descriptor:
            raise ProtocolError("CREATE_LEAF reached a peer not selected by its proof")
        for entry in entries:
            self._remember_descriptor(entry.owner)
        proposed = LeafNode(node_id, 1, int(payload["depth"]), parent, custody_seed, proof, entries)
        fingerprint = pack_message(proposed.to_wire())
        lock = self.node_locks.setdefault(node_id, asyncio.Lock())
        async with lock:
            existing = self.nodes.get(node_id)
            if existing is not None:
                if self.created_leaf_fingerprints.get(node_id) != fingerprint:
                    raise ProtocolError("CREATE_LEAF duplicate payload differs")
                return {"node_id": node_id, "already_exists": True}
            self.created_leaf_fingerprints[node_id] = fingerprint
            self.nodes[node_id] = proposed
            if len(proposed.entries) > int(self.parameters["leaf_capacity"]) and len(proposed.entries) >= int(payload["parent_entry_count"]):
                proposed.failed_split_size = len(proposed.entries)
            elif self._should_split(proposed):
                await self._split_leaf(proposed)
            return {"node_id": node_id, "entry_count": len(proposed.entries), "custody_proof": proof.to_wire()}

    def _maximum_local_depth(self) -> int:
        return max((int(node.depth) for node in self.nodes.values()), default=0)

    async def _on_tree_stats(self, _payload: Dict[str, Any], _request: Mapping[str, Any]) -> Dict[str, Any]:
        return {"peer_id": self.peer_id, "node_count": len(self.nodes), "maximum_local_depth": self._maximum_local_depth()}

    async def _on_leaf_scan(self, payload: Dict[str, Any], _request: Mapping[str, Any]) -> Mapping[str, Any] | tuple[str, Mapping[str, Any]]:
        node = self.nodes.get(str(payload["node_id"]))
        if isinstance(node, InternalNode):
            return "TREE_REDIRECT", {"node": node.to_wire()}
        if not isinstance(node, LeafNode):
            raise ProtocolError("LEAF_SCAN target is unknown")
        return {"node_id": node.node_id, "version": node.version, "entries": [entry.to_wire() for entry in node.entries]}

    async def _on_tree_adjacency_get(self, payload: Dict[str, Any], _request: Mapping[str, Any]) -> Dict[str, Any]:
        node = self.nodes.get(str(payload["node_id"]))
        if node is None:
            raise ProtocolError("Unknown adjacency node")
        neighbors: List[ChildDescriptor] = []
        if node.parent is not None:
            neighbors.append(node.parent)
        if isinstance(node, InternalNode):
            neighbors.extend([node.left, node.right])
        return {"node_id": node.node_id, "node_type": node.node_type, "version": node.version, "neighbors": [neighbor.to_wire() for neighbor in neighbors]}

    async def _on_profile_get(self, _payload: Dict[str, Any], _request: Mapping[str, Any]) -> Dict[str, Any]:
        if self.profile is None:
            raise ProtocolError("Profile is not ready")
        return {"profile": self.profile.to_wire()}

    async def _profile_for(self, value: int | PeerDescriptor) -> PeerProfile:
        peer_id = int(value.peer_id if isinstance(value, PeerDescriptor) else value)
        if peer_id in self.profile_cache:
            return self.profile_cache[peer_id]
        descriptor = value if isinstance(value, PeerDescriptor) else self.descriptor_cache.get(peer_id)
        if descriptor is None:
            raise ProtocolError(f"Profile owner {peer_id} has not been discovered")
        response = await self.rpc_descriptor(descriptor, "PROFILE_GET", {})
        profile = PeerProfile.from_wire(response["payload"]["profile"])
        if profile.peer_id != descriptor.peer_id or profile.endpoint != descriptor.endpoint:
            raise ProtocolError("PROFILE_GET returned a conflicting descriptor")
        self.profile_cache[peer_id] = profile
        self._remember_descriptor(profile.descriptor)
        return profile

    async def _advance_frontier(self, centroid_index: int, frontier: List[ChildDescriptor], visited: set[str], candidate_scores: List[Dict[int, float]], candidate_descriptors: Dict[int, PeerDescriptor]) -> List[ChildDescriptor]:
        next_frontier: List[ChildDescriptor] = []
        owners: Dict[int, PeerDescriptor] = {}
        for descriptor in frontier:
            node = await self._get_node(descriptor)
            if isinstance(node, LeafNode):
                response = await self.rpc_descriptor(descriptor.custodian, "LEAF_SCAN", {"node_id": descriptor.node_id})
                if response["message_type"] != "TREE_REDIRECT":
                    for raw in response["payload"]["entries"]:
                        entry = CentroidEntry.from_wire(raw)
                        if entry.peer_id != self.peer_id:
                            existing = owners.get(entry.peer_id)
                            if existing is not None and existing != entry.owner:
                                raise ProtocolError("Leaf contains conflicting owner descriptors")
                            owners[entry.peer_id] = entry.owner
                            self._remember_descriptor(entry.owner)
            response = await self.rpc_descriptor(descriptor.custodian, "TREE_ADJACENCY_GET", {"node_id": descriptor.node_id})
            for raw in response["payload"]["neighbors"]:
                neighbor = ChildDescriptor.from_wire(raw)
                self._remember_descriptor(neighbor.custodian)
                if neighbor.node_id not in visited:
                    visited.add(neighbor.node_id)
                    next_frontier.append(neighbor)
        source = self.profile.centroids[centroid_index]
        for owner in sorted(owners):
            candidate_descriptors[owner] = owners[owner]
            if owner not in candidate_scores[centroid_index]:
                profile = await self._profile_for(owners[owner])
                candidate_scores[centroid_index][owner] = peer_centroid_score(source, profile)
        return next_frontier

    async def _build_round0(self, maximum_tree_depth: Optional[int], *, require_complete: bool) -> bool:
        if self.profile is None:
            raise ProtocolError("Profile is not ready")
        start_descriptors = [(await self._locate_leaf(vector))[0] for vector in self.profile.centroids]
        frontiers = [[descriptor] for descriptor in start_descriptors]
        visited = [{descriptor.node_id} for descriptor in start_descriptors]
        candidate_scores: List[Dict[int, float]] = [dict() for _ in frontiers]
        candidate_descriptors: Dict[int, PeerDescriptor] = {}
        caps = self.profile.contact_caps.tolist()
        active = list(range(len(frontiers)))
        assigned: List[List[Tuple[int, float]]] = [[] for _ in frontiers]
        searched_distance = np.full(len(frontiers), -1, dtype=np.int32)
        maximum_distance = max(0, 4 * int(maximum_tree_depth)) if maximum_tree_depth is not None else None
        while active:
            for index in active:
                searched_distance[index] += 1
                frontiers[index] = await self._advance_frontier(index, frontiers[index], visited[index], candidate_scores, candidate_descriptors)
            assigned, underfilled = assign_unique_candidates(candidate_scores, caps)
            if not underfilled:
                break
            active = [index for index in underfilled if frontiers[index] and (maximum_distance is None or int(searched_distance[index]) < maximum_distance)]
            if not active:
                if require_complete:
                    missing = {index: int(caps[index] - len(assigned[index])) for index in underfilled}
                    raise RuntimeError(f"Unable to fill round-0 degree for peer {self.peer_id}: {missing}")
                return False
        contacts: List[List[Contact]] = []
        for values in assigned:
            centroid_contacts: List[Contact] = []
            for candidate, score in values:
                profile = await self._profile_for(candidate_descriptors[candidate])
                centroid_contacts.append(Contact(candidate, score, profile.endpoint, profile.version, "round0"))
            contacts.append(centroid_contacts)
        if sum(len(values) for values in contacts) != int(self.parameters["degree"]):
            if require_complete:
                raise RuntimeError("Round-0 contact degree is incomplete")
            return False
        self.contacts_by_round[0] = contacts
        self.initial_contact_sets = [{contact.peer_id for contact in values} for values in contacts]
        self.asked_referrers = [set() for _ in contacts]
        self.contact_pending = False
        return True

    def _contact_retry_delay(self, attempt: int) -> float:
        base = (1, 2, 4, 8, 16, 30)[min(int(attempt), 5)]
        jitter_ms = stable_int(f"contact-retry:{self.epoch}:{self.peer_id}:{int(attempt)}") % 251
        return float(base) + float(jitter_ms) / 1000.0

    async def _organize_active(self) -> None:
        self.phase = "ACTIVE_ORGANIZE"
        started = time.monotonic()
        while not self.stop_event.is_set():
            if await self._build_round0(None, require_complete=False):
                break
            self.contact_pending = True
            self._write_status(
                "contacts_pending",
                contact_pending=True,
                retry_attempt=self.contact_retry_attempt,
            )
            delay = self._contact_retry_delay(self.contact_retry_attempt)
            self.contact_retry_attempt += 1
            await asyncio.sleep(delay)
        if self.stop_event.is_set():
            return
        self.phase_timings["round0_seconds"] = time.monotonic() - started
        for round_index in range(1, int(self.parameters["expansion_rounds"]) + 1):
            while not self.stop_event.is_set():
                try:
                    await self._expand_round(round_index)
                    break
                except ContactsPending:
                    delay = self._contact_retry_delay(self.contact_retry_attempt)
                    self.contact_retry_attempt += 1
                    await asyncio.sleep(delay)
            if self.stop_event.is_set():
                return
        await self._run_local_query()
        self.phase = "ACTIVE"
        await self._export_state()
        self._write_status("active", contact_pending=False)
        if self.endpoint_queue is not None:
            self.endpoint_queue.put(("ACTIVE", self.peer_id, self.endpoint.host, self.endpoint.port))

    async def _on_expansion_request(self, payload: Dict[str, Any], request: Mapping[str, Any]) -> Dict[str, Any]:
        round_index = int(payload["round"])
        if round_index not in self.contacts_by_round:
            return {"round": round_index, "available": False, "peers": []}
        source_peer_id = int(request["source_peer_id"])
        descriptors: Dict[int, PeerDescriptor] = {}
        for values in self.contacts_by_round[round_index]:
            for contact in values:
                if contact.peer_id != source_peer_id:
                    descriptors[contact.peer_id] = contact.descriptor
        return {"round": round_index, "available": True, "peers": [descriptors[peer_id].to_wire() for peer_id in sorted(descriptors)]}

    async def _expand_round(self, round_index: int) -> None:
        asked_before = [set(values) for values in self.asked_referrers]
        try:
            await self._expand_round_once(round_index)
        except ContactsPending:
            self.asked_referrers = asked_before
            raise

    async def _expand_round_once(self, round_index: int) -> None:
        previous = self.contacts_by_round[round_index - 1]
        current = [[deepcopy(contact) for contact in values] for values in previous]
        owner_existing = {contact.peer_id for values in previous for contact in values}
        proposals: List[List[Tuple[int, float]]] = [[] for _ in current]
        selected_referrers: List[Optional[int]] = [None for _ in current]
        for centroid_index, values in enumerate(previous):
            ids = [contact.peer_id for contact in values]
            new_tier = [candidate for candidate in ids if candidate not in self.initial_contact_sets[centroid_index] and candidate not in self.asked_referrers[centroid_index]]
            initial_tier = [candidate for candidate in ids if candidate in self.initial_contact_sets[centroid_index] and candidate not in self.asked_referrers[centroid_index]]
            tier = new_tier if new_tier else initial_tier
            if not tier:
                continue
            rng = np.random.default_rng(stable_int(f"semantic-referrer:{self.parameters['seed']}:{round_index}:{self.peer_id}:{centroid_index}"))
            referrer = int(tier[int(rng.integers(0, len(tier)))])
            selected_referrers[centroid_index] = referrer
            self.asked_referrers[centroid_index].add(referrer)
            response = await self.rpc(referrer, "EXPANSION_REQUEST", {"round": round_index - 1})
            if not bool(response["payload"].get("available", True)):
                raise ContactsPending(
                    f"Referrer {referrer} has not published round {round_index - 1}"
                )
            for raw in response["payload"]["peers"]:
                descriptor = PeerDescriptor.from_wire(raw)
                self._remember_descriptor(descriptor)
                candidate = descriptor.peer_id
                if candidate == self.peer_id or candidate in owner_existing:
                    continue
                profile = await self._profile_for(descriptor)
                score = peer_centroid_score(self.profile.centroids[centroid_index], profile)
                if np.isfinite(score):
                    proposals[centroid_index].append((candidate, score))
        for centroid_index, values in enumerate(proposals):
            previous_contacts = {contact.peer_id: contact for contact in current[centroid_index]}
            excluded = {self.peer_id}
            excluded.update(contact.peer_id for other_index, contacts in enumerate(current) if other_index != centroid_index for contact in contacts)
            pairs = replace_centroid_contacts([(contact.peer_id, contact.score) for contact in current[centroid_index]], values, int(self.profile.contact_caps[centroid_index]), excluded)
            rewritten: List[Contact] = []
            for candidate, score in pairs:
                profile = await self._profile_for(candidate)
                rewritten.append(Contact(candidate, score, profile.endpoint, profile.version, previous_contacts[candidate].learned_at if candidate in previous_contacts else f"expansion_{round_index}"))
            current[centroid_index] = rewritten
        self.contacts_by_round[round_index] = current
        self.expansion_referrers[round_index] = selected_referrers

    def _best_query_neighbor(self, query: np.ndarray, visited: Iterable[int]) -> Optional[PeerDescriptor]:
        final_round = int(self.parameters["expansion_rounds"])
        if final_round not in self.contacts_by_round:
            return None
        visited_set = {int(value) for value in visited}
        contacts = {contact.peer_id: contact for values in self.contacts_by_round[final_round] for contact in values if contact.peer_id not in visited_set}
        ranked = []
        for candidate in sorted(contacts):
            profile = self.profile_cache.get(candidate)
            if profile is None:
                raise ProtocolError(f"Query neighbor {candidate} profile is not cached")
            ranked.append((-peer_centroid_score(query, profile), candidate))
        if not ranked:
            return None
        ranked.sort()
        return contacts[int(ranked[0][1])].descriptor

    async def _run_local_query(self) -> Dict[str, Any]:
        query_id = str(self.shard["query_id"].item())
        state = QueryState(query_id, normalize_rows(np.asarray(self.shard["query_embedding"], dtype=np.float32)[None, :])[0], str(self.shard["target_doc_id"].item()))
        self.query_states[query_id] = state
        event = asyncio.Event()
        self.query_events[query_id] = event
        next_peer = self._best_query_neighbor(state.query_vector, {self.peer_id})
        if next_peer is None:
            state.terminal_hop = 0
            state.terminal_reason = "dead_end"
            event.set()
        else:
            await self.rpc_descriptor(next_peer, "QUERY_HOP", {"query_id": query_id, "source": self.self_descriptor.to_wire(), "query_embedding": state.query_vector, "visited": [self.peer_id], "hop": 1, "hop_limit": int(self.parameters["query_hop_limit"])})
        await event.wait()
        for hop in sorted(state.results_by_hop):
            state.final_results = merge_topk(state.final_results, state.results_by_hop[hop], int(self.parameters["local_top_k"]))
        return state.to_wire()

    async def _on_query_hop(self, payload: Dict[str, Any], _request: Mapping[str, Any]) -> Dict[str, Any]:
        query_id = str(payload["query_id"])
        query = np.asarray(payload["query_embedding"], dtype=np.float32)
        source = PeerDescriptor.from_wire(payload["source"])
        self._remember_descriptor(source)
        hop = int(payload["hop"])
        visited = [int(value) for value in payload["visited"]]
        local = local_topk(
            query,
            self.shard["training_doc_ids"].tolist(),
            np.asarray(self.shard["training_embeddings"], dtype=np.float32),
            int(self.parameters["local_top_k"]),
            query_is_normalized=True,
            doc_order_keys=np.asarray(self.shard["training_doc_indices"], dtype=np.int64),
        )
        await self.rpc_descriptor(source, "QUERY_RESULT", {"query_id": query_id, "hop": hop, "peer": self.self_descriptor.to_wire(), "results": [[doc_id, score] for doc_id, score in local]})
        visited.append(self.peer_id)
        if hop >= int(payload["hop_limit"]):
            await self.rpc_descriptor(source, "QUERY_DONE", {"query_id": query_id, "hop": hop, "reason": "hop_limit"})
            return {"query_id": query_id, "hop": hop, "terminal": True}
        next_peer = self._best_query_neighbor(query, visited)
        if next_peer is None:
            await self.rpc_descriptor(source, "QUERY_DONE", {"query_id": query_id, "hop": hop, "reason": "exhausted_neighbors"})
            return {"query_id": query_id, "hop": hop, "terminal": True}
        await self.rpc_descriptor(next_peer, "QUERY_HOP", {**payload, "visited": visited, "hop": hop + 1})
        return {"query_id": query_id, "hop": hop, "terminal": False}

    async def _on_query_result(self, payload: Dict[str, Any], _request: Mapping[str, Any]) -> Dict[str, Any]:
        state = self.query_states[str(payload["query_id"])]
        hop = int(payload["hop"])
        peer = PeerDescriptor.from_wire(payload["peer"])
        self._remember_descriptor(peer)
        state.peer_by_hop[hop] = peer.peer_id
        state.results_by_hop[hop] = [(str(doc_id), float(score)) for doc_id, score in payload["results"]]
        return {"query_id": state.query_id, "hop": hop}

    async def _on_query_done(self, payload: Dict[str, Any], _request: Mapping[str, Any]) -> Dict[str, Any]:
        query_id = str(payload["query_id"])
        state = self.query_states[query_id]
        state.terminal_hop = int(payload["hop"])
        state.terminal_reason = str(payload["reason"])
        self.query_events[query_id].set()
        return {"query_id": query_id}

    async def _export_state(self) -> Dict[str, Any]:
        state_dir = self.run_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "decentralized_semantica_peer_state_v3",
            "implementation": "port_based_fully_decentralized_semantica_v3",
            "peer_id": self.peer_id,
            "endpoint": self.endpoint.to_wire(),
            "profile": self.profile.to_wire() if self.profile else None,
            "nodes": {node_id: node.to_wire(include_entries=True) if isinstance(node, LeafNode) else node.to_wire() for node_id, node in self.nodes.items()},
            "contacts_by_round": {str(round_index): contacts_to_wire(contacts) for round_index, contacts in sorted(self.contacts_by_round.items())},
            "queries": {query_id: state.to_wire() for query_id, state in self.query_states.items()},
            "expansion_referrers": {
                str(round_index): values
                for round_index, values in sorted(self.expansion_referrers.items())
            },
            "contact_pending": self.contact_pending,
            "contact_retry_attempt": self.contact_retry_attempt,
            "phase": self.phase,
            "phase_timings": self.phase_timings,
        }
        path = state_dir / f"peer_{self.peer_id:05d}.msgpack"
        path.write_bytes(pack_message(payload))
        return {"peer_id": self.peer_id, "state_path": str(path)}


def run_peer_process(peer_id: int, shard_path: str, run_dir: str, endpoint_queue: Any, bootstrap_wire: Optional[Mapping[str, Any]] = None, root_initialization: Optional[Mapping[str, Any]] = None) -> None:
    bootstrap = BootstrapConfig.from_wire(bootstrap_wire) if bootstrap_wire is not None else None
    runtime = PeerRuntime(peer_id, Path(shard_path), Path(run_dir), endpoint_queue, bootstrap=bootstrap, root_initialization=root_initialization)
    asyncio.run(runtime.serve())


def run_joining_peer(peer_id: int, shard_path: str, run_dir: str, bootstrap_wire: Mapping[str, Any]) -> None:
    run_peer_process(peer_id, shard_path, run_dir, None, bootstrap_wire)


def run_root_peer(peer_id: int, shard_path: str, run_dir: str, root_initialization: Mapping[str, Any]) -> None:
    run_peer_process(peer_id, shard_path, run_dir, None, root_initialization=root_initialization)

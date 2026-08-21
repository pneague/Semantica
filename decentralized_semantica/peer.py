from __future__ import annotations

import asyncio
import contextlib
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
    custodian_for_peers,
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
    BootstrapConfig,
    ChildDescriptor,
    Contact,
    CentroidEntry,
    Endpoint,
    InternalNode,
    LeafNode,
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
    "JOIN": "JOIN_ACK",
    "PEER_LOOKUP": "PEER_LOOKUP_REPLY",
    "CUSTODIAN_ASSIGN": "CUSTODIAN_ASSIGN_REPLY",
    "REGISTRY_STATUS": "REGISTRY_STATUS_REPLY",
    "ACTIVATE_SYNC": "ACTIVATE_SYNC_ACK",
    "SYNC_LINK": "SYNC_LINK_ACK",
    "SYNC_TOKEN": "SYNC_TOKEN_ACK",
    "SYNC_PHASE_COMPLETE": "SYNC_PHASE_COMPLETE_ACK",
    "SYNC_PHASE_FAILED": "SYNC_PHASE_FAILED_ACK",
    "FINALIZE": "FINALIZE_ACK",
    "TREE_STATS": "TREE_STATS_REPLY",
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
        bootstrap: Optional[BootstrapConfig] = None,
    ) -> None:
        self.peer_id = int(peer_id)
        self.shard_path = Path(shard_path)
        self.run_dir = Path(run_dir)
        self.endpoint_queue = endpoint_queue
        self.host = str(host)
        self.endpoint = Endpoint(self.host, 0)
        self.bootstrap: Optional[BootstrapConfig] = bootstrap
        self.epoch = bootstrap.epoch if bootstrap else "unconfigured"
        self.parameters: Dict[str, Any] = (
            dict(bootstrap.parameters) if bootstrap else {}
        )
        self.endpoint_cache: Dict[int, Endpoint] = {}
        if bootstrap is not None:
            self.endpoint_cache[int(bootstrap.root_peer_id)] = bootstrap.root_endpoint
        self.registry: Dict[int, Endpoint] = {}
        self.join_order: List[int] = []
        self.join_positions: Dict[int, int] = {}
        self.custodian_ready: set[int] = set()
        self.custodian_assignments: Dict[str, int] = {}
        self.initial_cohort: List[int] = []
        self.tree_joined: set[int] = set()
        self.active_peers: set[int] = set()
        self.sync_next_peer_id: Optional[int] = None
        self.sync_next_endpoint: Optional[Endpoint] = None
        self.sync_position: Optional[int] = None
        self.sync_size = 0
        self.sync_phase_waiters: Dict[str, asyncio.Future[Dict[str, Any]]] = {}
        self.seen_sync_tokens: set[str] = set()
        self.background_tasks: set[asyncio.Task[Any]] = set()
        self.admission_lock = asyncio.Lock()
        self.admission_task: Optional[asyncio.Task[None]] = None
        self.finalize_task: Optional[asyncio.Task[None]] = None
        self.checkpoint_state_on_active = False
        self.fatal_exception: Optional[BaseException] = None
        self.shard = load_shard(self.shard_path)
        self.profile: Optional[PeerProfile] = None
        self.inserted_profile = False
        self.profile_cache: Dict[int, PeerProfile] = {}
        self.nodes: Dict[str, TreeNode] = {}
        self.node_locks: Dict[str, asyncio.Lock] = {}
        self.contacts_by_round: Dict[int, List[List[Contact]]] = {}
        self.initial_contact_sets: List[set[int]] = []
        self.asked_referrers: List[set[int]] = []
        self.query_states: Dict[str, QueryState] = {}
        self.query_events: Dict[str, asyncio.Event] = {}
        self.profile_ready: set[int] = set()
        self.phase = "STARTING" if bootstrap is None else "BOOTSTRAP"
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
        if self.bootstrap is None:
            raise RuntimeError("Peer has not been configured")
        return root_node_id(self.bootstrap.epoch)

    @property
    def root_peer_id(self) -> int:
        if self.bootstrap is None:
            raise RuntimeError("Peer has not been configured")
        return int(self.bootstrap.root_peer_id)

    @property
    def root_endpoint(self) -> Endpoint:
        if self.bootstrap is None:
            raise RuntimeError("Peer has not been configured")
        return self.bootstrap.root_endpoint

    @property
    def is_root(self) -> bool:
        return self.bootstrap is not None and self.peer_id == self.root_peer_id

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
            "sync_phase_key": payload.get("phase_key"),
            "sync_position": payload.get("position"),
        }
        self.log_handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.log_handle.flush()

    def _peer_id_for_endpoint(self, endpoint: Endpoint) -> Optional[int]:
        if endpoint == self.endpoint:
            return self.peer_id
        candidates = self.registry if self.is_root else self.endpoint_cache
        return next(
            (peer_id for peer_id, member_endpoint in candidates.items() if member_endpoint == endpoint),
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
        endpoint = await self._resolve_peer(int(peer_id))
        return await self._rpc_client().call(endpoint, message_type, payload)

    async def _resolve_peer(self, peer_id: int) -> Endpoint:
        peer_id = int(peer_id)
        if peer_id == self.peer_id:
            return self.endpoint
        if self.is_root and peer_id in self.registry:
            return self.registry[peer_id]
        if peer_id in self.endpoint_cache:
            return self.endpoint_cache[peer_id]
        response = await self.rpc_endpoint(
            self.root_endpoint, "PEER_LOOKUP", {"peer_id": peer_id}
        )
        endpoint = Endpoint.from_wire(response["payload"]["endpoint"])
        self.endpoint_cache[peer_id] = endpoint
        return endpoint

    async def rpc_endpoint(
        self, endpoint: Endpoint, message_type: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return await self._rpc_client().call(endpoint, message_type, payload)

    async def serve(self) -> None:
        self.server = await asyncio.start_server(self._handle_client, self.host, 0)
        socket = self.server.sockets[0]
        self.endpoint = Endpoint(str(socket.getsockname()[0]), int(socket.getsockname()[1]))
        if self.endpoint_queue is not None:
            self.endpoint_queue.put(
                ("LISTENING", self.peer_id, self.endpoint.host, self.endpoint.port)
            )
        if self.bootstrap is not None:
            self.profile_task = self._spawn_background(self._join_and_initialize())
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
        if self.fatal_exception is not None:
            raise self.fatal_exception

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
        if self.is_root:
            for waiter in self.sync_phase_waiters.values():
                if not waiter.done():
                    waiter.set_exception(exception)
        (self.run_dir / "errors").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "errors" / f"peer_{self.peer_id:05d}_fatal.log").write_text(
            "".join(traceback.format_exception(exception)), encoding="utf-8"
        )
        self.stop_event.set()

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
        if self.bootstrap is not None:
            return {"peer_id": self.peer_id, "already_configured": True}
        self.bootstrap = BootstrapConfig.from_wire(payload["bootstrap"])
        self.epoch = self.bootstrap.epoch
        self.parameters = dict(self.bootstrap.parameters)
        if self.peer_id != self.root_peer_id or self.root_endpoint != self.endpoint:
            raise ProtocolError("CONFIGURE must target the declared root endpoint")
        self.endpoint_cache[self.peer_id] = self.endpoint
        self.nodes[self.root_id] = LeafNode(
            node_id=self.root_id, version=1, depth=0, parent=None
        )
        self.custodian_assignments[self.root_id] = self.peer_id
        self.phase = "BOOTSTRAP"
        self.profile_task = self._spawn_background(self._join_and_initialize())
        return {"peer_id": self.peer_id, "root_peer_id": self.root_peer_id}

    async def _join_and_initialize(self) -> None:
        response = await self.rpc_endpoint(
            self.root_endpoint,
            "JOIN",
            {"peer_id": self.peer_id, "endpoint": self.endpoint.to_wire()},
        )
        if int(response["payload"]["peer_id"]) != self.peer_id:
            raise ProtocolError("Root acknowledged the wrong joining peer")
        if self.endpoint_queue is not None:
            self.endpoint_queue.put(
                ("JOINED", self.peer_id, self.endpoint.host, self.endpoint.port)
            )
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
        if self.endpoint_queue is not None:
            self.endpoint_queue.put(
                ("READY", self.peer_id, self.endpoint.host, self.endpoint.port)
            )

    def _require_root(self) -> None:
        if not self.is_root:
            raise ProtocolError("Message must be handled by the root custodian")

    def _append_join_event(self, event: str, peer_id: int) -> None:
        with (self.run_dir / "join_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "event": str(event),
                        "peer_id": int(peer_id),
                        "phase": self.phase,
                        "join_position": self.join_positions.get(int(peer_id)),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    async def _on_join(
        self, payload: Dict[str, Any], request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self._require_root()
        if self.phase == "FINALIZING":
            raise ProtocolError("The overlay is finalizing")
        peer_id = int(payload["peer_id"])
        endpoint = Endpoint.from_wire(payload["endpoint"])
        if peer_id != int(request["source_peer_id"]):
            raise ProtocolError("JOIN peer ID differs from the envelope source")
        if endpoint != Endpoint.from_wire(request["source_endpoint"]):
            raise ProtocolError("JOIN endpoint differs from the envelope source")
        existing = self.registry.get(peer_id)
        if existing is not None:
            if existing != endpoint:
                raise ProtocolError(f"Peer ID {peer_id} is already registered")
            return {
                "peer_id": peer_id,
                "join_position": self.join_positions[peer_id],
                "already_registered": True,
            }
        if endpoint in self.registry.values():
            raise ProtocolError("Listening endpoint is already registered to another peer")
        self.registry[peer_id] = endpoint
        self.endpoint_cache[peer_id] = endpoint
        self.join_positions[peer_id] = len(self.join_order)
        self.join_order.append(peer_id)
        # A peer admitted while the compatibility token is in flight may
        # serve ordinary RPCs, but it must not affect synchronous custody.
        # The complete queued set becomes eligible only after ACTIVE.
        if self.phase in {"BOOTSTRAP", "ACTIVE"}:
            self.custodian_ready.add(peer_id)
        self._append_join_event("registered", peer_id)
        return {
            "peer_id": peer_id,
            "join_position": self.join_positions[peer_id],
            "already_registered": False,
        }

    async def _on_peer_lookup(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self._require_root()
        peer_id = int(payload["peer_id"])
        if peer_id not in self.registry:
            raise ProtocolError(f"Unknown peer ID: {peer_id}")
        return {"peer_id": peer_id, "endpoint": self.registry[peer_id].to_wire()}

    async def _on_custodian_assign(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self._require_root()
        node_id = str(payload["node_id"])
        if node_id == self.root_id:
            custodian = self.root_peer_id
        elif node_id in self.custodian_assignments:
            custodian = self.custodian_assignments[node_id]
        else:
            custodian = custodian_for_peers(
                node_id, self.epoch, self.custodian_ready
            )
            self.custodian_assignments[node_id] = custodian
        return {
            "node_id": node_id,
            "custodian_peer_id": custodian,
            "custodian_endpoint": self.registry[custodian].to_wire(),
        }

    async def _on_registry_status(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self._require_root()
        requested = payload.get("peer_id")
        return {
            "phase": self.phase,
            "registered_count": len(self.registry),
            "ready_count": len(self.profile_ready),
            "tree_joined_count": len(self.tree_joined),
            "active_count": len(self.active_peers),
            "peer_registered": (
                int(requested) in self.registry if requested is not None else None
            ),
            "peer_ready": (
                int(requested) in self.profile_ready if requested is not None else None
            ),
        }

    async def _on_profile_ready(
        self, payload: Dict[str, Any], request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self._require_root()
        peer_id = int(payload["peer_id"])
        if peer_id != int(request["source_peer_id"]) or peer_id not in self.registry:
            raise ProtocolError("PROFILE_READY source is not a registered peer")
        self.profile_ready.add(peer_id)
        self._append_join_event("profile_ready", peer_id)
        if self.phase == "ACTIVE":
            self._ensure_admission_task()
        return {"ready_count": len(self.profile_ready)}

    async def _on_activate_sync(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self._require_root()
        if self.phase != "BOOTSTRAP":
            raise ProtocolError(f"Cannot activate from phase {self.phase}")
        expected_count = int(payload["cohort_size"])
        if len(self.registry) < expected_count:
            raise ProtocolError(
                f"Registered peer count {len(self.registry)} < {expected_count}"
            )
        initial_cohort = sorted(self.registry)[:expected_count]
        if not set(initial_cohort).issubset(self.profile_ready):
            missing = sorted(set(initial_cohort).difference(self.profile_ready))
            raise ProtocolError(f"Profiles are not ready: {missing}")
        # Process creation is issued in centralized order, but operating-system
        # startup can complete out of order. The compatibility cohort is
        # therefore canonicalized by staged peer position before token links
        # are installed. No peer receives this complete ordering.
        self.initial_cohort = initial_cohort
        self.join_order = list(self.initial_cohort)
        self.join_order.extend(
            peer_id for peer_id in sorted(self.registry) if peer_id not in self.initial_cohort
        )
        self.join_positions = {
            peer_id: position for position, peer_id in enumerate(self.join_order)
        }
        self.custodian_ready = set(self.initial_cohort)
        self.checkpoint_state_on_active = bool(payload.get("checkpoint_state", False))
        self.phase = "SYNC_BUILD"
        self.coordinator_task = self._spawn_background(self._coordinate_protocol())
        return {"cohort_size": expected_count, "first_peer_id": self.initial_cohort[0]}

    async def _on_sync_link(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self.sync_position = int(payload["position"])
        self.sync_size = int(payload["cohort_size"])
        raw_next = payload.get("next")
        if raw_next is None:
            self.sync_next_peer_id = None
            self.sync_next_endpoint = None
        else:
            self.sync_next_peer_id = int(raw_next["peer_id"])
            self.sync_next_endpoint = Endpoint.from_wire(raw_next["endpoint"])
            self.endpoint_cache[self.sync_next_peer_id] = self.sync_next_endpoint
        return {"peer_id": self.peer_id, "position": self.sync_position}

    async def _on_sync_token(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        token_id = str(payload["token_id"])
        if token_id in self.seen_sync_tokens:
            return {"token_id": token_id, "duplicate": True}
        if self.sync_position is None or int(payload["position"]) != self.sync_position:
            raise ProtocolError("Synchronization token arrived at the wrong peer")
        self.seen_sync_tokens.add(token_id)
        self._spawn_background(self._process_sync_token(dict(payload)))
        return {"token_id": token_id, "accepted": True}

    async def _process_sync_token(self, token: Dict[str, Any]) -> None:
        try:
            phase = str(token["phase"])
            if phase == "BUILD":
                await self._perform_build()
            elif phase == "STATS":
                token["maximum_tree_depth"] = max(
                    int(token.get("maximum_tree_depth", 0)), self._maximum_local_depth()
                )
            elif phase == "ROUND0":
                self.phase = "SYNC_ORGANIZE"
                await self._build_round0(int(token["maximum_tree_depth"]))
            elif phase == "EXPANSION":
                self.phase = "SYNC_ORGANIZE"
                await self._expand_round(int(token["round"]))
            elif phase == "QUERY":
                self.phase = "SYNC_ORGANIZE"
                await self._run_local_query()
                if bool(token.get("checkpoint_state", False)):
                    await self._export_state()
            elif phase == "EXPORT":
                await self._export_state()
            else:
                raise ProtocolError(f"Unknown synchronization phase: {phase}")
            next_token = dict(token)
            next_token["position"] = int(token["position"]) + 1
            if self.sync_next_endpoint is not None:
                await self.rpc_endpoint(self.sync_next_endpoint, "SYNC_TOKEN", next_token)
            else:
                await self.rpc_endpoint(
                    self.root_endpoint,
                    "SYNC_PHASE_COMPLETE",
                    {
                        "phase_key": str(token["phase_key"]),
                        "token_id": str(token["token_id"]),
                        "maximum_tree_depth": int(token.get("maximum_tree_depth", 0)),
                    },
                )
        except Exception as exc:
            if not self.is_root:
                with contextlib.suppress(Exception):
                    await self.rpc_endpoint(
                        self.root_endpoint,
                        "SYNC_PHASE_FAILED",
                        {
                            "phase_key": str(token["phase_key"]),
                            "peer_id": self.peer_id,
                            "error": str(exc),
                        },
                    )
            raise

    async def _on_sync_phase_complete(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self._require_root()
        key = str(payload["phase_key"])
        waiter = self.sync_phase_waiters.get(key)
        if waiter is None or waiter.done():
            raise ProtocolError(f"No synchronization phase is waiting for {key}")
        waiter.set_result(dict(payload))
        return {"phase_key": key}

    async def _on_sync_phase_failed(
        self, payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self._require_root()
        key = str(payload["phase_key"])
        waiter = self.sync_phase_waiters.get(key)
        if waiter is not None and not waiter.done():
            waiter.set_exception(
                ProtocolError(
                    f"Sync phase {key} failed at peer {int(payload['peer_id'])}: "
                    f"{payload['error']}"
                )
            )
        return {"phase_key": key}

    async def _configure_sync_links(self) -> None:
        for position, peer_id in enumerate(self.initial_cohort):
            next_value = None
            if position + 1 < len(self.initial_cohort):
                next_peer = self.initial_cohort[position + 1]
                next_value = {
                    "peer_id": next_peer,
                    "endpoint": self.registry[next_peer].to_wire(),
                }
            await self.rpc(
                peer_id,
                "SYNC_LINK",
                {
                    "position": position,
                    "cohort_size": len(self.initial_cohort),
                    "next": next_value,
                },
            )

    async def _run_sync_phase(
        self, phase: str, *, round_index: Optional[int] = None, **payload: Any
    ) -> Dict[str, Any]:
        suffix = "" if round_index is None else f":{int(round_index)}"
        key = f"{phase}{suffix}"
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[Dict[str, Any]] = loop.create_future()
        self.sync_phase_waiters[key] = waiter
        token = {
            "token_id": f"{self.epoch}:{key}",
            "phase_key": key,
            "phase": phase,
            "position": 0,
            **payload,
        }
        if round_index is not None:
            token["round"] = int(round_index)
        first_peer = self.initial_cohort[0]
        await self.rpc(first_peer, "SYNC_TOKEN", token)
        timeout = float(self.parameters.get("rpc_timeout_seconds", 30.0)) * max(
            2, len(self.initial_cohort)
        )
        try:
            return await asyncio.wait_for(waiter, timeout=timeout)
        finally:
            self.sync_phase_waiters.pop(key, None)

    async def _coordinate_protocol(self) -> None:
        try:
            await self._configure_sync_links()
            self.phase = "SYNC_BUILD"
            started = time.monotonic()
            await self._run_sync_phase("BUILD")
            self.phase_timings["tree_build_seconds"] = time.monotonic() - started
            stats = await self._run_sync_phase("STATS", maximum_tree_depth=0)
            maximum_tree_depth = int(stats["maximum_tree_depth"])
            self.phase = "SYNC_ORGANIZE"
            started = time.monotonic()
            await self._run_sync_phase(
                "ROUND0", maximum_tree_depth=maximum_tree_depth
            )
            self.phase_timings["round0_seconds"] = time.monotonic() - started
            for round_index in range(1, int(self.parameters["expansion_rounds"]) + 1):
                started = time.monotonic()
                await self._run_sync_phase("EXPANSION", round_index=round_index)
                self.phase_timings[f"expansion_round_{round_index}_seconds"] = (
                    time.monotonic() - started
                )
            started = time.monotonic()
            await self._run_sync_phase(
                "QUERY", checkpoint_state=self.checkpoint_state_on_active
            )
            self.phase_timings["query_seconds"] = time.monotonic() - started
            self.tree_joined.update(self.initial_cohort)
            self.active_peers.update(self.initial_cohort)
            self.phase = "ACTIVE"
            self.custodian_ready.update(self.registry)
            (self.run_dir / "overlay_active.json").write_text(
                json.dumps(
                    {
                        "status": "active",
                        "epoch": self.epoch,
                        "initial_cohort_size": len(self.initial_cohort),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self._ensure_admission_task()
        except Exception:
            (self.run_dir / "root_failure.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            self.stop_event.set()
            raise

    def _ensure_admission_task(self) -> None:
        if not self.is_root or self.phase != "ACTIVE":
            return
        has_uninserted = any(
            peer_id in self.profile_ready and peer_id not in self.tree_joined
            for peer_id in self.join_order
        )
        has_organizable = (
            len(self.tree_joined) - 1 >= int(self.parameters["degree"])
            and any(
                peer_id in self.tree_joined and peer_id not in self.active_peers
                for peer_id in self.join_order
            )
        )
        if not has_uninserted and not has_organizable:
            return
        if self.admission_task is None or self.admission_task.done():
            self.admission_task = self._spawn_background(self._drain_admissions())
            self.admission_task.add_done_callback(
                lambda _task: asyncio.get_running_loop().call_soon(
                    self._ensure_admission_task
                )
            )

    async def _drain_admissions(self) -> None:
        async with self.admission_lock:
            pending = [
                peer_id
                for peer_id in self.join_order
                if peer_id in self.profile_ready and peer_id not in self.tree_joined
            ]
            for peer_id in pending:
                await self.rpc(peer_id, "BUILD_TURN", {"peer_id": peer_id})
                self.tree_joined.add(peer_id)
                self._append_join_event("tree_joined", peer_id)
            contact_pending = [
                peer_id
                for peer_id in self.join_order
                if peer_id in self.tree_joined and peer_id not in self.active_peers
            ]
            for peer_id in contact_pending:
                if len(self.tree_joined) - 1 < int(self.parameters["degree"]):
                    self._append_join_event("contacts_pending", peer_id)
                    continue
                maximum_depth = 0
                for candidate in self.join_order:
                    response = await self.rpc(candidate, "TREE_STATS", {})
                    maximum_depth = max(
                        maximum_depth,
                        int(response["payload"]["maximum_local_depth"]),
                    )
                await self.rpc(
                    peer_id,
                    "ROUND0_START",
                    {"maximum_tree_depth": maximum_depth},
                )
                for round_index in range(
                    1, int(self.parameters["expansion_rounds"]) + 1
                ):
                    await self.rpc(
                        peer_id, "EXPANSION_START", {"round": round_index}
                    )
                await self.rpc(peer_id, "QUERY_START", {})
                self.active_peers.add(peer_id)
                self._append_join_event("active", peer_id)

    async def _on_finalize(
        self, _payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self._require_root()
        if self.phase != "ACTIVE":
            raise ProtocolError(f"Cannot finalize from phase {self.phase}")
        if self.finalize_task is None:
            self.finalize_task = self._spawn_background(self._finalize_protocol())
        return {"status": "finalizing"}

    async def _finalize_protocol(self) -> None:
        self.phase = "FINALIZING"
        if self.admission_task is not None:
            await self.admission_task
        await self._run_sync_phase("EXPORT")
        for peer_id in self.join_order:
            if peer_id not in self.initial_cohort:
                await self.rpc(peer_id, "EXPORT_STATE", {})
        registry_wire = {
            "schema": "decentralized_semantica_discovered_registry_v2",
            "epoch": self.epoch,
            "join_order": list(self.join_order),
            "peers": {
                str(peer_id): self.registry[peer_id].to_wire()
                for peer_id in self.join_order
            },
            "initial_cohort": list(self.initial_cohort),
            "tree_joined": sorted(self.tree_joined),
            "active_peers": sorted(self.active_peers),
        }
        (self.run_dir / "discovered_peer_registry.json").write_text(
            json.dumps(registry_wire, indent=2, sort_keys=True), encoding="utf-8"
        )
        (self.run_dir / "custodian_assignments.json").write_text(
            json.dumps(
                {
                    "schema": "decentralized_semantica_custody_v2",
                    "epoch": self.epoch,
                    "assignments": dict(sorted(self.custodian_assignments.items())),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
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
        for peer_id in self.join_order:
            if peer_id != self.peer_id:
                await self.rpc(peer_id, "SHUTDOWN", {})
        self.stop_event.set()

    async def _on_build_turn(self, payload: Dict[str, Any], _request: Mapping[str, Any]) -> Dict[str, Any]:
        if int(payload["peer_id"]) != self.peer_id or self.profile is None:
            raise ProtocolError("Invalid build turn")
        await self._perform_build()
        return {"peer_id": self.peer_id, "inserted": len(self.profile.centroids)}

    async def _perform_build(self) -> None:
        if self.profile is None:
            raise ProtocolError("Profile is not ready for tree insertion")
        if self.inserted_profile:
            return
        self.phase = "SYNC_BUILD" if self.sync_position is not None else "ACTIVE"
        for centroid_index, vector in enumerate(self.profile.centroids):
            entry = CentroidEntry(
                peer_id=self.peer_id,
                centroid_index=centroid_index,
                vector=vector,
                profile_version=self.profile.version,
                endpoint=self.endpoint,
            )
            await self._insert_entry(entry)
        self.inserted_profile = True

    async def _descriptor(
        self, node_id: str, center: Optional[np.ndarray] = None
    ) -> ChildDescriptor:
        if node_id == self.root_id:
            return ChildDescriptor(
                node_id,
                self.root_peer_id,
                self.root_endpoint,
                center=center,
            )
        response = await self.rpc_endpoint(
            self.root_endpoint, "CUSTODIAN_ASSIGN", {"node_id": node_id}
        )
        value = response["payload"]
        endpoint = Endpoint.from_wire(value["custodian_endpoint"])
        custodian = int(value["custodian_peer_id"])
        self.endpoint_cache[custodian] = endpoint
        return ChildDescriptor(node_id, custodian, endpoint, center=center)

    async def _get_node(self, descriptor: ChildDescriptor) -> TreeNode:
        response = await self.rpc_endpoint(
            descriptor.custodian_endpoint,
            "TREE_NODE_GET",
            {"node_id": descriptor.node_id},
        )
        return node_from_wire(response["payload"]["node"])

    async def _locate_leaf(self, vector: np.ndarray) -> tuple[ChildDescriptor, LeafNode]:
        descriptor = await self._descriptor(self.root_id)
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
        parent_descriptor = await self._descriptor(node.node_id)
        left = await self._descriptor(left_id, left_center)
        right = await self._descriptor(right_id, right_center)
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
        # Retained as a compatibility RPC; v2 pauses mutation by phase instead
        # of permanently freezing the tree.
        return {
            "peer_id": self.peer_id,
            "node_count": len(self.nodes),
            "maximum_local_depth": self._maximum_local_depth(),
        }

    def _maximum_local_depth(self) -> int:
        return max((int(node.depth) for node in self.nodes.values()), default=0)

    async def _on_tree_stats(
        self, _payload: Dict[str, Any], _request: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "node_count": len(self.nodes),
            "maximum_local_depth": self._maximum_local_depth(),
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
        self.phase = "SYNC_ORGANIZE" if self.sync_position is not None else "ACTIVE"
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
        self.phase = "SYNC_ORGANIZE" if self.sync_position is not None else "ACTIVE"
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
        return await self._run_local_query()

    async def _run_local_query(self) -> Dict[str, Any]:
        self.phase = "SYNC_ORGANIZE" if self.sync_position is not None else "ACTIVE"
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
        return await self._export_state()

    async def _export_state(self) -> Dict[str, Any]:
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
    bootstrap_wire: Optional[Mapping[str, Any]] = None,
) -> None:
    bootstrap = (
        BootstrapConfig.from_wire(bootstrap_wire)
        if bootstrap_wire is not None
        else None
    )
    runtime = PeerRuntime(
        peer_id,
        Path(shard_path),
        Path(run_dir),
        endpoint_queue,
        bootstrap=bootstrap,
    )
    asyncio.run(runtime.serve())


def run_joining_peer(
    peer_id: int,
    shard_path: str,
    run_dir: str,
    bootstrap_wire: Mapping[str, Any],
) -> None:
    run_peer_process(peer_id, shard_path, run_dir, None, bootstrap_wire)

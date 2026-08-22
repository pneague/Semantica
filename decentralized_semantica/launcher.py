from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
import multiprocessing as mp
import os
import platform
import queue
import shutil
import sys
import time
import traceback
from collections import Counter
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import sklearn

from .algorithms import genesis_custody_proof, root_node_id
from .models import BootstrapConfig, ChildDescriptor, Endpoint, PeerDescriptor
from .peer import run_peer_process
from .protocol import (
    ProtocolError,
    RpcClient,
    envelope,
    pack_message,
    read_frame,
    unpack_message,
    validate_envelope,
    write_frame,
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(dict(value), indent=2, sort_keys=True), encoding="utf-8"
    )


def _runtime_environment() -> Dict[str, Any]:
    configuration = io.StringIO()
    with contextlib.redirect_stdout(configuration):
        np.show_config()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "msgpack": package_version("msgpack"),
        "numpy_build_configuration": configuration.getvalue(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


async def _launcher_call(
    bootstrap: BootstrapConfig,
    log_path: Path,
    message_type: str,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    log_stream = log_path.open("a", encoding="utf-8")

    def log(direction: str, message: Mapping[str, Any], target: Endpoint, size: int) -> None:
        source_endpoint = Endpoint.from_wire(message["source_endpoint"])
        destination_endpoint = target if direction == "send" else Endpoint("127.0.0.1", 0)
        destination_peer_id = bootstrap.root_peer_id if direction == "send" else -1
        row = {
            "time": time.time(),
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
            "payload_bytes": len(pack_message(message.get("payload", {}))),
            "phase": "LAUNCHER_CONTROL",
        }
        log_stream.write(json.dumps(row, sort_keys=True) + "\n")
        log_stream.flush()

    client = RpcClient(
        source_peer_id=-1,
        source_endpoint=Endpoint("127.0.0.1", 0),
        epoch=bootstrap.epoch,
        timeout=float(bootstrap.parameters.get("rpc_timeout_seconds", 30.0)),
        log=log,
    )
    try:
        return await client.call(bootstrap.root_endpoint, message_type, payload)
    finally:
        log_stream.close()


async def _configure_root(
    bootstrap: BootstrapConfig,
    log_path: Path,
) -> Dict[str, Any]:
    return await _launcher_call(
        bootstrap,
        log_path,
        "CONFIGURE",
        {"bootstrap": bootstrap.to_wire()},
    )


class CompatibilityCoordinator:
    """External total-order harness; it is deliberately not a Semantica peer."""

    def __init__(
        self,
        bootstrap: BootstrapConfig,
        peers: Sequence[PeerDescriptor],
        log_path: Path,
    ) -> None:
        self.bootstrap = bootstrap
        self.peers = list(peers)
        self.log_path = Path(log_path)
        self.endpoint = Endpoint("127.0.0.1", 0)
        self.server: asyncio.AbstractServer | None = None
        self.waiters: Dict[str, asyncio.Future[Dict[str, Any]]] = {}

    def _log(
        self, direction: str, message: Mapping[str, Any], target: Endpoint, size: int
    ) -> None:
        row = {
            "time": time.time(),
            "direction": direction,
            "remote": target.to_wire(),
            "source_peer_id": int(message["source_peer_id"]),
            "source_endpoint": message["source_endpoint"],
            "destination_peer_id": (
                next(
                    (
                        peer.peer_id
                        for peer in self.peers
                        if peer.endpoint == target
                    ),
                    None,
                )
                if direction == "send"
                else -1
            ),
            "destination_endpoint": (
                target.to_wire() if direction == "send" else self.endpoint.to_wire()
            ),
            "message_type": message.get("message_type"),
            "message_id": message.get("message_id"),
            "request_id": message.get("request_id"),
            "bytes": int(size),
            "payload_bytes": len(pack_message(message.get("payload", {}))),
            "phase": "EXTERNAL_COMPATIBILITY_HARNESS",
            "sync_phase_key": message.get("payload", {}).get("phase_key"),
            "sync_position": message.get("payload", {}).get("position"),
        }
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def _client(self) -> RpcClient:
        return RpcClient(
            source_peer_id=-1,
            source_endpoint=self.endpoint,
            epoch=self.bootstrap.epoch,
            timeout=float(self.bootstrap.parameters.get("rpc_timeout_seconds", 30.0)),
            log=self._log,
        )

    async def __aenter__(self) -> "CompatibilityCoordinator":
        self.server = await asyncio.start_server(self._handle_client, "127.0.0.1", 0)
        socket = self.server.sockets[0]
        self.endpoint = Endpoint(str(socket.getsockname()[0]), int(socket.getsockname()[1]))
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        request: Dict[str, Any] | None = None
        remote = writer.get_extra_info("peername")
        target = Endpoint(str(remote[0]), int(remote[1])) if remote else Endpoint("unknown", 0)
        try:
            request, size = await read_frame(reader)
            validate_envelope(request, self.bootstrap.epoch)
            target = Endpoint.from_wire(request["source_endpoint"])
            self._log("receive", request, target, size)
            message_type = str(request["message_type"])
            key = str(request["payload"]["phase_key"])
            waiter = self.waiters.get(key)
            if waiter is None or waiter.done():
                raise ProtocolError(f"No external phase is waiting for {key}")
            if message_type == "SYNC_PHASE_COMPLETE":
                waiter.set_result(dict(request["payload"]))
            elif message_type == "SYNC_PHASE_FAILED":
                waiter.set_exception(
                    ProtocolError(
                        f"Compatibility phase {key} failed at peer "
                        f"{request['payload']['peer_id']}: {request['payload']['error']}"
                    )
                )
            else:
                raise ProtocolError(f"Coordinator rejects {message_type}")
            response = envelope(
                f"{message_type}_ACK",
                epoch=self.bootstrap.epoch,
                source_peer_id=-1,
                source_endpoint=self.endpoint,
                payload={"phase_key": key},
                request_id=str(request["message_id"]),
            )
        except Exception as exc:
            response = envelope(
                "ERROR",
                epoch=self.bootstrap.epoch,
                source_peer_id=-1,
                source_endpoint=self.endpoint,
                payload={"error": str(exc)},
                request_id=str(request.get("message_id")) if request else None,
            )
        sent = await write_frame(writer, response)
        self._log("send", response, target, sent)
        writer.close()
        await writer.wait_closed()

    async def link(self) -> None:
        for position, peer in enumerate(self.peers):
            next_peer = self.peers[position + 1] if position + 1 < len(self.peers) else None
            await self._client().call(
                peer.endpoint,
                "SYNC_LINK",
                {
                    "position": position,
                    "cohort_size": len(self.peers),
                    "next": next_peer.to_wire() if next_peer else None,
                    "coordinator_endpoint": self.endpoint.to_wire(),
                },
            )

    async def phase(
        self, phase: str, *, round_index: int | None = None, **payload: Any
    ) -> Dict[str, Any]:
        suffix = "" if round_index is None else f":{round_index}"
        key = f"{phase}{suffix}"
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[Dict[str, Any]] = loop.create_future()
        self.waiters[key] = waiter
        token = {
            "token_id": f"{self.bootstrap.epoch}:{key}",
            "phase_key": key,
            "phase": phase,
            "position": 0,
            **payload,
        }
        if round_index is not None:
            token["round"] = int(round_index)
        await self._client().call(self.peers[0].endpoint, "SYNC_TOKEN", token)
        timeout = float(self.bootstrap.parameters.get("rpc_timeout_seconds", 30.0)) * max(2, len(self.peers))
        try:
            return await asyncio.wait_for(waiter, timeout=timeout)
        finally:
            self.waiters.pop(key, None)

    async def release(self) -> None:
        for peer in self.peers:
            await self._client().call(peer.endpoint, "SYNC_RELEASE", {})


async def _run_compatibility_protocol(
    bootstrap: BootstrapConfig,
    peers: Sequence[PeerDescriptor],
    log_path: Path,
    *,
    release_active: bool,
) -> Dict[str, float]:
    timings: Dict[str, float] = {}
    async with CompatibilityCoordinator(bootstrap, peers, log_path) as coordinator:
        await coordinator.link()
        started = time.monotonic()
        await coordinator.phase("BUILD")
        timings["tree_build_seconds"] = time.monotonic() - started
        stats = await coordinator.phase("STATS", maximum_tree_depth=0)
        maximum_depth = int(stats["maximum_tree_depth"])
        started = time.monotonic()
        await coordinator.phase("ROUND0", maximum_tree_depth=maximum_depth)
        timings["round0_seconds"] = time.monotonic() - started
        for round_index in range(1, int(bootstrap.parameters["expansion_rounds"]) + 1):
            started = time.monotonic()
            await coordinator.phase("EXPANSION", round_index=round_index)
            timings[f"expansion_round_{round_index}_seconds"] = time.monotonic() - started
        started = time.monotonic()
        await coordinator.phase("QUERY")
        timings["query_seconds"] = time.monotonic() - started
        await coordinator.phase("EXPORT")
        if release_active:
            await coordinator.release()
    return timings


def _terminate_processes(processes: Iterable[mp.Process]) -> None:
    values = list(processes)
    for process in values:
        if process.is_alive():
            process.terminate()
    for process in values:
        process.join(timeout=5.0)
    for process in values:
        if process.is_alive():
            process.kill()
    for process in values:
        process.join(timeout=5.0)


def _wait_peer_event(
    endpoint_queue: Any,
    processes: Sequence[mp.Process],
    expected_event: str,
    expected_peer_id: int,
    timeout_seconds: float,
) -> Endpoint:
    deadline = time.monotonic() + float(timeout_seconds)
    deferred: list[tuple[Any, ...]] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            event = tuple(endpoint_queue.get(timeout=min(1.0, remaining)))
        except queue.Empty:
            failed = [process.exitcode for process in processes if process.exitcode is not None]
            if failed:
                raise RuntimeError(f"A peer exited during startup: {failed}")
            continue
        if len(event) != 4:
            raise RuntimeError(f"Malformed peer startup event: {event!r}")
        kind, peer_id, host, port = event
        if str(kind) == str(expected_event) and int(peer_id) == int(expected_peer_id):
            for value in deferred:
                endpoint_queue.put(value)
            return Endpoint(str(host), int(port))
        deferred.append(event)
    for value in deferred:
        endpoint_queue.put(value)
    raise TimeoutError(
        f"Peer {expected_peer_id} did not report {expected_event} before timeout"
    )


def _collect_ready_peers(
    endpoint_queue: Any,
    processes: Sequence[mp.Process],
    expected_peer_ids: Sequence[int],
    timeout_seconds: float,
) -> None:
    expected = {int(peer_id) for peer_id in expected_peer_ids}
    ready: set[int] = set()
    deadline = time.monotonic() + float(timeout_seconds)
    while ready != expected:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            event = tuple(endpoint_queue.get(timeout=min(1.0, remaining)))
        except queue.Empty:
            failed = {
                process.name: process.exitcode
                for process in processes
                if process.exitcode not in (None, 0)
            }
            if failed:
                raise RuntimeError(f"Peer startup failures: {failed}")
            continue
        if len(event) != 4:
            raise RuntimeError(f"Malformed peer startup event: {event!r}")
        kind, peer_id, _host, _port = event
        peer_id = int(peer_id)
        if peer_id not in expected:
            raise RuntimeError(f"Unexpected peer startup event: {event!r}")
        if str(kind) == "READY":
            ready.add(peer_id)
    if ready != expected:
        raise TimeoutError(
            f"Only {len(ready)}/{len(expected)} peers completed profile setup"
        )


def load_peer_states(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    states: Dict[int, Dict[str, Any]] = {}
    for path in sorted((Path(run_dir) / "state").glob("peer_*.msgpack")):
        state = unpack_message(path.read_bytes())
        states[int(state["peer_id"])] = state
    return states


def aggregate_run_outputs(
    run_dir: Path, peers: Mapping[int, Endpoint], epoch: str
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    states = load_peer_states(run_dir)
    expected = set(peers)
    if set(states) != expected:
        missing = sorted(expected.difference(states))
        raise RuntimeError(f"Missing exported peer states: {missing}")

    profile_offsets = np.zeros(len(states) + 1, dtype=np.int64)
    profile_begin_by_peer: Dict[int, int] = {}
    centers = []
    counts = []
    caps = []
    for position, peer_id in enumerate(sorted(states)):
        profile_begin_by_peer[peer_id] = int(profile_offsets[position])
        profile = states[peer_id]["profile"]
        peer_centers = np.asarray(profile["centroids"], dtype=np.float32)
        centers.append(peer_centers)
        counts.append(np.asarray(profile["cluster_doc_counts"], dtype=np.int32))
        caps.append(np.asarray(profile["contact_caps"], dtype=np.int32))
        profile_offsets[position + 1] = profile_offsets[position] + len(peer_centers)
    np.savez_compressed(
        run_dir / "centroid_profiles.npz",
        peer_ids=np.asarray(sorted(states), dtype=np.int32),
        offsets=profile_offsets,
        centroids=np.vstack(centers),
        cluster_doc_counts=np.concatenate(counts),
        contact_caps=np.concatenate(caps),
    )
    with (run_dir / "centroid_profile_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "peer_id",
                "centroid_index",
                "flat_centroid_index",
                "cluster_doc_count",
                "contact_quota",
                "profile_version",
                "host",
                "port",
            ),
        )
        writer.writeheader()
        for peer_id in sorted(states):
            profile = states[peer_id]["profile"]
            begin = profile_begin_by_peer[peer_id]
            for centroid_index, (count, cap) in enumerate(
                zip(profile["cluster_doc_counts"], profile["contact_caps"])
            ):
                writer.writerow(
                    {
                        "peer_id": peer_id,
                        "centroid_index": centroid_index,
                        "flat_centroid_index": begin + centroid_index,
                        "cluster_doc_count": int(count),
                        "contact_quota": int(cap),
                        "profile_version": int(profile["version"]),
                        "host": profile["endpoint"]["host"],
                        "port": int(profile["endpoint"]["port"]),
                    }
                )

    nodes: Dict[str, Dict[str, Any]] = {}
    node_custodians: Dict[str, int] = {}
    for peer_id, state in states.items():
        for node_id, node in state["nodes"].items():
            if node_id in nodes:
                raise RuntimeError(f"Tree node {node_id} was exported by multiple custodians")
            nodes[node_id] = node
            node_custodians[node_id] = peer_id
    root_id = root_node_id(epoch)
    canonical_nodes: Dict[str, Dict[str, Any]] = {}
    visited: set[str] = set()

    def walk(node_id: str, path: str) -> None:
        if node_id in visited:
            raise RuntimeError(f"Tree node cycle or duplicate reference at {node_id}")
        visited.add(node_id)
        node = nodes[node_id]
        row = {
            **node,
            "canonical_path": path or "root",
            "custodian_peer_id": node_custodians[node_id],
        }
        canonical_nodes[path or "root"] = row
        if node["node_type"] == "internal":
            walk(str(node["left"]["node_id"]), path + "L")
            walk(str(node["right"]["node_id"]), path + "R")

    walk(root_id, "")
    if visited != set(nodes):
        raise RuntimeError(f"Exported tree contains {len(set(nodes) - visited)} unreachable nodes")
    (run_dir / "tree_snapshot.msgpack").write_bytes(pack_message(canonical_nodes))
    custody_audit = {
        path: {
            "node_id": node["node_id"],
            "custodian_peer_id": int(node["custodian_peer_id"]),
            "custody_seed": int(node["custody_seed"]),
            "custody_proof": node["custody_proof"],
        }
        for path, node in canonical_nodes.items()
    }
    (run_dir / "custody_audit.msgpack").write_bytes(pack_message(custody_audit))
    leaf_nodes = [node for node in nodes.values() if node["node_type"] == "leaf"]
    _write_json(
        run_dir / "tree_snapshot_summary.json",
        {
            "root_node_id": root_id,
            "node_count": len(nodes),
            "internal_node_count": len(nodes) - len(leaf_nodes),
            "leaf_node_count": len(leaf_nodes),
            "maximum_depth": max((int(node["depth"]) for node in nodes.values()), default=0),
            "leaf_entry_count": sum(len(node["entries"]) for node in leaf_nodes),
            "reported_unsplittable_overflow_count": sum(
                node.get("failed_split_size") is not None for node in leaf_nodes
            ),
        },
    )

    contacts = {
        str(peer_id): states[peer_id]["contacts_by_round"] for peer_id in sorted(states)
    }
    (run_dir / "contacts_rounds_0_20.msgpack").write_bytes(pack_message(contacts))
    # The combined 100-query result corpus is intentionally larger than the
    # 16 MiB network frame ceiling. Keep that ceiling strict and export one
    # bounded artifact per source peer instead.
    query_dir = run_dir / "query_paths_results"
    query_dir.mkdir(parents=True, exist_ok=True)
    query_index: Dict[str, str] = {}
    for peer_id in sorted(states):
        path = query_dir / f"peer_{peer_id:05d}.msgpack"
        path.write_bytes(pack_message(states[peer_id]["queries"]))
        query_index[str(peer_id)] = str(path.relative_to(run_dir))
    _write_json(run_dir / "query_paths_results_index.json", query_index)

    type_counts: Counter[str] = Counter()
    type_bytes: Counter[str] = Counter()
    type_payload_bytes: Counter[str] = Counter()
    for log_path in sorted((run_dir / "logs").glob("*.jsonl")):
        with log_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("direction") != "send":
                    continue
                message_type = str(row["message_type"])
                type_counts[message_type] += 1
                type_bytes[message_type] += int(row["bytes"])
                type_payload_bytes[message_type] += int(row.get("payload_bytes", 0))
    network_summary = {
        "messages_by_type": dict(sorted(type_counts.items())),
        "wire_bytes_by_type": dict(sorted(type_bytes.items())),
        "payload_bytes_by_type": dict(sorted(type_payload_bytes.items())),
        "message_total": int(sum(type_counts.values())),
        "wire_bytes_total": int(sum(type_bytes.values())),
        "payload_bytes_total": int(sum(type_payload_bytes.values())),
    }
    _write_json(run_dir / "network_summary.json", network_summary)
    _write_json(
        run_dir / "phase_timings.json",
        {
            str(peer_id): states[peer_id].get("phase_timings", {})
            for peer_id in sorted(states)
        },
    )
    return {
        "peer_state_count": len(states),
        "tree_node_count": len(nodes),
        "centroid_count": int(profile_offsets[-1]),
        **network_summary,
    }


def run_experiment(
    prepared_dir: Path,
    run_dir: Path,
    *,
    endpoint_timeout_seconds: float = 120.0,
    run_timeout_seconds: float = 7200.0,
    rpc_timeout_seconds: float = 30.0,
    start_method: str = "spawn",
    root_peer_id: int | None = None,
    release_active: bool = False,
) -> Dict[str, Any]:
    prepared_dir = Path(prepared_dir).resolve()
    run_dir = Path(run_dir).resolve()
    prepared = _read_json(prepared_dir / "prepared_manifest.json")
    shard_paths = sorted((prepared_dir / "shards").glob("peer_*.npz"))
    if not shard_paths:
        raise FileNotFoundError("Prepared directory contains no private peer shards")
    peer_ids: list[int] = []
    shards: Dict[int, Path] = {}
    for shard in shard_paths:
        with np.load(shard, allow_pickle=False) as values:
            peer_id = int(np.asarray(values["peer_id"]).item())
        if peer_id in shards:
            raise ValueError(f"Duplicate private shard for peer {peer_id}")
        peer_ids.append(peer_id)
        shards[peer_id] = shard
    peer_ids.sort()
    if peer_ids != list(range(len(peer_ids))):
        raise ValueError("Synchronous compatibility mode requires peer positions 0..N-1")
    selected_root = peer_ids[0] if root_peer_id is None else int(root_peer_id)
    if selected_root != peer_ids[0]:
        raise ValueError(
            "Synchronous compatibility mode requires the root to be first in peer order"
        )
    if (run_dir / "run_manifest.json").exists() or (run_dir / "state").exists():
        raise FileExistsError(f"Run directory already contains a run: {run_dir}")
    for name in ("logs", "errors", "state"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    context = mp.get_context(start_method)
    endpoint_queue = context.Queue()
    processes: list[mp.Process] = []
    started_at = time.time()
    try:
        parameters = dict(prepared["parameters"])
        parameters.pop("peer_count", None)
        parameters["rpc_timeout_seconds"] = float(rpc_timeout_seconds)
        parameters["compatibility_mode"] = True
        root_process = context.Process(
            name=f"semantica-peer-{selected_root}",
            target=run_peer_process,
            args=(
                selected_root,
                str(shards[selected_root]),
                str(run_dir),
                endpoint_queue,
            ),
        )
        root_process.start()
        processes.append(root_process)
        root_endpoint = _wait_peer_event(
            endpoint_queue,
            processes,
            "LISTENING",
            selected_root,
            endpoint_timeout_seconds,
        )
        root_peer = PeerDescriptor(
            selected_root, root_endpoint.host, root_endpoint.port, 1
        )
        root_id = root_node_id(str(prepared["epoch"]))
        root_proof = genesis_custody_proof(
            str(prepared["epoch"]), root_id, root_peer
        )
        bootstrap = BootstrapConfig(
            str(prepared["epoch"]),
            int(prepared["seed"]),
            ChildDescriptor(root_id, root_peer, 0, root_proof),
            parameters,
        )
        bootstrap_wire = bootstrap.to_wire()
        run_manifest: Dict[str, Any] = {
            "schema": "decentralized_semantica_run_v3",
            "implementation": "port_based_fully_decentralized_semantica_v3",
            "mode": "external_compatibility_experiment",
            "status": "running",
            "prepared_dir": str(prepared_dir),
            "run_dir": str(run_dir),
            "started_at_unix": started_at,
            "root_node_id": root_node_id(bootstrap.epoch),
            "root_peer_id": selected_root,
            "bootstrap": bootstrap_wire,
            "experiment_peer_count": len(peer_ids),
            "compatibility_order_policy": "sequential_private_shard_position",
            "sync_schedule": "external_coordinator_successor_token_chain",
            "release_active": bool(release_active),
            "launcher_role": "external_non_peer_compatibility_harness_and_process_supervisor",
            "prepared_manifest_schema": str(prepared["schema"]),
            "prepared_epoch": str(prepared["epoch"]),
            "prepared_seed": int(prepared["seed"]),
            "prepared_parameters": dict(prepared["parameters"]),
            "prepared_shards_sha256": str(prepared["shards_sha256"]),
            "runtime_environment": _runtime_environment(),
        }
        launcher_log = run_dir / "logs" / "launcher.jsonl"
        asyncio.run(_configure_root(bootstrap, launcher_log))
        _wait_peer_event(
            endpoint_queue,
            processes,
            "READY",
            selected_root,
            endpoint_timeout_seconds,
        )
        # Publish only the genesis descriptor and public algorithm parameters.
        _write_json(run_dir / "bootstrap.json", bootstrap_wire)
        _write_json(run_dir / "run_manifest.json", run_manifest)

        peer_descriptors = [root_peer]
        for peer_id in peer_ids[1:]:
            process = context.Process(
                name=f"semantica-peer-{peer_id}",
                target=run_peer_process,
                args=(
                    peer_id,
                    str(shards[peer_id]),
                    str(run_dir),
                    endpoint_queue,
                    bootstrap_wire,
                ),
            )
            process.start()
            processes.append(process)
            endpoint = _wait_peer_event(
                endpoint_queue,
                processes,
                "LISTENING",
                peer_id,
                endpoint_timeout_seconds,
            )
            peer_descriptors.append(
                PeerDescriptor(peer_id, endpoint.host, endpoint.port, 1)
            )
            _wait_peer_event(
                endpoint_queue,
                processes,
                "READY",
                peer_id,
                endpoint_timeout_seconds,
            )

        inventory = {
            "schema": "decentralized_semantica_experiment_inventory_v3",
            "runtime_input": False,
            "peer_order": [peer.peer_id for peer in peer_descriptors],
            "peers": {
                str(peer.peer_id): peer.to_wire() for peer in peer_descriptors
            },
        }
        _write_json(run_dir / "experiment_peer_inventory.json", inventory)
        timings = asyncio.run(
            asyncio.wait_for(
                _run_compatibility_protocol(
                    bootstrap,
                    peer_descriptors,
                    run_dir / "logs" / "compatibility_coordinator.jsonl",
                    release_active=release_active,
                ),
                timeout=float(run_timeout_seconds),
            )
        )
        if release_active:
            checkpoint_dir = run_dir / "compatibility_checkpoint"
            shutil.copytree(run_dir / "state", checkpoint_dir / "state")
            active_parameters = dict(bootstrap.parameters)
            active_parameters["compatibility_mode"] = False
            active_bootstrap = BootstrapConfig(
                bootstrap.epoch,
                bootstrap.seed,
                bootstrap.root,
                active_parameters,
            )
            bootstrap = active_bootstrap
            bootstrap_wire = active_bootstrap.to_wire()
            run_manifest["bootstrap"] = bootstrap_wire
            _write_json(run_dir / "bootstrap.json", bootstrap_wire)
            _write_json(
                run_dir / "overlay_released.json",
                {
                    "status": "active",
                    "epoch": bootstrap.epoch,
                    "compatibility_coordinator_closed": True,
                    "stop_request": str(run_dir / "stop_requested"),
                },
            )
            deadline = time.monotonic() + float(run_timeout_seconds)
            while not (run_dir / "stop_requested").exists():
                failed = {
                    process.name: process.exitcode
                    for process in processes
                    if process.exitcode not in (None, 0)
                }
                if failed:
                    raise RuntimeError(f"Released peer process failures: {failed}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Released overlay did not receive a local stop request")
                time.sleep(0.1)
        _terminate_processes(processes)
        failed = {
            process.name: process.exitcode
            for process in processes
            if process.exitcode not in (0, None)
        }
        if failed:
            raise RuntimeError(f"Peer processes failed during local stop: {failed}")
        if release_active:
            shutil.move(str(run_dir / "state"), str(run_dir / "released_state"))
            shutil.copytree(
                run_dir / "compatibility_checkpoint" / "state", run_dir / "state"
            )
        peers = {peer.peer_id: peer.endpoint for peer in peer_descriptors}
        aggregate = aggregate_run_outputs(run_dir, peers, bootstrap.epoch)
        _write_json(
            run_dir / "run_complete.json",
            {
                "status": "complete",
                "epoch": bootstrap.epoch,
                "coordinator_role": "external_compatibility_only",
                "phase_timings": timings,
            },
        )
        run_manifest.update(
            {
                "status": "complete",
                "finished_at_unix": time.time(),
                "wall_seconds": time.time() - started_at,
                "aggregate": aggregate,
            }
        )
        _write_json(run_dir / "run_manifest.json", run_manifest)
        return run_manifest
    except BaseException as exc:
        _terminate_processes(processes)
        failure = {
            "status": "failed",
            "failed_at_unix": time.time(),
            "wall_seconds": time.time() - started_at,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(run_dir / "launcher_failure.json", failure)
        raise
    finally:
        _terminate_processes(processes)
        endpoint_queue.close()
        endpoint_queue.join_thread()

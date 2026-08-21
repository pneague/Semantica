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
import sys
import time
import traceback
from collections import Counter
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import sklearn

from .algorithms import root_node_id
from .models import BootstrapConfig, Endpoint
from .peer import run_peer_process
from .protocol import RpcClient, pack_message, unpack_message


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


def _terminate_processes(processes: Iterable[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
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


def finalize_overlay(bootstrap_path: Path) -> Dict[str, Any]:
    bootstrap_path = Path(bootstrap_path).resolve()
    bootstrap = BootstrapConfig.from_wire(_read_json(bootstrap_path))
    return asyncio.run(
        _launcher_call(
            bootstrap,
            bootstrap_path.parent / "logs" / "finalizer.jsonl",
            "FINALIZE",
            {},
        )
    )


def run_experiment(
    prepared_dir: Path,
    run_dir: Path,
    *,
    endpoint_timeout_seconds: float = 120.0,
    run_timeout_seconds: float = 7200.0,
    rpc_timeout_seconds: float = 30.0,
    start_method: str = "spawn",
    root_peer_id: int | None = None,
    keep_active: bool = False,
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
        parameters["rpc_timeout_seconds"] = float(rpc_timeout_seconds)
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
        bootstrap = BootstrapConfig(
            epoch=str(prepared["epoch"]),
            seed=int(prepared["seed"]),
            root_peer_id=selected_root,
            root_endpoint=root_endpoint,
            parameters=parameters,
        )
        bootstrap_wire = bootstrap.to_wire()
        run_manifest: Dict[str, Any] = {
            "schema": "decentralized_semantica_run_v2",
            "implementation": "port_based_decentralized_semantica_v2",
            "status": "running",
            "prepared_dir": str(prepared_dir),
            "run_dir": str(run_dir),
            "started_at_unix": started_at,
            "root_node_id": root_node_id(bootstrap.epoch),
            "root_peer_id": selected_root,
            "bootstrap": bootstrap_wire,
            "initial_cohort_size": len(peer_ids),
            "join_order_policy": "sequential_private_shard_position",
            "sync_schedule": "successor_token_chain",
            "keep_active": bool(keep_active),
            "launcher_role": "root_startup_ordered_peer_start_activation_wait_and_collection_only",
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
            "JOINED",
            selected_root,
            endpoint_timeout_seconds,
        )
        # Publish the out-of-band bootstrap only after CONFIGURE has created
        # the empty root node and the root's own JOIN has been acknowledged.
        _write_json(run_dir / "bootstrap.json", bootstrap_wire)
        _write_json(run_dir / "run_manifest.json", run_manifest)

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
            _wait_peer_event(
                endpoint_queue,
                processes,
                "JOINED",
                peer_id,
                endpoint_timeout_seconds,
            )
        _collect_ready_peers(
            endpoint_queue,
            processes,
            peer_ids,
            endpoint_timeout_seconds,
        )

        asyncio.run(
            _launcher_call(
                bootstrap,
                launcher_log,
                "ACTIVATE_SYNC",
                {
                    "cohort_size": len(peer_ids),
                    "checkpoint_state": bool(keep_active),
                },
            )
        )

        deadline = time.monotonic() + float(run_timeout_seconds)
        finalized = False
        while True:
            for process in processes:
                process.join(timeout=0.0)
            bad_exit_codes = {
                process.name: process.exitcode
                for process in processes
                if process.exitcode not in (None, 0)
            }
            if bad_exit_codes:
                raise RuntimeError(f"Peer process failures: {bad_exit_codes}")
            exited = [process.name for process in processes if process.exitcode is not None]
            run_complete = (run_dir / "run_complete.json").exists()
            overlay_active = (run_dir / "overlay_active.json").exists()
            if overlay_active and not keep_active and not finalized:
                asyncio.run(
                    _launcher_call(bootstrap, launcher_log, "FINALIZE", {})
                )
                finalized = True
            if exited and not run_complete:
                raise RuntimeError(
                    "Peer processes exited before root completion: " + ", ".join(exited)
                )
            if len(exited) == len(processes):
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        alive = [process.name for process in processes if process.is_alive()]
        if alive:
            raise TimeoutError(f"Peer processes did not finish before timeout: {alive}")
        if not (run_dir / "run_complete.json").exists():
            raise RuntimeError("Root custodian exited without declaring run completion")
        registry_wire = _read_json(run_dir / "discovered_peer_registry.json")
        peers = {
            int(peer_id): Endpoint.from_wire(endpoint)
            for peer_id, endpoint in registry_wire["peers"].items()
        }
        aggregate = aggregate_run_outputs(run_dir, peers, bootstrap.epoch)
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
        endpoint_queue.close()
        endpoint_queue.join_thread()

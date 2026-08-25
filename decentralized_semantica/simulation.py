from __future__ import annotations

import json
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any, Dict

from .dataset import prepare_demo_shards
from .models import Endpoint
from .peer import run_peer_process


def _wait_for_file(path: Path, processes: list[mp.Process], timeout: float) -> None:
    deadline = time.monotonic() + float(timeout)
    while not path.exists():
        failures = {
            process.name: process.exitcode
            for process in processes
            if process.exitcode not in (None, 0)
        }
        if failures:
            raise RuntimeError(f"Peer process failed during startup: {failures}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.05)


def _wait_for_startup(
    event_queue: Any,
    processes: list[mp.Process],
    peer_count: int,
    timeout: float,
) -> Dict[int, Endpoint]:
    endpoints: Dict[int, Endpoint] = {}
    joined: set[int] = set()
    deadline = time.monotonic() + float(timeout)
    while len(endpoints) < int(peer_count) or len(joined) < int(peer_count):
        failures = {
            process.name: process.exitcode
            for process in processes
            if process.exitcode not in (None, 0)
        }
        if failures:
            raise RuntimeError(f"Peer process failed during startup: {failures}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            expected = set(range(int(peer_count)))
            missing_listeners = sorted(expected.difference(endpoints))
            missing_tree_joins = sorted(expected.difference(joined))
            raise TimeoutError(
                "Peers did not complete startup; "
                f"missing listeners={missing_listeners}, "
                f"missing tree joins={missing_tree_joins}"
            )
        try:
            kind, peer_id, host, port = event_queue.get(timeout=min(0.2, remaining))
        except queue.Empty:
            continue
        if str(kind) == "LISTENING":
            endpoints[int(peer_id)] = Endpoint(str(host), int(port))
        elif str(kind) == "TREE_JOINED":
            joined.add(int(peer_id))
    return endpoints


def _stop_processes(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def run_startup_simulation(
    work_dir: Path,
    *,
    peer_count: int = 4,
    seed: int = 42,
    startup_timeout_seconds: float = 30.0,
    hold_seconds: float = 1.0,
    start_method: str = "spawn",
) -> Dict[str, Any]:
    """Start a synthetic decentralized network and stop after a short smoke run."""
    work_dir = Path(work_dir).resolve()
    prepared_dir = work_dir / "prepared"
    run_dir = work_dir / "run"
    manifest = prepare_demo_shards(
        prepared_dir, peer_count=int(peer_count), seed=int(seed)
    )
    parameters = dict(manifest["parameters"])
    parameters.pop("peer_count", None)
    parameters["rpc_timeout_seconds"] = 5.0

    context = mp.get_context(str(start_method))
    events = context.Queue()
    processes: list[mp.Process] = []
    try:
        bootstrap_path = run_dir / "bootstrap.json"
        bootstrap_path.unlink(missing_ok=True)
        root = context.Process(
            name="semantica-peer-0",
            target=run_peer_process,
            args=(
                0,
                str(prepared_dir / "shards" / "peer_00000.npz"),
                str(run_dir),
                events,
            ),
            kwargs={
                "root_initialization": {
                    "epoch": str(manifest["epoch"]),
                    "seed": int(manifest["seed"]),
                    "parameters": parameters,
                    "bootstrap_path": str(bootstrap_path),
                }
            },
        )
        root.start()
        processes.append(root)
        _wait_for_file(bootstrap_path, processes, float(startup_timeout_seconds))
        bootstrap = json.loads(
            bootstrap_path.read_text(encoding="utf-8")
        )

        for peer_id in range(1, int(peer_count)):
            process = context.Process(
                name=f"semantica-peer-{peer_id}",
                target=run_peer_process,
                args=(
                    peer_id,
                    str(prepared_dir / "shards" / f"peer_{peer_id:05d}.npz"),
                    str(run_dir),
                    events,
                    bootstrap,
                ),
            )
            process.start()
            processes.append(process)

        endpoints = _wait_for_startup(
            events,
            processes,
            int(peer_count),
            float(startup_timeout_seconds),
        )
        deadline = time.monotonic() + max(0.0, float(hold_seconds))
        while time.monotonic() < deadline:
            failures = {
                process.name: process.exitcode
                for process in processes
                if process.exitcode not in (None, 0)
            }
            if failures:
                raise RuntimeError(f"Peer process failed after startup: {failures}")
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return {
            "status": "started",
            "peer_count": int(peer_count),
            "listening_peer_ids": sorted(endpoints),
        }
    finally:
        _stop_processes(processes)
        events.close()
        events.join_thread()

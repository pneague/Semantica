from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .dataset import prepare_demo_shards


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m decentralized_semantica",
        description="Process-isolated decentralized Semantica simulation",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare-demo", help="Create deterministic synthetic private peer shards"
    )
    prepare.add_argument("--output-dir", type=_path, required=True)
    prepare.add_argument("--peer-count", type=int, default=4)
    prepare.add_argument("--documents-per-peer", type=int, default=4)
    prepare.add_argument("--dimensions", type=int, default=3)
    prepare.add_argument("--seed", type=int, default=42)

    simulate = commands.add_parser(
        "simulate", help="Start a local synthetic peer network and smoke-test startup"
    )
    simulate.add_argument("--work-dir", type=_path, required=True)
    simulate.add_argument("--peer-count", type=int, default=4)
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument("--startup-timeout-seconds", type=float, default=30.0)
    simulate.add_argument("--hold-seconds", type=float, default=1.0)
    simulate.add_argument(
        "--start-method",
        choices=("spawn", "forkserver", "fork"),
        default="spawn",
    )

    centralized = commands.add_parser(
        "centralized-simulate",
        help="Run the same shard/profile/contact/query workflow in one process",
    )
    centralized.add_argument("--work-dir", type=_path, required=True)
    centralized.add_argument("--peer-count", type=int, default=4)
    centralized.add_argument("--seed", type=int, default=42)
    centralized.add_argument(
        "--aol-bundle",
        type=_path,
        help="Optional compact AOL document bundle instead of synthetic shards",
    )

    start_root = commands.add_parser(
        "start-root", help="Start the genesis peer from a prepared demo"
    )
    start_root.add_argument("--prepared-dir", type=_path, required=True)
    start_root.add_argument("--peer-id", type=int, default=0)
    start_root.add_argument("--shard", type=_path)
    start_root.add_argument("--run-dir", type=_path, required=True)
    start_root.add_argument("--bootstrap", type=_path)
    start_root.add_argument("--rpc-timeout-seconds", type=float, default=30.0)

    join = commands.add_parser(
        "join", help="Start an independent peer and join through the genesis custodian"
    )
    join.add_argument("--bootstrap", type=_path, required=True)
    join.add_argument("--peer-id", type=int, required=True)
    join.add_argument("--shard", type=_path, required=True)
    join.add_argument("--run-dir", type=_path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result: dict[str, Any]
    if arguments.command == "prepare-demo":
        result = prepare_demo_shards(
            arguments.output_dir,
            peer_count=arguments.peer_count,
            documents_per_peer=arguments.documents_per_peer,
            dimensions=arguments.dimensions,
            seed=arguments.seed,
        )
    elif arguments.command == "simulate":
        from .simulation import run_startup_simulation

        result = run_startup_simulation(
            arguments.work_dir,
            peer_count=arguments.peer_count,
            seed=arguments.seed,
            startup_timeout_seconds=arguments.startup_timeout_seconds,
            hold_seconds=arguments.hold_seconds,
            start_method=arguments.start_method,
        )
    elif arguments.command == "centralized-simulate":
        from .centralized_simulation import run_centralized_simulation

        result = run_centralized_simulation(
            arguments.work_dir,
            peer_count=arguments.peer_count,
            seed=arguments.seed,
            aol_bundle=arguments.aol_bundle,
        )
    elif arguments.command == "join":
        from .models import BootstrapConfig
        from .peer import run_joining_peer

        bootstrap_wire = json.loads(arguments.bootstrap.read_text(encoding="utf-8"))
        BootstrapConfig.from_wire(bootstrap_wire)
        run_joining_peer(
            arguments.peer_id,
            str(arguments.shard),
            str(arguments.run_dir),
            bootstrap_wire,
        )
        result = {"status": "stopped", "peer_id": arguments.peer_id}
    elif arguments.command == "start-root":
        from .peer import run_root_peer

        prepared = json.loads(
            (arguments.prepared_dir / "prepared_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        shard = arguments.shard or (
            arguments.prepared_dir / "shards" / f"peer_{arguments.peer_id:05d}.npz"
        )
        bootstrap_path = arguments.bootstrap or (arguments.run_dir / "bootstrap.json")
        parameters = dict(prepared["parameters"])
        parameters.pop("peer_count", None)
        parameters["rpc_timeout_seconds"] = float(arguments.rpc_timeout_seconds)
        run_root_peer(
            arguments.peer_id,
            str(shard),
            str(arguments.run_dir),
            {
                "epoch": str(prepared["epoch"]),
                "seed": int(prepared["seed"]),
                "parameters": parameters,
                "bootstrap_path": str(bootstrap_path),
            },
        )
        result = {"status": "stopped", "peer_id": arguments.peer_id}
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(arguments.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

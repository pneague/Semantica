from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .dataset import (
    DEFAULT_PUBMEDQA_DATA_ROOT,
    DEFAULT_PUBMEDQA_OVERLAY_ROOT,
    prepare_pubmedqa_shards,
)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m decentralized_semantica",
        description="Port-based, custodian-only fully decentralized Semantica v3",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Stage private PubMedQA peer shards")
    prepare.add_argument("--output-dir", type=_path, required=True)
    prepare.add_argument("--data-root", type=_path, default=DEFAULT_PUBMEDQA_DATA_ROOT)
    prepare.add_argument("--overlay-root", type=_path, default=DEFAULT_PUBMEDQA_OVERLAY_ROOT)
    prepare.add_argument("--peer-count", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=42)

    compat = commands.add_parser(
        "compat-run", help="Run the external ordered centralized-parity experiment"
    )
    compat.add_argument("--prepared-dir", type=_path, required=True)
    compat.add_argument("--run-dir", type=_path, required=True)
    compat.add_argument("--endpoint-timeout-seconds", type=float, default=120.0)
    compat.add_argument("--run-timeout-seconds", type=float, default=7200.0)
    compat.add_argument("--rpc-timeout-seconds", type=float, default=30.0)
    compat.add_argument("--root-peer-id", type=int)
    compat.add_argument("--release-active", action="store_true")
    compat.add_argument(
        "--start-method",
        choices=("spawn", "forkserver", "fork"),
        default="spawn",
    )

    validate = commands.add_parser(
        "validate", help="Validate invariants and compare with the centralized oracle"
    )
    validate.add_argument("--prepared-dir", type=_path, required=True)
    validate.add_argument("--run-dir", type=_path, required=True)
    validate.add_argument("--structural-only", action="store_true")
    validate.add_argument(
        "--float32-atol",
        type=float,
        default=0.0,
        help="Diagnostic override; acceptance uses the default exact comparison",
    )
    validate.add_argument("--reference-run", type=_path)
    validate.add_argument("--prejoin-run", type=_path)

    start_root = commands.add_parser(
        "start-root", help="Start the genesis peer as an ordinary seed-0 custodian"
    )
    start_root.add_argument("--prepared-dir", type=_path, required=True)
    start_root.add_argument("--peer-id", type=int, default=0)
    start_root.add_argument("--shard", type=_path)
    start_root.add_argument("--run-dir", type=_path, required=True)
    start_root.add_argument("--bootstrap", type=_path)
    start_root.add_argument("--rpc-timeout-seconds", type=float, default=30.0)

    join = commands.add_parser(
        "join", help="Start an independent peer and join through the root custodian"
    )
    join.add_argument("--bootstrap", type=_path, required=True)
    join.add_argument("--peer-id", type=int, required=True)
    join.add_argument("--shard", type=_path, required=True)
    join.add_argument("--run-dir", type=_path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result: dict[str, Any]
    if arguments.command == "prepare":
        result = prepare_pubmedqa_shards(
            arguments.output_dir,
            data_root=arguments.data_root,
            overlay_root=arguments.overlay_root,
            peer_count=arguments.peer_count,
            seed=arguments.seed,
        )
    elif arguments.command == "compat-run":
        from .launcher import run_experiment

        result = run_experiment(
            arguments.prepared_dir,
            arguments.run_dir,
            endpoint_timeout_seconds=arguments.endpoint_timeout_seconds,
            run_timeout_seconds=arguments.run_timeout_seconds,
            rpc_timeout_seconds=arguments.rpc_timeout_seconds,
            start_method=arguments.start_method,
            root_peer_id=arguments.root_peer_id,
            release_active=arguments.release_active,
        )
    elif arguments.command == "validate":
        from .validation import validate_run

        result = validate_run(
            arguments.prepared_dir,
            arguments.run_dir,
            structural_only=arguments.structural_only,
            float32_atol=arguments.float32_atol,
            reference_run=arguments.reference_run,
            prejoin_run=arguments.prejoin_run,
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
        parameters["compatibility_mode"] = False
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

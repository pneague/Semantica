from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .launcher import load_peer_states
from .models import BootstrapConfig, Endpoint
from .protocol import pack_message, unpack_message


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_run_provenance(
    prepared_dir: Path,
    run_dir: Path,
    prepared: Mapping[str, Any],
    bootstrap_wire: Mapping[str, Any],
    bootstrap: BootstrapConfig,
) -> Dict[str, Any]:
    if prepared.get("schema") != "decentralized_semantica_prepared_v2":
        raise AssertionError("Unexpected prepared-manifest schema")
    if "peer_ids" in prepared or "endpoints" in prepared:
        raise AssertionError("Prepared manifest contains a static peer list")
    if bootstrap_wire.get("schema") != "decentralized_semantica_bootstrap_v2":
        raise AssertionError("Unexpected decentralized bootstrap schema")
    if bootstrap_wire.get("implementation") != "port_based_decentralized_semantica_v2":
        raise AssertionError("Bootstrap is not labelled as the decentralized implementation")
    if "peers" in bootstrap_wire or "peer_ids" in bootstrap_wire:
        raise AssertionError("Bootstrap configuration contains a peer list")
    if bootstrap.epoch != str(prepared["epoch"]):
        raise AssertionError("Prepared and runtime epochs differ")
    if bootstrap.seed != int(prepared["seed"]):
        raise AssertionError("Prepared and runtime seeds differ")
    runtime_parameters = dict(bootstrap.parameters)
    rpc_timeout = runtime_parameters.pop("rpc_timeout_seconds", None)
    if rpc_timeout is None or runtime_parameters != dict(prepared["parameters"]):
        raise AssertionError("Prepared and runtime algorithm parameters differ")

    digest = hashlib.sha256()
    shard_paths = sorted((prepared_dir / "shards").glob("peer_*.npz"))
    if not shard_paths:
        raise AssertionError("A prepared private shard is missing")
    for path in shard_paths:
        digest.update(path.read_bytes())
    observed_digest = digest.hexdigest()
    if observed_digest != str(prepared.get("shards_sha256", "")):
        raise AssertionError("Prepared private-shard digest differs")

    run_manifest = _read_json(run_dir / "run_manifest.json")
    if run_manifest.get("schema") != "decentralized_semantica_run_v2":
        raise AssertionError("Unexpected decentralized run-manifest schema")
    if run_manifest.get("implementation") != "port_based_decentralized_semantica_v2":
        raise AssertionError("Run is not labelled as the decentralized implementation")
    if run_manifest.get("status") != "complete":
        raise AssertionError("Decentralized run manifest is not complete")
    if Path(run_manifest["prepared_dir"]).resolve() != prepared_dir:
        raise AssertionError("Run manifest references a different prepared directory")
    for field, expected in (
        ("prepared_manifest_schema", prepared["schema"]),
        ("prepared_epoch", prepared["epoch"]),
        ("prepared_seed", int(prepared["seed"])),
        ("prepared_parameters", prepared["parameters"]),
        ("prepared_shards_sha256", observed_digest),
    ):
        if run_manifest.get(field) != expected:
            raise AssertionError(f"Run-manifest provenance field differs: {field}")
    if run_manifest.get("bootstrap") != dict(bootstrap_wire):
        raise AssertionError("Run and bootstrap configurations differ")
    if "membership" in run_manifest or "peer_ids" in run_manifest:
        raise AssertionError("Run manifest contains static membership")
    if not run_manifest.get("runtime_environment"):
        raise AssertionError("Run manifest lacks runtime environment metadata")
    return {
        "implementation": "port_based_decentralized_semantica_v2",
        "prepared_schema": str(prepared["schema"]),
        "bootstrap_schema": str(bootstrap_wire["schema"]),
        "epoch": bootstrap.epoch,
        "seed": bootstrap.seed,
        "parameters_match": True,
        "shards_sha256": observed_digest,
    }


def _contacts_as_pairs(values: Sequence[Sequence[Mapping[str, Any]]]) -> List[List[Tuple[int, float]]]:
    return [
        [(int(contact["peer_id"]), float(contact["score"])) for contact in centroid]
        for centroid in values
    ]


def _compare_contact_snapshots(
    observed: Sequence[Sequence[Mapping[str, Any]]],
    expected: Sequence[Sequence[Tuple[int, float]]],
    *,
    atol: float,
    context: str,
) -> None:
    actual = _contacts_as_pairs(observed)
    if len(actual) != len(expected):
        raise AssertionError(f"{context}: centroid count differs")
    for centroid_index, (actual_values, expected_values) in enumerate(zip(actual, expected)):
        actual_ids = [peer for peer, _score in actual_values]
        expected_ids = [int(peer) for peer, _score in expected_values]
        if actual_ids != expected_ids:
            raise AssertionError(
                f"{context} centroid {centroid_index}: contacts {actual_ids} != {expected_ids}"
            )
        if not np.allclose(
            [score for _peer, score in actual_values],
            [score for _peer, score in expected_values],
            rtol=0.0,
            atol=float(atol),
        ):
            raise AssertionError(f"{context} centroid {centroid_index}: scores differ")


def _structural_validation(
    prepared_dir: Path,
    run_dir: Path,
    bootstrap: BootstrapConfig,
    registry_wire: Mapping[str, Any],
    states: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    parameters = bootstrap.parameters
    peers = {
        int(peer_id): Endpoint.from_wire(endpoint)
        for peer_id, endpoint in registry_wire["peers"].items()
    }
    custody_wire = _read_json(run_dir / "custodian_assignments.json")
    assignments = {
        str(node_id): int(peer_id)
        for node_id, peer_id in custody_wire["assignments"].items()
    }
    expected_rounds = set(range(int(parameters["expansion_rounds"]) + 1))
    centroid_entries: List[Tuple[int, int]] = []
    overflows: List[Dict[str, Any]] = []
    node_count = 0
    for peer_id, state in states.items():
        if Endpoint.from_wire(state["endpoint"]) != peers[peer_id]:
            raise AssertionError(f"Peer {peer_id} exported the wrong endpoint")
        profile = state["profile"]
        caps = np.asarray(profile["contact_caps"], dtype=np.int32)
        if int(np.sum(caps)) != int(parameters["degree"]) or np.any(caps < 1):
            raise AssertionError(f"Peer {peer_id} has invalid centroid quotas")
        rounds = {int(value) for value in state["contacts_by_round"]}
        if rounds != expected_rounds:
            raise AssertionError(f"Peer {peer_id} has contact rounds {sorted(rounds)}")
        previous_provenance: Dict[int, str] = {}
        for round_index, by_centroid in sorted(
            state["contacts_by_round"].items(), key=lambda item: int(item[0])
        ):
            ids = [
                int(contact["peer_id"])
                for centroid in by_centroid
                for contact in centroid
            ]
            if len(ids) != int(parameters["degree"]):
                raise AssertionError(f"Peer {peer_id} round {round_index} degree is {len(ids)}")
            if peer_id in ids or len(ids) != len(set(ids)):
                raise AssertionError(f"Peer {peer_id} round {round_index} has self/duplicate contacts")
            current_provenance: Dict[int, str] = {}
            for centroid in by_centroid:
                for contact in centroid:
                    candidate = int(contact["peer_id"])
                    if Endpoint.from_wire(contact["endpoint"]) != peers[candidate]:
                        raise AssertionError(
                            f"Peer {peer_id} round {round_index} contact {candidate} endpoint differs"
                        )
                    expected_version = int(states[candidate]["profile"]["version"])
                    if int(contact["profile_version"]) != expected_version:
                        raise AssertionError(
                            f"Peer {peer_id} round {round_index} contact {candidate} profile version differs"
                        )
                    learned_at = str(contact["learned_at"])
                    expected_learned_at = previous_provenance.get(
                        candidate,
                        "round0" if int(round_index) == 0 else f"expansion_{round_index}",
                    )
                    if learned_at != expected_learned_at:
                        raise AssertionError(
                            f"Peer {peer_id} round {round_index} contact {candidate} provenance differs"
                        )
                    current_provenance[candidate] = learned_at
            previous_provenance = current_provenance
        queries = state["queries"]
        if len(queries) != 1:
            raise AssertionError(f"Peer {peer_id} exported {len(queries)} queries")
        query = next(iter(queries.values()))
        path = [int(value) for _hop, value in sorted(
            ((int(hop), value) for hop, value in query["peer_by_hop"].items())
        )]
        if peer_id in path or len(path) != len(set(path)):
            raise AssertionError(f"Peer {peer_id} query path repeats a peer")
        if len(path) > int(parameters["query_hop_limit"]):
            raise AssertionError(f"Peer {peer_id} exceeded the query hop limit")
        if len(query["final_results"]) > int(parameters["local_top_k"]):
            raise AssertionError(f"Peer {peer_id} exported too many merged results")
        for node_id, node in state["nodes"].items():
            node_count += 1
            expected_custodian = assignments.get(str(node_id))
            if expected_custodian != peer_id:
                raise AssertionError(
                    f"Node {node_id} is on peer {peer_id}, expected {expected_custodian}"
                )
            if node["node_type"] == "leaf":
                entries = [
                    (int(entry["peer_id"]), int(entry["centroid_index"]))
                    for entry in node["entries"]
                ]
                centroid_entries.extend(entries)
                if len(entries) > int(parameters["leaf_capacity"]):
                    if node.get("failed_split_size") != len(entries):
                        raise AssertionError(f"Leaf {node_id} has an unreported overflow")
                    overflows.append(
                        {"node_id": node_id, "entry_count": len(entries), "reported": True}
                    )
    expected_entries = {
        (peer_id, centroid_index)
        for peer_id, state in states.items()
        for centroid_index in range(len(state["profile"]["centroids"]))
    }
    if len(centroid_entries) != len(set(centroid_entries)):
        raise AssertionError("Delta-zero tree contains cloned centroid entries")
    if set(centroid_entries) != expected_entries:
        raise AssertionError("Canonical leaves do not contain exactly the fitted centroids")
    if (run_dir / "root_failure.log").exists():
        raise AssertionError("Root failure log exists")
    error_logs = [path for path in (run_dir / "errors").glob("*.log") if path.stat().st_size]
    if error_logs:
        raise AssertionError(f"Protocol error logs exist: {[path.name for path in error_logs]}")

    sends: Dict[str, int] = {}
    receives: Dict[str, int] = {}
    send_count = receive_count = 0
    for path in sorted((run_dir / "logs").glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                message_id = str(row["message_id"])
                if row["direction"] == "send":
                    sends[message_id] = sends.get(message_id, 0) + 1
                    send_count += 1
                elif row["direction"] == "receive":
                    receives[message_id] = receives.get(message_id, 0) + 1
                    receive_count += 1
    if sends != receives:
        missing_receive = sorted(set(sends).difference(receives))[:5]
        missing_send = sorted(set(receives).difference(sends))[:5]
        raise AssertionError(
            f"Send/receive correlation failed: missing_receive={missing_receive}, "
            f"missing_send={missing_send}"
        )
    return {
        "peer_count": len(states),
        "node_count": node_count,
        "centroid_entry_count": len(centroid_entries),
        "reported_unsplittable_overflow_count": len(overflows),
        "send_log_count": send_count,
        "receive_log_count": receive_count,
        "private_shard_count": len(list((prepared_dir / "shards").glob("peer_*.npz"))),
    }


def _central_dataset(prepared: Mapping[str, Any], fixture: Mapping[str, np.ndarray]) -> Any:
    import pandas as pd
    from unified_retrieval_benchmark import EmbeddingBundle, PreparedDataset

    peer_ids = np.asarray(fixture["peer_ids"], dtype=np.int32)
    inventories = np.asarray(fixture["training_doc_indices"], dtype=np.int32)
    offsets = np.arange(len(peer_ids) + 1, dtype=np.int64) * inventories.shape[1]
    bundle = EmbeddingBundle(**dict(prepared["embedding_bundle"]))
    query_indices = np.asarray(fixture["query_indices"], dtype=np.int64)
    target_indices = np.asarray(fixture["target_doc_indices"], dtype=np.int64)
    frame = pd.DataFrame(
        {
            "manifest_row": np.arange(len(peer_ids), dtype=np.int64),
            "source_peer_position": np.arange(len(peer_ids), dtype=np.int32),
            "query_embedding_index": query_indices,
            "target_doc_index": target_indices,
            "target_available_in_training_network": np.asarray(
                fixture["target_available"], dtype=bool
            ),
        }
    )
    return PreparedDataset(
        dataset_name="pubmedqa",
        seed=int(prepared["seed"]),
        peer_ids=peer_ids,
        inventory_offsets=offsets,
        inventory_doc_indices=inventories.reshape(-1),
        manifest=frame,
        bundle=bundle,
        validation={},
    )


def _central_tree_snapshot(layout: Any, leaf_capacity: int, delta: float, seed: int) -> Dict[str, Any]:
    from peer_graph_comparison import _CentroidTree, _CentroidTreeLeafNode, _CentroidTreeSplitNode

    tree = _CentroidTree(
        record_limit_per_leafnode=int(leaf_capacity),
        delta=float(delta),
        seed=int(seed),
        capacity_metric="entries",
        normalize_split_centroids=True,
        eager_recursive_splitting=True,
    )
    for peer_id in range(layout.peer_count):
        begin = int(layout.offsets[peer_id])
        end = int(layout.offsets[peer_id + 1])
        for flat in range(begin, end):
            tree.position_record(
                {
                    "CloneID": flat,
                    "OwnerID": peer_id,
                    "CentroidIndex": flat - begin,
                    "embedding": layout.embeddings[flat],
                }
            )
    result: Dict[str, Any] = {}

    def walk(node: Any, path: str) -> None:
        key = path or "root"
        if isinstance(node, _CentroidTreeLeafNode):
            result[key] = {
                "node_type": "leaf",
                "entries": sorted(
                    (int(record["OwnerID"]), int(record["CentroidIndex"]))
                    for record in node.records
                ),
                "failed_split_size": node.failed_split_size,
            }
        elif isinstance(node, _CentroidTreeSplitNode):
            result[key] = {
                "node_type": "internal",
                "left_center": np.asarray(node.centroid_left, dtype=np.float32),
                "right_center": np.asarray(node.centroid_right, dtype=np.float32),
            }
            walk(node.left_child, path + "L")
            walk(node.right_child, path + "R")
        else:
            raise TypeError(type(node))

    walk(tree.root, "")
    return result


def _compare_trees(run_dir: Path, expected: Mapping[str, Any], atol: float) -> None:
    observed = unpack_message((run_dir / "tree_snapshot.msgpack").read_bytes())
    if set(observed) != set(expected):
        raise AssertionError("Canonical tree paths differ from centralized oracle")
    for path in sorted(expected):
        actual_node = observed[path]
        expected_node = expected[path]
        if actual_node["node_type"] != expected_node["node_type"]:
            raise AssertionError(f"Tree node type differs at {path}")
        if expected_node["node_type"] == "leaf":
            actual_entries = sorted(
                (int(entry["peer_id"]), int(entry["centroid_index"]))
                for entry in actual_node["entries"]
            )
            if actual_entries != expected_node["entries"]:
                raise AssertionError(f"Leaf membership differs at {path}")
        else:
            if not np.allclose(
                actual_node["left"]["center"], expected_node["left_center"], rtol=0.0, atol=atol
            ) or not np.allclose(
                actual_node["right"]["center"], expected_node["right_center"], rtol=0.0, atol=atol
            ):
                raise AssertionError(f"Split centers differ at {path}")


def _full_oracle_validation(
    prepared_dir: Path,
    run_dir: Path,
    prepared: Mapping[str, Any],
    states: Mapping[int, Mapping[str, Any]],
    *,
    atol: float,
) -> Dict[str, Any]:
    from unified_retrieval_benchmark import (
        ALL_LEAF_ROUND0_CANDIDATE_POLICY,
        _build_round0_contacts,
        _contact_lists_to_graph,
        expand_semantica_contacts,
        fit_dataset_centroids,
        trace_semantic_chain_query,
    )

    with np.load(Path(prepared["fixture"]), allow_pickle=False) as raw_fixture:
        fixture = {name: np.asarray(raw_fixture[name]) for name in raw_fixture.files}
    dataset = _central_dataset(prepared, fixture)
    parameters = prepared["parameters"]
    layout = fit_dataset_centroids(
        dataset,
        method="elbow_silhouette",
        degree=int(parameters["degree"]),
        protected_random_fraction=0.0,
        seed=int(prepared["seed"]),
        adaptive_max_k=int(parameters["max_centroids"]),
        progress_every=0,
    )
    for peer_id, state in states.items():
        actual = state["profile"]
        begin = int(layout.offsets[peer_id])
        end = int(layout.offsets[peer_id + 1])
        if not np.allclose(actual["centroids"], layout.embeddings[begin:end], rtol=0.0, atol=atol):
            raise AssertionError(f"Peer {peer_id} centroid centers differ")
        if not np.array_equal(actual["cluster_doc_counts"], layout.cluster_doc_counts[begin:end]):
            raise AssertionError(f"Peer {peer_id} cluster counts differ")
        if not np.array_equal(actual["contact_caps"], layout.contact_caps[begin:end]):
            raise AssertionError(f"Peer {peer_id} contact quotas differ")

    protected = [np.empty(0, dtype=np.int32) for _ in states]
    round0, _diagnostics = _build_round0_contacts(
        layout,
        leaf_capacity=int(parameters["leaf_capacity"]),
        delta=float(parameters["delta"]),
        protected_random=protected,
        seed=int(prepared["seed"]),
        candidate_policy=ALL_LEAF_ROUND0_CANDIDATE_POLICY,
        tree_insertion_mode="canonical",
    )
    oracle_rounds: Dict[int, Any] = {0: round0}

    def checkpoint(round_index: int, contacts: Any) -> None:
        oracle_rounds[int(round_index)] = [
            [list(centroid) for centroid in peer] for peer in contacts
        ]

    final_contacts, _expansion = expand_semantica_contacts(
        round0,
        layout,
        protected,
        rounds=int(parameters["expansion_rounds"]),
        seed=int(prepared["seed"]),
        checkpoint_callback=checkpoint,
    )
    for round_index, expected_by_peer in sorted(oracle_rounds.items()):
        for peer_id, expected in enumerate(expected_by_peer):
            _compare_contact_snapshots(
                states[peer_id]["contacts_by_round"][str(round_index)],
                expected,
                atol=atol,
                context=f"peer {peer_id} round {round_index}",
            )
    expected_tree = _central_tree_snapshot(
        layout,
        int(parameters["leaf_capacity"]),
        float(parameters["delta"]),
        int(prepared["seed"]),
    )
    _compare_trees(run_dir, expected_tree, atol)

    final_graph = _contact_lists_to_graph(
        final_contacts,
        protected,
        degree=int(parameters["degree"]),
        graph_source="decentralized_validation_centralized_oracle",
    )
    for peer_id, state in states.items():
        observed = next(iter(state["queries"].values()))
        row = dataset.manifest.iloc[int(peer_id)]
        expected = trace_semantic_chain_query(
            dataset,
            final_graph,
            layout,
            source_peer=int(row["source_peer_position"]),
            query_embedding_index=int(row["query_embedding_index"]),
            target_doc_index=int(row["target_doc_index"]),
            manifest_row=int(row["manifest_row"]),
            max_hops=int(parameters["query_hop_limit"]),
            top_k=int(parameters["local_top_k"]),
            seed=int(prepared["seed"]),
            query_doc_index=-1,
        )
        actual_path = {int(hop): int(peer) for hop, peer in observed["peer_by_hop"].items()}
        if str(observed["query_id"]) != expected.query_id:
            raise AssertionError(f"Peer {peer_id} query ID differs")
        if str(observed["target_doc_id"]) != expected.target_doc_id:
            raise AssertionError(f"Peer {peer_id} target document ID differs")
        if actual_path != expected.peer_by_hop:
            raise AssertionError(f"Peer {peer_id} query path differs")
        if int(observed["terminal_hop"]) != expected.terminal_hop:
            raise AssertionError(f"Peer {peer_id} terminal query hop differs")
        if str(observed["terminal_reason"]) != expected.terminal_reason:
            raise AssertionError(f"Peer {peer_id} terminal query reason differs")
        for key, expected_values in (
            ("final_results", expected.final_results),
        ):
            actual_values = [(str(doc), float(score)) for doc, score in observed[key]]
            if [doc for doc, _score in actual_values] != [doc for doc, _score in expected_values]:
                raise AssertionError(f"Peer {peer_id} {key} document IDs differ")
            if not np.allclose(
                [score for _doc, score in actual_values],
                [score for _doc, score in expected_values],
                rtol=0.0,
                atol=atol,
            ):
                raise AssertionError(f"Peer {peer_id} {key} scores differ")
        if set(int(hop) for hop in observed["results_by_hop"]) != set(
            expected.results_by_hop
        ):
            raise AssertionError(f"Peer {peer_id} per-hop result keys differ")
        for hop, expected_values in expected.results_by_hop.items():
            actual_values = [
                (str(doc), float(score)) for doc, score in observed["results_by_hop"][str(hop)]
            ]
            if [doc for doc, _score in actual_values] != [doc for doc, _score in expected_values]:
                raise AssertionError(f"Peer {peer_id} hop {hop} result IDs differ")
            if not np.allclose(
                [score for _doc, score in actual_values],
                [score for _doc, score in expected_values],
                rtol=0.0,
                atol=atol,
            ):
                raise AssertionError(f"Peer {peer_id} hop {hop} scores differ")
    return {
        "oracle": "unified_retrieval_benchmark_sequential_query_trace",
        "centroid_count": int(layout.embeddings.shape[0]),
        "tree_node_count": len(expected_tree),
        "rounds_compared": len(oracle_rounds),
        "queries_compared": len(states),
        "float32_absolute_tolerance": float(atol),
    }


def _assert_exact(actual: Any, expected: Any, context: str) -> None:
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray):
            raise AssertionError(f"{context}: array type differs")
        if not np.array_equal(actual, expected):
            raise AssertionError(f"{context}: array values differ")
        return
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            raise AssertionError(f"{context}: mapping type differs")
        if set(actual) != set(expected):
            raise AssertionError(f"{context}: mapping keys differ")
        for key in sorted(actual, key=str):
            _assert_exact(actual[key], expected[key], f"{context}.{key}")
        return
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or not isinstance(expected, (list, tuple)):
            raise AssertionError(f"{context}: sequence type differs")
        if len(actual) != len(expected):
            raise AssertionError(f"{context}: sequence length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_exact(left, right, f"{context}[{index}]")
        return
    if actual != expected:
        raise AssertionError(f"{context}: {actual!r} != {expected!r}")


def _profile_algorithm_state(profile: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in profile.items()
        if key not in {"endpoint"}
    }


def _contacts_algorithm_state(
    contacts_by_round: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        str(round_index): [
            [
                {key: value for key, value in contact.items() if key != "endpoint"}
                for contact in centroid
            ]
            for centroid in by_centroid
        ]
        for round_index, by_centroid in contacts_by_round.items()
    }


def _tree_algorithm_state(run_dir: Path) -> Dict[str, Any]:
    tree = unpack_message((run_dir / "tree_snapshot.msgpack").read_bytes())
    result: Dict[str, Any] = {}
    for path, node in tree.items():
        if node["node_type"] == "leaf":
            result[path] = {
                "node_type": "leaf",
                "depth": int(node["depth"]),
                "failed_split_size": node.get("failed_split_size"),
                "entries": sorted(
                    [
                        {
                            "peer_id": int(entry["peer_id"]),
                            "centroid_index": int(entry["centroid_index"]),
                            "profile_version": int(entry["profile_version"]),
                            "vector": entry["vector"],
                        }
                        for entry in node["entries"]
                    ],
                    key=lambda entry: (entry["peer_id"], entry["centroid_index"]),
                ),
            }
        else:
            result[path] = {
                "node_type": "internal",
                "depth": int(node["depth"]),
                "left_center": node["left"]["center"],
                "right_center": node["right"]["center"],
            }
    return result


def _reference_run_validation(run_dir: Path, reference_run: Path) -> Dict[str, Any]:
    observed = load_peer_states(run_dir)
    expected = load_peer_states(reference_run)
    if not set(expected).issubset(observed):
        raise AssertionError("Observed run is missing peers from the reference run")
    for peer_id in sorted(expected):
        _assert_exact(
            _profile_algorithm_state(observed[peer_id]["profile"]),
            _profile_algorithm_state(expected[peer_id]["profile"]),
            f"peer {peer_id} profile",
        )
        _assert_exact(
            _contacts_algorithm_state(observed[peer_id]["contacts_by_round"]),
            _contacts_algorithm_state(expected[peer_id]["contacts_by_round"]),
            f"peer {peer_id} contacts",
        )
        _assert_exact(
            observed[peer_id]["queries"],
            expected[peer_id]["queries"],
            f"peer {peer_id} queries",
        )
    tree_compared = set(observed) == set(expected)
    if tree_compared:
        _assert_exact(
            _tree_algorithm_state(run_dir),
            _tree_algorithm_state(reference_run),
            "canonical tree",
        )
    return {
        "status": "exact_match",
        "reference_run": str(reference_run),
        "peers_compared": len(expected),
        "late_peers_excluded": len(observed) - len(expected),
        "canonical_tree_compared": tree_compared,
        "physical_custody_compared": False,
        "network_totals_compared": False,
        "float_comparison": "bitwise",
    }


def _contact_stability_validation(
    run_dir: Path, prejoin_run: Path
) -> Dict[str, Any]:
    observed = load_peer_states(run_dir)
    checkpoint = load_peer_states(prejoin_run)
    if not checkpoint or not set(checkpoint).issubset(observed):
        raise AssertionError("Pre-join checkpoint has an invalid peer set")
    for peer_id in sorted(checkpoint):
        before = pack_message(checkpoint[peer_id]["contacts_by_round"])
        after = pack_message(observed[peer_id]["contacts_by_round"])
        if before != after:
            raise AssertionError(
                f"Peer {peer_id} contacts changed after asynchronous admission"
            )
    return {
        "status": "byte_identical",
        "peers_compared": len(checkpoint),
        "prejoin_run": str(prejoin_run),
    }


def validate_run(
    prepared_dir: Path,
    run_dir: Path,
    *,
    structural_only: bool = False,
    float32_atol: float = 1e-5,
    reference_run: Path | None = None,
    prejoin_run: Path | None = None,
) -> Dict[str, Any]:
    prepared_dir = Path(prepared_dir).resolve()
    run_dir = Path(run_dir).resolve()
    prepared = _read_json(prepared_dir / "prepared_manifest.json")
    bootstrap_wire = _read_json(run_dir / "bootstrap.json")
    bootstrap = BootstrapConfig.from_wire(bootstrap_wire)
    registry_wire = _read_json(run_dir / "discovered_peer_registry.json")
    registry_peers = {int(peer_id) for peer_id in registry_wire["peers"]}
    states = load_peer_states(run_dir)
    if set(states) != registry_peers:
        raise AssertionError("Exported peer state set differs from discovered registry")
    report: Dict[str, Any] = {
        "schema": "decentralized_semantica_validation_v2",
        "status": "running",
        "mode": "structural_only" if structural_only else "full_centralized_oracle",
    }
    report["provenance"] = _validate_run_provenance(
        prepared_dir, run_dir, prepared, bootstrap_wire, bootstrap
    )
    report["structural"] = _structural_validation(
        prepared_dir, run_dir, bootstrap, registry_wire, states
    )
    if not structural_only:
        report["oracle"] = _full_oracle_validation(
            prepared_dir,
            run_dir,
            prepared,
            states,
            atol=float(float32_atol),
        )
    if reference_run is not None:
        report["reference"] = _reference_run_validation(
            run_dir, Path(reference_run).resolve()
        )
    if prejoin_run is not None:
        report["contact_stability"] = _contact_stability_validation(
            run_dir, Path(prejoin_run).resolve()
        )
    report["status"] = "passed"
    (run_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report

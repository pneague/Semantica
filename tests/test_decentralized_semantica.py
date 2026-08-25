from __future__ import annotations

from pathlib import Path

import msgpack
import numpy as np
import pytest

from decentralized_semantica.algorithms import (
    assign_unique_candidates,
    custody_proof_for_entries,
    fit_peer_profile,
    genesis_custody_proof,
    root_node_id,
    split_entries,
    verify_custody_proof,
)
from decentralized_semantica.centralized_simulation import (
    CentralCentroidTree,
    run_centralized_simulation,
)
from decentralized_semantica.dataset import load_shard, prepare_demo_shards
from decentralized_semantica.models import (
    BootstrapConfig,
    CentroidEntry,
    ChildDescriptor,
    Endpoint,
    PeerDescriptor,
)
from decentralized_semantica.protocol import ProtocolError, pack_message, unpack_message
from decentralized_semantica.simulation import run_startup_simulation


def test_messagepack_round_trip_and_rejects_malformed_arrays() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    decoded = unpack_message(pack_message({"values": values}))["values"]
    assert decoded.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(decoded, values.astype(np.float32))

    malformed = {
        "__ndarray__": True,
        "dtype": "<f4",
        "shape": [2, 3],
        "data": b"short",
    }
    with pytest.raises(ProtocolError, match="length"):
        unpack_message(msgpack.packb({"values": malformed}, use_bin_type=True))


def test_profiles_splits_and_contact_assignment_are_valid() -> None:
    endpoint = Endpoint("127.0.0.1", 9000)
    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]],
        dtype=np.float32,
    )
    profile = fit_peer_profile(
        0, endpoint, embeddings, degree=2, seed=42, max_k=2
    )
    assert profile.peer_id == 0
    assert int(np.sum(profile.contact_caps)) == 2
    assert 1 <= len(profile.centroids) <= 2

    entries = [
        CentroidEntry(
            peer_id=index,
            centroid_index=0,
            vector=vector,
            profile_version=1,
            endpoint=Endpoint("127.0.0.1", 9000 + index),
        )
        for index, vector in enumerate(embeddings)
    ]
    split = split_entries(entries, seed=42)
    assert split is not None
    _left_center, _right_center, left, right = split
    assert left and right

    assigned, underfilled = assign_unique_candidates(
        [{1: 0.9, 2: 0.8}, {1: 0.95, 3: 0.7}], [1, 1]
    )
    assert underfilled == []
    assert len({peer_id for group in assigned for peer_id, _score in group}) == 2


def test_custody_is_deterministic_and_independently_verifiable() -> None:
    entries = [
        CentroidEntry(
            peer_id=peer_id,
            centroid_index=0,
            vector=np.asarray([float(peer_id), 1.0], dtype=np.float32),
            profile_version=1,
            endpoint=Endpoint("127.0.0.1", 9100 + peer_id),
        )
        for peer_id in (4, 2, 7)
    ]
    proof = custody_proof_for_entries("demo", "node", 3, entries)
    assert proof.candidate_peer_ids == (2, 4, 7)
    assert verify_custody_proof("demo", "node", 3, entries, proof)
    assert not verify_custody_proof("demo", "other-node", 3, entries, proof)


def test_bootstrap_contains_only_the_genesis_descriptor() -> None:
    descriptor = PeerDescriptor(0, "127.0.0.1", 9000)
    node_id = root_node_id("demo")
    proof = genesis_custody_proof("demo", node_id, descriptor)
    bootstrap = BootstrapConfig(
        "demo",
        42,
        ChildDescriptor(node_id, descriptor, 0, proof),
        {"degree": 2},
    )
    wire = bootstrap.to_wire()
    assert "peers" not in wire and "peer_ids" not in wire
    assert BootstrapConfig.from_wire(wire) == bootstrap
    with pytest.raises(ValueError, match="peer list"):
        BootstrapConfig.from_wire({**wire, "peer_ids": [0]})


def test_demo_preparation_creates_one_private_shard_per_peer(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    manifest = prepare_demo_shards(prepared, peer_count=4, seed=7)
    shards = sorted((prepared / "shards").glob("peer_*.npz"))
    assert manifest["dataset"] == "synthetic-demo"
    assert len(shards) == 4
    for peer_id, path in enumerate(shards):
        shard = load_shard(path)
        assert int(shard["peer_id"].item()) == peer_id
        assert shard["training_embeddings"].shape == (4, 3)


def test_local_simulation_starts_every_peer(tmp_path: Path) -> None:
    summary = run_startup_simulation(
        tmp_path / "simulation",
        peer_count=4,
        startup_timeout_seconds=30,
        hold_seconds=0.5,
    )
    assert summary == {
        "status": "started",
        "peer_count": 4,
        "listening_peer_ids": [0, 1, 2, 3],
    }


def test_local_simulation_can_reuse_work_directory(tmp_path: Path) -> None:
    work_dir = tmp_path / "reusable-simulation"
    first = run_startup_simulation(
        work_dir, peer_count=2, startup_timeout_seconds=30, hold_seconds=0.1
    )
    second = run_startup_simulation(
        work_dir, peer_count=2, startup_timeout_seconds=30, hold_seconds=0.1
    )
    assert first == second
    assert second["listening_peer_ids"] == [0, 1]


def test_centralized_simulation_runs_without_result_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locate_calls = 0
    original_locate = CentralCentroidTree.locate_leaf

    def traced_locate(
        tree: CentralCentroidTree, vector: np.ndarray
    ) -> object:
        nonlocal locate_calls
        locate_calls += 1
        return original_locate(tree, vector)

    monkeypatch.setattr(CentralCentroidTree, "locate_leaf", traced_locate)
    work_dir = tmp_path / "centralized"
    summary = run_centralized_simulation(work_dir, peer_count=4, seed=42)
    assert summary["status"] == "complete"
    assert summary["mode"] == "centralized"
    assert summary["peer_count"] == 4
    assert summary["tree_entry_count"] == summary["centroid_count"]
    assert summary["directed_contact_count"] == 8
    assert summary["query_count"] == 4
    assert locate_calls == summary["centroid_count"]
    assert not (work_dir / "results").exists()


def test_centralized_simulation_accepts_compact_aol_bundle(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    bundle = tmp_path / "aol_bundle.npz"
    np.savez_compressed(
        bundle,
        document_ids=np.asarray(
            [[f"aol-{peer}-{doc}" for doc in range(4)] for peer in range(3)],
            dtype="S16",
        ),
        document_embeddings=rng.normal(size=(3, 4, 6)).astype(np.float32),
    )
    work_dir = tmp_path / "aol-centralized"
    summary = run_centralized_simulation(
        work_dir,
        peer_count=3,
        seed=7,
        aol_bundle=bundle,
    )
    assert summary["status"] == "complete"
    assert summary["peer_count"] == 3
    assert summary["tree_entry_count"] == summary["centroid_count"]
    assert summary["query_count"] == 3
    first_shard = load_shard(work_dir / "prepared" / "shards" / "peer_00000.npz")
    assert first_shard["training_doc_ids"].tolist()[0] == "aol-0-0"
    assert first_shard["target_doc_id"].item() == "aol-1-0"

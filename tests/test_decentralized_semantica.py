from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing as mp
import queue
import shutil
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from decentralized_semantica.algorithms import (
    allocate_centroid_contact_caps,
    assign_unique_candidates,
    child_node_id,
    custody_digest,
    custody_proof_for_entries,
    fit_peer_profile,
    genesis_custody_proof,
    local_topk,
    merge_topk,
    normalize_rows,
    root_node_id,
    split_entries,
    verify_custody_proof,
)
from decentralized_semantica.launcher import load_peer_states, run_experiment
from decentralized_semantica.models import (
    BootstrapConfig,
    ChildDescriptor,
    CentroidEntry,
    Contact,
    CustodyProof,
    Endpoint,
    InternalNode,
    LeafNode,
    PeerDescriptor,
    PeerProfile,
    QueryState,
)
from decentralized_semantica.peer import PeerRuntime, RESPONSE_TYPES, run_peer_process
from decentralized_semantica.protocol import (
    MAX_FRAME_BYTES,
    ProtocolError,
    RpcClient,
    envelope,
    pack_message,
    read_frame,
    unpack_message,
    write_frame,
)
from decentralized_semantica.validation import validate_run


def _write_shard(path: Path, peer_id: int, vector: np.ndarray | None = None) -> None:
    base = np.asarray(vector if vector is not None else [1.0, 0.0], dtype=np.float32)
    values = np.vstack([base, base * 0.9, -base, -base * 0.9]).astype(np.float32)
    np.savez_compressed(
        path,
        peer_id=np.asarray(peer_id, dtype=np.int32),
        training_doc_indices=np.arange(peer_id * 4, peer_id * 4 + 4, dtype=np.int32),
        training_doc_ids=np.asarray([f"doc-{peer_id}-{index}" for index in range(4)]),
        training_embeddings=values,
        query_id=np.asarray(f"query-{peer_id}"),
        query_embedding=base,
        target_doc_id=np.asarray("target"),
    )


def _entry(peer_id: int, vector: list[float], *, port: int | None = None) -> CentroidEntry:
    return CentroidEntry(
        peer_id,
        0,
        np.asarray(vector, dtype=np.float32),
        1,
        Endpoint("127.0.0.1", port if port is not None else 10_000 + peer_id),
    )


def test_messagepack_framing_arrays_limits_and_correlation() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    decoded = unpack_message(pack_message({"values": values}))["values"]
    assert decoded.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(decoded, values.astype(np.float32))

    import msgpack

    malformed = {"__ndarray__": True, "dtype": "<f4", "shape": [2, 3], "data": b"short"}
    with pytest.raises(ProtocolError, match="length"):
        unpack_message(msgpack.packb({"values": malformed}, use_bin_type=True))
    with pytest.raises(ProtocolError, match="larger"):
        pack_message({"payload": b"x" * MAX_FRAME_BYTES})

    async def framing() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack(">I", MAX_FRAME_BYTES + 1))
        reader.feed_eof()
        with pytest.raises(ProtocolError, match="Invalid frame length"):
            await read_frame(reader)

        async def handler(incoming: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            request, _ = await read_frame(incoming)
            await write_frame(
                writer,
                envelope(
                    "PONG",
                    epoch=request["epoch"],
                    source_peer_id=1,
                    source_endpoint=Endpoint("127.0.0.1", 1),
                    request_id="wrong",
                ),
            )
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        endpoint = Endpoint("127.0.0.1", int(server.sockets[0].getsockname()[1]))
        async with server:
            with pytest.raises(ProtocolError, match="request_id"):
                await RpcClient(0, Endpoint("127.0.0.1", 2), "epoch", 2).call(endpoint, "PING")

    asyncio.run(framing())


def test_peer_rejects_wrong_epoch_and_unknown_message(tmp_path: Path) -> None:
    shard = tmp_path / "peer.npz"
    _write_shard(shard, 0)

    async def scenario() -> None:
        runtime = PeerRuntime(0, shard, tmp_path / "run", None)
        runtime.epoch = "expected"
        server = await asyncio.start_server(runtime._handle_client, "127.0.0.1", 0)
        runtime.endpoint = Endpoint("127.0.0.1", int(server.sockets[0].getsockname()[1]))
        async with server:
            with pytest.raises(ProtocolError, match="epoch"):
                await RpcClient(9, Endpoint("127.0.0.1", 9999), "wrong", 2).call(
                    runtime.endpoint, "PROFILE_GET"
                )
            with pytest.raises(ProtocolError, match="Unknown message type"):
                await RpcClient(9, Endpoint("127.0.0.1", 9999), "expected", 2).call(
                    runtime.endpoint, "NOT_A_MESSAGE"
                )
        runtime.log_handle.close()

    asyncio.run(scenario())


def test_seeded_custody_is_order_independent_salted_and_owner_only() -> None:
    epoch = "custody-epoch"
    parent = root_node_id(epoch)
    left = child_node_id(epoch, parent, "left")
    right = child_node_id(epoch, parent, "right")
    entries = [_entry(9, [1, 0]), _entry(2, [0.9, 0.1]), _entry(5, [0.8, 0.2])]
    proof = custody_proof_for_entries(epoch, left, 1, entries)
    reordered = custody_proof_for_entries(epoch, left, 1, list(reversed(entries)))
    assert proof == reordered
    assert proof.candidate_peer_ids == (2, 5, 9)
    assert proof.custodian.peer_id in proof.candidate_peer_ids
    assert proof.digest == custody_digest(epoch, left, 1, (9, 2, 5))
    candidate_bytes = b"".join(
        struct.pack(">I", 8) + struct.pack(">Q", peer_id)
        for peer_id in (2, 5, 9)
    )
    expected_digest = hashlib.sha256(
        b"semantica-custody-v1"
        + epoch.encode("utf-8")
        + struct.pack(">Q", 1)
        + left.encode("ascii")
        + candidate_bytes
    ).hexdigest()
    assert proof.digest == expected_digest
    assert custody_proof_for_entries(epoch, right, 1, entries).digest != proof.digest
    assert custody_proof_for_entries(epoch, left, 2, entries).digest != proof.digest
    assert custody_proof_for_entries(epoch, left, 1, [_entry(7, [1, 0])]).custodian.peer_id == 7
    assert verify_custody_proof(epoch, left, 1, entries, proof)
    with pytest.raises(ValueError, match="conflicting"):
        custody_proof_for_entries(epoch, left, 1, [entries[0], _entry(9, [0, 1], port=9999)])


def test_genesis_and_bootstrap_v3_have_no_membership(tmp_path: Path) -> None:
    peer = PeerDescriptor(3, "127.0.0.1", 1234, 1)
    node_id = root_node_id("epoch")
    proof = genesis_custody_proof("epoch", node_id, peer)
    bootstrap = BootstrapConfig("epoch", 42, ChildDescriptor(node_id, peer, 0, proof), {"degree": 20})
    wire = bootstrap.to_wire()
    assert wire["schema"] == "decentralized_semantica_bootstrap_v3"
    assert wire["implementation"] == "port_based_fully_decentralized_semantica_v3"
    assert "peers" not in wire and "peer_ids" not in wire
    assert BootstrapConfig.from_wire(wire) == bootstrap
    invalid = BootstrapConfig(
        "epoch",
        42,
        ChildDescriptor("not-the-root", peer, 0, proof),
        {"degree": 20},
    )
    shard = tmp_path / "peer.npz"
    _write_shard(shard, 3)
    with pytest.raises(ValueError, match="invalid genesis"):
        PeerRuntime(3, shard, tmp_path / "run", None, bootstrap=invalid)


def test_create_leaf_recomputes_every_custody_input(tmp_path: Path) -> None:
    shard = tmp_path / "peer.npz"
    epoch = "create-leaf-epoch"
    parent_peer = PeerDescriptor(0, "127.0.0.1", 10000, 1)
    parent_id = root_node_id(epoch)
    parent_proof = genesis_custody_proof(epoch, parent_id, parent_peer)
    parent = ChildDescriptor(parent_id, parent_peer, 0, parent_proof)
    child_id = child_node_id(epoch, parent_id, "left")
    entries = [_entry(1, [1, 0], port=10001), _entry(2, [0.8, 0.2], port=10002)]
    proof = custody_proof_for_entries(epoch, child_id, 1, entries)
    _write_shard(shard, proof.custodian.peer_id)
    runtime = PeerRuntime(proof.custodian.peer_id, shard, tmp_path / "run", None)
    runtime.epoch = epoch
    runtime.endpoint = proof.custodian.endpoint
    runtime.parameters = {"leaf_capacity": 20, "seed": 42, "delta": 0.0}
    payload = {
        "node_id": child_id,
        "side": "left",
        "depth": 1,
        "parent": parent.to_wire(),
        "parent_entry_count": len(entries),
        "custody_seed": 1,
        "custody_proof": proof.to_wire(),
        "entries": [entry.to_wire() for entry in entries],
    }
    request = {"source_peer_id": 0, "source_endpoint": parent_peer.endpoint.to_wire()}

    async def scenario() -> None:
        accepted = await runtime._on_create_leaf(payload, request)
        assert accepted["node_id"] == child_id
        duplicate = await runtime._on_create_leaf(payload, request)
        assert duplicate["already_exists"] is True
        bad = dict(payload)
        bad["custody_seed"] = 2
        with pytest.raises(ProtocolError, match="seed"):
            await runtime._on_create_leaf(bad, request)
        bad = dict(payload)
        bad_proof = CustodyProof(1, proof.candidate_peer_ids, "0" * 64, proof.custodian)
        bad["custody_proof"] = bad_proof.to_wire()
        with pytest.raises(ProtocolError, match="proof"):
            await runtime._on_create_leaf(bad, request)

    asyncio.run(scenario())
    runtime.log_handle.close()


def test_descriptor_conflicts_are_rejected_and_root_has_no_global_power(tmp_path: Path) -> None:
    shard = tmp_path / "peer.npz"
    _write_shard(shard, 0)
    runtime = PeerRuntime(0, shard, tmp_path / "run", None)
    runtime._remember_descriptor(PeerDescriptor(4, "127.0.0.1", 4000, 1))
    with pytest.raises(ProtocolError, match="conflicting"):
        runtime._remember_descriptor(PeerDescriptor(4, "127.0.0.1", 4001, 1))
    for name in (
        "registry",
        "join_order",
        "active_peers",
        "custodian_ready",
        "custodian_assignments",
        "admission_task",
        "sync_phase_waiters",
    ):
        assert not hasattr(runtime, name)
    for message_type in (
        "JOIN",
        "PEER_LOOKUP",
        "CUSTODIAN_ASSIGN",
        "REGISTRY_STATUS",
        "PROFILE_READY",
        "ACTIVATE_SYNC",
        "FINALIZE",
        "SHUTDOWN",
        "BUILD_TURN",
        "ROUND0_START",
        "EXPANSION_START",
        "QUERY_START",
        "EXPORT_STATE",
    ):
        assert message_type not in RESPONSE_TYPES
    runtime.log_handle.close()


def test_profile_quota_split_assignment_and_query_kernels_match() -> None:
    values = np.asarray([[1, 0], [0.9, 0.1], [-1, 0], [-0.9, -0.1]], dtype=np.float32)
    profile = fit_peer_profile(3, Endpoint("127.0.0.1", 1234), values, degree=4, seed=42, max_k=2)
    assert profile.cluster_doc_counts.tolist() == [2, 2]
    assert profile.contact_caps.tolist() == [2, 2]
    assert allocate_centroid_contact_caps([5, 3, 2], 7) == [3, 2, 2]
    split = split_entries([_entry(index, vector.tolist()) for index, vector in enumerate(values)], seed=42)
    assert split is not None
    assert sorted(entry.peer_id for entry in split[2] + split[3]) == [0, 1, 2, 3]
    assert split_entries([_entry(0, [1, 0]), _entry(1, [1, 0])], seed=42) is None
    assigned, underfilled = assign_unique_candidates([{1: 0.9, 2: 0.8}, {1: 0.95, 3: 0.7}], [1, 1])
    assert assigned == [[(2, 0.8)], [(1, 0.95)]] and underfilled == []
    local = local_topk(np.asarray([1, 0]), ["b", "a", "c"], np.asarray([[1, 0], [1, 0], [0, 1]]), 3)
    assert [doc for doc, _ in local] == ["a", "b", "c"]
    rng = np.random.default_rng(42)
    query = normalize_rows(rng.normal(size=(1, 64)).astype(np.float32))[0]
    embeddings = rng.normal(size=(133, 64)).astype(np.float32)
    keys = rng.permutation(np.arange(1000, 1133, dtype=np.int64))
    positions = np.argsort(keys)
    expected = local_topk(
        query,
        [f"doc-{value}" for value in keys[positions]],
        embeddings[positions],
        100,
        query_is_normalized=True,
    )
    observed = local_topk(
        query,
        [f"doc-{value}" for value in keys],
        embeddings,
        100,
        query_is_normalized=True,
        doc_order_keys=keys,
    )
    assert [doc for doc, _score in observed] == [doc for doc, _score in expected]
    assert np.asarray([score for _doc, score in observed], dtype="<f4").tobytes() == np.asarray(
        [score for _doc, score in expected], dtype="<f4"
    ).tobytes()
    assert merge_topk([("a", 0.9), ("b", 0.8)], [("b", 0.9), ("c", 0.7)], 3) == [("a", 0.9), ("b", 0.9), ("c", 0.7)]


def test_expansion_returns_descriptors_and_requester_fetches_profiles(tmp_path: Path) -> None:
    requester_shard = tmp_path / "peer-0.npz"
    referrer_shard = tmp_path / "peer-1.npz"
    _write_shard(requester_shard, 0)
    _write_shard(referrer_shard, 1)
    endpoint0 = Endpoint("127.0.0.1", 10000)
    endpoint1 = Endpoint("127.0.0.1", 10001)
    endpoint2 = Endpoint("127.0.0.1", 10002)

    async def scenario() -> None:
        referrer = PeerRuntime(1, referrer_shard, tmp_path / "referrer", None)
        referrer.contacts_by_round[0] = [[Contact(0, 0.8, endpoint0, 1, "round0"), Contact(2, 0.7, endpoint2, 1, "round0")]]
        reply = await referrer._on_expansion_request({"round": 0}, {"source_peer_id": 0})
        assert reply == {"round": 0, "available": True, "peers": [PeerDescriptor(2, endpoint2.host, endpoint2.port, 1).to_wire()]}

        requester = PeerRuntime(0, requester_shard, tmp_path / "requester", None)
        requester.parameters = {"seed": 42}
        requester.profile = PeerProfile(0, endpoint0, np.asarray([[1, 0]], dtype=np.float32), np.asarray([1], dtype=np.int32), np.asarray([1], dtype=np.int32))
        requester.contacts_by_round[0] = [[Contact(1, 0.1, endpoint1, 1, "round0")]]
        requester.initial_contact_sets = [{1}]
        requester.asked_referrers = [set()]
        observed: dict[str, Any] = {}

        async def fake_rpc(peer_id: int, message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
            observed.update(peer_id=peer_id, message_type=message_type, payload=payload)
            return {"message_type": "EXPANSION_REPLY", "payload": reply}

        fetched: list[int] = []

        async def fake_profile(value: int | PeerDescriptor) -> PeerProfile:
            peer_id = value.peer_id if isinstance(value, PeerDescriptor) else int(value)
            fetched.append(peer_id)
            endpoint = {0: endpoint0, 1: endpoint1, 2: endpoint2}[peer_id]
            vector = [0, 1] if peer_id == 1 else [1, 0]
            return PeerProfile(peer_id, endpoint, np.asarray([vector], dtype=np.float32), np.asarray([1], dtype=np.int32), np.asarray([1], dtype=np.int32))

        requester.rpc = fake_rpc  # type: ignore[method-assign]
        requester._profile_for = fake_profile  # type: ignore[method-assign]
        await requester._expand_round(1)
        assert observed == {"peer_id": 1, "message_type": "EXPANSION_REQUEST", "payload": {"round": 0}}
        assert 2 in fetched
        assert [contact.peer_id for contact in requester.contacts_by_round[1][0]] == [2]
        referrer.log_handle.close()
        requester.log_handle.close()

    asyncio.run(scenario())


def test_pending_contacts_use_deterministic_capped_backoff(tmp_path: Path) -> None:
    shard = tmp_path / "peer.npz"
    _write_shard(shard, 7)
    runtime = PeerRuntime(7, shard, tmp_path / "run", None)
    runtime.epoch = "retry"
    delays = [runtime._contact_retry_delay(index) for index in range(8)]
    assert [int(value) for value in delays] == [1, 2, 4, 8, 16, 30, 30, 30]
    assert delays == [runtime._contact_retry_delay(index) for index in range(8)]
    assert all(0 <= value - int(value) <= 0.250 for value in delays)
    runtime.log_handle.close()


def test_query_results_buffer_out_of_order_and_merge_by_hop(tmp_path: Path) -> None:
    shard = tmp_path / "peer.npz"
    _write_shard(shard, 0)
    runtime = PeerRuntime(0, shard, tmp_path / "run", None)
    runtime.query_states["q"] = QueryState(
        "q", np.asarray([1, 0], dtype=np.float32), "target"
    )

    async def scenario() -> None:
        await runtime._on_query_result(
            {
                "query_id": "q",
                "hop": 2,
                "peer": PeerDescriptor(2, "127.0.0.1", 10002, 1).to_wire(),
                "results": [["b", 0.9]],
            },
            {},
        )
        await runtime._on_query_result(
            {
                "query_id": "q",
                "hop": 1,
                "peer": PeerDescriptor(1, "127.0.0.1", 10001, 1).to_wire(),
                "results": [["a", 0.9], ["b", 0.8]],
            },
            {},
        )

    asyncio.run(scenario())
    state = runtime.query_states["q"]
    for hop in sorted(state.results_by_hop):
        state.final_results = merge_topk(state.final_results, state.results_by_hop[hop], 2)
    assert state.peer_by_hop == {2: 2, 1: 1}
    assert state.final_results == [("a", 0.9), ("b", 0.9)]
    runtime.log_handle.close()


def _write_small_prepared_fixture(path: Path, peer_count: int = 4) -> dict[str, Any]:
    (path / "shards").mkdir(parents=True)
    bases = np.asarray([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0.7, 0.7, 0]], dtype=np.float32)
    all_doc_ids: list[str] = []
    all_doc_embeddings: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    digest = hashlib.sha256()
    inventories = []
    for peer_id in range(peer_count):
        base = bases[peer_id]
        vectors = np.vstack([base, base + [0.03, 0.02, 0.01], -base, -base + [0.01, -0.02, 0.03]]).astype(np.float32)
        indices = np.arange(peer_id * 4, peer_id * 4 + 4, dtype=np.int32)
        inventories.append(indices)
        target = (peer_id + 1) % peer_count
        all_doc_ids.extend(f"doc-{peer_id}-{index}" for index in range(4))
        all_doc_embeddings.extend(vectors)
        queries.append(bases[target])
        shard = path / "shards" / f"peer_{peer_id:05d}.npz"
        np.savez_compressed(
            shard,
            peer_id=np.asarray(peer_id, dtype=np.int32),
            training_doc_indices=indices,
            training_doc_ids=np.asarray([f"doc-{peer_id}-{index}" for index in range(4)]),
            training_embeddings=vectors,
            query_index=np.asarray(peer_id, dtype=np.int32),
            query_id=np.asarray(f"query-{peer_id}"),
            query_embedding=bases[target],
            target_doc_index=np.asarray(target * 4, dtype=np.int32),
            target_doc_id=np.asarray(f"doc-{target}-0"),
            target_available=np.asarray(True),
        )
        digest.update(shard.read_bytes())
    bundle = path / "embedding_bundle"
    bundle.mkdir()
    paths = {
        "doc_ids_path": bundle / "doc_ids.npy",
        "doc_embeddings_path": bundle / "doc_embeddings.npy",
        "query_keys_path": bundle / "query_keys.npy",
        "query_embeddings_path": bundle / "query_embeddings.npy",
    }
    np.save(paths["doc_ids_path"], np.asarray(all_doc_ids), allow_pickle=False)
    np.save(paths["doc_embeddings_path"], np.asarray(all_doc_embeddings, dtype=np.float32), allow_pickle=False)
    np.save(paths["query_keys_path"], np.asarray([f"query-{peer}" for peer in range(peer_count)]), allow_pickle=False)
    np.save(paths["query_embeddings_path"], np.asarray(queries, dtype=np.float32), allow_pickle=False)
    bundle_manifest = bundle / "manifest.json"
    bundle_manifest.write_text("{}", encoding="utf-8")
    fixture = path / "central_fixture.npz"
    np.savez_compressed(
        fixture,
        peer_ids=np.arange(peer_count, dtype=np.int32),
        training_doc_indices=np.asarray(inventories, dtype=np.int32),
        query_indices=np.arange(peer_count, dtype=np.int32),
        target_doc_indices=np.asarray([((peer + 1) % peer_count) * 4 for peer in range(peer_count)], dtype=np.int32),
        target_available=np.ones(peer_count, dtype=np.bool_),
    )
    parameters = {
        "peer_count": peer_count,
        "seed": 42,
        "degree": 2,
        "leaf_capacity": 2,
        "delta": 0.0,
        "max_centroids": 2,
        "expansion_rounds": 2,
        "query_hop_limit": 4,
        "local_top_k": 3,
        "centroid_method": "cosine_silhouette",
        "protected_random_fraction": 0.0,
    }
    manifest = {
        "schema": "decentralized_semantica_prepared_v2",
        "dataset": "unit-fixture",
        "epoch": "fully-decentralized-socket-fixture",
        "seed": 42,
        "parameters": parameters,
        "shards_sha256": digest.hexdigest(),
        "fixture": str(fixture.resolve()),
        "embedding_bundle": {
            "manifest_path": str(bundle_manifest.resolve()),
            **{name: str(value.resolve()) for name, value in paths.items()},
            "model_name": "sentence-transformers/all-mpnet-base-v2",
        },
    }
    (path / "prepared_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def test_external_compatibility_run_exactly_matches_centralized_oracle(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_small_prepared_fixture(prepared_dir)
    run_dir = tmp_path / "run"
    result = run_experiment(prepared_dir, run_dir, endpoint_timeout_seconds=30, run_timeout_seconds=120, rpc_timeout_seconds=15, start_method="spawn")
    assert result["status"] == "complete"
    bootstrap = json.loads((run_dir / "bootstrap.json").read_text())
    assert bootstrap["implementation"] == "port_based_fully_decentralized_semantica_v3"
    assert "peers" not in bootstrap and "peer_ids" not in bootstrap
    assert not (run_dir / "discovered_peer_registry.json").exists()
    assert not (run_dir / "custodian_assignments.json").exists()
    inventory = json.loads((run_dir / "experiment_peer_inventory.json").read_text())
    assert inventory["runtime_input"] is False
    assert inventory["peer_order"] == [0, 1, 2, 3]
    states = load_peer_states(run_dir)
    assert set(states) == {0, 1, 2, 3}
    for state in states.values():
        assert set(state["contacts_by_round"]) == {"0", "1", "2"}
        assert len(state["queries"]) == 1
    log_rows = [json.loads(line) for path in (run_dir / "logs").glob("peer_*.jsonl") for line in path.read_text().splitlines()]
    for phase in ("BUILD", "STATS", "ROUND0", "EXPANSION:1", "EXPANSION:2", "QUERY", "EXPORT"):
        rows = [row for row in log_rows if row["direction"] == "receive" and row["message_type"] == "SYNC_TOKEN" and row["sync_phase_key"] == phase]
        rows.sort(key=lambda row: row["time"])
        assert [int(row["peer_id"]) for row in rows] == [0, 1, 2, 3]
    forbidden = {"JOIN", "PEER_LOOKUP", "CUSTODIAN_ASSIGN", "REGISTRY_STATUS", "PROFILE_READY", "ACTIVATE_SYNC", "FINALIZE", "SHUTDOWN"}
    assert not {row["message_type"] for row in log_rows}.intersection(forbidden)
    report = validate_run(prepared_dir, run_dir, reference_run=run_dir)
    assert report["status"] == "passed"
    assert report["oracle"]["float32_absolute_tolerance"] == 0.0
    assert report["reference"]["status"] == "exact_match"
    assert (run_dir / "custody_audit.msgpack").exists()


def _active_root_target(peer_id: int, shard: str, run_dir: str, event_queue: Any, epoch: str, parameters: dict[str, Any]) -> None:
    run_peer_process(peer_id, shard, run_dir, event_queue, root_initialization={"epoch": epoch, "seed": 42, "parameters": parameters, "bootstrap_path": str(Path(run_dir) / "bootstrap.json")})


def _active_join_target(peer_id: int, shard: str, run_dir: str, event_queue: Any, bootstrap: dict[str, Any]) -> None:
    run_peer_process(peer_id, shard, run_dir, event_queue, bootstrap)


def _released_compatibility_target(prepared_dir: str, run_dir: str) -> None:
    run_experiment(
        Path(prepared_dir),
        Path(run_dir),
        endpoint_timeout_seconds=30,
        run_timeout_seconds=120,
        rpc_timeout_seconds=15,
        start_method="spawn",
        release_active=True,
    )


def _wait_events(event_queue: Any, wanted: set[tuple[str, int]], processes: list[mp.Process], timeout: float = 60) -> dict[int, Endpoint]:
    found: set[tuple[str, int]] = set()
    endpoints: dict[int, Endpoint] = {}
    deadline = time.monotonic() + timeout
    while not wanted.issubset(found):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Missing events {wanted.difference(found)}")
        try:
            kind, peer_id, host, port = event_queue.get(timeout=0.2)
        except queue.Empty:
            failures = {process.name: process.exitcode for process in processes if process.exitcode not in (None, 0)}
            if failures:
                raise RuntimeError(f"Active peer failures: {failures}")
            continue
        found.add((str(kind), int(peer_id)))
        endpoints[int(peer_id)] = Endpoint(str(host), int(port))
    return endpoints


def _wait_status(path: Path, expected: str, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if path.exists() and json.loads(path.read_text()).get("status") == expected:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}={expected}")
        time.sleep(0.1)


def test_released_compatibility_overlay_accepts_late_peer_without_coordinator(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    manifest = _write_small_prepared_fixture(prepared_dir)
    run_dir = tmp_path / "released"
    context = mp.get_context("spawn")
    runner = context.Process(
        name="released-compatibility-runner",
        target=_released_compatibility_target,
        args=(str(prepared_dir), str(run_dir)),
    )
    runner.start()
    late: mp.Process | None = None
    try:
        deadline = time.monotonic() + 60
        while not (run_dir / "overlay_released.json").exists():
            if runner.exitcode is not None:
                raise RuntimeError(f"Compatibility runner exited {runner.exitcode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Compatibility coordinator did not release the overlay")
            time.sleep(0.1)
        released = json.loads((run_dir / "overlay_released.json").read_text())
        assert released["compatibility_coordinator_closed"] is True
        bootstrap = json.loads((run_dir / "bootstrap.json").read_text())
        assert bootstrap["parameters"]["compatibility_mode"] is False

        late_shard = tmp_path / "released-peer-4.npz"
        _write_shard(late_shard, 4, np.asarray([0.7, 0.7, 0], dtype=np.float32))
        late = context.Process(
            name="released-late-peer",
            target=_active_join_target,
            args=(4, str(late_shard), str(run_dir), None, bootstrap),
        )
        late.start()
        _wait_status(run_dir / "status" / "peer_00004.json", "active")
        late.terminate()
        late.join(timeout=15)
        assert late.exitcode == 0
        (run_dir / "stop_requested").touch()
        runner.join(timeout=60)
        assert runner.exitcode == 0
        assert (run_dir / "released_state" / "peer_00004.msgpack").exists()
        checkpoint = load_peer_states(run_dir)
        released_states = {
            int(state["peer_id"]): state
            for path in (run_dir / "released_state").glob("peer_*.msgpack")
            for state in [unpack_message(path.read_bytes())]
        }
        for peer_id in range(4):
            assert pack_message(checkpoint[peer_id]["contacts_by_round"]) == pack_message(
                released_states[peer_id]["contacts_by_round"]
            )
        assert set(released_states[4]["contacts_by_round"]) == {"0", "1", "2"}
        assert str(manifest["epoch"]) == BootstrapConfig.from_wire(bootstrap).epoch
    finally:
        if late is not None and late.is_alive():
            late.terminate()
            late.join(timeout=10)
        if runner.is_alive():
            (run_dir / "stop_requested").parent.mkdir(parents=True, exist_ok=True)
            (run_dir / "stop_requested").touch()
            runner.join(timeout=20)
            if runner.is_alive():
                runner.terminate()
                runner.join(timeout=10)


def test_active_mode_concurrent_and_late_join_preserves_existing_contacts(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    manifest = _write_small_prepared_fixture(prepared_dir)
    run_dir = tmp_path / "active"
    (run_dir / "logs").mkdir(parents=True)
    parameters = dict(manifest["parameters"])
    parameters.pop("peer_count", None)
    parameters.update(rpc_timeout_seconds=10.0, compatibility_mode=False)
    context = mp.get_context("spawn")
    events = context.Queue()
    processes: list[mp.Process] = []
    root = context.Process(name="active-root", target=_active_root_target, args=(0, str(prepared_dir / "shards" / "peer_00000.npz"), str(run_dir), events, str(manifest["epoch"]), parameters))
    root.start()
    processes.append(root)
    root_events = _wait_events(events, {("LISTENING", 0), ("TREE_JOINED", 0)}, processes)
    deadline = time.monotonic() + 10
    while not (run_dir / "bootstrap.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Genesis bootstrap was not published")
        time.sleep(0.05)
    bootstrap = json.loads((run_dir / "bootstrap.json").read_text())
    assert "peers" not in bootstrap
    for peer_id in (1, 2, 3):
        process = context.Process(name=f"active-{peer_id}", target=_active_join_target, args=(peer_id, str(prepared_dir / "shards" / f"peer_{peer_id:05d}.npz"), str(run_dir), events, bootstrap))
        process.start()
        processes.append(process)
    endpoints = dict(root_events)
    endpoints.update(_wait_events(events, {(kind, peer_id) for kind in ("LISTENING", "ACTIVE") for peer_id in (1, 2, 3)} | {("ACTIVE", 0)}, processes, timeout=90))
    before = {peer_id: pack_message(state["contacts_by_round"]) for peer_id, state in load_peer_states(run_dir).items()}
    prejoin_run = tmp_path / "prejoin"
    shutil.copytree(run_dir / "state", prejoin_run / "state")

    late_shard = tmp_path / "peer_00004.npz"
    _write_shard(late_shard, 4, np.asarray([0.7, 0.7, 0], dtype=np.float32))
    late = context.Process(name="active-4", target=_active_join_target, args=(4, str(late_shard), str(run_dir), events, bootstrap))
    late.start()
    processes.append(late)
    endpoints.update(_wait_events(events, {("LISTENING", 4), ("ACTIVE", 4)}, processes, timeout=90))
    for process in processes:
        process.terminate()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    after = load_peer_states(run_dir)
    for peer_id in range(4):
        assert pack_message(after[peer_id]["contacts_by_round"]) == before[peer_id]
    assert set(after[4]["contacts_by_round"]) == {"0", "1", "2"}
    assert len(after[4]["queries"]) == 1
    assert not (run_dir / "experiment_peer_inventory.json").exists()
    forbidden = {
        "JOIN",
        "PEER_LOOKUP",
        "CUSTODIAN_ASSIGN",
        "REGISTRY_STATUS",
        "PROFILE_READY",
        "ACTIVATE_SYNC",
        "FINALIZE",
        "SHUTDOWN",
        "BUILD_TURN",
        "ROUND0_START",
        "EXPANSION_START",
        "QUERY_START",
        "EXPORT_STATE",
    }
    message_types = {json.loads(line)["message_type"] for path in (run_dir / "logs").glob("peer_*.jsonl") for line in path.read_text().splitlines()}
    assert not message_types.intersection(forbidden)
    report = validate_run(
        prepared_dir,
        run_dir,
        structural_only=True,
        prejoin_run=prejoin_run,
    )
    assert report["status"] == "passed"
    assert report["contact_stability"]["status"] == "byte_identical"
    events.close()
    events.join_thread()

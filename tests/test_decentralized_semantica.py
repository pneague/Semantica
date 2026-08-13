from __future__ import annotations

import asyncio
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import pytest

from decentralized_semantica.algorithms import (
    allocate_centroid_contact_caps,
    assign_unique_candidates,
    child_node_id,
    custodian_for,
    fit_peer_profile,
    local_topk,
    merge_topk,
    root_node_id,
    split_entries,
)
from decentralized_semantica.launcher import load_peer_states, run_experiment
from decentralized_semantica.models import (
    ChildDescriptor,
    CentroidEntry,
    Contact,
    Endpoint,
    InternalNode,
    LeafNode,
    MembershipManifest,
    PeerProfile,
    QueryState,
)
from decentralized_semantica.peer import PeerRuntime
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


def test_messagepack_float32_array_roundtrip_and_malformed_dimensions() -> None:
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
    import msgpack

    with pytest.raises(ProtocolError, match="length"):
        unpack_message(msgpack.packb({"values": malformed}, use_bin_type=True))
    with pytest.raises(ProtocolError, match="larger"):
        pack_message({"payload": b"x" * MAX_FRAME_BYTES})


def test_frame_size_limit_is_rejected_before_payload_read() -> None:
    async def scenario() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack(">I", MAX_FRAME_BYTES + 1))
        reader.feed_eof()
        with pytest.raises(ProtocolError, match="Invalid frame length"):
            await read_frame(reader)

    asyncio.run(scenario())


def test_expansion_exchanges_only_round_and_peer_ids_and_fetches_profiles_at_requester(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "peer.npz"
    np.savez_compressed(
        shard,
        training_doc_ids=np.asarray(["local"]),
        training_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        query_id=np.asarray("q"),
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        target_doc_id=np.asarray("target"),
    )

    class QueueStub:
        def put(self, _value: object) -> None:
            pass

    async def scenario() -> None:
        endpoint = Endpoint("127.0.0.1", 1)
        referrer = PeerRuntime(1, shard, tmp_path / "referrer", QueueStub())
        referrer.contacts_by_round[0] = [
            [
                Contact(0, 0.8, endpoint, 1, "round0"),
                Contact(2, 0.7, endpoint, 1, "round0"),
            ]
        ]

        async def forbidden_profile_fetch(_peer_id: int) -> PeerProfile:
            raise AssertionError("The referrer must not fetch candidate profiles")

        referrer._profile_for = forbidden_profile_fetch  # type: ignore[method-assign]
        reply = await referrer._on_expansion_request(
            {"round": 0}, {"source_peer_id": 0}
        )
        assert reply == {"round": 0, "peer_ids": [2]}

        requester = PeerRuntime(0, shard, tmp_path / "requester", QueueStub())
        requester.parameters = {"seed": 42}
        requester.profile = PeerProfile(
            0,
            endpoint,
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([1], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
        )
        requester.contacts_by_round[0] = [
            [Contact(1, 0.1, endpoint, 1, "round0")]
        ]
        requester.initial_contact_sets = [{1}]
        requester.asked_referrers = [set()]
        observed_request: dict[str, object] = {}

        async def fake_rpc(
            peer_id: int, message_type: str, payload: dict[str, object]
        ) -> dict[str, object]:
            observed_request.update(
                {"peer_id": peer_id, "message_type": message_type, "payload": payload}
            )
            return {
                "message_type": "EXPANSION_REPLY",
                "payload": {"round": 0, "peer_ids": [2]},
            }

        profile_fetches: list[int] = []

        async def fake_profile_for(peer_id: int) -> PeerProfile:
            profile_fetches.append(int(peer_id))
            vector = [0.0, 1.0] if int(peer_id) == 1 else [1.0, 0.0]
            return PeerProfile(
                int(peer_id),
                endpoint,
                np.asarray([vector], dtype=np.float32),
                np.asarray([1], dtype=np.int32),
                np.asarray([1], dtype=np.int32),
            )

        requester.rpc = fake_rpc  # type: ignore[method-assign]
        requester._profile_for = fake_profile_for  # type: ignore[method-assign]
        await requester._expand_round(1)
        assert observed_request == {
            "peer_id": 1,
            "message_type": "EXPANSION_REQUEST",
            "payload": {"round": 0},
        }
        assert 2 in profile_fetches
        assert [contact.peer_id for contact in requester.contacts_by_round[1][0]] == [2]
        referrer.log_handle.close()
        requester.log_handle.close()

    asyncio.run(scenario())


def test_rpc_rejects_wrong_request_correlation() -> None:
    async def scenario() -> None:
        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            request, _size = await read_frame(reader)
            response = envelope(
                "PONG",
                epoch=request["epoch"],
                source_peer_id=1,
                source_endpoint=Endpoint("127.0.0.1", 1),
                request_id="not-the-request-id",
            )
            await write_frame(writer, response)
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = int(server.sockets[0].getsockname()[1])
        async with server:
            client = RpcClient(0, Endpoint("127.0.0.1", 2), "epoch", timeout=2)
            with pytest.raises(ProtocolError, match="request_id"):
                await client.call(Endpoint("127.0.0.1", port), "PING")

    asyncio.run(scenario())


def test_peer_server_rejects_wrong_epoch_and_unknown_message(tmp_path: Path) -> None:
    shard = tmp_path / "peer.npz"
    np.savez_compressed(
        shard,
        training_doc_ids=np.asarray(["local"]),
        training_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        query_id=np.asarray("q"),
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        target_doc_id=np.asarray("target"),
    )

    class QueueStub:
        def put(self, _value: object) -> None:
            pass

    async def scenario() -> None:
        runtime = PeerRuntime(0, shard, tmp_path / "rejector", QueueStub())
        runtime.epoch = "expected-epoch"
        server = await asyncio.start_server(runtime._handle_client, "127.0.0.1", 0)
        runtime.endpoint = Endpoint("127.0.0.1", int(server.sockets[0].getsockname()[1]))
        async with server:
            wrong_epoch = RpcClient(
                9, Endpoint("127.0.0.1", 9999), "wrong-epoch", timeout=2
            )
            with pytest.raises(ProtocolError, match="epoch"):
                await wrong_epoch.call(runtime.endpoint, "PROFILE_GET")
            correct_epoch = RpcClient(
                9, Endpoint("127.0.0.1", 9999), runtime.epoch, timeout=2
            )
            with pytest.raises(ProtocolError, match="Unknown message type"):
                await correct_epoch.call(runtime.endpoint, "NOT_A_MESSAGE")
        runtime.log_handle.close()

    asyncio.run(scenario())


def test_ids_rendezvous_and_tie_break_are_deterministic() -> None:
    manifest = MembershipManifest(
        epoch="epoch-42",
        seed=42,
        peers={peer: Endpoint("127.0.0.1", 10_000 + peer) for peer in range(8)},
        parameters={},
    )
    root = root_node_id(manifest.epoch)
    assert root == hashlib.sha256(b"epoch-42root").hexdigest()
    assert child_node_id(manifest.epoch, root, "left") == hashlib.sha256(
        b"epoch-42" + root.encode("ascii") + b"left"
    ).hexdigest()
    assert child_node_id(manifest.epoch, root, "left") != child_node_id(
        manifest.epoch, root, "right"
    )
    assert custodian_for(root, manifest) == custodian_for(root, manifest)
    with pytest.raises(ValueError, match="side"):
        child_node_id(manifest.epoch, root, "middle")


def test_profile_fitting_and_proportional_quotas_are_deterministic() -> None:
    values = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]], dtype=np.float32
    )
    profile = fit_peer_profile(
        3, Endpoint("127.0.0.1", 1234), values, degree=4, seed=42, max_k=2
    )
    assert profile.centroids.shape == (2, 2)
    assert profile.cluster_doc_counts.tolist() == [2, 2]
    assert profile.contact_caps.tolist() == [2, 2]
    assert allocate_centroid_contact_caps([5, 3, 2], 7) == [3, 2, 2]
    np.testing.assert_allclose(np.linalg.norm(profile.centroids, axis=1), 1.0, atol=1e-6)


def _entry(peer_id: int, vector: list[float]) -> CentroidEntry:
    return CentroidEntry(peer_id, 0, np.asarray(vector, dtype=np.float32), 1, Endpoint("h", peer_id + 1))


def test_split_entries_and_unsplittable_overflow() -> None:
    entries = [
        _entry(0, [1.0, 0.0]),
        _entry(1, [0.9, 0.1]),
        _entry(2, [-1.0, 0.0]),
        _entry(3, [-0.9, -0.1]),
    ]
    split = split_entries(entries, seed=42, delta=0.0)
    assert split is not None
    _left_center, _right_center, left, right = split
    assert sorted(entry.peer_id for entry in left + right) == [0, 1, 2, 3]
    assert left and right
    assert split_entries([_entry(0, [1, 0]), _entry(1, [1, 0])], seed=42) is None


def test_split_centers_match_centralized_float64_arithmetic() -> None:
    from peer_graph_comparison import _CentroidTree, _CentroidTreeSplitNode

    vectors = np.asarray(
        [
            [0.9912341, 0.1321027, 0.0012345],
            [0.9523811, 0.3049832, -0.0007311],
            [-0.9817314, -0.1902717, 0.0021921],
            [-0.9251284, -0.3796502, -0.0019137],
        ],
        dtype=np.float32,
    )
    entries = [_entry(peer_id, vector.tolist()) for peer_id, vector in enumerate(vectors)]
    split = split_entries(entries, seed=42, delta=0.0)
    assert split is not None
    left_center, right_center, _left, _right = split

    tree = _CentroidTree(
        record_limit_per_leafnode=3,
        delta=0.0,
        seed=42,
        capacity_metric="entries",
        normalize_split_centroids=True,
        eager_recursive_splitting=True,
    )
    for peer_id, vector in enumerate(vectors):
        tree.position_record(
            {
                "CloneID": peer_id,
                "OwnerID": peer_id,
                "CentroidIndex": 0,
                "embedding": vector,
            }
        )
    assert isinstance(tree.root, _CentroidTreeSplitNode)
    np.testing.assert_array_equal(left_center, np.asarray(tree.root.centroid_left, dtype=np.float32))
    np.testing.assert_array_equal(right_center, np.asarray(tree.root.centroid_right, dtype=np.float32))


def test_internal_insert_redirect_and_failed_split_retry_state(tmp_path: Path) -> None:
    shard = tmp_path / "peer.npz"
    np.savez_compressed(
        shard,
        training_doc_ids=np.asarray(["local"]),
        training_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        query_id=np.asarray("q"),
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        target_doc_id=np.asarray("target"),
    )

    class QueueStub:
        def put(self, _value: object) -> None:
            pass

    async def scenario() -> None:
        runtime = PeerRuntime(0, shard, tmp_path / "tree-handler", QueueStub())
        runtime.parameters = {"leaf_capacity": 1, "seed": 42, "delta": 0.0}
        child = ChildDescriptor("child", 0, Endpoint("127.0.0.1", 1), np.asarray([1, 0]))
        runtime.nodes["internal"] = InternalNode(
            "internal", 1, 0, None, child, child
        )
        redirected = await runtime._on_centroid_insert(
            {"node_id": "internal", "entry": _entry(9, [1, 0]).to_wire()}, {}
        )
        assert redirected[0] == "TREE_REDIRECT"

        runtime.nodes["leaf"] = LeafNode(
            "leaf", 1, 0, None, entries=[_entry(0, [1, 0])]
        )
        await runtime._on_centroid_insert(
            {"node_id": "leaf", "entry": _entry(1, [1, 0]).to_wire()}, {}
        )
        leaf = runtime.nodes["leaf"]
        assert isinstance(leaf, LeafNode)
        assert leaf.failed_split_size == 2
        assert not runtime._should_split(leaf)

        # A subsequent new entry changes the overflow size, so a split is
        # attempted again and the diagnosed stable size advances.
        await runtime._on_centroid_insert(
            {"node_id": "leaf", "entry": _entry(2, [1, 0]).to_wire()}, {}
        )
        assert leaf.failed_split_size == 3
        assert not runtime._should_split(leaf)
        runtime.log_handle.close()

    asyncio.run(scenario())


def test_joint_assignment_local_topk_and_out_of_order_merge() -> None:
    assigned, underfilled = assign_unique_candidates(
        [{1: 0.9, 2: 0.8}, {1: 0.95, 3: 0.7}], [1, 1]
    )
    assert assigned == [[(2, 0.8)], [(1, 0.95)]]
    assert underfilled == []
    local = local_topk(
        np.asarray([1.0, 0.0]),
        ["b", "a", "c"],
        np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        3,
    )
    assert [doc for doc, _score in local] == ["a", "b", "c"]
    merged = []
    by_hop = {2: [("b", 0.9), ("c", 0.7)], 1: [("a", 0.9), ("b", 0.8)]}
    for hop in sorted(by_hop):
        merged = merge_topk(merged, by_hop[hop], 3)
    assert merged == [("a", 0.9), ("b", 0.9), ("c", 0.7)]


def test_local_topk_matches_centralized_float32_operation_order() -> None:
    from types import SimpleNamespace

    from unified_retrieval_benchmark import _local_topk

    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(133, 31)).astype(np.float32)
    query = rng.normal(size=31).astype(np.float32)
    doc_ids = np.asarray([f"doc-{index:03d}" for index in range(len(embeddings))])

    central_dataset = SimpleNamespace(
        doc_embeddings=embeddings,
        doc_norms=np.linalg.norm(embeddings, axis=1).astype(np.float32),
        doc_ids=doc_ids,
    )

    central_query = np.asarray(query, dtype=np.float32)[None, :]
    central_query /= np.linalg.norm(central_query, axis=1, keepdims=True)
    expected = _local_topk(
        central_dataset,
        central_query[0],
        np.arange(len(embeddings), dtype=np.int64),
        100,
    )
    actual = local_topk(
        central_query[0],
        doc_ids,
        embeddings,
        100,
        query_is_normalized=True,
    )
    assert [doc for doc, _score in actual] == [
        str(doc_ids[index]) for index, _score in expected
    ]
    np.testing.assert_array_equal(
        np.asarray([score for _doc, score in actual]),
        np.asarray([score for _index, score in expected]),
    )


def test_query_results_arriving_out_of_order_over_tcp_merge_by_hop(tmp_path: Path) -> None:
    shard = tmp_path / "peer.npz"
    np.savez_compressed(
        shard,
        training_doc_ids=np.asarray(["local"]),
        training_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        query_id=np.asarray("q"),
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        target_doc_id=np.asarray("target"),
    )

    class QueueStub:
        def put(self, _value: object) -> None:
            pass

    async def scenario() -> None:
        runtime = PeerRuntime(0, shard, tmp_path / "collector", QueueStub())
        runtime.epoch = "query-epoch"
        runtime.query_states["q"] = QueryState(
            "q", np.asarray([1.0, 0.0], dtype=np.float32), "target"
        )
        server = await asyncio.start_server(runtime._handle_client, "127.0.0.1", 0)
        runtime.endpoint = Endpoint("127.0.0.1", int(server.sockets[0].getsockname()[1]))
        client = RpcClient(9, Endpoint("127.0.0.1", 9999), runtime.epoch, timeout=2)
        async with server:
            await client.call(
                runtime.endpoint,
                "QUERY_RESULT",
                {"query_id": "q", "hop": 2, "peer_id": 2, "results": [["b", 0.9]]},
            )
            await client.call(
                runtime.endpoint,
                "QUERY_RESULT",
                {"query_id": "q", "hop": 1, "peer_id": 1, "results": [["a", 0.9]]},
            )
        state = runtime.query_states["q"]
        for hop in sorted(state.results_by_hop):
            state.final_results = merge_topk(state.final_results, state.results_by_hop[hop], 2)
        assert state.peer_by_hop == {2: 2, 1: 1}
        assert state.final_results == [("a", 0.9), ("b", 0.9)]
        runtime.log_handle.close()

    asyncio.run(scenario())


def _epoch_with_self_and_remote_root_children(peer_count: int) -> str:
    peers = {peer: Endpoint("127.0.0.1", 10_000 + peer) for peer in range(peer_count)}
    for index in range(10_000):
        epoch = f"socket-integration-{index}"
        manifest = MembershipManifest(epoch, 42, peers, {})
        root = root_node_id(epoch)
        root_peer = custodian_for(root, manifest)
        child_peers = {
            custodian_for(child_node_id(epoch, root, side), manifest)
            for side in ("left", "right")
        }
        if root_peer in child_peers and any(peer != root_peer for peer in child_peers):
            return epoch
    raise AssertionError("Could not construct deterministic custody fixture")


def _write_small_prepared_fixture(path: Path) -> dict[str, object]:
    peer_count = 4
    (path / "shards").mkdir(parents=True)
    bases = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    all_doc_ids: list[str] = []
    all_doc_embeddings: list[np.ndarray] = []
    query_embeddings: list[np.ndarray] = []
    shard_digest = hashlib.sha256()
    for peer_id in range(peer_count):
        vectors = np.vstack(
            [
                bases[peer_id],
                bases[peer_id] + np.asarray([0.03, 0.02, 0.01], dtype=np.float32),
                -bases[peer_id],
                -bases[peer_id] + np.asarray([0.01, -0.02, 0.03], dtype=np.float32),
            ]
        )
        target_peer = (peer_id + 1) % peer_count
        all_doc_ids.extend(f"doc-{peer_id}-{index}" for index in range(4))
        all_doc_embeddings.extend(vectors)
        query_embeddings.append(bases[target_peer])
        shard_path = path / "shards" / f"peer_{peer_id:05d}.npz"
        np.savez_compressed(
            shard_path,
            peer_id=np.asarray(peer_id, dtype=np.int32),
            training_doc_indices=np.arange(peer_id * 4, peer_id * 4 + 4, dtype=np.int32),
            training_doc_ids=np.asarray([f"doc-{peer_id}-{index}" for index in range(4)]),
            training_embeddings=vectors,
            query_index=np.asarray(peer_id, dtype=np.int32),
            query_id=np.asarray(f"query-{peer_id}"),
            query_embedding=bases[target_peer],
            target_doc_index=np.asarray(target_peer * 4, dtype=np.int32),
            target_doc_id=np.asarray(f"doc-{target_peer}-0"),
            target_available=np.asarray(True),
        )
        shard_digest.update(shard_path.read_bytes())
    embedding_dir = path / "embedding_bundle"
    embedding_dir.mkdir()
    doc_ids_path = embedding_dir / "doc_ids.npy"
    doc_embeddings_path = embedding_dir / "doc_embeddings.npy"
    query_keys_path = embedding_dir / "query_keys.npy"
    query_embeddings_path = embedding_dir / "query_embeddings.npy"
    np.save(doc_ids_path, np.asarray(all_doc_ids), allow_pickle=False)
    np.save(doc_embeddings_path, np.asarray(all_doc_embeddings, dtype=np.float32), allow_pickle=False)
    np.save(query_keys_path, np.asarray([f"query-{peer}" for peer in range(peer_count)]), allow_pickle=False)
    np.save(query_embeddings_path, np.asarray(query_embeddings, dtype=np.float32), allow_pickle=False)
    bundle_manifest = embedding_dir / "manifest.json"
    bundle_manifest.write_text("{}", encoding="utf-8")
    fixture = path / "central_fixture.npz"
    np.savez_compressed(
        fixture,
        peer_ids=np.arange(peer_count, dtype=np.int32),
        training_doc_indices=np.arange(peer_count * 4, dtype=np.int32).reshape(peer_count, 4),
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
    manifest: dict[str, object] = {
        "schema": "decentralized_semantica_prepared_v1",
        "dataset": "unit-fixture",
        "epoch": _epoch_with_self_and_remote_root_children(peer_count),
        "seed": 42,
        "parameters": parameters,
        "peer_ids": list(range(peer_count)),
        "endpoints": {},
        "shards_sha256": shard_digest.hexdigest(),
        "fixture": str(fixture.resolve()),
        "embedding_bundle": {
            "manifest_path": str(bundle_manifest.resolve()),
            "doc_ids_path": str(doc_ids_path.resolve()),
            "doc_embeddings_path": str(doc_embeddings_path.resolve()),
            "query_keys_path": str(query_keys_path.resolve()),
            "query_embeddings_path": str(query_embeddings_path.resolve()),
            "model_name": "sentence-transformers/all-mpnet-base-v2",
        },
    }
    (path / "prepared_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def test_process_isolated_port_end_to_end(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    prepared = _write_small_prepared_fixture(prepared_dir)
    run_dir = tmp_path / "run"
    result = run_experiment(
        prepared_dir,
        run_dir,
        endpoint_timeout_seconds=30,
        run_timeout_seconds=120,
        rpc_timeout_seconds=15,
        start_method="spawn",
    )
    assert result["status"] == "complete"
    assert len(list((run_dir / "query_paths_results").glob("peer_*.msgpack"))) == 4
    membership = MembershipManifest.from_wire(
        json.loads((run_dir / "membership_manifest.json").read_text())
    )
    assert len({endpoint.port for endpoint in membership.peers.values()}) == 4
    assert all(endpoint.port > 0 for endpoint in membership.peers.values())
    states = load_peer_states(run_dir)
    assert set(states) == {0, 1, 2, 3}
    root = root_node_id(str(prepared["epoch"]))
    root_peer = custodian_for(root, membership)
    root_node = states[root_peer]["nodes"][root]
    assert root_node["node_type"] == "internal"
    child_custodians = {
        int(root_node["left"]["custodian_peer_id"]),
        int(root_node["right"]["custodian_peer_id"]),
    }
    assert root_peer in child_custodians
    assert any(peer != root_peer for peer in child_custodians)
    all_nodes = [node for state in states.values() for node in state["nodes"].values()]
    assert max(int(node["depth"]) for node in all_nodes) >= 2
    assert any(
        node["node_type"] == "internal" and int(node["depth"]) >= 1
        for node in all_nodes
    )
    for state in states.values():
        assert set(state["contacts_by_round"]) == {"0", "1", "2"}
        assert len(state["queries"]) == 1
    root_log = run_dir / "logs" / f"peer_{root_peer:05d}.jsonl"
    log_rows = [json.loads(line) for line in root_log.read_text().splitlines()]
    round1_done = [
        row["time"]
        for row in log_rows
        if row["direction"] == "receive"
        and row["message_type"] == "EXPANSION_DONE"
        and row["round"] == 1
    ]
    round2_start = [
        row["time"]
        for row in log_rows
        if row["direction"] == "send"
        and row["message_type"] == "EXPANSION_START"
        and row["round"] == 2
    ]
    assert len(round1_done) == 4 and len(round2_start) == 4
    assert max(round1_done) <= min(round2_start)
    round0_done = [
        row["time"]
        for row in log_rows
        if row["direction"] == "receive" and row["message_type"] == "ROUND0_DONE"
    ]
    round1_start = [
        row["time"]
        for row in log_rows
        if row["direction"] == "send"
        and row["message_type"] == "EXPANSION_START"
        and row["round"] == 1
    ]
    round2_done = [
        row["time"]
        for row in log_rows
        if row["direction"] == "receive"
        and row["message_type"] == "EXPANSION_DONE"
        and row["round"] == 2
    ]
    query_start = [
        row["time"]
        for row in log_rows
        if row["direction"] == "send" and row["message_type"] == "QUERY_START"
    ]
    assert len(round0_done) == len(round1_start) == 4
    assert len(round2_done) == len(query_start) == 4
    assert max(round0_done) <= min(round1_start)
    assert max(round2_done) <= min(query_start)
    create_leaf_sends = [
        row for row in log_rows if row["direction"] == "send" and row["message_type"] == "CREATE_LEAF"
    ]
    assert any(row["remote"]["port"] == membership.peers[root_peer].port for row in create_leaf_sends)
    assert any(row["remote"]["port"] != membership.peers[root_peer].port for row in create_leaf_sends)
    for row in log_rows:
        assert {
            "source_peer_id",
            "source_endpoint",
            "destination_peer_id",
            "destination_endpoint",
        } <= row.keys()
    peer_receive_rows = [
        row
        for row in log_rows
        if row["direction"] == "receive" and int(row["source_peer_id"]) >= 0
    ]
    assert peer_receive_rows
    assert all(
        Endpoint.from_wire(row["source_endpoint"])
        == membership.peers[int(row["source_peer_id"])]
        for row in peer_receive_rows
    )
    report = validate_run(prepared_dir, run_dir, structural_only=False)
    assert report["status"] == "passed"
    assert report["oracle"]["rounds_compared"] == 3
    assert report["oracle"]["queries_compared"] == 4
    assert (
        report["oracle"]["oracle"]
        == "unified_retrieval_benchmark_sequential_query_trace"
    )
    summary = json.loads((run_dir / "network_summary.json").read_text())
    assert summary["messages_by_type"]["CREATE_LEAF"] >= 2
    assert summary["messages_by_type"]["QUERY_HOP"] >= 4


def test_launcher_aborts_promptly_when_protocol_cannot_fill_degree(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    manifest = _write_small_prepared_fixture(prepared_dir)
    manifest["parameters"]["degree"] = 5
    (prepared_dir / "prepared_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    run_dir = tmp_path / "failed-run"
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="Peer process failure"):
        run_experiment(
            prepared_dir,
            run_dir,
            endpoint_timeout_seconds=20,
            run_timeout_seconds=30,
            rpc_timeout_seconds=5,
            start_method="spawn",
        )
    assert time.monotonic() - started < 15
    failure = json.loads((run_dir / "launcher_failure.json").read_text())
    assert failure["error"]
    assert "traceback" in failure
    assert (run_dir / "root_failure.log").exists()

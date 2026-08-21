from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

DEFAULT_PUBMEDQA_DATA_ROOT = Path(
    "/var/scratch/pmneague/Low_Coord_Semantic_DHT/data/"
    "pubmedqa_expanded_all_mesh_alpha0p00007_q30/pubmedqa"
)
DEFAULT_PUBMEDQA_OVERLAY_ROOT = Path(
    "/var/scratch/pmneague/Low_Coord_Semantic_DHT/cache/"
    "pubmedqa_alpha0p0005_events200_train133_test67"
)


def overlay_dir(overlay_root: Path, seed: int) -> Path:
    return overlay_root / (
        f"seed_{int(seed)}__peers_50000__alpha_0p0005__events_200__"
        "train_133__test_67__docs_per_peer_133"
    )


def default_parameters(peer_count: int = 100, seed: int = 42) -> Dict[str, Any]:
    return {
        "peer_count": int(peer_count),
        "seed": int(seed),
        "degree": 20,
        "leaf_capacity": 20,
        "delta": 0.0,
        "max_centroids": 8,
        "expansion_rounds": 20,
        "query_hop_limit": 100,
        "local_top_k": 100,
        "centroid_method": "cosine_silhouette",
        "protected_random_fraction": 0.0,
    }


def prepare_pubmedqa_shards(
    output_dir: Path,
    *,
    data_root: Path = DEFAULT_PUBMEDQA_DATA_ROOT,
    overlay_root: Path = DEFAULT_PUBMEDQA_OVERLAY_ROOT,
    peer_count: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    # Keep the unified benchmark import out of peer processes; only the
    # trusted staging command needs its full dataset loader.
    from unified_retrieval_benchmark import EmbeddingBundle

    output_dir = Path(output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    source_overlay = overlay_dir(Path(overlay_root), int(seed))
    if not source_overlay.exists():
        raise FileNotFoundError(f"Missing PubMedQA overlay: {source_overlay}")
    bundle = EmbeddingBundle.load(str(data_root))
    peer_docs = np.load(source_overlay / "peer_doc_indices.npy", mmap_mode="r", allow_pickle=False)
    peer_test_docs = np.load(
        source_overlay / "peer_test_doc_indices.npy", mmap_mode="r", allow_pickle=False
    )
    peer_test_queries = np.load(
        source_overlay / "peer_test_query_indices.npy", mmap_mode="r", allow_pickle=False
    )
    if int(peer_count) < 2 or int(peer_count) > int(peer_docs.shape[0]):
        raise ValueError("peer_count must select at least two available overlay peers")
    selected_docs = np.asarray(peer_docs[:peer_count], dtype=np.int32)
    selected_targets = np.asarray(peer_test_docs[:peer_count, 0], dtype=np.int32)
    selected_queries = np.asarray(peer_test_queries[:peer_count, 0], dtype=np.int32)
    doc_embeddings = np.load(bundle.doc_embeddings_path, mmap_mode="r", allow_pickle=False)
    query_embeddings = np.load(bundle.query_embeddings_path, mmap_mode="r", allow_pickle=False)
    doc_ids = np.load(bundle.doc_ids_path, mmap_mode="r", allow_pickle=False)
    query_ids = np.load(bundle.query_keys_path, mmap_mode="r", allow_pickle=False)
    owners: Dict[int, set[int]] = {}
    for peer_id, row in enumerate(selected_docs):
        for doc_index in row.tolist():
            owners.setdefault(int(doc_index), set()).add(int(peer_id))
    availability = np.asarray(
        [
            any(owner != peer_id for owner in owners.get(int(target), set()))
            for peer_id, target in enumerate(selected_targets.tolist())
        ],
        dtype=np.bool_,
    )
    digest = hashlib.sha256()
    for peer_id in range(int(peer_count)):
        train_indices = selected_docs[peer_id]
        # Store the private source rows as float32.  Deterministic kernels own
        # normalization, matching the centralized benchmark's single pass.
        train_vectors = np.asarray(doc_embeddings[train_indices], dtype=np.float32)
        query_vector = np.asarray(
            query_embeddings[int(selected_queries[peer_id])], dtype=np.float32
        )
        target_index = int(selected_targets[peer_id])
        path = shard_dir / f"peer_{peer_id:05d}.npz"
        np.savez_compressed(
            path,
            peer_id=np.asarray(peer_id, dtype=np.int32),
            training_doc_indices=train_indices,
            training_doc_ids=np.asarray(doc_ids[train_indices]),
            training_embeddings=train_vectors,
            query_index=np.asarray(selected_queries[peer_id], dtype=np.int32),
            query_id=np.asarray(str(query_ids[int(selected_queries[peer_id])])),
            query_embedding=query_vector,
            target_doc_index=np.asarray(target_index, dtype=np.int32),
            target_doc_id=np.asarray(str(doc_ids[target_index])),
            target_available=np.asarray(availability[peer_id], dtype=np.bool_),
        )
        digest.update(path.read_bytes())
    parameters = default_parameters(peer_count=peer_count, seed=seed)
    epoch = hashlib.sha256(
        json.dumps(
            {
                "dataset": "pubmedqa",
                "seed": int(seed),
                "peer_ids": list(range(int(peer_count))),
                "parameters": parameters,
                "shards_sha256": digest.hexdigest(),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:24]
    fixture_path = output_dir / "central_fixture.npz"
    np.savez_compressed(
        fixture_path,
        peer_ids=np.arange(int(peer_count), dtype=np.int32),
        training_doc_indices=selected_docs,
        query_indices=selected_queries,
        target_doc_indices=selected_targets,
        target_available=availability,
    )
    manifest = {
        "schema": "decentralized_semantica_prepared_v2",
        "dataset": "pubmedqa",
        "epoch": epoch,
        "seed": int(seed),
        "parameters": parameters,
        "data_root": str(Path(data_root).resolve()),
        "overlay_dir": str(source_overlay.resolve()),
        "embedding_bundle": {
            "manifest_path": bundle.manifest_path,
            "doc_ids_path": bundle.doc_ids_path,
            "doc_embeddings_path": bundle.doc_embeddings_path,
            "query_keys_path": bundle.query_keys_path,
            "query_embeddings_path": bundle.query_embeddings_path,
            "model_name": bundle.model_name,
        },
        "query_policy": "held_out_position_0_per_selected_peer",
        "shards_sha256": digest.hexdigest(),
        "target_available_count": int(np.count_nonzero(availability)),
        "fixture": str(fixture_path.resolve()),
    }
    (output_dir / "prepared_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def load_shard(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}

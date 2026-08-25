from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np


def default_parameters(peer_count: int, seed: int) -> Dict[str, Any]:
    """Small defaults that exercise joining, splitting, contacts, and queries."""
    if int(peer_count) < 2:
        raise ValueError("peer_count must be at least two")
    degree = min(2, int(peer_count) - 1)
    return {
        "peer_count": int(peer_count),
        "seed": int(seed),
        "degree": degree,
        "leaf_capacity": 2,
        "delta": 0.0,
        "max_centroids": min(2, degree),
        "expansion_rounds": 1,
        "query_hop_limit": int(peer_count),
        "local_top_k": 3,
    }


def prepare_demo_shards(
    output_dir: Path,
    *,
    peer_count: int = 4,
    documents_per_peer: int = 4,
    dimensions: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create deterministic synthetic private shards for a local simulation."""
    peer_count = int(peer_count)
    documents_per_peer = int(documents_per_peer)
    dimensions = int(dimensions)
    seed = int(seed)
    if peer_count < 2:
        raise ValueError("peer_count must be at least two")
    if documents_per_peer < 2:
        raise ValueError("documents_per_peer must be at least two")
    if dimensions < 2:
        raise ValueError("dimensions must be at least two")

    output_dir = Path(output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()

    bases: list[np.ndarray] = []
    for peer_id in range(peer_count):
        angle = 2.0 * math.pi * peer_id / peer_count
        base = np.zeros(dimensions, dtype=np.float32)
        base[0] = math.cos(angle)
        base[1] = math.sin(angle)
        bases.append(base)

    for peer_id, base in enumerate(bases):
        split = max(1, documents_per_peer // 2)
        vectors = []
        for document_index in range(documents_per_peer):
            center = base if document_index < split else -base
            noise = rng.normal(0.0, 0.025, size=dimensions).astype(np.float32)
            vectors.append(center + noise)
        embeddings = np.asarray(vectors, dtype=np.float32)
        indices = np.arange(
            peer_id * documents_per_peer,
            (peer_id + 1) * documents_per_peer,
            dtype=np.int32,
        )
        target_peer = (peer_id + 1) % peer_count
        shard_path = shard_dir / f"peer_{peer_id:05d}.npz"
        np.savez_compressed(
            shard_path,
            peer_id=np.asarray(peer_id, dtype=np.int32),
            training_doc_indices=indices,
            training_doc_ids=np.asarray(
                [f"doc-{peer_id}-{index}" for index in range(documents_per_peer)]
            ),
            training_embeddings=embeddings,
            query_index=np.asarray(peer_id, dtype=np.int32),
            query_id=np.asarray(f"query-{peer_id}"),
            query_embedding=bases[target_peer],
            target_doc_index=np.asarray(
                target_peer * documents_per_peer, dtype=np.int32
            ),
            target_doc_id=np.asarray(f"doc-{target_peer}-0"),
            target_available=np.asarray(True),
        )
        digest.update(shard_path.read_bytes())

    parameters = default_parameters(peer_count, seed)
    epoch = hashlib.sha256(
        json.dumps(
            {
                "dataset": "synthetic-demo",
                "seed": seed,
                "peer_count": peer_count,
                "shards_sha256": digest.hexdigest(),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:24]
    manifest = {
        "schema": "decentralized_semantica_demo_v1",
        "dataset": "synthetic-demo",
        "epoch": epoch,
        "seed": seed,
        "parameters": parameters,
        "shards_sha256": digest.hexdigest(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prepared_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def prepare_aol_shards(
    bundle_path: Path,
    output_dir: Path,
    *,
    peer_count: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Convert a compact AOL peer bundle into the standard private shards."""
    bundle_path = Path(bundle_path)
    output_dir = Path(output_dir)
    with np.load(bundle_path, allow_pickle=False) as bundle:
        document_ids = np.asarray(bundle["document_ids"])
        document_embeddings = np.asarray(
            bundle["document_embeddings"], dtype=np.float32
        )
    if document_ids.ndim != 2 or document_embeddings.ndim != 3:
        raise ValueError("AOL bundle arrays must have shapes [peers, documents, ...]")
    if document_ids.shape != document_embeddings.shape[:2]:
        raise ValueError("AOL document IDs and embeddings have incompatible shapes")
    if document_ids.dtype.kind == "S":
        document_ids = np.char.decode(document_ids, "utf-8")
    else:
        document_ids = document_ids.astype(str)
    peer_count = int(peer_count)
    if peer_count < 2 or peer_count > int(document_ids.shape[0]):
        raise ValueError(
            f"peer_count must be between 2 and {int(document_ids.shape[0])}"
        )

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    documents_per_peer = int(document_ids.shape[1])
    digest = hashlib.sha256()
    for peer_id in range(peer_count):
        target_peer = (peer_id + 1) % peer_count
        indices = np.arange(
            peer_id * documents_per_peer,
            (peer_id + 1) * documents_per_peer,
            dtype=np.int32,
        )
        target_doc_id = str(document_ids[target_peer, 0])
        shard_path = shard_dir / f"peer_{peer_id:05d}.npz"
        np.savez_compressed(
            shard_path,
            peer_id=np.asarray(peer_id, dtype=np.int32),
            training_doc_indices=indices,
            training_doc_ids=np.asarray(document_ids[peer_id]),
            training_embeddings=np.asarray(
                document_embeddings[peer_id], dtype=np.float32
            ),
            query_index=np.asarray(peer_id, dtype=np.int32),
            query_id=np.asarray(f"aol-query-{peer_id}"),
            query_embedding=np.asarray(
                document_embeddings[target_peer, 0], dtype=np.float32
            ),
            target_doc_index=np.asarray(
                target_peer * documents_per_peer, dtype=np.int32
            ),
            target_doc_id=np.asarray(target_doc_id),
            target_available=np.asarray(True),
        )
        digest.update(shard_path.read_bytes())

    parameters = default_parameters(peer_count, int(seed))
    epoch = hashlib.sha256(
        json.dumps(
            {
                "dataset": "aol-ia",
                "seed": int(seed),
                "peer_count": peer_count,
                "shards_sha256": digest.hexdigest(),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:24]
    manifest = {
        "schema": "decentralized_semantica_aol_demo_v1",
        "dataset": "aol-ia",
        "epoch": epoch,
        "seed": int(seed),
        "parameters": parameters,
        "shards_sha256": digest.hexdigest(),
    }
    (output_dir / "prepared_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def load_shard(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}

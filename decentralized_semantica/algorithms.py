from __future__ import annotations

import hashlib
import math
import struct
import warnings
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from .models import (
    CentroidEntry,
    CustodyProof,
    Endpoint,
    PeerDescriptor,
    PeerProfile,
)


def stable_int(value: object, bits: int = 63) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << bits) - 1)


def normalize_vector(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.zeros_like(value, dtype=np.float32)
    return value / norm


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[~np.isfinite(norms) | (norms <= 0.0)] = 1.0
    return values / norms


def _normalize_split_center(value: np.ndarray) -> np.ndarray:
    """Match the centralized tree's float64 split-center normalization."""
    value = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= 0.0 or not np.isfinite(norm):
        return value
    return value / norm


def allocate_centroid_contact_caps(
    cluster_doc_counts: Sequence[int],
    degree: int,
    *,
    min_quota: int = 1,
) -> List[int]:
    """Benchmark-equivalent proportional quota allocation."""
    counts = [max(0, int(value)) for value in cluster_doc_counts]
    if not counts:
        return []
    budget = max(0, int(degree))
    if len(counts) == 1:
        return [budget]
    minimum = max(0, int(min_quota))
    if budget < len(counts) * minimum:
        raise ValueError("contact budget is too small for the requested minimum quota")
    base = [minimum for _ in counts]
    remaining = budget - sum(base)
    if remaining <= 0:
        return base
    weights = np.asarray(counts, dtype=float)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones(len(counts), dtype=float)
    raw = remaining * weights / float(np.sum(weights))
    extras = np.floor(raw).astype(int)
    remainder = remaining - int(np.sum(extras))
    fractions = raw - extras
    order = sorted(
        range(len(counts)),
        key=lambda index: (-float(fractions[index]), -counts[index], index),
    )
    for index in order[:remainder]:
        extras[index] += 1
    return [int(base_value + extra) for base_value, extra in zip(base, extras)]


def fit_peer_profile(
    peer_id: int,
    endpoint: Endpoint,
    training_embeddings: np.ndarray,
    *,
    degree: int,
    seed: int,
    max_k: int = 8,
) -> PeerProfile:
    values = normalize_rows(training_embeddings)
    peer_seed = stable_int(f"centroids:{seed}:{peer_id}", bits=31)
    selected_k = 1
    best_score = -np.inf
    capped = max(1, min(int(max_k), len(values), int(degree)))
    if len(values) >= 3 and capped >= 2:
        for k in range(2, capped + 1):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Number of distinct clusters")
                with threadpool_limits(limits=1):
                    labels = KMeans(
                        n_clusters=k, random_state=peer_seed, n_init=10
                    ).fit_predict(values)
            unique = np.unique(labels)
            if len(unique) < 2 or len(unique) >= len(values):
                continue
            score = float(silhouette_score(values, labels, metric="cosine"))
            if score > best_score:
                best_score = score
                selected_k = k
    # The centralized layout first obtains normalized document rows and then
    # normalizes again inside its fitting helper. Preserve that second
    # float32 rounding pass so process-isolated peers are bitwise identical.
    fit_values = normalize_rows(values)
    if selected_k == 1:
        centers = normalize_vector(np.mean(fit_values, axis=0)).reshape(1, -1)
        counts = np.asarray([len(fit_values)], dtype=np.int32)
    else:
        with threadpool_limits(limits=1):
            model = KMeans(
                n_clusters=selected_k, random_state=peer_seed, n_init=10
            ).fit(fit_values)
        raw_counts = np.bincount(model.labels_, minlength=selected_k)
        nonempty = np.flatnonzero(raw_counts > 0)
        order = sorted(nonempty.tolist(), key=lambda position: (-int(raw_counts[position]), int(position)))
        centers = normalize_rows(np.asarray(model.cluster_centers_[order], dtype=np.float32))
        counts = np.asarray([raw_counts[position] for position in order], dtype=np.int32)
    caps = allocate_centroid_contact_caps(counts.tolist(), degree, min_quota=1)
    return PeerProfile(
        peer_id=peer_id,
        endpoint=endpoint,
        centroids=centers.astype(np.float32, copy=False),
        cluster_doc_counts=counts,
        contact_caps=np.asarray(caps, dtype=np.int32),
    )


def root_node_id(epoch: str) -> str:
    return hashlib.sha256(str(epoch).encode("utf-8") + b"root").hexdigest()


def child_node_id(epoch: str, parent_node_id: str, side: str) -> str:
    if side not in {"left", "right"}:
        raise ValueError("Child side must be left or right")
    return hashlib.sha256(
        str(epoch).encode("utf-8")
        + str(parent_node_id).encode("ascii")
        + str(side).encode("ascii")
    ).hexdigest()


CUSTODY_DOMAIN = b"semantica-custody-v1"
GENESIS_DOMAIN = b"semantica-genesis-v1"


def _length_prefixed(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def custody_digest(
    epoch: str,
    node_id: str,
    custody_seed: int,
    candidate_peer_ids: Iterable[int],
) -> str:
    candidates = tuple(sorted({int(peer_id) for peer_id in candidate_peer_ids}))
    if not candidates:
        raise ValueError("Cannot select a custodian without child owners")
    if int(custody_seed) < 0:
        raise ValueError("Custody seed cannot be negative")
    if any(peer_id < 0 for peer_id in candidates):
        raise ValueError("Custodian peer IDs must be non-negative")
    epoch_bytes = str(epoch).encode("utf-8")
    node_bytes = str(node_id).encode("ascii")
    candidate_bytes = b"".join(
        _length_prefixed(struct.pack(">Q", peer_id)) for peer_id in candidates
    )
    material = (
        CUSTODY_DOMAIN
        + epoch_bytes
        + struct.pack(">Q", int(custody_seed))
        + node_bytes
        + candidate_bytes
    )
    return hashlib.sha256(material).hexdigest()


def custody_proof_for_entries(
    epoch: str,
    node_id: str,
    custody_seed: int,
    entries: Sequence[CentroidEntry],
) -> CustodyProof:
    owners: Dict[int, PeerDescriptor] = {}
    for entry in entries:
        descriptor = entry.owner
        existing = owners.get(descriptor.peer_id)
        if existing is not None and existing.endpoint != descriptor.endpoint:
            raise ValueError(
                f"Peer {descriptor.peer_id} has conflicting child-owner descriptors"
            )
        if existing is None or descriptor.profile_version > existing.profile_version:
            owners[descriptor.peer_id] = descriptor
    candidates = tuple(sorted(owners))
    digest = custody_digest(epoch, node_id, custody_seed, candidates)
    selected_index = int.from_bytes(bytes.fromhex(digest), "big") % len(candidates)
    custodian = owners[candidates[selected_index]]
    return CustodyProof(
        seed=int(custody_seed),
        candidate_peer_ids=candidates,
        digest=digest,
        custodian=custodian,
    )


def genesis_custody_proof(
    epoch: str, node_id: str, custodian: PeerDescriptor
) -> CustodyProof:
    material = (
        GENESIS_DOMAIN
        + _length_prefixed(str(epoch).encode("utf-8"))
        + _length_prefixed(str(node_id).encode("ascii"))
        + _length_prefixed(struct.pack(">Q", int(custodian.peer_id)))
    )
    return CustodyProof(
        seed=0,
        candidate_peer_ids=(int(custodian.peer_id),),
        digest=hashlib.sha256(material).hexdigest(),
        custodian=custodian,
    )


def verify_custody_proof(
    epoch: str,
    node_id: str,
    custody_seed: int,
    entries: Sequence[CentroidEntry],
    proof: CustodyProof,
) -> bool:
    try:
        expected = custody_proof_for_entries(
            epoch, node_id, custody_seed, entries
        )
    except ValueError:
        return False
    return expected == proof


def split_entries(
    entries: Sequence[CentroidEntry],
    *,
    seed: int,
    delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, List[CentroidEntry], List[CentroidEntry]] | None:
    buckets: Dict[tuple[tuple[int, ...], str, bytes], List[CentroidEntry]] = defaultdict(list)
    vectors: Dict[tuple[tuple[int, ...], str, bytes], np.ndarray] = {}
    for entry in entries:
        # Fit split points in NumPy's float64 dtype, then round centers to
        # float32 only when they cross the protocol boundary.
        array = np.ascontiguousarray(np.asarray(entry.vector, dtype=float))
        key = (tuple(array.shape), array.dtype.str, array.tobytes())
        buckets[key].append(entry)
        vectors[key] = array
    keys = list(buckets)
    if len(keys) < 2:
        return None
    embeddings = np.asarray([vectors[key] for key in keys], dtype=float)
    weights = np.asarray([len(buckets[key]) for key in keys], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Number of distinct clusters")
        with threadpool_limits(limits=1):
            model = KMeans(n_clusters=2, random_state=int(seed), n_init=10).fit(
                embeddings, sample_weight=weights
            )
    left_center = _normalize_split_center(model.cluster_centers_[0])
    right_center = _normalize_split_center(model.cluster_centers_[1])
    if np.isnan(left_center).any() or np.isnan(right_center).any():
        return None
    left: List[CentroidEntry] = []
    right: List[CentroidEntry] = []
    separated = 0
    for key in keys:
        vector = vectors[key]
        left_distance = float(np.linalg.norm(vector - left_center))
        right_distance = float(np.linalg.norm(vector - right_center))
        if abs(left_distance - right_distance) < float(delta):
            left.extend(buckets[key])
            right.extend(buckets[key])
        elif left_distance <= right_distance:
            left.extend(buckets[key])
            separated += 1
        else:
            right.extend(buckets[key])
            separated += 1
    if not left or not right or separated < int(math.ceil(len(keys) / 2.0)):
        return None
    return (
        np.asarray(left_center, dtype=np.float32),
        np.asarray(right_center, dtype=np.float32),
        left,
        right,
    )


def choose_child(vector: np.ndarray, left_center: np.ndarray, right_center: np.ndarray) -> str:
    left_distance = float(np.linalg.norm(np.asarray(vector) - np.asarray(left_center)))
    right_distance = float(np.linalg.norm(np.asarray(vector) - np.asarray(right_center)))
    return "left" if left_distance < right_distance else "right"


def peer_centroid_score(source_centroid: np.ndarray, profile: PeerProfile) -> float:
    if profile.centroids.size == 0:
        return -np.inf
    return float(np.max(profile.centroids @ np.asarray(source_centroid, dtype=np.float32)))


def assign_unique_candidates(
    candidate_scores: Sequence[Mapping[int, float]],
    caps: Sequence[int],
) -> tuple[List[List[Tuple[int, float]]], List[int]]:
    ranked = sorted(
        (
            (float(score), int(candidate), int(centroid_index))
            for centroid_index, values in enumerate(candidate_scores)
            for candidate, score in values.items()
            if np.isfinite(score)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    assigned: List[List[Tuple[int, float]]] = [[] for _ in caps]
    used: set[int] = set()
    for score, candidate, centroid_index in ranked:
        if candidate in used or len(assigned[centroid_index]) >= int(caps[centroid_index]):
            continue
        assigned[centroid_index].append((candidate, score))
        used.add(candidate)
    for values in assigned:
        values.sort(key=lambda item: (-item[1], item[0]))
    underfilled = [index for index, values in enumerate(assigned) if len(values) < int(caps[index])]
    return assigned, underfilled


def replace_centroid_contacts(
    current: Sequence[Tuple[int, float]],
    proposals: Sequence[Tuple[int, float]],
    cap: int,
    globally_excluded: Iterable[int],
) -> List[Tuple[int, float]]:
    before = {int(candidate) for candidate, _score in current}
    excluded = {int(value) for value in globally_excluded}
    combined = list(current) + [
        (int(candidate), float(score))
        for candidate, score in proposals
        if int(candidate) not in excluded or int(candidate) in before
    ]
    best: Dict[int, float] = {}
    for candidate, score in combined:
        best[int(candidate)] = max(float(score), best.get(int(candidate), -np.inf))
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))[: int(cap)]


def local_topk(
    query: np.ndarray,
    doc_ids: Sequence[object],
    embeddings: np.ndarray,
    top_k: int = 100,
    *,
    query_is_normalized: bool = False,
    doc_order_keys: Optional[Sequence[int]] = None,
) -> List[Tuple[str, float]]:
    # Keep a stable float32 operation order across peers.
    # Normalizing every document first is mathematically equivalent, but the
    # extra rounding can change which document crosses the top-k boundary when
    # scores are extremely close.
    values = np.asarray(embeddings, dtype=np.float32)
    ids = np.asarray(doc_ids)
    if doc_order_keys is not None:
        keys = np.asarray(doc_order_keys, dtype=np.int64)
        if keys.shape != (len(values),):
            raise ValueError("doc_order_keys must contain one value per embedding")
        # The centralized kernel calls np.unique on the global document
        # indices before its matrix multiplication.  Reproduce that physical
        # row order: although row permutation is mathematically irrelevant,
        # BLAS may choose a different float32 accumulation path for a
        # differently ordered matrix.
        _unique_keys, first_positions = np.unique(keys, return_index=True)
        values = values[first_positions]
        ids = ids[first_positions]
    norms = np.linalg.norm(values, axis=1)
    norms[~np.isfinite(norms) | (norms <= 0.0)] = 1.0
    query_vector = (
        np.asarray(query, dtype=np.float32)
        if query_is_normalized
        else normalize_vector(query)
    )
    scores = (values @ query_vector) / norms
    ranked = sorted(
        ((str(doc_id), float(score)) for doc_id, score in zip(ids, scores)),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[: int(top_k)]


def merge_topk(
    current: Sequence[Tuple[str, float]],
    incoming: Sequence[Tuple[str, float]],
    top_k: int = 100,
) -> List[Tuple[str, float]]:
    best: Dict[str, float] = {}
    for doc_id, score in list(current) + list(incoming):
        best[str(doc_id)] = max(float(score), best.get(str(doc_id), -np.inf))
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))[: int(top_k)]

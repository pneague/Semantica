"""PubMedQA peer-overlay benchmark.

This module reuses the expanded PubMedQA all-MeSH embeddings already present on
scratch and adds only compact peer ownership overlays plus aggregate metrics.
It intentionally does not write per-query rankings or graph artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import multiprocessing as mp
import os
import contextlib
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from retrieval_benchmark import (
    BenchmarkConfig,
    SemanticTree,
    _metric_values,
    config_to_dict,
    normalize_rows,
    normalize_vector,
    stable_int,
)


DEFAULT_PUBMEDQA_DATA_DIR = (
    "/var/scratch/pmneague/Low_Coord_Semantic_DHT/data/"
    "pubmedqa_expanded_all_mesh_alpha0p00007_q30/pubmedqa"
)
DEFAULT_PUBMEDQA_INVENTORY_ALPHA = 5e-4
DEFAULT_PUBMEDQA_EVENTS_PER_PEER = 200
DEFAULT_PUBMEDQA_TRAIN_RECORDS = 133
DEFAULT_PUBMEDQA_TEST_RECORDS = 67
DEFAULT_PUBMEDQA_DATASET_NAME = "pubmedqa_alpha0p0005_events200_train133_test67"
DEFAULT_CACHE_ROOT = (
    "/var/scratch/pmneague/Low_Coord_Semantic_DHT/cache/"
    "pubmedqa_alpha0p0005_events200_train133_test67"
)
DEFAULT_RESULTS_ROOT = (
    "/var/scratch/pmneague/Low_Coord_Semantic_DHT/results/"
    "pubmedqa_alpha0p0005_events200_train133_test67_centroid_ablation_metrics_only"
)
DEFAULT_GRAPH_SOURCES = ("semantica_fixed_k", "random_graph")
DEFAULT_ROUTING_POLICIES = ("chain_only", "chain_plus_direct_topk", "direct_topk_entry")
DEFAULT_ALPHA_SWEEP = (7e-5, 1e-4, 5e-4, 1e-3, 5e-3)
DEFAULT_CENTROID_SPECS = (
    "fixed_1",
    "fixed_2",
    "fixed_3",
    "fixed_4",
    "fixed_5",
    "fixed_6",
    "fixed_7",
    "fixed_8",
    "elbow_knee",
    "elbow_relative_drop",
    "elbow_silhouette",
)
BAD_ARTIFACT_NAMES = (
    "query_hop_metrics.csv",
    "query_hop_metrics.csv.gz",
    "query_rankings.csv",
    "final_graphs.pkl",
    "debug_peer_graph_adjacency.json",
    "hnsw.index",
    "kgraph.index",
)


@contextlib.contextmanager
def _overlay_build_lock(overlay_dir: str, *, stale_seconds: int = 6 * 60 * 60):
    os.makedirs(overlay_dir, exist_ok=True)
    lock_path = os.path.join(overlay_dir, ".build.lock")
    acquired = False
    while not acquired:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()} time={time.time()}\n")
            acquired = True
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
            except OSError:
                age = 0.0
            if age > stale_seconds:
                try:
                    os.remove(lock_path)
                    continue
                except OSError:
                    pass
            time.sleep(5)
    try:
        yield
    finally:
        if acquired:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass


def _open_text(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return open(path, "r", newline="", encoding="utf-8")


def _write_json(path: str, payload: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_manifest(data_dir: str) -> dict:
    path = os.path.join(data_dir, "embeddings", "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return _read_json(path)


def _load_doc_keys(manifest: Mapping[str, object]) -> List[str]:
    return [str(value) for value in np.load(str(manifest["doc_keys_path"]), allow_pickle=False).tolist()]


def _load_query_key_index(manifest: Mapping[str, object]) -> Dict[str, int]:
    keys = np.load(str(manifest["query_keys_path"]), allow_pickle=False)
    return {str(value): int(idx) for idx, value in enumerate(keys.tolist())}


def _load_normalized_doc_embeddings(manifest: Mapping[str, object]) -> np.ndarray:
    embeddings = np.asarray(np.load(str(manifest["doc_embeddings_path"]), mmap_mode="r"), dtype=np.float32)
    return normalize_rows(embeddings)


def _overlay_dir(
    cache_root: str,
    *,
    seed: int,
    peer_count: int,
    interests_per_peer: int,
    docs_per_interest: int,
) -> str:
    docs_per_peer = int(interests_per_peer) * int(docs_per_interest)
    return os.path.join(
        cache_root,
        f"seed_{int(seed)}__peers_{int(peer_count)}"
        f"__interests_{int(interests_per_peer)}__docs_per_interest_{int(docs_per_interest)}"
        f"__docs_per_peer_{docs_per_peer}",
    )


def _overlay_paths(overlay_dir: str) -> Dict[str, str]:
    return {
        "peer_doc_indices": os.path.join(overlay_dir, "peer_doc_indices.npy"),
        "peer_topic_ids": os.path.join(overlay_dir, "peer_topic_ids.npy"),
        "peer_embeddings": os.path.join(overlay_dir, "peer_embeddings.npy"),
        "owner_offsets": os.path.join(overlay_dir, "owner_offsets.npy"),
        "owner_peer_ids": os.path.join(overlay_dir, "owner_peer_ids.npy"),
        "summary": os.path.join(overlay_dir, "peer_inventory_summary.json"),
    }


def _alpha_label(value: float) -> str:
    text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "m")


def _alpha_train_overlay_dir(
    cache_root: str,
    *,
    seed: int,
    peer_count: int,
    docs_per_peer: int,
    events_per_peer: int,
    inventory_alpha: float,
    train_records: int,
    test_records: int,
) -> str:
    return os.path.join(
        cache_root,
        f"seed_{int(seed)}__peers_{int(peer_count)}"
        f"__alpha_{_alpha_label(float(inventory_alpha))}"
        f"__events_{int(events_per_peer)}__train_{int(train_records)}__test_{int(test_records)}"
        f"__docs_per_peer_{int(docs_per_peer)}",
    )


def _alpha_train_overlay_paths(overlay_dir: str) -> Dict[str, str]:
    return {
        "peer_doc_indices": os.path.join(overlay_dir, "peer_doc_indices.npy"),
        "peer_test_doc_indices": os.path.join(overlay_dir, "peer_test_doc_indices.npy"),
        "peer_test_query_indices": os.path.join(overlay_dir, "peer_test_query_indices.npy"),
        "peer_embeddings": os.path.join(overlay_dir, "peer_embeddings.npy"),
        "owner_offsets": os.path.join(overlay_dir, "owner_offsets.npy"),
        "owner_peer_ids": os.path.join(overlay_dir, "owner_peer_ids.npy"),
        "summary": os.path.join(overlay_dir, "peer_inventory_summary.json"),
    }


def _centroid_label(mode: str, *, fixed_k: int, elbow_method: str) -> str:
    mode = str(mode).strip().lower().replace("-", "_")
    elbow_method = str(elbow_method).strip().lower().replace("-", "_")
    if mode == "fixed":
        return f"fixed_{int(fixed_k)}"
    if mode == "elbow":
        return f"elbow_{elbow_method}"
    raise ValueError(f"Unsupported PubMedQA centroid mode: {mode}")


def _centroid_cache_dir(
    overlay_dir: str,
    *,
    mode: str,
    fixed_k: int,
    elbow_max_k: int,
    elbow_method: str,
    relative_drop_threshold: float,
) -> str:
    label = _centroid_label(mode, fixed_k=fixed_k, elbow_method=elbow_method)
    parts = [label]
    if str(mode).strip().lower().replace("-", "_") == "elbow":
        parts.append(f"max_{int(elbow_max_k)}")
        parts.append(f"drop_{_alpha_label(float(relative_drop_threshold))}")
    return os.path.join(overlay_dir, "centroids", "__".join(parts))


def _centroid_cache_paths(cache_dir: str) -> Dict[str, str]:
    return {
        "centroid_embeddings": os.path.join(cache_dir, "centroid_embeddings.npy"),
        "centroid_offsets": os.path.join(cache_dir, "centroid_offsets.npy"),
        "selected_k": os.path.join(cache_dir, "selected_k.npy"),
        "summary": os.path.join(cache_dir, "centroid_summary.json"),
    }


@dataclass
class PubMedQAOverlay:
    data_dir: str
    overlay_dir: str
    seed: int
    peer_doc_indices: np.ndarray
    peer_topic_ids: np.ndarray
    peer_embeddings: np.ndarray
    owner_offsets: np.ndarray
    owner_peer_ids: np.ndarray
    summary: Dict[str, object]
    peer_test_doc_indices: Optional[np.ndarray] = None
    peer_test_query_indices: Optional[np.ndarray] = None

    @property
    def peer_count(self) -> int:
        return int(self.peer_doc_indices.shape[0])

    @property
    def docs_per_peer(self) -> int:
        return int(self.peer_doc_indices.shape[1])

    def owners_for_doc(self, doc_index: int) -> np.ndarray:
        doc_index = int(doc_index)
        start = int(self.owner_offsets[doc_index])
        end = int(self.owner_offsets[doc_index + 1])
        return np.asarray(self.owner_peer_ids[start:end], dtype=np.int32)


@dataclass
class PubMedQAEvents:
    rows: pd.DataFrame
    query_embeddings: np.ndarray

    @property
    def query_count(self) -> int:
        return int(self.rows.shape[0])


@dataclass
class PubMedQACentroids:
    mode: str
    fixed_k: int
    elbow_method: str
    elbow_max_k: int
    relative_drop_threshold: float
    centroid_embeddings: np.ndarray
    centroid_offsets: np.ndarray
    selected_k: np.ndarray
    summary: Dict[str, object]

    @property
    def label(self) -> str:
        return _centroid_label(
            self.mode,
            fixed_k=self.fixed_k,
            elbow_method=self.elbow_method,
        )

    def peer_centroids(self, peer_id: int) -> np.ndarray:
        start = int(self.centroid_offsets[int(peer_id)])
        end = int(self.centroid_offsets[int(peer_id) + 1])
        return np.asarray(self.centroid_embeddings[start:end], dtype=np.float32)

    def score_peers(self, peer_ids: Sequence[int], query_embedding: np.ndarray) -> np.ndarray:
        query = normalize_vector(query_embedding)
        scores = np.full(len(peer_ids), -np.inf, dtype=np.float32)
        for idx, peer_id in enumerate(peer_ids):
            centers = self.peer_centroids(int(peer_id))
            if centers.size:
                scores[idx] = float(np.max(centers @ query))
        return scores


@dataclass
class PubMedQATrainTestSplit:
    train_rows: pd.DataFrame
    test_events: PubMedQAEvents
    summary: Dict[str, object]


@dataclass
class PubMedQATopicData:
    topic_ids: np.ndarray
    pools: Dict[int, np.ndarray]
    cdf: np.ndarray
    doc_topics_by_index: List[np.ndarray]
    summary: Dict[str, object]


@dataclass
class PubMedQAGraph:
    graph_source: str
    adjacency: np.ndarray
    peer_embeddings: np.ndarray
    diagnostics: Dict[str, object]
    centroids: Optional[PubMedQACentroids] = None

    def neighbors(self, peer_id: int, limit: Optional[int] = None) -> np.ndarray:
        values = self.adjacency[int(peer_id)]
        values = values[values >= 0]
        if limit is not None:
            values = values[: max(0, int(limit))]
        return values.astype(np.int32, copy=False)

    def score_peers(self, peer_ids: Sequence[int], query_embedding: np.ndarray) -> np.ndarray:
        if self.centroids is not None:
            return self.centroids.score_peers(peer_ids, query_embedding)
        query = normalize_vector(query_embedding)
        return self.peer_embeddings[np.asarray(peer_ids, dtype=np.int64)] @ query

    def best_neighbor(self, peer_id: int, query_embedding: np.ndarray, excluded: set[int]) -> Optional[int]:
        neighbors = [int(value) for value in self.neighbors(peer_id).tolist() if int(value) not in excluded]
        if not neighbors:
            return None
        scores = self.score_peers(neighbors, query_embedding)
        order = np.lexsort((np.asarray(neighbors, dtype=np.int64), -scores))
        return int(neighbors[int(order[0])])


def load_pubmedqa_overlay(
    *,
    data_dir: str,
    cache_root: str,
    seed: int,
    peer_count: int = 50_000,
    interests_per_peer: int = 10,
    docs_per_interest: int = 20,
) -> PubMedQAOverlay:
    overlay_dir = _overlay_dir(
        cache_root,
        seed=seed,
        peer_count=peer_count,
        interests_per_peer=interests_per_peer,
        docs_per_interest=docs_per_interest,
    )
    paths = _overlay_paths(overlay_dir)
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing PubMedQA overlay artifacts under {overlay_dir}: {missing[:3]}")
    summary = _read_json(paths["summary"])
    return PubMedQAOverlay(
        data_dir=data_dir,
        overlay_dir=overlay_dir,
        seed=int(seed),
        peer_doc_indices=np.load(paths["peer_doc_indices"], mmap_mode="r"),
        peer_topic_ids=np.load(paths["peer_topic_ids"], mmap_mode="r"),
        peer_embeddings=normalize_rows(np.asarray(np.load(paths["peer_embeddings"], mmap_mode="r"), dtype=np.float32)),
        owner_offsets=np.load(paths["owner_offsets"], mmap_mode="r"),
        owner_peer_ids=np.load(paths["owner_peer_ids"], mmap_mode="r"),
        summary=summary,
    )


def _topic_pools_from_doc_topics(
    *,
    doc_topics_path: str,
    doc_id_to_pos: Mapping[str, int],
    min_docs_per_topic: int,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Dict[str, object]]:
    columns = ["topic_id", "doc_id"]
    frame = pd.read_csv(doc_topics_path, usecols=columns)
    frame["doc_id"] = frame["doc_id"].astype(str)
    frame = frame[frame["doc_id"].isin(doc_id_to_pos)].copy()
    frame["doc_index"] = frame["doc_id"].map(doc_id_to_pos).astype(np.int32)
    frame = frame[["topic_id", "doc_index"]].drop_duplicates()
    pools: Dict[int, np.ndarray] = {}
    topic_sizes = []
    for topic_id, group in frame.groupby("topic_id", sort=True):
        values = np.asarray(sorted(group["doc_index"].astype(int).tolist()), dtype=np.int32)
        if values.size >= int(min_docs_per_topic):
            pools[int(topic_id)] = values
        topic_sizes.append(int(values.size))
    eligible_topic_ids = np.asarray(sorted(pools), dtype=np.int32)
    sizes = np.asarray(topic_sizes, dtype=np.int64)
    summary = {
        "topic_count": int(len(topic_sizes)),
        "eligible_topic_count": int(len(eligible_topic_ids)),
        "min_docs_per_topic": int(min_docs_per_topic),
        "median_topic_docs": float(np.median(sizes)) if sizes.size else 0.0,
        "max_topic_docs": int(np.max(sizes)) if sizes.size else 0,
    }
    if eligible_topic_ids.size < 1:
        raise RuntimeError("No PubMedQA MeSH topics satisfy the minimum docs-per-topic threshold.")
    return eligible_topic_ids, pools, summary


def _topic_data_from_doc_topics(
    *,
    doc_topics_path: str,
    doc_id_to_pos: Mapping[str, int],
    min_docs_per_topic: int = 1,
    allowed_doc_indices: Optional[set[int]] = None,
) -> PubMedQATopicData:
    frame = pd.read_csv(doc_topics_path, usecols=["topic_id", "doc_id"])
    frame["doc_id"] = frame["doc_id"].astype(str)
    frame = frame[frame["doc_id"].isin(doc_id_to_pos)].copy()
    frame["doc_index"] = frame["doc_id"].map(doc_id_to_pos).astype(np.int32)
    if allowed_doc_indices is not None:
        frame = frame[frame["doc_index"].isin(allowed_doc_indices)].copy()
    frame = frame[["topic_id", "doc_index"]].drop_duplicates()

    doc_topics_by_index: List[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(len(doc_id_to_pos))]
    for doc_index, group in frame.groupby("doc_index", sort=False):
        doc_topics_by_index[int(doc_index)] = np.asarray(group["topic_id"].astype(int).tolist(), dtype=np.int32)

    pools: Dict[int, np.ndarray] = {}
    topic_sizes = []
    for topic_id, group in frame.groupby("topic_id", sort=True):
        values = np.asarray(sorted(group["doc_index"].astype(int).tolist()), dtype=np.int32)
        topic_sizes.append(int(values.size))
        if values.size >= int(min_docs_per_topic):
            pools[int(topic_id)] = values
    topic_ids = np.asarray(sorted(pools), dtype=np.int32)
    if topic_ids.size < 1:
        raise RuntimeError("No PubMedQA MeSH topics available for alpha overlay generation.")
    weights = np.asarray([pools[int(topic_id)].size for topic_id in topic_ids.tolist()], dtype=np.float64)
    cdf = np.cumsum(weights / weights.sum())
    cdf[-1] = 1.0
    sizes = np.asarray(topic_sizes, dtype=np.int64)
    summary = {
        "topic_count": int(len(topic_sizes)),
        "eligible_topic_count": int(len(topic_ids)),
        "min_docs_per_topic": int(min_docs_per_topic),
        "allowed_doc_count": int(len(allowed_doc_indices)) if allowed_doc_indices is not None else "",
        "topic_doc_assignment_count": int(frame.shape[0]),
        "median_topic_docs": float(np.median(sizes)) if sizes.size else 0.0,
        "max_topic_docs": int(np.max(sizes)) if sizes.size else 0,
    }
    return PubMedQATopicData(
        topic_ids=topic_ids,
        pools=pools,
        cdf=cdf,
        doc_topics_by_index=doc_topics_by_index,
        summary=summary,
    )


def _draw_global_topic(topic_data: PubMedQATopicData, rng: np.random.Generator) -> int:
    position = int(np.searchsorted(topic_data.cdf, rng.random(), side="right"))
    return int(topic_data.topic_ids[min(position, len(topic_data.topic_ids) - 1)])


def _draw_reinforced_topic(
    topic_counts: Counter,
    *,
    topic_data: PubMedQATopicData,
    inventory_alpha: float,
    rng: np.random.Generator,
) -> int:
    total_existing = int(sum(topic_counts.values()))
    total_alpha = float(len(topic_data.topic_ids)) * float(inventory_alpha)
    if total_existing <= 0 or rng.random() < total_alpha / (total_existing + total_alpha):
        topic_id = _draw_global_topic(topic_data, rng)
    else:
        keys = np.asarray(list(topic_counts.keys()), dtype=np.int32)
        weights = np.asarray([topic_counts[int(key)] for key in keys.tolist()], dtype=np.float64)
        cdf = np.cumsum(weights / weights.sum())
        position = int(np.searchsorted(cdf, rng.random(), side="right"))
        topic_id = int(keys[min(position, len(keys) - 1)])
    topic_counts[int(topic_id)] += 1
    return int(topic_id)


def _sample_one_topic_doc(
    *,
    topic_id: int,
    topic_data: PubMedQATopicData,
    used_docs: set[int],
    forbidden_docs: set[int],
    rng: np.random.Generator,
) -> Optional[int]:
    pool = topic_data.pools.get(int(topic_id))
    if pool is None or pool.size == 0:
        return None
    for _attempt in range(12):
        doc_index = int(pool[int(rng.integers(0, int(pool.size)))])
        if doc_index not in used_docs and doc_index not in forbidden_docs:
            return doc_index
    available = [int(value) for value in pool.tolist() if int(value) not in used_docs and int(value) not in forbidden_docs]
    if not available:
        return None
    return int(available[int(rng.integers(0, len(available)))])


def _sample_any_doc(
    *,
    doc_count: int,
    used_docs: set[int],
    forbidden_docs: set[int],
    rng: np.random.Generator,
) -> int:
    for _attempt in range(128):
        doc_index = int(rng.integers(0, int(doc_count)))
        if doc_index not in used_docs and doc_index not in forbidden_docs:
            return doc_index
    for doc_index in range(int(doc_count)):
        if doc_index not in used_docs and doc_index not in forbidden_docs:
            return int(doc_index)
    raise RuntimeError("No available document remains for PubMedQA alpha inventory sampling.")


def _query_options_by_doc_index(
    *,
    data_dir: str,
    manifest: Mapping[str, object],
    doc_id_to_pos: Mapping[str, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    path = os.path.join(data_dir, "query_topics.csv.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    query_key_to_pos = _load_query_key_index(manifest)
    frame = pd.read_csv(path, usecols=["query", "target_doc_id"])
    frame["query"] = frame["query"].astype(str)
    frame["target_doc_id"] = frame["target_doc_id"].astype(str)
    frame = frame[frame["query"].isin(query_key_to_pos) & frame["target_doc_id"].isin(doc_id_to_pos)].copy()
    if frame.empty:
        raise RuntimeError("No PubMedQA query-topic rows map to existing query and document embeddings.")
    frame["doc_index"] = frame["target_doc_id"].map(doc_id_to_pos).astype(np.int32)
    frame["query_index"] = frame["query"].map(query_key_to_pos).astype(np.int32)
    frame = frame[["doc_index", "query_index"]].drop_duplicates().sort_values(["doc_index", "query_index"])
    counts = np.bincount(frame["doc_index"].to_numpy(dtype=np.int64), minlength=len(doc_id_to_pos)).astype(np.int64)
    offsets = np.zeros(len(doc_id_to_pos) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    query_indices = frame["query_index"].to_numpy(dtype=np.int32)
    nonzero = counts[counts > 0]
    summary = {
        "query_topic_rows": int(frame.shape[0]),
        "docs_with_available_queries": int(nonzero.shape[0]),
        "mean_queries_per_queryable_doc": float(np.mean(nonzero)) if nonzero.size else 0.0,
        "median_queries_per_queryable_doc": float(np.median(nonzero)) if nonzero.size else 0.0,
        "max_queries_per_queryable_doc": int(np.max(nonzero)) if nonzero.size else 0,
    }
    return offsets, query_indices, summary


def _sample_query_index_for_doc(
    *,
    doc_index: int,
    query_offsets: np.ndarray,
    query_indices: np.ndarray,
    rng: np.random.Generator,
) -> int:
    start = int(query_offsets[int(doc_index)])
    end = int(query_offsets[int(doc_index) + 1])
    if end <= start:
        raise RuntimeError(f"Sampled PubMedQA doc index {doc_index} has no available query.")
    return int(query_indices[int(rng.integers(start, end))])


def _select_k_by_knee(inertias: Sequence[float]) -> int:
    if not inertias:
        return 1
    if len(inertias) <= 2:
        return len(inertias)
    points = np.column_stack((np.arange(1, len(inertias) + 1, dtype=np.float64), np.asarray(inertias, dtype=np.float64)))
    start = points[0]
    end = points[-1]
    line = end - start
    norm = float(np.linalg.norm(line))
    if norm <= 1e-12:
        return 1
    distances = np.abs(np.cross(line, points - start) / norm)
    return int(np.argmax(distances) + 1)


def _select_k_by_relative_drop(inertias: Sequence[float], threshold: float) -> int:
    if not inertias:
        return 1
    previous = float(inertias[0])
    for index in range(1, len(inertias)):
        current = float(inertias[index])
        if previous <= 1e-12:
            return int(index)
        improvement = (previous - current) / previous
        if improvement < float(threshold):
            return max(1, int(index))
        previous = current
    return int(len(inertias))


def _select_k_by_silhouette(embeddings: np.ndarray, max_k: int, seed: int) -> int:
    max_k = max(1, min(int(max_k), int(embeddings.shape[0]) - 1))
    if max_k < 2:
        return 1
    best_k = 2
    best_score = -np.inf
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=int(k), random_state=int(seed), n_init=3)
        labels = model.fit_predict(embeddings)
        if len(set(labels.tolist())) < 2:
            continue
        score = float(silhouette_score(embeddings, labels, metric="cosine"))
        if score > best_score:
            best_score = score
            best_k = int(k)
    return int(best_k)


def _select_peer_centroid_count(
    embeddings: np.ndarray,
    *,
    mode: str,
    fixed_k: int,
    elbow_max_k: int,
    elbow_method: str,
    relative_drop_threshold: float,
    seed: int,
) -> int:
    max_possible = max(1, int(embeddings.shape[0]))
    mode = str(mode).strip().lower().replace("-", "_")
    elbow_method = str(elbow_method).strip().lower().replace("-", "_")
    if mode == "fixed":
        return max(1, min(int(fixed_k), max_possible))
    if mode != "elbow":
        raise ValueError(f"Unsupported PubMedQA centroid mode: {mode}")
    max_k = max(1, min(int(elbow_max_k), max_possible))
    if max_k <= 1:
        return 1
    if elbow_method == "silhouette":
        return _select_k_by_silhouette(embeddings, max_k, int(seed))
    inertias: List[float] = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=int(k), random_state=int(seed), n_init=3)
        model.fit(embeddings)
        inertias.append(float(model.inertia_))
    if elbow_method == "knee":
        return _select_k_by_knee(inertias)
    if elbow_method == "relative_drop":
        return _select_k_by_relative_drop(inertias, float(relative_drop_threshold))
    raise ValueError(f"Unsupported PubMedQA centroid elbow method: {elbow_method}")


def _fit_centers(embeddings: np.ndarray, k: int, seed: int) -> np.ndarray:
    k = max(1, min(int(k), int(embeddings.shape[0])))
    if k <= 1:
        return normalize_rows(np.mean(embeddings, axis=0, keepdims=True)).astype(np.float32, copy=False)
    model = KMeans(n_clusters=int(k), random_state=int(seed), n_init=3)
    model.fit(embeddings)
    return normalize_rows(np.asarray(model.cluster_centers_, dtype=np.float32)).astype(np.float32, copy=False)


def _draw_unique_topics(
    rng: np.random.Generator,
    topic_ids: np.ndarray,
    cdf: np.ndarray,
    count: int,
) -> List[int]:
    selected: List[int] = []
    selected_set: set[int] = set()
    while len(selected) < int(count):
        draw_count = max((int(count) - len(selected)) * 3, int(count))
        random_values = rng.random(draw_count)
        positions = np.searchsorted(cdf, random_values, side="right")
        for pos in positions.tolist():
            topic_id = int(topic_ids[min(int(pos), len(topic_ids) - 1)])
            if topic_id in selected_set:
                continue
            selected.append(topic_id)
            selected_set.add(topic_id)
            if len(selected) >= int(count):
                break
    return selected


def _sample_topic_docs(
    *,
    pool: np.ndarray,
    used_docs: set[int],
    count: int,
    rng: np.random.Generator,
) -> Optional[List[int]]:
    selected: List[int] = []
    selected_set: set[int] = set()
    for _attempt in range(8):
        draw_size = min(int(pool.size), max((int(count) - len(selected)) * 4, int(count)))
        if draw_size <= 0:
            break
        candidates = rng.choice(pool, size=draw_size, replace=False)
        for value in candidates.tolist():
            doc_index = int(value)
            if doc_index in used_docs or doc_index in selected_set:
                continue
            selected.append(doc_index)
            selected_set.add(doc_index)
            if len(selected) >= int(count):
                return selected
    available = [int(value) for value in pool.tolist() if int(value) not in used_docs and int(value) not in selected_set]
    if len(available) + len(selected) < int(count):
        return None
    needed = int(count) - len(selected)
    if needed:
        extra = rng.choice(np.asarray(available, dtype=np.int32), size=needed, replace=False)
        selected.extend(int(value) for value in extra.tolist())
    return selected


def generate_pubmedqa_overlay(
    *,
    data_dir: str = DEFAULT_PUBMEDQA_DATA_DIR,
    cache_root: str = DEFAULT_CACHE_ROOT,
    seed: int = 42,
    peer_count: int = 50_000,
    interests_per_peer: int = 10,
    docs_per_interest: int = 20,
    embedding_batch_size: int = 100,
    force: bool = False,
) -> PubMedQAOverlay:
    manifest = _load_manifest(data_dir)
    doc_ids = _load_doc_keys(manifest)
    doc_id_to_pos = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    overlay_dir = _overlay_dir(
        cache_root,
        seed=seed,
        peer_count=peer_count,
        interests_per_peer=interests_per_peer,
        docs_per_interest=docs_per_interest,
    )
    paths = _overlay_paths(overlay_dir)
    if not force and all(os.path.exists(path) for path in paths.values()):
        return load_pubmedqa_overlay(
            data_dir=data_dir,
            cache_root=cache_root,
            seed=seed,
            peer_count=peer_count,
            interests_per_peer=interests_per_peer,
            docs_per_interest=docs_per_interest,
        )

    with _overlay_build_lock(overlay_dir):
        if not force and all(os.path.exists(path) for path in paths.values()):
            return load_pubmedqa_overlay(
                data_dir=data_dir,
                cache_root=cache_root,
                seed=seed,
                peer_count=peer_count,
                interests_per_peer=interests_per_peer,
                docs_per_interest=docs_per_interest,
            )
        return _generate_pubmedqa_overlay_unlocked(
            data_dir=data_dir,
            cache_root=cache_root,
            seed=seed,
            peer_count=peer_count,
            interests_per_peer=interests_per_peer,
            docs_per_interest=docs_per_interest,
            embedding_batch_size=embedding_batch_size,
            manifest=manifest,
            doc_ids=doc_ids,
            doc_id_to_pos=doc_id_to_pos,
            overlay_dir=overlay_dir,
            paths=paths,
        )


def _generate_pubmedqa_overlay_unlocked(
    *,
    data_dir: str,
    cache_root: str,
    seed: int,
    peer_count: int,
    interests_per_peer: int,
    docs_per_interest: int,
    embedding_batch_size: int,
    manifest: Mapping[str, object],
    doc_ids: Sequence[str],
    doc_id_to_pos: Mapping[str, int],
    overlay_dir: str,
    paths: Mapping[str, str],
) -> PubMedQAOverlay:
    start = time.perf_counter()
    topic_ids, pools, topic_summary = _topic_pools_from_doc_topics(
        doc_topics_path=os.path.join(data_dir, "doc_topics.csv.gz"),
        doc_id_to_pos=doc_id_to_pos,
        min_docs_per_topic=docs_per_interest,
    )
    weights = np.asarray([pools[int(topic_id)].size for topic_id in topic_ids.tolist()], dtype=np.float64)
    cdf = np.cumsum(weights / weights.sum())
    cdf[-1] = 1.0

    rng = np.random.default_rng(int(seed))
    docs_per_peer = int(interests_per_peer) * int(docs_per_interest)
    peer_doc_indices = np.empty((int(peer_count), docs_per_peer), dtype=np.int32)
    peer_topic_ids = np.empty((int(peer_count), int(interests_per_peer)), dtype=np.int32)
    replacement_topic_draws = 0

    for peer_id in range(int(peer_count)):
        topics = _draw_unique_topics(rng, topic_ids, cdf, int(interests_per_peer))
        used_docs: set[int] = set()
        row_docs: List[int] = []
        for topic_pos in range(int(interests_per_peer)):
            topic_id = int(topics[topic_pos])
            selected = _sample_topic_docs(
                pool=pools[topic_id],
                used_docs=used_docs,
                count=int(docs_per_interest),
                rng=rng,
            )
            while selected is None:
                replacement_topic_draws += 1
                replacement = int(topic_ids[int(np.searchsorted(cdf, rng.random(), side="right"))])
                if replacement in topics:
                    continue
                selected = _sample_topic_docs(
                    pool=pools[replacement],
                    used_docs=used_docs,
                    count=int(docs_per_interest),
                    rng=rng,
                )
                if selected is not None:
                    topics[topic_pos] = replacement
                    topic_id = replacement
            used_docs.update(selected)
            row_docs.extend(selected)
        if len(row_docs) != docs_per_peer or len(set(row_docs)) != docs_per_peer:
            raise RuntimeError(f"Peer {peer_id} did not receive {docs_per_peer} unique docs.")
        peer_doc_indices[peer_id] = np.asarray(row_docs, dtype=np.int32)
        peer_topic_ids[peer_id] = np.asarray(topics, dtype=np.int32)

    doc_embeddings = np.asarray(np.load(str(manifest["doc_embeddings_path"]), mmap_mode="r"), dtype=np.float32)
    peer_embeddings = np.empty((int(peer_count), int(doc_embeddings.shape[1])), dtype=np.float16)
    for start_row in range(0, int(peer_count), int(embedding_batch_size)):
        end_row = min(start_row + int(embedding_batch_size), int(peer_count))
        doc_batch = peer_doc_indices[start_row:end_row]
        values = np.asarray(doc_embeddings[doc_batch.reshape(-1)], dtype=np.float32).reshape(
            end_row - start_row,
            docs_per_peer,
            -1,
        )
        peer_embeddings[start_row:end_row] = normalize_rows(values.mean(axis=1)).astype(np.float16)

    flat_docs = peer_doc_indices.reshape(-1)
    counts = np.bincount(flat_docs.astype(np.int64), minlength=len(doc_ids)).astype(np.int64)
    owner_offsets = np.zeros(len(doc_ids) + 1, dtype=np.int64)
    owner_offsets[1:] = np.cumsum(counts)
    flat_peers = np.repeat(np.arange(int(peer_count), dtype=np.int32), docs_per_peer)
    order = np.argsort(flat_docs, kind="mergesort")
    owner_peer_ids = flat_peers[order].astype(np.int32, copy=False)

    np.save(paths["peer_doc_indices"], peer_doc_indices)
    np.save(paths["peer_topic_ids"], peer_topic_ids)
    np.save(paths["peer_embeddings"], peer_embeddings)
    np.save(paths["owner_offsets"], owner_offsets)
    np.save(paths["owner_peer_ids"], owner_peer_ids)

    topic_counts_per_peer = np.asarray([len(set(row.tolist())) for row in peer_topic_ids], dtype=np.int32)
    doc_counts_per_peer = np.asarray([len(set(row.tolist())) for row in peer_doc_indices], dtype=np.int32)
    summary: Dict[str, object] = {
        "data_dir": data_dir,
        "overlay_dir": overlay_dir,
        "seed": int(seed),
        "peer_count": int(peer_count),
        "interests_per_peer": int(interests_per_peer),
        "docs_per_interest": int(docs_per_interest),
        "docs_per_peer": int(docs_per_peer),
        "doc_count": int(len(doc_ids)),
        "owned_doc_assignment_count": int(peer_doc_indices.size),
        "unique_owned_doc_count": int(np.count_nonzero(counts)),
        "min_docs_per_peer": int(doc_counts_per_peer.min()),
        "max_docs_per_peer": int(doc_counts_per_peer.max()),
        "min_interests_per_peer": int(topic_counts_per_peer.min()),
        "max_interests_per_peer": int(topic_counts_per_peer.max()),
        "replacement_topic_draws": int(replacement_topic_draws),
        "build_runtime_seconds": float(time.perf_counter() - start),
        "topic_pool_summary": topic_summary,
        "paths": paths,
    }
    _write_json(paths["summary"], summary)
    return load_pubmedqa_overlay(
        data_dir=data_dir,
        cache_root=cache_root,
        seed=seed,
        peer_count=peer_count,
        interests_per_peer=interests_per_peer,
        docs_per_interest=docs_per_interest,
    )


def _prepare_pubmedqa_event_rows(
    *,
    data_dir: str,
    manifest: Mapping[str, object],
    doc_ids: Sequence[str],
) -> pd.DataFrame:
    metadata_path = os.path.join(data_dir, "metadata_seeded_randomized.csv.gz")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(metadata_path)
    usecols = ["qid", "user_id", "query", "target_doc_id"]
    events = pd.read_csv(metadata_path, usecols=usecols).reset_index(names="event_index")
    doc_id_to_pos = {str(doc_id): idx for idx, doc_id in enumerate(doc_ids)}
    query_key_to_pos = _load_query_key_index(manifest)
    events["query"] = events["query"].astype(str)
    events["target_doc_id"] = events["target_doc_id"].astype(str)
    events = events[events["query"].isin(query_key_to_pos) & events["target_doc_id"].isin(doc_id_to_pos)].copy()
    events["query_id"] = events["qid"].astype(str)
    events["query_index"] = events["query"].map(query_key_to_pos).astype(np.int64)
    events["target_doc_index"] = events["target_doc_id"].map(doc_id_to_pos).astype(np.int64)
    events["source_peer_id"] = events["user_id"].astype(np.int64)
    keep_cols = [
        "event_index",
        "query_id",
        "qid",
        "user_id",
        "source_peer_id",
        "query",
        "query_index",
        "target_doc_id",
        "target_doc_index",
    ]
    return events[keep_cols].sort_values("event_index").reset_index(drop=True)


def _events_from_rows(rows: pd.DataFrame, manifest: Mapping[str, object]) -> PubMedQAEvents:
    query_embeddings_mmap = np.load(str(manifest["query_embeddings_path"]), mmap_mode="r")
    query_embeddings = normalize_rows(
        np.asarray(query_embeddings_mmap[rows["query_index"].to_numpy(dtype=np.int64)], dtype=np.float32)
    )
    keep_cols = [
        "event_index",
        "query_id",
        "qid",
        "source_peer_id",
        "query",
        "query_index",
        "target_doc_id",
        "target_doc_index",
    ]
    return PubMedQAEvents(rows=rows[keep_cols].reset_index(drop=True), query_embeddings=query_embeddings)


def load_pubmedqa_train_test_split(
    *,
    data_dir: str,
    manifest: Mapping[str, object],
    doc_ids: Sequence[str],
    train_records: int = DEFAULT_PUBMEDQA_TRAIN_RECORDS,
    test_records: int = DEFAULT_PUBMEDQA_TEST_RECORDS,
    sample_queries: Optional[int],
    seed: int,
) -> PubMedQATrainTestSplit:
    if int(train_records) <= 0 or int(test_records) <= 0:
        raise ValueError("PubMedQA train_records and test_records must both be positive.")
    events = _prepare_pubmedqa_event_rows(data_dir=data_dir, manifest=manifest, doc_ids=doc_ids)
    min_records = int(train_records) + int(test_records)
    train_parts = []
    test_parts = []
    eligible_users = 0
    for user_id, group in events.groupby("user_id", sort=False):
        if len(group) < min_records:
            continue
        eligible_users += 1
        rng = np.random.default_rng(stable_int(f"pubmedqa-split:{seed}:{user_id}"))
        positions = np.arange(len(group), dtype=np.int64)
        test_positions = set(rng.choice(positions, size=int(test_records), replace=False).tolist())
        remaining = [int(pos) for pos in positions.tolist() if int(pos) not in test_positions]
        if len(remaining) > int(train_records):
            remaining = sorted(rng.choice(np.asarray(remaining, dtype=np.int64), size=int(train_records), replace=False))
        train_parts.append(group.iloc[remaining])
        test_parts.append(group.iloc[sorted(test_positions)])
    if not train_parts or not test_parts:
        raise RuntimeError(
            f"No PubMedQA users have at least {min_records} embedded events for train/test splitting."
        )
    train_rows = pd.concat(train_parts, ignore_index=True).sort_values("event_index").reset_index(drop=True)
    test_rows = pd.concat(test_parts, ignore_index=True).sort_values("event_index").reset_index(drop=True)
    if sample_queries is not None and int(sample_queries) > 0 and int(sample_queries) < len(test_rows):
        test_rows = test_rows.sample(n=int(sample_queries), random_state=int(seed)).sort_values("event_index")
    test_rows = test_rows.reset_index(drop=True)
    summary = {
        "split_strategy": "aol_style_random_test_per_user_train_from_remaining",
        "seed": int(seed),
        "train_records_per_user": int(train_records),
        "test_records_per_user": int(test_records),
        "eligible_user_count": int(eligible_users),
        "train_event_count": int(train_rows.shape[0]),
        "test_event_count_before_query_sampling": int(sum(len(part) for part in test_parts)),
        "test_event_count": int(test_rows.shape[0]),
        "sample_queries": sample_queries,
    }
    return PubMedQATrainTestSplit(
        train_rows=train_rows,
        test_events=_events_from_rows(test_rows, manifest),
        summary=summary,
    )


def load_pubmedqa_events(
    *,
    data_dir: str,
    manifest: Mapping[str, object],
    doc_ids: Sequence[str],
    sample_queries: Optional[int],
    seed: int,
) -> PubMedQAEvents:
    events = _prepare_pubmedqa_event_rows(data_dir=data_dir, manifest=manifest, doc_ids=doc_ids)
    if sample_queries is not None and int(sample_queries) > 0 and int(sample_queries) < len(events):
        events = events.sample(n=int(sample_queries), random_state=int(seed)).sort_values("event_index")
    return _events_from_rows(events.reset_index(drop=True), manifest)


def load_pubmedqa_alpha_train_overlay(
    *,
    data_dir: str,
    cache_root: str,
    seed: int,
    peer_count: int = 50_000,
    docs_per_peer: int = DEFAULT_PUBMEDQA_TRAIN_RECORDS,
    events_per_peer: int = DEFAULT_PUBMEDQA_EVENTS_PER_PEER,
    inventory_alpha: float = DEFAULT_PUBMEDQA_INVENTORY_ALPHA,
    train_records: int = DEFAULT_PUBMEDQA_TRAIN_RECORDS,
    test_records: int = DEFAULT_PUBMEDQA_TEST_RECORDS,
) -> PubMedQAOverlay:
    if int(train_records) + int(test_records) != int(events_per_peer):
        raise ValueError("train_records + test_records must equal events_per_peer.")
    overlay_dir = _alpha_train_overlay_dir(
        cache_root,
        seed=seed,
        peer_count=peer_count,
        docs_per_peer=docs_per_peer,
        events_per_peer=events_per_peer,
        inventory_alpha=inventory_alpha,
        train_records=train_records,
        test_records=test_records,
    )
    paths = _alpha_train_overlay_paths(overlay_dir)
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing PubMedQA alpha-train overlay artifacts under {overlay_dir}: {missing[:3]}")
    summary = _read_json(paths["summary"])
    peer_doc_indices = np.load(paths["peer_doc_indices"], mmap_mode="r")
    return PubMedQAOverlay(
        data_dir=data_dir,
        overlay_dir=overlay_dir,
        seed=int(seed),
        peer_doc_indices=peer_doc_indices,
        peer_topic_ids=np.empty((int(peer_doc_indices.shape[0]), 0), dtype=np.int32),
        peer_embeddings=normalize_rows(np.asarray(np.load(paths["peer_embeddings"], mmap_mode="r"), dtype=np.float32)),
        owner_offsets=np.load(paths["owner_offsets"], mmap_mode="r"),
        owner_peer_ids=np.load(paths["owner_peer_ids"], mmap_mode="r"),
        summary=summary,
        peer_test_doc_indices=np.load(paths["peer_test_doc_indices"], mmap_mode="r"),
        peer_test_query_indices=np.load(paths["peer_test_query_indices"], mmap_mode="r"),
    )


def generate_pubmedqa_alpha_train_overlay(
    *,
    data_dir: str = DEFAULT_PUBMEDQA_DATA_DIR,
    cache_root: str = DEFAULT_CACHE_ROOT,
    seed: int = 42,
    peer_count: int = 50_000,
    docs_per_peer: int = DEFAULT_PUBMEDQA_TRAIN_RECORDS,
    events_per_peer: int = DEFAULT_PUBMEDQA_EVENTS_PER_PEER,
    inventory_alpha: float = DEFAULT_PUBMEDQA_INVENTORY_ALPHA,
    train_records: int = DEFAULT_PUBMEDQA_TRAIN_RECORDS,
    test_records: int = DEFAULT_PUBMEDQA_TEST_RECORDS,
    embedding_batch_size: int = 100,
    force: bool = False,
) -> PubMedQAOverlay:
    if int(train_records) + int(test_records) != int(events_per_peer):
        raise ValueError("train_records + test_records must equal events_per_peer.")
    if int(docs_per_peer) != int(train_records):
        raise ValueError("For alpha_train PubMedQA, docs_per_peer must equal train_records.")
    manifest = _load_manifest(data_dir)
    doc_ids = _load_doc_keys(manifest)
    overlay_dir = _alpha_train_overlay_dir(
        cache_root,
        seed=seed,
        peer_count=peer_count,
        docs_per_peer=docs_per_peer,
        events_per_peer=events_per_peer,
        inventory_alpha=inventory_alpha,
        train_records=train_records,
        test_records=test_records,
    )
    paths = _alpha_train_overlay_paths(overlay_dir)
    if not force and all(os.path.exists(path) for path in paths.values()):
        return load_pubmedqa_alpha_train_overlay(
            data_dir=data_dir,
            cache_root=cache_root,
            seed=seed,
            peer_count=peer_count,
            docs_per_peer=docs_per_peer,
            events_per_peer=events_per_peer,
            inventory_alpha=inventory_alpha,
            train_records=train_records,
            test_records=test_records,
        )
    with _overlay_build_lock(overlay_dir):
        if not force and all(os.path.exists(path) for path in paths.values()):
            return load_pubmedqa_alpha_train_overlay(
                data_dir=data_dir,
                cache_root=cache_root,
                seed=seed,
                peer_count=peer_count,
                docs_per_peer=docs_per_peer,
                events_per_peer=events_per_peer,
                inventory_alpha=inventory_alpha,
                train_records=train_records,
                test_records=test_records,
            )
        return _generate_pubmedqa_alpha_train_overlay_unlocked(
            data_dir=data_dir,
            cache_root=cache_root,
            seed=seed,
            peer_count=peer_count,
            docs_per_peer=docs_per_peer,
            events_per_peer=events_per_peer,
            inventory_alpha=inventory_alpha,
            train_records=train_records,
            test_records=test_records,
            embedding_batch_size=embedding_batch_size,
            manifest=manifest,
            doc_ids=doc_ids,
            overlay_dir=overlay_dir,
            paths=paths,
        )


def _generate_pubmedqa_alpha_train_overlay_unlocked(
    *,
    data_dir: str,
    cache_root: str,
    seed: int,
    peer_count: int,
    docs_per_peer: int,
    events_per_peer: int,
    inventory_alpha: float,
    train_records: int,
    test_records: int,
    embedding_batch_size: int,
    manifest: Mapping[str, object],
    doc_ids: Sequence[str],
    overlay_dir: str,
    paths: Mapping[str, str],
) -> PubMedQAOverlay:
    start = time.perf_counter()
    doc_id_to_pos = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    query_offsets, query_indices, query_pool_summary = _query_options_by_doc_index(
        data_dir=data_dir,
        manifest=manifest,
        doc_id_to_pos=doc_id_to_pos,
    )
    queryable_doc_indices = {int(value) for value in np.flatnonzero(np.diff(query_offsets) > 0).tolist()}
    topic_data = _topic_data_from_doc_topics(
        doc_topics_path=os.path.join(data_dir, "doc_topics.csv.gz"),
        doc_id_to_pos=doc_id_to_pos,
        min_docs_per_topic=1,
        allowed_doc_indices=queryable_doc_indices,
    )

    peer_doc_indices = np.empty((int(peer_count), int(docs_per_peer)), dtype=np.int32)
    peer_test_doc_indices = np.empty((int(peer_count), int(test_records)), dtype=np.int32)
    peer_test_query_indices = np.empty((int(peer_count), int(test_records)), dtype=np.int32)
    nonempty_topic_counts = np.zeros(int(peer_count), dtype=np.int16)
    training_topic_counts = np.zeros(int(peer_count), dtype=np.int16)
    fallback_doc_draws = 0
    global_topic_draws = 0
    query_draws = 0

    for peer_id in range(int(peer_count)):
        rng = np.random.default_rng(stable_int(f"pubmedqa-alpha-inventory:{seed}:{peer_id}"))
        used_docs: set[int] = set()
        event_docs: List[int] = []
        event_queries: List[int] = []
        event_topics: List[int] = []
        topic_counts: Counter = Counter()

        while len(event_docs) < int(events_per_peer):
            topic_id = _draw_reinforced_topic(
                topic_counts,
                topic_data=topic_data,
                inventory_alpha=inventory_alpha,
                rng=rng,
            )
            accepted_topic_id = int(topic_id)
            if topic_counts[int(topic_id)] == 1:
                global_topic_draws += 1
            doc_index = _sample_one_topic_doc(
                topic_id=topic_id,
                topic_data=topic_data,
                used_docs=used_docs,
                forbidden_docs=set(),
                rng=rng,
            )
            if doc_index is None:
                fallback_doc_draws += 1
                for _attempt in range(16):
                    replacement_topic = _draw_global_topic(topic_data, rng)
                    topic_counts[int(replacement_topic)] += 1
                    doc_index = _sample_one_topic_doc(
                        topic_id=replacement_topic,
                        topic_data=topic_data,
                        used_docs=used_docs,
                        forbidden_docs=set(),
                        rng=rng,
                    )
                    if doc_index is not None:
                        accepted_topic_id = int(replacement_topic)
                        break
            if doc_index is None:
                doc_index = _sample_any_doc(
                    doc_count=len(doc_ids),
                    used_docs=used_docs,
                    forbidden_docs=set(range(len(doc_ids))).difference(queryable_doc_indices),
                    rng=rng,
                )
                doc_topics = topic_data.doc_topics_by_index[int(doc_index)]
                accepted_topic_id = int(np.min(doc_topics)) if doc_topics.size else -1
            used_docs.add(int(doc_index))
            event_docs.append(int(doc_index))
            event_topics.append(int(accepted_topic_id))
            event_queries.append(
                _sample_query_index_for_doc(
                    doc_index=int(doc_index),
                    query_offsets=query_offsets,
                    query_indices=query_indices,
                    rng=rng,
                )
            )
            query_draws += 1

        if len(event_docs) != int(events_per_peer) or len(set(event_docs)) != int(events_per_peer):
            raise RuntimeError(f"Peer {peer_id} did not receive {events_per_peer} unique events/docs.")
        order = rng.permutation(int(events_per_peer))
        train_positions = order[: int(train_records)]
        test_positions = order[int(train_records) :]
        peer_doc_indices[peer_id] = np.asarray([event_docs[int(pos)] for pos in train_positions], dtype=np.int32)
        peer_test_doc_indices[peer_id] = np.asarray([event_docs[int(pos)] for pos in test_positions], dtype=np.int32)
        peer_test_query_indices[peer_id] = np.asarray([event_queries[int(pos)] for pos in test_positions], dtype=np.int32)
        nonempty_topic_counts[peer_id] = int(len(topic_counts))
        training_topic_counts[peer_id] = int(
            len({event_topics[int(pos)] for pos in train_positions if event_topics[int(pos)] >= 0})
        )

    doc_embeddings = np.asarray(np.load(str(manifest["doc_embeddings_path"]), mmap_mode="r"), dtype=np.float32)
    peer_embeddings = np.empty((int(peer_count), int(doc_embeddings.shape[1])), dtype=np.float16)
    for start_row in range(0, int(peer_count), int(embedding_batch_size)):
        end_row = min(start_row + int(embedding_batch_size), int(peer_count))
        doc_batch = peer_doc_indices[start_row:end_row]
        values = np.asarray(doc_embeddings[doc_batch.reshape(-1)], dtype=np.float32).reshape(
            end_row - start_row,
            int(docs_per_peer),
            -1,
        )
        peer_embeddings[start_row:end_row] = normalize_rows(values.mean(axis=1)).astype(np.float16)

    flat_docs = peer_doc_indices.reshape(-1)
    counts = np.bincount(flat_docs.astype(np.int64), minlength=len(doc_ids)).astype(np.int64)
    owner_offsets = np.zeros(len(doc_ids) + 1, dtype=np.int64)
    owner_offsets[1:] = np.cumsum(counts)
    flat_peers = np.repeat(np.arange(int(peer_count), dtype=np.int32), int(docs_per_peer))
    order = np.argsort(flat_docs, kind="mergesort")
    owner_peer_ids = flat_peers[order].astype(np.int32, copy=False)

    np.save(paths["peer_doc_indices"], peer_doc_indices)
    np.save(paths["peer_test_doc_indices"], peer_test_doc_indices)
    np.save(paths["peer_test_query_indices"], peer_test_query_indices)
    np.save(paths["peer_embeddings"], peer_embeddings)
    np.save(paths["owner_offsets"], owner_offsets)
    np.save(paths["owner_peer_ids"], owner_peer_ids)

    doc_counts_per_peer = np.asarray([len(set(row.tolist())) for row in peer_doc_indices], dtype=np.int32)
    topic_counts_float = nonempty_topic_counts.astype(np.float32)
    training_topic_counts_float = training_topic_counts.astype(np.float32)
    peers_at_most_8 = int(np.count_nonzero(training_topic_counts <= 8))
    peers_at_least_8 = int(np.count_nonzero(training_topic_counts >= 8))
    peers_above_8 = int(np.count_nonzero(training_topic_counts > 8))
    summary: Dict[str, object] = {
        "inventory_mode": "alpha_train",
        "data_dir": data_dir,
        "overlay_dir": overlay_dir,
        "seed": int(seed),
        "peer_count": int(peer_count),
        "docs_per_peer": int(docs_per_peer),
        "events_per_peer": int(events_per_peer),
        "inventory_alpha": float(inventory_alpha),
        "train_records_per_user": int(train_records),
        "test_records_per_user": int(test_records),
        "doc_count": int(len(doc_ids)),
        "owned_doc_assignment_count": int(peer_doc_indices.size),
        "unique_owned_doc_count": int(np.count_nonzero(counts)),
        "min_docs_per_peer": int(doc_counts_per_peer.min()),
        "max_docs_per_peer": int(doc_counts_per_peer.max()),
        "mean_nonempty_topics_per_peer": float(np.mean(topic_counts_float)),
        "median_nonempty_topics_per_peer": float(np.median(topic_counts_float)),
        "p95_nonempty_topics_per_peer": float(np.percentile(topic_counts_float, 95)),
        "max_nonempty_topics_per_peer": int(np.max(nonempty_topic_counts)),
        "training_interest_definition": (
            "distinct_reinforced_sampling_topics_that_produced_accepted_training_documents"
        ),
        "mean_training_interests_per_peer": float(np.mean(training_topic_counts_float)),
        "median_training_interests_per_peer": float(np.median(training_topic_counts_float)),
        "p95_training_interests_per_peer": float(np.percentile(training_topic_counts_float, 95)),
        "max_training_interests_per_peer": int(np.max(training_topic_counts)),
        "peers_at_most_8_training_interests": peers_at_most_8,
        "peer_share_at_most_8_training_interests": float(peers_at_most_8 / int(peer_count)),
        "peers_at_least_8_training_interests": peers_at_least_8,
        "peer_share_at_least_8_training_interests": float(peers_at_least_8 / int(peer_count)),
        "peers_above_8_training_interests": peers_above_8,
        "peer_share_above_8_training_interests": float(peers_above_8 / int(peer_count)),
        "training_interest_count_histogram": {
            str(value): int(np.count_nonzero(training_topic_counts == value))
            for value in np.unique(training_topic_counts).tolist()
        },
        "query_draws": int(query_draws),
        "fallback_doc_draws": int(fallback_doc_draws),
        "global_topic_draws": int(global_topic_draws),
        "build_runtime_seconds": float(time.perf_counter() - start),
        "topic_pool_summary": topic_data.summary,
        "query_pool_summary": query_pool_summary,
        "split_summary": {
            "split_strategy": "sample_200_alpha_doc_query_events_per_peer_then_seeded_random_permutation",
            "events_per_peer": int(events_per_peer),
            "train_records_per_user": int(train_records),
            "test_records_per_user": int(test_records),
            "train_fraction": int(train_records) / int(events_per_peer),
            "test_fraction": int(test_records) / int(events_per_peer),
        },
        "paths": paths,
    }
    _write_json(paths["summary"], summary)
    return load_pubmedqa_alpha_train_overlay(
        data_dir=data_dir,
        cache_root=cache_root,
        seed=seed,
        peer_count=peer_count,
        docs_per_peer=docs_per_peer,
        events_per_peer=events_per_peer,
        inventory_alpha=inventory_alpha,
        train_records=train_records,
        test_records=test_records,
    )


def load_pubmedqa_overlay_test_events(
    *,
    overlay: PubMedQAOverlay,
    manifest: Mapping[str, object],
    doc_ids: Sequence[str],
    sample_queries: Optional[int],
    seed: int,
) -> PubMedQAEvents:
    if overlay.peer_test_doc_indices is None or overlay.peer_test_query_indices is None:
        raise RuntimeError("PubMedQA alpha overlay does not include held-out test event arrays.")
    peer_test_docs = np.asarray(overlay.peer_test_doc_indices)
    peer_test_queries = np.asarray(overlay.peer_test_query_indices)
    if peer_test_docs.shape != peer_test_queries.shape:
        raise RuntimeError("PubMedQA test doc/query arrays have mismatched shapes.")
    peer_count, test_records = peer_test_docs.shape
    total = int(peer_count) * int(test_records)
    if sample_queries is not None and int(sample_queries) > 0 and int(sample_queries) < total:
        rng = np.random.default_rng(stable_int(f"pubmedqa-test-sample:{seed}:{sample_queries}:{total}"))
        flat_positions = np.sort(rng.choice(total, size=int(sample_queries), replace=False))
    else:
        flat_positions = np.arange(total, dtype=np.int64)

    source_peer_ids = (flat_positions // int(test_records)).astype(np.int64)
    test_offsets = (flat_positions % int(test_records)).astype(np.int64)
    target_doc_indices = peer_test_docs[source_peer_ids, test_offsets].astype(np.int64)
    query_indices = peer_test_queries[source_peer_ids, test_offsets].astype(np.int64)
    query_keys = np.load(str(manifest["query_keys_path"]), mmap_mode="r")
    query_values = [str(query_keys[int(index)]) for index in query_indices.tolist()]
    rows = pd.DataFrame(
        {
            "event_index": flat_positions.astype(np.int64),
            "query_id": [f"pmq_syn_{int(value)}" for value in flat_positions.tolist()],
            "qid": [f"pmq_syn_{int(value)}" for value in flat_positions.tolist()],
            "source_peer_id": source_peer_ids.astype(np.int64),
            "query": query_values,
            "query_index": query_indices.astype(np.int64),
            "target_doc_id": [doc_ids[int(index)] for index in target_doc_indices.tolist()],
            "target_doc_index": target_doc_indices.astype(np.int64),
        }
    )
    query_embeddings_mmap = np.load(str(manifest["query_embeddings_path"]), mmap_mode="r")
    query_embeddings = normalize_rows(np.asarray(query_embeddings_mmap[query_indices], dtype=np.float32))
    return PubMedQAEvents(rows=rows.reset_index(drop=True), query_embeddings=query_embeddings)


def load_pubmedqa_centroids(
    *,
    overlay: PubMedQAOverlay,
    mode: str,
    fixed_k: int,
    elbow_max_k: int,
    elbow_method: str,
    relative_drop_threshold: float,
) -> PubMedQACentroids:
    mode = str(mode).strip().lower().replace("-", "_")
    elbow_method = str(elbow_method).strip().lower().replace("-", "_")
    cache_dir = _centroid_cache_dir(
        overlay.overlay_dir,
        mode=mode,
        fixed_k=fixed_k,
        elbow_max_k=elbow_max_k,
        elbow_method=elbow_method,
        relative_drop_threshold=relative_drop_threshold,
    )
    paths = _centroid_cache_paths(cache_dir)
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing PubMedQA centroid cache under {cache_dir}: {missing[:3]}")
    summary = _read_json(paths["summary"])
    centroid_embeddings = normalize_rows(
        np.asarray(np.load(paths["centroid_embeddings"], mmap_mode="r"), dtype=np.float32)
    )
    return PubMedQACentroids(
        mode=mode,
        fixed_k=int(fixed_k),
        elbow_method=elbow_method,
        elbow_max_k=int(elbow_max_k),
        relative_drop_threshold=float(relative_drop_threshold),
        centroid_embeddings=centroid_embeddings,
        centroid_offsets=np.load(paths["centroid_offsets"], mmap_mode="r"),
        selected_k=np.load(paths["selected_k"], mmap_mode="r"),
        summary=summary,
    )


def generate_pubmedqa_centroids(
    *,
    overlay: PubMedQAOverlay,
    doc_embeddings: np.ndarray,
    mode: str,
    fixed_k: int,
    elbow_max_k: int,
    elbow_method: str,
    relative_drop_threshold: float,
    force: bool = False,
) -> PubMedQACentroids:
    mode = str(mode).strip().lower().replace("-", "_")
    elbow_method = str(elbow_method).strip().lower().replace("-", "_")
    label = _centroid_label(mode, fixed_k=fixed_k, elbow_method=elbow_method)
    cache_dir = _centroid_cache_dir(
        overlay.overlay_dir,
        mode=mode,
        fixed_k=fixed_k,
        elbow_max_k=elbow_max_k,
        elbow_method=elbow_method,
        relative_drop_threshold=relative_drop_threshold,
    )
    paths = _centroid_cache_paths(cache_dir)
    if not force and all(os.path.exists(path) for path in paths.values()):
        return load_pubmedqa_centroids(
            overlay=overlay,
            mode=mode,
            fixed_k=fixed_k,
            elbow_max_k=elbow_max_k,
            elbow_method=elbow_method,
            relative_drop_threshold=relative_drop_threshold,
        )
    with _overlay_build_lock(cache_dir):
        if not force and all(os.path.exists(path) for path in paths.values()):
            return load_pubmedqa_centroids(
                overlay=overlay,
                mode=mode,
                fixed_k=fixed_k,
                elbow_max_k=elbow_max_k,
                elbow_method=elbow_method,
                relative_drop_threshold=relative_drop_threshold,
            )
        os.makedirs(cache_dir, exist_ok=True)
        start = time.perf_counter()
        centroid_blocks: List[np.ndarray] = []
        selected_k = np.empty(int(overlay.peer_count), dtype=np.int16)
        offsets = np.zeros(int(overlay.peer_count) + 1, dtype=np.int64)
        for peer_id in range(int(overlay.peer_count)):
            doc_indices = np.asarray(overlay.peer_doc_indices[peer_id], dtype=np.int64)
            values = np.asarray(doc_embeddings[doc_indices], dtype=np.float32)
            fit_seed = stable_int(f"pubmedqa-centroids:{overlay.seed}:{label}:{peer_id}")
            k = _select_peer_centroid_count(
                values,
                mode=mode,
                fixed_k=fixed_k,
                elbow_max_k=elbow_max_k,
                elbow_method=elbow_method,
                relative_drop_threshold=relative_drop_threshold,
                seed=fit_seed,
            )
            centers = _fit_centers(values, k, fit_seed)
            selected_k[peer_id] = int(centers.shape[0])
            offsets[peer_id + 1] = offsets[peer_id] + int(centers.shape[0])
            centroid_blocks.append(centers.astype(np.float32, copy=False))
            if overlay.peer_count >= 10000 and ((peer_id + 1) % 5000 == 0 or peer_id + 1 == overlay.peer_count):
                print(
                    f"Built PubMedQA centroids {label}: {peer_id + 1}/{overlay.peer_count} peers, "
                    f"{int(offsets[peer_id + 1])} centroids.",
                    flush=True,
                )

        centroid_embeddings = (
            normalize_rows(np.vstack(centroid_blocks)).astype(np.float16)
            if centroid_blocks
            else np.empty((0, int(doc_embeddings.shape[1])), dtype=np.float16)
        )
        np.save(paths["centroid_embeddings"], centroid_embeddings)
        np.save(paths["centroid_offsets"], offsets)
        np.save(paths["selected_k"], selected_k)
        unique_k, counts = np.unique(selected_k.astype(np.int64), return_counts=True)
        summary = {
            "label": label,
            "mode": mode,
            "fixed_k": int(fixed_k),
            "elbow_max_k": int(elbow_max_k),
            "elbow_method": elbow_method,
            "relative_drop_threshold": float(relative_drop_threshold),
            "seed": int(overlay.seed),
            "peer_count": int(overlay.peer_count),
            "docs_per_peer": int(overlay.docs_per_peer),
            "embedding_dim": int(doc_embeddings.shape[1]),
            "total_centroids": int(centroid_embeddings.shape[0]),
            "mean_selected_k": float(np.mean(selected_k)) if selected_k.size else 0.0,
            "median_selected_k": float(np.median(selected_k)) if selected_k.size else 0.0,
            "min_selected_k": int(np.min(selected_k)) if selected_k.size else 0,
            "max_selected_k": int(np.max(selected_k)) if selected_k.size else 0,
            "selected_k_histogram": {str(int(k)): int(v) for k, v in zip(unique_k.tolist(), counts.tolist())},
            "cache_dir": cache_dir,
            "build_runtime_seconds": float(time.perf_counter() - start),
        }
        _write_json(paths["summary"], summary)
    return load_pubmedqa_centroids(
        overlay=overlay,
        mode=mode,
        fixed_k=fixed_k,
        elbow_max_k=elbow_max_k,
        elbow_method=elbow_method,
        relative_drop_threshold=relative_drop_threshold,
    )


def build_pubmedqa_graph(
    *,
    overlay: PubMedQAOverlay,
    centroids: PubMedQACentroids,
    graph_source: str,
    graph_degree: int,
    seed: int,
    semantica_leaf_size: int = 50,
    semantica_delta: float = 5e-4,
) -> PubMedQAGraph:
    start = time.perf_counter()
    peer_count = overlay.peer_count
    graph_degree = max(0, min(int(graph_degree), max(0, peer_count - 1)))
    adjacency = np.full((peer_count, graph_degree), -1, dtype=np.int32)
    centroid_label = centroids.label
    centroid_extra = {
        "centroid_label": centroid_label,
        "centroid_mode": centroids.mode,
        "centroid_fixed_k": int(centroids.fixed_k),
        "centroid_elbow_method": centroids.elbow_method if centroids.mode == "elbow" else "",
        "centroid_elbow_max_k": int(centroids.elbow_max_k),
        "centroid_relative_drop_threshold": float(centroids.relative_drop_threshold),
        "total_centroids": int(centroids.centroid_embeddings.shape[0]),
        "mean_selected_k": float(centroids.summary.get("mean_selected_k", 0.0)),
        "min_selected_k": int(centroids.summary.get("min_selected_k", 0)),
        "max_selected_k": int(centroids.summary.get("max_selected_k", 0)),
    }

    def rank_peer_candidates(source_peer_id: int, candidates: Iterable[int]) -> List[int]:
        source_centers = centroids.peer_centroids(int(source_peer_id))
        ranked = []
        seen: set[int] = set()
        for candidate in candidates:
            candidate = int(candidate)
            if candidate == int(source_peer_id) or candidate in seen:
                continue
            seen.add(candidate)
            candidate_centers = centroids.peer_centroids(candidate)
            if not source_centers.size or not candidate_centers.size:
                continue
            score = float(np.max(source_centers @ candidate_centers.T))
            ranked.append((candidate, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return [peer for peer, _score in ranked]

    if graph_source == "random_graph":
        rng = np.random.default_rng(int(seed))
        for peer_id in range(peer_count):
            if graph_degree <= 0:
                continue
            sampled = rng.choice(peer_count - 1, size=graph_degree, replace=False)
            sampled = np.where(sampled >= peer_id, sampled + 1, sampled)
            adjacency[peer_id] = sampled.astype(np.int32)
        extra = {"random_seed": int(seed)}
        extra.update(centroid_extra)
    elif graph_source == "semantica_fixed_k":
        tree = SemanticTree(
            centroids.centroid_embeddings,
            leaf_size=int(semantica_leaf_size),
            delta=float(semantica_delta),
            seed=int(seed),
        )
        centroid_positions = np.arange(int(centroids.centroid_embeddings.shape[0]), dtype=np.int64)
        centroid_peer_ids = np.searchsorted(centroids.centroid_offsets[1:], centroid_positions, side="right").astype(
            np.int32,
            copy=False,
        )
        fallback_links_added = 0
        max_selected_k = max(1, int(centroids.summary.get("max_selected_k", 1)))
        query_limit = min(
            int(centroids.centroid_embeddings.shape[0]),
            max(graph_degree * max_selected_k * 4, graph_degree + max_selected_k + 2, 1),
        )
        duplicate_links_removed = 0
        self_links_removed = 0
        for peer_id in range(peer_count):
            raw_candidates: List[int] = []
            for centroid_pos in range(int(centroids.centroid_offsets[peer_id]), int(centroids.centroid_offsets[peer_id + 1])):
                raw = tree.query_indices(centroids.centroid_embeddings[int(centroid_pos)], query_limit)
                raw_candidates.extend(int(centroid_peer_ids[int(value)]) for value in raw)
            seen: set[int] = {peer_id}
            candidates: List[int] = []
            for candidate in raw_candidates:
                candidate = int(candidate)
                if candidate == int(peer_id):
                    self_links_removed += 1
                    continue
                if candidate in seen:
                    duplicate_links_removed += 1
                    continue
                seen.add(candidate)
                candidates.append(candidate)
            selected = rank_peer_candidates(int(peer_id), candidates)[:graph_degree]
            seen = {int(peer_id), *selected}
            if len(selected) < graph_degree:
                # Guardrail only: the centroid tree should normally produce enough
                # unique owners. Fill rare short rows by mean-embedding similarity.
                scores = overlay.peer_embeddings @ overlay.peer_embeddings[int(peer_id)]
                for candidate in np.argsort(-scores, kind="mergesort").tolist():
                    candidate = int(candidate)
                    if candidate in seen:
                        continue
                    selected.append(candidate)
                    seen.add(candidate)
                    fallback_links_added += 1
                    if len(selected) >= graph_degree:
                        break
            adjacency[peer_id, : len(selected)] = np.asarray(selected, dtype=np.int32)
        extra = {
            "semantica_leaf_size": int(semantica_leaf_size),
            "semantica_delta": float(semantica_delta),
            "fallback_links_added": int(fallback_links_added),
            "duplicate_links_removed": int(duplicate_links_removed),
            "self_links_removed": int(self_links_removed),
            "topology_rule": "semantic_tree_over_peer_centroids",
            "centroid_query_limit": int(query_limit),
        }
        extra.update(centroid_extra)
    else:
        raise ValueError(f"Unsupported PubMedQA graph_source: {graph_source}")

    degrees = np.sum(adjacency >= 0, axis=1)
    diagnostics: Dict[str, object] = {
        "graph_source": graph_source,
        "graph_degree": int(graph_degree),
        "peer_count": int(peer_count),
        "mean_out_degree": float(np.mean(degrees)) if degrees.size else 0.0,
        "min_out_degree": int(np.min(degrees)) if degrees.size else 0,
        "max_out_degree": int(np.max(degrees)) if degrees.size else 0,
        "build_runtime_seconds": float(time.perf_counter() - start),
    }
    diagnostics.update(extra)
    return PubMedQAGraph(
        graph_source=graph_source,
        adjacency=adjacency,
        peer_embeddings=overlay.peer_embeddings,
        diagnostics=diagnostics,
        centroids=centroids,
    )


class IncrementalCandidateRanker:
    def __init__(
        self,
        *,
        overlay: PubMedQAOverlay,
        doc_embeddings: np.ndarray,
        query_embedding: np.ndarray,
        target_doc_index: int,
        top_k: int,
    ) -> None:
        self.overlay = overlay
        self.doc_embeddings = doc_embeddings
        self.query_embedding = normalize_vector(query_embedding)
        self.target_doc_index = int(target_doc_index)
        self.top_k = int(top_k)
        self.seen_docs: set[int] = set()
        self.seen_doc_order: List[int] = []
        self.target_seen = False
        self.target_score = 0.0
        self.docs_ahead_of_target = 0
        self.touched_peers: set[int] = set()

    def add_peers(self, peer_ids: Iterable[int]) -> None:
        new_docs: List[int] = []
        for peer_id in peer_ids:
            peer_id = int(peer_id)
            if peer_id < 0 or peer_id >= self.overlay.peer_count or peer_id in self.touched_peers:
                continue
            self.touched_peers.add(peer_id)
            for doc_index in self.overlay.peer_doc_indices[peer_id].tolist():
                doc_index = int(doc_index)
                if doc_index in self.seen_docs:
                    continue
                self.seen_docs.add(doc_index)
                self.seen_doc_order.append(doc_index)
                new_docs.append(doc_index)
        if not new_docs:
            return

        if not self.target_seen:
            if self.target_doc_index not in self.seen_docs:
                return
            self.target_seen = True
            self.target_score = float(self.doc_embeddings[self.target_doc_index] @ self.query_embedding)
            docs_to_score = self.seen_doc_order
        else:
            docs_to_score = new_docs

        doc_array = np.asarray(docs_to_score, dtype=np.int64)
        scores = self.doc_embeddings[doc_array] @ self.query_embedding
        ahead = (scores > self.target_score) | ((scores == self.target_score) & (doc_array < self.target_doc_index))
        self.docs_ahead_of_target += int(np.count_nonzero(ahead))

    def metric_row(self, target_doc_id: str, doc_ids: Sequence[str], messages_sent: int) -> Dict[str, object]:
        if self.target_seen:
            target_rank = int(self.docs_ahead_of_target) + 1
            hit = target_rank <= self.top_k
        else:
            target_rank = 0
            hit = False
        relevant_count = 1 if hit else 0
        return {
            "precision_at_10": (relevant_count / self.top_k) if self.top_k > 0 else 0.0,
            "recall_at_10": float(relevant_count),
            "hit_at_10": float(relevant_count),
            "relevant_count_at_10": float(relevant_count),
            "mrr_at_10": (1.0 / target_rank) if hit else 0.0,
            "ndcg_at_10": (1.0 / math.log2(target_rank + 1)) if hit else 0.0,
            "unique_peers_touched": int(len(self.touched_peers)),
            "candidate_docs_scored": int(len(self.seen_docs)),
            "messages_sent": int(messages_sent),
        }


class RunningMetricSummary:
    def __init__(self) -> None:
        self.sums: Dict[Tuple[object, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.counts: Dict[Tuple[object, ...], int] = defaultdict(int)

    def add(self, key: Tuple[object, ...], row: Mapping[str, object]) -> None:
        self.counts[key] += 1
        for field in (
            "precision_at_10",
            "recall_at_10",
            "hit_at_10",
            "relevant_count_at_10",
            "mrr_at_10",
            "ndcg_at_10",
            "unique_peers_touched",
            "candidate_docs_scored",
            "messages_sent",
        ):
            self.sums[key][field] += float(row.get(field, 0.0))

    def payload(self) -> Dict[str, object]:
        return {
            "counts": dict(self.counts),
            "sums": {key: dict(values) for key, values in self.sums.items()},
        }

    def merge_payload(self, payload: Mapping[str, object]) -> None:
        for key, count in payload.get("counts", {}).items():
            self.counts[key] += int(count)
        for key, values in payload.get("sums", {}).items():
            for field, value in values.items():
                self.sums[key][field] += float(value)

    def to_frame(self, columns: Sequence[str]) -> pd.DataFrame:
        rows = []
        for key in sorted(self.counts):
            count = self.counts[key]
            row = {column: value for column, value in zip(columns, key)}
            for field, value in self.sums[key].items():
                row[field] = value / count if count else 0.0
            row["query_hit_rate"] = row.get("hit_at_10", 0.0)
            row["query_hit_count"] = int(round(self.sums[key].get("hit_at_10", 0.0)))
            row["evaluated_query_count"] = int(count)
            row["primary_metric"] = "query_hit_rate"
            rows.append(row)
        return pd.DataFrame(rows)


def _chain_path(
    *,
    graph: PubMedQAGraph,
    source_peer_id: int,
    query_embedding: np.ndarray,
    max_hops: int,
    random_walk: bool,
    seed_token: object,
) -> List[int]:
    path: List[int] = []
    current = int(source_peer_id)
    visited: set[int] = {current}
    rng = np.random.default_rng(stable_int(seed_token))
    for _hop in range(int(max_hops)):
        candidates = [int(value) for value in graph.neighbors(current).tolist() if int(value) not in visited]
        if not candidates:
            break
        if random_walk:
            next_peer = int(rng.choice(np.asarray(candidates, dtype=np.int64)))
        else:
            next_peer = graph.best_neighbor(current, query_embedding, visited)
            if next_peer is None:
                break
        path.append(int(next_peer))
        visited.add(int(next_peer))
        current = int(next_peer)
    return path


def _evaluate_graph_hop_summary_range(
    *,
    data_dir: str,
    doc_ids: Sequence[str],
    doc_embeddings: np.ndarray,
    events: PubMedQAEvents,
    overlay: PubMedQAOverlay,
    graph: PubMedQAGraph,
    seed: int,
    graph_degree: int,
    routing_policies: Sequence[str],
    top_k: int,
    max_hops: int,
    start_query: int,
    end_query: int,
) -> RunningMetricSummary:
    summary = RunningMetricSummary()
    dataset_name = DEFAULT_PUBMEDQA_DATASET_NAME
    centroid_label = str(graph.diagnostics.get("centroid_label", "fixed_1"))
    centroid_mode = str(graph.diagnostics.get("centroid_mode", "fixed"))
    centroid_fixed_k = int(graph.diagnostics.get("centroid_fixed_k", 1))
    centroid_elbow_method = str(graph.diagnostics.get("centroid_elbow_method", ""))
    base_key = (
        dataset_name,
        seed,
        graph.graph_source,
        graph_degree,
        centroid_label,
        centroid_mode,
        centroid_fixed_k,
        centroid_elbow_method,
    )
    event_rows = events.rows.iloc[int(start_query) : int(end_query)]
    for query_pos, event in enumerate(event_rows.itertuples(index=False), start=int(start_query)):
        source_peer_id = int(event.source_peer_id)
        if source_peer_id < 0 or source_peer_id >= overlay.peer_count:
            continue
        query_embedding = events.query_embeddings[query_pos]
        target_doc_id = str(event.target_doc_id)
        target_doc_index = int(event.target_doc_index)
        semantic_path: Optional[List[int]] = None
        if "chain_only" in routing_policies or "chain_plus_direct_topk" in routing_policies:
            semantic_path = _chain_path(
                graph=graph,
                source_peer_id=source_peer_id,
                query_embedding=query_embedding,
                max_hops=max_hops,
                random_walk=False,
                seed_token=f"{seed}:{event.query_id}:semantic_chain",
            )
        if "chain_only" in routing_policies:
            ranker = IncrementalCandidateRanker(
                overlay=overlay,
                doc_embeddings=doc_embeddings,
                query_embedding=query_embedding,
                target_doc_index=target_doc_index,
                top_k=top_k,
            )
            row = ranker.metric_row(target_doc_id, doc_ids, messages_sent=0)
            summary.add(base_key + ("chain_only", 0), row)
            path = semantic_path or []
            for hop, peer_id in enumerate(path, start=1):
                ranker.add_peers([peer_id])
                row = ranker.metric_row(target_doc_id, doc_ids, messages_sent=hop)
                summary.add(base_key + ("chain_only", hop), row)
            for hop in range(len(path) + 1, int(max_hops) + 1):
                row = ranker.metric_row(target_doc_id, doc_ids, messages_sent=len(path))
                summary.add(base_key + ("chain_only", hop), row)

        if "chain_plus_direct_topk" in routing_policies:
            ranker = IncrementalCandidateRanker(
                overlay=overlay,
                doc_embeddings=doc_embeddings,
                query_embedding=query_embedding,
                target_doc_index=target_doc_index,
                top_k=top_k,
            )
            initial = graph.neighbors(source_peer_id, graph_degree)
            ranker.add_peers(initial)
            row = ranker.metric_row(target_doc_id, doc_ids, messages_sent=len(initial))
            summary.add(base_key + ("chain_plus_direct_topk", 0), row)
            path = semantic_path or []
            for hop, peer_id in enumerate(path, start=1):
                peers_to_add = [int(peer_id)]
                peers_to_add.extend(int(value) for value in graph.neighbors(peer_id, graph_degree).tolist())
                ranker.add_peers(peers_to_add)
                row = ranker.metric_row(target_doc_id, doc_ids, messages_sent=len(ranker.touched_peers))
                summary.add(base_key + ("chain_plus_direct_topk", hop), row)
            for hop in range(len(path) + 1, int(max_hops) + 1):
                row = ranker.metric_row(target_doc_id, doc_ids, messages_sent=len(ranker.touched_peers))
                summary.add(base_key + ("chain_plus_direct_topk", hop), row)

        if "direct_topk_entry" in routing_policies:
            ranker = IncrementalCandidateRanker(
                overlay=overlay,
                doc_embeddings=doc_embeddings,
                query_embedding=query_embedding,
                target_doc_index=target_doc_index,
                top_k=top_k,
            )
            neighbors = graph.neighbors(source_peer_id, graph_degree).tolist()
            row = ranker.metric_row(target_doc_id, doc_ids, messages_sent=0)
            summary.add(base_key + ("direct_topk_entry", 0), row)
            for hop, peer_id in enumerate(neighbors, start=1):
                ranker.add_peers([int(peer_id)])
                row = ranker.metric_row(target_doc_id, doc_ids, messages_sent=hop)
                summary.add(base_key + ("direct_topk_entry", hop), row)

        if "random_agglomeration" in routing_policies:
            rng = np.random.default_rng(stable_int(f"{seed}:{event.query_id}:random_agglomeration"))
            ranker = IncrementalCandidateRanker(
                overlay=overlay,
                doc_embeddings=doc_embeddings,
                query_embedding=query_embedding,
                target_doc_index=target_doc_index,
                top_k=top_k,
            )
            touched: set[int] = {source_peer_id}
            frontier: set[int] = {source_peer_id}
            row = ranker.metric_row(target_doc_id, doc_ids, messages_sent=0)
            summary.add(base_key + ("random_agglomeration", 0), row)
            for hop in range(1, int(max_hops) + 1):
                candidates: List[int] = []
                seen_candidates: set[int] = set()
                for peer_id in sorted(frontier):
                    for neighbor in graph.neighbors(peer_id).tolist():
                        neighbor = int(neighbor)
                        if neighbor in touched or neighbor in seen_candidates:
                            continue
                        seen_candidates.add(neighbor)
                        candidates.append(neighbor)
                if candidates:
                    sample_size = min(int(graph_degree), len(candidates))
                    sampled = rng.choice(np.asarray(candidates, dtype=np.int64), size=sample_size, replace=False)
                    frontier = {int(value) for value in sampled.tolist()}
                    touched.update(frontier)
                    ranker.add_peers(frontier)
                row = ranker.metric_row(target_doc_id, doc_ids, messages_sent=len(ranker.touched_peers))
                summary.add(base_key + ("random_agglomeration", hop), row)
    return summary


_PUBMED_HOP_WORKER_CONTEXT: Optional[Dict[str, object]] = None


def _query_ranges(total_queries: int, chunk_size: int) -> List[Tuple[int, int]]:
    chunk_size = max(1, int(chunk_size))
    return [
        (start, min(start + chunk_size, int(total_queries)))
        for start in range(0, int(total_queries), chunk_size)
    ]


def _evaluate_graph_hop_worker(bounds: Tuple[int, int]) -> Tuple[Dict[str, object], int]:
    if _PUBMED_HOP_WORKER_CONTEXT is None:
        raise RuntimeError("PubMedQA hop worker context is not initialized.")
    start_query, end_query = bounds
    summary = _evaluate_graph_hop_summary_range(
        start_query=start_query,
        end_query=end_query,
        **_PUBMED_HOP_WORKER_CONTEXT,
    )
    return summary.payload(), int(end_query - start_query)


def _log_query_progress(
    *,
    phase: str,
    completed: int,
    total: int,
    next_threshold: int,
    every: int,
) -> int:
    while every > 0 and completed >= next_threshold and next_threshold <= total:
        print(
            f"PubMedQA {phase} progress: {next_threshold}/{total} queries processed.",
            flush=True,
        )
        next_threshold += every
    return next_threshold


def _evaluate_graph_hop_summaries(
    *,
    data_dir: str,
    doc_ids: Sequence[str],
    doc_embeddings: np.ndarray,
    events: PubMedQAEvents,
    overlay: PubMedQAOverlay,
    graph: PubMedQAGraph,
    seed: int,
    graph_degree: int,
    routing_policies: Sequence[str],
    top_k: int,
    max_hops: int,
    query_workers: int = 1,
    query_chunk_size: int = 50_000,
    query_progress_every: int = 500_000,
) -> pd.DataFrame:
    key_columns = (
        "dataset_name",
        "seed",
        "graph_source",
        "graph_degree",
        "centroid_label",
        "centroid_mode",
        "centroid_fixed_k",
        "centroid_elbow_method",
        "routing_policy",
        "hop",
    )
    total_queries = int(events.rows.shape[0])
    ranges = _query_ranges(total_queries, query_chunk_size)
    workers = max(1, min(int(query_workers), len(ranges) or 1))
    context_values = {
        "data_dir": data_dir,
        "doc_ids": doc_ids,
        "doc_embeddings": doc_embeddings,
        "events": events,
        "overlay": overlay,
        "graph": graph,
        "seed": seed,
        "graph_degree": graph_degree,
        "routing_policies": routing_policies,
        "top_k": top_k,
        "max_hops": max_hops,
    }
    combined = RunningMetricSummary()
    completed = 0
    every = max(0, int(query_progress_every))
    next_threshold = every
    print(
        f"PubMedQA routing start: {total_queries} queries, {workers} worker(s), "
        f"progress interval={every}.",
        flush=True,
    )

    global _PUBMED_HOP_WORKER_CONTEXT
    _PUBMED_HOP_WORKER_CONTEXT = context_values
    use_pool = workers > 1 and "fork" in mp.get_all_start_methods()
    if use_pool:
        context = mp.get_context("fork")
        with context.Pool(processes=workers) as pool:
            results = pool.imap_unordered(_evaluate_graph_hop_worker, ranges, chunksize=1)
            for payload, processed in results:
                combined.merge_payload(payload)
                completed += int(processed)
                next_threshold = _log_query_progress(
                    phase="routing",
                    completed=completed,
                    total=total_queries,
                    next_threshold=next_threshold,
                    every=every,
                )
    else:
        if workers > 1:
            print("Fork multiprocessing is unavailable; routing with one worker.", flush=True)
        for bounds in ranges:
            payload, processed = _evaluate_graph_hop_worker(bounds)
            combined.merge_payload(payload)
            completed += int(processed)
            next_threshold = _log_query_progress(
                phase="routing",
                completed=completed,
                total=total_queries,
                next_threshold=next_threshold,
                every=every,
            )
    _PUBMED_HOP_WORKER_CONTEXT = None
    print(f"PubMedQA routing complete: {completed}/{total_queries} queries processed.", flush=True)

    frame = combined.to_frame(key_columns)
    if frame.empty:
        return frame
    frame["query_hop"] = frame["hop"].astype(int)
    frame["measurement_step_type"] = "query_routing_hop"
    hop_semantics = {
        "chain_only": "one_additional_chain_peer_contacted_per_query_hop",
        "chain_plus_direct_topk": "one_chain_step_then_direct_neighbors_contacted_per_query_hop",
        "direct_topk_entry": "one_additional_entry_neighbor_contacted_per_query_hop",
        "random_agglomeration": "one_random_frontier_expansion_per_query_hop",
    }
    frame["query_hop_semantics"] = frame["routing_policy"].map(hop_semantics).fillna("query_routing_hop")
    return frame


def _doc_distance_counts_range(
    *,
    events: PubMedQAEvents,
    overlay: PubMedQAOverlay,
    graph: PubMedQAGraph,
    max_distance: int,
    start_query: int,
    end_query: int,
) -> Dict[str, object]:
    buckets: Dict[str, int] = defaultdict(int)
    total = 0
    unavailable = 0
    event_rows = events.rows.iloc[int(start_query) : int(end_query)]
    for event in event_rows.itertuples(index=False):
        source_peer_id = int(event.source_peer_id)
        target_doc_index = int(event.target_doc_index)
        if source_peer_id < 0 or source_peer_id >= overlay.peer_count:
            buckets["unavailable"] += 1
            unavailable += 1
            total += 1
            continue
        owners = {int(value) for value in overlay.owners_for_doc(target_doc_index).tolist() if int(value) != source_peer_id}
        total += 1
        if not owners:
            buckets["unavailable"] += 1
            unavailable += 1
            continue
        visited: set[int] = {source_peer_id}
        frontier: set[int] = {source_peer_id}
        found_distance: Optional[int] = None
        for distance in range(1, int(max_distance) + 1):
            next_frontier: set[int] = set()
            for peer_id in frontier:
                for neighbor in graph.neighbors(peer_id).tolist():
                    neighbor = int(neighbor)
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
            if next_frontier & owners:
                found_distance = distance
                break
            if not next_frontier:
                break
            frontier = next_frontier
        if found_distance is None:
            buckets["unreachable"] += 1
        else:
            buckets[str(found_distance)] += 1
    return {
        "buckets": dict(buckets),
        "total": int(total),
        "unavailable": int(unavailable),
    }


_PUBMED_DISTANCE_WORKER_CONTEXT: Optional[Dict[str, object]] = None


def _doc_distance_worker(bounds: Tuple[int, int]) -> Tuple[Dict[str, object], int]:
    if _PUBMED_DISTANCE_WORKER_CONTEXT is None:
        raise RuntimeError("PubMedQA distance worker context is not initialized.")
    start_query, end_query = bounds
    result = _doc_distance_counts_range(
        start_query=start_query,
        end_query=end_query,
        **_PUBMED_DISTANCE_WORKER_CONTEXT,
    )
    return result, int(end_query - start_query)


def compute_doc_distance_histogram(
    *,
    events: PubMedQAEvents,
    overlay: PubMedQAOverlay,
    graph: PubMedQAGraph,
    seed: int,
    graph_degree: int,
    max_distance: int,
    query_workers: int = 1,
    query_chunk_size: int = 50_000,
    query_progress_every: int = 500_000,
) -> pd.DataFrame:
    centroid_label = str(graph.diagnostics.get("centroid_label", "fixed_1"))
    centroid_mode = str(graph.diagnostics.get("centroid_mode", "fixed"))
    centroid_fixed_k = int(graph.diagnostics.get("centroid_fixed_k", 1))
    centroid_elbow_method = str(graph.diagnostics.get("centroid_elbow_method", ""))
    total_queries = int(events.rows.shape[0])
    ranges = _query_ranges(total_queries, query_chunk_size)
    workers = max(1, min(int(query_workers), len(ranges) or 1))
    context_values = {
        "events": events,
        "overlay": overlay,
        "graph": graph,
        "max_distance": max_distance,
    }
    buckets: Counter = Counter()
    total = 0
    unavailable = 0
    completed = 0
    every = max(0, int(query_progress_every))
    next_threshold = every
    print(
        f"PubMedQA distance start: {total_queries} queries, {workers} worker(s), "
        f"progress interval={every}.",
        flush=True,
    )

    global _PUBMED_DISTANCE_WORKER_CONTEXT
    _PUBMED_DISTANCE_WORKER_CONTEXT = context_values
    use_pool = workers > 1 and "fork" in mp.get_all_start_methods()
    if use_pool:
        context = mp.get_context("fork")
        with context.Pool(processes=workers) as pool:
            results = pool.imap_unordered(_doc_distance_worker, ranges, chunksize=1)
            for result, processed in results:
                buckets.update(result["buckets"])
                total += int(result["total"])
                unavailable += int(result["unavailable"])
                completed += int(processed)
                next_threshold = _log_query_progress(
                    phase="distance",
                    completed=completed,
                    total=total_queries,
                    next_threshold=next_threshold,
                    every=every,
                )
    else:
        if workers > 1:
            print("Fork multiprocessing is unavailable; distance evaluation uses one worker.", flush=True)
        for bounds in ranges:
            result, processed = _doc_distance_worker(bounds)
            buckets.update(result["buckets"])
            total += int(result["total"])
            unavailable += int(result["unavailable"])
            completed += int(processed)
            next_threshold = _log_query_progress(
                phase="distance",
                completed=completed,
                total=total_queries,
                next_threshold=next_threshold,
                every=every,
            )
    _PUBMED_DISTANCE_WORKER_CONTEXT = None
    print(f"PubMedQA distance complete: {completed}/{total_queries} queries processed.", flush=True)

    numeric_distances = sorted(int(key) for key in buckets if key.isdigit())
    cumulative = 0
    rows = []
    for distance in numeric_distances:
        count = int(buckets[str(distance)])
        cumulative += count
        rows.append(
            {
                "dataset_name": DEFAULT_PUBMEDQA_DATASET_NAME,
                "seed": int(seed),
                "graph_source": graph.graph_source,
                "graph_degree": int(graph_degree),
                "centroid_label": centroid_label,
                "centroid_mode": centroid_mode,
                "centroid_fixed_k": int(centroid_fixed_k),
                "centroid_elbow_method": centroid_elbow_method,
                "distance": int(distance),
                "doc_query_count": count,
                "fraction": count / total if total else 0.0,
                "cumulative_count": cumulative,
                "cumulative_fraction": cumulative / total if total else 0.0,
                "total_evaluated": int(total),
                "unavailable_count": int(unavailable),
            }
        )
    for special in ("unreachable", "unavailable"):
        count = int(buckets.get(special, 0))
        rows.append(
            {
                "dataset_name": DEFAULT_PUBMEDQA_DATASET_NAME,
                "seed": int(seed),
                "graph_source": graph.graph_source,
                "graph_degree": int(graph_degree),
                "centroid_label": centroid_label,
                "centroid_mode": centroid_mode,
                "centroid_fixed_k": int(centroid_fixed_k),
                "centroid_elbow_method": centroid_elbow_method,
                "distance": special,
                "doc_query_count": count,
                "fraction": count / total if total else 0.0,
                "cumulative_count": "",
                "cumulative_fraction": "",
                "total_evaluated": int(total),
                "unavailable_count": int(unavailable),
            }
        )
    return pd.DataFrame(rows)


def evaluate_hnsw_reference(
    *,
    doc_embeddings: np.ndarray,
    doc_ids: Sequence[str],
    events: PubMedQAEvents,
    seed: int,
    top_k: int,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
    query_batch_size: int = 512,
) -> pd.DataFrame:
    try:
        import hnswlib
    except ImportError as exc:
        raise ImportError("Install hnswlib to compute the PubMedQA HNSW reference.") from exc
    start = time.perf_counter()
    index = hnswlib.Index(space="cosine", dim=int(doc_embeddings.shape[1]))
    index.init_index(
        max_elements=int(doc_embeddings.shape[0]),
        ef_construction=int(hnsw_ef_construction),
        M=int(hnsw_m),
        random_seed=int(seed),
    )
    index.add_items(doc_embeddings, np.arange(int(doc_embeddings.shape[0])))
    index.set_ef(int(hnsw_ef_search))
    build_runtime = time.perf_counter() - start
    metric_rows = []
    query_runtime = 0.0
    for start_row in range(0, events.query_count, int(query_batch_size)):
        end_row = min(start_row + int(query_batch_size), events.query_count)
        batch = events.query_embeddings[start_row:end_row]
        q_start = time.perf_counter()
        labels, _distances = index.knn_query(batch, k=min(int(top_k), len(doc_ids)))
        query_runtime += time.perf_counter() - q_start
        for offset, doc_indices in enumerate(labels.tolist()):
            event = events.rows.iloc[start_row + offset]
            retrieved = [doc_ids[int(idx)] for idx in doc_indices]
            metric_rows.append(_metric_values(retrieved, {str(event.target_doc_id)}, int(top_k)))

    def mean_metric(name: str) -> float:
        return float(np.mean([row[name] for row in metric_rows])) if metric_rows else 0.0

    return pd.DataFrame(
        [
            {
                "dataset_name": DEFAULT_PUBMEDQA_DATASET_NAME,
                "method_name": "hnsw",
                "seed": int(seed),
                "top_k": int(top_k),
                "evaluated_query_count": int(len(metric_rows)),
                "precision_at_10": mean_metric("precision"),
                "recall_at_10": mean_metric("recall"),
                "hit_rate_at_10": mean_metric("hit"),
                "query_hit_rate": mean_metric("hit"),
                "primary_metric": "query_hit_rate",
                "mean_relevant_count_at_10": mean_metric("relevant_count"),
                "mrr_at_10": mean_metric("mrr"),
                "ndcg_at_10": mean_metric("ndcg"),
                "build_runtime_seconds": float(build_runtime),
                "query_runtime_seconds": float(query_runtime),
                "runtime_seconds": float(build_runtime + query_runtime),
                "hnsw_m": int(hnsw_m),
                "hnsw_ef_construction": int(hnsw_ef_construction),
                "hnsw_ef_search": int(hnsw_ef_search),
            }
        ]
    )


def _metrics_summary_from_hop_summary(hop_summary: pd.DataFrame, hnsw_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not hop_summary.empty:
        group_cols = [
            "dataset_name",
            "seed",
            "graph_source",
            "graph_degree",
            "centroid_label",
            "centroid_mode",
            "centroid_fixed_k",
            "centroid_elbow_method",
            "routing_policy",
        ]
        for column, default in (
            ("centroid_label", "fixed_1"),
            ("centroid_mode", "fixed"),
            ("centroid_fixed_k", 1),
            ("centroid_elbow_method", ""),
        ):
            if column not in hop_summary.columns:
                hop_summary[column] = default
        for keys, group in hop_summary.groupby(
            group_cols,
            dropna=False,
        ):
            (
                dataset_name,
                seed,
                graph_source,
                graph_degree,
                centroid_label,
                centroid_mode,
                centroid_fixed_k,
                centroid_elbow_method,
                routing_policy,
            ) = keys
            group = group.sort_values("hop")
            final = group.iloc[-1].to_dict()
            best = group.sort_values(["precision_at_10", "hit_at_10", "hop"], ascending=[False, False, True]).iloc[0]
            final.update(
                {
                    "method_name": f"{graph_source}:{centroid_label}:{routing_policy}",
                    "selected_hop": int(final["hop"]),
                    "best_precision_at_10": float(best["precision_at_10"]),
                    "best_precision_hop": int(best["hop"]),
                    "dataset_name": dataset_name,
                    "seed": int(seed),
                    "graph_source": graph_source,
                    "graph_degree": int(graph_degree),
                    "centroid_label": centroid_label,
                    "centroid_mode": centroid_mode,
                    "centroid_fixed_k": int(centroid_fixed_k),
                    "centroid_elbow_method": centroid_elbow_method,
                    "routing_policy": routing_policy,
                }
            )
            rows.append(final)
    if hnsw_summary is not None and not hnsw_summary.empty:
        for row in hnsw_summary.to_dict("records"):
            if "precision_at_10" not in row:
                continue
            rows.append(
                {
                    "dataset_name": row["dataset_name"],
                    "seed": int(row["seed"]),
                    "graph_source": "hnsw",
                    "graph_degree": -1,
                    "centroid_label": row.get("centroid_label", ""),
                    "centroid_mode": row.get("centroid_mode", ""),
                    "centroid_fixed_k": row.get("centroid_fixed_k", ""),
                    "centroid_elbow_method": row.get("centroid_elbow_method", ""),
                    "routing_policy": "full_corpus",
                    "method_name": "hnsw",
                    "selected_hop": -1,
                    "precision_at_10": row["precision_at_10"],
                    "recall_at_10": row["recall_at_10"],
                    "hit_at_10": row["hit_rate_at_10"],
                    "query_hit_rate": row["hit_rate_at_10"],
                    "query_hit_count": int(
                        round(float(row["hit_rate_at_10"]) * int(row["evaluated_query_count"]))
                    ),
                    "relevant_count_at_10": row["mean_relevant_count_at_10"],
                    "mrr_at_10": row["mrr_at_10"],
                    "ndcg_at_10": row["ndcg_at_10"],
                    "evaluated_query_count": row["evaluated_query_count"],
                    "primary_metric": "query_hit_rate",
                    "query_hop": -1,
                    "measurement_step_type": "full_corpus_ann_reference",
                    "query_hop_semantics": "not_applicable_full_corpus_hnsw",
                    "runtime_seconds": row["runtime_seconds"],
                }
            )
    return pd.DataFrame(rows)


def validate_metrics_only_output(output_dir: str) -> None:
    for name in BAD_ARTIFACT_NAMES:
        if os.path.exists(os.path.join(output_dir, name)):
            raise RuntimeError(f"Metrics-only PubMedQA run emitted forbidden artifact: {name}")


def run_pubmedqa_overlay_benchmark(
    *,
    data_dir: str,
    cache_root: str,
    output_dir: str,
    graph_source: str,
    graph_degree: int,
    seed: int,
    sample_queries: Optional[int],
    routing_policies: Sequence[str],
    top_k: int = 10,
    max_hops: int = 100,
    inventory_mode: str = "alpha_train",
    inventory_alpha: float = DEFAULT_PUBMEDQA_INVENTORY_ALPHA,
    events_per_peer: int = DEFAULT_PUBMEDQA_EVENTS_PER_PEER,
    train_records: int = DEFAULT_PUBMEDQA_TRAIN_RECORDS,
    test_records: int = DEFAULT_PUBMEDQA_TEST_RECORDS,
    docs_per_peer: Optional[int] = None,
    interests_per_peer: int = 10,
    docs_per_interest: int = 20,
    peer_count: int = 50_000,
    semantica_leaf_size: int = 50,
    semantica_delta: float = 5e-4,
    centroid_mode: str = "fixed",
    centroid_k: int = 1,
    centroid_elbow_method: str = "knee",
    centroid_elbow_max_k: int = 8,
    centroid_relative_drop_threshold: float = 0.10,
    force_centroids: bool = False,
    compute_hnsw_reference: bool = True,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 100,
    force_overlay: bool = False,
    query_workers: int = 1,
    query_chunk_size: int = 50_000,
    query_progress_every: int = 500_000,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    started = time.perf_counter()
    inventory_mode = str(inventory_mode).strip().lower().replace("-", "_")
    if inventory_mode in {"exact", "exact10", "exact_interests"}:
        inventory_mode = "exact_interests"
    elif inventory_mode in {"alpha", "alpha_train", "train_alpha"}:
        inventory_mode = "alpha_train"
    else:
        raise ValueError(f"Unsupported PubMedQA inventory_mode: {inventory_mode}")
    centroid_mode = str(centroid_mode).strip().lower().replace("-", "_")
    centroid_elbow_method = str(centroid_elbow_method).strip().lower().replace("-", "_")
    centroid_label = _centroid_label(centroid_mode, fixed_k=centroid_k, elbow_method=centroid_elbow_method)
    if int(train_records) + int(test_records) != int(events_per_peer):
        raise ValueError("train_records + test_records must equal events_per_peer.")
    resolved_docs_per_peer = (
        int(docs_per_peer)
        if docs_per_peer is not None
        else (int(train_records) if inventory_mode == "alpha_train" else int(interests_per_peer) * int(docs_per_interest))
    )
    config = BenchmarkConfig(
        top_k=int(top_k),
        seed=int(seed),
        semantica_leaf_size=int(semantica_leaf_size),
        semantica_delta=float(semantica_delta),
        hnsw_m=int(hnsw_m),
        hnsw_ef_construction=int(hnsw_ef_construction),
        hnsw_ef_search=int(hnsw_ef_search),
    )
    run_config = {
        "dataset_name": DEFAULT_PUBMEDQA_DATASET_NAME,
        "data_dir": data_dir,
        "cache_root": cache_root,
        "graph_source": graph_source,
        "graph_degree": int(graph_degree),
        "seed": int(seed),
        "sample_queries": sample_queries,
        "routing_policies": list(routing_policies),
        "top_k": int(top_k),
        "max_hops": int(max_hops),
        "inventory_mode": inventory_mode,
        "inventory_alpha": float(inventory_alpha),
        "events_per_peer": int(events_per_peer),
        "train_records": int(train_records),
        "test_records": int(test_records),
        "peer_count": int(peer_count),
        "interests_per_peer": int(interests_per_peer),
        "docs_per_interest": int(docs_per_interest),
        "docs_per_peer": int(resolved_docs_per_peer),
        "centroid_label": centroid_label,
        "centroid_mode": centroid_mode,
        "centroid_k": int(centroid_k),
        "centroid_elbow_method": centroid_elbow_method if centroid_mode == "elbow" else "",
        "centroid_elbow_max_k": int(centroid_elbow_max_k),
        "centroid_relative_drop_threshold": float(centroid_relative_drop_threshold),
        "compute_hnsw_reference": bool(compute_hnsw_reference),
        "query_workers": int(query_workers),
        "query_chunk_size": int(query_chunk_size),
        "query_progress_every": int(query_progress_every),
        "metrics_only": True,
        "retrieval_metric_definitions": {
            "primary_metric": "query_hit_rate",
            "query_hit_rate": "fraction_of_evaluated_queries_with_the_target_document_in_the_retrieved_top_k",
            "precision_at_10": "relevant_documents_in_top_10_divided_by_10",
            "recall_at_10": "relevant_documents_in_top_10_divided_by_relevant_document_count",
            "rate_scale": "0_to_1",
        },
        "query_hop_semantics": {
            "chain_only": "one_additional_chain_peer_contacted_per_query_hop",
            "chain_plus_direct_topk": "one_chain_step_then_direct_neighbors_contacted_per_query_hop",
            "direct_topk_entry": "one_additional_entry_neighbor_contacted_per_query_hop",
            "random_agglomeration": "one_random_frontier_expansion_per_query_hop",
        },
        "benchmark_config": config_to_dict(config),
    }
    _write_json(os.path.join(output_dir, "run_config.json"), run_config)

    manifest = _load_manifest(data_dir)
    doc_ids = _load_doc_keys(manifest)
    if inventory_mode == "alpha_train":
        overlay = generate_pubmedqa_alpha_train_overlay(
            data_dir=data_dir,
            cache_root=cache_root,
            seed=seed,
            peer_count=peer_count,
            docs_per_peer=resolved_docs_per_peer,
            events_per_peer=events_per_peer,
            inventory_alpha=inventory_alpha,
            train_records=train_records,
            test_records=test_records,
            force=force_overlay,
        )
        events = load_pubmedqa_overlay_test_events(
            overlay=overlay,
            manifest=manifest,
            doc_ids=doc_ids,
            sample_queries=sample_queries,
            seed=seed,
        )
        split_summary = overlay.summary.get("split_summary", {})
    else:
        overlay = generate_pubmedqa_overlay(
            data_dir=data_dir,
            cache_root=cache_root,
            seed=seed,
            peer_count=peer_count,
            interests_per_peer=interests_per_peer,
            docs_per_interest=docs_per_interest,
            force=force_overlay,
        )
        events = load_pubmedqa_events(
            data_dir=data_dir,
            manifest=manifest,
            doc_ids=doc_ids,
            sample_queries=sample_queries,
            seed=seed,
        )
        split_summary = {"split_strategy": "none_exact_interest_overlay"}
    _write_json(os.path.join(output_dir, "peer_inventory_summary.json"), overlay.summary)
    validation = {
        "dataset_name": DEFAULT_PUBMEDQA_DATASET_NAME,
        "source_data_dir": data_dir,
        "inventory_mode": inventory_mode,
        "inventory_alpha": float(inventory_alpha) if inventory_mode == "alpha_train" else "",
        "events_per_peer": int(events_per_peer) if inventory_mode == "alpha_train" else "",
        "train_records": int(train_records) if inventory_mode == "alpha_train" else "",
        "test_records": int(test_records) if inventory_mode == "alpha_train" else "",
        "split_summary": split_summary,
        "documents": int(len(doc_ids)),
        "queries_loaded": int(events.query_count),
        "embedding_dim": int(manifest.get("embedding_dim", 0)),
        "peer_count": int(overlay.peer_count),
        "docs_per_peer": int(overlay.docs_per_peer),
        "centroid_label": centroid_label,
        "centroid_mode": centroid_mode,
        "centroid_k": int(centroid_k),
        "centroid_elbow_method": centroid_elbow_method if centroid_mode == "elbow" else "",
        "centroid_elbow_max_k": int(centroid_elbow_max_k),
        "centroid_relative_drop_threshold": float(centroid_relative_drop_threshold),
        "interests_per_peer": int(overlay.peer_topic_ids.shape[1]) if inventory_mode == "exact_interests" else "",
        "mean_nonempty_topics_per_peer": overlay.summary.get("mean_nonempty_topics_per_peer", ""),
        "single_target_relevance": True,
        "source_peer_column": "user_id/source_peer_id",
    }
    _write_json(os.path.join(output_dir, "dataset_validation.json"), validation)

    doc_embeddings = _load_normalized_doc_embeddings(manifest)
    centroids = generate_pubmedqa_centroids(
        overlay=overlay,
        doc_embeddings=doc_embeddings,
        mode=centroid_mode,
        fixed_k=centroid_k,
        elbow_max_k=centroid_elbow_max_k,
        elbow_method=centroid_elbow_method,
        relative_drop_threshold=centroid_relative_drop_threshold,
        force=force_centroids,
    )
    graph = build_pubmedqa_graph(
        overlay=overlay,
        centroids=centroids,
        graph_source=graph_source,
        graph_degree=graph_degree,
        seed=seed,
        semantica_leaf_size=semantica_leaf_size,
        semantica_delta=semantica_delta,
    )
    pd.DataFrame([graph.diagnostics]).to_csv(os.path.join(output_dir, "peer_graph_diagnostics.csv"), index=False)
    hop_summary = _evaluate_graph_hop_summaries(
        data_dir=data_dir,
        doc_ids=doc_ids,
        doc_embeddings=doc_embeddings,
        events=events,
        overlay=overlay,
        graph=graph,
        seed=seed,
        graph_degree=graph_degree,
        routing_policies=routing_policies,
        top_k=top_k,
        max_hops=max_hops,
        query_workers=query_workers,
        query_chunk_size=query_chunk_size,
        query_progress_every=query_progress_every,
    )
    hop_summary.to_csv(os.path.join(output_dir, "hop_curve_summary.csv"), index=False)
    distance_histogram = compute_doc_distance_histogram(
        events=events,
        overlay=overlay,
        graph=graph,
        seed=seed,
        graph_degree=graph_degree,
        max_distance=max_hops,
        query_workers=query_workers,
        query_chunk_size=query_chunk_size,
        query_progress_every=query_progress_every,
    )
    distance_histogram.to_csv(os.path.join(output_dir, "doc_distance_histogram.csv"), index=False)

    if compute_hnsw_reference:
        hnsw_summary = evaluate_hnsw_reference(
            doc_embeddings=doc_embeddings,
            doc_ids=doc_ids,
            events=events,
            seed=seed,
            top_k=top_k,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
        )
    else:
        hnsw_summary = pd.DataFrame(
            [
                {
                    "dataset_name": DEFAULT_PUBMEDQA_DATASET_NAME,
                    "method_name": "hnsw",
                    "seed": int(seed),
                    "evaluated_query_count": 0,
                    "skipped_reason": "disabled",
                }
            ]
        )
    for column, value in (
        ("centroid_label", centroid_label),
        ("centroid_mode", centroid_mode),
        ("centroid_fixed_k", int(centroid_k)),
        ("centroid_elbow_method", centroid_elbow_method if centroid_mode == "elbow" else ""),
    ):
        hnsw_summary[column] = value
    hnsw_summary.to_csv(os.path.join(output_dir, "hnsw_reference_summary.csv"), index=False)
    metrics_summary = _metrics_summary_from_hop_summary(hop_summary, hnsw_summary)
    metrics_summary.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    validate_metrics_only_output(output_dir)
    return {
        "output_dir": output_dir,
        "runtime_seconds": float(time.perf_counter() - started),
        "query_count": int(events.query_count),
        "graph_source": graph_source,
        "graph_degree": int(graph_degree),
    }


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).replace(":", ",").split(",") if item.strip()]


def optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "false", "off", "0"}:
        return None
    return int(text)


def bool_env(value: object) -> bool:
    return str(value).strip().lower() not in {"", "0", "false", "no", "off", "none"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_PUBMEDQA_DATA_DIR)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--graph-source", choices=list(DEFAULT_GRAPH_SOURCES), required=True)
    parser.add_argument("--graph-degree", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-queries", type=optional_int, default=10_000)
    parser.add_argument("--routing-policies", type=parse_csv, default=list(DEFAULT_ROUTING_POLICIES))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-hops", type=int, default=100)
    parser.add_argument("--query-workers", type=int, default=1)
    parser.add_argument("--query-chunk-size", type=int, default=50_000)
    parser.add_argument("--query-progress-every", type=int, default=500_000)
    parser.add_argument("--peer-count", type=int, default=50_000)
    parser.add_argument("--inventory-mode", choices=["alpha_train", "exact_interests"], default="alpha_train")
    parser.add_argument("--inventory-alpha", type=float, default=DEFAULT_PUBMEDQA_INVENTORY_ALPHA)
    parser.add_argument("--events-per-peer", type=int, default=DEFAULT_PUBMEDQA_EVENTS_PER_PEER)
    parser.add_argument("--train-records", type=int, default=DEFAULT_PUBMEDQA_TRAIN_RECORDS)
    parser.add_argument("--test-records", type=int, default=DEFAULT_PUBMEDQA_TEST_RECORDS)
    parser.add_argument("--docs-per-peer", type=optional_int, default=None)
    parser.add_argument("--interests-per-peer", type=int, default=10)
    parser.add_argument("--docs-per-interest", type=int, default=20)
    parser.add_argument("--semantica-leaf-size", type=int, default=50)
    parser.add_argument("--semantica-delta", type=float, default=5e-4)
    parser.add_argument("--centroid-mode", choices=["fixed", "elbow"], default="fixed")
    parser.add_argument("--centroid-k", type=int, default=1)
    parser.add_argument("--centroid-elbow-method", choices=["knee", "relative_drop", "silhouette"], default="knee")
    parser.add_argument("--centroid-elbow-max-k", type=int, default=8)
    parser.add_argument("--centroid-relative-drop-threshold", type=float, default=0.10)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=100)
    parser.add_argument("--skip-hnsw-reference", action="store_true")
    parser.add_argument("--force-overlay", action="store_true")
    parser.add_argument("--force-centroids", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.prepare_only:
        if args.inventory_mode == "alpha_train":
            overlay = generate_pubmedqa_alpha_train_overlay(
                data_dir=args.data_dir,
                cache_root=args.cache_root,
                seed=args.seed,
                peer_count=args.peer_count,
                docs_per_peer=args.docs_per_peer or args.train_records,
                events_per_peer=args.events_per_peer,
                inventory_alpha=args.inventory_alpha,
                train_records=args.train_records,
                test_records=args.test_records,
                force=args.force_overlay,
            )
        else:
            overlay = generate_pubmedqa_overlay(
                data_dir=args.data_dir,
                cache_root=args.cache_root,
                seed=args.seed,
                peer_count=args.peer_count,
                interests_per_peer=args.interests_per_peer,
                docs_per_interest=args.docs_per_interest,
                force=args.force_overlay,
            )
        manifest = _load_manifest(args.data_dir)
        doc_embeddings = _load_normalized_doc_embeddings(manifest)
        centroids = generate_pubmedqa_centroids(
            overlay=overlay,
            doc_embeddings=doc_embeddings,
            mode=args.centroid_mode,
            fixed_k=args.centroid_k,
            elbow_max_k=args.centroid_elbow_max_k,
            elbow_method=args.centroid_elbow_method,
            relative_drop_threshold=args.centroid_relative_drop_threshold,
            force=args.force_centroids,
        )
        print(
            json.dumps(
                {
                    "overlay_dir": overlay.overlay_dir,
                    "summary": overlay.summary,
                    "centroid_summary": centroids.summary,
                },
                indent=2,
            ),
            flush=True,
        )
        return
    summary = run_pubmedqa_overlay_benchmark(
        data_dir=args.data_dir,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        graph_source=args.graph_source,
        graph_degree=args.graph_degree,
        seed=args.seed,
        sample_queries=args.sample_queries,
        routing_policies=args.routing_policies,
        top_k=args.top_k,
        max_hops=args.max_hops,
        inventory_mode=args.inventory_mode,
        inventory_alpha=args.inventory_alpha,
        events_per_peer=args.events_per_peer,
        train_records=args.train_records,
        test_records=args.test_records,
        docs_per_peer=args.docs_per_peer,
        peer_count=args.peer_count,
        interests_per_peer=args.interests_per_peer,
        docs_per_interest=args.docs_per_interest,
        semantica_leaf_size=args.semantica_leaf_size,
        semantica_delta=args.semantica_delta,
        centroid_mode=args.centroid_mode,
        centroid_k=args.centroid_k,
        centroid_elbow_method=args.centroid_elbow_method,
        centroid_elbow_max_k=args.centroid_elbow_max_k,
        centroid_relative_drop_threshold=args.centroid_relative_drop_threshold,
        compute_hnsw_reference=not args.skip_hnsw_reference,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        force_overlay=args.force_overlay,
        force_centroids=args.force_centroids,
        query_workers=args.query_workers,
        query_chunk_size=args.query_chunk_size,
        query_progress_every=args.query_progress_every,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

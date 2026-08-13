"""Unified metrics-only AOL-IA, PubMedQA, and Stack Overflow retrieval benchmark.

The module deliberately keeps large embedding arrays outside result folders.  A
prepared dataset consists of compact integer manifests that refer back to the
existing MPNet arrays.  Graphs and ANN indexes are process-local and are never
serialized by the runners in this module.
"""

from __future__ import annotations

import gc
import gzip
import hashlib
import json
import math
import os
import re
import resource
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from peer_graph_comparison import (
    _CentroidTree,
    _candidate_owners_for_centroid,
    _order_centroid_records_for_tree,
    _select_k_by_knee,
    _select_k_by_relative_drop,
    allocate_centroid_contact_caps,
    select_peer_centroid_count,
)
from pubmedqa_overlay_benchmark import (
    DEFAULT_PUBMEDQA_EVENTS_PER_PEER,
    DEFAULT_PUBMEDQA_INVENTORY_ALPHA,
    DEFAULT_PUBMEDQA_TEST_RECORDS,
    DEFAULT_PUBMEDQA_TRAIN_RECORDS,
    generate_pubmedqa_alpha_train_overlay,
)


DEFAULT_SEEDS = (42, 101, 202)
DEFAULT_DATASETS = ("aol_ia", "pubmedqa", "stackoverflow")
DEFAULT_QUERY_COUNT = 100_000
DEFAULT_MAX_HOPS = 100
DEFAULT_EXPANSION_ROUNDS = 10
ADAPTIVE_CONFIGURED_MAX_K = 8
RETRIEVAL_CUTOFFS = (1, 5, 10, 100)
LOCAL_RESPONSE_K = max(RETRIEVAL_CUTOFFS)
MULTICUTOFF_PROTOCOL_ID = "top100_cutoffs_1_5_10_100_adaptive8_exactneighbors"
TREE_LEAF_CAPACITY_METRIC = "entries"
TREE_SPLIT_CENTROIDS_NORMALIZED = True
TREE_EAGER_RECURSIVE_SPLITTING = True
TREE_INSERTION_ORDER = "owner_major_peer_position_then_centroid_index"
TREE_INSERTION_MODES = ("canonical", "peer_shuffle")
TREE_CLONE_LEAF_CONTACT_BUDGET = "single_shared_contact_cap_across_all_clone_leaves"
RESULT_WIRE_BYTES_PER_DOCUMENT = 8
LEGACY_REPLACEMENT_POLICY = "legacy_rank_truncation"
HISTORICAL_STD_REPLACEMENT_POLICY = "historical_std_margin"
EXPANSION_REPLACEMENT_POLICIES = (
    LEGACY_REPLACEMENT_POLICY,
    HISTORICAL_STD_REPLACEMENT_POLICY,
)
FIRST_SEEN_ROUND0_CANDIDATE_POLICY = "first_seen_quota"
ALL_LEAF_ROUND0_CANDIDATE_POLICY = "all_source_leaf_candidates_global_greedy"
ROUND0_CANDIDATE_POLICIES = (
    FIRST_SEEN_ROUND0_CANDIDATE_POLICY,
    ALL_LEAF_ROUND0_CANDIDATE_POLICY,
)
DEFAULT_AOL_TOP_USERS = 50_000
DEFAULT_AOL_DATA_ROOT = "/var/scratch/pmneague/Low_Coord_Semantic_DHT/data/aol4foltr"
DEFAULT_AOL_LEGACY_CACHE = "/var/scratch/pmneague/Low_Coord_Semantic_DHT/cache/aol_ia_peer_graph"
DEFAULT_PUBMEDQA_DATA_ROOT = (
    "/var/scratch/pmneague/Low_Coord_Semantic_DHT/data/"
    "pubmedqa_expanded_all_mesh_alpha0p00007_q30/pubmedqa"
)
DEFAULT_CACHE_ROOT = (
    "/var/scratch/pmneague/Low_Coord_Semantic_DHT/cache/"
    "unified_aolia_pubmedqa_top10_chainhop100"
)
DEFAULT_PUBMEDQA_OVERLAY_CACHE = (
    "/var/scratch/pmneague/Low_Coord_Semantic_DHT/cache/"
    "pubmedqa_alpha0p0005_events200_train133_test67"
)
DEFAULT_STACKOVERFLOW_DATA_ROOT = (
    "/var/scratch/pmneague/Low_Coord_Semantic_DHT/data/"
    "stackoverflow_duplicate_posts_repeat2_leakage_safe"
)
DEFAULT_STACKOVERFLOW_OVERLAY_CACHE = (
    "/var/scratch/pmneague/Low_Coord_Semantic_DHT/cache/"
    "stackoverflow_duplicate_posts_repeat2_peer_overlays"
)
MANIFEST_SCHEMA_VERSION = 2


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
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional embedding matrix.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[~np.isfinite(norms) | (norms <= 0.0)] = 1.0
    return values / norms


def atomic_json(path: os.PathLike[str] | str, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    temporary.replace(path)


def peak_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.uname().sysname == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


class QueryIndex:
    """Resolve normalized query text to an existing query embedding row."""

    def __init__(self, query_keys_path: str, query_embeddings_path: str):
        self.query_keys_path = str(query_keys_path)
        self.query_embeddings_path = str(query_embeddings_path)
        keys = np.load(self.query_keys_path, mmap_mode="r", allow_pickle=False)
        self.key_to_position = {str(key): int(position) for position, key in enumerate(keys)}
        if len(self.key_to_position) != len(keys):
            raise ValueError("Query embedding keys must be unique.")
        self.embeddings = np.load(self.query_embeddings_path, mmap_mode="r")
        if len(keys) != self.embeddings.shape[0]:
            raise ValueError("Query keys and embeddings have different row counts.")

    def position(self, query_text: object) -> Optional[int]:
        return self.key_to_position.get(str(query_text))

    def rows(self, positions: Sequence[int]) -> np.ndarray:
        return normalize_rows(np.asarray(self.embeddings[np.asarray(positions, dtype=np.int64)], dtype=np.float32))


@dataclass(frozen=True)
class EmbeddingBundle:
    manifest_path: str
    doc_ids_path: str
    doc_embeddings_path: str
    query_keys_path: str
    query_embeddings_path: str
    model_name: Optional[str] = None

    @classmethod
    def load(
        cls,
        data_root: str,
        doc_embeddings_path: Optional[str] = None,
    ) -> "EmbeddingBundle":
        manifest_path = os.path.join(data_root, "embeddings", "manifest.json")
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        model = manifest.get("model")
        if model is not None and str(model) != "sentence-transformers/all-mpnet-base-v2":
            raise ValueError(f"Unified retrieval requires MPNet embeddings, found {model!r}.")

        def resolve(key: str, fallback: str) -> str:
            value = str(manifest.get(key) or os.path.join(data_root, "embeddings", fallback))
            return value

        bundle = cls(
            manifest_path=manifest_path,
            doc_ids_path=resolve("doc_keys_path", "doc_ids.npy"),
            doc_embeddings_path=(
                os.path.realpath(doc_embeddings_path)
                if doc_embeddings_path
                else resolve("doc_embeddings_path", "doc_embeddings.npy")
            ),
            query_keys_path=resolve("query_keys_path", "query_texts.npy"),
            query_embeddings_path=resolve("query_embeddings_path", "query_embeddings.npy"),
            model_name=str(model) if model is not None else None,
        )
        doc_keys = np.load(bundle.doc_ids_path, mmap_mode="r", allow_pickle=False)
        doc_embeddings = np.load(bundle.doc_embeddings_path, mmap_mode="r")
        query_keys = np.load(bundle.query_keys_path, mmap_mode="r", allow_pickle=False)
        query_embeddings = np.load(bundle.query_embeddings_path, mmap_mode="r")
        if len(doc_keys) != doc_embeddings.shape[0] or len(query_keys) != query_embeddings.shape[0]:
            raise ValueError("Embedding bundle key and vector row counts do not match.")
        if doc_embeddings.ndim != 2 or query_embeddings.ndim != 2 or doc_embeddings.shape[1] != query_embeddings.shape[1]:
            raise ValueError("Document and query embedding dimensions do not match.")
        return bundle


@dataclass
class PreparedDataset:
    dataset_name: str
    seed: int
    peer_ids: np.ndarray
    inventory_offsets: np.ndarray
    inventory_doc_indices: np.ndarray
    manifest: pd.DataFrame
    bundle: EmbeddingBundle
    validation: Dict[str, object]
    _doc_ids: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _doc_embeddings: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _query_embeddings: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _query_keys: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _train_doc_indices: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _normalized_train_doc_embeddings: Optional[np.ndarray] = field(
        default=None, init=False, repr=False
    )
    _doc_norms: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _doc_tie_ranks: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @property
    def peer_count(self) -> int:
        return int(self.peer_ids.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.manifest.shape[0])

    @property
    def doc_ids(self) -> np.ndarray:
        if self._doc_ids is None:
            self._doc_ids = np.load(self.bundle.doc_ids_path, mmap_mode="r", allow_pickle=False)
        return self._doc_ids

    @property
    def doc_embeddings(self) -> np.ndarray:
        if self._doc_embeddings is None:
            self._doc_embeddings = np.load(self.bundle.doc_embeddings_path, mmap_mode="r")
        return self._doc_embeddings

    @property
    def query_embeddings(self) -> np.ndarray:
        if self._query_embeddings is None:
            self._query_embeddings = np.load(self.bundle.query_embeddings_path, mmap_mode="r")
        return self._query_embeddings

    @property
    def query_keys(self) -> np.ndarray:
        if self._query_keys is None:
            self._query_keys = np.load(
                self.bundle.query_keys_path, mmap_mode="r", allow_pickle=False
            )
        return self._query_keys

    @property
    def train_doc_indices(self) -> np.ndarray:
        if self._train_doc_indices is None:
            self._train_doc_indices = np.unique(self.inventory_doc_indices)
        return self._train_doc_indices

    @property
    def normalized_train_doc_embeddings(self) -> np.ndarray:
        if self._normalized_train_doc_embeddings is None:
            values = np.asarray(
                self.doc_embeddings[self.train_doc_indices], dtype=np.float32
            )
            normalized = normalize_rows(values)
            normalized.flags.writeable = False
            self._normalized_train_doc_embeddings = normalized
        return self._normalized_train_doc_embeddings

    def peer_documents(self, peer_position: int) -> np.ndarray:
        start = int(self.inventory_offsets[int(peer_position)])
        end = int(self.inventory_offsets[int(peer_position) + 1])
        return self.inventory_doc_indices[start:end]

    def normalized_doc_rows(self, indices: Sequence[int]) -> np.ndarray:
        return normalize_rows(np.asarray(self.doc_embeddings[np.asarray(indices, dtype=np.int64)], dtype=np.float32))

    def normalized_query_rows(self, indices: Sequence[int]) -> np.ndarray:
        return normalize_rows(np.asarray(self.query_embeddings[np.asarray(indices, dtype=np.int64)], dtype=np.float32))

    @property
    def doc_norms(self) -> np.ndarray:
        if self._doc_norms is None:
            norms = np.empty(self.doc_embeddings.shape[0], dtype=np.float32)
            block_size = 25_000
            for start in range(0, self.doc_embeddings.shape[0], block_size):
                end = min(start + block_size, self.doc_embeddings.shape[0])
                values = np.asarray(self.doc_embeddings[start:end], dtype=np.float32)
                norms[start:end] = np.linalg.norm(values, axis=1)
            norms[~np.isfinite(norms) | (norms <= 0.0)] = 1.0
            self._doc_norms = norms
        return self._doc_norms

    @property
    def doc_tie_ranks(self) -> np.ndarray:
        if self._doc_tie_ranks is None:
            order = sorted(range(len(self.doc_ids)), key=lambda index: str(self.doc_ids[index]))
            ranks = np.empty(len(order), dtype=np.int64)
            ranks[np.asarray(order, dtype=np.int64)] = np.arange(len(order), dtype=np.int64)
            self._doc_tie_ranks = ranks
        return self._doc_tie_ranks


def _manifest_paths(cache_root: str, dataset_name: str, seed: int) -> Dict[str, str]:
    root = os.path.join(cache_root, "manifests", dataset_name, f"seed_{int(seed)}")
    return {
        "root": root,
        "queries": os.path.join(root, "query_manifest.csv.gz"),
        "inventory": os.path.join(root, "training_inventory.npz"),
        "validation": os.path.join(root, "dataset_validation.json"),
        "config": os.path.join(root, "manifest_config.json"),
    }


def _write_inventory(path: str, peer_ids: np.ndarray, inventories: Sequence[Sequence[int]]) -> None:
    offsets = np.zeros(len(inventories) + 1, dtype=np.int64)
    flattened: List[np.ndarray] = []
    for position, values in enumerate(inventories):
        array = np.unique(np.asarray(values, dtype=np.int32))
        flattened.append(array)
        offsets[position + 1] = offsets[position] + len(array)
    doc_indices = np.concatenate(flattened) if flattened else np.empty(0, dtype=np.int32)
    np.savez_compressed(path, peer_ids=np.asarray(peer_ids), offsets=offsets, doc_indices=doc_indices)


def _frame_checksum(frame: pd.DataFrame) -> str:
    canonical = frame.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _inventory_checksum(path: str) -> str:
    inventory = np.load(path, allow_pickle=False)
    digest = hashlib.sha256()
    for key in ("peer_ids", "offsets", "doc_indices"):
        values = np.ascontiguousarray(inventory[key])
        digest.update(key.encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _file_identity(path: str, *, hash_limit_bytes: int = 8 * 1024 * 1024) -> Dict[str, object]:
    real_path = os.path.realpath(path)
    stat = os.stat(real_path)
    identity: Dict[str, object] = {
        "path": real_path,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if stat.st_size <= int(hash_limit_bytes):
        digest = hashlib.sha256()
        with open(real_path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        identity["sha256"] = digest.hexdigest()
    return identity


def _file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sample_sha256(path: str, sample_count: int = 64) -> str:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    count = min(int(sample_count), int(values.shape[0]))
    positions = np.linspace(0, int(values.shape[0]) - 1, num=count, dtype=np.int64)
    sample = np.ascontiguousarray(values[positions])
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(positions.tobytes())
    digest.update(sample.tobytes())
    return digest.hexdigest()


def _embedding_bundle_identity(bundle: EmbeddingBundle) -> Dict[str, object]:
    identities = {
        "manifest": _file_identity(bundle.manifest_path),
        "doc_ids": _file_identity(bundle.doc_ids_path),
        "doc_embeddings": _file_identity(bundle.doc_embeddings_path),
        "query_keys": _file_identity(bundle.query_keys_path),
        "query_embeddings": _file_identity(bundle.query_embeddings_path),
    }
    for name, path in (
        ("doc_ids", bundle.doc_ids_path),
        ("doc_embeddings", bundle.doc_embeddings_path),
        ("query_keys", bundle.query_keys_path),
        ("query_embeddings", bundle.query_embeddings_path),
    ):
        identities[name]["deterministic_sample_sha256"] = _array_sample_sha256(path)
    return identities


def _pubmed_overlay_identity(
    cache_root: str,
    seed: int,
    inventory_alpha: float = DEFAULT_PUBMEDQA_INVENTORY_ALPHA,
) -> Dict[str, object]:
    alpha_label = f"{float(inventory_alpha):.10f}".rstrip("0").rstrip(".").replace(".", "p")
    overlay_dir = os.path.join(
        cache_root,
        f"seed_{int(seed)}__peers_50000__alpha_{alpha_label}__events_200"
        "__train_133__test_67__docs_per_peer_133",
    )
    filenames = (
        "peer_doc_indices.npy",
        "peer_test_doc_indices.npy",
        "peer_test_query_indices.npy",
        "peer_embeddings.npy",
        "owner_offsets.npy",
        "owner_peer_ids.npy",
        "peer_inventory_summary.json",
    )
    missing = [filename for filename in filenames if not os.path.exists(os.path.join(overlay_dir, filename))]
    if missing:
        return {
            "overlay_dir": os.path.realpath(overlay_dir),
            "complete": False,
            "missing_files": missing,
        }
    return {
        "overlay_dir": os.path.realpath(overlay_dir),
        "complete": True,
        "files": {
            filename: _file_identity(os.path.join(overlay_dir, filename)) for filename in filenames
        },
    }


def _stackoverflow_overlay_dir(cache_root: str, seed: int) -> str:
    candidates = (
        os.path.join(cache_root, f"seed_{int(seed)}"),
        os.path.join(
            cache_root,
            f"peer_simulation_alpha0p0005_events200_seed{int(seed)}",
        ),
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def _stackoverflow_overlay_identity(cache_root: str, seed: int) -> Dict[str, object]:
    overlay_dir = _stackoverflow_overlay_dir(cache_root, seed)
    filenames = (
        "peer_train_doc_ids.npy",
        "peer_test_doc_ids.npy",
        "peer_test_query_ids.npy",
        "summary.json",
    )
    missing = [
        filename
        for filename in filenames
        if not os.path.exists(os.path.join(overlay_dir, filename))
    ]
    if missing:
        return {
            "overlay_dir": os.path.realpath(overlay_dir),
            "complete": False,
            "missing_files": missing,
        }
    return {
        "overlay_dir": os.path.realpath(overlay_dir),
        "complete": True,
        "files": {
            filename: _file_identity(os.path.join(overlay_dir, filename))
            for filename in filenames
        },
    }


def _validate_query_self_document_mapping(
    manifest: pd.DataFrame,
    bundle: EmbeddingBundle,
    validation: Mapping[str, object],
) -> None:
    if validation.get("query_self_document_excluded", False):
        if "query_doc_index" not in manifest:
            raise RuntimeError(
                "Query self-document exclusion is enabled, but query_doc_index is missing."
            )
        query_docs = manifest["query_doc_index"].to_numpy(dtype=np.int64)
        positive = query_docs >= 0
        doc_ids = np.load(bundle.doc_ids_path, mmap_mode="r", allow_pickle=False)
        query_keys = np.load(
            bundle.query_keys_path, mmap_mode="r", allow_pickle=False
        )
        query_rows = manifest["query_embedding_index"].to_numpy(dtype=np.int64)
        if np.any(query_rows < 0) or np.any(query_rows >= len(query_keys)):
            raise RuntimeError("Query embedding indices are out of bounds.")
        doc_position_by_external_id = {
            str(doc_id): position for position, doc_id in enumerate(doc_ids.tolist())
        }
        expected_query_docs = np.fromiter(
            (
                doc_position_by_external_id.get(str(query_keys[row_position]), -1)
                for row_position in query_rows
            ),
            dtype=np.int64,
            count=len(query_rows),
        )
        expected_count = int(validation.get("query_self_document_event_count", -1))
        expected_unique = int(validation.get("query_self_document_unique_count", -1))
        if (
            np.any(query_docs < -1)
            or np.any(query_docs >= len(doc_ids))
            or int(np.count_nonzero(positive)) != expected_count
            or int(np.unique(query_docs[positive]).size) != expected_unique
        ):
            raise RuntimeError("Query self-document mapping validation failed.")
        if not np.array_equal(query_docs, expected_query_docs):
            raise RuntimeError("Query self-document external ID mapping failed.")
        targets = manifest["target_doc_index"].to_numpy(dtype=np.int64)
        if np.any(positive & (query_docs == targets)):
            raise RuntimeError("A query self-document is also its designated target.")


def _load_prepared_dataset(
    dataset_name: str,
    seed: int,
    cache_root: str,
    bundle: EmbeddingBundle,
) -> PreparedDataset:
    paths = _manifest_paths(cache_root, dataset_name, seed)
    inventory = np.load(paths["inventory"], allow_pickle=False)
    manifest = pd.read_csv(paths["queries"])
    with open(paths["validation"], encoding="utf-8") as stream:
        validation = json.load(stream)
    _validate_query_self_document_mapping(manifest, bundle, validation)
    return PreparedDataset(
        dataset_name=dataset_name,
        seed=int(seed),
        peer_ids=np.asarray(inventory["peer_ids"]),
        inventory_offsets=np.asarray(inventory["offsets"], dtype=np.int64),
        inventory_doc_indices=np.asarray(inventory["doc_indices"], dtype=np.int32),
        manifest=manifest,
        bundle=bundle,
        validation=validation,
    )


def _load_if_compatible(
    dataset_name: str,
    seed: int,
    cache_root: str,
    bundle: EmbeddingBundle,
    query_count: int,
    expected_config: Mapping[str, object],
) -> Optional[PreparedDataset]:
    paths = _manifest_paths(cache_root, dataset_name, seed)
    if not all(
        os.path.exists(paths[key]) for key in ("queries", "inventory", "validation", "config")
    ):
        return None
    prepared = _load_prepared_dataset(dataset_name, seed, cache_root, bundle)
    if prepared.query_count != int(query_count):
        raise RuntimeError(
            f"Cached {dataset_name} seed {seed} manifest has {prepared.query_count} rows, "
            f"not the requested {query_count}. Use a separate cache root or --force-manifest."
        )
    config_path = paths["config"]
    with open(config_path, encoding="utf-8") as stream:
        observed_config = json.load(stream)
    if observed_config != dict(expected_config):
        raise RuntimeError(
            f"Cached {dataset_name} seed {seed} was prepared with a different dataset configuration. "
            "Use a separate cache root or --force-manifest."
        )
    expected_checksum = str(prepared.validation.get("query_manifest_sha256", ""))
    expected_inventory = str(prepared.validation.get("training_inventory_sha256", ""))
    expected_config_digest = str(prepared.validation.get("manifest_config_sha256", ""))
    if not expected_checksum or not expected_inventory or not expected_config_digest:
        raise RuntimeError(
            f"Cached {dataset_name} seed {seed} is missing mandatory provenance checksums."
        )
    if _frame_checksum(prepared.manifest) != expected_checksum:
        raise RuntimeError(f"Cached {dataset_name} seed {seed} query manifest checksum mismatch.")
    inventory_path = _manifest_paths(cache_root, dataset_name, seed)["inventory"]
    if _inventory_checksum(inventory_path) != expected_inventory:
        raise RuntimeError(f"Cached {dataset_name} seed {seed} training inventory checksum mismatch.")
    observed_config_digest = hashlib.sha256(
        json.dumps(observed_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if observed_config_digest != expected_config_digest:
        raise RuntimeError(f"Cached {dataset_name} seed {seed} manifest config checksum mismatch.")
    if prepared.validation.get("dataset_name") != dataset_name or int(
        prepared.validation.get("seed", -1)
    ) != int(seed):
        raise RuntimeError(f"Cached {dataset_name} seed {seed} validation identity mismatch.")
    if dataset_name == "aol_ia":
        doc_to_position, canonical_positions = canonical_doc_id_positions(prepared.doc_ids)
        verify_duplicate_doc_embedding_rows(
            prepared.doc_ids,
            prepared.doc_embeddings,
            doc_to_position,
        )
        referenced_indices = np.unique(
            np.concatenate(
                [
                    prepared.inventory_doc_indices.astype(np.int64, copy=False),
                    prepared.manifest["target_doc_index"].to_numpy(dtype=np.int64),
                ]
            )
        )
        insertion_points = np.searchsorted(canonical_positions, referenced_indices)
        in_range = insertion_points < len(canonical_positions)
        canonical = np.zeros(len(referenced_indices), dtype=bool)
        canonical[in_range] = (
            canonical_positions[insertion_points[in_range]] == referenced_indices[in_range]
        )
        if not np.all(canonical):
            raise RuntimeError(
                "Cached AOL manifest or inventory references non-canonical duplicate document rows."
            )
        if prepared.validation.get("duplicate_doc_id_policy") != "canonical_first_embedding_row":
            raise RuntimeError("Cached AOL manifest is missing the canonical document-ID policy.")
    return prepared


def _uniform_manifest_sample(frame: pd.DataFrame, query_count: int, seed: int, token: str) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError(f"No eligible query events remain for {token}.")
    if int(query_count) > len(frame):
        raise RuntimeError(
            f"Requested {query_count} query events for {token}, but only {len(frame)} are eligible."
        )
    rng = np.random.default_rng(stable_int(f"query-manifest:{token}:{int(seed)}"))
    positions = np.sort(rng.choice(len(frame), size=int(query_count), replace=False))
    sampled = frame.iloc[positions].copy().reset_index(drop=True)
    sampled.insert(0, "manifest_row", np.arange(len(sampled), dtype=np.int64))
    return sampled


def canonical_doc_id_positions(doc_ids: Sequence[object]) -> Tuple[Dict[str, int], np.ndarray]:
    first_position: Dict[str, int] = {}
    for position, doc_id in enumerate(doc_ids):
        first_position.setdefault(str(doc_id), int(position))
    positions = np.asarray(sorted(first_position.values()), dtype=np.int32)
    return first_position, positions


def verify_duplicate_doc_embedding_rows(
    doc_ids: Sequence[object],
    doc_embeddings: np.ndarray,
    canonical_positions: Mapping[str, int],
) -> int:
    duplicate_count = 0
    for position, doc_id in enumerate(doc_ids):
        canonical_position = int(canonical_positions[str(doc_id)])
        if canonical_position == int(position):
            continue
        duplicate_count += 1
        canonical_bytes = np.ascontiguousarray(doc_embeddings[canonical_position]).tobytes()
        duplicate_bytes = np.ascontiguousarray(doc_embeddings[int(position)]).tobytes()
        if canonical_bytes != duplicate_bytes:
            raise RuntimeError(
                f"Duplicate document ID {doc_id!r} has conflicting embedding rows "
                f"{canonical_position} and {position}."
            )
    return int(duplicate_count)


def _sample_pubmedqa_overlay_manifest(
    overlay: object,
    *,
    query_count: int,
    seed: int,
) -> pd.DataFrame:
    test_docs = np.asarray(overlay.peer_test_doc_indices)
    test_queries = np.asarray(overlay.peer_test_query_indices)
    if test_docs.ndim != 2 or test_docs.shape != test_queries.shape:
        raise ValueError("PubMedQA held-out document/query arrays must have the same 2D shape.")
    total_events = int(test_docs.size)
    if int(query_count) > total_events:
        raise RuntimeError(f"Requested {query_count} PubMed events, but only {total_events} exist.")
    rng = np.random.default_rng(stable_int(f"query-manifest:pubmedqa:{int(seed)}"))
    flat_positions = np.sort(rng.choice(total_events, size=int(query_count), replace=False))
    events_per_peer = int(test_docs.shape[1])
    source_positions = (flat_positions // events_per_peer).astype(np.int32)
    test_offsets = (flat_positions % events_per_peer).astype(np.int32)
    target_indices = np.asarray(test_docs[source_positions, test_offsets], dtype=np.int32)
    query_indices = np.asarray(test_queries[source_positions, test_offsets], dtype=np.int32)
    availability = [
        bool(np.any(np.asarray(overlay.owners_for_doc(int(doc_index))) != int(source_peer)))
        for source_peer, doc_index in zip(source_positions.tolist(), target_indices.tolist())
    ]
    return pd.DataFrame(
        {
            "manifest_row": np.arange(int(query_count), dtype=np.int64),
            "source_event_index": flat_positions.astype(np.int64),
            "qid": [f"pmq_syn_{int(position)}" for position in flat_positions.tolist()],
            "source_peer_position": source_positions,
            "query_embedding_index": query_indices,
            "target_doc_index": target_indices,
            "target_available_in_training_network": availability,
        }
    )


def _verify_pubmedqa_overlay_source_pairs(
    bundle: EmbeddingBundle,
    data_root: str,
    manifest: pd.DataFrame,
    *,
    audit_count: int = 1_000,
) -> int:
    count = min(int(audit_count), len(manifest))
    if count <= 0:
        return 0
    positions = np.linspace(0, len(manifest) - 1, num=count, dtype=np.int64)
    audited = manifest.iloc[positions]
    query_keys = np.load(bundle.query_keys_path, mmap_mode="r", allow_pickle=False)
    doc_keys = np.load(bundle.doc_ids_path, mmap_mode="r", allow_pickle=False)
    expected = {
        (str(query_keys[int(query_index)]), str(doc_keys[int(doc_index)]))
        for query_index, doc_index in zip(
            audited["query_embedding_index"], audited["target_doc_index"]
        )
    }
    remaining = set(expected)
    path = os.path.join(data_root, "query_topics.csv.gz")
    for chunk in pd.read_csv(path, usecols=["query", "target_doc_id"], chunksize=250_000):
        if not remaining:
            break
        observed = set(zip(chunk["query"].astype(str), chunk["target_doc_id"].astype(str)))
        remaining.difference_update(observed)
    if remaining:
        raise RuntimeError(
            f"PubMedQA overlay/source mapping audit failed for {len(remaining)}/{len(expected)} pairs."
        )
    return len(expected)


def _verify_pubmedqa_overlay_peer_embeddings(
    bundle: EmbeddingBundle,
    overlay: object,
    *,
    audit_count: int = 256,
) -> int:
    count = min(int(audit_count), int(overlay.peer_doc_indices.shape[0]))
    peer_positions = np.linspace(
        0,
        int(overlay.peer_doc_indices.shape[0]) - 1,
        num=count,
        dtype=np.int64,
    )
    doc_embeddings = np.load(bundle.doc_embeddings_path, mmap_mode="r")
    expected_rows = []
    for peer_position in peer_positions:
        indices = np.asarray(overlay.peer_doc_indices[int(peer_position)], dtype=np.int64)
        expected_rows.append(normalize_vector(np.asarray(doc_embeddings[indices], dtype=np.float32).mean(axis=0)))
    expected = np.asarray(expected_rows, dtype=np.float32)
    observed = np.asarray(overlay.peer_embeddings[peer_positions], dtype=np.float32)
    similarities = np.sum(normalize_rows(expected) * normalize_rows(observed), axis=1)
    if not np.all(similarities >= 0.999):
        raise RuntimeError(
            "PubMedQA overlay peer embeddings do not match the current document embedding rows."
        )
    return int(count)


def _query_embedding_audit_targets(
    bundle: EmbeddingBundle,
    manifest: pd.DataFrame,
    *,
    audit_count: int = 64,
) -> Dict[str, str]:
    count = min(int(audit_count), len(manifest))
    positions = np.linspace(0, len(manifest) - 1, num=count, dtype=np.int64)
    audited = manifest.iloc[positions].drop_duplicates("query_embedding_index")
    query_keys = np.load(bundle.query_keys_path, mmap_mode="r", allow_pickle=False)
    doc_keys = np.load(bundle.doc_ids_path, mmap_mode="r", allow_pickle=False)
    targets: Dict[str, str] = {}
    for record in audited.itertuples(index=False):
        query_key = str(query_keys[int(record.query_embedding_index)])
        target_doc_id = str(doc_keys[int(record.target_doc_index)])
        previous = targets.setdefault(query_key, target_doc_id)
        if previous != target_doc_id:
            raise RuntimeError(
                f"PubMedQA query key {query_key!r} maps to multiple manifest targets."
            )
    return targets


def _load_pubmedqa_query_text_lookup(
    answer_targets_path: str,
    query_keys: Sequence[str],
    *,
    expected_target_by_query_key: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    remaining = set(map(str, query_keys))
    found: Dict[str, str] = {}
    with open(answer_targets_path, encoding="utf-8") as stream:
        for line in stream:
            if not remaining:
                break
            record = json.loads(line)
            query_key = str(record.get("query_key", ""))
            if query_key not in remaining:
                continue
            question = str(record.get("question", "")).strip()
            if not question:
                raise RuntimeError(f"PubMedQA source query {query_key!r} has no question text.")
            if expected_target_by_query_key is not None:
                expected_target = str(expected_target_by_query_key[query_key])
                observed_target = str(record.get("target_doc_id", ""))
                if observed_target != expected_target:
                    raise RuntimeError(
                        f"PubMedQA source query {query_key!r} target {observed_target!r} "
                        f"does not match manifest target {expected_target!r}."
                    )
            found[query_key] = question
            remaining.remove(query_key)
    if remaining:
        raise RuntimeError(
            f"PubMedQA answer source is missing {len(remaining)} embedding audit query keys."
        )
    return found


def _verify_pubmedqa_owner_index(overlay: object, doc_count: int) -> int:
    peer_documents = np.asarray(overlay.peer_doc_indices, dtype=np.int32)
    flat_docs = peer_documents.reshape(-1)
    flat_peers = np.repeat(
        np.arange(peer_documents.shape[0], dtype=np.int32),
        peer_documents.shape[1],
    )
    order = np.argsort(flat_docs, kind="stable")
    expected_offsets = np.zeros(int(doc_count) + 1, dtype=np.int64)
    expected_offsets[1:] = np.cumsum(
        np.bincount(flat_docs, minlength=int(doc_count)),
        dtype=np.int64,
    )
    observed_offsets = np.asarray(overlay.owner_offsets, dtype=np.int64)
    observed_peers = np.asarray(overlay.owner_peer_ids, dtype=np.int32)
    expected_peers = flat_peers[order]
    if not np.array_equal(observed_offsets, expected_offsets) or not np.array_equal(
        observed_peers, expected_peers
    ):
        raise RuntimeError("PubMedQA owner index does not match the peer training inventory.")
    return int(len(flat_docs))


def _verify_query_embedding_rows(
    bundle: EmbeddingBundle,
    manifest: pd.DataFrame,
    *,
    audit_count: int = 64,
    query_text_lookup: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    if bundle.model_name is None:
        return {"audit_count": 0, "status": "skipped_manifest_has_no_model_identity"}
    count = min(int(audit_count), len(manifest))
    positions = np.linspace(0, len(manifest) - 1, num=count, dtype=np.int64)
    query_indices = np.unique(
        manifest.iloc[positions]["query_embedding_index"].to_numpy(dtype=np.int64)
    )
    query_keys = np.load(bundle.query_keys_path, mmap_mode="r", allow_pickle=False)
    stored_embeddings = np.load(bundle.query_embeddings_path, mmap_mode="r")
    texts = [str(query_keys[int(index)]) for index in query_indices]
    opaque_keys = bool(texts) and all(re.fullmatch(r"pmq_q_\d+", text) for text in texts)
    independent_source_text = query_text_lookup is not None
    target_diagnostics: Dict[str, object] = {}
    if opaque_keys or independent_source_text:
        if query_text_lookup is None:
            raise RuntimeError("Opaque query keys require independent source question text.")
        missing = [text for text in texts if text not in query_text_lookup]
        if missing:
            raise RuntimeError(
                f"Missing independent source text for {len(missing)} query keys."
            )
        source_texts = [str(query_text_lookup[text]) for text in texts]
        if opaque_keys:
            audited = manifest.iloc[positions].drop_duplicates("query_embedding_index")
            observed = normalize_rows(
                np.asarray(
                    stored_embeddings[
                        audited["query_embedding_index"].to_numpy(dtype=np.int64)
                    ],
                    dtype=np.float32,
                )
            )
            doc_embeddings = np.load(bundle.doc_embeddings_path, mmap_mode="r")
            targets = normalize_rows(
                np.asarray(
                    doc_embeddings[
                        audited["target_doc_index"].to_numpy(dtype=np.int64)
                    ],
                    dtype=np.float32,
                )
            )
            similarities = np.sum(observed * targets, axis=1)
            mean_similarity = float(np.mean(similarities))
            p05_similarity = float(np.quantile(similarities, 0.05))
            target_diagnostics = {
                "mean_target_cosine_similarity": mean_similarity,
                "p05_target_cosine_similarity": p05_similarity,
            }
    else:
        source_texts = texts
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(bundle.model_name, device="cpu", local_files_only=True)
    recomputed = np.asarray(
        model.encode(
            source_texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    observed = normalize_rows(
        np.asarray(stored_embeddings[query_indices.astype(np.int64)], dtype=np.float32)
    )
    similarities = np.sum(recomputed * observed, axis=1)
    minimum_similarity = float(np.min(similarities)) if len(similarities) else 1.0
    if minimum_similarity < 0.999:
        raise RuntimeError(
            f"Query embedding row attestation failed: minimum cosine={minimum_similarity:.6f}."
        )
    return {
        "audit_count": int(len(query_indices)),
        "status": (
            "verified_by_independent_source_text_mpnet_reencoding"
            if opaque_keys or independent_source_text
            else "verified_by_mpnet_reencoding"
        ),
        "minimum_cosine_similarity": minimum_similarity,
        "query_embedding_sample_sha256": _array_sample_sha256(
            bundle.query_embeddings_path
        ),
        "model": bundle.model_name,
        **target_diagnostics,
    }


def _aol_top_peer_cache_path(cache_root: str, top_users: int) -> str:
    candidates = [
        os.path.join(cache_root, f"aol_ia_top{top_users}_by_doc_count_users_by_doc_count.pkl"),
        os.path.join(cache_root, f"aol_ia_top{top_users}_users_by_query_count.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Missing AOL top-user cache. Expected one of: " + ", ".join(candidates)
    )


def _aol_top_peer_ids(cache_root: str, top_users: int) -> np.ndarray:
    frame = pd.read_pickle(_aol_top_peer_cache_path(cache_root, top_users))
    return np.asarray(sorted(frame["AnonID"].tolist(), key=lambda value: (str(type(value)), str(value))))


def prepare_aol_manifest(
    *,
    seed: int,
    data_root: str = DEFAULT_AOL_DATA_ROOT,
    legacy_cache_root: str = DEFAULT_AOL_LEGACY_CACHE,
    cache_root: str = DEFAULT_CACHE_ROOT,
    query_count: int = DEFAULT_QUERY_COUNT,
    top_users: int = DEFAULT_AOL_TOP_USERS,
    doc_embeddings_path: Optional[str] = None,
    force: bool = False,
) -> PreparedDataset:
    paths = _manifest_paths(cache_root, "aol_ia", seed)
    bundle = EmbeddingBundle.load(data_root, doc_embeddings_path=doc_embeddings_path)
    manifest_config = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": "aol_ia",
        "seed": int(seed),
        "query_count": int(query_count),
        "top_users": int(top_users),
        "data_root": os.path.realpath(data_root),
        "embedding_bundle_identity": _embedding_bundle_identity(bundle),
        "metadata_identity": _file_identity(os.path.join(data_root, "metadata.csv")),
        "peer_cohort_identity": _file_identity(_aol_top_peer_cache_path(legacy_cache_root, top_users)),
    }
    if not force:
        prepared = _load_if_compatible(
            "aol_ia", seed, cache_root, bundle, query_count, manifest_config
        )
        if prepared is not None:
            return prepared
    os.makedirs(paths["root"], exist_ok=True)

    peer_ids = _aol_top_peer_ids(legacy_cache_root, top_users)
    peer_to_position = {str(peer): position for position, peer in enumerate(peer_ids.tolist())}
    doc_ids = np.load(bundle.doc_ids_path, mmap_mode="r", allow_pickle=False)
    doc_to_position, source_corpus_positions = canonical_doc_id_positions(doc_ids)
    duplicate_embedding_rows_verified = verify_duplicate_doc_embedding_rows(
        doc_ids,
        np.load(bundle.doc_embeddings_path, mmap_mode="r"),
        doc_to_position,
    )
    query_index = QueryIndex(bundle.query_keys_path, bundle.query_embeddings_path)

    metadata_path = os.path.join(data_root, "metadata.csv")
    chunks: List[pd.DataFrame] = []
    usecols = ["qid", "user_id", "timestamp", "query", "target_doc_id"]
    for chunk in pd.read_csv(metadata_path, usecols=usecols, chunksize=500_000):
        peer_positions = chunk["user_id"].astype(str).map(peer_to_position)
        chunk = chunk[peer_positions.notna()].copy()
        if chunk.empty:
            continue
        chunk["source_peer_position"] = peer_positions[peer_positions.notna()].astype(np.int32)
        chunk["query_embedding_index"] = chunk["query"].astype(str).map(query_index.key_to_position)
        chunk["target_doc_index"] = chunk["target_doc_id"].astype(str).map(doc_to_position)
        chunk = chunk[chunk["query_embedding_index"].notna() & chunk["target_doc_index"].notna()].copy()
        if chunk.empty:
            continue
        chunks.append(
            chunk[
                ["qid", "source_peer_position", "query_embedding_index", "target_doc_index"]
            ].copy()
        )
    if not chunks:
        raise RuntimeError("No AOL-IA top-peer events map to both query and document embeddings.")
    events = pd.concat(chunks, ignore_index=True)
    events.insert(0, "source_event_index", np.arange(len(events), dtype=np.int64))
    events["query_embedding_index"] = events["query_embedding_index"].astype(np.int64)
    events["target_doc_index"] = events["target_doc_index"].astype(np.int64)

    pair_frame = events[["source_peer_position", "target_doc_index"]].drop_duplicates()
    train_pairs: List[pd.DataFrame] = []
    test_pairs: List[pd.DataFrame] = []
    for peer_position, group in pair_frame.groupby("source_peer_position", sort=True):
        group = group.sort_values("target_doc_index").reset_index(drop=True)
        train_size = int(math.floor(len(group) * (2.0 / 3.0)))
        if train_size <= 0 or train_size >= len(group):
            continue
        rng = np.random.default_rng(stable_int(f"aol-pair-split:{seed}:{int(peer_position)}"))
        selected = set(rng.choice(len(group), size=train_size, replace=False).tolist())
        mask = np.asarray([position in selected for position in range(len(group))], dtype=bool)
        train_pairs.append(group[mask])
        test_pairs.append(group[~mask])
    if not train_pairs or not test_pairs:
        raise RuntimeError("No AOL peers have enough owner-document pairs for a 2/3-1/3 split.")
    train_pairs_frame = pd.concat(train_pairs, ignore_index=True)
    test_pairs_frame = pd.concat(test_pairs, ignore_index=True)
    eligible_peer_positions = np.sort(train_pairs_frame["source_peer_position"].unique())
    if len(eligible_peer_positions) != len(peer_ids):
        remap = {int(old): new for new, old in enumerate(eligible_peer_positions.tolist())}
        peer_ids = peer_ids[eligible_peer_positions]
        for frame in (events, train_pairs_frame, test_pairs_frame):
            frame["source_peer_position"] = frame["source_peer_position"].map(remap)
        events = events[events["source_peer_position"].notna()].copy()
        train_pairs_frame = train_pairs_frame[train_pairs_frame["source_peer_position"].notna()].copy()
        test_pairs_frame = test_pairs_frame[test_pairs_frame["source_peer_position"].notna()].copy()
        for frame in (events, train_pairs_frame, test_pairs_frame):
            frame["source_peer_position"] = frame["source_peer_position"].astype(np.int32)

    inventories: List[np.ndarray] = []
    train_by_peer = train_pairs_frame.groupby("source_peer_position")["target_doc_index"].apply(list).to_dict()
    for peer_position in range(len(peer_ids)):
        inventories.append(np.asarray(train_by_peer.get(peer_position, []), dtype=np.int32))
    _write_inventory(paths["inventory"], peer_ids, inventories)

    held_out = test_pairs_frame.assign(_test_pair=True)
    test_events = events.merge(
        held_out,
        on=["source_peer_position", "target_doc_index"],
        how="inner",
    )
    test_events = test_events.sort_values(["source_event_index", "qid"]).reset_index(drop=True)
    train_union = set(train_pairs_frame["target_doc_index"].astype(int).tolist())
    owners_by_doc = train_pairs_frame.groupby("target_doc_index")["source_peer_position"].apply(set).to_dict()
    test_events["target_available_in_training_network"] = [
        any(int(owner) != int(source) for owner in owners_by_doc.get(int(doc), set()))
        for source, doc in zip(test_events["source_peer_position"], test_events["target_doc_index"])
    ]
    keep = [
        "source_event_index",
        "qid",
        "source_peer_position",
        "query_embedding_index",
        "target_doc_index",
        "target_available_in_training_network",
    ]
    manifest = _uniform_manifest_sample(test_events[keep], query_count, seed, "aol_ia")
    query_embedding_attestation = _verify_query_embedding_rows(bundle, manifest)
    manifest.to_csv(paths["queries"], index=False, compression="gzip")
    validation: Dict[str, object] = {
        "dataset_name": "aol_ia",
        "seed": int(seed),
        "embedding_bundle_identity": manifest_config["embedding_bundle_identity"],
        "split_unit": "unique_owner_document_pair",
        "train_fraction": 2.0 / 3.0,
        "test_fraction": 1.0 / 3.0,
        "peer_count": int(len(peer_ids)),
        "training_owner_doc_pair_count": int(len(train_pairs_frame)),
        "held_out_owner_doc_pair_count": int(len(test_pairs_frame)),
        "held_out_event_count_before_sampling": int(len(test_events)),
        "query_manifest_count": int(len(manifest)),
        "query_manifest_unique_source_event_count": int(manifest["source_event_index"].nunique()),
        "query_manifest_sha256": _frame_checksum(manifest),
        "training_inventory_sha256": _inventory_checksum(paths["inventory"]),
        "unique_training_doc_count": int(len(train_union)),
        "available_query_count": int(manifest["target_available_in_training_network"].sum()),
        "query_embedding_source": "actual_query_text_via_QueryIndex",
        "query_embedding_row_attestation": query_embedding_attestation,
        "source_peer_documents_searched": False,
        "doc_ids_unique": bool(len(source_corpus_positions) == len(doc_ids)),
        "doc_embedding_row_count": int(len(doc_ids)),
        "source_corpus_unique_doc_count": int(len(source_corpus_positions)),
        "duplicate_doc_id_row_count": int(len(doc_ids) - len(source_corpus_positions)),
        "duplicate_doc_id_policy": "canonical_first_embedding_row",
        "duplicate_embedding_rows_verified_identical": int(duplicate_embedding_rows_verified),
        "owner_document_split_overlap": int(
            len(
                set(map(tuple, train_pairs_frame[["source_peer_position", "target_doc_index"]].to_numpy()))
                .intersection(
                    set(map(tuple, test_pairs_frame[["source_peer_position", "target_doc_index"]].to_numpy()))
                )
            )
        ),
        "manifest_config_sha256": hashlib.sha256(
            json.dumps(manifest_config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    if int(validation["owner_document_split_overlap"]) != 0:
        raise AssertionError("AOL train/test owner-document split leaked pairs across partitions.")
    atomic_json(paths["validation"], validation)
    atomic_json(
        paths["config"],
        manifest_config,
    )
    return _load_prepared_dataset("aol_ia", seed, cache_root, bundle)


def prepare_pubmedqa_manifest(
    *,
    seed: int,
    data_root: str = DEFAULT_PUBMEDQA_DATA_ROOT,
    cache_root: str = DEFAULT_CACHE_ROOT,
    overlay_cache_root: str = DEFAULT_PUBMEDQA_OVERLAY_CACHE,
    query_count: int = DEFAULT_QUERY_COUNT,
    inventory_alpha: float = DEFAULT_PUBMEDQA_INVENTORY_ALPHA,
    force: bool = False,
) -> PreparedDataset:
    paths = _manifest_paths(cache_root, "pubmedqa", seed)
    bundle = EmbeddingBundle.load(data_root)
    answer_targets_path = os.path.join(data_root, "answer_targets.jsonl")
    manifest_config = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": "pubmedqa",
        "seed": int(seed),
        "query_count": int(query_count),
        "overlay_alpha": float(inventory_alpha),
        "events_per_peer": 200,
        "train_records_per_peer": 133,
        "test_records_per_peer": 67,
        "data_root": os.path.realpath(data_root),
        "embedding_bundle_identity": _embedding_bundle_identity(bundle),
        "query_topics_identity": _file_identity(os.path.join(data_root, "query_topics.csv.gz")),
        "doc_topics_identity": _file_identity(os.path.join(data_root, "doc_topics.csv.gz")),
        "answer_targets_identity": _file_identity(answer_targets_path),
        "reused_overlay_cache_root": os.path.realpath(overlay_cache_root),
        "overlay_identity": _pubmed_overlay_identity(
            overlay_cache_root, seed, inventory_alpha
        ),
    }
    if not force:
        prepared = _load_if_compatible(
            "pubmedqa", seed, cache_root, bundle, query_count, manifest_config
        )
        if prepared is not None:
            return prepared
    os.makedirs(paths["root"], exist_ok=True)

    overlay = generate_pubmedqa_alpha_train_overlay(
        data_dir=data_root,
        cache_root=overlay_cache_root,
        seed=int(seed),
        peer_count=50_000,
        docs_per_peer=DEFAULT_PUBMEDQA_TRAIN_RECORDS,
        events_per_peer=DEFAULT_PUBMEDQA_EVENTS_PER_PEER,
        inventory_alpha=float(inventory_alpha),
        train_records=DEFAULT_PUBMEDQA_TRAIN_RECORDS,
        test_records=DEFAULT_PUBMEDQA_TEST_RECORDS,
        force=False,
    )
    manifest_config["overlay_identity"] = _pubmed_overlay_identity(
        overlay_cache_root, seed, inventory_alpha
    )
    expected_overlay_summary = {
        "inventory_mode": "alpha_train",
        "seed": int(seed),
        "peer_count": 50_000,
        "docs_per_peer": 133,
        "events_per_peer": 200,
        "inventory_alpha": float(inventory_alpha),
        "train_records_per_user": 133,
        "test_records_per_user": 67,
    }
    for key, expected_value in expected_overlay_summary.items():
        if overlay.summary.get(key) != expected_value:
            raise RuntimeError(
                f"PubMedQA overlay summary mismatch for {key}: "
                f"{overlay.summary.get(key)!r} != {expected_value!r}."
            )
    if os.path.realpath(str(overlay.summary.get("data_dir", ""))) != os.path.realpath(data_root):
        raise RuntimeError("PubMedQA overlay was generated from a different source data directory.")
    if overlay.peer_test_doc_indices is None or overlay.peer_test_query_indices is None:
        raise RuntimeError("PubMedQA overlay lacks its held-out event arrays.")
    peer_ids = np.arange(overlay.peer_doc_indices.shape[0], dtype=np.int64)
    doc_keys = np.load(bundle.doc_ids_path, mmap_mode="r", allow_pickle=False)
    doc_ids_unique = len(np.unique(doc_keys)) == len(doc_keys)
    if not doc_ids_unique:
        raise ValueError("PubMedQA document embedding IDs must be unique.")
    if int(overlay.summary.get("doc_count", -1)) != len(doc_keys):
        raise RuntimeError("PubMedQA overlay document count does not match the embedding bundle.")
    if (
        np.min(overlay.peer_doc_indices) < 0
        or np.min(overlay.peer_test_doc_indices) < 0
        or np.max(overlay.peer_doc_indices) >= len(doc_keys)
        or np.max(overlay.peer_test_doc_indices) >= len(doc_keys)
    ):
        raise RuntimeError("PubMedQA overlay contains document indices outside the current bundle.")
    query_embedding_rows = int(np.load(bundle.query_embeddings_path, mmap_mode="r").shape[0])
    if np.min(overlay.peer_test_query_indices) < 0 or np.max(
        overlay.peer_test_query_indices
    ) >= query_embedding_rows:
        raise RuntimeError("PubMedQA overlay contains query indices outside the current bundle.")
    overlay_summary_mtime = os.stat(
        os.path.join(overlay.overlay_dir, "peer_inventory_summary.json")
    ).st_mtime_ns
    for source_path in (bundle.doc_ids_path, bundle.query_keys_path):
        if os.stat(source_path).st_mtime_ns > overlay_summary_mtime:
            raise RuntimeError(
                "PubMedQA source key arrays are newer than the reused overlay; regenerate the overlay."
            )
    _write_inventory(paths["inventory"], peer_ids, overlay.peer_doc_indices)

    train_shape = tuple(int(value) for value in overlay.peer_doc_indices.shape)
    test_doc_shape = tuple(int(value) for value in overlay.peer_test_doc_indices.shape)
    test_query_shape = tuple(int(value) for value in overlay.peer_test_query_indices.shape)
    expected_train_shape = (50_000, 133)
    expected_test_shape = (50_000, 67)
    if train_shape != expected_train_shape or test_doc_shape != expected_test_shape or test_query_shape != expected_test_shape:
        raise RuntimeError(
            f"Unexpected PubMed overlay shapes: train={train_shape}, test_docs={test_doc_shape}, "
            f"test_queries={test_query_shape}."
        )
    total_events = int(np.prod(test_doc_shape))
    manifest = _sample_pubmedqa_overlay_manifest(
        overlay,
        query_count=query_count,
        seed=seed,
    )
    source_mapping_audit_count = _verify_pubmedqa_overlay_source_pairs(
        bundle,
        data_root,
        manifest,
    )
    peer_embedding_source_audit_count = _verify_pubmedqa_overlay_peer_embeddings(bundle, overlay)
    owner_index_pair_audit_count = _verify_pubmedqa_owner_index(overlay, len(doc_keys))
    query_audit_targets = _query_embedding_audit_targets(bundle, manifest)
    query_text_lookup = _load_pubmedqa_query_text_lookup(
        answer_targets_path,
        list(query_audit_targets),
        expected_target_by_query_key=query_audit_targets,
    )
    query_embedding_attestation = _verify_query_embedding_rows(
        bundle,
        manifest,
        query_text_lookup=query_text_lookup,
    )
    manifest.to_csv(paths["queries"], index=False, compression="gzip")
    validation: Dict[str, object] = {
        "dataset_name": "pubmedqa",
        "seed": int(seed),
        "embedding_bundle_identity": manifest_config["embedding_bundle_identity"],
        "overlay_alpha": float(inventory_alpha),
        "events_per_peer": int(DEFAULT_PUBMEDQA_EVENTS_PER_PEER),
        "train_records_per_peer": int(DEFAULT_PUBMEDQA_TRAIN_RECORDS),
        "test_records_per_peer": int(DEFAULT_PUBMEDQA_TEST_RECORDS),
        "peer_count": int(len(peer_ids)),
        "training_owner_doc_pair_count": int(overlay.peer_doc_indices.size),
        "held_out_event_count_before_sampling": int(total_events),
        "query_manifest_count": int(len(manifest)),
        "query_manifest_unique_source_event_count": int(manifest["source_event_index"].nunique()),
        "query_manifest_sha256": _frame_checksum(manifest),
        "training_inventory_sha256": _inventory_checksum(paths["inventory"]),
        "unique_training_doc_count": int(np.unique(np.asarray(overlay.peer_doc_indices)).size),
        "available_query_count": int(manifest["target_available_in_training_network"].sum()),
        "query_embedding_source": "actual_sampled_query_for_target_document",
        "query_embedding_row_attestation": query_embedding_attestation,
        "overlay_source_mapping_audit_count": int(source_mapping_audit_count),
        "overlay_peer_embedding_source_audit_count": int(peer_embedding_source_audit_count),
        "overlay_owner_index_pair_audit_count": int(owner_index_pair_audit_count),
        "query_text_source": os.path.realpath(answer_targets_path),
        "source_peer_documents_searched": False,
        "doc_ids_unique": bool(doc_ids_unique),
        "doc_embedding_row_count": int(len(doc_keys)),
        "source_corpus_unique_doc_count": int(len(doc_keys)),
        "duplicate_doc_id_row_count": 0,
        "duplicate_doc_id_policy": "not_required_ids_are_unique",
        "training_inventory_shape": list(train_shape),
        "held_out_doc_shape": list(test_doc_shape),
        "held_out_query_shape": list(test_query_shape),
        "source_train_test_overlap_count": int(
            sum(
                np.intersect1d(
                    np.asarray(overlay.peer_doc_indices[peer_position]),
                    np.asarray(overlay.peer_test_doc_indices[peer_position]),
                    assume_unique=False,
                ).size
                for peer_position in range(overlay.peer_count)
            )
        ),
        "min_unique_training_docs_per_peer": int(
            min(len(np.unique(overlay.peer_doc_indices[position])) for position in range(overlay.peer_count))
        ),
        "max_unique_training_docs_per_peer": int(
            max(len(np.unique(overlay.peer_doc_indices[position])) for position in range(overlay.peer_count))
        ),
        "training_interest_definition": overlay.summary.get("training_interest_definition"),
        "mean_training_interests_per_peer": overlay.summary.get(
            "mean_training_interests_per_peer"
        ),
        "median_training_interests_per_peer": overlay.summary.get(
            "median_training_interests_per_peer"
        ),
        "p95_training_interests_per_peer": overlay.summary.get(
            "p95_training_interests_per_peer"
        ),
        "max_training_interests_per_peer": overlay.summary.get(
            "max_training_interests_per_peer"
        ),
        "peers_at_most_8_training_interests": overlay.summary.get(
            "peers_at_most_8_training_interests"
        ),
        "peer_share_at_most_8_training_interests": overlay.summary.get(
            "peer_share_at_most_8_training_interests"
        ),
        "peers_at_least_8_training_interests": overlay.summary.get(
            "peers_at_least_8_training_interests"
        ),
        "peer_share_at_least_8_training_interests": overlay.summary.get(
            "peer_share_at_least_8_training_interests"
        ),
        "peers_above_8_training_interests": overlay.summary.get(
            "peers_above_8_training_interests"
        ),
        "peer_share_above_8_training_interests": overlay.summary.get(
            "peer_share_above_8_training_interests"
        ),
        "training_interest_count_histogram": overlay.summary.get(
            "training_interest_count_histogram"
        ),
        "manifest_config_sha256": hashlib.sha256(
            json.dumps(manifest_config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    if int(validation["source_train_test_overlap_count"]) != 0:
        raise AssertionError("PubMedQA overlay contains source-peer train/test document overlap.")
    if int(validation["min_unique_training_docs_per_peer"]) != 133 or int(
        validation["max_unique_training_docs_per_peer"]
    ) != 133:
        raise AssertionError("PubMedQA peers must each expose exactly 133 unique training documents.")
    if validation["training_interest_definition"] is not None:
        at_most = int(validation["peers_at_most_8_training_interests"])
        above = int(validation["peers_above_8_training_interests"])
        if at_most + above != len(peer_ids):
            raise AssertionError("PubMedQA training-interest groups do not cover every peer.")
        histogram = {
            int(key): int(value)
            for key, value in validation["training_interest_count_histogram"].items()
        }
        at_least = int(validation["peers_at_least_8_training_interests"])
        if at_least != sum(count for interests, count in histogram.items() if interests >= 8):
            raise AssertionError("PubMedQA >=8 training-interest count disagrees with its histogram.")
        if not np.isclose(
            float(validation["peer_share_at_most_8_training_interests"])
            + float(validation["peer_share_above_8_training_interests"]),
            1.0,
        ):
            raise AssertionError("PubMedQA training-interest shares do not sum to one.")
    elif not np.isclose(float(inventory_alpha), DEFAULT_PUBMEDQA_INVENTORY_ALPHA):
        raise RuntimeError("The requested PubMedQA alpha overlay lacks training-interest diagnostics.")
    atomic_json(paths["validation"], validation)
    atomic_json(
        paths["config"],
        manifest_config,
    )
    return _load_prepared_dataset("pubmedqa", seed, cache_root, bundle)


def _embedding_positions_for_external_ids(
    external_ids: np.ndarray,
    embedding_keys: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    keys = np.asarray(embedding_keys)
    if keys.ndim != 1 or len(np.unique(keys)) != len(keys):
        raise ValueError(f"{label} embedding keys must be a unique one-dimensional array.")
    if not np.issubdtype(keys.dtype, np.integer):
        raise ValueError(f"{label} embedding keys must contain integer Stack Overflow IDs.")
    values = np.asarray(external_ids)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    insertion = np.searchsorted(sorted_keys, values)
    in_range = insertion < len(sorted_keys)
    safe_insertion = np.minimum(insertion, len(sorted_keys) - 1)
    matched = in_range & (sorted_keys[safe_insertion] == values)
    if not np.all(matched):
        missing = np.unique(values[~matched])
        raise RuntimeError(
            f"Stack Overflow overlay references {len(missing)} {label} IDs "
            "that are absent from the embedding bundle."
        )
    return np.asarray(order[insertion], dtype=np.int32)


def _load_stackoverflow_query_text_lookup(
    path: str,
    query_keys: Sequence[str],
) -> Dict[str, str]:
    remaining = set(map(str, query_keys))
    found: Dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not remaining:
                break
            record = json.loads(line)
            query_key = str(record.get("id", ""))
            if query_key not in remaining:
                continue
            text = str(record.get("text", "")).strip()
            if not text:
                raise RuntimeError(
                    f"Stack Overflow source query {query_key!r} has no text."
                )
            found[query_key] = text
            remaining.remove(query_key)
    if remaining:
        raise RuntimeError(
            "Stack Overflow query source is missing "
            f"{len(remaining)} embedding audit query IDs."
        )
    return found


def _verify_stackoverflow_qrel_pairs(
    qrels_path: str,
    bundle: EmbeddingBundle,
    manifest: pd.DataFrame,
) -> int:
    query_keys = np.load(bundle.query_keys_path, mmap_mode="r", allow_pickle=False)
    doc_keys = np.load(bundle.doc_ids_path, mmap_mode="r", allow_pickle=False)
    expected = {
        (
            int(query_keys[int(query_index)]),
            int(doc_keys[int(doc_index)]),
        )
        for query_index, doc_index in zip(
            manifest["query_embedding_index"],
            manifest["target_doc_index"],
        )
    }
    remaining = set(expected)
    for chunk in pd.read_csv(
        qrels_path,
        usecols=["query_id", "doc_id"],
        chunksize=250_000,
    ):
        observed = set(
            zip(
                chunk["query_id"].astype(np.int64),
                chunk["doc_id"].astype(np.int64),
            )
        )
        remaining.difference_update(observed)
        if not remaining:
            break
    if remaining:
        raise RuntimeError(
            "Stack Overflow overlay/qrel mapping audit failed for "
            f"{len(remaining)}/{len(expected)} sampled pairs."
        )
    return int(len(expected))


def prepare_stackoverflow_manifest(
    *,
    seed: int,
    data_root: str = DEFAULT_STACKOVERFLOW_DATA_ROOT,
    cache_root: str = DEFAULT_CACHE_ROOT,
    overlay_cache_root: str = DEFAULT_STACKOVERFLOW_OVERLAY_CACHE,
    query_count: int = DEFAULT_QUERY_COUNT,
    force: bool = False,
) -> PreparedDataset:
    dataset_name = "stackoverflow"
    paths = _manifest_paths(cache_root, dataset_name, seed)
    bundle = EmbeddingBundle.load(data_root)
    input_root = os.path.join(data_root, "embedding_inputs")
    source_manifest_path = os.path.join(input_root, "manifest.json")
    queries_path = os.path.join(input_root, "queries.jsonl.gz")
    qrels_path = os.path.join(input_root, "qrels.csv.gz")
    overlay_dir = _stackoverflow_overlay_dir(overlay_cache_root, seed)
    overlay_identity = _stackoverflow_overlay_identity(overlay_cache_root, seed)
    if not overlay_identity.get("complete"):
        raise FileNotFoundError(
            f"Incomplete Stack Overflow peer overlay: {overlay_identity}"
        )
    manifest_config = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "seed": int(seed),
        "query_count": int(query_count),
        "overlay_alpha": 5e-4,
        "events_per_peer": 200,
        "train_records_per_peer": 133,
        "test_records_per_peer": 67,
        "data_root": os.path.realpath(data_root),
        "embedding_bundle_identity": _embedding_bundle_identity(bundle),
        "embedding_input_manifest_identity": _file_identity(source_manifest_path),
        "queries_identity": _file_identity(queries_path),
        "qrels_identity": _file_identity(qrels_path),
        "reused_overlay_cache_root": os.path.realpath(overlay_cache_root),
        "overlay_identity": overlay_identity,
    }
    if not force:
        prepared = _load_if_compatible(
            dataset_name,
            seed,
            cache_root,
            bundle,
            query_count,
            manifest_config,
        )
        if prepared is not None:
            return prepared
    os.makedirs(paths["root"], exist_ok=True)

    with open(os.path.join(overlay_dir, "summary.json"), encoding="utf-8") as stream:
        overlay_summary = json.load(stream)
    expected_summary = {
        "seed": int(seed),
        "peer_count": 50_000,
        "events_per_peer": 200,
        "train_records_per_peer": 133,
        "test_records_per_peer": 67,
        "inventory_alpha": 5e-4,
        "eligible_target_document_count": 65_836,
    }
    for key, expected_value in expected_summary.items():
        if overlay_summary.get(key) != expected_value:
            raise RuntimeError(
                f"Stack Overflow overlay summary mismatch for {key}: "
                f"{overlay_summary.get(key)!r} != {expected_value!r}."
            )

    train_doc_ids = np.load(
        os.path.join(overlay_dir, "peer_train_doc_ids.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    test_doc_ids = np.load(
        os.path.join(overlay_dir, "peer_test_doc_ids.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    test_query_ids = np.load(
        os.path.join(overlay_dir, "peer_test_query_ids.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    expected_train_shape = (50_000, 133)
    expected_test_shape = (50_000, 67)
    if (
        tuple(train_doc_ids.shape) != expected_train_shape
        or tuple(test_doc_ids.shape) != expected_test_shape
        or tuple(test_query_ids.shape) != expected_test_shape
    ):
        raise RuntimeError(
            "Unexpected Stack Overflow overlay shapes: "
            f"train={train_doc_ids.shape}, test_docs={test_doc_ids.shape}, "
            f"test_queries={test_query_ids.shape}."
        )

    doc_keys = np.load(bundle.doc_ids_path, mmap_mode="r", allow_pickle=False)
    query_keys = np.load(bundle.query_keys_path, mmap_mode="r", allow_pickle=False)
    train_doc_indices = _embedding_positions_for_external_ids(
        train_doc_ids,
        doc_keys,
        label="document",
    )
    test_doc_indices = _embedding_positions_for_external_ids(
        test_doc_ids,
        doc_keys,
        label="document",
    )
    test_query_indices = _embedding_positions_for_external_ids(
        test_query_ids,
        query_keys,
        label="query",
    )
    combined_doc_indices = np.concatenate(
        [train_doc_indices, test_doc_indices],
        axis=1,
    )
    combined_doc_indices.sort(axis=1)
    source_train_test_overlap_count = int(
        np.count_nonzero(np.diff(combined_doc_indices, axis=1) == 0)
    )
    del combined_doc_indices
    if source_train_test_overlap_count:
        raise AssertionError(
            "Stack Overflow overlay contains source-peer train/test document overlap."
        )

    peer_ids = np.arange(expected_train_shape[0], dtype=np.int64)
    _write_inventory(paths["inventory"], peer_ids, train_doc_indices)

    total_events = int(test_doc_indices.size)
    if int(query_count) > total_events:
        raise RuntimeError(
            f"Requested {query_count} Stack Overflow events, but only {total_events} exist."
        )
    rng = np.random.default_rng(
        stable_int(f"query-manifest:stackoverflow:{int(seed)}")
    )
    flat_positions = np.sort(
        rng.choice(total_events, size=int(query_count), replace=False)
    )
    events_per_peer = expected_test_shape[1]
    source_positions = (flat_positions // events_per_peer).astype(np.int32)
    test_offsets = (flat_positions % events_per_peer).astype(np.int32)
    target_indices = np.asarray(
        test_doc_indices[source_positions, test_offsets],
        dtype=np.int32,
    )
    query_indices = np.asarray(
        test_query_indices[source_positions, test_offsets],
        dtype=np.int32,
    )
    doc_position_by_external_id = {
        int(doc_id): position for position, doc_id in enumerate(doc_keys.tolist())
    }
    query_doc_indices = np.fromiter(
        (
            doc_position_by_external_id.get(int(query_keys[int(query_index)]), -1)
            for query_index in query_indices
        ),
        dtype=np.int32,
        count=len(query_indices),
    )
    training_owner_counts = np.bincount(
        train_doc_indices.reshape(-1),
        minlength=len(doc_keys),
    )
    availability = training_owner_counts[target_indices] > 0
    manifest = pd.DataFrame(
        {
            "manifest_row": np.arange(int(query_count), dtype=np.int64),
            "source_event_index": flat_positions.astype(np.int64),
            "qid": np.asarray(query_keys[query_indices]),
            "source_peer_position": source_positions,
            "query_embedding_index": query_indices,
            "query_doc_index": query_doc_indices,
            "target_doc_index": target_indices,
            "target_available_in_training_network": availability,
        }
    )
    qrel_pair_audit_count = _verify_stackoverflow_qrel_pairs(
        qrels_path,
        bundle,
        manifest,
    )
    audit_count = min(64, len(manifest))
    audit_positions = np.linspace(
        0,
        len(manifest) - 1,
        num=audit_count,
        dtype=np.int64,
    )
    audited = manifest.iloc[audit_positions].drop_duplicates(
        "query_embedding_index"
    )
    audited_query_keys = [
        str(query_keys[int(index)])
        for index in audited["query_embedding_index"].to_numpy(dtype=np.int64)
    ]
    query_text_lookup = _load_stackoverflow_query_text_lookup(
        queries_path,
        audited_query_keys,
    )
    query_embedding_attestation = _verify_query_embedding_rows(
        bundle,
        manifest,
        query_text_lookup=query_text_lookup,
    )
    manifest.to_csv(paths["queries"], index=False, compression="gzip")

    validation: Dict[str, object] = {
        "dataset_name": dataset_name,
        "seed": int(seed),
        "embedding_bundle_identity": manifest_config["embedding_bundle_identity"],
        "overlay_alpha": 5e-4,
        "events_per_peer": 200,
        "train_records_per_peer": 133,
        "test_records_per_peer": 67,
        "peer_count": int(len(peer_ids)),
        "training_owner_doc_pair_count": int(train_doc_indices.size),
        "held_out_event_count_before_sampling": int(total_events),
        "query_manifest_count": int(len(manifest)),
        "query_manifest_unique_source_event_count": int(
            manifest["source_event_index"].nunique()
        ),
        "query_manifest_sha256": _frame_checksum(manifest),
        "training_inventory_sha256": _inventory_checksum(paths["inventory"]),
        "unique_training_doc_count": int(np.unique(train_doc_indices).size),
        "available_query_count": int(np.count_nonzero(availability)),
        "query_embedding_source": "actual_sampled_duplicate_post_text",
        "query_self_document_excluded": True,
        "query_self_document_event_count": int(
            np.count_nonzero(query_doc_indices >= 0)
        ),
        "query_self_document_unique_count": int(
            np.unique(query_doc_indices[query_doc_indices >= 0]).size
        ),
        "query_embedding_row_attestation": query_embedding_attestation,
        "overlay_qrel_pair_audit_count": int(qrel_pair_audit_count),
        "query_text_source": os.path.realpath(queries_path),
        "qrels_source": os.path.realpath(qrels_path),
        "source_peer_documents_searched": False,
        "doc_ids_unique": True,
        "doc_embedding_row_count": int(len(doc_keys)),
        "source_corpus_unique_doc_count": int(len(doc_keys)),
        "duplicate_doc_id_row_count": 0,
        "duplicate_doc_id_policy": "not_required_ids_are_unique",
        "training_inventory_shape": list(expected_train_shape),
        "held_out_doc_shape": list(expected_test_shape),
        "held_out_query_shape": list(expected_test_shape),
        "source_train_test_overlap_count": int(source_train_test_overlap_count),
        "min_unique_training_docs_per_peer": 133,
        "max_unique_training_docs_per_peer": 133,
        "manifest_config_sha256": hashlib.sha256(
            json.dumps(manifest_config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    atomic_json(paths["validation"], validation)
    atomic_json(paths["config"], manifest_config)
    return _load_prepared_dataset(dataset_name, seed, cache_root, bundle)


def load_or_prepare_dataset(
    dataset_name: str,
    seed: int,
    cache_root: str = DEFAULT_CACHE_ROOT,
    query_count: int = DEFAULT_QUERY_COUNT,
    force: bool = False,
    aol_data_root: str = DEFAULT_AOL_DATA_ROOT,
    aol_doc_embeddings_path: Optional[str] = None,
    pubmedqa_data_root: str = DEFAULT_PUBMEDQA_DATA_ROOT,
    stackoverflow_data_root: str = DEFAULT_STACKOVERFLOW_DATA_ROOT,
    aol_legacy_cache: str = DEFAULT_AOL_LEGACY_CACHE,
    pubmedqa_overlay_cache: str = DEFAULT_PUBMEDQA_OVERLAY_CACHE,
    pubmedqa_inventory_alpha: float = DEFAULT_PUBMEDQA_INVENTORY_ALPHA,
    stackoverflow_overlay_cache: str = DEFAULT_STACKOVERFLOW_OVERLAY_CACHE,
) -> PreparedDataset:
    normalized = str(dataset_name).lower().replace("-", "_")
    if normalized in {"aol", "aolia", "aol_ia"}:
        return prepare_aol_manifest(
            seed=seed,
            data_root=aol_data_root,
            legacy_cache_root=aol_legacy_cache,
            cache_root=cache_root,
            query_count=query_count,
            doc_embeddings_path=aol_doc_embeddings_path,
            force=force,
        )
    if normalized in {"pubmed", "pubmedqa"}:
        return prepare_pubmedqa_manifest(
            seed=seed,
            data_root=pubmedqa_data_root,
            cache_root=cache_root,
            overlay_cache_root=pubmedqa_overlay_cache,
            query_count=query_count,
            inventory_alpha=pubmedqa_inventory_alpha,
            force=force,
        )
    if normalized in {
        "stackoverflow",
        "stack_overflow",
        "stackoverflow_duplicate_posts",
    }:
        return prepare_stackoverflow_manifest(
            seed=seed,
            data_root=stackoverflow_data_root,
            cache_root=cache_root,
            overlay_cache_root=stackoverflow_overlay_cache,
            query_count=query_count,
            force=force,
        )
    raise ValueError(f"Unsupported unified dataset: {dataset_name}")


@dataclass
class CentroidLayout:
    embeddings: np.ndarray
    offsets: np.ndarray
    cluster_doc_counts: np.ndarray
    contact_caps: np.ndarray
    method: str

    @property
    def peer_count(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def peer_centroids(self, peer_position: int) -> np.ndarray:
        start = int(self.offsets[int(peer_position)])
        end = int(self.offsets[int(peer_position) + 1])
        return self.embeddings[start:end]

    def peer_caps(self, peer_position: int) -> np.ndarray:
        start = int(self.offsets[int(peer_position)])
        end = int(self.offsets[int(peer_position) + 1])
        return self.contact_caps[start:end]

    def peer_counts(self, peer_position: int) -> np.ndarray:
        start = int(self.offsets[int(peer_position)])
        end = int(self.offsets[int(peer_position) + 1])
        return self.cluster_doc_counts[start:end]


@dataclass
class CompactGraph:
    offsets: np.ndarray
    neighbors: np.ndarray
    graph_source: str
    degree: int

    @property
    def peer_count(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def outgoing(self, peer_position: int) -> np.ndarray:
        start = int(self.offsets[int(peer_position)])
        end = int(self.offsets[int(peer_position) + 1])
        return self.neighbors[start:end]


@dataclass
class SemanticaTopology:
    centroids: CentroidLayout
    round0_graph: CompactGraph
    final_graph: CompactGraph
    checkpoint_graphs: Dict[int, CompactGraph]
    protected_random: List[np.ndarray]
    diagnostics: Dict[str, object]
    expansion_rounds: List[Dict[str, object]]

    @property
    def round10_graph(self) -> CompactGraph:
        """Compatibility alias for older round-10 callers."""
        return self.final_graph


def centroid_method_parameters(method: str) -> Tuple[str, int, str]:
    normalized = str(method).lower().replace("-", "_")
    if normalized.startswith("fixed_"):
        fixed_k = int(normalized.split("_", 1)[1])
        if fixed_k < 1 or fixed_k > 8:
            raise ValueError("Fixed centroid count must be between 1 and 8.")
        return "fixed", fixed_k, ""
    aliases = {
        "elbow_knee": "knee",
        "elbow_relative_drop": "relative_drop",
        "elbow_silhouette": "silhouette",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported centroid method: {method}")
    return "elbow", 0, aliases[normalized]


def _fit_centroids_with_counts(values: np.ndarray, selected_k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    values = normalize_rows(values)
    selected_k = max(1, min(int(selected_k), len(values)))
    if selected_k == 1:
        center = normalize_vector(np.mean(values, axis=0)).reshape(1, -1)
        return center, np.asarray([len(values)], dtype=np.int32)
    model = KMeans(n_clusters=selected_k, random_state=int(seed), n_init=10).fit(values)
    counts = np.bincount(model.labels_, minlength=selected_k)
    nonempty = np.flatnonzero(counts > 0)
    order = sorted(nonempty.tolist(), key=lambda position: (-int(counts[position]), int(position)))
    centers = normalize_rows(np.asarray(model.cluster_centers_[order], dtype=np.float32))
    return centers, np.asarray([counts[position] for position in order], dtype=np.int32)


def fit_dataset_centroids(
    dataset: PreparedDataset,
    method: str,
    degree: int,
    protected_random_fraction: float,
    seed: int,
    adaptive_max_k: int = ADAPTIVE_CONFIGURED_MAX_K,
    selected_k_by_peer: Optional[np.ndarray] = None,
    progress_every: int = 5_000,
) -> CentroidLayout:
    mode, fixed_k, elbow_method = centroid_method_parameters(method)
    protected_slots = int(math.floor(int(degree) * float(protected_random_fraction)))
    semantic_degree = int(degree) - protected_slots
    if semantic_degree <= 0:
        raise ValueError("At least one semantic graph slot is required.")
    blocks: List[np.ndarray] = []
    counts_flat: List[int] = []
    caps_flat: List[int] = []
    offsets = np.zeros(dataset.peer_count + 1, dtype=np.int64)
    for peer_position in range(dataset.peer_count):
        doc_indices = dataset.peer_documents(peer_position)
        if len(doc_indices) == 0:
            raise RuntimeError(f"Peer position {peer_position} has no training documents.")
        values = dataset.normalized_doc_rows(doc_indices)
        peer_seed = stable_int(f"centroids:{seed}:{peer_position}", bits=31)
        if mode == "fixed":
            selected_k = min(fixed_k, len(values), semantic_degree)
        elif selected_k_by_peer is not None:
            if len(selected_k_by_peer) != dataset.peer_count:
                raise ValueError("Selected-k cache length does not match the peer count.")
            selected_k = int(selected_k_by_peer[peer_position])
            if not 1 <= selected_k <= min(adaptive_max_k, len(values), semantic_degree):
                raise ValueError(f"Invalid cached selected-k={selected_k} for peer {peer_position}.")
        else:
            selected_k = select_peer_centroid_count(
                values,
                mode="elbow",
                max_k=min(adaptive_max_k, len(values), semantic_degree),
                elbow_method=elbow_method,
                relative_drop_threshold=0.10,
                seed=peer_seed,
            )
        centers, cluster_counts = _fit_centroids_with_counts(values, selected_k, peer_seed)
        caps = allocate_centroid_contact_caps(
            cluster_counts.tolist(),
            semantic_degree,
            min_quota=1,
            weight_mode="proportional",
        )
        if sum(caps) != semantic_degree or any(cap < 1 for cap in caps):
            raise AssertionError("Centroid degree allocation violated the unified protocol.")
        blocks.append(centers.astype(np.float32, copy=False))
        counts_flat.extend(int(value) for value in cluster_counts.tolist())
        caps_flat.extend(int(value) for value in caps)
        offsets[peer_position + 1] = offsets[peer_position] + len(centers)
        if progress_every and (peer_position + 1) % int(progress_every) == 0:
            print(f"Fitted centroids for {peer_position + 1}/{dataset.peer_count} peers.", flush=True)
    embeddings = np.vstack(blocks).astype(np.float32, copy=False)
    return CentroidLayout(
        embeddings=embeddings,
        offsets=offsets,
        cluster_doc_counts=np.asarray(counts_flat, dtype=np.int32),
        contact_caps=np.asarray(caps_flat, dtype=np.int32),
        method=str(method),
    )


def _selected_k_cache_dir(cache_root: str, dataset: PreparedDataset, effective_cap: int) -> Path:
    return (
        Path(cache_root)
        / "selected_k"
        / dataset.dataset_name
        / f"seed_{dataset.seed}"
        / f"cap_{int(effective_cap)}"
    )


def _select_all_adaptive_k(values: np.ndarray, max_k: int, seed: int) -> Dict[str, int]:
    values = np.asarray(values, dtype=float)
    capped = max(1, min(int(max_k), len(values)))
    if capped == 1:
        return {method: 1 for method in ("elbow_knee", "elbow_relative_drop", "elbow_silhouette")}
    inertias: List[float] = []
    silhouette_best = (1, -np.inf)
    for k in range(1, capped + 1):
        model = KMeans(n_clusters=k, random_state=int(seed), n_init=10).fit(values)
        inertias.append(float(model.inertia_))
        unique_labels = np.unique(model.labels_)
        if 2 <= len(unique_labels) < len(values):
            score = float(silhouette_score(values, model.labels_, metric="cosine"))
            if score > silhouette_best[1]:
                silhouette_best = (k, score)
    return {
        "elbow_knee": int(_select_k_by_knee(inertias)),
        "elbow_relative_drop": int(_select_k_by_relative_drop(inertias, 0.10)),
        "elbow_silhouette": int(silhouette_best[0]),
    }


def prepare_selected_k_cache(
    dataset: PreparedDataset,
    cache_root: str,
    effective_cap: int,
    *,
    force: bool = False,
) -> Dict[str, object]:
    cache_dir = _selected_k_cache_dir(cache_root, dataset, effective_cap)
    metadata_path = cache_dir / "validation.json"
    methods = ("elbow_knee", "elbow_relative_drop", "elbow_silhouette")
    paths = {method: cache_dir / f"{method}.npy" for method in methods}
    expected = {
        "protocol_id": MULTICUTOFF_PROTOCOL_ID,
        "dataset_name": dataset.dataset_name,
        "seed": int(dataset.seed),
        "effective_cap": int(effective_cap),
        "peer_count": int(dataset.peer_count),
        "training_inventory_sha256": dataset.validation.get("training_inventory_sha256"),
        "manifest_config_sha256": dataset.validation.get("manifest_config_sha256"),
        "doc_embeddings_identity": dataset.validation.get("embedding_bundle_identity", {}).get(
            "doc_embeddings"
        ),
        "dtype": "uint8",
        "selector": "normalized_kmeans_n_init_10_exact_cosine_silhouette",
    }
    if not force and metadata_path.exists() and all(path.exists() for path in paths.values()):
        observed = json.loads(metadata_path.read_text())
        observed_hashes = observed.get("array_sha256", {})
        if all(observed.get(key) == value for key, value in expected.items()) and all(
            observed_hashes.get(method) == _file_sha256(path)
            for method, path in paths.items()
        ):
            return observed

    arrays = {method: np.empty(dataset.peer_count, dtype=np.uint8) for method in methods}
    for peer_position in range(dataset.peer_count):
        docs = dataset.peer_documents(peer_position)
        values = dataset.normalized_doc_rows(docs)
        peer_seed = stable_int(f"centroids:{dataset.seed}:{peer_position}", bits=31)
        selected = _select_all_adaptive_k(values, min(int(effective_cap), len(values)), peer_seed)
        for method in methods:
            arrays[method][peer_position] = selected[method]
        if (peer_position + 1) % 5_000 == 0:
            print(f"Selected adaptive k for {peer_position + 1}/{dataset.peer_count} peers.", flush=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    for method, values in arrays.items():
        temporary = paths[method].with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        temporary.replace(paths[method])
    metadata = {
        **expected,
        "array_sha256": {
            method: _file_sha256(path) for method, path in paths.items()
        },
        "selected_k_histograms": {
            method: {
                str(int(k)): int(count)
                for k, count in zip(*np.unique(values, return_counts=True))
            }
            for method, values in arrays.items()
        },
    }
    atomic_json(metadata_path, metadata)
    return metadata


def load_selected_k_cache(
    dataset: PreparedDataset,
    cache_root: str,
    effective_cap: int,
    method: str,
) -> np.ndarray:
    cache_dir = _selected_k_cache_dir(cache_root, dataset, effective_cap)
    metadata_path = cache_dir / "validation.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing selected-k cache metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "protocol_id": MULTICUTOFF_PROTOCOL_ID,
        "dataset_name": dataset.dataset_name,
        "seed": int(dataset.seed),
        "effective_cap": int(effective_cap),
        "peer_count": int(dataset.peer_count),
        "training_inventory_sha256": dataset.validation.get("training_inventory_sha256"),
        "manifest_config_sha256": dataset.validation.get("manifest_config_sha256"),
        "doc_embeddings_identity": dataset.validation.get("embedding_bundle_identity", {}).get(
            "doc_embeddings"
        ),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Selected-k cache validation failed: {metadata_path}")
    values = np.load(cache_dir / f"{method}.npy", allow_pickle=False)
    if values.dtype != np.uint8 or values.shape != (dataset.peer_count,):
        raise ValueError(f"Malformed selected-k array for {method}: {values.shape} {values.dtype}")
    if metadata.get("array_sha256", {}).get(method) != _file_sha256(
        cache_dir / f"{method}.npy"
    ):
        raise ValueError(f"Selected-k cache checksum failed for {method}.")
    return values


def _peer_centroid_score(source_centroid: np.ndarray, candidate: int, layout: CentroidLayout) -> float:
    candidate_centroids = layout.peer_centroids(int(candidate))
    if candidate_centroids.size == 0:
        return -np.inf
    return float(np.max(candidate_centroids @ source_centroid))


def _rank_for_centroid(
    source_centroid: np.ndarray,
    candidates: Iterable[int],
    layout: CentroidLayout,
    cap: int,
    excluded: Iterable[int] = (),
) -> List[Tuple[int, float]]:
    excluded_set = set(int(value) for value in excluded)
    ranked: List[Tuple[int, float]] = []
    for candidate in sorted(set(int(value) for value in candidates)):
        if candidate in excluded_set:
            continue
        score = _peer_centroid_score(source_centroid, candidate, layout)
        if np.isfinite(score):
            ranked.append((candidate, score))
    ranked.sort(key=lambda item: (-item[1], int(item[0])))
    return ranked[: max(0, int(cap))]


def _assign_unique_candidates_to_centroids(
    candidate_scores: Sequence[Mapping[int, float]],
    caps: Sequence[int],
) -> Tuple[List[List[Tuple[int, float]]], List[int]]:
    if len(candidate_scores) != len(caps):
        raise ValueError("Candidate-score and centroid-cap lengths differ.")
    ranked_pairs = sorted(
        (
            (float(score), int(candidate), int(centroid_index))
            for centroid_index, values in enumerate(candidate_scores)
            for candidate, score in values.items()
            if np.isfinite(score)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    assigned: List[List[Tuple[int, float]]] = [[] for _ in caps]
    assigned_candidates: set[int] = set()
    for score, candidate, centroid_index in ranked_pairs:
        if candidate in assigned_candidates or len(assigned[centroid_index]) >= int(caps[centroid_index]):
            continue
        assigned[centroid_index].append((candidate, score))
        assigned_candidates.add(candidate)
    for values in assigned:
        values.sort(key=lambda item: (-item[1], item[0]))
    underfilled = [
        centroid_index
        for centroid_index, (values, cap) in enumerate(zip(assigned, caps))
        if len(values) < int(cap)
    ]
    return assigned, underfilled


def _leaf_nodes_at_distance(
    tree: _CentroidTree,
    start_leaf_nodes: Sequence[object],
    distance: int,
    visited_node_ids: set[int],
) -> List[object]:
    nodes = []
    for start_leaf in start_leaf_nodes:
        for node in tree.nodes_at_distance(start_leaf, int(distance)):
            if not hasattr(node, "records") or int(node.node_id) in visited_node_ids:
                continue
            visited_node_ids.add(int(node.node_id))
            nodes.append(node)
    return nodes


def _build_all_leaf_contacts(
    layout: CentroidLayout,
    tree: _CentroidTree,
    clone_to_leaves: Mapping[int, Sequence[object]],
    protected_random: Sequence[np.ndarray],
) -> Tuple[List[List[List[Tuple[int, float]]]], Dict[str, object]]:
    contacts: List[List[List[Tuple[int, float]]]] = []
    per_centroid_initial_candidates: List[int] = []
    per_centroid_total_candidates: List[int] = []
    per_source_initial_unique_candidates: List[int] = []
    per_source_total_unique_candidates: List[int] = []
    per_source_adjacent_unique_candidates: List[int] = []
    per_source_candidate_pairs: List[int] = []
    per_source_shared_candidates: List[int] = []
    per_source_shared_candidate_excess: List[int] = []
    per_source_cosine_checks: List[int] = []
    per_source_max_tree_distance: List[int] = []
    per_source_final_contacts: List[int] = []
    max_distance = max(0, int(tree.maximum_attained_depth) * 4)

    for peer_position in range(layout.peer_count):
        begin = int(layout.offsets[peer_position])
        end = int(layout.offsets[peer_position + 1])
        caps = [int(value) for value in layout.contact_caps[begin:end]]
        excluded = {int(peer_position)}
        excluded.update(int(value) for value in protected_random[peer_position])
        start_leaves = [
            list(clone_to_leaves.get(flat_position, ()))
            for flat_position in range(begin, end)
        ]
        if any(not values for values in start_leaves):
            raise RuntimeError(f"Peer {peer_position} has a centroid without a source leaf.")

        candidate_scores: List[Dict[int, float]] = [dict() for _ in caps]
        visited_nodes: List[set[int]] = [set() for _ in caps]
        searched_distance = np.full(len(caps), -1, dtype=np.int32)
        cosine_checks = 0

        def add_frontier(centroid_indices: Iterable[int]) -> None:
            nonlocal cosine_checks
            for centroid_index in centroid_indices:
                centroid_index = int(centroid_index)
                searched_distance[centroid_index] += 1
                distance = int(searched_distance[centroid_index])
                nodes = _leaf_nodes_at_distance(
                    tree,
                    start_leaves[centroid_index],
                    distance,
                    visited_nodes[centroid_index],
                )
                owners = {
                    int(record["OwnerID"])
                    for node in nodes
                    for record in node.records
                    if int(record["OwnerID"]) not in excluded
                }
                owners.difference_update(candidate_scores[centroid_index])
                source_flat = begin + centroid_index
                for candidate in sorted(owners):
                    score = _peer_centroid_score(
                        layout.embeddings[source_flat],
                        candidate,
                        layout,
                    )
                    cosine_checks += int(layout.offsets[candidate + 1] - layout.offsets[candidate])
                    if np.isfinite(score):
                        candidate_scores[centroid_index][candidate] = score

        add_frontier(range(len(caps)))
        initial_sets = [set(values) for values in candidate_scores]
        initial_union = set().union(*initial_sets) if initial_sets else set()
        assigned, underfilled = _assign_unique_candidates_to_centroids(candidate_scores, caps)
        while underfilled:
            if all(int(searched_distance[index]) >= max_distance for index in underfilled):
                missing = {
                    int(index): int(caps[index] - len(assigned[index]))
                    for index in underfilled
                }
                raise RuntimeError(
                    f"Unable to fill degree for peer {peer_position}; missing centroid slots {missing}."
                )
            add_frontier(
                index
                for index in underfilled
                if int(searched_distance[index]) < max_distance
            )
            assigned, underfilled = _assign_unique_candidates_to_centroids(candidate_scores, caps)

        final_sets = [set(values) for values in candidate_scores]
        final_union = set().union(*final_sets) if final_sets else set()
        occurrence_counts = Counter(
            candidate for values in final_sets for candidate in values
        )
        final_contact_count = sum(len(values) for values in assigned)
        if final_contact_count != sum(caps):
            raise AssertionError("All-leaf assignment did not fill every centroid quota.")
        if len({candidate for values in assigned for candidate, _score in values}) != final_contact_count:
            raise AssertionError("All-leaf assignment produced duplicate contacts.")

        contacts.append(assigned)
        per_centroid_initial_candidates.extend(len(values) for values in initial_sets)
        per_centroid_total_candidates.extend(len(values) for values in final_sets)
        per_source_initial_unique_candidates.append(len(initial_union))
        per_source_total_unique_candidates.append(len(final_union))
        per_source_adjacent_unique_candidates.append(len(final_union - initial_union))
        per_source_candidate_pairs.append(sum(len(values) for values in final_sets))
        per_source_shared_candidates.append(
            sum(1 for count in occurrence_counts.values() if count > 1)
        )
        per_source_shared_candidate_excess.append(
            sum(max(0, count - 1) for count in occurrence_counts.values())
        )
        per_source_cosine_checks.append(cosine_checks)
        per_source_max_tree_distance.append(int(np.max(searched_distance)))
        per_source_final_contacts.append(final_contact_count)
        if (peer_position + 1) % 10_000 == 0:
            print(
                f"Collected all-leaf contacts for {peer_position + 1}/{layout.peer_count} peers.",
                flush=True,
            )

    diagnostics = {
        "round0_candidate_policy": ALL_LEAF_ROUND0_CANDIDATE_POLICY,
        "round0_candidate_scoring": "max_source_centroid_to_candidate_peer_centroid_cosine",
        "round0_cross_centroid_assignment": (
            "global_score_desc_candidate_id_asc_centroid_index_asc_unique_owner"
        ),
        "round0_candidate_adjacency_expansion": (
            "complete_tree_distance_frontiers_for_underfilled_centroids"
        ),
        "round0_candidate_peer_centroid_cosine_checks_total": int(sum(per_source_cosine_checks)),
        **_distribution_summary(
            np.asarray(per_centroid_initial_candidates),
            "round0_initial_leaf_unique_candidates_per_centroid",
        ),
        **_distribution_summary(
            np.asarray(per_centroid_total_candidates),
            "round0_total_unique_candidates_per_centroid",
        ),
        **_distribution_summary(
            np.asarray(per_source_initial_unique_candidates),
            "round0_initial_leaf_unique_candidate_peers_per_source",
        ),
        **_distribution_summary(
            np.asarray(per_source_total_unique_candidates),
            "round0_total_unique_candidate_peers_seen_per_source",
        ),
        **_distribution_summary(
            np.asarray(per_source_adjacent_unique_candidates),
            "round0_adjacent_unique_candidate_peers_per_source",
        ),
        **_distribution_summary(
            np.asarray(per_source_candidate_pairs),
            "round0_centroid_candidate_pairs_per_source",
        ),
        **_distribution_summary(
            np.asarray(per_source_shared_candidates),
            "round0_candidates_shared_across_centroids_per_source",
        ),
        **_distribution_summary(
            np.asarray(per_source_shared_candidate_excess),
            "round0_shared_candidate_pair_excess_per_source",
        ),
        **_distribution_summary(
            np.asarray(per_source_cosine_checks),
            "round0_candidate_peer_centroid_cosine_checks_per_source",
        ),
        **_distribution_summary(
            np.asarray(per_source_max_tree_distance),
            "round0_max_tree_distance_inspected_per_source",
        ),
        **_distribution_summary(
            np.asarray(per_source_final_contacts),
            "round0_final_unique_contacts_per_source",
        ),
    }
    return contacts, diagnostics


def _protected_random_links(peer_count: int, slots: int, seed: int) -> List[np.ndarray]:
    links: List[np.ndarray] = []
    for peer_position in range(int(peer_count)):
        if slots <= 0 or peer_count <= 1:
            links.append(np.empty(0, dtype=np.int32))
            continue
        rng = np.random.default_rng(stable_int(f"protected-random:{seed}:{peer_position}"))
        raw = rng.choice(peer_count - 1, size=min(slots, peer_count - 1), replace=False)
        raw = raw + (raw >= peer_position)
        links.append(np.sort(raw.astype(np.int32)))
    return links


def _contact_lists_to_graph(
    contacts: Sequence[Sequence[Sequence[Tuple[int, float]]]],
    protected_random: Sequence[np.ndarray],
    degree: int,
    graph_source: str,
) -> CompactGraph:
    offsets = np.zeros(len(contacts) + 1, dtype=np.int64)
    rows: List[np.ndarray] = []
    for peer_position, by_centroid in enumerate(contacts):
        ordered: List[int] = []
        seen = {int(peer_position)}
        for values in by_centroid:
            for candidate, _score in values:
                candidate = int(candidate)
                if candidate not in seen:
                    seen.add(candidate)
                    ordered.append(candidate)
        for candidate in protected_random[peer_position]:
            candidate = int(candidate)
            if candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
        row = np.asarray(ordered[: int(degree)], dtype=np.int32)
        if len(row) != int(degree):
            raise RuntimeError(
                f"{graph_source} peer {peer_position} has {len(row)} unique outgoing contacts; "
                f"the requested degree is {degree}."
            )
        rows.append(row)
        offsets[peer_position + 1] = offsets[peer_position] + len(row)
    neighbors = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    return CompactGraph(offsets=offsets, neighbors=neighbors, graph_source=graph_source, degree=int(degree))


def compact_graph_sha256(graph: CompactGraph) -> str:
    """Hash exact row boundaries and neighbor order for pairing assertions."""

    digest = hashlib.sha256()
    for label, values in (
        ("offsets", np.asarray(graph.offsets, dtype="<i8")),
        ("neighbors", np.asarray(graph.neighbors, dtype="<i4")),
    ):
        digest.update(label.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def centroid_layout_sha256(layout: CentroidLayout) -> str:
    """Hash the complete centroid layout used to interpret graph contacts."""

    digest = hashlib.sha256()
    arrays = (
        ("embeddings", np.asarray(layout.embeddings, dtype="<f4")),
        ("offsets", np.asarray(layout.offsets, dtype="<i8")),
        ("cluster_doc_counts", np.asarray(layout.cluster_doc_counts, dtype="<i4")),
        ("contact_caps", np.asarray(layout.contact_caps, dtype="<i4")),
    )
    for label, values in arrays:
        digest.update(label.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(values).tobytes())
    digest.update(str(layout.method).encode("utf-8"))
    return digest.hexdigest()


def graph_diagnostics(graph: CompactGraph, reachability_sample_sources: int = 16) -> Dict[str, object]:
    degrees = np.diff(graph.offsets)
    peer_count = graph.peer_count
    try:
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components
    except ImportError as exc:
        raise ImportError("scipy is required for compact graph diagnostics.") from exc
    rows = np.repeat(np.arange(peer_count, dtype=np.int32), degrees.astype(np.int32))
    adjacency = sparse.csr_matrix(
        (np.ones(len(graph.neighbors), dtype=np.int32), (rows, graph.neighbors)),
        shape=(peer_count, peer_count),
    )
    adjacency.sum_duplicates()
    adjacency.data[:] = 1
    incoming = np.asarray(adjacency.sum(axis=0)).reshape(-1).astype(np.int64)
    incoming_total = int(incoming.sum())
    incoming_mean = float(np.mean(incoming)) if len(incoming) else 0.0
    incoming_std = float(np.std(incoming, ddof=0)) if len(incoming) else 0.0
    if incoming_total and len(incoming):
        ordered_incoming = np.sort(incoming)
        positions = np.arange(1, len(ordered_incoming) + 1, dtype=np.float64)
        incoming_gini = float(
            2.0 * np.sum(positions * ordered_incoming)
            / (len(ordered_incoming) * incoming_total)
            - (len(ordered_incoming) + 1.0) / len(ordered_incoming)
        )
        top_count = int(math.ceil(0.01 * len(incoming)))
        top_one_percent_share = float(
            np.sum(np.partition(incoming, len(incoming) - top_count)[-top_count:])
            / incoming_total
        )
    else:
        incoming_gini = 0.0
        top_count = int(math.ceil(0.01 * len(incoming))) if len(incoming) else 0
        top_one_percent_share = 0.0
    weak_count, weak_labels = connected_components(adjacency, directed=True, connection="weak")
    strong_count, strong_labels = connected_components(adjacency, directed=True, connection="strong")
    weak_sizes = np.bincount(weak_labels, minlength=weak_count)
    strong_sizes = np.bincount(strong_labels, minlength=strong_count)
    if strong_count == 1:
        reachable_fractions = [1.0]
        reachability_exact_for_all_sources = True
    else:
        from scipy.sparse.csgraph import breadth_first_order

        sample_count = min(max(1, int(reachability_sample_sources)), peer_count)
        sample_positions = np.linspace(0, max(0, peer_count - 1), sample_count, dtype=np.int64)
        reachable_fractions = [
            len(breadth_first_order(adjacency, int(source), directed=True, return_predecessors=False))
            / max(1, peer_count)
            for source in sample_positions.tolist()
        ]
        reachability_exact_for_all_sources = sample_count == peer_count
    return {
        "exact_adjacency_sha256": compact_graph_sha256(graph),
        "peer_count": int(peer_count),
        "mean_out_degree": float(np.mean(degrees)) if len(degrees) else 0.0,
        "min_out_degree": int(np.min(degrees)) if len(degrees) else 0,
        "max_out_degree": int(np.max(degrees)) if len(degrees) else 0,
        "incoming_unique_edge_definition": (
            "number_of_distinct_directed_source_peers_with_an_edge_to_the_peer"
        ),
        "incoming_unique_edge_min": int(np.min(incoming)) if len(incoming) else 0,
        "incoming_unique_edge_mean": incoming_mean,
        "incoming_unique_edge_median": (
            float(np.median(incoming)) if len(incoming) else 0.0
        ),
        "incoming_unique_edge_p95": (
            float(np.percentile(incoming, 95)) if len(incoming) else 0.0
        ),
        "incoming_unique_edge_max": int(np.max(incoming)) if len(incoming) else 0,
        "incoming_unique_edge_population_std": incoming_std,
        "incoming_unique_edge_coefficient_of_variation": (
            float(incoming_std / incoming_mean) if incoming_mean else 0.0
        ),
        "incoming_unique_edge_zero_fraction": (
            float(np.mean(incoming == 0)) if len(incoming) else 0.0
        ),
        "incoming_unique_edge_gini": incoming_gini,
        "incoming_unique_edge_top_one_percent_peer_count": top_count,
        "incoming_unique_edge_top_one_percent_share": top_one_percent_share,
        "component_count_weak": int(weak_count),
        "component_count_strong": int(strong_count),
        "largest_weak_component_fraction": float(np.max(weak_sizes) / max(1, peer_count)),
        "largest_strong_component_fraction": float(np.max(strong_sizes) / max(1, peer_count)),
        "mean_directed_reachable_peer_fraction": float(np.mean(reachable_fractions)),
        "min_directed_reachable_peer_fraction": float(np.min(reachable_fractions)),
        "max_directed_reachable_peer_fraction": float(np.max(reachable_fractions)),
        "directed_reachability_source_count": int(len(reachable_fractions)),
        "directed_reachability_exact_for_all_sources": bool(reachability_exact_for_all_sources),
        "directed_reachability_definition": "outgoing_directed_BFS_including_source",
    }


def _build_round0_contacts(
    layout: CentroidLayout,
    leaf_capacity: int,
    delta: float,
    protected_random: Sequence[np.ndarray],
    seed: int,
    candidate_policy: str = FIRST_SEEN_ROUND0_CANDIDATE_POLICY,
    tree_insertion_mode: str = "canonical",
    tree_insertion_seed: Optional[int] = None,
) -> Tuple[List[List[List[Tuple[int, float]]]], Dict[str, object]]:
    if candidate_policy not in ROUND0_CANDIDATE_POLICIES:
        raise ValueError(f"Unsupported round-0 candidate policy: {candidate_policy}")
    if tree_insertion_mode not in TREE_INSERTION_MODES:
        raise ValueError(f"Unsupported tree insertion mode: {tree_insertion_mode}")
    if tree_insertion_mode == "canonical" and tree_insertion_seed is not None:
        raise ValueError("Canonical tree insertion does not accept an insertion seed.")
    if tree_insertion_mode == "peer_shuffle" and tree_insertion_seed is None:
        raise ValueError("Peer-shuffled tree insertion requires an insertion seed.")
    start = time.perf_counter()
    tree = _CentroidTree(
        record_limit_per_leafnode=int(leaf_capacity),
        delta=float(delta),
        seed=int(seed),
        capacity_metric=TREE_LEAF_CAPACITY_METRIC,
        normalize_split_centroids=TREE_SPLIT_CENTROIDS_NORMALIZED,
        eager_recursive_splitting=TREE_EAGER_RECURSIVE_SPLITTING,
    )
    records: List[Dict[str, object]] = []
    for peer_position in range(layout.peer_count):
        begin = int(layout.offsets[peer_position])
        end = int(layout.offsets[peer_position + 1])
        for flat_position in range(begin, end):
            records.append(
                {
                    "CloneID": int(flat_position),
                    "OwnerID": int(peer_position),
                    "CentroidIndex": int(flat_position - begin),
                    "embedding": layout.embeddings[flat_position],
                }
            )
    ordered_records = _order_centroid_records_for_tree(
        records,
        tree_insertion_seed=(
            int(tree_insertion_seed) if tree_insertion_mode == "peer_shuffle" else None
        ),
    )
    insertion_digest = hashlib.sha256()
    for record in ordered_records:
        insertion_digest.update(
            np.asarray(
                [int(record["OwnerID"]), int(record["CentroidIndex"])],
                dtype="<i8",
            ).tobytes()
        )
    for index, record in enumerate(ordered_records, start=1):
        tree.position_record(record)
        if index % 25_000 == 0:
            print(f"Inserted {index}/{len(records)} centroid entries into the tree.", flush=True)

    leaves = tree.get_all_leaf_nodes()
    clone_to_leaves: Dict[int, List[object]] = defaultdict(list)
    for leaf in leaves:
        for record in leaf.records:
            clone_to_leaves[int(record["CloneID"])].append(leaf)
    contact_diagnostics: Dict[str, object] = {
        "round0_candidate_policy": candidate_policy,
    }
    if candidate_policy == ALL_LEAF_ROUND0_CANDIDATE_POLICY:
        contacts, contact_diagnostics = _build_all_leaf_contacts(
            layout,
            tree,
            clone_to_leaves,
            protected_random,
        )
    else:
        contacts = []
        valid_owners = set(range(layout.peer_count))
        for peer_position in range(layout.peer_count):
            begin = int(layout.offsets[peer_position])
            end = int(layout.offsets[peer_position + 1])
            seen = {peer_position}
            seen.update(int(value) for value in protected_random[peer_position])
            by_centroid: List[List[Tuple[int, float]]] = []
            for flat_position in range(begin, end):
                cap = int(layout.contact_caps[flat_position])
                candidates = _candidate_owners_for_centroid(
                    tree=tree,
                    start_leaf_nodes=clone_to_leaves.get(flat_position, []),
                    owner=peer_position,
                    contact_cap=cap,
                    valid_owners=valid_owners,
                    seen_owners=set(seen),
                )
                ranked = _rank_for_centroid(
                    layout.embeddings[flat_position],
                    candidates,
                    layout,
                    cap,
                    excluded={peer_position},
                )
                seen.update(candidate for candidate, _score in ranked)
                by_centroid.append(ranked)
            contacts.append(by_centroid)
            if (peer_position + 1) % 10_000 == 0:
                print(
                    f"Collected tree contacts for {peer_position + 1}/{layout.peer_count} peers.",
                    flush=True,
                )

    leaf_entry_counts = np.asarray([len(leaf.records) for leaf in leaves], dtype=np.int64)
    leaf_distinct_point_counts = np.asarray(
        [leaf.distinct_point_count for leaf in leaves], dtype=np.int64
    )
    leaf_unique_peer_counts = np.asarray(
        [len({int(record["OwnerID"]) for record in leaf.records}) for leaf in leaves], dtype=np.int64
    )
    total_leaf_entries = int(leaf_entry_counts.sum())
    diagnostics: Dict[str, object] = {
        **contact_diagnostics,
        "leaf_capacity_metric": "inserted_centroid_or_clone_entries",
        "tree_split_centroids_normalized": bool(TREE_SPLIT_CENTROIDS_NORMALIZED),
        "tree_eager_recursive_splitting": bool(TREE_EAGER_RECURSIVE_SPLITTING),
        "tree_insertion_order": TREE_INSERTION_ORDER,
        "tree_insertion_mode": tree_insertion_mode,
        "tree_insertion_seed": (
            int(tree_insertion_seed) if tree_insertion_seed is not None else None
        ),
        "tree_insertion_sequence_sha256": insertion_digest.hexdigest(),
        "tree_clone_leaf_contact_budget": TREE_CLONE_LEAF_CONTACT_BUDGET,
        "leaf_capacity": int(leaf_capacity),
        "leaf_count": int(len(leaves)),
        "tree_max_depth": int(tree.maximum_attained_depth),
        "split_count": int(sum(1 for node in tree.nodes.values() if hasattr(node, "left_child"))),
        "original_centroid_entry_count": int(len(records)),
        "leaf_entry_count_total": total_leaf_entries,
        "clone_entry_count": int(total_leaf_entries - len(records)),
        "leaf_entry_count_min": int(leaf_entry_counts.min()) if len(leaves) else 0,
        "leaf_entry_count_mean": float(leaf_entry_counts.mean()) if len(leaves) else 0.0,
        "leaf_entry_count_p50": float(np.percentile(leaf_entry_counts, 50)) if len(leaves) else 0.0,
        "leaf_entry_count_p95": float(np.percentile(leaf_entry_counts, 95)) if len(leaves) else 0.0,
        "leaf_entry_count_max": int(leaf_entry_counts.max()) if len(leaves) else 0,
        "failed_split_leaf_count": int(sum(leaf.failed_split_size is not None for leaf in leaves)),
        "leaf_distinct_point_count_min": (
            int(leaf_distinct_point_counts.min()) if len(leaves) else 0
        ),
        "leaf_distinct_point_count_mean": (
            float(leaf_distinct_point_counts.mean()) if len(leaves) else 0.0
        ),
        "leaf_distinct_point_count_p50": (
            float(np.percentile(leaf_distinct_point_counts, 50)) if len(leaves) else 0.0
        ),
        "leaf_distinct_point_count_p95": (
            float(np.percentile(leaf_distinct_point_counts, 95)) if len(leaves) else 0.0
        ),
        "leaf_distinct_point_count_max": (
            int(leaf_distinct_point_counts.max()) if len(leaves) else 0
        ),
        "over_capacity_leaf_count": int(np.sum(leaf_entry_counts > int(leaf_capacity))),
        "maximum_leaf_capacity_overflow": int(
            max(0, int(leaf_entry_counts.max()) - int(leaf_capacity))
        ) if len(leaves) else 0,
        "leaf_unique_peer_count_min": int(leaf_unique_peer_counts.min()) if len(leaves) else 0,
        "leaf_unique_peer_count_mean": float(leaf_unique_peer_counts.mean()) if len(leaves) else 0.0,
        "leaf_unique_peer_count_p50": (
            float(np.percentile(leaf_unique_peer_counts, 50)) if len(leaves) else 0.0
        ),
        "leaf_unique_peer_count_p95": (
            float(np.percentile(leaf_unique_peer_counts, 95)) if len(leaves) else 0.0
        ),
        "leaf_unique_peer_count_max": int(leaf_unique_peer_counts.max()) if len(leaves) else 0,
        "tree_construction_runtime_seconds": float(time.perf_counter() - start),
    }
    return contacts, diagnostics


def _replace_centroid_contacts(
    contacts: MutableMapping[int, List[Tuple[int, float]]] | List[List[Tuple[int, float]]],
    centroid_index: int,
    proposals: Sequence[Tuple[int, float]],
    cap: int,
    globally_excluded: Iterable[int],
) -> int:
    current = list(contacts[centroid_index])
    before = {int(candidate) for candidate, _score in current}
    excluded = set(int(value) for value in globally_excluded)
    combined = current + [
        (int(candidate), float(score))
        for candidate, score in proposals
        if int(candidate) not in excluded or int(candidate) in before
    ]
    best_by_candidate: Dict[int, float] = {}
    for candidate, score in combined:
        best_by_candidate[candidate] = max(score, best_by_candidate.get(candidate, -np.inf))
    ranked = sorted(best_by_candidate.items(), key=lambda item: (-item[1], item[0]))[: int(cap)]
    contacts[centroid_index] = ranked
    after = {int(candidate) for candidate, _score in ranked}
    return len(after - before)


def _replace_centroid_contacts_with_margin(
    contacts: MutableMapping[int, List[Tuple[int, float]]] | List[List[Tuple[int, float]]],
    centroid_index: int,
    proposals: Sequence[Tuple[int, float, str]],
    cap: int,
    globally_excluded: Iterable[int],
    tolerance: float,
) -> Dict[str, object]:
    """Apply one synchronous, deterministic batch of margin-gated replacements."""
    current = [(int(candidate), float(score)) for candidate, score in contacts[centroid_index]]
    if len(current) != int(cap):
        raise RuntimeError(
            f"Centroid {centroid_index} has {len(current)} contacts before replacement; expected {cap}."
        )
    current_ids = {candidate for candidate, _score in current}
    excluded = set(int(value) for value in globally_excluded)
    source_order = {"semantic": 0, "random_channel": 1}
    best_by_candidate: Dict[int, Tuple[float, str]] = {}
    for candidate, score, source in proposals:
        candidate = int(candidate)
        score = float(score)
        source = str(source)
        if candidate in current_ids or candidate in excluded or not np.isfinite(score):
            continue
        previous = best_by_candidate.get(candidate)
        if previous is None or score > previous[0] or (
            score == previous[0]
            and source_order.get(source, 99) < source_order.get(previous[1], 99)
        ):
            best_by_candidate[candidate] = (score, source)

    candidates = sorted(
        (
            (candidate, score, source)
            for candidate, (score, source) in best_by_candidate.items()
        ),
        key=lambda item: (-item[1], item[0], source_order.get(item[2], 99)),
    )
    # For equal incumbent scores, evict the larger peer ID first so the stable
    # lower-ID tie break is preserved.
    incumbents = sorted(current, key=lambda item: (item[1], -item[0]))
    paired_count = min(len(candidates), len(incumbents))
    accepted: List[Tuple[int, float, str]] = []
    removed_ids: set[int] = set()
    strict_rejections = 0
    margin_rejections = 0
    accepted_by_source = Counter()
    for (candidate, candidate_score, source), (incumbent, incumbent_score) in zip(
        candidates[:paired_count], incumbents[:paired_count]
    ):
        if candidate_score <= incumbent_score:
            strict_rejections += 1
            continue
        if candidate_score <= incumbent_score + float(tolerance):
            margin_rejections += 1
            continue
        accepted.append((candidate, candidate_score, source))
        removed_ids.add(incumbent)
        accepted_by_source[source] += 1

    retained = [(candidate, score) for candidate, score in current if candidate not in removed_ids]
    retained.extend((candidate, score) for candidate, score, _source in accepted)
    retained.sort(key=lambda item: (-item[1], item[0]))
    if len(retained) != int(cap) or len({candidate for candidate, _score in retained}) != int(cap):
        raise AssertionError("Historical-margin replacement did not preserve the centroid contact cap.")
    contacts[centroid_index] = retained
    return {
        "eligible_candidates": int(len(candidates)),
        "paired_candidates": int(paired_count),
        "capacity_rejections": int(len(candidates) - paired_count),
        "strict_rejections": int(strict_rejections),
        "margin_rejections": int(margin_rejections),
        "accepted_contacts": int(len(accepted)),
        "accepted_by_source": dict(accepted_by_source),
    }


def _update_running_history(
    flat_centroid: int,
    candidate: int,
    score: float,
    seen_candidates: Sequence[set[int]],
    counts: np.ndarray,
    means: np.ndarray,
    m2: np.ndarray,
) -> bool:
    candidate = int(candidate)
    seen = seen_candidates[int(flat_centroid)]
    if candidate in seen:
        return False
    seen.add(candidate)
    old_count = int(counts[int(flat_centroid)])
    new_count = old_count + 1
    delta = float(score) - float(means[int(flat_centroid)])
    means[int(flat_centroid)] += delta / new_count
    m2[int(flat_centroid)] += delta * (float(score) - float(means[int(flat_centroid)]))
    counts[int(flat_centroid)] = new_count
    return True


def _distribution_summary(values: np.ndarray, prefix: str) -> Dict[str, object]:
    values = np.asarray(values)
    if values.size == 0:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def _historical_margin_snapshot(
    counts: np.ndarray,
    m2: np.ndarray,
    multiplier: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=np.int32)
    m2 = np.asarray(m2, dtype=np.float64)
    eligible = counts >= 3
    standard_deviations = np.zeros_like(m2)
    standard_deviations[eligible] = np.sqrt(
        np.maximum(0.0, m2[eligible] / (counts[eligible] - 1))
    )
    margins = float(multiplier) * standard_deviations
    return margins, eligible, standard_deviations


def _contact_state_diagnostics(
    initial_sets: Sequence[Sequence[set[int]]],
    initial_peer_sets: Sequence[set[int]],
    initial_score_mean: float,
    contacts: Sequence[Sequence[Sequence[Tuple[int, float]]]],
) -> Dict[str, object]:
    initial_centroid_total = sum(len(values) for by_centroid in initial_sets for values in by_centroid)
    retained_centroid_total = 0
    initial_outgoing_total = sum(len(values) for values in initial_peer_sets)
    retained_outgoing_total = 0
    for initial_by_centroid, initial_union, current_by_centroid in zip(
        initial_sets, initial_peer_sets, contacts
    ):
        current_union: set[int] = set()
        for initial_values, current_values in zip(initial_by_centroid, current_by_centroid):
            current_ids = {int(candidate) for candidate, _score in current_values}
            retained_centroid_total += len(initial_values & current_ids)
            current_union.update(current_ids)
        retained_outgoing_total += len(initial_union & current_union)
    current_score_array = np.fromiter(
        (
            float(score)
            for by_centroid in contacts
            for values in by_centroid
            for _candidate, score in values
        ),
        dtype=np.float64,
    )
    current_mean = float(np.mean(current_score_array)) if current_score_array.size else 0.0
    return {
        "round0_centroid_contact_retention_fraction": float(
            retained_centroid_total / max(1, initial_centroid_total)
        ),
        "round0_outgoing_edge_retention_fraction": float(
            retained_outgoing_total / max(1, initial_outgoing_total)
        ),
        "semantic_edge_score_mean": current_mean,
        "semantic_edge_score_p50": (
            float(np.percentile(current_score_array, 50)) if current_score_array.size else 0.0
        ),
        "semantic_edge_score_p95": (
            float(np.percentile(current_score_array, 95)) if current_score_array.size else 0.0
        ),
        "semantic_edge_score_mean_delta_from_round0": float(current_mean - initial_score_mean),
    }


def _apply_historical_proposals_for_peer(
    contacts: List[List[List[Tuple[int, float]]]],
    peer_position: int,
    grouped_proposals: Mapping[int, Sequence[Tuple[int, float, str]]],
    protected_random: Sequence[np.ndarray],
    layout: CentroidLayout,
    round_margins: np.ndarray,
) -> Counter:
    counts = Counter()
    for centroid_index in sorted(grouped_proposals):
        all_existing = {
            int(candidate)
            for local_index, centroid_values in enumerate(contacts[peer_position])
            if local_index != centroid_index
            for candidate, _score in centroid_values
        }
        all_existing.update(int(candidate) for candidate in protected_random[peer_position])
        all_existing.add(int(peer_position))
        flat_position = int(layout.offsets[peer_position]) + int(centroid_index)
        outcome = _replace_centroid_contacts_with_margin(
            contacts[peer_position],
            int(centroid_index),
            grouped_proposals[int(centroid_index)],
            int(layout.contact_caps[flat_position]),
            all_existing,
            float(round_margins[flat_position]),
        )
        for name in (
            "eligible_candidates",
            "paired_candidates",
            "capacity_rejections",
            "strict_rejections",
            "margin_rejections",
            "accepted_contacts",
        ):
            counts[f"replacement_{name}"] += int(outcome[name])
        for source_name, count in dict(outcome["accepted_by_source"]).items():
            counts[f"{source_name}_accepted_contacts"] += int(count)
    return counts


def expand_semantica_contacts(
    initial_contacts: Sequence[Sequence[Sequence[Tuple[int, float]]]],
    layout: CentroidLayout,
    protected_random: Sequence[np.ndarray],
    rounds: int,
    seed: int,
    replacement_policy: str = LEGACY_REPLACEMENT_POLICY,
    replacement_tolerance_multiplier: float = 0.0,
    checkpoint_callback: Optional[
        Callable[[int, Sequence[Sequence[Sequence[Tuple[int, float]]]]], None]
    ] = None,
) -> Tuple[List[List[List[Tuple[int, float]]]], List[Dict[str, object]]]:
    replacement_policy = str(replacement_policy)
    if replacement_policy not in EXPANSION_REPLACEMENT_POLICIES:
        raise ValueError(f"Unsupported expansion replacement policy: {replacement_policy}")
    if float(replacement_tolerance_multiplier) < 0.0:
        raise ValueError("Replacement tolerance multiplier must be non-negative.")
    contacts = [[list(values) for values in by_centroid] for by_centroid in initial_contacts]
    initial_sets = [
        [{int(candidate) for candidate, _score in values} for values in by_centroid]
        for by_centroid in initial_contacts
    ]
    initial_peer_sets = [
        set().union(*by_centroid) if by_centroid else set() for by_centroid in initial_sets
    ]
    initial_scores = np.fromiter(
        (
            float(score)
            for by_centroid in initial_contacts
            for values in by_centroid
            for _candidate, score in values
        ),
        dtype=np.float64,
    )
    initial_score_mean = float(np.mean(initial_scores)) if initial_scores.size else 0.0
    asked = [[set() for _values in by_centroid] for by_centroid in initial_contacts]
    asked_random = [set() for _ in initial_contacts]
    diagnostics: List[Dict[str, object]] = []
    cumulative = Counter()
    history_seen: Optional[List[set[int]]] = None
    history_counts: Optional[np.ndarray] = None
    history_means: Optional[np.ndarray] = None
    history_m2: Optional[np.ndarray] = None
    if replacement_policy == HISTORICAL_STD_REPLACEMENT_POLICY:
        centroid_count = int(layout.embeddings.shape[0])
        history_seen = [set() for _ in range(centroid_count)]
        history_counts = np.zeros(centroid_count, dtype=np.int32)
        history_means = np.zeros(centroid_count, dtype=np.float64)
        history_m2 = np.zeros(centroid_count, dtype=np.float64)
        for peer_position, by_centroid in enumerate(initial_contacts):
            for centroid_index, values in enumerate(by_centroid):
                flat_centroid = int(layout.offsets[peer_position]) + int(centroid_index)
                for candidate, score in values:
                    _update_running_history(
                        flat_centroid,
                        int(candidate),
                        float(score),
                        history_seen,
                        history_counts,
                        history_means,
                        history_m2,
                    )
    diagnostic_fields = (
        "semantic_messages",
        "semantic_new_contact_requests",
        "semantic_initial_contact_requests",
        "semantic_returned_ids",
        "semantic_peer_centroid_cosine_checks",
        "semantic_accepted_contacts",
        "random_channel_messages",
        "random_channel_returned_ids",
        "random_channel_peer_centroid_cosine_checks",
        "random_channel_accepted_contacts",
        "replacement_eligible_candidates",
        "replacement_paired_candidates",
        "replacement_capacity_rejections",
        "replacement_strict_rejections",
        "replacement_margin_rejections",
        "replacement_accepted_contacts",
    )
    for round_index in range(1, int(rounds) + 1):
        round_counts = Counter()
        peak_buffered_replacement_proposals = 0
        round_margins = np.zeros(int(layout.embeddings.shape[0]), dtype=np.float64)
        history_start_diagnostics: Dict[str, object] = {}
        if replacement_policy == HISTORICAL_STD_REPLACEMENT_POLICY:
            assert history_counts is not None and history_m2 is not None
            round_margins, history_eligible, history_std = _historical_margin_snapshot(
                history_counts,
                history_m2,
                replacement_tolerance_multiplier,
            )
            active_tolerance = history_eligible & (float(replacement_tolerance_multiplier) > 0.0)
            positive_tolerance = active_tolerance & (round_margins > 0.0)
            history_start_diagnostics = {
                "round_history_eligible_centroids": int(np.count_nonzero(history_eligible)),
                "round_active_tolerance_centroids": int(np.count_nonzero(active_tolerance)),
                "round_positive_tolerance_centroids": int(np.count_nonzero(positive_tolerance)),
                **_distribution_summary(history_counts, "round_start_occupant_history_size"),
                **_distribution_summary(
                    round_margins[active_tolerance], "round_replacement_tolerance"
                ),
            }
        snapshot_neighbors = []
        for peer_position, by_centroid in enumerate(contacts):
            values = {
                int(candidate)
                for centroid_values in by_centroid
                for candidate, _score in centroid_values
            }
            values.update(int(candidate) for candidate in protected_random[peer_position])
            values.discard(peer_position)
            snapshot_neighbors.append(sorted(values))
        semantic_proposals: List[Tuple[int, int, List[Tuple[int, float]]]] = []
        random_proposals: List[Tuple[int, int, List[Tuple[int, float]]]] = []
        for peer_position, by_centroid in enumerate(contacts):
            owner_existing = set(snapshot_neighbors[peer_position])
            peer_historical_proposals: Dict[int, List[Tuple[int, float, str]]] = defaultdict(list)
            for centroid_index, current in enumerate(by_centroid):
                current_ids = [int(candidate) for candidate, _score in current]
                new_tier = [
                    candidate
                    for candidate in current_ids
                    if candidate not in initial_sets[peer_position][centroid_index]
                    and candidate not in asked[peer_position][centroid_index]
                ]
                initial_tier = [
                    candidate
                    for candidate in current_ids
                    if candidate in initial_sets[peer_position][centroid_index]
                    and candidate not in asked[peer_position][centroid_index]
                ]
                tier = new_tier if new_tier else initial_tier
                if not tier:
                    continue
                rng = np.random.default_rng(
                    stable_int(f"semantic-referrer:{seed}:{round_index}:{peer_position}:{centroid_index}")
                )
                referrer = int(tier[int(rng.integers(0, len(tier)))])
                asked[peer_position][centroid_index].add(referrer)
                round_counts["semantic_messages"] += 1
                round_counts["semantic_new_contact_requests" if new_tier else "semantic_initial_contact_requests"] += 1
                candidates = [
                    candidate
                    for candidate in snapshot_neighbors[referrer]
                    if candidate != peer_position and candidate not in owner_existing
                ]
                round_counts["semantic_returned_ids"] += len(candidates)
                source_flat = int(layout.offsets[peer_position]) + centroid_index
                proposals = []
                for candidate in candidates:
                    score = _peer_centroid_score(layout.embeddings[source_flat], candidate, layout)
                    round_counts["semantic_peer_centroid_cosine_checks"] += int(
                        layout.offsets[candidate + 1] - layout.offsets[candidate]
                    )
                    if np.isfinite(score):
                        proposals.append((candidate, score))
                if replacement_policy == HISTORICAL_STD_REPLACEMENT_POLICY:
                    peer_historical_proposals[int(centroid_index)].extend(
                        (int(candidate), float(score), "semantic") for candidate, score in proposals
                    )
                else:
                    semantic_proposals.append((peer_position, centroid_index, proposals))

            remaining_random = [
                int(candidate)
                for candidate in protected_random[peer_position]
                if int(candidate) not in asked_random[peer_position]
            ]
            if remaining_random:
                rng = np.random.default_rng(
                    stable_int(f"random-channel-referrer:{seed}:{round_index}:{peer_position}")
                )
                referrer = remaining_random[int(rng.integers(0, len(remaining_random)))]
                asked_random[peer_position].add(referrer)
                round_counts["random_channel_messages"] += 1
                candidates = [
                    candidate
                    for candidate in snapshot_neighbors[referrer]
                    if candidate != peer_position and candidate not in owner_existing
                ]
                round_counts["random_channel_returned_ids"] += len(candidates)
                for candidate in candidates:
                    best: Optional[Tuple[float, int]] = None
                    begin = int(layout.offsets[peer_position])
                    end = int(layout.offsets[peer_position + 1])
                    for source_flat in range(begin, end):
                        centroid_index = source_flat - begin
                        score = _peer_centroid_score(layout.embeddings[source_flat], candidate, layout)
                        round_counts["random_channel_peer_centroid_cosine_checks"] += int(
                            layout.offsets[candidate + 1] - layout.offsets[candidate]
                        )
                        if best is None or score > best[0] or (score == best[0] and centroid_index < best[1]):
                            best = (score, centroid_index)
                    if best is not None and np.isfinite(best[0]):
                        if replacement_policy == HISTORICAL_STD_REPLACEMENT_POLICY:
                            peer_historical_proposals[int(best[1])].append(
                                (int(candidate), float(best[0]), "random_channel")
                            )
                        else:
                            random_proposals.append(
                                (peer_position, best[1], [(candidate, best[0])])
                            )
            if replacement_policy == HISTORICAL_STD_REPLACEMENT_POLICY:
                peak_buffered_replacement_proposals = max(
                    peak_buffered_replacement_proposals,
                    sum(len(values) for values in peer_historical_proposals.values()),
                )
                round_counts.update(
                    _apply_historical_proposals_for_peer(
                        contacts,
                        peer_position,
                        peer_historical_proposals,
                        protected_random,
                        layout,
                        round_margins,
                    )
                )

        if replacement_policy == LEGACY_REPLACEMENT_POLICY:
            for source_name, proposals in (
                ("semantic", semantic_proposals),
                ("random_channel", random_proposals),
            ):
                for peer_position, centroid_index, values in proposals:
                    all_existing = {
                        int(candidate)
                        for local_index, centroid_values in enumerate(contacts[peer_position])
                        if local_index != centroid_index
                        for candidate, _score in centroid_values
                    }
                    all_existing.update(int(candidate) for candidate in protected_random[peer_position])
                    all_existing.add(peer_position)
                    flat_position = int(layout.offsets[peer_position]) + int(centroid_index)
                    accepted = _replace_centroid_contacts(
                        contacts[peer_position],
                        centroid_index,
                        values,
                        int(layout.contact_caps[flat_position]),
                        all_existing,
                    )
                    round_counts[f"{source_name}_accepted_contacts"] += accepted
        else:
            assert history_seen is not None
            assert history_counts is not None
            assert history_means is not None
            assert history_m2 is not None
            for peer_position, by_centroid in enumerate(contacts):
                for centroid_index, values in enumerate(by_centroid):
                    flat_centroid = int(layout.offsets[peer_position]) + int(centroid_index)
                    for candidate, score in values:
                        _update_running_history(
                            flat_centroid,
                            int(candidate),
                            float(score),
                            history_seen,
                            history_counts,
                            history_means,
                            history_m2,
                        )
            history_start_diagnostics.update(
                _distribution_summary(history_counts, "round_end_occupant_history_size")
            )
        contact_diagnostics = _contact_state_diagnostics(
            initial_sets,
            initial_peer_sets,
            initial_score_mean,
            contacts,
        )
        cumulative.update(round_counts)
        diagnostics.append(
            {
                "expansion_round": int(round_index),
                "replacement_policy": replacement_policy,
                "replacement_tolerance_multiplier": float(replacement_tolerance_multiplier),
                "round_peak_buffered_replacement_proposals": int(
                    peak_buffered_replacement_proposals
                ),
                **{f"round_{key}": int(round_counts.get(key, 0)) for key in diagnostic_fields},
                **{f"cumulative_{key}": int(cumulative.get(key, 0)) for key in diagnostic_fields},
                **history_start_diagnostics,
                **contact_diagnostics,
            }
        )
        if checkpoint_callback is not None:
            checkpoint_callback(int(round_index), contacts)
    return contacts, diagnostics


def assign_random_graph_to_centroid_quotas(
    graph: CompactGraph,
    layout: CentroidLayout,
) -> List[List[List[Tuple[int, float]]]]:
    """Assign an exact random adjacency to centroid quotas without changing its edges."""
    if graph.peer_count != layout.peer_count:
        raise ValueError("Random graph and centroid layout peer counts differ.")
    contacts: List[List[List[Tuple[int, float]]]] = []
    for source in range(graph.peer_count):
        candidates = [int(value) for value in graph.outgoing(source)]
        caps = layout.peer_caps(source).astype(np.int64)
        if int(np.sum(caps)) != len(candidates):
            raise ValueError("Centroid quotas must sum to the random graph degree.")
        pairs: List[Tuple[float, int, int]] = []
        begin = int(layout.offsets[source])
        for candidate in candidates:
            for centroid_index in range(len(caps)):
                score = _peer_centroid_score(
                    layout.embeddings[begin + centroid_index],
                    candidate,
                    layout,
                )
                pairs.append((-float(score), int(candidate), int(centroid_index)))
        pairs.sort()
        assigned: set[int] = set()
        remaining = caps.copy()
        by_centroid: List[List[Tuple[int, float]]] = [[] for _ in range(len(caps))]
        for negative_score, candidate, centroid_index in pairs:
            if candidate in assigned or remaining[centroid_index] <= 0:
                continue
            assigned.add(candidate)
            remaining[centroid_index] -= 1
            by_centroid[centroid_index].append((candidate, -negative_score))
            if len(assigned) == len(candidates):
                break
        if len(assigned) != len(candidates) or np.any(remaining != 0):
            raise RuntimeError(f"Could not assign random edges to centroid quotas for peer {source}.")
        for values in by_centroid:
            values.sort(key=lambda item: (-item[1], item[0]))
        contacts.append(by_centroid)
    return contacts


def build_semantica_topology(
    dataset: PreparedDataset,
    *,
    centroid_method: str,
    degree: int,
    delta: float,
    leaf_capacity: int,
    protected_random_fraction: float,
    seed: int,
    expansion_rounds: int = DEFAULT_EXPANSION_ROUNDS,
    replacement_policy: str = LEGACY_REPLACEMENT_POLICY,
    replacement_tolerance_multiplier: float = 0.0,
    round0_candidate_policy: str = FIRST_SEEN_ROUND0_CANDIDATE_POLICY,
    adaptive_max_k: int = ADAPTIVE_CONFIGURED_MAX_K,
    selected_k_by_peer: Optional[np.ndarray] = None,
    evaluation_expansion_rounds: Optional[Sequence[int]] = None,
    initial_topology: str = "semantica_tree",
    tree_insertion_mode: str = "canonical",
    tree_insertion_seed: Optional[int] = None,
    checkpoint_consumer: Optional[
        Callable[[int, CompactGraph, CentroidLayout], None]
    ] = None,
    retain_checkpoint_graphs: bool = True,
) -> SemanticaTopology:
    build_start = time.perf_counter()
    checkpoint_consumer_runtime = 0.0
    if initial_topology not in {"semantica_tree", "random_graph"}:
        raise ValueError(f"Unsupported initial topology: {initial_topology}")
    requested_checkpoints = sorted(
        set(
            int(value)
            for value in (
                evaluation_expansion_rounds
                if evaluation_expansion_rounds is not None
                else (0, int(expansion_rounds))
            )
        )
    )
    if not requested_checkpoints or requested_checkpoints[0] < 0:
        raise ValueError("Expansion checkpoints must be non-negative.")
    if requested_checkpoints[-1] > int(expansion_rounds):
        raise ValueError("An expansion checkpoint exceeds the requested expansion rounds.")
    protected_slots = int(math.floor(int(degree) * float(protected_random_fraction)))
    if initial_topology == "random_graph" and protected_slots:
        raise ValueError("Random-start expansion does not support protected-random slots.")
    protected_random = _protected_random_links(dataset.peer_count, protected_slots, seed)
    layout = fit_dataset_centroids(
        dataset,
        method=centroid_method,
        degree=degree,
        protected_random_fraction=protected_random_fraction,
        seed=seed,
        adaptive_max_k=adaptive_max_k,
        selected_k_by_peer=selected_k_by_peer,
    )
    random_round0_graph: Optional[CompactGraph] = None
    if initial_topology == "semantica_tree":
        round0_contacts, initial_diagnostics = _build_round0_contacts(
            layout,
            leaf_capacity=leaf_capacity,
            delta=delta,
            protected_random=protected_random,
            seed=seed,
            candidate_policy=round0_candidate_policy,
            tree_insertion_mode=tree_insertion_mode,
            tree_insertion_seed=tree_insertion_seed,
        )
        round0_source = "semantica_round0"
    else:
        random_round0_graph = build_random_graph(dataset.peer_count, degree, seed)
        round0_contacts = assign_random_graph_to_centroid_quotas(random_round0_graph, layout)
        initial_diagnostics = {
            "initial_topology": "random_graph",
            "random_round0_graph": graph_diagnostics(random_round0_graph),
            "tree_clone_count": 0,
        }
        round0_source = "random_round0"
    round0_graph = _contact_lists_to_graph(
        round0_contacts, protected_random, degree, round0_source
    )
    if random_round0_graph is not None:
        for peer_position in range(dataset.peer_count):
            if set(round0_graph.outgoing(peer_position)) != set(
                random_round0_graph.outgoing(peer_position)
            ):
                raise RuntimeError("Centroid quota assignment changed random round-0 adjacency.")
        round0_graph = random_round0_graph
        initial_diagnostics["random_round0_adjacency_preserved"] = True
        initial_diagnostics["random_round0_adjacency_definition"] = (
            "exact_offsets_neighbors_and_row_order"
        )
    round0_build_runtime_seconds = float(time.perf_counter() - build_start)
    checkpoint_graphs: Dict[int, CompactGraph] = {}
    checkpoint_diagnostics: Dict[int, Dict[str, object]] = {}
    if 0 in requested_checkpoints:
        checkpoint_diagnostics[0] = graph_diagnostics(round0_graph)
        if checkpoint_consumer is not None:
            consumer_start = time.perf_counter()
            checkpoint_consumer(0, round0_graph, layout)
            checkpoint_consumer_runtime += time.perf_counter() - consumer_start
        if retain_checkpoint_graphs:
            checkpoint_graphs[0] = round0_graph

    def capture_checkpoint(
        round_index: int,
        contacts: Sequence[Sequence[Sequence[Tuple[int, float]]]],
    ) -> None:
        nonlocal checkpoint_consumer_runtime
        if round_index in requested_checkpoints:
            graph = _contact_lists_to_graph(
                contacts,
                protected_random,
                degree,
                f"{initial_topology}_round{round_index}",
            )
            checkpoint_diagnostics[round_index] = graph_diagnostics(graph)
            if checkpoint_consumer is not None:
                consumer_start = time.perf_counter()
                checkpoint_consumer(round_index, graph, layout)
                checkpoint_consumer_runtime += time.perf_counter() - consumer_start
            if retain_checkpoint_graphs:
                checkpoint_graphs[round_index] = graph

    expansion_start = time.perf_counter()
    expansion_consumer_before = checkpoint_consumer_runtime
    expanded_contacts, expansion_diagnostics = expand_semantica_contacts(
        round0_contacts,
        layout,
        protected_random,
        rounds=expansion_rounds,
        seed=seed,
        replacement_policy=replacement_policy,
        replacement_tolerance_multiplier=replacement_tolerance_multiplier,
        checkpoint_callback=capture_checkpoint,
    )
    expansion_build_runtime_seconds = float(
        time.perf_counter()
        - expansion_start
        - (checkpoint_consumer_runtime - expansion_consumer_before)
    )
    final_graph = checkpoint_graphs.get(int(expansion_rounds))
    if int(expansion_rounds) == 0:
        final_graph = round0_graph
    elif final_graph is None:
        final_graph = _contact_lists_to_graph(
            expanded_contacts,
            protected_random,
            degree,
            f"{initial_topology}_round{int(expansion_rounds)}",
        )
    if int(expansion_rounds) in requested_checkpoints and retain_checkpoint_graphs:
        checkpoint_graphs[int(expansion_rounds)] = final_graph
    if int(expansion_rounds) not in checkpoint_diagnostics:
        checkpoint_diagnostics[int(expansion_rounds)] = graph_diagnostics(final_graph)
    selected_k = np.diff(layout.offsets)
    training_doc_counts = np.diff(dataset.inventory_offsets)
    semantic_degree = int(degree - protected_slots)
    effective_adaptive_caps = np.minimum(
        training_doc_counts,
        min(int(adaptive_max_k), semantic_degree),
    )
    is_adaptive = centroid_method_parameters(centroid_method)[0] != "fixed"
    diagnostics: Dict[str, object] = {
        **initial_diagnostics,
        "protocol_id": MULTICUTOFF_PROTOCOL_ID,
        "initial_topology": initial_topology,
        "tree_insertion_mode": tree_insertion_mode,
        "tree_insertion_seed": (
            int(tree_insertion_seed) if tree_insertion_seed is not None else None
        ),
        "centroid_method": centroid_method,
        "degree": int(degree),
        "leaf_capacity": int(leaf_capacity),
        "delta": float(delta),
        "protected_random_fraction": float(protected_random_fraction),
        "protected_random_slots": int(protected_slots),
        "semantic_slots": int(degree - protected_slots),
        "expansion_rounds_requested": int(expansion_rounds),
        "expansion_rounds_completed": int(len(expansion_diagnostics)),
        "evaluation_expansion_rounds": requested_checkpoints,
        "expansion_replacement_policy": str(replacement_policy),
        "round0_candidate_policy": str(round0_candidate_policy),
        "replacement_tolerance_multiplier": float(replacement_tolerance_multiplier),
        "replacement_tolerance_definition": (
            "multiplier_times_sample_std_ddof_1_of_distinct_actual_centroid_occupants"
            if replacement_policy == HISTORICAL_STD_REPLACEMENT_POLICY
            else "not_applicable_legacy_rank_truncation"
        ),
        "replacement_history_activation_distinct_occupants": (
            3 if replacement_policy == HISTORICAL_STD_REPLACEMENT_POLICY else None
        ),
        "replacement_history_update_rule": (
            "round0_contacts_plus_post_round_accepted_occupants_effective_next_round"
            if replacement_policy == HISTORICAL_STD_REPLACEMENT_POLICY
            else "not_applicable"
        ),
        "centroid_count": int(layout.embeddings.shape[0]),
        "adaptive_configured_max_k": int(adaptive_max_k),
        "selected_k_cache_used": bool(is_adaptive and selected_k_by_peer is not None),
        "adaptive_effective_degree_cap": int(min(int(adaptive_max_k), semantic_degree)),
        "adaptive_min_documents_per_centroid": None,
        "mean_selected_k": float(np.mean(selected_k)),
        "min_selected_k": int(np.min(selected_k)),
        "max_selected_k": int(np.max(selected_k)),
        "median_selected_k": float(np.median(selected_k)),
        "fraction_selected_k_at_effective_cap": float(
            np.mean(selected_k == effective_adaptive_caps)
        ) if is_adaptive else None,
        "mean_training_docs_per_centroid": float(
            np.mean(training_doc_counts / np.maximum(selected_k, 1))
        ),
        "total_build_runtime_seconds": float(
            time.perf_counter() - build_start - checkpoint_consumer_runtime
        ),
        "round0_build_runtime_seconds": round0_build_runtime_seconds,
        "incremental_expansion_runtime_seconds": expansion_build_runtime_seconds,
        "checkpoint_evaluation_runtime_excluded_seconds": float(
            checkpoint_consumer_runtime
        ),
        "peak_memory_mb": peak_memory_mb(),
        "peak_memory_measurement_note": (
            "process high-water RSS after streamed checkpoint evaluations; "
            "not topology-construction-only memory"
        ),
        "round0_graph": graph_diagnostics(round0_graph),
        "round0_graph_sha256": compact_graph_sha256(round0_graph),
        "centroid_layout_sha256": centroid_layout_sha256(layout),
        "final_graph_label": f"round_{int(expansion_rounds)}",
        "final_graph": graph_diagnostics(final_graph),
        "checkpoint_graphs": {
            f"round_{round_index}": diagnostic
            for round_index, diagnostic in sorted(checkpoint_diagnostics.items())
        },
    }
    if int(expansion_rounds) == 10:
        diagnostics["round10_graph"] = diagnostics["final_graph"]
    return SemanticaTopology(
        centroids=layout,
        round0_graph=round0_graph,
        final_graph=final_graph,
        checkpoint_graphs=checkpoint_graphs,
        protected_random=protected_random,
        diagnostics=diagnostics,
        expansion_rounds=expansion_diagnostics,
    )


def build_random_graph(peer_count: int, degree: int, seed: int) -> CompactGraph:
    if int(degree) >= int(peer_count):
        raise ValueError("Random graph degree must be smaller than peer count.")
    offsets = np.arange(0, (int(peer_count) + 1) * int(degree), int(degree), dtype=np.int64)
    neighbors = np.empty(int(peer_count) * int(degree), dtype=np.int32)
    for source in range(int(peer_count)):
        rng = np.random.default_rng(stable_int(f"random-graph:{seed}:{degree}:{source}"))
        values = rng.choice(int(peer_count) - 1, size=int(degree), replace=False)
        values = values + (values >= source)
        neighbors[offsets[source] : offsets[source + 1]] = np.sort(values.astype(np.int32))
    return CompactGraph(offsets=offsets, neighbors=neighbors, graph_source="random_graph", degree=int(degree))


@dataclass
class ChainCounters:
    max_hops: int
    peer_count: int
    cutoffs: Tuple[int, ...] = RETRIEVAL_CUTOFFS
    query_count: int = 0
    available_query_count: int = 0
    current_hit_counts: Dict[int, np.ndarray] = field(init=False)
    cumulative_entry_counts: Dict[int, np.ndarray] = field(init=False)
    reciprocal_rank_sums: Dict[int, np.ndarray] = field(init=False)
    ndcg_sums: Dict[int, np.ndarray] = field(init=False)
    exact_neighbor_overlap_sums: Dict[int, np.ndarray] = field(init=False)
    message_sums: np.ndarray = field(init=False)
    peer_cosine_check_sums: np.ndarray = field(init=False)
    document_cosine_check_sums: np.ndarray = field(init=False)
    unique_peer_sums: np.ndarray = field(init=False)
    unique_document_sums: np.ndarray = field(init=False)
    returned_id_sums: np.ndarray = field(init=False)
    returned_byte_sums: np.ndarray = field(init=False)
    available_hit_counts: Dict[int, np.ndarray] = field(init=False)
    target_owner_contact_counts: np.ndarray = field(init=False)
    target_rank_sums: np.ndarray = field(init=False)
    target_rank_observation_counts: np.ndarray = field(init=False)
    owner_query_counts: Counter = field(default_factory=Counter)
    owner_hit_counts: Dict[int, Dict[int, np.ndarray]] = field(init=False)
    first_target_owner_contact_histogram: np.ndarray = field(init=False)
    first_entry_histograms: Dict[int, np.ndarray] = field(init=False)
    terminal_contact_histogram: np.ndarray = field(init=False)
    stop_reason_counts: Counter = field(default_factory=Counter)
    process_peak_memory_mb_sum_upper_bound: float = 0.0
    process_peak_memory_mb_max: float = 0.0

    def __post_init__(self) -> None:
        self.cutoffs = tuple(sorted(set(int(value) for value in self.cutoffs)))
        if not self.cutoffs or self.cutoffs[-1] != LOCAL_RESPONSE_K:
            raise ValueError(f"Chain cutoffs must end at local response K={LOCAL_RESPONSE_K}.")
        length = int(self.max_hops) + 1
        self.current_hit_counts = {
            cutoff: np.zeros(length, dtype=np.int64) for cutoff in self.cutoffs
        }
        self.cumulative_entry_counts = {
            cutoff: np.zeros(length, dtype=np.int64) for cutoff in self.cutoffs
        }
        self.reciprocal_rank_sums = {
            cutoff: np.zeros(length, dtype=np.float64) for cutoff in self.cutoffs
        }
        self.ndcg_sums = {
            cutoff: np.zeros(length, dtype=np.float64) for cutoff in self.cutoffs
        }
        self.exact_neighbor_overlap_sums = {
            cutoff: np.zeros(length, dtype=np.float64) for cutoff in self.cutoffs
        }
        self.message_sums = np.zeros(length, dtype=np.int64)
        self.peer_cosine_check_sums = np.zeros(length, dtype=np.int64)
        self.document_cosine_check_sums = np.zeros(length, dtype=np.int64)
        self.unique_peer_sums = np.zeros(length, dtype=np.int64)
        self.unique_document_sums = np.zeros(length, dtype=np.int64)
        self.returned_id_sums = np.zeros(length, dtype=np.int64)
        self.returned_byte_sums = np.zeros(length, dtype=np.int64)
        self.available_hit_counts = {
            cutoff: np.zeros(length, dtype=np.int64) for cutoff in self.cutoffs
        }
        self.target_owner_contact_counts = np.zeros(length, dtype=np.int64)
        self.target_rank_sums = np.zeros(length, dtype=np.float64)
        self.target_rank_observation_counts = np.zeros(length, dtype=np.int64)
        self.owner_hit_counts = {cutoff: {} for cutoff in self.cutoffs}
        self.first_target_owner_contact_histogram = np.zeros(length + 1, dtype=np.int64)
        self.first_entry_histograms = {
            cutoff: np.zeros(length + 1, dtype=np.int64) for cutoff in self.cutoffs
        }
        self.terminal_contact_histogram = np.zeros(length, dtype=np.int64)

    @property
    def first_target_seen_histogram(self) -> np.ndarray:
        return self.first_target_owner_contact_histogram

    @property
    def first_top10_entry_histogram(self) -> np.ndarray:
        return self.first_entry_histograms[10]

    def merge(self, other: "ChainCounters") -> None:
        if self.max_hops != other.max_hops or self.peer_count != other.peer_count:
            raise ValueError("Cannot merge incompatible chain counters.")
        self.query_count += int(other.query_count)
        self.available_query_count += int(other.available_query_count)
        if self.cutoffs != other.cutoffs:
            raise ValueError("Cannot merge chain counters with different retrieval cutoffs.")
        for cutoff in self.cutoffs:
            for values_by_cutoff in (
                "current_hit_counts",
                "cumulative_entry_counts",
                "reciprocal_rank_sums",
                "ndcg_sums",
                "exact_neighbor_overlap_sums",
                "available_hit_counts",
                "first_entry_histograms",
            ):
                getattr(self, values_by_cutoff)[cutoff] += getattr(other, values_by_cutoff)[cutoff]
        for name in (
            "message_sums",
            "peer_cosine_check_sums",
            "document_cosine_check_sums",
            "unique_peer_sums",
            "unique_document_sums",
            "returned_id_sums",
            "returned_byte_sums",
            "target_owner_contact_counts",
            "target_rank_sums",
            "target_rank_observation_counts",
            "first_target_owner_contact_histogram",
            "terminal_contact_histogram",
        ):
            getattr(self, name)[:] += getattr(other, name)
        self.owner_query_counts.update(other.owner_query_counts)
        length = self.max_hops + 1
        for cutoff in self.cutoffs:
            for owner, incoming in other.owner_hit_counts[cutoff].items():
                current = self.owner_hit_counts[cutoff].setdefault(
                    int(owner), np.zeros(length, dtype=np.int32)
                )
                current += incoming
        self.stop_reason_counts.update(other.stop_reason_counts)
        self.process_peak_memory_mb_max = max(
            self.process_peak_memory_mb_max,
            float(other.process_peak_memory_mb_max),
        )

    def curve_frame(self, metadata: Mapping[str, object]) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for hop in range(self.max_hops + 1):
            query_count = max(1, self.query_count)
            row: Dict[str, object] = {
                **dict(metadata),
                "query_hop": int(hop),
                "query_count": int(self.query_count),
                "available_query_count": int(self.available_query_count),
                "target_owner_contact_rate": float(
                    self.target_owner_contact_counts[hop] / query_count
                ),
                "mean_messages": float(self.message_sums[hop] / query_count),
                "mean_peer_centroid_cosine_checks": float(
                    self.peer_cosine_check_sums[hop] / query_count
                ),
                "mean_document_cosine_checks": float(
                    self.document_cosine_check_sums[hop] / query_count
                ),
                "mean_unique_peers_contacted": float(self.unique_peer_sums[hop] / query_count),
                "mean_unique_documents_scored": float(
                    self.unique_document_sums[hop] / query_count
                ),
                "mean_returned_result_ids": float(self.returned_id_sums[hop] / query_count),
                "mean_returned_result_bytes": float(self.returned_byte_sums[hop] / query_count),
                "mean_target_rank_when_in_top_100": (
                    float(self.target_rank_sums[hop] / self.target_rank_observation_counts[hop])
                    if self.target_rank_observation_counts[hop]
                    else np.nan
                ),
                "process_peak_memory_mb_sum_upper_bound": float(
                    self.process_peak_memory_mb_sum_upper_bound
                ),
                "process_peak_memory_mb_max": float(self.process_peak_memory_mb_max),
            }
            for cutoff in self.cutoffs:
                hit_rate = float(self.current_hit_counts[cutoff][hop] / query_count)
                macro_owner = 0.0
                if self.owner_query_counts:
                    macro_owner = float(
                        np.mean(
                            [
                                self.owner_hit_counts[cutoff][owner][hop] / count
                                for owner, count in self.owner_query_counts.items()
                            ]
                        )
                    )
                row[f"hit_rate_at_{cutoff}"] = hit_rate
                row[f"recall_at_{cutoff}"] = hit_rate
                row[f"cumulative_target_entry_rate_at_{cutoff}"] = float(
                    self.cumulative_entry_counts[cutoff][hop] / query_count
                )
                row[f"macro_owner_hit_rate_at_{cutoff}"] = macro_owner
                row[f"availability_conditioned_hit_rate_at_{cutoff}"] = (
                    float(self.available_hit_counts[cutoff][hop] / self.available_query_count)
                    if self.available_query_count
                    else 0.0
                )
                row[f"target_owner_contact_minus_recall_at_{cutoff}"] = float(
                    row["target_owner_contact_rate"] - hit_rate
                )
                row[f"exact_neighbor_recall_at_{cutoff}"] = float(
                    self.exact_neighbor_overlap_sums[cutoff][hop] / query_count
                )
                row[f"exact_overlap_at_{cutoff}"] = row[
                    f"exact_neighbor_recall_at_{cutoff}"
                ]
            row["target_owner_visited_rate"] = row["target_owner_contact_rate"]
            row["precision_at_10"] = float(
                self.current_hit_counts[10][hop] / (10.0 * query_count)
            )
            row["mrr_at_10"] = float(self.reciprocal_rank_sums[10][hop] / query_count)
            row["ndcg_at_10"] = float(self.ndcg_sums[10][hop] / query_count)
            row["mrr_at_100"] = float(
                self.reciprocal_rank_sums[LOCAL_RESPONSE_K][hop] / query_count
            )
            row["ndcg_at_100"] = float(
                self.ndcg_sums[LOCAL_RESPONSE_K][hop] / query_count
            )
            row["cumulative_target_discovery_rate"] = row[
                "cumulative_target_entry_rate_at_10"
            ]
            rows.append(row)
        return pd.DataFrame(rows)

    def histogram_frame(self, metadata: Mapping[str, object]) -> pd.DataFrame:
        rows = []
        never_index = self.max_hops + 1
        histograms = [("first_target_owner_contact", self.first_target_owner_contact_histogram)]
        histograms.extend(
            (f"first_top_{cutoff}_entry", self.first_entry_histograms[cutoff])
            for cutoff in self.cutoffs
        )
        for metric_name, values in histograms:
            for index, count in enumerate(values.tolist()):
                rows.append(
                    {
                        **dict(metadata),
                        "first_hit_metric": metric_name,
                        "query_hop": None if index == never_index else int(index),
                        "never_reached": bool(index == never_index),
                        "query_count": int(count),
                        "query_fraction": float(count / max(1, self.query_count)),
                    }
                )
        return pd.DataFrame(rows)

    def terminal_contact_frame(self, metadata: Mapping[str, object]) -> pd.DataFrame:
        counts = self.terminal_contact_histogram
        total = max(1, int(self.query_count))
        cumulative = np.cumsum(counts)

        def percentile(value: float) -> int:
            if self.query_count <= 0:
                return 0
            rank = max(1, int(math.ceil(value * int(self.query_count))))
            return int(np.searchsorted(cumulative, rank, side="left"))

        summary = {
            "terminal_contacts_mean": float(
                np.dot(np.arange(len(counts), dtype=np.float64), counts) / total
            ),
            "terminal_contacts_median": percentile(0.5),
            "terminal_contacts_p95": percentile(0.95),
            "terminal_hop_limit_rate": float(
                self.stop_reason_counts.get("hop_limit", 0) / total
            ),
            "terminal_early_termination_rate": float(
                (int(self.query_count) - self.stop_reason_counts.get("hop_limit", 0))
                / total
            ),
        }
        return pd.DataFrame(
            [
                {
                    **dict(metadata),
                    **summary,
                    "terminal_contacts": int(contact_count),
                    "query_count": int(count),
                    "query_fraction": float(count / total),
                }
                for contact_count, count in enumerate(counts.tolist())
            ]
        )


def _local_topk(
    dataset: PreparedDataset,
    query_embedding: np.ndarray,
    doc_indices: np.ndarray,
    top_k: int,
) -> List[Tuple[int, float]]:
    if len(doc_indices) == 0:
        return []
    indices = np.unique(np.asarray(doc_indices, dtype=np.int64))
    values = np.asarray(dataset.doc_embeddings[indices], dtype=np.float32)
    scores = (values @ query_embedding) / dataset.doc_norms[indices]
    positions = np.arange(len(indices), dtype=np.int64)
    if len(indices) > int(top_k):
        boundary = len(indices) - int(top_k)
        threshold = float(np.partition(scores, boundary)[boundary])
        greater = np.flatnonzero(scores > threshold).tolist()
        tied = np.flatnonzero(scores == threshold).tolist()
        tied.sort(key=lambda position: str(dataset.doc_ids[int(indices[position])]))
        positions = np.asarray(
            greater + tied[: max(0, int(top_k) - len(greater))], dtype=np.int64
        )
    ranked = sorted(
        (
            (int(indices[position]), float(scores[position]))
            for position in positions.tolist()
        ),
        key=lambda item: (-item[1], str(dataset.doc_ids[item[0]])),
    )
    return ranked[: int(top_k)]


def _local_top10(
    dataset: PreparedDataset,
    query_embedding: np.ndarray,
    doc_indices: np.ndarray,
) -> List[Tuple[int, float]]:
    return _local_topk(dataset, query_embedding, doc_indices, 10)


def _merge_global_topk(
    dataset: PreparedDataset,
    current: Sequence[Tuple[int, float]],
    incoming: Sequence[Tuple[int, float]],
    top_k: int,
) -> List[Tuple[int, float]]:
    best: Dict[int, float] = {}
    for doc_index, score in list(current) + list(incoming):
        best[int(doc_index)] = max(float(score), best.get(int(doc_index), -np.inf))
    doc_indices = np.fromiter(best.keys(), dtype=np.int64, count=len(best))
    scores = np.fromiter(best.values(), dtype=np.float64, count=len(best))
    selected_docs, selected_scores = _topk_score_pairs(
        scores, doc_indices, dataset.doc_tie_ranks, int(top_k)
    )
    return list(zip(selected_docs.tolist(), selected_scores.tolist()))


def _merge_global_top10(
    dataset: PreparedDataset,
    current: Sequence[Tuple[int, float]],
    incoming: Sequence[Tuple[int, float]],
) -> List[Tuple[int, float]]:
    return _merge_global_topk(dataset, current, incoming, 10)


def _route_next_peer(
    *,
    source_peer: int,
    graph: CompactGraph,
    layout: CentroidLayout,
    query_embedding: np.ndarray,
    visited: set[int],
    routing_policy: str,
    seed: int,
    manifest_row: int,
    hop: int,
    peer_ids: Optional[np.ndarray] = None,
) -> Tuple[Optional[int], int, str]:
    outgoing = [int(value) for value in graph.outgoing(source_peer)]
    if not outgoing:
        return None, 0, "dead_end"
    candidates = [candidate for candidate in outgoing if candidate not in visited]
    if not candidates:
        return None, 0, "exhausted_neighbors"
    if routing_policy == "uniform_random_walk":
        rng = np.random.default_rng(stable_int(f"random-walk:{seed}:{manifest_row}:{hop}"))
        return candidates[int(rng.integers(0, len(candidates)))], 0, "running"
    if routing_policy != "semantic_chain":
        raise ValueError(f"Unsupported chain routing policy: {routing_policy}")
    ranked: List[Tuple[float, Tuple[int, object], int]] = []
    checks = 0
    for candidate in candidates:
        centroids = layout.peer_centroids(candidate)
        checks += len(centroids)
        score = float(np.max(centroids @ query_embedding)) if len(centroids) else -np.inf
        raw_peer_id = peer_ids[candidate] if peer_ids is not None else candidate
        if isinstance(raw_peer_id, (int, np.integer)):
            tie_value: Tuple[int, object] = (0, int(raw_peer_id))
        else:
            tie_value = (1, str(raw_peer_id))
        ranked.append((-score, tie_value, candidate))
    ranked.sort()
    return int(ranked[0][2]), int(checks), "running"


@dataclass
class SequentialQueryTrace:
    """Complete observable trace for one centralized semantic-chain query."""

    query_id: str
    source_peer: int
    target_doc_id: str
    peer_by_hop: Dict[int, int]
    results_by_hop: Dict[int, List[Tuple[str, float]]]
    final_results: List[Tuple[str, float]]
    terminal_hop: int
    terminal_reason: str


def trace_semantic_chain_query(
    dataset: PreparedDataset,
    graph: CompactGraph,
    layout: CentroidLayout,
    *,
    source_peer: int,
    query_embedding_index: int,
    target_doc_index: int,
    manifest_row: int,
    max_hops: int = DEFAULT_MAX_HOPS,
    top_k: int = LOCAL_RESPONSE_K,
    seed: int,
    query_doc_index: int = -1,
) -> SequentialQueryTrace:
    """Run one read-only query sequentially through the centralized topology.

    This function deliberately uses the established centralized routing,
    document-ranking, and global-merge implementations.  It is intended for
    differential validation of distributed executions, not aggregate metrics.
    """
    source_peer = int(source_peer)
    query_embedding_index = int(query_embedding_index)
    target_doc_index = int(target_doc_index)
    query_doc_index = int(query_doc_index)
    query = dataset.normalized_query_rows([query_embedding_index])[0]
    visited = {source_peer}
    current_peer = source_peer
    peer_by_hop: Dict[int, int] = {}
    index_results_by_hop: Dict[int, List[Tuple[int, float]]] = {}
    global_top: List[Tuple[int, float]] = []
    terminal_hop = 0
    terminal_reason = "dead_end"

    for hop in range(1, int(max_hops) + 1):
        next_peer, _checks, reason = _route_next_peer(
            source_peer=current_peer,
            graph=graph,
            layout=layout,
            query_embedding=query,
            visited=visited,
            routing_policy="semantic_chain",
            seed=int(seed),
            manifest_row=int(manifest_row),
            hop=hop,
            peer_ids=dataset.peer_ids,
        )
        if next_peer is None:
            terminal_reason = reason
            terminal_hop = hop - 1
            break
        visited.add(next_peer)
        current_peer = next_peer
        peer_by_hop[hop] = next_peer
        docs = dataset.peer_documents(next_peer)
        if query_doc_index >= 0:
            docs = docs[docs != query_doc_index]
        local_top = _local_topk(dataset, query, docs, int(top_k))
        index_results_by_hop[hop] = local_top
        global_top = _merge_global_topk(dataset, global_top, local_top, int(top_k))
        terminal_hop = hop
        if hop == int(max_hops):
            terminal_reason = "hop_limit"
            break

    def externalize(values: Sequence[Tuple[int, float]]) -> List[Tuple[str, float]]:
        return [
            (str(dataset.doc_ids[int(doc_index)]), float(score))
            for doc_index, score in values
        ]

    return SequentialQueryTrace(
        query_id=str(dataset.query_keys[query_embedding_index]),
        source_peer=source_peer,
        target_doc_id=str(dataset.doc_ids[target_doc_index]),
        peer_by_hop=peer_by_hop,
        results_by_hop={
            hop: externalize(values) for hop, values in index_results_by_hop.items()
        },
        final_results=externalize(global_top),
        terminal_hop=terminal_hop,
        terminal_reason=terminal_reason,
    )


def evaluate_chain_rows(
    dataset: PreparedDataset,
    graph: CompactGraph,
    layout: CentroidLayout,
    rows: pd.DataFrame,
    *,
    routing_policy: str = "semantic_chain",
    max_hops: int = DEFAULT_MAX_HOPS,
    seed: int,
    progress_every: int = 10_000,
    progress_prefix: str = "",
    exact_topk_doc_indices: Optional[np.ndarray] = None,
) -> ChainCounters:
    counters = ChainCounters(max_hops=int(max_hops), peer_count=dataset.peer_count)
    required = {
        "manifest_row",
        "source_peer_position",
        "query_embedding_index",
        "target_doc_index",
        "target_available_in_training_network",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Query manifest is missing columns: {sorted(missing)}")
    query_indices = rows["query_embedding_index"].to_numpy(dtype=np.int64)
    query_embeddings = dataset.normalized_query_rows(query_indices)
    if exact_topk_doc_indices is not None and exact_topk_doc_indices.shape != (
        len(rows),
        LOCAL_RESPONSE_K,
    ):
        raise ValueError("Exact-neighbor rows must align with query rows and local response K.")
    never_index = int(max_hops) + 1
    exact_rank_lookup = (
        np.full(len(dataset.doc_ids), LOCAL_RESPONSE_K + 1, dtype=np.uint16)
        if exact_topk_doc_indices is not None
        else None
    )
    cutoff_array = np.asarray(counters.cutoffs, dtype=np.int32)
    for local_position, (record, query_embedding) in enumerate(
        zip(rows.itertuples(index=False), query_embeddings), start=1
    ):
        source_peer = int(record.source_peer_position)
        target_doc = int(record.target_doc_index)
        query_doc = int(getattr(record, "query_doc_index", -1))
        available = bool(record.target_available_in_training_network)
        manifest_row = int(record.manifest_row)
        counters.query_count += 1
        counters.available_query_count += int(available)
        counters.owner_query_counts[source_peer] += 1
        for cutoff in counters.cutoffs:
            counters.owner_hit_counts[cutoff].setdefault(
                source_peer, np.zeros(int(max_hops) + 1, dtype=np.int32)
            )
        visited = {source_peer}
        current_peer = source_peer
        unique_documents: set[int] = set()
        global_top: List[Tuple[int, float]] = []
        messages = 0
        peer_checks = 0
        doc_checks = 0
        returned_ids = 0
        returned_bytes = 0
        exact_row = exact_topk_doc_indices[local_position - 1] if exact_topk_doc_indices is not None else None
        valid_exact = exact_row[exact_row >= 0] if exact_row is not None else np.empty(0, dtype=np.int32)
        if exact_rank_lookup is not None:
            exact_rank_lookup[valid_exact] = np.arange(1, len(valid_exact) + 1, dtype=np.uint16)
        target_owner_contacted = False
        first_owner_contact: Optional[int] = None
        first_entries: Dict[int, Optional[int]] = {cutoff: None for cutoff in counters.cutoffs}
        cumulative_entries = {cutoff: False for cutoff in counters.cutoffs}
        terminal_reason: Optional[str] = None

        def record_hop(hop: int) -> None:
            target_rank = (
                next(
                    (index for index, (doc_index, _score) in enumerate(global_top, start=1) if doc_index == target_doc),
                    None,
                )
            )
            counters.target_owner_contact_counts[hop] += int(target_owner_contacted)
            overlap_counts = None
            if exact_rank_lookup is not None and global_top:
                global_docs = np.fromiter(
                    (doc_index for doc_index, _score in global_top),
                    dtype=np.int64,
                    count=len(global_top),
                )
                thresholds = np.maximum(
                    np.arange(1, len(global_docs) + 1, dtype=np.int32),
                    exact_rank_lookup[global_docs].astype(np.int32),
                )
                bins = np.searchsorted(cutoff_array, thresholds, side="left")
                overlap_counts = np.cumsum(
                    np.bincount(bins, minlength=len(cutoff_array) + 1)[: len(cutoff_array)]
                )
            if target_rank is not None and target_rank <= LOCAL_RESPONSE_K:
                counters.target_rank_sums[hop] += float(target_rank)
                counters.target_rank_observation_counts[hop] += 1
            for cutoff in counters.cutoffs:
                hit = int(target_rank is not None and target_rank <= cutoff)
                counters.current_hit_counts[cutoff][hop] += hit
                counters.cumulative_entry_counts[cutoff][hop] += int(cumulative_entries[cutoff])
                counters.available_hit_counts[cutoff][hop] += int(available and hit)
                counters.owner_hit_counts[cutoff][source_peer][hop] += hit
                if hit:
                    counters.reciprocal_rank_sums[cutoff][hop] += 1.0 / target_rank
                    counters.ndcg_sums[cutoff][hop] += 1.0 / math.log2(target_rank + 1.0)
                if overlap_counts is not None:
                    cutoff_position = int(np.searchsorted(cutoff_array, cutoff))
                    counters.exact_neighbor_overlap_sums[cutoff][hop] += (
                        float(overlap_counts[cutoff_position]) / float(cutoff)
                    )
            counters.message_sums[hop] += messages
            counters.peer_cosine_check_sums[hop] += peer_checks
            counters.document_cosine_check_sums[hop] += doc_checks
            counters.unique_peer_sums[hop] += len(visited) - 1
            counters.unique_document_sums[hop] += len(unique_documents)
            counters.returned_id_sums[hop] += returned_ids
            counters.returned_byte_sums[hop] += returned_bytes

        record_hop(0)
        last_recorded_hop = 0
        for hop in range(1, int(max_hops) + 1):
            next_peer, checks, reason = _route_next_peer(
                source_peer=current_peer,
                graph=graph,
                layout=layout,
                query_embedding=query_embedding,
                visited=visited,
                routing_policy=routing_policy,
                seed=seed,
                manifest_row=manifest_row,
                hop=hop,
                peer_ids=dataset.peer_ids,
            )
            peer_checks += checks
            if next_peer is None:
                terminal_reason = reason
                break
            messages += 1
            visited.add(next_peer)
            current_peer = next_peer
            docs = dataset.peer_documents(next_peer)
            if query_doc >= 0:
                docs = docs[docs != query_doc]
            unique_documents.update(int(value) for value in docs.tolist())
            doc_checks += len(docs)
            if not target_owner_contacted and np.any(docs == target_doc):
                target_owner_contacted = True
                first_owner_contact = hop
            local_top = _local_topk(dataset, query_embedding, docs, LOCAL_RESPONSE_K)
            returned_ids += len(local_top)
            returned_bytes += len(local_top) * RESULT_WIRE_BYTES_PER_DOCUMENT
            global_top = _merge_global_topk(dataset, global_top, local_top, LOCAL_RESPONSE_K)
            target_rank = (
                next(
                    (index for index, (doc_index, _score) in enumerate(global_top, start=1) if doc_index == target_doc),
                    None,
                )
            )
            for cutoff in counters.cutoffs:
                if target_rank is not None and target_rank <= cutoff:
                    cumulative_entries[cutoff] = True
                    if first_entries[cutoff] is None:
                        first_entries[cutoff] = hop
            record_hop(hop)
            last_recorded_hop = hop
        if terminal_reason is None:
            terminal_reason = "hop_limit"
        for hop in range(last_recorded_hop + 1, int(max_hops) + 1):
            record_hop(hop)
        counters.stop_reason_counts[terminal_reason] += 1
        counters.terminal_contact_histogram[messages] += 1
        if not available:
            counters.stop_reason_counts["unavailable_target"] += 1
        counters.first_target_owner_contact_histogram[
            first_owner_contact if first_owner_contact is not None else never_index
        ] += 1
        for cutoff in counters.cutoffs:
            counters.first_entry_histograms[cutoff][
                first_entries[cutoff] if first_entries[cutoff] is not None else never_index
            ] += 1
        if exact_rank_lookup is not None:
            exact_rank_lookup[valid_exact] = LOCAL_RESPONSE_K + 1
        if progress_every and local_position % int(progress_every) == 0:
            print(
                f"{progress_prefix}completed {local_position}/{len(rows)} query events",
                flush=True,
            )
    worker_peak = peak_memory_mb()
    counters.process_peak_memory_mb_sum_upper_bound = worker_peak
    counters.process_peak_memory_mb_max = worker_peak
    return counters


_CHAIN_WORKER_STATE: Dict[str, object] = {}


def _initialize_chain_worker(
    dataset: PreparedDataset,
    graph: CompactGraph,
    layout: CentroidLayout,
    routing_policy: str,
    max_hops: int,
    seed: int,
    exact_topk_doc_indices: Optional[np.ndarray],
    progress_prefix: str,
) -> None:
    global _CHAIN_WORKER_STATE
    _CHAIN_WORKER_STATE = {
        "dataset": dataset,
        "graph": graph,
        "layout": layout,
        "routing_policy": routing_policy,
        "max_hops": int(max_hops),
        "seed": int(seed),
        "exact_topk_doc_indices": exact_topk_doc_indices,
        "progress_prefix": str(progress_prefix),
    }


def _evaluate_chain_worker(payload: Tuple[int, pd.DataFrame]) -> Tuple[int, int, ChainCounters]:
    chunk_index, rows = payload
    state = _CHAIN_WORKER_STATE
    result = evaluate_chain_rows(
        state["dataset"],
        state["graph"],
        state["layout"],
        rows,
        routing_policy=str(state["routing_policy"]),
        max_hops=int(state["max_hops"]),
        seed=int(state["seed"]),
        progress_every=10_000,
        progress_prefix=f"{state['progress_prefix']}worker_chunk={chunk_index} ",
        exact_topk_doc_indices=(
            state["exact_topk_doc_indices"][rows.index.to_numpy(dtype=np.int64)]
            if state["exact_topk_doc_indices"] is not None
            else None
        ),
    )
    return chunk_index, os.getpid(), result


def evaluate_chain(
    dataset: PreparedDataset,
    graph: CompactGraph,
    layout: CentroidLayout,
    *,
    routing_policy: str = "semantic_chain",
    max_hops: int = DEFAULT_MAX_HOPS,
    seed: int,
    workers: int = 1,
    exact_topk_doc_indices: Optional[np.ndarray] = None,
    progress_prefix: str = "",
) -> ChainCounters:
    workers = max(1, int(workers))
    if workers == 1 or len(dataset.manifest) < workers:
        return evaluate_chain_rows(
            dataset,
            graph,
            layout,
            dataset.manifest,
            routing_policy=routing_policy,
            max_hops=max_hops,
            seed=seed,
            exact_topk_doc_indices=exact_topk_doc_indices,
            progress_prefix=progress_prefix,
        )
    import multiprocessing as mp

    try:
        context = mp.get_context("fork")
    except ValueError:
        context = mp.get_context()
    _ = dataset.doc_norms
    _ = dataset.doc_tie_ranks
    manifest = dataset.manifest.reset_index(drop=True)
    chunk_count = min(len(manifest), max(workers * 2, int(math.ceil(len(manifest) / 10_000))))
    boundaries = np.linspace(0, len(manifest), num=chunk_count + 1, dtype=np.int64)
    chunks = [
        manifest.iloc[int(start) : int(end)].copy()
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if int(end) > int(start)
    ]
    combined = ChainCounters(max_hops=int(max_hops), peer_count=dataset.peer_count)
    completed_queries = 0
    next_progress = 10_000
    worker_peaks_by_pid: Dict[int, float] = {}
    with context.Pool(
        processes=workers,
        initializer=_initialize_chain_worker,
        initargs=(
            dataset,
            graph,
            layout,
            routing_policy,
            max_hops,
            seed,
            exact_topk_doc_indices,
            progress_prefix,
        ),
    ) as pool:
        for _chunk_index, worker_pid, partial in pool.imap_unordered(
            _evaluate_chain_worker,
            list(enumerate(chunks)),
        ):
            combined.merge(partial)
            worker_peaks_by_pid[worker_pid] = max(
                worker_peaks_by_pid.get(worker_pid, 0.0),
                partial.process_peak_memory_mb_max,
            )
            completed_queries += partial.query_count
            while completed_queries >= next_progress:
                print(
                    f"{progress_prefix}completed at least "
                    f"{next_progress}/{len(dataset.manifest)} query events",
                    flush=True,
                )
                next_progress += 10_000
    parent_peak = peak_memory_mb()
    combined.process_peak_memory_mb_sum_upper_bound = parent_peak + sum(worker_peaks_by_pid.values())
    combined.process_peak_memory_mb_max = max(
        [combined.process_peak_memory_mb_max, parent_peak, *worker_peaks_by_pid.values()]
    )
    return combined


@dataclass
class RankedMetrics:
    query_count: int = 0
    available_query_count: int = 0
    hit_counts: Counter = field(default_factory=Counter)
    available_hit_counts: Counter = field(default_factory=Counter)
    reciprocal_rank_sums: Counter = field(default_factory=Counter)
    ndcg_sums: Counter = field(default_factory=Counter)
    exact_neighbor_overlap_sums: Counter = field(default_factory=Counter)
    exact_reference_query_count: int = 0
    returned_id_count: int = 0
    unique_candidate_count: int = 0
    message_count: int = 0
    document_cosine_checks: int = 0
    runtime_seconds: float = 0.0
    owner_query_counts: Counter = field(default_factory=Counter)
    owner_hit_counts: Dict[int, Counter] = field(
        default_factory=lambda: {cutoff: Counter() for cutoff in RETRIEVAL_CUTOFFS}
    )

    def merge(self, other: "RankedMetrics") -> None:
        self.query_count += int(other.query_count)
        self.available_query_count += int(other.available_query_count)
        self.hit_counts.update(other.hit_counts)
        self.available_hit_counts.update(other.available_hit_counts)
        self.reciprocal_rank_sums.update(other.reciprocal_rank_sums)
        self.ndcg_sums.update(other.ndcg_sums)
        self.exact_neighbor_overlap_sums.update(other.exact_neighbor_overlap_sums)
        self.exact_reference_query_count += int(other.exact_reference_query_count)
        self.returned_id_count += int(other.returned_id_count)
        self.unique_candidate_count += int(other.unique_candidate_count)
        self.message_count += int(other.message_count)
        self.document_cosine_checks += int(other.document_cosine_checks)
        self.runtime_seconds += float(other.runtime_seconds)
        self.owner_query_counts.update(other.owner_query_counts)
        for cutoff in RETRIEVAL_CUTOFFS:
            self.owner_hit_counts[cutoff].update(other.owner_hit_counts[cutoff])

    def add(
        self,
        ranked_doc_indices: Sequence[int],
        target_doc_index: int,
        available: bool,
        *,
        returned_ids: int,
        unique_candidates: int,
        messages: int,
        document_checks: int,
        source_owner: Optional[int] = None,
        exact_topk_doc_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.query_count += 1
        self.available_query_count += int(available)
        if source_owner is not None:
            self.owner_query_counts[int(source_owner)] += 1
        ranked = [int(value) for value in ranked_doc_indices]
        exact = (
            [int(value) for value in exact_topk_doc_indices if int(value) >= 0]
            if exact_topk_doc_indices is not None
            else None
        )
        self.exact_reference_query_count += int(exact is not None)
        for cutoff in RETRIEVAL_CUTOFFS:
            rank = next(
                (
                    position
                    for position, value in enumerate(ranked[:cutoff], start=1)
                    if value == target_doc_index
                ),
                None,
            )
            hit = int(rank is not None)
            self.hit_counts[cutoff] += hit
            self.available_hit_counts[cutoff] += int(available and hit)
            if rank is not None:
                self.reciprocal_rank_sums[cutoff] += 1.0 / rank
                self.ndcg_sums[cutoff] += 1.0 / math.log2(rank + 1.0)
            if source_owner is not None:
                self.owner_hit_counts[cutoff][int(source_owner)] += hit
            if exact is not None:
                self.exact_neighbor_overlap_sums[cutoff] += (
                    len(set(ranked[:cutoff]).intersection(exact[:cutoff])) / float(cutoff)
                )
        self.returned_id_count += int(returned_ids)
        self.unique_candidate_count += int(unique_candidates)
        self.message_count += int(messages)
        self.document_cosine_checks += int(document_checks)

    def summary(self) -> Dict[str, object]:
        denominator = max(1, int(self.query_count))
        summary: Dict[str, object] = {
            "query_count": int(self.query_count),
            "available_query_count": int(self.available_query_count),
            "exact_reference_query_count": int(self.exact_reference_query_count),
            "mean_returned_ids_before_deduplication": float(self.returned_id_count / denominator),
            "mean_unique_candidates_scored": float(self.unique_candidate_count / denominator),
            "mean_messages": float(self.message_count / denominator),
            "mean_document_cosine_checks": float(self.document_cosine_checks / denominator),
            "runtime_seconds": float(self.runtime_seconds),
            "process_peak_memory_mb_to_date": peak_memory_mb(),
        }
        for cutoff in RETRIEVAL_CUTOFFS:
            hit_rate = float(self.hit_counts[cutoff] / denominator)
            macro_owner_hit_rate = 0.0
            if self.owner_query_counts:
                macro_owner_hit_rate = float(
                    np.mean(
                        [
                            self.owner_hit_counts[cutoff][owner] / count
                            for owner, count in self.owner_query_counts.items()
                        ]
                    )
                )
            summary[f"hit_rate_at_{cutoff}"] = hit_rate
            summary[f"recall_at_{cutoff}"] = hit_rate
            summary[f"availability_conditioned_hit_rate_at_{cutoff}"] = (
                float(self.available_hit_counts[cutoff] / self.available_query_count)
                if self.available_query_count
                else 0.0
            )
            summary[f"macro_owner_hit_rate_at_{cutoff}"] = macro_owner_hit_rate
            summary[f"exact_neighbor_recall_at_{cutoff}"] = (
                float(
                    self.exact_neighbor_overlap_sums[cutoff]
                    / self.exact_reference_query_count
                )
                if self.exact_reference_query_count
                else np.nan
            )
            summary[f"exact_overlap_at_{cutoff}"] = summary[
                f"exact_neighbor_recall_at_{cutoff}"
            ]
        summary["precision_at_10"] = float(self.hit_counts[10] / (10.0 * denominator))
        summary["mrr_at_10"] = float(self.reciprocal_rank_sums[10] / denominator)
        summary["ndcg_at_10"] = float(self.ndcg_sums[10] / denominator)
        summary["mrr_at_100"] = float(self.reciprocal_rank_sums[100] / denominator)
        summary["ndcg_at_100"] = float(self.ndcg_sums[100] / denominator)
        return summary


def _topk_score_pairs(
    scores: np.ndarray,
    doc_indices: np.ndarray,
    doc_tie_ranks: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.flatnonzero(np.isfinite(scores))
    if len(finite) > int(top_k):
        finite_scores = scores[finite]
        boundary = len(finite_scores) - int(top_k)
        threshold = float(np.partition(finite_scores, boundary)[boundary])
        greater = finite[finite_scores > threshold]
        tied = finite[finite_scores == threshold]
        tied = tied[np.argsort(doc_tie_ranks[doc_indices[tied]], kind="stable")]
        positions = np.concatenate([greater, tied[: int(top_k) - len(greater)]])
    else:
        positions = finite
    order = np.lexsort((doc_tie_ranks[doc_indices[positions]], -scores[positions]))
    positions = positions[order][: int(top_k)]
    return doc_indices[positions].astype(np.int32, copy=False), scores[positions]


def exact_topk(
    dataset: PreparedDataset,
    query_embedding_indices: Sequence[int],
    corpus_doc_indices: Sequence[int],
    top_k: int,
    progress_every: int = 10_000,
    source_peer_positions: Optional[Sequence[int]] = None,
    sole_training_owner_by_doc: Optional[np.ndarray] = None,
    excluded_doc_indices: Optional[Sequence[int]] = None,
    output_path: Optional[os.PathLike[str] | str] = None,
    query_block_size: int = 64,
    doc_block_size: int = 25_000,
) -> np.ndarray:
    corpus = np.asarray(corpus_doc_indices, dtype=np.int64)
    queries = dataset.normalized_query_rows(query_embedding_indices)
    width = min(int(top_k), len(corpus))
    shape = (len(queries), width)
    result = (
        np.lib.format.open_memmap(output_path, mode="w+", dtype=np.int32, shape=shape)
        if output_path is not None
        else np.empty(shape, dtype=np.int32)
    )
    result[:] = -1
    tie_ranks = dataset.doc_tie_ranks
    sources = (
        np.asarray(source_peer_positions, dtype=np.int32)
        if source_peer_positions is not None
        else None
    )
    excluded = (
        np.asarray(excluded_doc_indices, dtype=np.int64)
        if excluded_doc_indices is not None
        else None
    )
    if excluded is not None and excluded.shape != (len(queries),):
        raise ValueError("Excluded document indices must align with query rows.")
    for query_start in range(0, len(queries), int(query_block_size)):
        query_end = min(query_start + int(query_block_size), len(queries))
        query_values = queries[query_start:query_end]
        best_docs = [np.empty(0, dtype=np.int32) for _ in range(len(query_values))]
        best_scores = [np.empty(0, dtype=np.float32) for _ in range(len(query_values))]
        for start in range(0, len(corpus), doc_block_size):
            block_indices = corpus[start : start + doc_block_size]
            values = np.asarray(dataset.doc_embeddings[block_indices], dtype=np.float32)
            values /= dataset.doc_norms[block_indices, None]
            score_block = np.asarray(query_values @ values.T, dtype=np.float32)
            if sources is not None and sole_training_owner_by_doc is not None:
                owners = sole_training_owner_by_doc[block_indices]
                for local_query, source in enumerate(sources[query_start:query_end]):
                    score_block[local_query, owners == int(source)] = -np.inf
            if excluded is not None:
                for local_query, excluded_doc in enumerate(
                    excluded[query_start:query_end]
                ):
                    if int(excluded_doc) >= 0:
                        score_block[
                            local_query, block_indices == int(excluded_doc)
                        ] = -np.inf
            for local_query in range(len(query_values)):
                incoming_docs, incoming_scores = _topk_score_pairs(
                    score_block[local_query], block_indices, tie_ranks, width
                )
                combined_docs = np.concatenate([best_docs[local_query], incoming_docs])
                combined_scores = np.concatenate([best_scores[local_query], incoming_scores])
                best_docs[local_query], best_scores[local_query] = _topk_score_pairs(
                    combined_scores, combined_docs.astype(np.int64), tie_ranks, width
                )
        for local_query, docs in enumerate(best_docs):
            result[query_start + local_query, : len(docs)] = docs
        if progress_every and query_end % int(progress_every) < int(query_block_size):
            print(f"Exact top-{width} completed {query_end}/{len(queries)} queries.", flush=True)
    if hasattr(result, "flush"):
        result.flush()
    return result


def exact_top10(
    dataset: PreparedDataset,
    query_embedding_indices: Sequence[int],
    corpus_doc_indices: Sequence[int],
    progress_every: int = 100,
    source_peer_positions: Optional[Sequence[int]] = None,
    sole_training_owner_by_doc: Optional[np.ndarray] = None,
    excluded_doc_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    return exact_topk(
        dataset,
        query_embedding_indices,
        corpus_doc_indices,
        top_k=10,
        progress_every=progress_every,
        source_peer_positions=source_peer_positions,
        sole_training_owner_by_doc=sole_training_owner_by_doc,
        excluded_doc_indices=excluded_doc_indices,
    )


def _exact_neighbor_cache_dir(cache_root: str, dataset: PreparedDataset) -> Path:
    return Path(cache_root) / "exact_top100" / dataset.dataset_name / f"seed_{dataset.seed}"


def prepare_exact_neighbor_cache(dataset: PreparedDataset, cache_root: str) -> Dict[str, object]:
    cache_dir = _exact_neighbor_cache_dir(cache_root, dataset)
    array_path = cache_dir / "exact_top100_doc_indices.npy"
    metadata_path = cache_dir / "validation.json"
    expected = {
        "protocol_id": MULTICUTOFF_PROTOCOL_ID,
        "dataset_name": dataset.dataset_name,
        "seed": int(dataset.seed),
        "query_count": int(dataset.query_count),
        "top_k": int(LOCAL_RESPONSE_K),
        "dtype": "int32",
        "query_manifest_sha256": dataset.validation.get("query_manifest_sha256"),
        "training_inventory_sha256": dataset.validation.get("training_inventory_sha256"),
        "manifest_config_sha256": dataset.validation.get("manifest_config_sha256"),
        "doc_embeddings_identity": dataset.validation.get("embedding_bundle_identity", {}).get(
            "doc_embeddings"
        ),
        "query_embeddings_identity": dataset.validation.get("embedding_bundle_identity", {}).get(
            "query_embeddings"
        ),
        "corpus_scope": (
            "unique_training_docs_excluding_docs_solely_owned_by_query_source"
            + (
                "_and_query_self_document"
                if dataset.validation.get("query_self_document_excluded", False)
                else ""
            )
        ),
    }
    if dataset.validation.get("query_self_document_excluded", False):
        expected["query_self_document_excluded"] = True
    if metadata_path.exists() and array_path.exists():
        observed = json.loads(metadata_path.read_text())
        if (
            all(observed.get(key) == value for key, value in expected.items())
            and observed.get("array_sha256") == _file_sha256(array_path)
        ):
            values = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if values.shape == (dataset.query_count, LOCAL_RESPONSE_K) and values.dtype == np.int32:
                return observed
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_suffix(".npy.tmp")
    exact_topk(
        dataset,
        dataset.manifest["query_embedding_index"].to_numpy(dtype=np.int64),
        dataset.train_doc_indices,
        top_k=LOCAL_RESPONSE_K,
        source_peer_positions=dataset.manifest["source_peer_position"].to_numpy(dtype=np.int32),
        sole_training_owner_by_doc=training_sole_owner_by_doc(dataset),
        excluded_doc_indices=(
            dataset.manifest["query_doc_index"].to_numpy(dtype=np.int32)
            if "query_doc_index" in dataset.manifest
            else None
        ),
        output_path=temporary,
    )
    temporary.replace(array_path)
    metadata = {
        **expected,
        "array_bytes": int(array_path.stat().st_size),
        "array_sha256": _file_sha256(array_path),
    }
    atomic_json(metadata_path, metadata)
    return metadata


def load_exact_neighbor_cache(dataset: PreparedDataset, cache_root: str) -> np.ndarray:
    cache_dir = _exact_neighbor_cache_dir(cache_root, dataset)
    metadata_path = cache_dir / "validation.json"
    array_path = cache_dir / "exact_top100_doc_indices.npy"
    if not metadata_path.exists() or not array_path.exists():
        raise FileNotFoundError(f"Missing exact-neighbor cache: {cache_dir}")
    metadata = json.loads(metadata_path.read_text())
    full_query_count = int(dataset.validation.get("query_manifest_count", dataset.query_count))
    expected = {
        "protocol_id": MULTICUTOFF_PROTOCOL_ID,
        "dataset_name": dataset.dataset_name,
        "seed": int(dataset.seed),
        "query_count": full_query_count,
        "top_k": int(LOCAL_RESPONSE_K),
        "query_manifest_sha256": dataset.validation.get("query_manifest_sha256"),
        "training_inventory_sha256": dataset.validation.get("training_inventory_sha256"),
        "manifest_config_sha256": dataset.validation.get("manifest_config_sha256"),
        "doc_embeddings_identity": dataset.validation.get("embedding_bundle_identity", {}).get(
            "doc_embeddings"
        ),
        "query_embeddings_identity": dataset.validation.get("embedding_bundle_identity", {}).get(
            "query_embeddings"
        ),
        "corpus_scope": (
            "unique_training_docs_excluding_docs_solely_owned_by_query_source"
            + (
                "_and_query_self_document"
                if dataset.validation.get("query_self_document_excluded", False)
                else ""
            )
        ),
    }
    if dataset.validation.get("query_self_document_excluded", False):
        expected["query_self_document_excluded"] = True
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Exact-neighbor cache validation failed: {metadata_path}")
    values = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if values.shape != (full_query_count, LOCAL_RESPONSE_K) or values.dtype != np.int32:
        raise ValueError(f"Malformed exact-neighbor cache: {values.shape} {values.dtype}")
    if metadata.get("array_sha256") != _file_sha256(array_path):
        raise ValueError("Exact-neighbor cache checksum failed.")
    manifest_rows = dataset.manifest["manifest_row"].to_numpy(dtype=np.int64)
    if np.any(manifest_rows < 0) or np.any(manifest_rows >= full_query_count):
        raise ValueError("Evaluation rows do not map into the immutable exact-neighbor cache.")
    selected = values[manifest_rows]
    if dataset.validation.get("query_self_document_excluded", False):
        query_docs = dataset.manifest["query_doc_index"].to_numpy(dtype=np.int32)
        for start in range(0, len(query_docs), 10_000):
            stop = min(start + 10_000, len(query_docs))
            positive = query_docs[start:stop] >= 0
            if np.any(
                positive[:, None]
                & (selected[start:stop] == query_docs[start:stop, None])
            ):
                raise ValueError("Exact-neighbor cache contains a query self-document.")
    return selected


def training_sole_owner_by_doc(dataset: PreparedDataset) -> np.ndarray:
    """Return sole owner position, or -2 for replicated docs and -1 for unowned docs."""
    owners = np.full(dataset.doc_embeddings.shape[0], -1, dtype=np.int32)
    for peer_position in range(dataset.peer_count):
        docs = dataset.peer_documents(peer_position).astype(np.int64)
        current = owners[docs]
        owners[docs[current == -1]] = int(peer_position)
        replicated = (current >= 0) & (current != int(peer_position))
        owners[docs[replicated]] = -2
    return owners


def deterministic_audit_rows(dataset: PreparedDataset, count: int = 2_000) -> pd.DataFrame:
    count = min(int(count), len(dataset.manifest))
    rng = np.random.default_rng(stable_int(f"exact-audit:{dataset.dataset_name}:{dataset.seed}"))
    positions = np.sort(rng.choice(len(dataset.manifest), size=count, replace=False))
    return dataset.manifest.iloc[positions].reset_index(drop=True)


def top10_overlap(approximate: np.ndarray, exact: np.ndarray) -> Dict[str, float]:
    if approximate.shape != exact.shape:
        raise ValueError("Approximate and exact result matrices must have equal shapes.")
    overlaps = [
        len(set(approximate[row].tolist()).intersection(exact[row].tolist())) / max(1, exact.shape[1])
        for row in range(exact.shape[0])
    ]
    exact_match = [set(approximate[row].tolist()) == set(exact[row].tolist()) for row in range(exact.shape[0])]
    return {
        "exact_audit_query_count": int(exact.shape[0]),
        "mean_exact_top10_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "exact_top10_set_match_rate": float(np.mean(exact_match)) if exact_match else 0.0,
    }


@dataclass
class CompactHashTable:
    projections: np.ndarray
    offsets: np.ndarray
    sorted_keys: np.ndarray
    sorted_doc_indices: np.ndarray

    def signature(self, vector: np.ndarray, width: float) -> np.ndarray:
        values = np.floor((self.projections[:, :-1] @ vector + self.projections[:, -1]) / width)
        return np.ascontiguousarray(values.astype(np.int32))

    @staticmethod
    def key_view(signatures: np.ndarray) -> np.ndarray:
        signatures = np.ascontiguousarray(signatures, dtype=np.int32)
        return signatures.view(np.dtype((np.void, signatures.shape[-1] * signatures.dtype.itemsize))).reshape(-1)

    def bucket(self, signature: np.ndarray) -> np.ndarray:
        key = self.key_view(signature.reshape(1, -1))[0]
        left = int(np.searchsorted(self.sorted_keys, key, side="left"))
        right = int(np.searchsorted(self.sorted_keys, key, side="right"))
        return self.sorted_doc_indices[left:right]


class CompactPStableIndex:
    def __init__(
        self,
        dataset: PreparedDataset,
        *,
        k: int,
        table_count: int,
        width: float,
        seed: int,
    ):
        self.dataset = dataset
        self.k = int(k)
        self.table_count = int(table_count)
        self.width = float(width)
        self.seed = int(seed)
        self.doc_indices = dataset.train_doc_indices.astype(np.int32, copy=False)
        self.tables: List[CompactHashTable] = []
        self.bucket_diagnostics: List[Dict[str, object]] = []
        self.build_runtime_seconds = 0.0
        self.owner_offsets, self.owner_positions = self._build_owner_index()
        self.doc_to_train_position = np.full(dataset.doc_embeddings.shape[0], -1, dtype=np.int32)
        self.doc_to_train_position[self.doc_indices.astype(np.int64)] = np.arange(
            len(self.doc_indices), dtype=np.int32
        )
        self._build()

    def _build_owner_index(self) -> Tuple[np.ndarray, np.ndarray]:
        owners: Dict[int, List[int]] = defaultdict(list)
        for peer_position in range(self.dataset.peer_count):
            for doc_index in self.dataset.peer_documents(peer_position):
                owners[int(doc_index)].append(peer_position)
        offsets = np.zeros(len(self.doc_indices) + 1, dtype=np.int64)
        values: List[np.ndarray] = []
        for position, doc_index in enumerate(self.doc_indices.tolist()):
            row = np.asarray(sorted(set(owners.get(int(doc_index), []))), dtype=np.int32)
            values.append(row)
            offsets[position + 1] = offsets[position] + len(row)
        return offsets, np.concatenate(values) if values else np.empty(0, dtype=np.int32)

    def _build(self) -> None:
        start_time = time.perf_counter()
        dimension = int(self.dataset.doc_embeddings.shape[1])
        rng = np.random.default_rng(self.seed)
        block_size = 25_000
        for table_index in range(self.table_count):
            a = rng.standard_normal((self.k, dimension), dtype=np.float32)
            b = rng.uniform(0.0, self.width, size=self.k).astype(np.float32)
            signatures = np.empty((len(self.doc_indices), self.k), dtype=np.int32)
            for start in range(0, len(self.doc_indices), block_size):
                end = min(start + block_size, len(self.doc_indices))
                indices = self.doc_indices[start:end].astype(np.int64)
                values = np.asarray(self.dataset.doc_embeddings[indices], dtype=np.float32)
                values /= self.dataset.doc_norms[indices, None]
                signatures[start:end] = np.floor((values @ a.T + b) / self.width).astype(np.int32)
            keys = CompactHashTable.key_view(signatures)
            order = np.argsort(keys, kind="stable")
            sorted_keys = keys[order]
            sorted_docs = self.doc_indices[order]
            changes = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
            starts = np.flatnonzero(changes)
            ends = np.r_[starts[1:], len(sorted_keys)]
            lengths = ends - starts
            largest_positions = sorted(
                range(len(starts)),
                key=lambda position: (-int(lengths[position]), bytes(sorted_keys[starts[position]])),
            )[:3]
            for bucket_rank, position in enumerate(largest_positions, start=1):
                raw_key = bytes(sorted_keys[starts[position]])
                self.bucket_diagnostics.append(
                    {
                        "table_index": int(table_index),
                        "bucket_rank_within_table": int(bucket_rank),
                        "bucket_fingerprint": hashlib.sha256(raw_key).hexdigest()[:20],
                        "document_count": int(lengths[position]),
                        "document_fraction": float(lengths[position] / max(1, len(self.doc_indices))),
                    }
                )
            self.tables.append(
                CompactHashTable(
                    projections=np.concatenate([a, b[:, None]], axis=1),
                    offsets=np.empty(0, dtype=np.int64),
                    sorted_keys=sorted_keys,
                    sorted_doc_indices=sorted_docs,
                )
            )
            del signatures, keys, order
        self.bucket_diagnostics.sort(
            key=lambda row: (-int(row["document_count"]), int(row["table_index"]), str(row["bucket_fingerprint"]))
        )
        for overall_rank, row in enumerate(self.bucket_diagnostics[:3], start=1):
            row["overall_bucket_rank"] = overall_rank
        self.bucket_diagnostics = self.bucket_diagnostics[:3]
        self.build_runtime_seconds = float(time.perf_counter() - start_time)

    @property
    def largest_bucket_fraction(self) -> float:
        return float(self.bucket_diagnostics[0]["document_fraction"]) if self.bucket_diagnostics else 0.0

    @property
    def in_memory_index_size_estimate_bytes(self) -> int:
        table_bytes = sum(
            table.projections.nbytes + table.sorted_keys.nbytes + table.sorted_doc_indices.nbytes
            for table in self.tables
        )
        return int(
            table_bytes
            + self.doc_indices.nbytes
            + self.owner_offsets.nbytes
            + self.owner_positions.nbytes
            + self.doc_to_train_position.nbytes
        )

    def has_non_source_owner(self, doc_index: int, source_peer: Optional[int]) -> bool:
        if source_peer is None:
            return True
        train_position = int(self.doc_to_train_position[int(doc_index)])
        if train_position < 0:
            return False
        start = int(self.owner_offsets[train_position])
        end = int(self.owner_offsets[train_position + 1])
        owners = self.owner_positions[start:end]
        return bool(np.any(owners != int(source_peer)))

    def retrieve(
        self,
        query_embedding: np.ndarray,
        candidate_cap: Optional[int],
        source_peer: Optional[int] = None,
        collision_limit: Optional[int] = None,
        excluded_doc_index: int = -1,
    ) -> Tuple[List[int], int, int, int]:
        candidates: List[int] = []
        seen: set[int] = set()
        returned = 0
        requests = 0
        for table in self.tables:
            requests += 1
            signature = table.signature(query_embedding, self.width)
            bucket = table.bucket(signature)
            for doc_index in bucket:
                if collision_limit is not None and returned >= int(collision_limit):
                    break
                value = int(doc_index)
                returned += 1
                if value == int(excluded_doc_index):
                    continue
                if not self.has_non_source_owner(value, source_peer):
                    continue
                if value not in seen:
                    seen.add(value)
                    candidates.append(value)
                    if candidate_cap is not None and len(candidates) >= int(candidate_cap):
                        break
            if candidate_cap is not None and len(candidates) >= int(candidate_cap):
                break
            if collision_limit is not None and returned >= int(collision_limit):
                break
        if not candidates:
            return [], returned, 0, requests
        candidate_indices = np.asarray(candidates, dtype=np.int64)
        values = np.asarray(self.dataset.doc_embeddings[candidate_indices], dtype=np.float32)
        scores = (values @ query_embedding) / self.dataset.doc_norms[candidate_indices]
        ranked = sorted(
            ((int(index), float(score)) for index, score in zip(candidate_indices.tolist(), scores.tolist())),
            key=lambda item: (-item[1], str(self.dataset.doc_ids[item[0]])),
        )[:LOCAL_RESPONSE_K]
        return [index for index, _score in ranked], returned, len(candidate_indices), requests


_PSTABLE_WORKER_STATE: Dict[str, object] = {}


def _initialize_pstable_worker(
    index: CompactPStableIndex,
    query_embeddings: np.ndarray,
    target_doc_indices: np.ndarray,
    query_doc_indices: np.ndarray,
    availability: np.ndarray,
    source_peers: np.ndarray,
    manifest_rows: np.ndarray,
    audit_manifest_rows: set[int],
    candidate_cap: Optional[int],
    collision_limit: Optional[int],
    exact_topk_doc_indices: Optional[np.ndarray],
    start_barrier: Optional[object] = None,
    ready_queue: Optional[object] = None,
) -> None:
    global _PSTABLE_WORKER_STATE
    _PSTABLE_WORKER_STATE = {
        "index": index,
        "query_embeddings": query_embeddings,
        "target_doc_indices": target_doc_indices,
        "query_doc_indices": query_doc_indices,
        "availability": availability,
        "source_peers": source_peers,
        "manifest_rows": manifest_rows,
        "audit_manifest_rows": audit_manifest_rows,
        "candidate_cap": candidate_cap,
        "collision_limit": collision_limit,
        "exact_topk_doc_indices": exact_topk_doc_indices,
        "start_barrier": start_barrier,
    }
    if ready_queue is not None:
        ready_queue.put(os.getpid())


def _evaluate_pstable_worker(
    payload: Tuple[int, int, int],
) -> Tuple[int, int, RankedMetrics, List[Tuple[int, List[int]]], float]:
    chunk_index, start, end = payload
    state = _PSTABLE_WORKER_STATE
    index = state["index"]
    query_embeddings = state["query_embeddings"]
    target_doc_indices = state["target_doc_indices"]
    query_doc_indices = state["query_doc_indices"]
    availability = state["availability"]
    source_peers = state["source_peers"]
    manifest_rows = state["manifest_rows"]
    audit_manifest_rows = state["audit_manifest_rows"]
    candidate_cap = state["candidate_cap"]
    collision_limit = state["collision_limit"]
    exact_topk_doc_indices = state["exact_topk_doc_indices"]
    start_barrier = state["start_barrier"]
    if start_barrier is not None:
        start_barrier.wait(timeout=120.0)
    metrics = RankedMetrics()
    audit_rankings: List[Tuple[int, List[int]]] = []
    for position in range(int(start), int(end)):
        ranked, returned, unique_candidates, requests = index.retrieve(
            query_embeddings[position],
            candidate_cap,
            source_peer=int(source_peers[position]),
            collision_limit=collision_limit,
            excluded_doc_index=int(query_doc_indices[position]),
        )
        metrics.add(
            ranked,
            int(target_doc_indices[position]),
            bool(availability[position]),
            returned_ids=returned,
            unique_candidates=unique_candidates,
            messages=requests,
            document_checks=unique_candidates,
            source_owner=int(source_peers[position]),
            exact_topk_doc_indices=(
                exact_topk_doc_indices[position]
                if exact_topk_doc_indices is not None
                else None
            ),
        )
        manifest_row = int(manifest_rows[position])
        if manifest_row in audit_manifest_rows:
            audit_rankings.append(
                (manifest_row, ranked[:10] + [-1] * max(0, 10 - len(ranked)))
            )
    return chunk_index, os.getpid(), metrics, audit_rankings, peak_memory_mb()


def evaluate_pstable_configuration(
    dataset: PreparedDataset,
    *,
    k: int,
    table_count: int,
    width: float,
    candidate_cap: Optional[int],
    seed: int,
    exact_audit_rows: Optional[pd.DataFrame] = None,
    exact_audit_top10: Optional[np.ndarray] = None,
    prebuilt_index: Optional[CompactPStableIndex] = None,
    collision_limit: Optional[int] = None,
    workers: int = 1,
    exact_topk_doc_indices: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    if exact_topk_doc_indices is not None and exact_topk_doc_indices.shape != (
        len(dataset.manifest),
        LOCAL_RESPONSE_K,
    ):
        raise ValueError("P-stable exact-neighbor rows must align with the query manifest.")
    owns_index = prebuilt_index is None
    index = prebuilt_index or CompactPStableIndex(
        dataset, k=k, table_count=table_count, width=width, seed=seed
    )
    if (index.k, index.table_count, index.width, index.seed) != (
        int(k), int(table_count), float(width), int(seed)
    ):
        raise ValueError("Prebuilt P-stable index does not match the requested configuration.")
    base = {
        "dataset_name": dataset.dataset_name,
        "seed": int(seed),
        "method": "pstable_lsh_dht",
        "pstable_k": int(k),
        "pstable_L": int(table_count),
        "pstable_r": float(width),
        "candidate_policy": (
            f"collision_limit_{int(collision_limit)}"
            if collision_limit is not None
            else "all_collisions"
            if candidate_cap is None
            else f"cap_{int(candidate_cap)}"
        ),
        "unique_training_doc_count": int(len(index.doc_indices)),
        "largest_bucket_fraction": index.largest_bucket_fraction,
        "build_runtime_seconds": index.build_runtime_seconds,
        "in_memory_index_size_estimate_bytes": index.in_memory_index_size_estimate_bytes,
        "query_workers_requested": int(workers),
        "protocol_id": MULTICUTOFF_PROTOCOL_ID,
        "retrieval_cutoffs": tuple(RETRIEVAL_CUTOFFS),
        "local_response_k": int(LOCAL_RESPONSE_K),
        "query_self_document_excluded": bool(
            dataset.validation.get("query_self_document_excluded", False)
        ),
    }
    diagnostics = [{**base, **row} for row in index.bucket_diagnostics]
    if index.largest_bucket_fraction > 0.30:
        return {
            **base,
            "status": "skipped_oversized_bucket",
            "query_count": 0,
            "query_worker_count": 0,
            "peak_memory_mb": peak_memory_mb(),
        }, diagnostics
    metrics = RankedMetrics()
    start = time.perf_counter()
    query_embeddings = dataset.normalized_query_rows(
        dataset.manifest["query_embedding_index"].to_numpy(dtype=np.int64)
    )
    manifest_rows = dataset.manifest["manifest_row"].to_numpy(dtype=np.int64)
    target_doc_indices = dataset.manifest["target_doc_index"].to_numpy(dtype=np.int32)
    query_doc_indices = (
        dataset.manifest["query_doc_index"].to_numpy(dtype=np.int32)
        if "query_doc_index" in dataset.manifest
        else np.full(len(dataset.manifest), -1, dtype=np.int32)
    )
    availability = dataset.manifest["target_available_in_training_network"].to_numpy(dtype=bool)
    source_peers = dataset.manifest["source_peer_position"].to_numpy(dtype=np.int32)
    audit_order = (
        exact_audit_rows["manifest_row"].astype(int).tolist()
        if exact_audit_rows is not None
        else []
    )
    audit_manifest_rows = set(audit_order)
    audit_by_manifest_row: Dict[int, List[int]] = {}
    workers = max(1, min(int(workers), len(dataset.manifest)))
    worker_pids: set[int] = set()
    initialized_worker_pids: set[int] = set()
    worker_peaks_by_pid: Dict[int, float] = {}
    if workers == 1:
        _initialize_pstable_worker(
            index,
            query_embeddings,
            target_doc_indices,
            query_doc_indices,
            availability,
            source_peers,
            manifest_rows,
            audit_manifest_rows,
            candidate_cap,
            collision_limit,
            exact_topk_doc_indices,
            None,
            None,
        )
        _chunk, worker_pid, partial, audit_rankings, worker_peak = _evaluate_pstable_worker(
            (0, 0, len(dataset.manifest))
        )
        metrics.merge(partial)
        worker_pids.add(worker_pid)
        initialized_worker_pids.add(worker_pid)
        worker_peaks_by_pid[worker_pid] = worker_peak
        audit_by_manifest_row.update(audit_rankings)
        print(
            f"P-stable completed {len(dataset.manifest)}/{len(dataset.manifest)} query events.",
            flush=True,
        )
    else:
        import multiprocessing as mp

        try:
            context = mp.get_context("fork")
        except ValueError:
            context = mp.get_context()
        chunk_count = min(len(dataset.manifest), workers)
        boundaries = np.linspace(0, len(dataset.manifest), num=chunk_count + 1, dtype=np.int64)
        chunks = [
            (chunk_index, int(chunk_start), int(chunk_end))
            for chunk_index, (chunk_start, chunk_end) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            )
            if int(chunk_end) > int(chunk_start)
        ]
        completed_queries = 0
        next_progress = 10_000
        partials_by_chunk: Dict[
            int, Tuple[RankedMetrics, List[Tuple[int, List[int]]]]
        ] = {}
        ready_queue = context.Queue()
        start_barrier = context.Barrier(workers)
        with context.Pool(
            processes=workers,
            initializer=_initialize_pstable_worker,
            initargs=(
                index,
                query_embeddings,
                target_doc_indices,
                query_doc_indices,
                availability,
                source_peers,
                manifest_rows,
                audit_manifest_rows,
                candidate_cap,
                collision_limit,
                exact_topk_doc_indices,
                start_barrier,
                ready_queue,
            ),
        ) as pool:
            while len(initialized_worker_pids) < workers:
                initialized_worker_pids.add(int(ready_queue.get(timeout=60.0)))
            for _chunk, worker_pid, partial, audit_rankings, worker_peak in pool.imap_unordered(
                _evaluate_pstable_worker,
                chunks,
                chunksize=1,
            ):
                worker_pids.add(worker_pid)
                worker_peaks_by_pid[worker_pid] = max(
                    worker_peaks_by_pid.get(worker_pid, 0.0), float(worker_peak)
                )
                partials_by_chunk[int(_chunk)] = (partial, audit_rankings)
                completed_queries += int(partial.query_count)
                while completed_queries >= next_progress:
                    print(
                        f"P-stable completed at least {next_progress}/{len(dataset.manifest)} "
                        "query events.",
                        flush=True,
                    )
                    next_progress += 10_000
        ready_queue.close()
        ready_queue.join_thread()
        for chunk_index in sorted(partials_by_chunk):
            partial, audit_rankings = partials_by_chunk[chunk_index]
            metrics.merge(partial)
            audit_by_manifest_row.update(audit_rankings)
    metrics.runtime_seconds = float(time.perf_counter() - start)
    summary = {
        **base,
        "status": "completed",
        **metrics.summary(),
        "query_worker_count": int(len(worker_pids)),
        "query_worker_processes_initialized": int(len(initialized_worker_pids)),
        "query_worker_processes_observed": int(len(worker_pids)),
        "process_peak_memory_mb_max": float(
            max([peak_memory_mb(), *worker_peaks_by_pid.values()])
        ),
        "process_peak_memory_mb_sum_upper_bound": float(
            peak_memory_mb() + sum(worker_peaks_by_pid.values())
        ),
        "process_peak_memory_note": (
            "max is the largest per-process RSS; sum_upper_bound double-counts "
            "forked copy-on-write shared pages"
        ),
    }
    if exact_audit_top10 is not None:
        missing_audit_rows = [row for row in audit_order if row not in audit_by_manifest_row]
        if missing_audit_rows:
            raise RuntimeError(
                f"P-stable workers omitted {len(missing_audit_rows)} exact-audit query rows."
            )
        audit_approximate = np.asarray(
            [audit_by_manifest_row[row] for row in audit_order], dtype=np.int32
        )
        summary.update(top10_overlap(audit_approximate, exact_audit_top10))
    if workers == 1:
        _PSTABLE_WORKER_STATE.clear()
    if owns_index:
        del index
        gc.collect()
    return summary, diagnostics


def evaluate_hnsw_grid(
    dataset: PreparedDataset,
    *,
    seed: int,
    m_values: Sequence[int] = (8, 16, 32, 48),
    ef_construction_values: Sequence[int] = (40, 100, 200, 500),
    ef_search_values: Sequence[int] = (10, 25, 50, 100, 200, 500),
    audit_count: int = 2_000,
    workers: int = 1,
    exact_topk_doc_indices: Optional[np.ndarray] = None,
    corpus_scope: str = "network_training",
    progress_prefix: str = "",
) -> pd.DataFrame:
    try:
        import hnswlib
    except ImportError as exc:
        raise ImportError("Install hnswlib before running the HNSW benchmark.") from exc
    if corpus_scope == "full_source_corpus":
        _doc_to_position, corpus = canonical_doc_id_positions(dataset.doc_ids)
        del _doc_to_position
        expected_corpus_size = int(dataset.validation.get("source_corpus_unique_doc_count", -1))
        if expected_corpus_size != len(corpus):
            raise RuntimeError("HNSW canonical document corpus does not match dataset validation.")
        sole_owners = None
    elif corpus_scope == "network_training":
        corpus = dataset.train_doc_indices.astype(np.int32, copy=False)
        expected_corpus_size = int(dataset.validation.get("unique_training_doc_count", -1))
        if expected_corpus_size != len(corpus):
            raise RuntimeError("HNSW training corpus does not match dataset validation.")
        sole_owners = training_sole_owner_by_doc(dataset)
    else:
        raise ValueError(f"Unsupported HNSW corpus scope: {corpus_scope}")
    workers = max(1, int(workers))
    if exact_topk_doc_indices is not None and exact_topk_doc_indices.shape != (
        len(dataset.manifest),
        LOCAL_RESPONSE_K,
    ):
        raise ValueError("HNSW exact-neighbor rows must align with the query manifest.")
    audit_rows = deterministic_audit_rows(dataset, audit_count)
    exact_audit = exact_top10(
        dataset,
        audit_rows["query_embedding_index"].to_numpy(dtype=np.int64),
        corpus,
        source_peer_positions=(
            audit_rows["source_peer_position"].to_numpy(dtype=np.int64)
            if sole_owners is not None
            else None
        ),
        sole_training_owner_by_doc=sole_owners,
        excluded_doc_indices=(
            audit_rows["query_doc_index"].to_numpy(dtype=np.int32)
            if "query_doc_index" in audit_rows
            else None
        ),
    )
    queries = dataset.normalized_query_rows(
        dataset.manifest["query_embedding_index"].to_numpy(dtype=np.int64)
    )
    query_doc_indices = (
        dataset.manifest["query_doc_index"].to_numpy(dtype=np.int32)
        if "query_doc_index" in dataset.manifest
        else np.full(len(dataset.manifest), -1, dtype=np.int32)
    )
    audit_positions = audit_rows["manifest_row"].to_numpy(dtype=np.int64)
    rows: List[Dict[str, object]] = []
    dimension = int(dataset.doc_embeddings.shape[1])
    result_width = min(LOCAL_RESPONSE_K, len(corpus))
    add_block = 25_000
    for m in m_values:
        for ef_construction in ef_construction_values:
            if int(ef_construction) < int(m):
                continue
            build_start = time.perf_counter()
            index = hnswlib.Index(space="cosine", dim=dimension)
            index.init_index(
                max_elements=len(corpus),
                ef_construction=int(ef_construction),
                M=int(m),
                random_seed=int(seed),
            )
            for start in range(0, len(corpus), add_block):
                indices = corpus[start : start + add_block].astype(np.int64)
                values = np.asarray(dataset.doc_embeddings[indices], dtype=np.float32)
                values /= dataset.doc_norms[indices, None]
                index.add_items(values, indices, num_threads=workers)
            build_runtime = float(time.perf_counter() - build_start)
            for ef_search in ef_search_values:
                effective_ef_search = max(LOCAL_RESPONSE_K, int(ef_search))
                index.set_ef(effective_ef_search)
                query_start = time.perf_counter()
                metrics = RankedMetrics()
                labels = np.full((len(queries), LOCAL_RESPONSE_K), -1, dtype=np.int32)
                query_block_size = 10_000
                max_filter_request_k = result_width
                for query_start_position in range(0, len(queries), query_block_size):
                    index.set_ef(effective_ef_search)
                    query_end_position = min(query_start_position + query_block_size, len(queries))
                    block_queries = queries[query_start_position:query_end_position]
                    block_sources = dataset.manifest.iloc[
                        query_start_position:query_end_position
                    ]["source_peer_position"].to_numpy(dtype=np.int32)
                    block_query_docs = query_doc_indices[
                        query_start_position:query_end_position
                    ]
                    block_rankings = np.full(
                        (len(block_queries), LOCAL_RESPONSE_K), -1, dtype=np.int32
                    )
                    pending = np.arange(len(block_queries), dtype=np.int64)
                    request_k = result_width
                    while len(pending):
                        if request_k > effective_ef_search:
                            index.set_ef(request_k)
                        max_filter_request_k = max(max_filter_request_k, request_k)
                        approximate, _distances = index.knn_query(
                            block_queries[pending],
                            k=request_k,
                            num_threads=workers,
                        )
                        still_pending = []
                        for pending_position, ranked in zip(pending.tolist(), approximate):
                            accepted = []
                            seen = set()
                            for doc_index in ranked.tolist():
                                doc_index = int(doc_index)
                                if doc_index in seen:
                                    continue
                                seen.add(doc_index)
                                if doc_index == int(
                                    block_query_docs[pending_position]
                                ):
                                    continue
                                if (
                                    sole_owners is not None
                                    and int(sole_owners[doc_index])
                                    == int(block_sources[pending_position])
                                ):
                                    continue
                                accepted.append(doc_index)
                                if len(accepted) == result_width:
                                    break
                            block_rankings[pending_position, : len(accepted)] = accepted
                            if len(accepted) < result_width and request_k < len(corpus):
                                still_pending.append(pending_position)
                        if not still_pending:
                            break
                        pending = np.asarray(still_pending, dtype=np.int64)
                        request_k = min(len(corpus), max(request_k + 1, request_k * 2))
                    labels[query_start_position:query_end_position] = block_rankings
                    records = dataset.manifest.iloc[query_start_position:query_end_position].itertuples(index=False)
                    for block_position, (record, ranked) in enumerate(zip(records, block_rankings)):
                        eligible = ranked[ranked >= 0]
                        metrics.add(
                            eligible.tolist(),
                            int(record.target_doc_index),
                            bool(record.target_available_in_training_network),
                            returned_ids=len(eligible),
                            unique_candidates=len(eligible),
                            messages=0,
                            document_checks=0,
                            source_owner=int(record.source_peer_position),
                            exact_topk_doc_indices=(
                                exact_topk_doc_indices[query_start_position + block_position]
                                if exact_topk_doc_indices is not None
                                else None
                            ),
                        )
                    print(
                        f"{progress_prefix}HNSW M={m} efC={ef_construction} "
                        f"efS={ef_search} completed "
                        f"{query_end_position}/{len(queries)} query events.",
                        flush=True,
                    )
                metrics.runtime_seconds = float(time.perf_counter() - query_start)
                audit_width = exact_audit.shape[1]
                audit_approximate = labels[audit_positions, :audit_width]
                estimate = int(len(corpus) * (dimension * 4 + int(m) * 2 * 4 + 64))
                row = {
                    "dataset_name": dataset.dataset_name,
                    "seed": int(seed),
                    "method": "hnsw_centralized_upper_bound",
                    "M": int(m),
                    "ef_construction": int(ef_construction),
                    "ef_search": int(ef_search),
                    "construction_threads": int(workers),
                    "query_threads": int(workers),
                    "requested_ef_search": int(ef_search),
                    "effective_ef_search": int(effective_ef_search),
                    "max_effective_ef_search_after_source_filtering": int(
                        max(effective_ef_search, max_filter_request_k)
                    ),
                    "max_ann_candidates_requested_after_source_filtering": int(
                        max_filter_request_k
                    ),
                    "protocol_id": MULTICUTOFF_PROTOCOL_ID,
                    "retrieval_cutoffs": tuple(RETRIEVAL_CUTOFFS),
                    "local_response_k": int(LOCAL_RESPONSE_K),
                    "build_runtime_seconds": build_runtime,
                    "index_size_estimate_bytes": estimate,
                    "centralized_upper_performance_bound": True,
                    "hnsw_corpus_scope": corpus_scope,
                    "source_only_documents_eligible": bool(sole_owners is None),
                    "query_self_document_excluded": bool(
                        dataset.validation.get(
                            "query_self_document_excluded", False
                        )
                    ),
                    "exact_overlap_reference_corpus": (
                        "network_training_documents_excluding_source_only_ownership"
                        if sole_owners is not None
                        else "full_source_corpus"
                    ),
                    "exact_overlap_retrieval_corpus": (
                        "network_training_documents_excluding_source_only_ownership"
                        if sole_owners is not None
                        else "full_source_corpus"
                    ),
                    **metrics.summary(),
                    **top10_overlap(audit_approximate.astype(np.int32), exact_audit),
                }
                for field_name in ("mean_messages", "mean_document_cosine_checks"):
                    row[field_name] = np.nan
                row["communication_costs_applicable"] = False
                rows.append(row)
                print(
                    f"HNSW M={m} efC={ef_construction} efS={ef_search} "
                    f"hit@10={row['hit_rate_at_10']:.6f}",
                    flush=True,
                )
            del index
            gc.collect()
    return pd.DataFrame(rows)


def diffuse_peer_embeddings(
    dataset: PreparedDataset,
    graph: CompactGraph,
    *,
    alpha: float,
    relative_tolerance: float = 1e-5,
    max_iterations: int = 200,
) -> Tuple[CentroidLayout, Dict[str, object]]:
    try:
        from scipy import sparse
    except ImportError as exc:
        raise ImportError("scipy is required for graph diffusion.") from exc
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("Diffusion alpha must be in (0, 1].")
    start = time.perf_counter()
    dimension = int(dataset.doc_embeddings.shape[1])
    personalization = np.zeros((dataset.peer_count, dimension), dtype=np.float32)
    for peer_position in range(dataset.peer_count):
        docs = dataset.peer_documents(peer_position).astype(np.int64)
        values = np.asarray(dataset.doc_embeddings[docs], dtype=np.float32)
        values /= dataset.doc_norms[docs, None]
        personalization[peer_position] = np.sum(values, axis=0)
    degrees = np.diff(graph.offsets).astype(np.float32)
    row_indices = np.repeat(np.arange(graph.peer_count, dtype=np.int32), degrees.astype(np.int32))
    edge_weights = 1.0 / np.maximum(1.0, degrees[row_indices])
    transition = sparse.csr_matrix(
        (edge_weights, (row_indices, graph.neighbors)),
        shape=(graph.peer_count, graph.peer_count),
        dtype=np.float32,
    )
    current = personalization.copy()
    converged = alpha == 1.0
    iterations = 0
    final_relative_change = 0.0
    if not converged:
        for iteration in range(1, int(max_iterations) + 1):
            updated = alpha * personalization + (1.0 - alpha) * transition.dot(current)
            denominator = max(float(np.linalg.norm(current)), 1e-12)
            final_relative_change = float(np.linalg.norm(updated - current) / denominator)
            current = np.asarray(updated, dtype=np.float32)
            iterations = iteration
            if final_relative_change <= float(relative_tolerance):
                converged = True
                break
    if not converged:
        raise RuntimeError(
            f"Graph diffusion did not converge in {max_iterations} iterations "
            f"(relative change={final_relative_change:.6g})."
        )
    diffused = normalize_rows(current)
    offsets = np.arange(dataset.peer_count + 1, dtype=np.int64)
    layout = CentroidLayout(
        embeddings=diffused,
        offsets=offsets,
        cluster_doc_counts=np.diff(dataset.inventory_offsets).astype(np.int32),
        contact_caps=np.diff(graph.offsets).astype(np.int32),
        method=f"graph_diffusion_alpha_{alpha:g}",
    )
    edge_count = int(len(graph.neighbors))
    diagnostics = {
        "alpha": alpha,
        "relative_tolerance": float(relative_tolerance),
        "max_iterations": int(max_iterations),
        "iterations": int(iterations),
        "converged": True,
        "final_relative_change": float(final_relative_change),
        "construction_messages": int(edge_count * iterations),
        "transmitted_vector_bytes": int(edge_count * iterations * dimension * 4),
        "runtime_seconds": float(time.perf_counter() - start),
        "peak_memory_mb": peak_memory_mb(),
        "directed_graph_preserved": True,
        "personalization": "sum_of_normalized_training_document_embeddings",
    }
    return layout, diagnostics

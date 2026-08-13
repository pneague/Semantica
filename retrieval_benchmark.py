"""Compact top-k retrieval benchmark for Semantica, P-stable LSH, and HNSW.

The benchmark intentionally writes only aggregate metrics by default.  Datasets
are represented as documents, queries, qrels, normalized embeddings, and a
synthetic peer assignment used by the Semantica-style retrievers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_BEIR_DATASETS = ("scifact", "nfcorpus", "fiqa")
DEFAULT_RETRIEVAL_DATASETS = DEFAULT_BEIR_DATASETS + ("aol-ia",)
DEFAULT_AOL_IA_CACHE_DIR = "/var/scratch/pmneague/Low_Coord_Semantic_DHT/cache/aol_ia_peer_graph"
DEFAULT_METHODS = (
    "semantica_random",
    "semantica_tree",
    "semantica_fixed_k",
    "semantica_elbow",
    "pstable_dht",
    "hnsw",
    "kgraph_nndescent",
)


def stable_int(value: object, modulo: int = 2**32) -> int:
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_").lower()


def _file_signature(path: str) -> str:
    abspath = os.path.abspath(path)
    if os.path.exists(path):
        stat = os.stat(path)
        token = f"{abspath}:{stat.st_size}:{stat.st_mtime_ns}"
    else:
        token = abspath
    return hashlib.md5(token.encode("utf-8")).hexdigest()[:12]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D embedding matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[~np.isfinite(norms)] = 1.0
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _download_zip(url: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading {url} -> {output_path}", flush=True)
    urllib.request.urlretrieve(url, output_path)


def _extract_zip(zip_path: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


@dataclass
class RetrievalDataset:
    name: str
    documents: pd.DataFrame
    queries: pd.DataFrame
    qrels: Dict[str, Set[str]]
    doc_embeddings: np.ndarray
    query_embeddings: np.ndarray
    peer_assignments: Optional["PeerAssignments"] = None

    def __post_init__(self) -> None:
        self.documents = self.documents.reset_index(drop=True).copy()
        self.queries = self.queries.reset_index(drop=True).copy()
        self.documents["doc_id"] = self.documents["doc_id"].astype(str)
        self.queries["query_id"] = self.queries["query_id"].astype(str)
        self.qrels = {
            str(query_id): {str(doc_id) for doc_id in doc_ids}
            for query_id, doc_ids in self.qrels.items()
        }
        self.doc_embeddings = normalize_rows(self.doc_embeddings)
        self.query_embeddings = normalize_rows(self.query_embeddings)
        if self.documents.shape[0] != self.doc_embeddings.shape[0]:
            raise ValueError("documents and doc_embeddings have different lengths.")
        if self.queries.shape[0] != self.query_embeddings.shape[0]:
            raise ValueError("queries and query_embeddings have different lengths.")

    @property
    def doc_ids(self) -> List[str]:
        return self.documents["doc_id"].tolist()

    @property
    def query_ids(self) -> List[str]:
        return self.queries["query_id"].tolist()

    @property
    def doc_id_to_pos(self) -> Dict[str, int]:
        return {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}

    def indexed_qrels(self) -> Dict[str, Set[str]]:
        indexed_doc_ids = set(self.doc_ids)
        return {
            query_id: set(doc_ids).intersection(indexed_doc_ids)
            for query_id, doc_ids in self.qrels.items()
        }

    def validation_summary(self) -> Dict[str, object]:
        indexed_doc_ids = set(self.doc_ids)
        query_ids = set(self.query_ids)
        raw_qrel_pairs = sum(len(values) for values in self.qrels.values())
        indexed_qrels = self.indexed_qrels()
        indexed_qrel_pairs = sum(len(values) for values in indexed_qrels.values())
        query_count_with_indexed_relevance = sum(1 for values in indexed_qrels.values() if values)
        return {
            "dataset_name": self.name,
            "documents": int(self.documents.shape[0]),
            "queries": int(self.queries.shape[0]),
            "qrel_queries": int(len(self.qrels)),
            "qrel_pairs": int(raw_qrel_pairs),
            "indexed_qrel_pairs": int(indexed_qrel_pairs),
            "missing_qrel_pairs": int(raw_qrel_pairs - indexed_qrel_pairs),
            "queries_without_indexed_relevance": int(
                sum(1 for query_id in query_ids if not indexed_qrels.get(query_id, set()))
            ),
            "queries_with_indexed_relevance": int(query_count_with_indexed_relevance),
            "embedding_dim": int(self.doc_embeddings.shape[1]) if self.doc_embeddings.size else 0,
            "peer_count": int(len(self.peer_assignments.peer_ids)) if self.peer_assignments else 0,
        }


@dataclass
class PeerAssignments:
    doc_peer_ids: np.ndarray
    peer_ids: List[int]
    peer_embeddings: np.ndarray
    peer_doc_indices: Dict[int, np.ndarray]


@dataclass
class RetrievalResult:
    doc_indices: List[int]
    scores: List[float]
    touched_peers: int
    candidate_docs_scored: int


@dataclass
class BenchmarkConfig:
    top_k: int = 10
    seed: int = 42
    target_docs_per_peer: int = 50
    max_peers: int = 5000
    semantica_contact_peers: int = 50
    semantica_leaf_size: int = 50
    semantica_delta: float = 5e-4
    semantica_fixed_k: int = 2
    semantica_elbow_max_k: int = 8
    pstable_k: int = 6
    pstable_l: int = 64
    pstable_w: float = 0.75
    pstable_candidate_cap: int = 2500
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100
    kgraph_index_k: int = 25
    kgraph_index_l: int = 100
    kgraph_index_iterations: int = 30
    kgraph_index_recall: float = 0.99
    kgraph_index_delta: float = 0.002
    kgraph_index_reverse: int = -1
    kgraph_search_p: int = 100
    kgraph_search_m: int = 0
    kgraph_search_s: int = 10
    kgraph_search_t: int = 1
    kgraph_threads: int = 0
    kgraph_cache_root: Optional[str] = None
    aol_ia_cache_dir: str = DEFAULT_AOL_IA_CACHE_DIR
    aol_ia_top_users: int = 10000
    aol_ia_top_user_metric: str = "query_count"
    aol_ia_nbr_train_recs: int = 20
    aol_ia_nbr_test_records: int = 10
    aol_ia_doc_embeddings_path: Optional[str] = None


def parse_beir_dir(dataset_dir: str, split: str = "test") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    corpus_path = os.path.join(dataset_dir, "corpus.jsonl")
    queries_path = os.path.join(dataset_dir, "queries.jsonl")
    qrels_path = os.path.join(dataset_dir, "qrels", f"{split}.tsv")
    if not os.path.exists(qrels_path):
        qrels_dir = os.path.join(dataset_dir, "qrels")
        candidates = sorted(
            os.path.join(qrels_dir, filename)
            for filename in os.listdir(qrels_dir)
            if filename.endswith(".tsv")
        )
        if not candidates:
            raise FileNotFoundError(f"No BEIR qrels TSV found under {qrels_dir}")
        qrels_path = candidates[0]
    for path in (corpus_path, queries_path, qrels_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    corpus_rows = _read_jsonl(corpus_path)
    query_rows = _read_jsonl(queries_path)
    documents = pd.DataFrame(
        {
            "doc_id": str(row.get("_id", "")),
            "title": str(row.get("title", "") or ""),
            "text": str(row.get("text", "") or ""),
        }
        for row in corpus_rows
        if row.get("_id") is not None
    )
    documents["full_text"] = (documents["title"].fillna("") + ". " + documents["text"].fillna("")).str.strip()
    queries = pd.DataFrame(
        {
            "query_id": str(row.get("_id", "")),
            "text": str(row.get("text", "") or ""),
        }
        for row in query_rows
        if row.get("_id") is not None
    )

    qrels_df = pd.read_csv(qrels_path, sep="\t")
    if not {"query-id", "corpus-id", "score"}.issubset(qrels_df.columns):
        qrels_df = pd.read_csv(qrels_path, sep="\t", names=["query-id", "corpus-id", "score"], header=None)
    qrels: Dict[str, Set[str]] = {}
    for row in qrels_df.itertuples(index=False):
        query_id = str(getattr(row, "_0", None) or getattr(row, "query_id", None) or row[0])
        corpus_id = str(getattr(row, "_1", None) or getattr(row, "corpus_id", None) or row[1])
        score = float(getattr(row, "_2", None) or getattr(row, "score", None) or row[2])
        if score > 0:
            qrels.setdefault(query_id, set()).add(corpus_id)
    return documents, queries, qrels


def ensure_beir_dataset(dataset_name: str, data_root: str) -> str:
    dataset_name = safe_slug(dataset_name).replace("_", "-")
    beir_root = os.path.join(data_root, "beir")
    candidates = [
        os.path.join(beir_root, dataset_name),
        os.path.join(data_root, dataset_name),
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "corpus.jsonl")):
            return path

    zip_path = os.path.join(beir_root, f"{dataset_name}.zip")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    if not os.path.exists(zip_path):
        _download_zip(url, zip_path)
    _extract_zip(zip_path, beir_root)
    extracted = os.path.join(beir_root, dataset_name)
    if not os.path.exists(os.path.join(extracted, "corpus.jsonl")):
        raise FileNotFoundError(f"Downloaded BEIR dataset but did not find corpus.jsonl in {extracted}")
    return extracted


def normalize_aol_ia_top_user_metric(value: str) -> str:
    metric = str(value).lower().replace("-", "_")
    if metric in {"query", "queries", "query_count"}:
        return "query_count"
    if metric in {"doc", "docs", "doc_count", "document_count"}:
        return "doc_count"
    raise ValueError(f"Unsupported AOL-IA top-user metric: {value}")


def _aol_ia_prefix(top_users: int, top_user_metric: str) -> str:
    metric = normalize_aol_ia_top_user_metric(top_user_metric)
    prefix = f"aol_ia_top{int(top_users)}"
    if metric != "query_count":
        prefix = f"{prefix}_by_{metric}"
    return prefix


def _aol_ia_default_doc_embeddings_path(config: BenchmarkConfig) -> str:
    metric = normalize_aol_ia_top_user_metric(config.aol_ia_top_user_metric)
    metric_suffix = "" if metric == "query_count" else f"_by_{metric}"
    filename = f"embedded_docs_AOLIA_top{int(config.aol_ia_top_users)}{metric_suffix}_mpnet_df.pkl"
    return os.path.join(config.aol_ia_cache_dir, filename)


def _aol_ia_interactions_path(config: BenchmarkConfig) -> str:
    prefix = _aol_ia_prefix(config.aol_ia_top_users, config.aol_ia_top_user_metric)
    return os.path.join(config.aol_ia_cache_dir, f"{prefix}_clicked_interactions.pkl")


def _aol_ia_dataset_cache_path(cache_root: str, config: BenchmarkConfig) -> str:
    metric = normalize_aol_ia_top_user_metric(config.aol_ia_top_user_metric)
    doc_path = config.aol_ia_doc_embeddings_path or _aol_ia_default_doc_embeddings_path(config)
    doc_sig = _file_signature(doc_path)
    interaction_sig = _file_signature(_aol_ia_interactions_path(config))
    filename = (
        f"aol_ia__top{int(config.aol_ia_top_users)}__{metric}"
        f"__train{int(config.aol_ia_nbr_train_recs)}__test{int(config.aol_ia_nbr_test_records)}"
        f"__seed{int(config.seed)}__docsig{doc_sig}__intsig{interaction_sig}.pkl"
    )
    return os.path.join(cache_root, "datasets", filename)


def _load_aol_ia_interactions(config: BenchmarkConfig) -> pd.DataFrame:
    path = _aol_ia_interactions_path(config)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing AOL-IA clicked-interaction cache: {path}. "
            "Run prepare_aol_ia_peer_graph_inputs.py first."
        )
    interactions = pd.read_pickle(path).copy()
    required = {"AnonID", "DocIndex"}
    missing = sorted(required.difference(interactions.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if "QueryIndex" not in interactions.columns:
        interactions["QueryIndex"] = interactions.groupby("AnonID").cumcount().astype(str)
    return interactions


def _load_aol_ia_doc_embeddings(config: BenchmarkConfig) -> pd.DataFrame:
    path = config.aol_ia_doc_embeddings_path or _aol_ia_default_doc_embeddings_path(config)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing AOL-IA document embedding cache: {path}. "
            "Run prepare_aol_ia_peer_graph_inputs.py first."
        )
    docs = pd.read_pickle(path).copy()
    if "DocIndex" not in docs.columns:
        if "doc_id" in docs.columns:
            docs = docs.rename(columns={"doc_id": "DocIndex"})
        elif "docno" in docs.columns:
            docs = docs.rename(columns={"docno": "DocIndex"})
    required = {"DocIndex", "embedding"}
    missing = sorted(required.difference(docs.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    docs["DocIndex"] = docs["DocIndex"].astype(str)
    return docs[["DocIndex", "embedding"]].drop_duplicates("DocIndex").reset_index(drop=True)


def _limit_aol_train_records_per_user(train_data: pd.DataFrame, max_records: int, seed: int) -> pd.DataFrame:
    max_records = int(max_records)
    if max_records <= 0:
        raise ValueError("aol_ia_nbr_train_recs must be positive.")
    selected = []
    for user_id, group in train_data.groupby("AnonID", sort=False):
        if len(group) > max_records:
            random_state = stable_int(f"aol-ia-train:{seed}:{user_id}")
            group = group.sample(n=max_records, random_state=random_state).sort_index()
        selected.append(group)
    if not selected:
        return train_data.iloc[0:0].copy().reset_index(drop=True)
    return pd.concat(selected).sort_index().reset_index(drop=True)


def load_or_prepare_aol_ia_dataset(
    cache_root: str,
    config: BenchmarkConfig,
    force: bool = False,
) -> RetrievalDataset:
    cache_path = _aol_ia_dataset_cache_path(cache_root, config)
    if os.path.exists(cache_path) and not force:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from Construct_avg_embedding import split_train_test_by_user

    random_state = np.random.get_state()
    import random as _random

    python_random_state = _random.getstate()
    _random.seed(config.seed)
    np.random.seed(config.seed)
    try:
        interactions = _load_aol_ia_interactions(config)
        docs = _load_aol_ia_doc_embeddings(config)
        interactions["DocIndex"] = interactions["DocIndex"].astype(str)
        embedded_doc_ids = set(docs["DocIndex"].astype(str))
        interactions = interactions[interactions["DocIndex"].isin(embedded_doc_ids)].copy()
        if interactions.empty:
            raise RuntimeError("No AOL-IA interactions remain after filtering to embedded documents.")

        sort_columns = ["AnonID", "DocIndex"]
        if "QueryTime" in interactions.columns:
            sort_columns = ["QueryTime"] + sort_columns
        interactions = interactions.sort_values(sort_columns, ascending=True).groupby(
            ["AnonID", "DocIndex"],
            as_index=False,
        ).first()[["AnonID", "DocIndex", "QueryIndex"]]

        train_data, test_data = split_train_test_by_user(
            interactions,
            nbr_train_recs=config.aol_ia_nbr_train_recs,
            nbr_test_records=config.aol_ia_nbr_test_records,
            sample_rdn_test=True,
            filter_out_users_with_few_docs=True,
        )
        train_data = _limit_aol_train_records_per_user(
            train_data,
            max_records=config.aol_ia_nbr_train_recs,
            seed=config.seed,
        )
    finally:
        _random.setstate(python_random_state)
        np.random.set_state(random_state)

    doc_ids = sorted(set(train_data["DocIndex"].astype(str)).union(test_data["DocIndex"].astype(str)))
    embedding_by_doc = {
        str(row.DocIndex): np.asarray(row.embedding, dtype=np.float32)
        for row in docs.itertuples(index=False)
        if str(row.DocIndex) in set(doc_ids)
    }
    missing_embeddings = sorted(set(doc_ids).difference(embedding_by_doc))
    if missing_embeddings:
        raise RuntimeError(f"AOL-IA split referenced documents without embeddings: {missing_embeddings[:5]}")

    documents = pd.DataFrame(
        {"doc_id": doc_id, "title": "", "text": f"AOL-IA clicked document {doc_id}"}
        for doc_id in doc_ids
    )
    doc_embeddings = normalize_rows(np.vstack([embedding_by_doc[doc_id] for doc_id in doc_ids]))
    doc_id_to_pos = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    users = sorted(train_data["AnonID"].astype(str).unique().tolist())
    train_by_user = {
        str(user): group["DocIndex"].astype(str).tolist()
        for user, group in train_data.assign(AnonID=train_data["AnonID"].astype(str)).groupby("AnonID", sort=False)
    }
    test_by_user = {
        str(user): set(group["DocIndex"].astype(str).tolist())
        for user, group in test_data.assign(AnonID=test_data["AnonID"].astype(str)).groupby("AnonID", sort=False)
    }

    query_rows = []
    query_embeddings = []
    qrels: Dict[str, Set[str]] = {}
    peer_ids: List[int] = []
    peer_doc_indices: Dict[int, np.ndarray] = {}
    peer_embeddings = []
    doc_peer_ids = np.full(len(doc_ids), -1, dtype=np.int32)

    for peer_id, user in enumerate(users):
        train_doc_ids = [doc_id for doc_id in train_by_user.get(user, []) if doc_id in doc_id_to_pos]
        test_doc_ids = {doc_id for doc_id in test_by_user.get(user, set()) if doc_id in doc_id_to_pos}
        if not train_doc_ids or not test_doc_ids:
            continue
        train_positions = np.asarray(sorted({doc_id_to_pos[doc_id] for doc_id in train_doc_ids}), dtype=np.int64)
        profile_embedding = normalize_vector(np.mean(doc_embeddings[train_positions], axis=0))
        query_id = str(user)
        query_rows.append(
            {
                "query_id": query_id,
                "text": f"AOL-IA user profile {user}",
                "user_id": user,
                "source_peer_id": int(peer_id),
            }
        )
        query_embeddings.append(profile_embedding)
        qrels[query_id] = test_doc_ids
        peer_ids.append(int(peer_id))
        peer_doc_indices[int(peer_id)] = train_positions
        peer_embeddings.append(profile_embedding)
        for doc_pos in train_positions:
            if doc_peer_ids[int(doc_pos)] < 0:
                doc_peer_ids[int(doc_pos)] = int(peer_id)

    if not query_rows:
        raise RuntimeError("AOL-IA retrieval conversion produced no eligible profile queries.")

    dataset = RetrievalDataset(
        name="aol-ia",
        documents=documents,
        queries=pd.DataFrame(query_rows),
        qrels=qrels,
        doc_embeddings=doc_embeddings,
        query_embeddings=np.vstack(query_embeddings),
        peer_assignments=PeerAssignments(
            doc_peer_ids=doc_peer_ids,
            peer_ids=peer_ids,
            peer_embeddings=normalize_rows(np.vstack(peer_embeddings)),
            peer_doc_indices=peer_doc_indices,
        ),
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset


MIND_NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
MIND_BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]


def _read_mind_news(path: str) -> pd.DataFrame:
    news = pd.read_csv(
        path,
        sep="\t",
        names=MIND_NEWS_COLUMNS,
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )
    news["doc_id"] = news["news_id"].astype(str)
    news["full_text"] = (news["title"].fillna("") + ". " + news["abstract"].fillna("")).str.strip()
    return news[["doc_id", "title", "abstract", "category", "subcategory", "full_text"]].drop_duplicates("doc_id")


def _read_mind_behaviors(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        names=MIND_BEHAVIOR_COLUMNS,
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )


def parse_mind_small_dirs(
    train_dir: str,
    valid_dir: str,
    recent_history: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    train_news = _read_mind_news(os.path.join(train_dir, "news.tsv"))
    valid_news = _read_mind_news(os.path.join(valid_dir, "news.tsv"))
    documents = pd.concat([train_news, valid_news], ignore_index=True).drop_duplicates("doc_id").reset_index(drop=True)
    text_by_doc = documents.set_index("doc_id")["full_text"].to_dict()

    behaviors = _read_mind_behaviors(os.path.join(valid_dir, "behaviors.tsv"))
    query_rows = []
    qrels: Dict[str, Set[str]] = {}
    for row in behaviors.itertuples(index=False):
        impressions = [item for item in str(row.impressions).split() if "-" in item]
        clicked = [item.rsplit("-", 1)[0] for item in impressions if item.rsplit("-", 1)[-1] == "1"]
        clicked = [doc_id for doc_id in clicked if doc_id in text_by_doc]
        if not clicked:
            continue
        history_ids = [doc_id for doc_id in str(row.history).split() if doc_id in text_by_doc]
        history_ids = history_ids[-recent_history:]
        query_text = " ".join(text_by_doc[doc_id] for doc_id in history_ids).strip()
        if not query_text:
            query_text = " ".join(text_by_doc[doc_id] for doc_id in clicked).strip()
        if not query_text:
            continue
        query_id = str(row.impression_id)
        query_rows.append({"query_id": query_id, "text": query_text, "user_id": row.user_id})
        qrels[query_id] = set(clicked)
    queries = pd.DataFrame(query_rows)
    return documents, queries, qrels


def mind_hf_rows_to_frames(rows: Iterable[Mapping[str, object]]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """Convert the compact MTEB MIND-small rows into benchmark frames.

    The public Azure MIND blobs are not always accessible from the cluster.  The
    MTEB mirror exposes query history titles plus positive/negative candidate
    titles, so we treat each unique candidate title as a retrievable document.
    """
    title_to_doc_id: Dict[str, str] = {}
    doc_rows = []
    query_rows = []
    qrels: Dict[str, Set[str]] = {}

    def add_doc(title_value: object) -> Optional[str]:
        title = str(title_value or "").strip()
        if not title:
            return None
        doc_id = title_to_doc_id.get(title)
        if doc_id is None:
            doc_id = f"MINDDOC_{len(title_to_doc_id)}"
            title_to_doc_id[title] = doc_id
            doc_rows.append({"doc_id": doc_id, "title": title, "full_text": title})
        return doc_id

    for fallback_idx, row in enumerate(rows):
        row_map = dict(row)
        query_parts = row_map.get("query", [])
        if not isinstance(query_parts, list):
            query_parts = [query_parts]
        query_text = " ".join(str(part or "").strip() for part in query_parts if str(part or "").strip()).strip()
        if not query_text:
            continue

        positives = row_map.get("positive", []) or []
        negatives = row_map.get("negative", []) or []
        if not isinstance(positives, list):
            positives = [positives]
        if not isinstance(negatives, list):
            negatives = [negatives]

        positive_doc_ids = set()
        for title in positives:
            doc_id = add_doc(title)
            if doc_id is not None:
                positive_doc_ids.add(doc_id)
        for title in negatives:
            add_doc(title)
        if not positive_doc_ids:
            continue

        query_id = str(row_map.get("id", fallback_idx))
        query_rows.append({"query_id": query_id, "text": query_text})
        qrels[query_id] = positive_doc_ids

    documents = pd.DataFrame(doc_rows, columns=["doc_id", "title", "full_text"])
    queries = pd.DataFrame(query_rows, columns=["query_id", "text"])
    return documents, queries, qrels


def parse_mind_small_hf(split: str = "test") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    from datasets import load_dataset

    hf_split = "test" if split in {"dev", "validation"} else split
    rows = load_dataset("mteb/mind_small", split=hf_split)
    return mind_hf_rows_to_frames(rows)


def ensure_mind_small(data_root: str) -> Tuple[str, str]:
    mind_root = os.path.join(data_root, "mind")
    train_dir = os.path.join(mind_root, "MINDsmall_train")
    valid_dir = os.path.join(mind_root, "MINDsmall_dev")
    if os.path.exists(os.path.join(train_dir, "news.tsv")) and os.path.exists(os.path.join(valid_dir, "behaviors.tsv")):
        return train_dir, valid_dir
    urls = {
        "MINDsmall_train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
        "MINDsmall_dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
    }
    os.makedirs(mind_root, exist_ok=True)
    for name, url in urls.items():
        zip_path = os.path.join(mind_root, f"{name}.zip")
        if not os.path.exists(zip_path):
            _download_zip(url, zip_path)
        dest = os.path.join(mind_root, name)
        if not os.path.exists(os.path.join(dest, "news.tsv")):
            _extract_zip(zip_path, dest)
    return train_dir, valid_dir


def embed_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 128,
    show_progress: bool = True,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return normalize_rows(embeddings)


def _dataset_cache_path(cache_root: str, dataset_name: str, model_name: str, split: str = "test") -> str:
    model_slug = safe_slug(model_name)
    split_slug = safe_slug(split)
    return os.path.join(cache_root, "datasets", f"{safe_slug(dataset_name)}__{model_slug}__split{split_slug}.pkl")


def _load_raw_dataset(dataset_name: str, data_root: str, split: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    normalized = str(dataset_name).lower()
    if normalized == "mind-small":
        mind_root = os.path.join(data_root, "mind")
        train_dir = os.path.join(mind_root, "MINDsmall_train")
        valid_dir = os.path.join(mind_root, "MINDsmall_dev")
        if os.path.exists(os.path.join(train_dir, "news.tsv")) and os.path.exists(os.path.join(valid_dir, "behaviors.tsv")):
            return parse_mind_small_dirs(train_dir, valid_dir)
        print(
            "MIND-small TSV files not found; loading compact mteb/mind_small fallback.",
            flush=True,
        )
        return parse_mind_small_hf(split=split)
    dataset_dir = ensure_beir_dataset(normalized, data_root)
    return parse_beir_dir(dataset_dir, split=split)


def load_or_prepare_dataset(
    dataset_name: str,
    data_root: str,
    cache_root: str,
    config: Optional[BenchmarkConfig] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    split: str = "test",
    batch_size: int = 128,
    force: bool = False,
) -> RetrievalDataset:
    normalized_name = str(dataset_name).lower().replace("_", "-")
    if normalized_name in {"aol-ia", "aolia"}:
        return load_or_prepare_aol_ia_dataset(
            cache_root=cache_root,
            config=config or BenchmarkConfig(),
            force=force,
        )

    cache_path = _dataset_cache_path(cache_root, dataset_name, model_name, split=split)
    if os.path.exists(cache_path) and not force:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    documents, queries, qrels = _load_raw_dataset(dataset_name, data_root, split=split)
    doc_texts = documents["full_text"].fillna("").astype(str).tolist()
    query_texts = queries["text"].fillna("").astype(str).tolist()
    doc_embeddings = embed_texts(doc_texts, model_name=model_name, batch_size=batch_size)
    query_embeddings = embed_texts(query_texts, model_name=model_name, batch_size=batch_size)
    dataset = RetrievalDataset(
        name=dataset_name,
        documents=documents[["doc_id", "title", "full_text"]].rename(columns={"full_text": "text"}),
        queries=queries[["query_id", "text"]],
        qrels=qrels,
        doc_embeddings=doc_embeddings,
        query_embeddings=query_embeddings,
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset


def subset_dataset(
    dataset: RetrievalDataset,
    sample_queries: Optional[int] = None,
    sample_docs: Optional[int] = None,
    seed: int = 42,
) -> RetrievalDataset:
    rng = np.random.default_rng(seed)
    query_positions = np.arange(dataset.queries.shape[0])
    if sample_queries is not None and sample_queries > 0 and sample_queries < len(query_positions):
        eligible = [
            idx for idx, query_id in enumerate(dataset.query_ids)
            if dataset.qrels.get(query_id, set())
        ]
        source = np.array(eligible if eligible else query_positions, dtype=np.int64)
        query_positions = np.sort(rng.choice(source, size=min(sample_queries, len(source)), replace=False))

    selected_queries = dataset.queries.iloc[query_positions].reset_index(drop=True)
    selected_query_embeddings = dataset.query_embeddings[query_positions]
    selected_query_ids = selected_queries["query_id"].astype(str).tolist()
    selected_qrels = {
        query_id: set(dataset.qrels.get(query_id, set()))
        for query_id in selected_query_ids
    }

    doc_positions = np.arange(dataset.documents.shape[0])
    if sample_docs is not None and sample_docs > 0 and sample_docs < len(doc_positions):
        doc_id_to_pos = dataset.doc_id_to_pos
        relevant_positions = sorted(
            {
                doc_id_to_pos[doc_id]
                for doc_ids in selected_qrels.values()
                for doc_id in doc_ids
                if doc_id in doc_id_to_pos
            }
        )
        remaining = [idx for idx in doc_positions.tolist() if idx not in set(relevant_positions)]
        extra_count = max(0, sample_docs - len(relevant_positions))
        if extra_count < len(remaining):
            extra = rng.choice(np.array(remaining, dtype=np.int64), size=extra_count, replace=False).tolist()
        else:
            extra = remaining
        doc_positions = np.array(sorted(set(relevant_positions).union(extra)), dtype=np.int64)

    selected_docs = dataset.documents.iloc[doc_positions].reset_index(drop=True)
    selected_doc_embeddings = dataset.doc_embeddings[doc_positions]
    indexed_doc_ids = set(selected_docs["doc_id"].astype(str))
    filtered_qrels = {
        query_id: set(doc_ids).intersection(indexed_doc_ids)
        for query_id, doc_ids in selected_qrels.items()
    }
    subset = RetrievalDataset(
        name=dataset.name,
        documents=selected_docs,
        queries=selected_queries,
        qrels=filtered_qrels,
        doc_embeddings=selected_doc_embeddings,
        query_embeddings=selected_query_embeddings,
    )
    if dataset.peer_assignments is not None:
        subset.peer_assignments = _subset_peer_assignments(
            dataset.peer_assignments,
            doc_positions,
            subset.doc_embeddings,
        )
    return subset


def _subset_peer_assignments(
    assignments: PeerAssignments,
    doc_positions: np.ndarray,
    doc_embeddings: np.ndarray,
) -> PeerAssignments:
    doc_positions = np.asarray(doc_positions, dtype=np.int64)
    old_to_new = {int(old_pos): int(new_pos) for new_pos, old_pos in enumerate(doc_positions.tolist())}
    doc_peer_ids = np.full(len(doc_positions), -1, dtype=np.int32)
    for new_pos, old_pos in enumerate(doc_positions.tolist()):
        if int(old_pos) < len(assignments.doc_peer_ids):
            doc_peer_ids[new_pos] = int(assignments.doc_peer_ids[int(old_pos)])

    peer_ids: List[int] = []
    peer_doc_indices: Dict[int, np.ndarray] = {}
    peer_embeddings = []
    for peer_id in assignments.peer_ids:
        remapped = [
            old_to_new[int(old_pos)]
            for old_pos in assignments.peer_doc_indices.get(int(peer_id), np.asarray([], dtype=np.int64))
            if int(old_pos) in old_to_new
        ]
        if not remapped:
            continue
        indices = np.asarray(sorted(set(remapped)), dtype=np.int64)
        peer_ids.append(int(peer_id))
        peer_doc_indices[int(peer_id)] = indices
        peer_embeddings.append(normalize_vector(np.mean(doc_embeddings[indices], axis=0)))

    if not peer_ids:
        return PeerAssignments(
            doc_peer_ids=doc_peer_ids,
            peer_ids=[],
            peer_embeddings=np.zeros((0, doc_embeddings.shape[1]), dtype=np.float32),
            peer_doc_indices={},
        )
    return PeerAssignments(
        doc_peer_ids=doc_peer_ids,
        peer_ids=peer_ids,
        peer_embeddings=normalize_rows(np.vstack(peer_embeddings)),
        peer_doc_indices=peer_doc_indices,
    )


def _peer_cache_path(
    cache_root: str,
    dataset: RetrievalDataset,
    target_docs_per_peer: int,
    max_peers: int,
    seed: int,
) -> str:
    doc_hash = hashlib.md5("\n".join(dataset.doc_ids).encode("utf-8")).hexdigest()[:12]
    embedding_hash = _embedding_cache_signature(dataset)
    filename = (
        f"{safe_slug(dataset.name)}__docs{len(dataset.doc_ids)}__{doc_hash}"
        f"__emb{embedding_hash}__target{target_docs_per_peer}__max{max_peers}__seed{seed}.pkl"
    )
    return os.path.join(cache_root, "peer_assignments", filename)


def _embedding_cache_signature(dataset: RetrievalDataset) -> str:
    embeddings = np.asarray(dataset.doc_embeddings, dtype=np.float32)
    digest = hashlib.md5()
    digest.update(str(embeddings.shape).encode("utf-8"))
    if embeddings.size:
        sample_count = min(512, embeddings.shape[0])
        digest.update(np.ascontiguousarray(embeddings[:sample_count]).tobytes())
        if embeddings.shape[0] > sample_count:
            digest.update(np.ascontiguousarray(embeddings[-sample_count:]).tobytes())
    return digest.hexdigest()[:12]


def build_peer_assignments(
    dataset: RetrievalDataset,
    target_docs_per_peer: int = 50,
    max_peers: int = 5000,
    seed: int = 42,
    cache_root: Optional[str] = None,
    force: bool = False,
) -> PeerAssignments:
    if cache_root:
        path = _peer_cache_path(cache_root, dataset, target_docs_per_peer, max_peers, seed)
        if os.path.exists(path) and not force:
            with open(path, "rb") as f:
                return pickle.load(f)

    n_docs = dataset.doc_embeddings.shape[0]
    if n_docs == 0:
        raise ValueError("Cannot build peer assignments for an empty dataset.")
    n_peers = min(max(1, int(math.ceil(n_docs / max(1, target_docs_per_peer)))), int(max_peers), n_docs)
    if n_peers == n_docs:
        labels = np.arange(n_docs, dtype=np.int32)
    else:
        batch_size = max(1024, min(n_docs, n_peers * 10))
        model = MiniBatchKMeans(
            n_clusters=n_peers,
            random_state=seed,
            batch_size=batch_size,
            n_init=3,
            reassignment_ratio=0.01,
        )
        labels = model.fit_predict(dataset.doc_embeddings).astype(np.int32)

    peer_doc_indices: Dict[int, np.ndarray] = {}
    peer_embeddings = []
    peer_ids = sorted(int(value) for value in np.unique(labels))
    for peer_id in peer_ids:
        indices = np.where(labels == peer_id)[0].astype(np.int64)
        peer_doc_indices[peer_id] = indices
        peer_embeddings.append(normalize_vector(np.mean(dataset.doc_embeddings[indices], axis=0)))
    assignments = PeerAssignments(
        doc_peer_ids=labels,
        peer_ids=peer_ids,
        peer_embeddings=normalize_rows(np.vstack(peer_embeddings)),
        peer_doc_indices=peer_doc_indices,
    )
    if cache_root:
        path = _peer_cache_path(cache_root, dataset, target_docs_per_peer, max_peers, seed)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(assignments, f, protocol=pickle.HIGHEST_PROTOCOL)
    return assignments


def _top_docs_from_candidates(
    dataset: RetrievalDataset,
    query_embedding: np.ndarray,
    candidate_indices: Iterable[int],
    top_k: int,
) -> Tuple[List[int], List[float]]:
    seen = set()
    candidates = []
    for idx in candidate_indices:
        idx = int(idx)
        if idx < 0 or idx >= dataset.doc_embeddings.shape[0] or idx in seen:
            continue
        seen.add(idx)
        candidates.append(idx)
    if not candidates:
        return [], []
    candidate_array = np.asarray(candidates, dtype=np.int64)
    sims = dataset.doc_embeddings[candidate_array] @ normalize_vector(query_embedding)
    order = np.argsort(-sims, kind="mergesort")[:top_k]
    top_indices = candidate_array[order].astype(int).tolist()
    top_scores = sims[order].astype(float).tolist()
    return top_indices, top_scores


def _count_touched_peers(dataset: RetrievalDataset, doc_indices: Iterable[int]) -> int:
    if dataset.peer_assignments is None:
        return 0
    valid_doc_indices = [
        int(idx)
        for idx in doc_indices
        if 0 <= int(idx) < len(dataset.peer_assignments.doc_peer_ids)
    ]
    if not valid_doc_indices:
        return 0
    peer_ids = dataset.peer_assignments.doc_peer_ids[np.asarray(valid_doc_indices, dtype=np.int64)]
    return int(len({int(peer_id) for peer_id in peer_ids.tolist() if int(peer_id) >= 0}))


def _pad_matrix_dim_to_multiple_of_four(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D embedding matrix.")
    pad = (-matrix.shape[1]) % 4
    if pad:
        matrix = np.pad(matrix, ((0, 0), (0, pad)), mode="constant")
    return np.ascontiguousarray(matrix, dtype=np.float32)


class TreeNode:
    _counter = 0

    def __init__(self, indices: Sequence[int], depth: int, parent: Optional["TreeNode"] = None):
        self.node_id = TreeNode._counter
        TreeNode._counter += 1
        self.indices = list(indices)
        self.depth = depth
        self.parent = parent
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None
        self.left_centroid: Optional[np.ndarray] = None
        self.right_centroid: Optional[np.ndarray] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def children(self) -> List["TreeNode"]:
        return [child for child in (self.left, self.right) if child is not None]


class SemanticTree:
    def __init__(self, embeddings: np.ndarray, leaf_size: int = 50, delta: float = 5e-4, seed: int = 42):
        self.embeddings = normalize_rows(embeddings)
        self.leaf_size = max(1, int(leaf_size))
        self.delta = float(delta)
        self.seed = int(seed)
        TreeNode._counter = 0
        self.root = self._build(list(range(self.embeddings.shape[0])), depth=0, parent=None)

    def _build(self, indices: List[int], depth: int, parent: Optional[TreeNode]) -> TreeNode:
        node = TreeNode(indices, depth=depth, parent=parent)
        if len(indices) <= self.leaf_size or len(indices) < 2:
            return node
        values = self.embeddings[np.asarray(indices, dtype=np.int64)]
        try:
            model = KMeans(n_clusters=2, random_state=self.seed + depth, n_init=5).fit(values)
        except ValueError:
            return node
        labels = model.labels_
        left_indices = [idx for idx, label in zip(indices, labels) if int(label) == 0]
        right_indices = [idx for idx, label in zip(indices, labels) if int(label) == 1]
        if not left_indices or not right_indices:
            return node
        if min(len(left_indices), len(right_indices)) < max(1, int(0.05 * len(indices))):
            return node
        node.left_centroid = normalize_vector(model.cluster_centers_[0])
        node.right_centroid = normalize_vector(model.cluster_centers_[1])
        node.left = self._build(left_indices, depth=depth + 1, parent=node)
        node.right = self._build(right_indices, depth=depth + 1, parent=node)
        return node

    def _route_leaves(self, query_embedding: np.ndarray) -> List[TreeNode]:
        query_embedding = normalize_vector(query_embedding)
        leaves = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                leaves.append(node)
                continue
            left_sim = float(np.dot(query_embedding, node.left_centroid))
            right_sim = float(np.dot(query_embedding, node.right_centroid))
            if abs(left_sim - right_sim) < self.delta:
                stack.extend(node.children())
            elif left_sim >= right_sim:
                stack.append(node.left)
            else:
                stack.append(node.right)
        return leaves

    def query_indices(self, query_embedding: np.ndarray, limit: int) -> List[int]:
        limit = max(1, int(limit))
        output: List[int] = []
        seen_indices = set()
        seen_nodes = set()
        queue = list(self._route_leaves(query_embedding))
        while queue and len(output) < limit:
            node = queue.pop(0)
            if node is None or node.node_id in seen_nodes:
                continue
            seen_nodes.add(node.node_id)
            if node.is_leaf:
                for idx in node.indices:
                    if idx in seen_indices:
                        continue
                    seen_indices.add(idx)
                    output.append(idx)
                    if len(output) >= limit:
                        break
            else:
                queue.extend(node.children())
            if node.parent is not None:
                queue.append(node.parent)
        if len(output) < limit:
            query = normalize_vector(query_embedding)
            sims = self.embeddings @ query
            for idx in np.argsort(-sims, kind="mergesort"):
                idx = int(idx)
                if idx not in seen_indices:
                    output.append(idx)
                    if len(output) >= limit:
                        break
        return output[:limit]


class Retriever:
    def retrieve(self, query_id: str, query_embedding: np.ndarray, top_k: int) -> RetrievalResult:
        raise NotImplementedError


class SemanticaRandomRetriever(Retriever):
    def __init__(self, dataset: RetrievalDataset, config: BenchmarkConfig):
        self.dataset = dataset
        self.config = config
        if dataset.peer_assignments is None:
            raise ValueError("Semantica retrievers require peer assignments.")
        self.peers = dataset.peer_assignments

    def retrieve(self, query_id: str, query_embedding: np.ndarray, top_k: int) -> RetrievalResult:
        rng = np.random.default_rng(self.config.seed + stable_int(query_id))
        count = min(self.config.semantica_contact_peers, len(self.peers.peer_ids))
        peer_ids = rng.choice(np.asarray(self.peers.peer_ids, dtype=np.int64), size=count, replace=False).astype(int)
        candidate_indices = np.concatenate([self.peers.peer_doc_indices[int(peer)] for peer in peer_ids])
        doc_indices, scores = _top_docs_from_candidates(self.dataset, query_embedding, candidate_indices, top_k)
        return RetrievalResult(doc_indices, scores, int(len(peer_ids)), int(len(set(candidate_indices.tolist()))))


class SemanticaTreeRetriever(Retriever):
    def __init__(self, dataset: RetrievalDataset, config: BenchmarkConfig):
        self.dataset = dataset
        self.config = config
        if dataset.peer_assignments is None:
            raise ValueError("Semantica retrievers require peer assignments.")
        self.peers = dataset.peer_assignments
        self.tree = SemanticTree(
            self.peers.peer_embeddings,
            leaf_size=config.semantica_leaf_size,
            delta=config.semantica_delta,
            seed=config.seed,
        )

    def retrieve(self, query_id: str, query_embedding: np.ndarray, top_k: int) -> RetrievalResult:
        peer_positions = self.tree.query_indices(query_embedding, self.config.semantica_contact_peers)
        peer_ids = [self.peers.peer_ids[pos] for pos in peer_positions]
        candidate_indices = np.concatenate([self.peers.peer_doc_indices[int(peer)] for peer in peer_ids])
        doc_indices, scores = _top_docs_from_candidates(self.dataset, query_embedding, candidate_indices, top_k)
        return RetrievalResult(doc_indices, scores, int(len(peer_ids)), int(len(set(candidate_indices.tolist()))))


def _select_elbow_k(embeddings: np.ndarray, max_k: int, seed: int) -> int:
    max_k = max(1, min(int(max_k), embeddings.shape[0]))
    if max_k < 3:
        return max_k
    inertias = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(embeddings)
        inertias.append(float(model.inertia_))
    previous = inertias[0]
    for index in range(1, len(inertias)):
        if previous <= 1e-12:
            return index
        improvement = (previous - inertias[index]) / previous
        if improvement < 0.10:
            return max(1, index)
        previous = inertias[index]
    return max_k


class SemanticaCentroidRetriever(Retriever):
    def __init__(self, dataset: RetrievalDataset, config: BenchmarkConfig, mode: str):
        self.dataset = dataset
        self.config = config
        self.mode = mode
        if dataset.peer_assignments is None:
            raise ValueError("Semantica retrievers require peer assignments.")
        self.peers = dataset.peer_assignments
        self.centroid_peer_ids: List[int] = []
        self.centroid_embeddings: List[np.ndarray] = []
        self.peer_centroids: Dict[int, np.ndarray] = {}
        self._fit_centroids()
        self.centroid_matrix = normalize_rows(np.vstack(self.centroid_embeddings))
        self.tree = SemanticTree(
            self.centroid_matrix,
            leaf_size=config.semantica_leaf_size,
            delta=config.semantica_delta,
            seed=config.seed,
        )

    def _fit_centroids(self) -> None:
        for peer_id in self.peers.peer_ids:
            doc_indices = self.peers.peer_doc_indices[int(peer_id)]
            values = self.dataset.doc_embeddings[doc_indices]
            if self.mode == "fixed":
                k = min(max(1, self.config.semantica_fixed_k), values.shape[0])
            else:
                k = _select_elbow_k(values, self.config.semantica_elbow_max_k, self.config.seed)
            if k <= 1:
                centers = np.array([np.mean(values, axis=0)])
            else:
                model = KMeans(n_clusters=k, random_state=self.config.seed, n_init=3).fit(values)
                centers = model.cluster_centers_
            centers = normalize_rows(centers)
            self.peer_centroids[int(peer_id)] = centers
            for center in centers:
                self.centroid_peer_ids.append(int(peer_id))
                self.centroid_embeddings.append(center)

    def retrieve(self, query_id: str, query_embedding: np.ndarray, top_k: int) -> RetrievalResult:
        query = normalize_vector(query_embedding)
        centroid_limit = min(len(self.centroid_peer_ids), max(self.config.semantica_contact_peers * 4, top_k))
        centroid_positions = self.tree.query_indices(query, centroid_limit)
        peer_candidates = []
        seen = set()
        for pos in centroid_positions:
            peer_id = self.centroid_peer_ids[pos]
            if peer_id not in seen:
                seen.add(peer_id)
                peer_candidates.append(peer_id)
            if len(peer_candidates) >= self.config.semantica_contact_peers:
                break
        if len(peer_candidates) < self.config.semantica_contact_peers:
            peer_scores = []
            for peer_id, centers in self.peer_centroids.items():
                if peer_id in seen:
                    continue
                peer_scores.append((peer_id, float(np.max(centers @ query))))
            peer_scores.sort(key=lambda item: item[1], reverse=True)
            for peer_id, _score in peer_scores:
                peer_candidates.append(peer_id)
                if len(peer_candidates) >= self.config.semantica_contact_peers:
                    break
        candidate_indices = np.concatenate([self.peers.peer_doc_indices[int(peer)] for peer in peer_candidates])
        doc_indices, scores = _top_docs_from_candidates(self.dataset, query, candidate_indices, top_k)
        return RetrievalResult(
            doc_indices,
            scores,
            int(len(peer_candidates)),
            int(len(set(candidate_indices.tolist()))),
        )


class PStableDHTRetriever(Retriever):
    def __init__(self, dataset: RetrievalDataset, config: BenchmarkConfig):
        self.dataset = dataset
        self.config = config
        self.dim = int(dataset.doc_embeddings.shape[1])
        rng = np.random.default_rng(config.seed)
        self.projections = rng.normal(size=(config.pstable_l, config.pstable_k, self.dim)).astype(np.float32)
        self.offsets = rng.uniform(0.0, config.pstable_w, size=(config.pstable_l, config.pstable_k)).astype(np.float32)
        self.tables: List[Dict[Tuple[int, ...], List[int]]] = [dict() for _ in range(config.pstable_l)]
        self._build_tables()

    def _hash_table(self, table_idx: int, vectors: np.ndarray) -> np.ndarray:
        projected = vectors @ self.projections[table_idx].T
        return np.floor((projected + self.offsets[table_idx]) / self.config.pstable_w).astype(np.int64)

    def _build_tables(self) -> None:
        for table_idx in range(self.config.pstable_l):
            hashes = self._hash_table(table_idx, self.dataset.doc_embeddings)
            table = self.tables[table_idx]
            for doc_idx, row in enumerate(hashes):
                table.setdefault(tuple(int(value) for value in row), []).append(int(doc_idx))

    def bucket_size_stats(self) -> Dict[str, object]:
        sizes = np.asarray(
            [len(bucket) for table in self.tables for bucket in table.values()],
            dtype=np.int64,
        )
        if sizes.size == 0:
            return {
                "nonempty_bucket_count": 0,
                "total_bucket_slots": int(self.config.pstable_l),
                "total_bucket_entries": 0,
                "mean_nonempty_bucket_docs": 0.0,
                "median_nonempty_bucket_docs": 0.0,
                "p95_nonempty_bucket_docs": 0.0,
                "max_bucket_docs": 0,
            }
        return {
            "nonempty_bucket_count": int(sizes.size),
            "total_bucket_slots": int(self.config.pstable_l),
            "total_bucket_entries": int(sizes.sum()),
            "mean_nonempty_bucket_docs": float(np.mean(sizes)),
            "median_nonempty_bucket_docs": float(np.median(sizes)),
            "p95_nonempty_bucket_docs": float(np.percentile(sizes, 95)),
            "max_bucket_docs": int(np.max(sizes)),
        }

    def retrieve(self, query_id: str, query_embedding: np.ndarray, top_k: int) -> RetrievalResult:
        query = normalize_vector(query_embedding).reshape(1, -1)
        candidates = []
        seen = set()
        cap = int(self.config.pstable_candidate_cap)
        unlimited = cap <= 0
        for table_idx, table in enumerate(self.tables):
            key = tuple(int(value) for value in self._hash_table(table_idx, query)[0])
            for doc_idx in table.get(key, []):
                if doc_idx in seen:
                    continue
                seen.add(doc_idx)
                candidates.append(doc_idx)
                if not unlimited and len(candidates) >= cap:
                    break
            if not unlimited and len(candidates) >= cap:
                break
        doc_indices, scores = _top_docs_from_candidates(self.dataset, query_embedding, candidates, top_k)
        touched = _count_touched_peers(self.dataset, candidates)
        return RetrievalResult(doc_indices, scores, touched, int(len(candidates)))


class HNSWRetriever(Retriever):
    def __init__(self, dataset: RetrievalDataset, config: BenchmarkConfig):
        try:
            import hnswlib
        except ImportError as exc:
            raise ImportError("Install hnswlib to run the HNSW retrieval method.") from exc
        self.dataset = dataset
        self.config = config
        dim = int(dataset.doc_embeddings.shape[1])
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(
            max_elements=dataset.doc_embeddings.shape[0],
            ef_construction=config.hnsw_ef_construction,
            M=config.hnsw_m,
            random_seed=config.seed,
        )
        self.index.add_items(dataset.doc_embeddings, np.arange(dataset.doc_embeddings.shape[0]))
        self.index.set_ef(config.hnsw_ef_search)

    def retrieve(self, query_id: str, query_embedding: np.ndarray, top_k: int) -> RetrievalResult:
        count = min(top_k, self.dataset.doc_embeddings.shape[0])
        labels, distances = self.index.knn_query(normalize_vector(query_embedding).reshape(1, -1), k=count)
        doc_indices = [int(value) for value in labels[0].tolist()]
        scores = [float(1.0 - value) for value in distances[0].tolist()]
        touched = _count_touched_peers(self.dataset, doc_indices)
        return RetrievalResult(doc_indices, scores, touched, int(count))


def _effective_kgraph_index_params(config: BenchmarkConfig, n_items: int) -> Dict[str, object]:
    if int(n_items) < 2:
        raise ValueError("KGraph NN-Descent index construction requires at least two peers.")
    max_neighbors = int(n_items) - 1
    index_k = min(max(1, int(config.kgraph_index_k)), max_neighbors)
    index_l = min(max(index_k, int(config.kgraph_index_l)), max_neighbors)
    return {
        "iterations": int(config.kgraph_index_iterations),
        "L": int(index_l),
        "K": int(index_k),
        "S": 10,
        "controls": 100,
        "delta": float(config.kgraph_index_delta),
        "recall": float(config.kgraph_index_recall),
        "prune": 0,
        "reverse": int(config.kgraph_index_reverse),
        "metric": "angular",
    }


def _kgraph_index_cache_path(dataset: RetrievalDataset, config: BenchmarkConfig) -> Optional[str]:
    if not config.kgraph_cache_root:
        return None
    if dataset.peer_assignments is None:
        return None
    if dataset.peer_assignments.peer_embeddings.shape[0] < 2:
        return None
    peer_ids = [str(peer_id) for peer_id in dataset.peer_assignments.peer_ids]
    peer_hash = hashlib.md5("\n".join(peer_ids).encode("utf-8")).hexdigest()[:12]
    embeddings = np.asarray(dataset.peer_assignments.peer_embeddings, dtype=np.float32)
    digest = hashlib.md5()
    digest.update(str(embeddings.shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(embeddings).tobytes())
    embedding_hash = digest.hexdigest()[:12]
    params = _effective_kgraph_index_params(config, dataset.peer_assignments.peer_embeddings.shape[0])
    params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    metadata = {
        "cache_format": 1,
        "module": kgraph_module_metadata(),
    }
    module_hash = hashlib.md5(json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    filename = (
        f"{safe_slug(dataset.name)}__peers{len(peer_ids)}__{peer_hash}"
        f"__peeremb{embedding_hash}__kgraph{params_hash}__kgmod{module_hash}.index"
    )
    return os.path.join(config.kgraph_cache_root, "kgraph_indices", filename)


def _load_kgraph_module():
    try:
        import kgraph
    except ImportError as exc:
        raise ImportError(
            "Install the vendored KGraph Python extension to run kgraph_nndescent. "
            "From paper-implementation/Semantica, run: "
            "PYTHON_BIN=/path/to/python scripts/install_kgraph_python.sh"
        ) from exc
    return kgraph


def kgraph_module_metadata() -> Dict[str, object]:
    kgraph = _load_kgraph_module()
    return {
        "git_commit": getattr(kgraph, "git_commit", "unknown"),
        "arch": getattr(kgraph, "arch", "unknown"),
    }


def preflight_retrieval_methods(method_names: Sequence[str]) -> None:
    if "kgraph_nndescent" not in set(method_names):
        return
    metadata = kgraph_module_metadata()
    print(f"KGraph preflight OK: {metadata}", flush=True)


class KGraphNNDescentRetriever(Retriever):
    def __init__(self, dataset: RetrievalDataset, config: BenchmarkConfig):
        self.dataset = dataset
        self.config = config
        self.kgraph = _load_kgraph_module()
        if dataset.peer_assignments is None:
            raise ValueError("kgraph_nndescent requires peer assignments.")
        self.peers = dataset.peer_assignments
        if not self.peers.peer_ids or self.peers.peer_embeddings.shape[0] == 0:
            raise ValueError("kgraph_nndescent requires at least one peer embedding.")
        if "source_peer_id" in dataset.queries.columns:
            self.query_source_peer_id = {
                str(row.query_id): int(row.source_peer_id)
                for row in dataset.queries[["query_id", "source_peer_id"]].dropna().itertuples(index=False)
            }
        else:
            self.query_source_peer_id = {}
        self.peer_matrix = _pad_matrix_dim_to_multiple_of_four(self.peers.peer_embeddings)
        self.index = None
        if len(self.peers.peer_ids) > 1:
            self.index = self.kgraph.KGraph(self.peer_matrix, "angular")
            self._load_or_build_index()

    def _load_or_build_index(self) -> None:
        cache_path = _kgraph_index_cache_path(self.dataset, self.config)
        if cache_path and os.path.exists(cache_path):
            try:
                self.index.load(cache_path)
                print(f"Loaded KGraph index cache: {cache_path}", flush=True)
                return
            except Exception as exc:
                print(f"Could not load KGraph index cache {cache_path}: {exc}; rebuilding.", flush=True)

        params = _effective_kgraph_index_params(self.config, self.peers.peer_embeddings.shape[0])
        self.index.build(
            iterations=params["iterations"],
            L=params["L"],
            K=params["K"],
            S=params["S"],
            controls=params["controls"],
            delta=params["delta"],
            recall=params["recall"],
            prune=params["prune"],
            reverse=params["reverse"],
        )
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_path = f"{cache_path}.tmp.{os.getpid()}"
            self.index.save(tmp_path)
            os.replace(tmp_path, cache_path)
            print(f"Saved KGraph index cache: {cache_path}", flush=True)

    def search_peer_ids(
        self,
        query_embedding: np.ndarray,
        count: int,
        exclude_peer_id: Optional[int] = None,
    ) -> List[int]:
        peer_count = min(max(0, int(count)), len(self.peers.peer_ids))
        if peer_count <= 0:
            return []
        excluded = {int(exclude_peer_id)} if exclude_peer_id is not None else set()
        search_count = min(
            len(self.peers.peer_ids),
            peer_count + len(excluded),
        )
        query = normalize_vector(query_embedding).reshape(1, -1)
        query_matrix = _pad_matrix_dim_to_multiple_of_four(query)
        if self.index is None:
            labels = np.arange(len(self.peers.peer_ids), dtype=np.int64).reshape(1, -1)
        else:
            labels = np.asarray(
                self.index.search(
                    query_matrix,
                    K=search_count,
                    P=int(self.config.kgraph_search_p),
                    M=int(self.config.kgraph_search_m),
                    S=int(self.config.kgraph_search_s),
                    T=int(self.config.kgraph_search_t),
                    threads=int(self.config.kgraph_threads),
                )
            )
        peer_ids: List[int] = []
        seen = set()
        for value in labels.reshape(-1).tolist():
            idx = int(value)
            if idx in seen or idx < 0 or idx >= len(self.peers.peer_ids):
                continue
            seen.add(idx)
            peer_id = int(self.peers.peer_ids[idx])
            if peer_id in excluded:
                continue
            peer_ids.append(peer_id)
            if len(peer_ids) >= peer_count:
                break
        return peer_ids

    def retrieve(self, query_id: str, query_embedding: np.ndarray, top_k: int) -> RetrievalResult:
        peer_count = min(
            max(1, int(self.config.semantica_contact_peers)),
            len(self.peers.peer_ids),
        )
        source_peer_id = self.query_source_peer_id.get(str(query_id))
        peer_ids = self.search_peer_ids(query_embedding, peer_count, exclude_peer_id=source_peer_id)
        if not peer_ids:
            return RetrievalResult([], [], 0, 0)
        candidate_arrays = [
            self.peers.peer_doc_indices[int(peer_id)]
            for peer_id in peer_ids
            if int(peer_id) in self.peers.peer_doc_indices
        ]
        if not candidate_arrays:
            return RetrievalResult([], [], int(len(peer_ids)), 0)
        candidate_indices = np.concatenate(candidate_arrays)
        doc_indices, scores = _top_docs_from_candidates(
            self.dataset,
            query_embedding,
            candidate_indices,
            top_k,
        )
        return RetrievalResult(
            doc_indices,
            scores,
            int(len(peer_ids)),
            int(len(set(candidate_indices.astype(int).tolist()))),
        )


def build_retriever(method_name: str, dataset: RetrievalDataset, config: BenchmarkConfig) -> Retriever:
    if method_name == "semantica_random":
        return SemanticaRandomRetriever(dataset, config)
    if method_name == "semantica_tree":
        return SemanticaTreeRetriever(dataset, config)
    if method_name == "semantica_fixed_k":
        return SemanticaCentroidRetriever(dataset, config, mode="fixed")
    if method_name == "semantica_elbow":
        return SemanticaCentroidRetriever(dataset, config, mode="elbow")
    if method_name == "pstable_dht":
        return PStableDHTRetriever(dataset, config)
    if method_name == "hnsw":
        return HNSWRetriever(dataset, config)
    if method_name == "kgraph_nndescent":
        return KGraphNNDescentRetriever(dataset, config)
    raise ValueError(f"Unsupported retrieval method: {method_name}")


def _metric_values(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Set[str], top_k: int) -> Dict[str, float]:
    seen = set()
    ranked = []
    for doc_id in retrieved_doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
        if len(ranked) >= top_k:
            break
    rel_count = sum(1 for doc_id in ranked if doc_id in relevant_doc_ids)
    first_rel = next((idx + 1 for idx, doc_id in enumerate(ranked) if doc_id in relevant_doc_ids), None)
    dcg = 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_count = min(len(relevant_doc_ids), top_k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "hit": 1.0 if rel_count > 0 else 0.0,
        "relevant_count": float(rel_count),
        "recall": float(rel_count / len(relevant_doc_ids)) if relevant_doc_ids else 0.0,
        "precision": float(rel_count / top_k) if top_k else 0.0,
        "mrr": float(1.0 / first_rel) if first_rel else 0.0,
        "ndcg": float(dcg / idcg) if idcg > 0 else 0.0,
    }


def evaluate_retriever(
    dataset: RetrievalDataset,
    method_name: str,
    retriever: Retriever,
    config: BenchmarkConfig,
    build_runtime_seconds: float,
) -> Dict[str, object]:
    indexed_qrels = dataset.indexed_qrels()
    metric_rows = []
    touched_peers = []
    candidate_docs = []
    query_runtime = 0.0
    doc_ids = dataset.doc_ids
    unavailable = 0
    for idx, row in enumerate(dataset.queries.itertuples(index=False)):
        query_id = str(row.query_id)
        relevant = indexed_qrels.get(query_id, set())
        if not relevant:
            unavailable += 1
            continue
        start = time.perf_counter()
        result = retriever.retrieve(query_id, dataset.query_embeddings[idx], config.top_k)
        query_runtime += time.perf_counter() - start
        retrieved_doc_ids = [doc_ids[doc_idx] for doc_idx in result.doc_indices if 0 <= int(doc_idx) < len(doc_ids)]
        metric_rows.append(_metric_values(retrieved_doc_ids, relevant, config.top_k))
        touched_peers.append(result.touched_peers)
        candidate_docs.append(result.candidate_docs_scored)

    def mean_metric(name: str) -> float:
        return float(np.mean([row[name] for row in metric_rows])) if metric_rows else 0.0

    return {
        "dataset_name": dataset.name,
        "method_name": method_name,
        "seed": int(config.seed),
        "top_k": int(config.top_k),
        "query_count": int(dataset.queries.shape[0]),
        "evaluated_query_count": int(len(metric_rows)),
        "unavailable_query_count": int(unavailable),
        "hit_rate_at_10": mean_metric("hit"),
        "mean_relevant_count_at_10": mean_metric("relevant_count"),
        "recall_at_10": mean_metric("recall"),
        "precision_at_10": mean_metric("precision"),
        "mrr_at_10": mean_metric("mrr"),
        "ndcg_at_10": mean_metric("ndcg"),
        "mean_touched_peers": float(np.mean(touched_peers)) if touched_peers else 0.0,
        "p95_touched_peers": float(np.percentile(touched_peers, 95)) if touched_peers else 0.0,
        "mean_candidate_docs_scored": float(np.mean(candidate_docs)) if candidate_docs else 0.0,
        "p95_candidate_docs_scored": float(np.percentile(candidate_docs, 95)) if candidate_docs else 0.0,
        "build_runtime_seconds": float(build_runtime_seconds),
        "query_runtime_seconds": float(query_runtime),
        "runtime_seconds": float(build_runtime_seconds + query_runtime),
    }


def run_retrieval_benchmark(
    dataset: RetrievalDataset,
    method_names: Sequence[str],
    config: BenchmarkConfig,
    output_dir: str,
    debug_save_query_samples: bool = False,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    validation = dataset.validation_summary()
    with open(os.path.join(output_dir, "dataset_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)
    rows = []
    for method_name in method_names:
        print(f"Building retrieval method {method_name} for {dataset.name}.", flush=True)
        start = time.perf_counter()
        retriever = build_retriever(method_name, dataset, config)
        build_runtime = time.perf_counter() - start
        row = evaluate_retriever(dataset, method_name, retriever, config, build_runtime)
        rows.append(row)
        print(
            f"Finished {method_name}: hit@10={row['hit_rate_at_10']:.4f}, "
            f"mean_rel@10={row['mean_relevant_count_at_10']:.4f}.",
            flush=True,
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    if debug_save_query_samples:
        sample_path = os.path.join(output_dir, "debug_query_sample_note.txt")
        with open(sample_path, "w") as f:
            f.write("Debug query samples are intentionally not implemented for metrics-only benchmark runs.\n")
    return metrics


def prepare_dataset_for_run(
    dataset_name: str,
    data_root: str,
    cache_root: str,
    config: BenchmarkConfig,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    split: str = "test",
    sample_queries: Optional[int] = None,
    sample_docs: Optional[int] = None,
    batch_size: int = 128,
    force_dataset_cache: bool = False,
    force_peer_cache: bool = False,
) -> RetrievalDataset:
    dataset = load_or_prepare_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        cache_root=cache_root,
        config=config,
        model_name=model_name,
        split=split,
        batch_size=batch_size,
        force=force_dataset_cache,
    )
    dataset = subset_dataset(dataset, sample_queries=sample_queries, sample_docs=sample_docs, seed=config.seed)
    if dataset.peer_assignments is None:
        assignments = build_peer_assignments(
            dataset=dataset,
            target_docs_per_peer=config.target_docs_per_peer,
            max_peers=config.max_peers,
            seed=config.seed,
            cache_root=cache_root,
            force=force_peer_cache,
        )
        dataset.peer_assignments = assignments
    return dataset


def config_to_dict(config: BenchmarkConfig) -> Dict[str, object]:
    return {
        key: getattr(config, key)
        for key in BenchmarkConfig.__dataclass_fields__.keys()
    }

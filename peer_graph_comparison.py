"""Budgeted peer-graph comparison utilities.

This module implements the four-way comparison requested for the Semantica
follow-up experiments:

1. Random initialization + Semantica expansion.
2. Tree initialization + Semantica expansion.
3. Random initialization + full NN-Descent.
4. Tree initialization + full NN-Descent.

The public entry point for remote runs is ``run_budgeted_peer_graph_comparison``.
The algorithms below intentionally expose explicit communication and compute
counters so the plots compare methods by cumulative budget rather than by
unequal notions of a "round".
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


UserId = object
TopList = List[Tuple[UserId, float]]
TopByUser = Dict[UserId, TopList]
KnownByUser = Dict[UserId, Set[UserId]]
RandomByUser = Dict[UserId, Set[UserId]]
CentroidTopByUser = Dict[UserId, Dict[int, TopList]]
CentroidKnownByUser = Dict[UserId, Dict[int, Set[UserId]]]
CentroidEmbeddingsByUser = Dict[UserId, np.ndarray]
CentroidIntListsByUser = Dict[UserId, List[int]]
CentroidResponseTask = Tuple[int, UserId, int, TopList, int, Set[UserId], UserId]
CentroidResponseResult = Tuple[int, UserId, int, List[Tuple[UserId, float]], int]
CentroidResponseChunk = Tuple[int, List[CentroidResponseTask], CentroidTopByUser, RandomByUser]
UNBOUNDED_BUDGET = 1e30
SETUP_NAMES = [
    "random_init_semantica_expansion",
    "tree_init_semantica_expansion",
    "random_init_full_nndescent",
    "tree_init_full_nndescent",
]
MULTI_CENTROID_SETUP_NAMES = [
    "random_fixed_k_init_semantica_expansion",
    "random_elbow_k_init_semantica_expansion",
    "tree_fixed_k_init_semantica_expansion",
    "tree_elbow_k_init_semantica_expansion",
    "tree_fixed_k_init_full_nndescent",
    "tree_elbow_k_init_full_nndescent",
]
ALL_SETUP_NAMES = SETUP_NAMES + MULTI_CENTROID_SETUP_NAMES
CENTROID_SEMANTICA_REFERRER_POLICIES = (
    "legacy_mixed_referrers",
    "new_first_plus_random_channel",
)
CENTROID_SEMANTICA_REFERRER_POLICY_DEFINITIONS = {
    "legacy_mixed_referrers": {
        "semantic_referrer_tiers": ["current_ranked_and_protected_random"],
        "random_channel": "none",
        "round_eligibility": "live_topology",
    },
    "new_first_plus_random_channel": {
        "semantic_referrer_tiers": [
            "unasked_expansion_discovered_ranked_contacts",
            "unasked_round_0_ranked_contacts",
        ],
        "random_channel": (
            "one_unasked_protected_random_contact_per_peer_per_round; its unique combined "
            "round-start neighbors are transmitted once and matched against every local centroid"
        ),
        "round_eligibility": (
            "round-start_snapshot; contacts accepted in round_r become eligible in round_r_plus_1"
        ),
        "protected_random_links": "fixed_unranked_never_promoted_and_each_queried_at_most_once",
    },
}
EXPANSION_DIAGNOSTIC_FIELDS = (
    "semantic_new_requests",
    "semantic_initial_requests",
    "random_channel_requests",
    "semantic_candidate_ids",
    "random_channel_candidate_ids",
    "semantic_cosine_checks",
    "random_channel_cosine_checks",
    "semantic_accepted_contacts",
    "random_channel_accepted_contacts",
)
SETUP_CONFIGS = {
    "random_init_semantica_expansion": {"initializer": "random", "refiner": "semantica"},
    "tree_init_semantica_expansion": {"initializer": "tree", "refiner": "semantica"},
    "random_init_full_nndescent": {"initializer": "random", "refiner": "nndescent_full"},
    "tree_init_full_nndescent": {"initializer": "tree", "refiner": "nndescent_full"},
    "random_fixed_k_init_semantica_expansion": {"initializer": "random_fixed_k", "refiner": "semantica"},
    "random_elbow_k_init_semantica_expansion": {"initializer": "random_elbow_k", "refiner": "semantica"},
    "tree_fixed_k_init_semantica_expansion": {"initializer": "tree_fixed_k", "refiner": "semantica"},
    "tree_elbow_k_init_semantica_expansion": {"initializer": "tree_elbow_k", "refiner": "semantica"},
    "tree_fixed_k_init_full_nndescent": {"initializer": "tree_fixed_k", "refiner": "nndescent_full"},
    "tree_elbow_k_init_full_nndescent": {"initializer": "tree_elbow_k", "refiner": "nndescent_full"},
}


@dataclass
class BudgetCost:
    messages: int = 0
    candidate_ids_transmitted: int = 0
    cosine_computations: int = 0

    def per_peer(self, num_peers: int) -> Dict[str, float]:
        divisor = max(1, num_peers)
        return {
            "messages": self.messages / divisor,
            "candidate_ids": self.candidate_ids_transmitted / divisor,
            "cosine_computations": self.cosine_computations / divisor,
        }


@dataclass
class BudgetCaps:
    messages_per_peer: float
    candidate_ids_per_peer: float
    cosine_computations_per_peer: float

    def reached(self, cost: BudgetCost, num_peers: int) -> bool:
        per_peer = cost.per_peer(num_peers)
        return (
            per_peer["messages"] >= self.messages_per_peer
            or per_peer["candidate_ids"] >= self.candidate_ids_per_peer
            or per_peer["cosine_computations"] >= self.cosine_computations_per_peer
        )


@dataclass
class RunState:
    top_by_user: TopByUser
    active_peers: int
    updates_so_far: int
    converged: bool
    stop_reason: str
    runtime_seconds: float
    rounds_completed: int = 0
    random_by_user: RandomByUser = field(default_factory=dict)
    centroid_top_by_user: CentroidTopByUser = field(default_factory=dict)
    centroid_embeddings_by_user: CentroidEmbeddingsByUser = field(default_factory=dict)
    expansion_diagnostics: Dict[str, object] = field(default_factory=dict)


@dataclass
class AlgorithmResult:
    setup_name: str
    initializer: str
    refiner: str
    top_by_user: TopByUser
    known_by_user: KnownByUser
    trace_rows: List[Dict[str, object]]
    final_cost: BudgetCost
    converged: bool
    stop_reason: str
    runtime_seconds: float
    updates_so_far: int
    rounds_completed: int = 0
    random_by_user: RandomByUser = field(default_factory=dict)
    centroid_top_by_user: CentroidTopByUser = field(default_factory=dict)
    centroid_known_by_user: CentroidKnownByUser = field(default_factory=dict)
    centroid_embeddings_by_user: CentroidEmbeddingsByUser = field(default_factory=dict)
    centroid_contact_caps_by_user: CentroidIntListsByUser = field(default_factory=dict)
    centroid_cluster_doc_counts_by_user: CentroidIntListsByUser = field(default_factory=dict)
    expansion_diagnostics: Dict[str, object] = field(default_factory=dict)


@dataclass
class InitialGraph:
    initializer: str
    known_by_user: KnownByUser
    top_by_user: TopByUser
    runtime_seconds: float
    initial_candidates_per_peer: float
    candidate_ids_generated_per_peer: float
    initial_recall_at_n: float
    initial_common_exact_topn: float
    initial_document_retrieval_success: float
    initial_centroid_routed_document_retrieval_success: float = float("nan")
    initial_query_hit_rate: float = float("nan")
    initial_macro_peer_hit_rate: float = float("nan")
    initial_query_hit_count: int = 0
    initial_evaluated_query_count: int = 0
    initial_primary_hit_metric_variant: str = "direct_neighbors"
    random_by_user: RandomByUser = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)
    centroid_top_by_user: CentroidTopByUser = field(default_factory=dict)
    centroid_known_by_user: CentroidKnownByUser = field(default_factory=dict)
    centroid_embeddings_by_user: CentroidEmbeddingsByUser = field(default_factory=dict)
    centroid_contact_caps_by_user: CentroidIntListsByUser = field(default_factory=dict)
    centroid_cluster_doc_counts_by_user: CentroidIntListsByUser = field(default_factory=dict)


@dataclass(frozen=True)
class CentroidChainRouteResult:
    found: bool
    hops: int
    stop_reason: str


@dataclass
class NeighborEntry:
    sim: float
    is_new: bool = True


class SimilarityIndex:
    """Fast user-id aware cosine-similarity access.

    Small/default runs can use the historical dense similarity dataframe. Larger
    runs can provide average embeddings and compute only the similarities they
    need, avoiding a dense 50k x 50k dataframe per array task.
    """

    def __init__(
        self,
        cosine_sims: Optional[pd.DataFrame] = None,
        users: Optional[Sequence[UserId]] = None,
        embeddings: Optional[Mapping[UserId, np.ndarray]] = None,
    ):
        if cosine_sims is None and embeddings is None:
            raise ValueError("Provide either cosine_sims or embeddings.")
        if users is None:
            users = list(cosine_sims.index) if cosine_sims is not None else list(embeddings.keys())
        self.users = list(users)
        self.user_set = set(self.users)
        self.user_to_pos = {user: idx for idx, user in enumerate(self.users)}
        self.matrix = None
        self.embeddings = None
        self._exact_topn_cache: Dict[int, Dict[UserId, List[UserId]]] = {}

        if cosine_sims is not None:
            matrix_df = cosine_sims.loc[self.users, self.users].copy()
            matrix = matrix_df.to_numpy(dtype=float)
            matrix = np.nan_to_num(matrix, nan=-np.inf)
            np.fill_diagonal(matrix, -np.inf)
            self.matrix = matrix
        else:
            rows = []
            for user in self.users:
                if user not in embeddings:
                    raise ValueError(f"Missing average embedding for user {user!r}.")
                rows.append(np.asarray(embeddings[user], dtype=np.float32))
            matrix = np.vstack(rows).astype(np.float32, copy=False)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = matrix / norms

    @classmethod
    def from_avg_embeddings(cls, df_avgemb: pd.DataFrame, users: Optional[Sequence[UserId]] = None) -> "SimilarityIndex":
        if users is None:
            users = list(df_avgemb["AnonID"])
        embeddings = {
            row.AnonID: np.asarray(row.embedding, dtype=np.float32)
            for row in df_avgemb[df_avgemb["AnonID"].isin(users)].itertuples(index=False)
        }
        return cls(cosine_sims=None, users=users, embeddings=embeddings)

    def sim(self, user: UserId, candidate: UserId) -> float:
        if user == candidate:
            return -np.inf
        user_pos = self.user_to_pos.get(user)
        cand_pos = self.user_to_pos.get(candidate)
        if user_pos is None or cand_pos is None:
            return -np.inf
        if self.matrix is not None:
            return float(self.matrix[user_pos, cand_pos])
        return float(np.dot(self.embeddings[user_pos], self.embeddings[cand_pos]))

    def valid_candidates(self, user: UserId, candidates: Iterable[UserId]) -> List[UserId]:
        return [cand for cand in candidates if cand != user and cand in self.user_set]

    def rank_candidates(self, user: UserId, candidates: Iterable[UserId], k: int) -> TopList:
        if k <= 0:
            return []
        seen = set()
        candidate_list = []
        for candidate in candidates:
            if candidate in seen or candidate == user or candidate not in self.user_set:
                continue
            seen.add(candidate)
            candidate_list.append(candidate)
        if not candidate_list:
            return []

        if self.matrix is not None:
            ranked = []
            user_pos = self.user_to_pos[user]
            for candidate in candidate_list:
                cand_pos = self.user_to_pos[candidate]
                sim = float(self.matrix[user_pos, cand_pos])
                if np.isfinite(sim):
                    ranked.append((candidate, sim))
        else:
            user_pos = self.user_to_pos[user]
            candidate_positions = np.array([self.user_to_pos[candidate] for candidate in candidate_list], dtype=np.int64)
            sims = self.embeddings[candidate_positions] @ self.embeddings[user_pos]
            ranked = [
                (candidate, float(sim))
                for candidate, sim in zip(candidate_list, sims)
                if np.isfinite(sim)
            ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]

    def exact_topn(self, k: int) -> Dict[UserId, List[UserId]]:
        if k in self._exact_topn_cache:
            return self._exact_topn_cache[k]
        if k <= 0:
            exact = {user: [] for user in self.users}
            self._exact_topn_cache[k] = exact
            return exact

        exact = {}
        if self.matrix is not None:
            for user in self.users:
                pos = self.user_to_pos[user]
                row = self.matrix[pos]
                if len(row) <= 1:
                    exact[user] = []
                    continue
                top_positions = np.argsort(row)[::-1][: min(k, len(row) - 1)]
                exact[user] = [self.users[idx] for idx in top_positions if np.isfinite(row[idx])]
            self._exact_topn_cache[k] = exact
            return exact

        block_size = int(os.environ.get("SEMANTICA_EXACT_TOPN_BLOCK_SIZE", "256"))
        block_size = max(1, block_size)
        target_k = min(k, max(0, len(self.users) - 1))
        if target_k == 0:
            exact = {user: [] for user in self.users}
            self._exact_topn_cache[k] = exact
            return exact

        print(f"Computing exact top-{target_k} peers from embeddings in blocks of {block_size}...", flush=True)
        all_embeddings_t = self.embeddings.T
        for start in range(0, len(self.users), block_size):
            end = min(start + block_size, len(self.users))
            sims = self.embeddings[start:end] @ all_embeddings_t
            rows = np.arange(end - start)
            sims[rows, np.arange(start, end)] = -np.inf
            if target_k >= sims.shape[1]:
                top_positions = np.argsort(-sims, axis=1)[:, :target_k]
            else:
                partition = np.argpartition(-sims, target_k - 1, axis=1)[:, :target_k]
                partition_sims = np.take_along_axis(sims, partition, axis=1)
                order = np.argsort(-partition_sims, axis=1)
                top_positions = np.take_along_axis(partition, order, axis=1)
            for row_offset, positions in enumerate(top_positions):
                user = self.users[start + row_offset]
                exact[user] = [self.users[idx] for idx in positions if np.isfinite(sims[row_offset, idx])]
            print(f"Computed exact top peers for {end}/{len(self.users)} users.", flush=True)
        self._exact_topn_cache[k] = exact
        return exact


class MetricEvaluator:
    def __init__(
        self,
        users: Sequence[UserId],
        exact_topn: Mapping[UserId, Sequence[UserId]],
        user_docs_train: pd.DataFrame,
        user_docs_test: pd.DataFrame,
        n: int,
        doc_embeddings: Optional[pd.DataFrame] = None,
    ):
        self.users = list(users)
        self.exact_topn = {user: set(values) for user, values in exact_topn.items()}
        self.n = n
        self.user_docs_train_df = user_docs_train
        self.user_docs_test_df = user_docs_test
        self.doc_embeddings_df = doc_embeddings
        self.user_train_docs = user_docs_train.groupby("AnonID")["DocIndex"].apply(set).to_dict()
        self.user_test_docs = user_docs_test.groupby("AnonID")["DocIndex"].apply(list).to_dict()
        self.doc_embedding_by_id: Dict[object, np.ndarray] = {}
        if doc_embeddings is not None and {"DocIndex", "embedding"}.issubset(doc_embeddings.columns):
            for row in doc_embeddings[["DocIndex", "embedding"]].itertuples(index=False):
                self.doc_embedding_by_id[row.DocIndex] = _normalize_embedding(
                    np.asarray(row.embedding, dtype=np.float32)
                ).astype(np.float32, copy=False)
        if "embedding" in user_docs_test.columns:
            for row in user_docs_test[["DocIndex", "embedding"]].dropna(subset=["embedding"]).itertuples(index=False):
                self.doc_embedding_by_id[row.DocIndex] = _normalize_embedding(
                    np.asarray(row.embedding, dtype=np.float32)
                ).astype(np.float32, copy=False)

    def evaluate(
        self,
        top_by_user: Mapping[UserId, Sequence[Tuple[UserId, float]]],
        random_by_user: Optional[Mapping[UserId, Iterable[UserId]]] = None,
        centroid_top_by_user: Optional[Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]]] = None,
        centroid_embeddings_by_user: Optional[Mapping[UserId, np.ndarray]] = None,
        include_document_metrics: bool = True,
    ) -> Dict[str, object]:
        random_by_user = random_by_user or {}
        common_counts = []
        recalls = []
        retrieval_scores = []
        centroid_routed_scores = []
        retrieval_hit_count = 0
        retrieval_query_count = 0
        centroid_routed_hit_count = 0
        centroid_routed_query_count = 0
        has_centroid_routing = (
            include_document_metrics
            and bool(centroid_top_by_user)
            and bool(centroid_embeddings_by_user)
            and bool(self.doc_embedding_by_id)
        )
        for user in self.users:
            top_ids = [item[0] if isinstance(item, tuple) else item for item in top_by_user.get(user, [])]
            top_set = set(top_ids[: self.n])
            exact_set = self.exact_topn.get(user, set())
            common = len(top_set.intersection(exact_set))
            common_counts.append(common)
            recalls.append(common / max(1, min(self.n, len(exact_set))))

            if not include_document_metrics:
                continue
            test_docs = self.user_test_docs.get(user)
            if not test_docs:
                continue
            held_docs = set()
            reachable_peers = list(top_ids[: self.n]) + list(random_by_user.get(user, []))
            for peer in reachable_peers:
                held_docs.update(self.user_train_docs.get(peer, set()))
            user_retrieval_hits = sum(1 for doc_id in test_docs if doc_id in held_docs)
            retrieval_scores.append(user_retrieval_hits / len(test_docs))
            retrieval_hit_count += user_retrieval_hits
            retrieval_query_count += len(test_docs)

            if has_centroid_routing:
                routed_hits = 0
                routed_total = 0
                user_centroid_links = centroid_top_by_user.get(user, {})
                candidate_peers = []
                for values in user_centroid_links.values():
                    candidate_peers.extend(peer for peer, _score in values)
                protected_random = set(random_by_user.get(user, set()))
                for doc_id in test_docs:
                    doc_embedding = self.doc_embedding_by_id.get(doc_id)
                    if doc_embedding is None:
                        continue
                    routed_total += 1
                    ranked_routed = _rank_query_centroid_neighbors(
                        query_embedding=doc_embedding,
                        candidate_peers=candidate_peers,
                        centroid_embeddings_by_user=centroid_embeddings_by_user,
                        limit=self.n,
                        excluded_peers=protected_random.union({user}),
                    )
                    routed_peers = [peer for peer, _score in ranked_routed] + list(protected_random)
                    if any(doc_id in self.user_train_docs.get(peer, set()) for peer in routed_peers):
                        routed_hits += 1
                if routed_total:
                    centroid_routed_scores.append(routed_hits / routed_total)

                centroid_routed_hit_count += routed_hits
                centroid_routed_query_count += routed_total

        document_macro_peer_hit_rate = (
            float(np.mean(retrieval_scores))
            if include_document_metrics and retrieval_scores
            else float("nan")
        )
        document_query_hit_rate = (
            float(retrieval_hit_count / retrieval_query_count)
            if include_document_metrics and retrieval_query_count
            else float("nan")
        )
        centroid_routed_macro_peer_hit_rate = (
            float(np.mean(centroid_routed_scores))
            if include_document_metrics and centroid_routed_scores
            else float("nan")
        )
        centroid_routed_query_hit_rate = (
            float(centroid_routed_hit_count / centroid_routed_query_count)
            if include_document_metrics and centroid_routed_query_count
            else float("nan")
        )
        if has_centroid_routing:
            primary_hit_metric_variant = "centroid_routed_direct_neighbors"
            query_hit_rate = centroid_routed_query_hit_rate
            macro_peer_hit_rate = centroid_routed_macro_peer_hit_rate
            query_hit_count = centroid_routed_hit_count
            evaluated_query_count = centroid_routed_query_count
        else:
            primary_hit_metric_variant = "direct_neighbors"
            query_hit_rate = document_query_hit_rate
            macro_peer_hit_rate = document_macro_peer_hit_rate
            query_hit_count = retrieval_hit_count
            evaluated_query_count = retrieval_query_count

        return {
            "closest_peer_recall_at_50": float(np.mean(recalls)) if recalls else 0.0,
            "mean_common_exact_top50": float(np.mean(common_counts)) if common_counts else 0.0,
            "document_retrieval_success": (
                document_macro_peer_hit_rate * 100.0
            ),
            "centroid_routed_document_retrieval_success": (
                centroid_routed_macro_peer_hit_rate * 100.0
            ),
            "document_query_hit_rate": document_query_hit_rate,
            "document_macro_peer_hit_rate": document_macro_peer_hit_rate,
            "document_query_hit_count": int(retrieval_hit_count),
            "document_evaluated_query_count": int(retrieval_query_count),
            "centroid_routed_query_hit_rate": centroid_routed_query_hit_rate,
            "centroid_routed_macro_peer_hit_rate": centroid_routed_macro_peer_hit_rate,
            "centroid_routed_query_hit_count": int(centroid_routed_hit_count),
            "centroid_routed_evaluated_query_count": int(centroid_routed_query_count),
            "query_hit_rate": query_hit_rate,
            "macro_peer_hit_rate": macro_peer_hit_rate,
            "query_hit_count": int(query_hit_count),
            "evaluated_query_count": int(evaluated_query_count),
            "primary_hit_metric_variant": primary_hit_metric_variant,
        }


class BudgetTraceRecorder:
    def __init__(
        self,
        setup_name: str,
        initializer: str,
        refiner: str,
        num_peers: int,
        checkpoints_by_axis: Mapping[str, Sequence[float]],
        metric_evaluator: MetricEvaluator,
        include_document_metrics: bool = True,
    ):
        self.setup_name = setup_name
        self.initializer = initializer
        self.refiner = refiner
        self.num_peers = num_peers
        self.checkpoints_by_axis = {axis: list(values) for axis, values in checkpoints_by_axis.items()}
        self.next_index = {axis: 0 for axis in self.checkpoints_by_axis}
        self.metric_evaluator = metric_evaluator
        self.include_document_metrics = bool(include_document_metrics)
        self.rows: List[Dict[str, object]] = []

    def maybe_record(self, cost: BudgetCost, state: RunState) -> None:
        per_peer = cost.per_peer(self.num_peers)
        axis_values = dict(per_peer)
        axis_values["rounds"] = float(state.rounds_completed)
        metrics_cache: Optional[Dict[str, object]] = None
        for axis, checkpoints in self.checkpoints_by_axis.items():
            if axis not in axis_values:
                continue
            while (
                self.next_index[axis] < len(checkpoints)
                and axis_values[axis] >= checkpoints[self.next_index[axis]]
            ):
                if metrics_cache is None:
                    metrics_cache = self.metric_evaluator.evaluate(
                        state.top_by_user,
                        state.random_by_user,
                        centroid_top_by_user=state.centroid_top_by_user,
                        centroid_embeddings_by_user=state.centroid_embeddings_by_user,
                        include_document_metrics=self.include_document_metrics,
                    )
                self.rows.append(
                    self._row(
                        checkpoint_axis=axis,
                        checkpoint_value=checkpoints[self.next_index[axis]],
                        per_peer=per_peer,
                        metrics=metrics_cache,
                        state=state,
                    )
                )
                self.next_index[axis] += 1

    def carry_forward_converged(self, cost: BudgetCost, state: RunState) -> None:
        per_peer = cost.per_peer(self.num_peers)
        metrics = self.metric_evaluator.evaluate(
            state.top_by_user,
            state.random_by_user,
            centroid_top_by_user=state.centroid_top_by_user,
            centroid_embeddings_by_user=state.centroid_embeddings_by_user,
            include_document_metrics=self.include_document_metrics,
        )
        for axis, checkpoints in self.checkpoints_by_axis.items():
            while self.next_index[axis] < len(checkpoints):
                self.rows.append(
                    self._row(
                        checkpoint_axis=axis,
                        checkpoint_value=checkpoints[self.next_index[axis]],
                        per_peer=per_peer,
                        metrics=metrics,
                        state=state,
                    )
                )
                self.next_index[axis] += 1

    def _row(
        self,
        checkpoint_axis: str,
        checkpoint_value: float,
        per_peer: Mapping[str, float],
        metrics: Mapping[str, object],
        state: RunState,
    ) -> Dict[str, object]:
        row = {
            "setup_name": self.setup_name,
            "initializer": self.initializer,
            "refiner": self.refiner,
            "checkpoint_axis": checkpoint_axis,
            "checkpoint_value": checkpoint_value,
            "messages_per_peer": per_peer["messages"],
            "candidate_ids_transmitted_per_peer": per_peer["candidate_ids"],
            "cosine_computations_per_peer": per_peer["cosine_computations"],
            "closest_peer_recall_at_50": metrics["closest_peer_recall_at_50"],
            "mean_common_exact_top50": metrics["mean_common_exact_top50"],
            "document_retrieval_success": metrics["document_retrieval_success"],
            "centroid_routed_document_retrieval_success": metrics["centroid_routed_document_retrieval_success"],
            "query_hit_rate": metrics["query_hit_rate"],
            "macro_peer_hit_rate": metrics["macro_peer_hit_rate"],
            "query_hit_count": metrics["query_hit_count"],
            "evaluated_query_count": metrics["evaluated_query_count"],
            "document_query_hit_rate": metrics["document_query_hit_rate"],
            "document_macro_peer_hit_rate": metrics["document_macro_peer_hit_rate"],
            "centroid_routed_query_hit_rate": metrics["centroid_routed_query_hit_rate"],
            "centroid_routed_macro_peer_hit_rate": metrics["centroid_routed_macro_peer_hit_rate"],
            "primary_hit_metric_variant": metrics["primary_hit_metric_variant"],
            "active_peers": state.active_peers,
            "updates_so_far": state.updates_so_far,
            "converged": state.converged,
            "stop_reason": state.stop_reason,
            "runtime_seconds": state.runtime_seconds,
            "rounds_completed": int(state.rounds_completed),
            "expansion_round": int(state.rounds_completed),
            "query_hop": 1,
            "measurement_step_type": (
                "expansion_round_checkpoint" if checkpoint_axis == "rounds" else "budget_checkpoint"
            ),
            "query_hop_semantics": "direct_outgoing_peer_contact_after_expansion_checkpoint",
        }
        row.update(state.expansion_diagnostics)
        return row


def make_1_2_5_checkpoints(max_value: float) -> List[float]:
    if max_value <= 0:
        return []
    checkpoints = []
    scale = 1.0
    while scale <= max_value * 10:
        for multiplier in (1.0, 2.0, 5.0):
            value = multiplier * scale
            if value <= max_value:
                checkpoints.append(value)
        scale *= 10.0
    return checkpoints


def make_dense_message_checkpoints(max_messages_per_peer: float) -> List[float]:
    """Submission checkpoints: every integer 1..50, then 100/200/500."""
    if max_messages_per_peer <= 0:
        return []
    dense_stop = int(min(50, math.floor(max_messages_per_peer)))
    checkpoints = [float(value) for value in range(1, dense_stop + 1)]
    for value in (100.0, 200.0, 500.0):
        if value <= max_messages_per_peer and value not in checkpoints:
            checkpoints.append(value)
    return checkpoints


def make_budget_checkpoints(caps: BudgetCaps, dense_messages: bool = True) -> Dict[str, List[float]]:
    if dense_messages:
        return {
            "messages": make_dense_message_checkpoints(caps.messages_per_peer),
            "candidate_ids": [],
            "cosine_computations": [],
        }
    return {
        "messages": make_1_2_5_checkpoints(caps.messages_per_peer),
        "candidate_ids": make_1_2_5_checkpoints(caps.candidate_ids_per_peer),
        "cosine_computations": make_1_2_5_checkpoints(caps.cosine_computations_per_peer),
    }


def top_ids(top_by_user: Mapping[UserId, Sequence[Tuple[UserId, float]]], user: UserId, n: int) -> List[UserId]:
    return [peer for peer, _ in list(top_by_user.get(user, []))[:n]]


def copy_known(known_by_user: Mapping[UserId, Set[UserId]]) -> KnownByUser:
    return {user: set(candidates) for user, candidates in known_by_user.items()}


def copy_top(top_by_user: Mapping[UserId, TopList]) -> TopByUser:
    return {user: list(values) for user, values in top_by_user.items()}


def copy_random(random_by_user: Mapping[UserId, Iterable[UserId]]) -> RandomByUser:
    return {user: set(values) for user, values in random_by_user.items()}


def copy_centroid_top(centroid_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]]) -> CentroidTopByUser:
    return {
        user: {int(index): list(values) for index, values in by_centroid.items()}
        for user, by_centroid in centroid_top_by_user.items()
    }


def copy_centroid_known(centroid_known_by_user: Mapping[UserId, Mapping[int, Iterable[UserId]]]) -> CentroidKnownByUser:
    return {
        user: {int(index): set(values) for index, values in by_centroid.items()}
        for user, by_centroid in centroid_known_by_user.items()
    }


def snapshot_centroid_neighbor_ids(
    centroid_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]],
) -> Dict[UserId, Dict[int, Set[UserId]]]:
    return {
        user: {
            int(centroid_index): {peer for peer, _score in values}
            for centroid_index, values in by_centroid.items()
        }
        for user, by_centroid in centroid_top_by_user.items()
    }


def select_centroid_referrer(
    ranked_referrers: Sequence[UserId],
    protected_random_referrers: Sequence[UserId],
    initial_referrers: Set[UserId],
    asked_referrers: Set[UserId],
    rng: random.Random,
    policy: str,
) -> Optional[Tuple[UserId, str]]:
    if policy not in CENTROID_SEMANTICA_REFERRER_POLICIES:
        raise ValueError(f"Unsupported centroid Semantica referrer policy: {policy}")
    ranked = _ordered_unique(ranked_referrers)
    if policy == "legacy_mixed_referrers":
        eligible = [
            peer
            for peer in _ordered_unique(ranked + list(protected_random_referrers))
            if peer not in asked_referrers
        ]
        if not eligible:
            return None
        return rng.choice(eligible), "legacy"

    new_contacts = [
        peer for peer in ranked if peer not in initial_referrers and peer not in asked_referrers
    ]
    if new_contacts:
        return rng.choice(new_contacts), "new"
    initial_contacts = [
        peer for peer in ranked if peer in initial_referrers and peer not in asked_referrers
    ]
    if initial_contacts:
        return rng.choice(initial_contacts), "initial"
    return None


def combined_centroid_neighbor_ids(
    referrer: UserId,
    centroid_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]],
    random_by_user: Mapping[UserId, Iterable[UserId]],
) -> List[UserId]:
    candidates: List[UserId] = []
    for centroid_index in sorted(centroid_top_by_user.get(referrer, {})):
        candidates.extend(
            peer for peer, _score in centroid_top_by_user[referrer][centroid_index]
        )
    candidates.extend(sorted(random_by_user.get(referrer, set()), key=repr))
    return _ordered_unique(candidates)


def expansion_diagnostic_snapshot(
    cumulative: Mapping[str, int],
    current_round: Mapping[str, int],
    num_peers: int,
) -> Dict[str, object]:
    divisor = max(1, int(num_peers))
    output: Dict[str, object] = {}
    for field_name in EXPANSION_DIAGNOSTIC_FIELDS:
        cumulative_value = int(cumulative.get(field_name, 0))
        round_value = int(current_round.get(field_name, 0))
        output[f"cumulative_{field_name}"] = cumulative_value
        output[f"cumulative_{field_name}_per_peer"] = cumulative_value / divisor
        output[f"round_{field_name}"] = round_value
        output[f"round_{field_name}_per_peer"] = round_value / divisor
    return output


def _centroid_candidate_score(
    source_embedding: np.ndarray,
    candidate: UserId,
    centroid_embeddings_by_user: Mapping[UserId, np.ndarray],
) -> float:
    candidate_centroids = centroid_embeddings_by_user.get(candidate)
    if candidate_centroids is None or len(candidate_centroids) == 0:
        return -np.inf
    source_embedding = np.asarray(source_embedding, dtype=np.float32)
    candidate_centroids = np.asarray(candidate_centroids, dtype=np.float32)
    if candidate_centroids.ndim == 1:
        return float(np.dot(source_embedding, candidate_centroids))
    return float(np.max(candidate_centroids @ source_embedding))


def apply_random_channel_candidates(
    user: UserId,
    candidate_ids: Sequence[UserId],
    centroid_top_by_user: CentroidTopByUser,
    centroid_embeddings_by_user: Mapping[UserId, np.ndarray],
    centroid_contact_caps_by_user: Mapping[UserId, Sequence[int]],
    random_by_user: Mapping[UserId, Set[UserId]],
    valid_users: Set[UserId],
) -> Tuple[int, int]:
    user_centroids = centroid_embeddings_by_user.get(user)
    if user_centroids is None or len(user_centroids) == 0:
        return 0, 0

    existing_peers = _centroid_top_peer_set(centroid_top_by_user, user)
    protected_random = random_by_user.get(user, set())
    assignments: List[Tuple[float, str, UserId, int]] = []
    cosine_checks = 0
    for candidate in _ordered_unique(candidate_ids):
        if (
            candidate == user
            or candidate not in valid_users
            or candidate in existing_peers
            or candidate in protected_random
        ):
            continue
        best: Optional[Tuple[float, int]] = None
        for centroid_index, source_embedding in enumerate(user_centroids):
            cap_values = centroid_contact_caps_by_user.get(user, [])
            source_cap = int(cap_values[centroid_index]) if centroid_index < len(cap_values) else 0
            if source_cap <= 0:
                continue
            score = _centroid_candidate_score(
                source_embedding,
                candidate,
                centroid_embeddings_by_user,
            )
            cosine_checks += 1
            if not np.isfinite(score):
                continue
            source_top = centroid_top_by_user.get(user, {}).get(centroid_index, [])
            improves = len(source_top) < source_cap or score > source_top[-1][1]
            if improves and (best is None or score > best[0] or (score == best[0] and centroid_index < best[1])):
                best = (score, centroid_index)
        if best is not None:
            assignments.append((best[0], repr(candidate), candidate, best[1]))

    assignments.sort(key=lambda item: (-item[0], item[1], item[3]))
    accepted = 0
    for score, _candidate_key, candidate, centroid_index in assignments:
        if candidate in _centroid_top_peer_set(centroid_top_by_user, user):
            continue
        if _accept_centroid_candidate(
            user=user,
            centroid_index=centroid_index,
            candidate=candidate,
            sim=score,
            centroid_top_by_user=centroid_top_by_user,
            caps_by_user=centroid_contact_caps_by_user,
            protected_random=protected_random,
        ):
            accepted += 1
    return accepted, cosine_checks


def _rank_query_centroid_neighbors(
    query_embedding: np.ndarray,
    candidate_peers: Iterable[UserId],
    centroid_embeddings_by_user: Mapping[UserId, np.ndarray],
    limit: int,
    excluded_peers: Optional[Set[UserId]] = None,
) -> TopList:
    if limit <= 0:
        return []
    excluded_peers = excluded_peers or set()
    ranked: TopList = []
    seen: Set[UserId] = set()
    for candidate in candidate_peers:
        if candidate in seen or candidate in excluded_peers:
            continue
        seen.add(candidate)
        score = _centroid_candidate_score(query_embedding, candidate, centroid_embeddings_by_user)
        if np.isfinite(score):
            ranked.append((candidate, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def centroid_chain_contacts(
    user: UserId,
    centroid_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]],
    random_by_user: Optional[Mapping[UserId, Iterable[UserId]]] = None,
) -> List[UserId]:
    contacts: List[UserId] = []
    seen: Set[UserId] = set()
    for values in centroid_top_by_user.get(user, {}).values():
        for peer, _score in values:
            if peer not in seen:
                seen.add(peer)
                contacts.append(peer)
    if random_by_user:
        for peer in random_by_user.get(user, []):
            if peer not in seen:
                seen.add(peer)
                contacts.append(peer)
    return contacts


def route_centroid_chain_query(
    query_embedding: np.ndarray,
    source_user: UserId,
    target_doc: object,
    user_train_docs: Mapping[UserId, Set[object]],
    centroid_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]],
    random_by_user: Optional[Mapping[UserId, Iterable[UserId]]],
    centroid_embeddings_by_user: Mapping[UserId, np.ndarray],
    max_hops: int = 50,
) -> CentroidChainRouteResult:
    if max_hops <= 0:
        return CentroidChainRouteResult(False, 0, "max_hops")
    visited: Set[UserId] = {source_user}
    current_user = source_user
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    for hop in range(1, max_hops + 1):
        contacts = centroid_chain_contacts(current_user, centroid_top_by_user, random_by_user)
        if not contacts:
            return CentroidChainRouteResult(False, hop - 1, "no_contacts")
        ranked = _rank_query_centroid_neighbors(
            query_embedding=query_embedding,
            candidate_peers=contacts,
            centroid_embeddings_by_user=centroid_embeddings_by_user,
            limit=1,
            excluded_peers=visited,
        )
        if not ranked:
            return CentroidChainRouteResult(False, hop - 1, "no_unvisited_candidates")
        next_peer = ranked[0][0]
        visited.add(next_peer)
        if target_doc in user_train_docs.get(next_peer, set()):
            return CentroidChainRouteResult(True, hop, "found")
        current_user = next_peer
    return CentroidChainRouteResult(False, max_hops, "max_hops")


def _doc_embedding_map(
    doc_embeddings: Optional[pd.DataFrame],
    user_docs_test: Optional[pd.DataFrame] = None,
) -> Dict[object, np.ndarray]:
    result: Dict[object, np.ndarray] = {}
    for frame in (doc_embeddings, user_docs_test):
        if frame is None or not {"DocIndex", "embedding"}.issubset(frame.columns):
            continue
        for row in frame[["DocIndex", "embedding"]].dropna(subset=["embedding"]).itertuples(index=False):
            result[row.DocIndex] = _normalize_embedding(np.asarray(row.embedding, dtype=np.float32)).astype(
                np.float32,
                copy=False,
            )
    return result


def _train_doc_maps(user_docs_train: pd.DataFrame) -> Tuple[Dict[UserId, Set[object]], Dict[object, Set[UserId]]]:
    user_train_docs = user_docs_train.groupby("AnonID")["DocIndex"].apply(set).to_dict()
    doc_train_owners: Dict[object, Set[UserId]] = defaultdict(set)
    for row in user_docs_train[["AnonID", "DocIndex"]].itertuples(index=False):
        doc_train_owners[row.DocIndex].add(row.AnonID)
    return user_train_docs, doc_train_owners


def _finalize_centroid_chain_counts(counts: Mapping[str, object], hop_cutoffs: Sequence[int]) -> Dict[str, object]:
    query_count = int(counts.get("query_count", 0))
    available_query_count = int(counts.get("available_query_count", 0))
    routed_query_count = int(counts.get("routed_query_count", 0))
    success_count = int(counts.get("success_count", 0))
    available_success_count = int(counts.get("available_success_count", 0))
    success_hop_counts = {int(k): int(v) for k, v in dict(counts.get("success_hop_counts", {})).items()}
    failure_hop_counts = {int(k): int(v) for k, v in dict(counts.get("failure_hop_counts", {})).items()}
    termination_hop_counts = {
        int(k): int(v) for k, v in dict(counts.get("termination_hop_counts", {})).items()
    }
    stop_reason_counts = {str(k): int(v) for k, v in dict(counts.get("stop_reason_counts", {})).items()}
    total_routing_messages = int(counts.get("total_routing_messages", 0))
    success_hops: List[int] = []
    for hop, count in sorted(success_hop_counts.items()):
        success_hops.extend([hop] * count)
    summary: Dict[str, object] = {
        "query_count": query_count,
        "available_query_count": available_query_count,
        "routed_query_count": routed_query_count,
        "unavailable_target_count": int(counts.get("unavailable_target_count", 0)),
        "success_count": success_count,
        "available_success_count": available_success_count,
        "centroid_chain_doc_retrieval_success": 100.0 * success_count / query_count if query_count else 0.0,
        "centroid_chain_available_doc_recall": (
            100.0 * available_success_count / available_query_count if available_query_count else 0.0
        ),
        "mean_success_hops": float(np.mean(success_hops)) if success_hops else float("nan"),
        "median_success_hops": float(np.median(success_hops)) if success_hops else float("nan"),
        "total_routing_messages": total_routing_messages,
        "mean_routing_messages_per_query": (
            total_routing_messages / query_count if query_count else 0.0
        ),
        "mean_routing_messages_per_routed_query": (
            total_routing_messages / routed_query_count if routed_query_count else 0.0
        ),
        "success_hop_counts": success_hop_counts,
        "failure_hop_counts": failure_hop_counts,
        "termination_hop_counts": termination_hop_counts,
        "stop_reason_counts": stop_reason_counts,
    }
    for cutoff in hop_cutoffs:
        cutoff = int(cutoff)
        cutoff_success = sum(count for hop, count in success_hop_counts.items() if hop <= cutoff)
        summary[f"centroid_chain_doc_retrieval_success_at_{cutoff}_hops"] = (
            100.0 * cutoff_success / query_count if query_count else 0.0
        )
        summary[f"centroid_chain_available_doc_recall_at_{cutoff}_hops"] = (
            100.0 * cutoff_success / available_query_count if available_query_count else 0.0
        )
    return summary


def combine_centroid_chain_summaries(
    summaries: Sequence[Mapping[str, object]],
    hop_cutoffs: Sequence[int] = (1, 2, 5, 10, 20, 50),
) -> Dict[str, object]:
    combined: Dict[str, object] = {
        "query_count": 0,
        "available_query_count": 0,
        "routed_query_count": 0,
        "unavailable_target_count": 0,
        "success_count": 0,
        "available_success_count": 0,
        "success_hop_counts": Counter(),
        "failure_hop_counts": Counter(),
        "termination_hop_counts": Counter(),
        "stop_reason_counts": Counter(),
        "total_routing_messages": 0,
    }
    for summary in summaries:
        for key in (
            "query_count",
            "available_query_count",
            "routed_query_count",
            "unavailable_target_count",
            "success_count",
            "available_success_count",
            "total_routing_messages",
        ):
            combined[key] = int(combined.get(key, 0)) + int(summary.get(key, 0))
        combined["success_hop_counts"].update({int(k): int(v) for k, v in summary.get("success_hop_counts", {}).items()})
        combined["failure_hop_counts"].update(
            {int(k): int(v) for k, v in summary.get("failure_hop_counts", {}).items()}
        )
        combined["termination_hop_counts"].update(
            {int(k): int(v) for k, v in summary.get("termination_hop_counts", {}).items()}
        )
        combined["stop_reason_counts"].update({str(k): int(v) for k, v in summary.get("stop_reason_counts", {}).items()})
    return _finalize_centroid_chain_counts(combined, hop_cutoffs)


def centroid_chain_hop_rows(
    summary: Mapping[str, object],
    max_hops: int,
    expansion_round: int,
    topology_stage: str,
) -> List[Dict[str, object]]:
    query_count = int(summary.get("query_count", 0))
    available_query_count = int(summary.get("available_query_count", 0))
    success_hop_counts = {
        int(key): int(value) for key, value in summary.get("success_hop_counts", {}).items()
    }
    failure_hop_counts = {
        int(key): int(value) for key, value in summary.get("failure_hop_counts", {}).items()
    }
    termination_hop_counts = {
        int(key): int(value) for key, value in summary.get("termination_hop_counts", {}).items()
    }
    cumulative_success = 0
    rows: List[Dict[str, object]] = []
    for hop in range(1, int(max_hops) + 1):
        exact_success = success_hop_counts.get(hop, 0)
        cumulative_success += exact_success
        rows.append(
            {
                "topology_stage": topology_stage,
                "expansion_round": int(expansion_round),
                "query_hop": hop,
                "exact_success_count": exact_success,
                "exact_failed_termination_count": failure_hop_counts.get(hop, 0),
                "exact_termination_count": termination_hop_counts.get(hop, 0),
                "cumulative_success_count": cumulative_success,
                "cumulative_query_hit_rate": (
                    cumulative_success / query_count if query_count else 0.0
                ),
                "cumulative_available_doc_recall": (
                    cumulative_success / available_query_count if available_query_count else 0.0
                ),
                "remaining_query_failure_count": query_count - cumulative_success,
                "remaining_available_failure_count": available_query_count - cumulative_success,
                "query_count": query_count,
                "available_query_count": available_query_count,
                "routed_query_count": int(summary.get("routed_query_count", 0)),
                "total_routing_messages": int(summary.get("total_routing_messages", 0)),
                "mean_routing_messages_per_query": float(
                    summary.get("mean_routing_messages_per_query", 0.0)
                ),
                "mean_routing_messages_per_routed_query": float(
                    summary.get("mean_routing_messages_per_routed_query", 0.0)
                ),
            }
        )
    return rows


def evaluate_centroid_chain_document_rows(
    query_rows: Iterable[Tuple[UserId, object]],
    user_train_docs: Mapping[UserId, Set[object]],
    doc_train_owners: Mapping[object, Set[UserId]],
    doc_embedding_by_id: Mapping[object, np.ndarray],
    centroid_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]],
    random_by_user: Optional[Mapping[UserId, Iterable[UserId]]],
    centroid_embeddings_by_user: Mapping[UserId, np.ndarray],
    max_hops: int = 50,
    hop_cutoffs: Sequence[int] = (1, 2, 5, 10, 20, 50),
) -> Dict[str, object]:
    counts: Dict[str, object] = {
        "query_count": 0,
        "available_query_count": 0,
        "routed_query_count": 0,
        "unavailable_target_count": 0,
        "success_count": 0,
        "available_success_count": 0,
        "success_hop_counts": Counter(),
        "failure_hop_counts": Counter(),
        "termination_hop_counts": Counter(),
        "stop_reason_counts": Counter(),
        "total_routing_messages": 0,
    }
    for user, doc_id in query_rows:
        counts["query_count"] += 1
        owners = doc_train_owners.get(doc_id, set())
        is_available = any(owner != user for owner in owners)
        if not is_available:
            counts["unavailable_target_count"] += 1
            counts["stop_reason_counts"]["unavailable_target"] += 1
            continue
        counts["available_query_count"] += 1
        query_embedding = doc_embedding_by_id.get(doc_id)
        if query_embedding is None:
            counts["stop_reason_counts"]["missing_query_embedding"] += 1
            continue
        result = route_centroid_chain_query(
            query_embedding=query_embedding,
            source_user=user,
            target_doc=doc_id,
            user_train_docs=user_train_docs,
            centroid_top_by_user=centroid_top_by_user,
            random_by_user=random_by_user,
            centroid_embeddings_by_user=centroid_embeddings_by_user,
            max_hops=max_hops,
        )
        counts["routed_query_count"] += 1
        counts["total_routing_messages"] += int(result.hops)
        counts["termination_hop_counts"][result.hops] += 1
        counts["stop_reason_counts"][result.stop_reason] += 1
        if result.found:
            counts["success_count"] += 1
            counts["available_success_count"] += 1
            counts["success_hop_counts"][result.hops] += 1
        else:
            counts["failure_hop_counts"][result.hops] += 1
    return _finalize_centroid_chain_counts(counts, hop_cutoffs)


_INLINE_CHAIN_EVAL_CONTEXT: Optional[Dict[str, object]] = None


def _evaluate_inline_chain_chunk(
    query_rows: Sequence[Tuple[UserId, object]],
) -> Dict[str, object]:
    if _INLINE_CHAIN_EVAL_CONTEXT is None:
        raise RuntimeError("Inline chain evaluation worker context is not initialized.")
    return evaluate_centroid_chain_document_rows(
        query_rows=query_rows,
        **_INLINE_CHAIN_EVAL_CONTEXT,
    )


def evaluate_centroid_chain_document_retrieval(
    user_docs_train: pd.DataFrame,
    user_docs_test: pd.DataFrame,
    doc_embeddings: Optional[pd.DataFrame],
    centroid_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]],
    random_by_user: Optional[Mapping[UserId, Iterable[UserId]]],
    centroid_embeddings_by_user: Mapping[UserId, np.ndarray],
    max_hops: int = 50,
    hop_cutoffs: Sequence[int] = (1, 2, 5, 10, 20, 50),
    limit_queries: Optional[int] = None,
    workers: int = 1,
) -> Dict[str, object]:
    user_train_docs, doc_train_owners = _train_doc_maps(user_docs_train)
    doc_embedding_by_id = _doc_embedding_map(doc_embeddings, user_docs_test)
    query_frame = user_docs_test[["AnonID", "DocIndex"]]
    if limit_queries is not None:
        query_frame = query_frame.head(int(limit_queries))
    query_rows = list(query_frame.itertuples(index=False, name=None))
    context_values = {
        "user_train_docs": user_train_docs,
        "doc_train_owners": doc_train_owners,
        "doc_embedding_by_id": doc_embedding_by_id,
        "centroid_top_by_user": centroid_top_by_user,
        "random_by_user": random_by_user,
        "centroid_embeddings_by_user": centroid_embeddings_by_user,
        "max_hops": max_hops,
        "hop_cutoffs": hop_cutoffs,
    }
    workers = max(1, min(int(workers), len(query_rows) or 1))
    if workers == 1:
        return evaluate_centroid_chain_document_rows(
            query_rows=query_rows,
            **context_values,
        )

    try:
        context = mp.get_context("fork")
    except ValueError:
        warnings.warn("Fork multiprocessing is unavailable; using one chain-evaluation worker.")
        return evaluate_centroid_chain_document_rows(
            query_rows=query_rows,
            **context_values,
        )

    global _INLINE_CHAIN_EVAL_CONTEXT
    _INLINE_CHAIN_EVAL_CONTEXT = context_values
    chunk_size = int(math.ceil(len(query_rows) / workers))
    chunks = [
        query_rows[start : start + chunk_size]
        for start in range(0, len(query_rows), chunk_size)
    ]
    with context.Pool(processes=workers) as pool:
        summaries = pool.map(_evaluate_inline_chain_chunk, chunks, chunksize=1)
    _INLINE_CHAIN_EVAL_CONTEXT = None
    return combine_centroid_chain_summaries(summaries, hop_cutoffs=hop_cutoffs)


def flatten_centroid_chain_summary(summary: Mapping[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {}
    for key, value in summary.items():
        if isinstance(value, Mapping):
            for subkey, subvalue in value.items():
                row[f"{key}.{subkey}"] = subvalue
        else:
            row[key] = value
    return row


def _rank_centroid_candidates(
    owner: UserId,
    centroid_index: int,
    candidates: Iterable[UserId],
    centroid_embeddings_by_user: Mapping[UserId, np.ndarray],
    cap: int,
) -> TopList:
    if cap <= 0:
        return []
    owner_centroids = centroid_embeddings_by_user.get(owner)
    if owner_centroids is None or centroid_index >= len(owner_centroids):
        return []
    source_embedding = owner_centroids[centroid_index]
    ranked = []
    seen = set()
    for candidate in candidates:
        if candidate in seen or candidate == owner:
            continue
        seen.add(candidate)
        score = _centroid_candidate_score(source_embedding, candidate, centroid_embeddings_by_user)
        if np.isfinite(score):
            ranked.append((candidate, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:cap]


def _centroid_known_from_top(
    centroid_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]],
    users: Optional[Iterable[UserId]] = None,
) -> CentroidKnownByUser:
    owner_ids = list(users) if users is not None else list(centroid_top_by_user.keys())
    compact: CentroidKnownByUser = {}
    for user in owner_ids:
        compact[user] = {
            int(index): {peer for peer, _score in values}
            for index, values in centroid_top_by_user.get(user, {}).items()
        }
    return compact


def _project_centroid_top_to_peer_top(
    centroid_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]],
    similarity: SimilarityIndex,
    n: int,
) -> TopByUser:
    top_by_user: TopByUser = {}
    for user in similarity.users:
        candidates = []
        for values in centroid_top_by_user.get(user, {}).values():
            candidates.extend(peer for peer, _score in values)
        top_by_user[user] = similarity.rank_candidates(user, candidates, n)
    return top_by_user


def _project_centroid_known_to_peer_known(
    centroid_known_by_user: Mapping[UserId, Mapping[int, Iterable[UserId]]],
    random_by_user: Mapping[UserId, Iterable[UserId]],
    users: Iterable[UserId],
) -> KnownByUser:
    known_by_user: KnownByUser = {}
    for user in users:
        known = set()
        for values in centroid_known_by_user.get(user, {}).values():
            known.update(values)
        known.update(random_by_user.get(user, set()))
        known.discard(user)
        known_by_user[user] = known
    return known_by_user


def _refresh_peer_view_for_user(
    user: UserId,
    top_by_user: TopByUser,
    known_by_user: KnownByUser,
    centroid_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]],
    centroid_known_by_user: Mapping[UserId, Mapping[int, Iterable[UserId]]],
    random_by_user: Mapping[UserId, Iterable[UserId]],
    similarity: SimilarityIndex,
    n: int,
) -> None:
    candidates = []
    for values in centroid_top_by_user.get(user, {}).values():
        candidates.extend(peer for peer, _score in values)
    top_by_user[user] = similarity.rank_candidates(user, candidates, n)
    compact_centroid_known = {
        int(index): {peer for peer, _score in values}
        for index, values in centroid_top_by_user.get(user, {}).items()
    }
    centroid_known_by_user[user] = compact_centroid_known
    known = set().union(*compact_centroid_known.values()) if compact_centroid_known else set()
    known.update(random_by_user.get(user, set()))
    known.discard(user)
    known_by_user[user] = known


def _centroid_top_peer_set(
    centroid_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]],
    user: UserId,
) -> Set[UserId]:
    peers = set()
    for values in centroid_top_by_user.get(user, {}).values():
        peers.update(peer for peer, _score in values)
    return peers


def _centroid_referrer_candidate_order(
    source_embedding: np.ndarray,
    referrer: UserId,
    centroid_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]],
    centroid_embeddings_by_user: Mapping[UserId, np.ndarray],
    random_by_user: Mapping[UserId, Iterable[UserId]],
) -> List[UserId]:
    referrer_centroids = centroid_embeddings_by_user.get(referrer)
    if referrer_centroids is not None and len(referrer_centroids) > 0:
        closest_index = _nearest_centroid_index(source_embedding, referrer_centroids)
        centroid_order = [closest_index] + [
            index
            for index in sorted(centroid_top_by_user.get(referrer, {}).keys())
            if index != closest_index
        ]
    else:
        centroid_order = sorted(centroid_top_by_user.get(referrer, {}).keys())

    candidate_order: List[UserId] = []
    for centroid_index in centroid_order:
        candidate_order.extend(peer for peer, _score in centroid_top_by_user.get(referrer, {}).get(centroid_index, []))
    candidate_order.extend(sorted(random_by_user.get(referrer, set()), key=repr))
    return candidate_order


def _filter_centroid_response_candidates(
    source_user: UserId,
    source_embedding: np.ndarray,
    source_top_list: Sequence[Tuple[UserId, float]],
    source_cap: int,
    source_excluded_peers: Set[UserId],
    candidate_order: Sequence[UserId],
    centroid_embeddings_by_user: Mapping[UserId, np.ndarray],
    valid_users: Set[UserId],
) -> Tuple[List[Tuple[UserId, float]], int]:
    if source_cap <= 0:
        return [], 0
    threshold = -np.inf
    if len(source_top_list) >= source_cap and source_top_list:
        threshold = float(source_top_list[-1][1])

    response: List[Tuple[UserId, float]] = []
    seen: Set[UserId] = set()
    scored_candidates = 0
    for candidate in candidate_order:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate == source_user or candidate not in valid_users or candidate in source_excluded_peers:
            continue
        score = _centroid_candidate_score(source_embedding, candidate, centroid_embeddings_by_user)
        scored_candidates += 1
        if not np.isfinite(score):
            continue
        if len(source_top_list) < source_cap or score > threshold:
            response.append((candidate, score))

    response.sort(key=lambda item: item[1], reverse=True)
    return response, scored_candidates


def _build_centroid_referrer_response(
    source_user: UserId,
    source_embedding: np.ndarray,
    source_top_list: Sequence[Tuple[UserId, float]],
    source_cap: int,
    source_excluded_peers: Set[UserId],
    referrer: UserId,
    centroid_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]],
    centroid_embeddings_by_user: Mapping[UserId, np.ndarray],
    random_by_user: Mapping[UserId, Iterable[UserId]],
    valid_users: Set[UserId],
) -> Tuple[List[Tuple[UserId, float]], int]:
    candidate_order = _centroid_referrer_candidate_order(
        source_embedding=source_embedding,
        referrer=referrer,
        centroid_top_by_user=centroid_top_by_user,
        centroid_embeddings_by_user=centroid_embeddings_by_user,
        random_by_user=random_by_user,
    )
    return _filter_centroid_response_candidates(
        source_user=source_user,
        source_embedding=source_embedding,
        source_top_list=source_top_list,
        source_cap=source_cap,
        source_excluded_peers=source_excluded_peers,
        candidate_order=candidate_order,
        centroid_embeddings_by_user=centroid_embeddings_by_user,
        valid_users=valid_users,
    )


_CENTROID_RESPONSE_EMBEDDINGS_BY_USER: Optional[CentroidEmbeddingsByUser] = None
_CENTROID_RESPONSE_VALID_USERS: Optional[Set[UserId]] = None


def _init_centroid_response_worker(
    centroid_embeddings_by_user: CentroidEmbeddingsByUser,
    valid_users: Set[UserId],
) -> None:
    global _CENTROID_RESPONSE_EMBEDDINGS_BY_USER, _CENTROID_RESPONSE_VALID_USERS
    _CENTROID_RESPONSE_EMBEDDINGS_BY_USER = centroid_embeddings_by_user
    _CENTROID_RESPONSE_VALID_USERS = valid_users


def _compute_centroid_response_task(
    task: CentroidResponseTask,
    centroid_embeddings_by_user: Mapping[UserId, np.ndarray],
    valid_users: Set[UserId],
    referrer_top_by_user: Mapping[UserId, Mapping[int, Sequence[Tuple[UserId, float]]]],
    referrer_random_by_user: Mapping[UserId, Iterable[UserId]],
) -> CentroidResponseResult:
    task_index, user, centroid_index, source_top_list, source_cap, excluded, referrer = task
    user_centroids = centroid_embeddings_by_user.get(user)
    if user_centroids is None or centroid_index >= len(user_centroids):
        return task_index, user, centroid_index, [], 0
    source_embedding = user_centroids[centroid_index]
    candidate_order = _centroid_referrer_candidate_order(
        source_embedding=source_embedding,
        referrer=referrer,
        centroid_top_by_user=referrer_top_by_user,
        centroid_embeddings_by_user=centroid_embeddings_by_user,
        random_by_user=referrer_random_by_user,
    )
    response, scored_candidates = _filter_centroid_response_candidates(
        source_user=user,
        source_embedding=source_embedding,
        source_top_list=source_top_list,
        source_cap=source_cap,
        source_excluded_peers=excluded,
        candidate_order=candidate_order,
        centroid_embeddings_by_user=centroid_embeddings_by_user,
        valid_users=valid_users,
    )
    return task_index, user, centroid_index, response, scored_candidates


def _centroid_response_worker(
    chunk: CentroidResponseChunk,
) -> Tuple[int, List[CentroidResponseResult]]:
    if _CENTROID_RESPONSE_EMBEDDINGS_BY_USER is None or _CENTROID_RESPONSE_VALID_USERS is None:
        raise RuntimeError("Centroid response worker was not initialized.")
    return _compute_centroid_response_chunk(
        chunk,
        _CENTROID_RESPONSE_EMBEDDINGS_BY_USER,
        _CENTROID_RESPONSE_VALID_USERS,
    )


def _compute_centroid_response_chunk(
    chunk: CentroidResponseChunk,
    centroid_embeddings_by_user: Mapping[UserId, np.ndarray],
    valid_users: Set[UserId],
) -> Tuple[int, List[CentroidResponseResult]]:
    chunk_index, tasks, referrer_top_by_user, referrer_random_by_user = chunk
    return (
        chunk_index,
        [
            _compute_centroid_response_task(
                task,
                centroid_embeddings_by_user,
                valid_users,
                referrer_top_by_user,
                referrer_random_by_user,
            )
            for task in tasks
        ],
    )


def _mp_context_for_centroid_workers():
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context()


def exploration_slot_counts(n: int, fraction: float) -> Tuple[int, int]:
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("random_exploration_fraction must be between 0.0 and 1.0.")
    total_slots = max(0, int(n))
    random_slots = min(total_slots, int(math.floor(total_slots * fraction)))
    return total_slots - random_slots, random_slots


def build_protected_random_links(
    similarity: SimilarityIndex,
    random_slots: int,
    seed: int,
) -> RandomByUser:
    rng = np.random.default_rng(seed)
    random_by_user: RandomByUser = {}
    for user in similarity.users:
        choices = [candidate for candidate in similarity.users if candidate != user]
        sample_size = min(max(0, int(random_slots)), len(choices))
        if sample_size == 0:
            random_by_user[user] = set()
            continue
        sampled = rng.choice(np.array(choices, dtype=object), size=sample_size, replace=False).tolist()
        random_by_user[user] = set(sampled)
    return random_by_user


def combine_known_with_random(
    known_by_user: Mapping[UserId, Set[UserId]],
    random_by_user: Mapping[UserId, Iterable[UserId]],
    users: Iterable[UserId],
) -> KnownByUser:
    combined: KnownByUser = {}
    for user in users:
        known = set(known_by_user.get(user, set()))
        known.update(random_by_user.get(user, set()))
        known.discard(user)
        combined[user] = known
    return combined


def build_graph_from_candidates(
    initializer: str,
    initial_candidates: Mapping[UserId, Iterable[UserId]],
    similarity: SimilarityIndex,
    metric_evaluator: MetricEvaluator,
    n: int,
    runtime_seconds: float,
    random_by_user: Optional[Mapping[UserId, Iterable[UserId]]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> InitialGraph:
    known_by_user: KnownByUser = {}
    top_by_user: TopByUser = {}
    clean_random_by_user: RandomByUser = {}
    candidate_counts = []
    total_users = len(similarity.users)
    for idx, user in enumerate(similarity.users, start=1):
        protected_random = set(similarity.valid_candidates(user, (random_by_user or {}).get(user, [])))
        clean_random_by_user[user] = protected_random
        ranked_known = set(similarity.valid_candidates(user, initial_candidates.get(user, [])))
        ranked_known.difference_update(protected_random)
        combined_known = set(ranked_known)
        combined_known.update(protected_random)
        known_by_user[user] = combined_known
        candidate_counts.append(len(combined_known))
        top_by_user[user] = similarity.rank_candidates(user, ranked_known, n)
        if total_users >= 10000 and (idx % 10000 == 0 or idx == total_users):
            print(f"Phase progress: ranking initial candidates for {idx}/{total_users} peers.", flush=True)

    metrics = metric_evaluator.evaluate(top_by_user, clean_random_by_user)
    candidates_per_peer = float(np.mean(candidate_counts)) if candidate_counts else 0.0
    return InitialGraph(
        initializer=initializer,
        known_by_user=known_by_user,
        top_by_user=top_by_user,
        runtime_seconds=runtime_seconds,
        initial_candidates_per_peer=candidates_per_peer,
        candidate_ids_generated_per_peer=candidates_per_peer,
        initial_recall_at_n=metrics["closest_peer_recall_at_50"],
        initial_common_exact_topn=metrics["mean_common_exact_top50"],
        initial_document_retrieval_success=metrics["document_retrieval_success"],
        initial_centroid_routed_document_retrieval_success=metrics["centroid_routed_document_retrieval_success"],
        initial_query_hit_rate=metrics["query_hit_rate"],
        initial_macro_peer_hit_rate=metrics["macro_peer_hit_rate"],
        initial_query_hit_count=metrics["query_hit_count"],
        initial_evaluated_query_count=metrics["evaluated_query_count"],
        initial_primary_hit_metric_variant=metrics["primary_hit_metric_variant"],
        random_by_user=clean_random_by_user,
        metadata=dict(metadata or {}),
    )


def build_random_initial_graph(
    similarity: SimilarityIndex,
    metric_evaluator: MetricEvaluator,
    n: int,
    seed: int,
    random_by_user: Optional[Mapping[UserId, Iterable[UserId]]] = None,
) -> InitialGraph:
    start = time.perf_counter()
    rng = np.random.default_rng(seed + 1)
    initial_candidates = {}
    for user in similarity.users:
        protected_random = set((random_by_user or {}).get(user, []))
        choices = [candidate for candidate in similarity.users if candidate != user and candidate not in protected_random]
        sample_size = min(n, len(choices))
        if sample_size == 0:
            initial_candidates[user] = []
        else:
            initial_candidates[user] = rng.choice(np.array(choices, dtype=object), size=sample_size, replace=False).tolist()
    return build_graph_from_candidates(
        initializer="random",
        initial_candidates=initial_candidates,
        similarity=similarity,
        metric_evaluator=metric_evaluator,
        n=n,
        runtime_seconds=time.perf_counter() - start,
        random_by_user=random_by_user,
    )


def build_tree_initial_graph(
    df_avgemb: pd.DataFrame,
    user_docs_train: pd.DataFrame,
    user_docs_test: pd.DataFrame,
    cosine_sims: pd.DataFrame,
    similarity: SimilarityIndex,
    metric_evaluator: MetricEvaluator,
    n: int,
    m: int,
    record_limit_per_leafnode: int,
    tree_delta: float,
    random_by_user: Optional[Mapping[UserId, Iterable[UserId]]] = None,
) -> InitialGraph:
    from Construct_tree_verified_version import construct_tree

    start = time.perf_counter()
    tree_cosine_sims = pd.DataFrame(
        np.array(cosine_sims.loc[similarity.users, similarity.users].to_numpy(dtype=float), copy=True),
        index=similarity.users,
        columns=similarity.users,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        result = construct_tree(
            record_limit_per_leafnode=record_limit_per_leafnode,
            N=n,
            DELTA=tree_delta,
            df_avgemb=df_avgemb.copy(deep=True),
            user_docs_train=user_docs_train.copy(deep=True),
            user_docs_test=user_docs_test.copy(deep=True),
            cosine_sims=tree_cosine_sims,
            print_hists=False,
            M=m,
            perform_query_test=None,
            EXPANSION_ROUNDS=0,
        )
    tree_anon_closest_users_df = result[5]
    initial_candidates = {}
    for user in similarity.users:
        if user not in tree_anon_closest_users_df.index:
            initial_candidates[user] = []
            continue
        all_shared = tree_anon_closest_users_df.loc[user, "All_Shared"]
        if isinstance(all_shared, float) and np.isnan(all_shared):
            all_shared = []
        initial_candidates[user] = list(all_shared)

    return build_graph_from_candidates(
        initializer="tree",
        initial_candidates=initial_candidates,
        similarity=similarity,
        metric_evaluator=metric_evaluator,
        n=n,
        runtime_seconds=time.perf_counter() - start,
        random_by_user=random_by_user,
    )


class _CentroidTreeNode:
    def __init__(self, node_id: int, depth: int, parent: Optional["_CentroidTreeNode"] = None):
        self.node_id = node_id
        self.depth = depth
        self.parent = parent


class _CentroidTreeSplitNode(_CentroidTreeNode):
    def __init__(
        self,
        node_id: int,
        depth: int,
        centroid_left: np.ndarray,
        centroid_right: np.ndarray,
        parent: Optional[_CentroidTreeNode] = None,
    ):
        super().__init__(node_id=node_id, depth=depth, parent=parent)
        self.centroid_left = centroid_left
        self.centroid_right = centroid_right
        self.left_child: Optional[_CentroidTreeNode] = None
        self.right_child: Optional[_CentroidTreeNode] = None

    def get_children(self) -> List[_CentroidTreeNode]:
        return [child for child in (self.left_child, self.right_child) if child is not None]


def _embedding_bucket_key(embedding: object) -> Tuple[Tuple[int, ...], str, bytes]:
    array = np.ascontiguousarray(np.asarray(embedding, dtype=float))
    return tuple(array.shape), array.dtype.str, array.tobytes()


class _CentroidTreeLeafNode(_CentroidTreeNode):
    def __init__(self, node_id: int, depth: int, parent: Optional[_CentroidTreeNode] = None):
        super().__init__(node_id=node_id, depth=depth, parent=parent)
        self.records: List[Dict[str, object]] = []
        self.buckets: Dict[Tuple[Tuple[int, ...], str, bytes], Dict[str, object]] = {}
        self.failed_split_size: Optional[int] = None

    @property
    def failed_split_distinct_count(self) -> Optional[int]:
        """Legacy diagnostic alias used by the original distinct-point tree tests."""
        return self.failed_split_size

    def add_record(self, record: Mapping[str, object]) -> None:
        record = dict(record)
        key = _embedding_bucket_key(record["embedding"])
        bucket = self.buckets.get(key)
        if bucket is None:
            bucket = {
                "embedding": np.asarray(record["embedding"], dtype=float).copy(),
                "weight": 0,
                "records": [],
            }
            self.buckets[key] = bucket
        bucket["weight"] = int(bucket["weight"]) + 1
        bucket["records"].append(record)
        self.records.append(record)

    @property
    def distinct_point_count(self) -> int:
        return len(self.buckets)

    def bucket_items(self) -> List[Dict[str, object]]:
        return list(self.buckets.values())

    def capacity_size(self, capacity_metric: str) -> int:
        if capacity_metric == "entries":
            return len(self.records)
        if capacity_metric == "distinct_points":
            return self.distinct_point_count
        raise ValueError(f"Unsupported leaf capacity metric: {capacity_metric}")

    def should_attempt_split(
        self,
        record_limit_per_leafnode: int,
        capacity_metric: str = "distinct_points",
    ) -> bool:
        capacity_size = self.capacity_size(capacity_metric)
        return (
            capacity_size > record_limit_per_leafnode
            and self.failed_split_size != capacity_size
        )

    def mark_split_failed(self, capacity_metric: str = "distinct_points") -> None:
        self.failed_split_size = self.capacity_size(capacity_metric)


class _CentroidTree:
    def __init__(
        self,
        record_limit_per_leafnode: int,
        delta: float,
        seed: int,
        capacity_metric: str = "distinct_points",
        normalize_split_centroids: bool = False,
        eager_recursive_splitting: bool = True,
    ):
        self.record_limit_per_leafnode = max(1, int(record_limit_per_leafnode))
        self.delta = float(delta)
        self.seed = int(seed)
        if capacity_metric not in {"distinct_points", "entries"}:
            raise ValueError(f"Unsupported leaf capacity metric: {capacity_metric}")
        self.capacity_metric = capacity_metric
        self.normalize_split_centroids = bool(normalize_split_centroids)
        self.eager_recursive_splitting = bool(eager_recursive_splitting)
        self._next_node_id = 0
        self.nodes: Dict[int, _CentroidTreeNode] = {}
        self.root = self._new_leaf(depth=0, parent=None)
        self.maximum_attained_depth = 0

    def _new_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _register(self, node: _CentroidTreeNode) -> _CentroidTreeNode:
        self.nodes[node.node_id] = node
        return node

    def _new_leaf(self, depth: int, parent: Optional[_CentroidTreeNode]) -> _CentroidTreeLeafNode:
        return self._register(_CentroidTreeLeafNode(self._new_node_id(), depth=depth, parent=parent))

    def _new_split(
        self,
        depth: int,
        centroid_left: np.ndarray,
        centroid_right: np.ndarray,
        parent: Optional[_CentroidTreeNode],
    ) -> _CentroidTreeSplitNode:
        return self._register(
            _CentroidTreeSplitNode(
                self._new_node_id(),
                depth=depth,
                centroid_left=centroid_left,
                centroid_right=centroid_right,
                parent=parent,
            )
        )

    def position_record(self, record: Mapping[str, object]) -> None:
        self._insert_record_downstream(self.root, dict(record))

    def _insert_record_downstream(self, current_node: _CentroidTreeNode, record: Mapping[str, object]) -> None:
        while True:
            if isinstance(current_node, _CentroidTreeSplitNode):
                embedding = np.asarray(record["embedding"], dtype=float)
                dist_left = float(np.linalg.norm(embedding - current_node.centroid_left))
                dist_right = float(np.linalg.norm(embedding - current_node.centroid_right))
                diff = abs(dist_left - dist_right)
                if diff < self.delta:
                    self._insert_record_downstream(current_node.left_child, record)
                    self._insert_record_downstream(current_node.right_child, record)
                    return
                current_node = current_node.left_child if dist_left < dist_right else current_node.right_child
                continue

            if isinstance(current_node, _CentroidTreeLeafNode):
                current_node.add_record(record)
                if current_node.should_attempt_split(
                    self.record_limit_per_leafnode,
                    capacity_metric=self.capacity_metric,
                ):
                    self._replace_leaf_with_splitnode(current_node)
                return

            raise TypeError(f"Unsupported tree node: {type(current_node)!r}")

    def _replace_leaf_with_splitnode(self, leaf_node: _CentroidTreeLeafNode) -> None:
        if leaf_node.distinct_point_count < 2:
            leaf_node.mark_split_failed(self.capacity_metric)
            return
        parent_capacity_size = leaf_node.capacity_size(self.capacity_metric)
        split_node = self._turn_leaf_into_split(leaf_node)
        if split_node is None:
            return
        if leaf_node is self.root:
            self.root = split_node
        else:
            parent = leaf_node.parent
            if not isinstance(parent, _CentroidTreeSplitNode):
                return
            if parent.left_child is leaf_node:
                parent.left_child = split_node
            elif parent.right_child is leaf_node:
                parent.right_child = split_node
        self.nodes.pop(leaf_node.node_id, None)
        leaf_node.records.clear()
        leaf_node.buckets.clear()
        if self.eager_recursive_splitting:
            for child in list(split_node.get_children()):
                if isinstance(child, _CentroidTreeLeafNode):
                    child_capacity_size = child.capacity_size(self.capacity_metric)
                    if child_capacity_size >= parent_capacity_size:
                        child.mark_split_failed(self.capacity_metric)
                    elif child.should_attempt_split(
                        self.record_limit_per_leafnode,
                        capacity_metric=self.capacity_metric,
                    ):
                        self._replace_leaf_with_splitnode(child)

    def _turn_leaf_into_split(self, leaf_node: _CentroidTreeLeafNode) -> Optional[_CentroidTreeSplitNode]:
        buckets = leaf_node.bucket_items()
        embeddings = np.array([np.asarray(bucket["embedding"], dtype=float) for bucket in buckets])
        weights = np.array([int(bucket["weight"]) for bucket in buckets], dtype=float)
        if embeddings.shape[0] < 2:
            return None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Number of distinct clusters")
            model = KMeans(n_clusters=2, random_state=self.seed, n_init=10).fit(embeddings, sample_weight=weights)
        labels = model.labels_
        centroid_left = model.cluster_centers_[0]
        centroid_right = model.cluster_centers_[1]
        if self.normalize_split_centroids:
            centroid_left = _normalize_embedding(centroid_left)
            centroid_right = _normalize_embedding(centroid_right)
        if np.isnan(centroid_left).any() or np.isnan(centroid_right).any():
            leaf_node.mark_split_failed(self.capacity_metric)
            return None

        left_bucket_records: List[List[Dict[str, object]]] = []
        right_bucket_records: List[List[Dict[str, object]]] = []
        non_duplicated_bucket_count = 0
        for bucket, label in zip(buckets, labels):
            emb = np.asarray(bucket["embedding"], dtype=float)
            dist_left = float(np.linalg.norm(emb - centroid_left))
            dist_right = float(np.linalg.norm(emb - centroid_right))
            records = list(bucket["records"])
            route_left = (
                dist_left <= dist_right if self.normalize_split_centroids else label == 0
            )
            if abs(dist_left - dist_right) < self.delta:
                left_bucket_records.append(records)
                right_bucket_records.append(records)
            elif route_left:
                left_bucket_records.append(records)
                non_duplicated_bucket_count += 1
            else:
                right_bucket_records.append(records)
                non_duplicated_bucket_count += 1

        minimum_separated = int(math.ceil(leaf_node.distinct_point_count / 2.0))
        if (
            not left_bucket_records
            or not right_bucket_records
            or non_duplicated_bucket_count < minimum_separated
        ):
            leaf_node.mark_split_failed(self.capacity_metric)
            return None

        split_node = self._new_split(
            depth=leaf_node.depth,
            centroid_left=centroid_left,
            centroid_right=centroid_right,
            parent=leaf_node.parent,
        )
        left_leaf = self._new_leaf(depth=leaf_node.depth + 1, parent=split_node)
        right_leaf = self._new_leaf(depth=leaf_node.depth + 1, parent=split_node)
        split_node.left_child = left_leaf
        split_node.right_child = right_leaf

        for records in left_bucket_records:
            for rec in records:
                left_leaf.add_record(rec)
        for records in right_bucket_records:
            for rec in records:
                right_leaf.add_record(rec)

        self.maximum_attained_depth = max(self.maximum_attained_depth, left_leaf.depth, right_leaf.depth)
        return split_node

    def get_all_leaf_nodes(self) -> List[_CentroidTreeLeafNode]:
        leaf_nodes: List[_CentroidTreeLeafNode] = []
        queue: List[_CentroidTreeNode] = [self.root]
        while queue:
            node = queue.pop(0)
            if isinstance(node, _CentroidTreeLeafNode):
                leaf_nodes.append(node)
            elif isinstance(node, _CentroidTreeSplitNode):
                queue.extend(node.get_children())
        return leaf_nodes

    def nodes_at_distance(self, start_node: _CentroidTreeNode, distance: int) -> List[_CentroidTreeNode]:
        if distance == 0:
            return [start_node]
        visited = set()
        queue = [(start_node, 0)]
        result = []
        while queue:
            current, dist = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            neighbors = []
            if current.parent is not None:
                neighbors.append(current.parent)
            if isinstance(current, _CentroidTreeSplitNode):
                neighbors.extend(current.get_children())
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                if dist + 1 == distance:
                    result.append(neighbor)
                else:
                    queue.append((neighbor, dist + 1))
        return result


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=float)
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0 or not np.isfinite(norm):
        return embedding
    return embedding / norm


def _nearest_centroid_index(embedding: np.ndarray, centroids: np.ndarray) -> int:
    centroids = np.asarray(centroids, dtype=np.float32)
    if centroids.ndim == 1:
        return 0
    if centroids.shape[0] == 0:
        return 0
    embedding = _normalize_embedding(np.asarray(embedding, dtype=np.float32)).astype(np.float32, copy=False)
    sims = centroids @ embedding
    return int(np.argmax(sims))


def _embedding_matrix(values: Iterable[object]) -> np.ndarray:
    return np.vstack([np.asarray(value, dtype=float) for value in values])


def allocate_centroid_contact_caps(
    cluster_doc_counts: Sequence[int],
    m: int,
    min_quota: int = 3,
    weight_mode: str = "sqrt",
) -> List[int]:
    counts = [max(0, int(value)) for value in cluster_doc_counts]
    if not counts:
        return []
    total_budget = max(0, int(m))
    if len(counts) == 1:
        return [total_budget]
    min_quota = max(0, int(min_quota))
    if min_quota == 0:
        base = [0 for _ in counts]
    else:
        if total_budget < len(counts) * min_quota:
            raise ValueError("contact budget is too small for the requested centroid minimum quota")
        base = [min_quota for _ in counts]
    remaining = total_budget - sum(base)
    if remaining <= 0:
        return base
    if weight_mode == "sqrt":
        weights = np.sqrt(np.asarray(counts, dtype=float))
    elif weight_mode == "proportional":
        weights = np.asarray(counts, dtype=float)
    else:
        raise ValueError(f"Unsupported centroid contact weight mode: {weight_mode}")
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones(len(counts), dtype=float)
    raw = remaining * weights / float(np.sum(weights))
    extras = np.floor(raw).astype(int)
    remainder = remaining - int(np.sum(extras))
    fractions = raw - extras
    order = sorted(range(len(counts)), key=lambda idx: (-float(fractions[idx]), -counts[idx], idx))
    for idx in order[:remainder]:
        extras[idx] += 1
    return [int(base_value + extra) for base_value, extra in zip(base, extras)]


def _effective_k_with_min_quota(selected_k: int, doc_count: int, m: int, min_quota: int) -> int:
    selected_k = max(1, int(selected_k))
    doc_count = max(1, int(doc_count))
    m = max(0, int(m))
    min_quota = max(1, int(min_quota))
    max_by_budget = max(1, m // min_quota) if m >= min_quota else 1
    return max(1, min(selected_k, doc_count, max_by_budget))


def _kmeans_inertias(embeddings: np.ndarray, max_k: int, seed: int) -> List[float]:
    inertias = []
    for k in range(1, max_k + 1):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Number of distinct clusters")
            model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(embeddings)
        inertias.append(float(model.inertia_))
    return inertias


def _select_k_by_knee(inertias: Sequence[float]) -> int:
    if len(inertias) < 3:
        return 1
    y = np.asarray(inertias, dtype=float)
    if not np.all(np.isfinite(y)) or float(np.max(y) - np.min(y)) <= 1e-12:
        return 1
    x = np.linspace(0.0, 1.0, num=len(y))
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    start = np.array([x[0], y_norm[0]])
    end = np.array([x[-1], y_norm[-1]])
    line = end - start
    line_norm = np.linalg.norm(line)
    if line_norm <= 1e-12:
        return 1
    distances = []
    for x_value, y_value in zip(x, y_norm):
        point = np.array([x_value, y_value])
        distances.append(abs(np.cross(line, start - point)) / line_norm)
    best_index = int(np.argmax(distances))
    if distances[best_index] <= 0.05:
        return 1
    return best_index + 1


def _select_k_by_relative_drop(inertias: Sequence[float], threshold: float) -> int:
    if len(inertias) < 2:
        return 1
    previous = float(inertias[0])
    for index in range(1, len(inertias)):
        current = float(inertias[index])
        if abs(previous) <= 1e-12:
            return max(1, index)
        improvement = (previous - current) / abs(previous)
        if improvement < threshold:
            return max(1, index)
        previous = current
    return len(inertias)


def _select_k_by_silhouette(embeddings: np.ndarray, max_k: int, seed: int) -> int:
    if embeddings.shape[0] < 3 or max_k < 2:
        return 1
    best_k = 1
    best_score = -np.inf
    for k in range(2, max_k + 1):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Number of distinct clusters")
            labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embeddings)
        unique_labels = np.unique(labels)
        if unique_labels.shape[0] < 2 or unique_labels.shape[0] >= embeddings.shape[0]:
            continue
        score = float(silhouette_score(embeddings, labels, metric="cosine"))
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def select_peer_centroid_count(
    embeddings: np.ndarray,
    mode: str = "fixed",
    fixed_k: int = 2,
    max_k: int = 8,
    elbow_method: str = "knee",
    relative_drop_threshold: float = 0.10,
    seed: int = 42,
) -> int:
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return 1
    capped_max_k = max(1, min(int(max_k), embeddings.shape[0]))
    if mode == "fixed":
        return max(1, min(int(fixed_k), capped_max_k))
    if mode not in {"elbow", "auto"}:
        raise ValueError(f"Unsupported centroid mode: {mode}")
    if capped_max_k < 2:
        return 1
    if elbow_method == "silhouette":
        return _select_k_by_silhouette(embeddings, capped_max_k, seed)

    inertias = _kmeans_inertias(embeddings, capped_max_k, seed)
    if elbow_method == "knee":
        return _select_k_by_knee(inertias)
    if elbow_method == "relative_drop":
        return _select_k_by_relative_drop(inertias, relative_drop_threshold)
    raise ValueError(f"Unsupported elbow method: {elbow_method}")


def _fit_peer_centroids(embeddings: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k <= 1:
        return np.array([np.mean(embeddings, axis=0)])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Number of distinct clusters")
        model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(embeddings)
    return model.cluster_centers_


def _fit_peer_centroids_with_counts(embeddings: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, List[int]]:
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Cannot fit centroids for an empty embedding matrix.")
    if k <= 1:
        return np.array([np.mean(embeddings, axis=0)]), [int(embeddings.shape[0])]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Number of distinct clusters")
        model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(embeddings)
    counts = np.bincount(model.labels_, minlength=k)
    non_empty = [idx for idx, count in enumerate(counts) if int(count) > 0]
    if not non_empty:
        return np.array([np.mean(embeddings, axis=0)]), [int(embeddings.shape[0])]
    centers = model.cluster_centers_[non_empty]
    non_empty_counts = [int(counts[idx]) for idx in non_empty]
    order = sorted(range(len(non_empty_counts)), key=lambda idx: (-non_empty_counts[idx], idx))
    return centers[order], [non_empty_counts[idx] for idx in order]


def _json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(_json_safe(key)): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return _json_safe(value)


def _training_doc_embeddings(
    user_docs_train: pd.DataFrame,
    doc_embeddings: Optional[pd.DataFrame],
    users: Sequence[UserId],
) -> pd.DataFrame:
    user_docs_train = user_docs_train[user_docs_train["AnonID"].isin(users)].copy()
    if "embedding" in user_docs_train.columns:
        docs = user_docs_train[["AnonID", "DocIndex", "embedding"]].copy()
    else:
        if doc_embeddings is None:
            raise ValueError(
                "Multi-centroid tree initialization needs document embeddings. "
                "Pass doc_embeddings or include an embedding column in user_docs_train."
            )
        if "DocIndex" not in doc_embeddings.columns or "embedding" not in doc_embeddings.columns:
            raise ValueError("doc_embeddings must contain DocIndex and embedding columns.")
        docs = user_docs_train.merge(doc_embeddings[["DocIndex", "embedding"]], on="DocIndex", how="left")
    docs = docs.dropna(subset=["embedding"]).copy()
    return docs


def _candidate_owners_for_centroid(
    tree: _CentroidTree,
    start_leaf_nodes: Sequence[_CentroidTreeLeafNode],
    owner: UserId,
    contact_cap: int,
    valid_owners: Set[UserId],
    seen_owners: Optional[Set[UserId]] = None,
) -> List[UserId]:
    if contact_cap <= 0:
        return []
    candidates = []
    seen = set(seen_owners or set())
    seen.add(owner)
    max_distance = max(0, tree.maximum_attained_depth * 4)
    for distance in range(max_distance + 1):
        for leaf in start_leaf_nodes:
            for node in tree.nodes_at_distance(leaf, distance):
                if not isinstance(node, _CentroidTreeLeafNode):
                    continue
                for record in node.records:
                    candidate_owner = record["OwnerID"]
                    if candidate_owner in seen or candidate_owner not in valid_owners:
                        continue
                    seen.add(candidate_owner)
                    if seen_owners is not None:
                        seen_owners.add(candidate_owner)
                    candidates.append(candidate_owner)
                    if len(candidates) >= contact_cap:
                        return candidates
    return candidates


def _order_centroid_records_for_tree(
    records: Sequence[Mapping[str, object]],
    tree_insertion_seed: Optional[int] = None,
) -> List[Mapping[str, object]]:
    if tree_insertion_seed is None:
        return list(records)
    records_by_owner: Dict[UserId, List[Mapping[str, object]]] = defaultdict(list)
    owner_order: List[UserId] = []
    for record in records:
        owner = record["OwnerID"]
        if owner not in records_by_owner:
            owner_order.append(owner)
        records_by_owner[owner].append(record)
    rng = random.Random(int(tree_insertion_seed))
    rng.shuffle(owner_order)
    ordered: List[Mapping[str, object]] = []
    for owner in owner_order:
        ordered.extend(sorted(records_by_owner[owner], key=lambda item: int(item["CentroidIndex"])))
    return ordered


def _load_multi_centroid_cache(path: str) -> Tuple[CentroidEmbeddingsByUser, CentroidIntListsByUser, Mapping[str, object]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    embeddings = payload.get("centroid_embeddings_by_user")
    counts = payload.get("centroid_cluster_doc_counts_by_user")
    if not embeddings or not counts:
        raise ValueError(f"Centroid cache is missing embeddings or cluster counts: {path}")
    normalized_embeddings: CentroidEmbeddingsByUser = {}
    normalized_counts: CentroidIntListsByUser = {}
    for user, centers in embeddings.items():
        centers_array = np.asarray(centers, dtype=np.float32)
        if centers_array.ndim == 1:
            centers_array = centers_array.reshape(1, -1)
        normalized_embeddings[user] = np.vstack(
            [_normalize_embedding(center).astype(np.float32, copy=False) for center in centers_array]
        ).astype(np.float32, copy=False)
    for user, values in counts.items():
        normalized_counts[user] = [int(value) for value in values]
    return normalized_embeddings, normalized_counts, payload


def build_multi_centroid_tree_initial_graph(
    df_avgemb: pd.DataFrame,
    user_docs_train: pd.DataFrame,
    doc_embeddings: Optional[pd.DataFrame],
    similarity: SimilarityIndex,
    metric_evaluator: MetricEvaluator,
    n: int,
    m: int,
    record_limit_per_leafnode: int,
    tree_delta: float,
    seed: int,
    mode: str,
    fixed_k: int = 2,
    elbow_max_k: int = 8,
    elbow_method: str = "knee",
    relative_drop_threshold: float = 0.10,
    min_contact_quota: int = 3,
    random_by_user: Optional[Mapping[UserId, Iterable[UserId]]] = None,
    tree_insertion_seed: Optional[int] = None,
    centroid_cache_path: Optional[str] = None,
    topology: str = "tree",
) -> InitialGraph:
    if topology not in {"tree", "random"}:
        raise ValueError(f"Unsupported multi-centroid topology: {topology}")
    start = time.perf_counter()
    print(
        f"Phase start: multi-centroid {topology} initialization "
        f"mode={mode} fixed_k={fixed_k} elbow_method={elbow_method}.",
        flush=True,
    )
    cached_centroids_by_user: Optional[CentroidEmbeddingsByUser] = None
    cached_counts_by_user: Optional[CentroidIntListsByUser] = None
    centroid_cache_metadata: Mapping[str, object] = {}
    if centroid_cache_path:
        print(f"Phase start: loading centroid cache {centroid_cache_path}.", flush=True)
        cached_centroids_by_user, cached_counts_by_user, centroid_cache_metadata = _load_multi_centroid_cache(
            centroid_cache_path
        )
        print(
            f"Phase complete: loaded centroid cache for {len(cached_centroids_by_user)} peers.",
            flush=True,
        )
        docs_by_user = {}
    else:
        print("Phase start: joining training docs to embeddings.", flush=True)
        docs = _training_doc_embeddings(user_docs_train, doc_embeddings, similarity.users)
        print(f"Phase complete: joined {docs.shape[0]} training docs to embeddings.", flush=True)
        docs_by_user = {user: group for user, group in docs.groupby("AnonID", sort=False)}
    avg_by_user = {
        row.AnonID: np.asarray(row.embedding, dtype=float)
        for row in df_avgemb[df_avgemb["AnonID"].isin(similarity.users)].itertuples(index=False)
    }

    records = []
    selected_k_rows = []
    selected_k_by_owner: Dict[UserId, int] = {}
    centroid_embeddings_by_user: CentroidEmbeddingsByUser = {}
    centroid_cluster_doc_counts_by_user: CentroidIntListsByUser = {}
    centroid_contact_caps_by_user: CentroidIntListsByUser = {}
    effective_m = max(0, int(m))
    min_contact_quota = max(1, int(min_contact_quota))
    total_users = len(similarity.users)
    print(f"Phase start: fitting/selecting centroids for {total_users} peers.", flush=True)
    for idx, user in enumerate(similarity.users, start=1):
        if cached_centroids_by_user is not None and cached_counts_by_user is not None:
            if user not in cached_centroids_by_user or user not in cached_counts_by_user:
                raise ValueError(f"Centroid cache {centroid_cache_path} is missing user {user!r}.")
            normalized_centers = np.asarray(cached_centroids_by_user[user], dtype=np.float32)
            cluster_doc_counts = [int(value) for value in cached_counts_by_user[user]]
            if normalized_centers.shape[0] != len(cluster_doc_counts):
                raise ValueError(
                    f"Centroid cache {centroid_cache_path} has mismatched center/count lengths for user {user!r}."
                )
            doc_count = int(sum(cluster_doc_counts))
            selected_k = len(cluster_doc_counts)
        else:
            group = docs_by_user.get(user)
            if group is not None and not group.empty:
                embeddings = _embedding_matrix(group["embedding"].tolist())
                doc_count = embeddings.shape[0]
            elif user in avg_by_user:
                embeddings = np.array([avg_by_user[user]])
                doc_count = 1
            else:
                continue

            if mode == "fixed":
                selected_k = select_peer_centroid_count(
                    embeddings,
                    mode="fixed",
                    fixed_k=fixed_k,
                    max_k=max(1, min(doc_count, int(fixed_k))),
                    seed=seed,
                )
            else:
                selected_k = select_peer_centroid_count(
                    embeddings,
                    mode="elbow",
                    max_k=max(1, min(doc_count, int(elbow_max_k))),
                    elbow_method=elbow_method,
                    relative_drop_threshold=relative_drop_threshold,
                    seed=seed,
                )
            selected_k = _effective_k_with_min_quota(selected_k, doc_count, effective_m, min_contact_quota)
            centers, cluster_doc_counts = _fit_peer_centroids_with_counts(embeddings, selected_k, seed)
            selected_k = len(cluster_doc_counts)
            normalized_centers = np.vstack(
                [_normalize_embedding(center).astype(np.float32, copy=False) for center in centers]
            ).astype(np.float32, copy=False)

        selected_k_by_owner[user] = selected_k
        contact_caps = allocate_centroid_contact_caps(cluster_doc_counts, effective_m, min_quota=min_contact_quota)
        centroid_embeddings_by_user[user] = normalized_centers
        centroid_cluster_doc_counts_by_user[user] = [int(value) for value in cluster_doc_counts]
        centroid_contact_caps_by_user[user] = [int(value) for value in contact_caps]
        selected_k_rows.append(
            {
                "owner": _json_safe(user),
                "selected_k": selected_k,
                "doc_count": doc_count,
                "cluster_doc_counts": [int(value) for value in cluster_doc_counts],
                "contact_caps": [int(value) for value in contact_caps],
            }
        )

        for centroid_index in range(normalized_centers.shape[0]):
            records.append(
                {
                    "CloneID": f"{_json_safe(user)}::c{centroid_index}",
                    "OwnerID": user,
                    "CentroidIndex": centroid_index,
                    "SelectedK": selected_k,
                    "ClusterDocCount": int(cluster_doc_counts[centroid_index]),
                    "ContactCap": int(contact_caps[centroid_index]),
                    "embedding": normalized_centers[centroid_index],
                }
            )
        if total_users >= 10000 and (idx % 5000 == 0 or idx == total_users):
            print(
                f"Phase progress: centroid fitting/selection for {idx}/{total_users} peers "
                f"({len(records)} centroid clones so far).",
                flush=True,
            )

    initial_candidates: Dict[UserId, Set[UserId]] = {user: set() for user in similarity.users}
    centroid_candidates_by_user: Dict[UserId, Dict[int, List[UserId]]] = {
        user: {} for user in similarity.users
    }
    centroid_metadata = []
    if topology == "tree":
        insertion_records = _order_centroid_records_for_tree(records, tree_insertion_seed)
        if tree_insertion_seed is not None:
            print(
                f"Phase start: inserting {len(insertion_records)} centroid clones into centroid tree "
                f"with peer-order seed={tree_insertion_seed}.",
                flush=True,
            )
        else:
            print(f"Phase start: inserting {len(insertion_records)} centroid clones into centroid tree.", flush=True)
        tree = _CentroidTree(record_limit_per_leafnode=record_limit_per_leafnode, delta=tree_delta, seed=seed)
        for idx, record in enumerate(insertion_records, start=1):
            tree.position_record(record)
            if len(insertion_records) >= 10000 and (idx % 10000 == 0 or idx == len(insertion_records)):
                print(f"Phase progress: inserted {idx}/{len(insertion_records)} centroid clones into tree.", flush=True)

        print("Phase start: mapping centroid clones to leaf nodes.", flush=True)
        clone_to_leaf_nodes: Dict[str, List[_CentroidTreeLeafNode]] = defaultdict(list)
        for leaf in tree.get_all_leaf_nodes():
            for record in leaf.records:
                clone_to_leaf_nodes[str(record["CloneID"])].append(leaf)

        seen_candidates_by_owner: Dict[UserId, Set[UserId]] = {
            user: set(random_by_user.get(user, set()) if random_by_user is not None else set())
            for user in similarity.users
        }
        print("Phase start: collecting owner candidates from centroid leaf neighborhoods.", flush=True)
        for idx, record in enumerate(records, start=1):
            clone_id = str(record["CloneID"])
            owner = record["OwnerID"]
            centroid_index = int(record["CentroidIndex"])
            leaf_nodes = clone_to_leaf_nodes.get(clone_id, [])
            contact_cap = int(record["ContactCap"])
            candidate_owners = _candidate_owners_for_centroid(
                tree=tree,
                start_leaf_nodes=leaf_nodes,
                owner=owner,
                contact_cap=contact_cap,
                valid_owners=similarity.user_set,
                seen_owners=seen_candidates_by_owner[owner],
            )
            initial_candidates[owner].update(candidate_owners)
            centroid_candidates_by_user[owner][centroid_index] = candidate_owners
            centroid_metadata.append(
                {
                    "centroid_id": clone_id,
                    "owner": _json_safe(owner),
                    "centroid_index": centroid_index,
                    "selected_k": int(record["SelectedK"]),
                    "cluster_doc_count": int(record["ClusterDocCount"]),
                    "leaf_node_ids": [leaf.node_id for leaf in leaf_nodes],
                    "contact_cap": contact_cap,
                    "candidate_owners": [_json_safe(candidate) for candidate in candidate_owners],
                }
            )
            if len(records) >= 10000 and (idx % 10000 == 0 or idx == len(records)):
                print(f"Phase progress: collected candidates for {idx}/{len(records)} centroid clones.", flush=True)
    else:
        print("Phase start: assigning uniformly random owner candidates to peer centroids.", flush=True)
        rng = np.random.default_rng(seed + 17)
        all_users = list(similarity.users)
        records_by_owner: Dict[UserId, List[Mapping[str, object]]] = defaultdict(list)
        for record in records:
            records_by_owner[record["OwnerID"]].append(record)
        for idx, owner in enumerate(similarity.users, start=1):
            protected = set(random_by_user.get(owner, set()) if random_by_user is not None else set())
            available = [candidate for candidate in all_users if candidate != owner and candidate not in protected]
            owner_records = sorted(records_by_owner.get(owner, []), key=lambda item: int(item["CentroidIndex"]))
            requested = sum(int(record["ContactCap"]) for record in owner_records)
            sample_size = min(requested, len(available))
            sampled = (
                rng.choice(np.asarray(available, dtype=object), size=sample_size, replace=False).tolist()
                if sample_size
                else []
            )
            offset = 0
            for record in owner_records:
                centroid_index = int(record["CentroidIndex"])
                contact_cap = int(record["ContactCap"])
                candidate_owners = sampled[offset : offset + contact_cap]
                offset += contact_cap
                initial_candidates[owner].update(candidate_owners)
                centroid_candidates_by_user[owner][centroid_index] = candidate_owners
                centroid_metadata.append(
                    {
                        "centroid_id": str(record["CloneID"]),
                        "owner": _json_safe(owner),
                        "centroid_index": centroid_index,
                        "selected_k": int(record["SelectedK"]),
                        "cluster_doc_count": int(record["ClusterDocCount"]),
                        "leaf_node_ids": [],
                        "contact_cap": contact_cap,
                        "candidate_owners": [_json_safe(candidate) for candidate in candidate_owners],
                    }
                )
            if total_users >= 10000 and (idx % 10000 == 0 or idx == total_users):
                print(f"Phase progress: assigned random candidates for {idx}/{total_users} peers.", flush=True)

    centroid_top_by_user: CentroidTopByUser = {user: {} for user in similarity.users}
    centroid_known_by_user: CentroidKnownByUser = {user: {} for user in similarity.users}
    for user in similarity.users:
        for centroid_index, candidates in centroid_candidates_by_user.get(user, {}).items():
            cap = int(centroid_contact_caps_by_user.get(user, [])[centroid_index])
            ranked = _rank_centroid_candidates(
                owner=user,
                centroid_index=centroid_index,
                candidates=candidates,
                centroid_embeddings_by_user=centroid_embeddings_by_user,
                cap=cap,
            )
            centroid_top_by_user[user][centroid_index] = ranked
            centroid_known_by_user[user][centroid_index] = {peer for peer, _score in ranked}

    selected_k_values = list(selected_k_by_owner.values())
    initializer = f"{topology}_{'fixed_k' if mode == 'fixed' else 'elbow_k'}"
    metadata = {
        "initializer": initializer,
        "topology": topology,
        "mode": mode,
        "fixed_k": int(fixed_k),
        "elbow_max_k": int(elbow_max_k),
        "elbow_method": elbow_method,
        "relative_drop_threshold": float(relative_drop_threshold),
        "record_limit_per_leafnode": int(record_limit_per_leafnode),
        "tree_delta": float(tree_delta),
        "tree_insertion_seed": int(tree_insertion_seed) if tree_insertion_seed is not None else None,
        "centroid_cache_path": centroid_cache_path,
        "centroid_cache_key": centroid_cache_metadata.get("cache_key", "") if centroid_cache_metadata else "",
        "m": int(effective_m),
        "min_contact_quota": int(min_contact_quota),
        "selected_k_by_peer": selected_k_rows,
        "centroids": centroid_metadata,
        "summary": {
            "num_peers": len(selected_k_values),
            "num_centroids": len(records),
            "mean_selected_k": float(np.mean(selected_k_values)) if selected_k_values else 0.0,
            "min_selected_k": int(min(selected_k_values)) if selected_k_values else 0,
            "max_selected_k": int(max(selected_k_values)) if selected_k_values else 0,
        },
    }

    print("Phase start: collapsing centroid candidates to peer-level initial graph.", flush=True)
    initial_graph = build_graph_from_candidates(
        initializer=initializer,
        initial_candidates=initial_candidates,
        similarity=similarity,
        metric_evaluator=metric_evaluator,
        n=n,
        runtime_seconds=time.perf_counter() - start,
        random_by_user=random_by_user,
        metadata=metadata,
    )
    initial_graph.centroid_top_by_user = centroid_top_by_user
    initial_graph.centroid_known_by_user = centroid_known_by_user
    initial_graph.centroid_embeddings_by_user = centroid_embeddings_by_user
    initial_graph.centroid_contact_caps_by_user = centroid_contact_caps_by_user
    initial_graph.centroid_cluster_doc_counts_by_user = centroid_cluster_doc_counts_by_user
    initial_graph.top_by_user = _project_centroid_top_to_peer_top(centroid_top_by_user, similarity, n)
    initial_graph.known_by_user = _project_centroid_known_to_peer_known(
        centroid_known_by_user, initial_graph.random_by_user, similarity.users
    )
    routed_metrics = metric_evaluator.evaluate(
        initial_graph.top_by_user,
        initial_graph.random_by_user,
        centroid_top_by_user=initial_graph.centroid_top_by_user,
        centroid_embeddings_by_user=initial_graph.centroid_embeddings_by_user,
    )
    initial_graph.initial_recall_at_n = routed_metrics["closest_peer_recall_at_50"]
    initial_graph.initial_common_exact_topn = routed_metrics["mean_common_exact_top50"]
    initial_graph.initial_document_retrieval_success = routed_metrics["document_retrieval_success"]
    initial_graph.initial_centroid_routed_document_retrieval_success = routed_metrics[
        "centroid_routed_document_retrieval_success"
    ]
    initial_graph.initial_query_hit_rate = routed_metrics["query_hit_rate"]
    initial_graph.initial_macro_peer_hit_rate = routed_metrics["macro_peer_hit_rate"]
    initial_graph.initial_query_hit_count = routed_metrics["query_hit_count"]
    initial_graph.initial_evaluated_query_count = routed_metrics["evaluated_query_count"]
    initial_graph.initial_primary_hit_metric_variant = routed_metrics["primary_hit_metric_variant"]
    return initial_graph


def _accept_semantica_candidate(
    user: UserId,
    candidate: UserId,
    sim: float,
    top_by_user: TopByUser,
    known_by_user: KnownByUser,
    n: int,
    protected_random: Optional[Set[UserId]] = None,
) -> bool:
    if candidate == user:
        return False
    if protected_random is not None and candidate in protected_random:
        known_by_user[user].add(candidate)
        return False
    if n <= 0:
        return False
    top_list = top_by_user.setdefault(user, [])
    if any(peer == candidate for peer, _ in top_list):
        known_by_user[user].add(candidate)
        return False
    if len(top_list) < n:
        top_list.append((candidate, sim))
        top_list.sort(key=lambda item: item[1], reverse=True)
        known_by_user[user].add(candidate)
        return True
    if sim > top_list[-1][1]:
        top_list.append((candidate, sim))
        top_list.sort(key=lambda item: item[1], reverse=True)
        del top_list[n:]
        known_by_user[user].add(candidate)
        return True
    return False


def run_semantica_expansion(
    setup_name: str,
    initializer: str,
    initial_graph: InitialGraph,
    similarity: SimilarityIndex,
    metric_evaluator: MetricEvaluator,
    n: int,
    caps: BudgetCaps,
    checkpoints_by_axis: Mapping[str, Sequence[float]],
    seed: int,
    max_rounds: int,
    checkpoint_include_document_metrics: bool = True,
) -> AlgorithmResult:
    start = time.perf_counter()
    rng = random.Random(seed)
    known_by_user = copy_known(initial_graph.known_by_user)
    top_by_user = copy_top(initial_graph.top_by_user)
    random_by_user = copy_random(initial_graph.random_by_user)
    asked_by_user: Dict[UserId, Set[UserId]] = {user: set() for user in similarity.users}
    cost = BudgetCost()
    updates_so_far = 0
    rounds_completed = 0
    stop_reason = "max_rounds"
    converged = False
    recorder = BudgetTraceRecorder(
        setup_name=setup_name,
        initializer=initializer,
        refiner="semantica",
        num_peers=len(similarity.users),
        checkpoints_by_axis=checkpoints_by_axis,
        metric_evaluator=metric_evaluator,
        include_document_metrics=checkpoint_include_document_metrics,
    )

    for _round_idx in range(max_rounds):
        round_updates = 0
        active_peers = 0
        for user in similarity.users:
            ranked_referrers = [peer for peer, _ in top_by_user.get(user, [])]
            random_referrers = sorted(random_by_user.get(user, set()), key=repr)
            potential_referrers = [
                peer for peer in _ordered_unique(ranked_referrers + random_referrers)
                if peer not in asked_by_user[user]
            ]
            if not potential_referrers:
                continue
            active_peers += 1
            referrer = rng.choice(potential_referrers)
            asked_by_user[user].add(referrer)
            response = list(known_by_user.get(referrer, set()))
            cost.messages += 1
            cost.candidate_ids_transmitted += len(response)

            for candidate in response:
                if candidate == user or candidate in known_by_user[user] or candidate not in similarity.user_set:
                    continue
                cost.cosine_computations += 1
                sim = similarity.sim(user, candidate)
                if not np.isfinite(sim):
                    continue
                if _accept_semantica_candidate(
                    user,
                    candidate,
                    sim,
                    top_by_user,
                    known_by_user,
                    n,
                    protected_random=random_by_user.get(user, set()),
                ):
                    round_updates += 1
                    updates_so_far += 1

            runtime_seconds = time.perf_counter() - start
            state = RunState(
                top_by_user=top_by_user,
                active_peers=active_peers,
                updates_so_far=updates_so_far,
                converged=False,
                stop_reason="running",
                runtime_seconds=runtime_seconds,
                rounds_completed=rounds_completed,
                random_by_user=random_by_user,
            )
            recorder.maybe_record(cost, state)
            if caps.reached(cost, len(similarity.users)):
                stop_reason = "max_budget"
                return AlgorithmResult(
                    setup_name=setup_name,
                    initializer=initializer,
                    refiner="semantica",
                    top_by_user=top_by_user,
                    known_by_user=known_by_user,
                    trace_rows=recorder.rows,
                    final_cost=cost,
                    converged=False,
                    stop_reason=stop_reason,
                    runtime_seconds=runtime_seconds,
                    updates_so_far=updates_so_far,
                    rounds_completed=rounds_completed,
                    random_by_user=random_by_user,
                )

        rounds_completed = _round_idx + 1
        round_state = RunState(
            top_by_user=top_by_user,
            active_peers=active_peers,
            updates_so_far=updates_so_far,
            converged=False,
            stop_reason="running",
            runtime_seconds=time.perf_counter() - start,
            rounds_completed=rounds_completed,
            random_by_user=random_by_user,
        )
        recorder.maybe_record(cost, round_state)
        if active_peers == 0:
            stop_reason = "no_referrers_remaining"
            converged = True
            break
        if round_updates == 0:
            stop_reason = "no_topn_updates"
            converged = True
            break

    runtime_seconds = time.perf_counter() - start
    final_state = RunState(
        top_by_user=top_by_user,
        active_peers=0,
        updates_so_far=updates_so_far,
        converged=converged,
        stop_reason=stop_reason,
        runtime_seconds=runtime_seconds,
        rounds_completed=rounds_completed,
        random_by_user=random_by_user,
    )
    recorder.maybe_record(cost, final_state)
    if converged:
        recorder.carry_forward_converged(cost, final_state)
    return AlgorithmResult(
        setup_name=setup_name,
        initializer=initializer,
        refiner="semantica",
        top_by_user=top_by_user,
        known_by_user=known_by_user,
        trace_rows=recorder.rows,
        final_cost=cost,
        converged=converged,
        stop_reason=stop_reason,
        runtime_seconds=runtime_seconds,
        updates_so_far=updates_so_far,
        rounds_completed=rounds_completed,
        random_by_user=random_by_user,
    )


def _accept_centroid_candidate(
    user: UserId,
    centroid_index: int,
    candidate: UserId,
    sim: float,
    centroid_top_by_user: CentroidTopByUser,
    caps_by_user: Mapping[UserId, Sequence[int]],
    protected_random: Optional[Set[UserId]] = None,
) -> bool:
    if candidate == user or not np.isfinite(sim):
        return False
    if protected_random is not None and candidate in protected_random:
        return False
    cap_values = caps_by_user.get(user, [])
    cap = int(cap_values[centroid_index]) if centroid_index < len(cap_values) else 0
    if cap <= 0:
        return False
    top_list = centroid_top_by_user.setdefault(user, {}).setdefault(centroid_index, [])
    if any(peer == candidate for peer, _ in top_list):
        return False
    if len(top_list) < cap:
        top_list.append((candidate, sim))
        top_list.sort(key=lambda item: item[1], reverse=True)
        return True
    if sim > top_list[-1][1]:
        top_list.append((candidate, sim))
        top_list.sort(key=lambda item: item[1], reverse=True)
        del top_list[cap:]
        return True
    return False


def _run_centroid_semantica_expansion_batched(
    setup_name: str,
    initializer: str,
    similarity: SimilarityIndex,
    n: int,
    caps: BudgetCaps,
    seed: int,
    max_rounds: int,
    start: float,
    random_by_user: RandomByUser,
    centroid_top_by_user: CentroidTopByUser,
    centroid_known_by_user: CentroidKnownByUser,
    centroid_embeddings_by_user: CentroidEmbeddingsByUser,
    centroid_contact_caps_by_user: CentroidIntListsByUser,
    centroid_cluster_doc_counts_by_user: CentroidIntListsByUser,
    top_by_user: TopByUser,
    known_by_user: KnownByUser,
    asked_by_centroid: Dict[UserId, Dict[int, Set[UserId]]],
    recorder: BudgetTraceRecorder,
    batch_size: int,
    workers: int,
    worker_peer_batch_size: int,
) -> AlgorithmResult:
    rng = random.Random(seed)
    cost = BudgetCost()
    updates_so_far = 0
    rounds_completed = 0
    stop_reason = "max_rounds"
    converged = False
    users = list(similarity.users)
    batch_size = max(1, int(batch_size))
    workers = max(1, int(workers))
    worker_peer_batch_size = max(1, int(worker_peer_batch_size))

    def result_for_current_state(converged_value: bool, stop_reason_value: str, active_peers: int) -> AlgorithmResult:
        runtime_seconds = time.perf_counter() - start
        final_state = RunState(
            top_by_user=top_by_user,
            active_peers=active_peers,
            updates_so_far=updates_so_far,
            converged=converged_value,
            stop_reason=stop_reason_value,
            runtime_seconds=runtime_seconds,
            rounds_completed=rounds_completed,
            random_by_user=random_by_user,
            centroid_top_by_user=centroid_top_by_user,
            centroid_embeddings_by_user=centroid_embeddings_by_user,
        )
        recorder.maybe_record(cost, final_state)
        if converged_value:
            recorder.carry_forward_converged(cost, final_state)
        return AlgorithmResult(
            setup_name=setup_name,
            initializer=initializer,
            refiner="semantica_centroid",
            top_by_user=top_by_user,
            known_by_user=known_by_user,
            trace_rows=recorder.rows,
            final_cost=cost,
            converged=converged_value,
            stop_reason=stop_reason_value,
            runtime_seconds=runtime_seconds,
            updates_so_far=updates_so_far,
            rounds_completed=rounds_completed,
            random_by_user=random_by_user,
            centroid_top_by_user=centroid_top_by_user,
            centroid_known_by_user=centroid_known_by_user,
            centroid_embeddings_by_user=centroid_embeddings_by_user,
            centroid_contact_caps_by_user=centroid_contact_caps_by_user,
            centroid_cluster_doc_counts_by_user=centroid_cluster_doc_counts_by_user,
        )

    pool = None
    if workers > 1:
        context = _mp_context_for_centroid_workers()
        pool = context.Pool(
            processes=workers,
            initializer=_init_centroid_response_worker,
            initargs=(centroid_embeddings_by_user, similarity.user_set),
        )

    try:
        for _round_idx in range(max_rounds):
            round_updates = 0
            round_active_centroids = 0
            for batch_start in range(0, len(users), batch_size):
                batch_users = users[batch_start : batch_start + batch_size]
                chunks: List[CentroidResponseChunk] = []
                task_count = 0
                for chunk_start in range(0, len(batch_users), worker_peer_batch_size):
                    chunk_users = batch_users[chunk_start : chunk_start + worker_peer_batch_size]
                    chunk_tasks: List[CentroidResponseTask] = []
                    chunk_referrers: Set[UserId] = set()
                    for user in chunk_users:
                        user_centroids = centroid_embeddings_by_user.get(user)
                        if user_centroids is None or len(user_centroids) == 0:
                            continue
                        for centroid_index in range(len(user_centroids)):
                            ranked_referrers = [
                                peer for peer, _score in centroid_top_by_user.get(user, {}).get(centroid_index, [])
                            ]
                            random_referrers = sorted(random_by_user.get(user, set()), key=repr)
                            asked = asked_by_centroid.setdefault(user, {}).setdefault(centroid_index, set())
                            potential_referrers = [
                                peer for peer in _ordered_unique(ranked_referrers + random_referrers)
                                if peer not in asked
                            ]
                            if not potential_referrers:
                                continue
                            round_active_centroids += 1
                            referrer = rng.choice(potential_referrers)
                            asked.add(referrer)

                            source_top_list = list(centroid_top_by_user.get(user, {}).get(centroid_index, []))
                            cap_values = centroid_contact_caps_by_user.get(user, [])
                            source_cap = int(cap_values[centroid_index]) if centroid_index < len(cap_values) else 0
                            excluded_peers = _centroid_top_peer_set(centroid_top_by_user, user)
                            excluded_peers.update(random_by_user.get(user, set()))
                            chunk_referrers.add(referrer)
                            chunk_tasks.append(
                                (
                                    task_count,
                                    user,
                                    centroid_index,
                                    source_top_list,
                                    source_cap,
                                    excluded_peers,
                                    referrer,
                                )
                            )
                            task_count += 1
                    if not chunk_tasks:
                        continue
                    referrer_top_snapshot: CentroidTopByUser = {
                        referrer: {
                            int(index): list(values)
                            for index, values in centroid_top_by_user.get(referrer, {}).items()
                        }
                        for referrer in chunk_referrers
                    }
                    referrer_random_snapshot: RandomByUser = {
                        referrer: set(random_by_user.get(referrer, set()))
                        for referrer in chunk_referrers
                    }
                    chunks.append(
                        (
                            len(chunks),
                            chunk_tasks,
                            referrer_top_snapshot,
                            referrer_random_snapshot,
                        )
                    )

                if not chunks:
                    continue

                cost.messages += task_count
                if pool is not None:
                    chunk_results = pool.map(_centroid_response_worker, chunks, chunksize=1)
                else:
                    chunk_results = [
                        _compute_centroid_response_chunk(chunk, centroid_embeddings_by_user, similarity.user_set)
                        for chunk in chunks
                    ]
                results: List[CentroidResponseResult] = []
                for _chunk_index, chunk_items in sorted(chunk_results, key=lambda item: item[0]):
                    results.extend(chunk_items)
                results.sort(key=lambda item: item[0])

                changed_users: Set[UserId] = set()
                accepted_peer_sets: Dict[UserId, Set[UserId]] = {}
                batch_updates = 0
                for _task_index, user, centroid_index, response, scored_candidates in results:
                    cost.candidate_ids_transmitted += len(response)
                    cost.cosine_computations += scored_candidates
                    protected_random = random_by_user.get(user, set())
                    accepted_peer_sets.setdefault(user, _centroid_top_peer_set(centroid_top_by_user, user))
                    for candidate, sim in response:
                        if candidate in accepted_peer_sets[user] or candidate in protected_random:
                            continue
                        if _accept_centroid_candidate(
                            user=user,
                            centroid_index=centroid_index,
                            candidate=candidate,
                            sim=sim,
                            centroid_top_by_user=centroid_top_by_user,
                            caps_by_user=centroid_contact_caps_by_user,
                            protected_random=protected_random,
                        ):
                            batch_updates += 1
                            updates_so_far += 1
                            changed_users.add(user)
                            accepted_peer_sets[user] = _centroid_top_peer_set(centroid_top_by_user, user)

                round_updates += batch_updates
                for user in batch_users:
                    if user not in changed_users:
                        continue
                    _refresh_peer_view_for_user(
                        user=user,
                        top_by_user=top_by_user,
                        known_by_user=known_by_user,
                        centroid_top_by_user=centroid_top_by_user,
                        centroid_known_by_user=centroid_known_by_user,
                        random_by_user=random_by_user,
                        similarity=similarity,
                        n=n,
                    )

                runtime_seconds = time.perf_counter() - start
                state = RunState(
                    top_by_user=top_by_user,
                    active_peers=round_active_centroids,
                    updates_so_far=updates_so_far,
                    converged=False,
                    stop_reason="running",
                    runtime_seconds=runtime_seconds,
                    rounds_completed=rounds_completed,
                    random_by_user=random_by_user,
                    centroid_top_by_user=centroid_top_by_user,
                    centroid_embeddings_by_user=centroid_embeddings_by_user,
                )
                recorder.maybe_record(cost, state)
                if caps.reached(cost, len(similarity.users)):
                    stop_reason = "max_budget"
                    return result_for_current_state(False, stop_reason, round_active_centroids)
                if len(users) >= 10000:
                    processed = min(len(users), batch_start + len(batch_users))
                    progress_stride = max(batch_size, batch_size * 5)
                    if processed == len(users) or processed % progress_stride == 0:
                        per_peer = cost.per_peer(len(users))
                        print(
                            "Phase progress: centroid Semantica "
                            f"round {_round_idx + 1}, processed {processed}/{len(users)} peers, "
                            f"messages_per_peer={per_peer['messages']:.3f}, "
                            f"updates={updates_so_far}, runtime={runtime_seconds:.1f}s.",
                            flush=True,
                        )

            rounds_completed = _round_idx + 1
            round_state = RunState(
                top_by_user=top_by_user,
                active_peers=round_active_centroids,
                updates_so_far=updates_so_far,
                converged=False,
                stop_reason="running",
                runtime_seconds=time.perf_counter() - start,
                rounds_completed=rounds_completed,
                random_by_user=random_by_user,
                centroid_top_by_user=centroid_top_by_user,
                centroid_embeddings_by_user=centroid_embeddings_by_user,
            )
            recorder.maybe_record(cost, round_state)
            if round_active_centroids == 0:
                stop_reason = "no_referrers_remaining"
                converged = True
                break
            if round_updates == 0:
                stop_reason = "no_topn_updates"
                converged = True
                break
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return result_for_current_state(converged, stop_reason, 0)


def _run_centroid_semantica_expansion_frontier_priority(
    setup_name: str,
    initializer: str,
    similarity: SimilarityIndex,
    n: int,
    caps: BudgetCaps,
    seed: int,
    max_rounds: int,
    start: float,
    random_by_user: RandomByUser,
    centroid_top_by_user: CentroidTopByUser,
    centroid_known_by_user: CentroidKnownByUser,
    centroid_embeddings_by_user: CentroidEmbeddingsByUser,
    centroid_contact_caps_by_user: CentroidIntListsByUser,
    centroid_cluster_doc_counts_by_user: CentroidIntListsByUser,
    top_by_user: TopByUser,
    known_by_user: KnownByUser,
    initial_centroid_neighbors: Mapping[UserId, Mapping[int, Set[UserId]]],
    asked_by_centroid: Dict[UserId, Dict[int, Set[UserId]]],
    recorder: BudgetTraceRecorder,
    batch_size: int,
    workers: int,
    worker_peer_batch_size: int,
) -> AlgorithmResult:
    rng = random.Random(seed)
    cost = BudgetCost()
    updates_so_far = 0
    rounds_completed = 0
    stop_reason = "max_rounds"
    converged = False
    users = list(similarity.users)
    batch_size = max(1, int(batch_size))
    workers = max(1, int(workers))
    worker_peer_batch_size = max(1, int(worker_peer_batch_size))
    asked_random_by_user: Dict[UserId, Set[UserId]] = {user: set() for user in users}
    cumulative_diagnostics: Counter = Counter()
    round_diagnostics: Counter = Counter()

    def bump(field_name: str, value: int = 1) -> None:
        round_diagnostics[field_name] += int(value)
        cumulative_diagnostics[field_name] += int(value)

    def diagnostic_values() -> Dict[str, object]:
        return expansion_diagnostic_snapshot(
            cumulative=cumulative_diagnostics,
            current_round=round_diagnostics,
            num_peers=len(users),
        )

    def result_for_current_state(
        converged_value: bool,
        stop_reason_value: str,
        active_requests: int,
    ) -> AlgorithmResult:
        runtime_seconds = time.perf_counter() - start
        diagnostics = diagnostic_values()
        final_state = RunState(
            top_by_user=top_by_user,
            active_peers=active_requests,
            updates_so_far=updates_so_far,
            converged=converged_value,
            stop_reason=stop_reason_value,
            runtime_seconds=runtime_seconds,
            rounds_completed=rounds_completed,
            random_by_user=random_by_user,
            centroid_top_by_user=centroid_top_by_user,
            centroid_embeddings_by_user=centroid_embeddings_by_user,
            expansion_diagnostics=diagnostics,
        )
        recorder.maybe_record(cost, final_state)
        if converged_value:
            recorder.carry_forward_converged(cost, final_state)
        return AlgorithmResult(
            setup_name=setup_name,
            initializer=initializer,
            refiner="semantica_centroid",
            top_by_user=top_by_user,
            known_by_user=known_by_user,
            trace_rows=recorder.rows,
            final_cost=cost,
            converged=converged_value,
            stop_reason=stop_reason_value,
            runtime_seconds=runtime_seconds,
            updates_so_far=updates_so_far,
            rounds_completed=rounds_completed,
            random_by_user=random_by_user,
            centroid_top_by_user=centroid_top_by_user,
            centroid_known_by_user=centroid_known_by_user,
            centroid_embeddings_by_user=centroid_embeddings_by_user,
            centroid_contact_caps_by_user=centroid_contact_caps_by_user,
            centroid_cluster_doc_counts_by_user=centroid_cluster_doc_counts_by_user,
            expansion_diagnostics=diagnostics,
        )

    pool = None
    if workers > 1:
        context = _mp_context_for_centroid_workers()
        pool = context.Pool(
            processes=workers,
            initializer=_init_centroid_response_worker,
            initargs=(centroid_embeddings_by_user, similarity.user_set),
        )

    try:
        for round_index in range(max_rounds):
            round_diagnostics = Counter()
            round_updates = 0
            round_active_requests = 0
            round_top_snapshot = copy_centroid_top(centroid_top_by_user)

            for batch_start in range(0, len(users), batch_size):
                batch_users = users[batch_start : batch_start + batch_size]
                chunks: List[CentroidResponseChunk] = []
                random_requests: List[Tuple[UserId, UserId, List[UserId]]] = []
                task_count = 0

                for chunk_start in range(0, len(batch_users), worker_peer_batch_size):
                    chunk_users = batch_users[chunk_start : chunk_start + worker_peer_batch_size]
                    chunk_tasks: List[CentroidResponseTask] = []
                    chunk_referrers: Set[UserId] = set()
                    for user in chunk_users:
                        user_centroids = centroid_embeddings_by_user.get(user)
                        if user_centroids is None or len(user_centroids) == 0:
                            continue
                        for centroid_index in range(len(user_centroids)):
                            ranked_referrers = [
                                peer
                                for peer, _score in round_top_snapshot.get(user, {}).get(centroid_index, [])
                            ]
                            asked = asked_by_centroid.setdefault(user, {}).setdefault(centroid_index, set())
                            selected = select_centroid_referrer(
                                ranked_referrers=ranked_referrers,
                                protected_random_referrers=(),
                                initial_referrers=set(
                                    initial_centroid_neighbors.get(user, {}).get(centroid_index, set())
                                ),
                                asked_referrers=asked,
                                rng=rng,
                                policy="new_first_plus_random_channel",
                            )
                            if selected is None:
                                continue
                            referrer, tier = selected
                            asked.add(referrer)
                            round_active_requests += 1
                            bump(
                                "semantic_new_requests"
                                if tier == "new"
                                else "semantic_initial_requests"
                            )

                            source_top_list = list(
                                round_top_snapshot.get(user, {}).get(centroid_index, [])
                            )
                            cap_values = centroid_contact_caps_by_user.get(user, [])
                            source_cap = (
                                int(cap_values[centroid_index])
                                if centroid_index < len(cap_values)
                                else 0
                            )
                            excluded_peers = _centroid_top_peer_set(round_top_snapshot, user)
                            excluded_peers.update(random_by_user.get(user, set()))
                            chunk_referrers.add(referrer)
                            chunk_tasks.append(
                                (
                                    task_count,
                                    user,
                                    centroid_index,
                                    source_top_list,
                                    source_cap,
                                    excluded_peers,
                                    referrer,
                                )
                            )
                            task_count += 1

                        unasked_random = [
                            peer
                            for peer in sorted(random_by_user.get(user, set()), key=repr)
                            if peer not in asked_random_by_user[user]
                        ]
                        if unasked_random:
                            random_referrer = rng.choice(unasked_random)
                            asked_random_by_user[user].add(random_referrer)
                            advertised = combined_centroid_neighbor_ids(
                                referrer=random_referrer,
                                centroid_top_by_user=round_top_snapshot,
                                random_by_user=random_by_user,
                            )
                            random_requests.append((user, random_referrer, advertised))
                            round_active_requests += 1
                            bump("random_channel_requests")

                    if chunk_tasks:
                        referrer_top_snapshot: CentroidTopByUser = {
                            referrer: {
                                int(index): list(values)
                                for index, values in round_top_snapshot.get(referrer, {}).items()
                            }
                            for referrer in chunk_referrers
                        }
                        referrer_random_snapshot: RandomByUser = {
                            referrer: set(random_by_user.get(referrer, set()))
                            for referrer in chunk_referrers
                        }
                        chunks.append(
                            (
                                len(chunks),
                                chunk_tasks,
                                referrer_top_snapshot,
                                referrer_random_snapshot,
                            )
                        )

                cost.messages += task_count + len(random_requests)
                if pool is not None and chunks:
                    chunk_results = pool.map(_centroid_response_worker, chunks, chunksize=1)
                else:
                    chunk_results = [
                        _compute_centroid_response_chunk(
                            chunk,
                            centroid_embeddings_by_user,
                            similarity.user_set,
                        )
                        for chunk in chunks
                    ]
                results: List[CentroidResponseResult] = []
                for _chunk_index, chunk_items in sorted(chunk_results, key=lambda item: item[0]):
                    results.extend(chunk_items)
                results.sort(key=lambda item: item[0])

                changed_users: Set[UserId] = set()
                accepted_peer_sets: Dict[UserId, Set[UserId]] = {}
                for _task_index, user, centroid_index, response, scored_candidates in results:
                    cost.candidate_ids_transmitted += len(response)
                    cost.cosine_computations += scored_candidates
                    bump("semantic_candidate_ids", len(response))
                    bump("semantic_cosine_checks", scored_candidates)
                    protected_random = random_by_user.get(user, set())
                    accepted_peer_sets.setdefault(
                        user,
                        _centroid_top_peer_set(centroid_top_by_user, user),
                    )
                    for candidate, sim in response:
                        if candidate in accepted_peer_sets[user] or candidate in protected_random:
                            continue
                        if _accept_centroid_candidate(
                            user=user,
                            centroid_index=centroid_index,
                            candidate=candidate,
                            sim=sim,
                            centroid_top_by_user=centroid_top_by_user,
                            caps_by_user=centroid_contact_caps_by_user,
                            protected_random=protected_random,
                        ):
                            round_updates += 1
                            updates_so_far += 1
                            bump("semantic_accepted_contacts")
                            changed_users.add(user)
                            accepted_peer_sets[user] = _centroid_top_peer_set(
                                centroid_top_by_user,
                                user,
                            )

                for user, _random_referrer, advertised in random_requests:
                    cost.candidate_ids_transmitted += len(advertised)
                    bump("random_channel_candidate_ids", len(advertised))
                    accepted, scored_candidates = apply_random_channel_candidates(
                        user=user,
                        candidate_ids=advertised,
                        centroid_top_by_user=centroid_top_by_user,
                        centroid_embeddings_by_user=centroid_embeddings_by_user,
                        centroid_contact_caps_by_user=centroid_contact_caps_by_user,
                        random_by_user=random_by_user,
                        valid_users=similarity.user_set,
                    )
                    cost.cosine_computations += scored_candidates
                    bump("random_channel_cosine_checks", scored_candidates)
                    bump("random_channel_accepted_contacts", accepted)
                    if accepted:
                        round_updates += accepted
                        updates_so_far += accepted
                        changed_users.add(user)

                for user in changed_users:
                    _refresh_peer_view_for_user(
                        user=user,
                        top_by_user=top_by_user,
                        known_by_user=known_by_user,
                        centroid_top_by_user=centroid_top_by_user,
                        centroid_known_by_user=centroid_known_by_user,
                        random_by_user=random_by_user,
                        similarity=similarity,
                        n=n,
                    )

                state = RunState(
                    top_by_user=top_by_user,
                    active_peers=round_active_requests,
                    updates_so_far=updates_so_far,
                    converged=False,
                    stop_reason="running",
                    runtime_seconds=time.perf_counter() - start,
                    rounds_completed=rounds_completed,
                    random_by_user=random_by_user,
                    centroid_top_by_user=centroid_top_by_user,
                    centroid_embeddings_by_user=centroid_embeddings_by_user,
                    expansion_diagnostics=diagnostic_values(),
                )
                recorder.maybe_record(cost, state)
                if caps.reached(cost, len(users)):
                    return result_for_current_state(False, "max_budget", round_active_requests)
                if len(users) >= 10000:
                    processed = min(len(users), batch_start + len(batch_users))
                    progress_stride = max(batch_size, batch_size * 5)
                    if processed == len(users) or processed % progress_stride == 0:
                        per_peer = cost.per_peer(len(users))
                        print(
                            "Phase progress: frontier-priority centroid Semantica "
                            f"round {round_index + 1}, processed {processed}/{len(users)} peers, "
                            f"messages_per_peer={per_peer['messages']:.3f}, "
                            f"updates={updates_so_far}, runtime={time.perf_counter() - start:.1f}s.",
                            flush=True,
                        )

            rounds_completed = round_index + 1
            round_state = RunState(
                top_by_user=top_by_user,
                active_peers=round_active_requests,
                updates_so_far=updates_so_far,
                converged=False,
                stop_reason="running",
                runtime_seconds=time.perf_counter() - start,
                rounds_completed=rounds_completed,
                random_by_user=random_by_user,
                centroid_top_by_user=centroid_top_by_user,
                centroid_embeddings_by_user=centroid_embeddings_by_user,
                expansion_diagnostics=diagnostic_values(),
            )
            recorder.maybe_record(cost, round_state)
            if round_active_requests == 0:
                stop_reason = "no_referrers_remaining"
                converged = True
                break
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return result_for_current_state(converged, stop_reason, 0)


def run_centroid_semantica_expansion(
    setup_name: str,
    initializer: str,
    initial_graph: InitialGraph,
    similarity: SimilarityIndex,
    metric_evaluator: MetricEvaluator,
    n: int,
    caps: BudgetCaps,
    checkpoints_by_axis: Mapping[str, Sequence[float]],
    seed: int,
    max_rounds: int,
    batch_size: int = 1,
    workers: int = 1,
    worker_peer_batch_size: Optional[int] = None,
    checkpoint_include_document_metrics: bool = True,
    referrer_policy: str = "legacy_mixed_referrers",
) -> AlgorithmResult:
    if referrer_policy not in CENTROID_SEMANTICA_REFERRER_POLICIES:
        raise ValueError(f"Unsupported centroid Semantica referrer policy: {referrer_policy}")
    start = time.perf_counter()
    rng = random.Random(seed)
    random_by_user = copy_random(initial_graph.random_by_user)
    centroid_top_by_user = copy_centroid_top(initial_graph.centroid_top_by_user)
    initial_centroid_neighbors = snapshot_centroid_neighbor_ids(centroid_top_by_user)
    centroid_known_by_user = _centroid_known_from_top(centroid_top_by_user, similarity.users)
    centroid_embeddings_by_user = initial_graph.centroid_embeddings_by_user
    centroid_contact_caps_by_user = {
        user: list(values) for user, values in initial_graph.centroid_contact_caps_by_user.items()
    }
    centroid_cluster_doc_counts_by_user = {
        user: list(values) for user, values in initial_graph.centroid_cluster_doc_counts_by_user.items()
    }
    top_by_user = _project_centroid_top_to_peer_top(centroid_top_by_user, similarity, n)
    known_by_user = _project_centroid_known_to_peer_known(centroid_known_by_user, random_by_user, similarity.users)
    asked_by_centroid: Dict[UserId, Dict[int, Set[UserId]]] = {
        user: {idx: set() for idx in centroid_top_by_user.get(user, {})}
        for user in similarity.users
    }
    cost = BudgetCost()
    updates_so_far = 0
    rounds_completed = 0
    stop_reason = "max_rounds"
    converged = False
    recorder = BudgetTraceRecorder(
        setup_name=setup_name,
        initializer=initializer,
        refiner="semantica_centroid",
        num_peers=len(similarity.users),
        checkpoints_by_axis=checkpoints_by_axis,
        metric_evaluator=metric_evaluator,
        include_document_metrics=checkpoint_include_document_metrics,
    )

    batch_size = max(1, int(batch_size))
    workers = max(1, int(workers))
    if worker_peer_batch_size is None or int(worker_peer_batch_size) <= 0:
        worker_peer_batch_size = max(1, int(math.ceil(batch_size / workers)))
    else:
        worker_peer_batch_size = max(1, int(worker_peer_batch_size))
    if referrer_policy == "new_first_plus_random_channel":
        return _run_centroid_semantica_expansion_frontier_priority(
            setup_name=setup_name,
            initializer=initializer,
            similarity=similarity,
            n=n,
            caps=caps,
            seed=seed,
            max_rounds=max_rounds,
            start=start,
            random_by_user=random_by_user,
            centroid_top_by_user=centroid_top_by_user,
            centroid_known_by_user=centroid_known_by_user,
            centroid_embeddings_by_user=centroid_embeddings_by_user,
            centroid_contact_caps_by_user=centroid_contact_caps_by_user,
            centroid_cluster_doc_counts_by_user=centroid_cluster_doc_counts_by_user,
            top_by_user=top_by_user,
            known_by_user=known_by_user,
            initial_centroid_neighbors=initial_centroid_neighbors,
            asked_by_centroid=asked_by_centroid,
            recorder=recorder,
            batch_size=batch_size,
            workers=workers,
            worker_peer_batch_size=worker_peer_batch_size,
        )
    if batch_size > 1 or workers > 1:
        return _run_centroid_semantica_expansion_batched(
            setup_name=setup_name,
            initializer=initializer,
            similarity=similarity,
            n=n,
            caps=caps,
            seed=seed,
            max_rounds=max_rounds,
            start=start,
            random_by_user=random_by_user,
            centroid_top_by_user=centroid_top_by_user,
            centroid_known_by_user=centroid_known_by_user,
            centroid_embeddings_by_user=centroid_embeddings_by_user,
            centroid_contact_caps_by_user=centroid_contact_caps_by_user,
            centroid_cluster_doc_counts_by_user=centroid_cluster_doc_counts_by_user,
            top_by_user=top_by_user,
            known_by_user=known_by_user,
            asked_by_centroid=asked_by_centroid,
            recorder=recorder,
            batch_size=batch_size,
            workers=workers,
            worker_peer_batch_size=worker_peer_batch_size,
        )

    for _round_idx in range(max_rounds):
        round_updates = 0
        active_centroids = 0
        for user in similarity.users:
            user_centroids = centroid_embeddings_by_user.get(user)
            if user_centroids is None or len(user_centroids) == 0:
                continue
            for centroid_index in range(len(user_centroids)):
                ranked_referrers = [
                    peer for peer, _score in centroid_top_by_user.get(user, {}).get(centroid_index, [])
                ]
                random_referrers = sorted(random_by_user.get(user, set()), key=repr)
                asked = asked_by_centroid.setdefault(user, {}).setdefault(centroid_index, set())
                potential_referrers = [
                    peer for peer in _ordered_unique(ranked_referrers + random_referrers)
                    if peer not in asked
                ]
                if not potential_referrers:
                    continue
                active_centroids += 1
                referrer = rng.choice(potential_referrers)
                asked.add(referrer)

                cost.messages += 1
                existing_top_peers = _centroid_top_peer_set(centroid_top_by_user, user)
                excluded_peers = set(existing_top_peers)
                excluded_peers.update(random_by_user.get(user, set()))
                source_top_list = centroid_top_by_user.get(user, {}).get(centroid_index, [])
                cap_values = centroid_contact_caps_by_user.get(user, [])
                source_cap = int(cap_values[centroid_index]) if centroid_index < len(cap_values) else 0
                response, scored_candidates = _build_centroid_referrer_response(
                    source_user=user,
                    source_embedding=user_centroids[centroid_index],
                    source_top_list=source_top_list,
                    source_cap=source_cap,
                    source_excluded_peers=excluded_peers,
                    referrer=referrer,
                    centroid_top_by_user=centroid_top_by_user,
                    centroid_embeddings_by_user=centroid_embeddings_by_user,
                    random_by_user=random_by_user,
                    valid_users=similarity.user_set,
                )
                cost.candidate_ids_transmitted += len(response)
                cost.cosine_computations += scored_candidates

                for candidate, sim in response:
                    if _accept_centroid_candidate(
                        user=user,
                        centroid_index=centroid_index,
                        candidate=candidate,
                        sim=sim,
                        centroid_top_by_user=centroid_top_by_user,
                        caps_by_user=centroid_contact_caps_by_user,
                        protected_random=random_by_user.get(user, set()),
                    ):
                        existing_top_peers.add(candidate)
                        round_updates += 1
                        updates_so_far += 1

                _refresh_peer_view_for_user(
                    user=user,
                    top_by_user=top_by_user,
                    known_by_user=known_by_user,
                    centroid_top_by_user=centroid_top_by_user,
                    centroid_known_by_user=centroid_known_by_user,
                    random_by_user=random_by_user,
                    similarity=similarity,
                    n=n,
                )
                runtime_seconds = time.perf_counter() - start
                state = RunState(
                    top_by_user=top_by_user,
                    active_peers=active_centroids,
                    updates_so_far=updates_so_far,
                    converged=False,
                    stop_reason="running",
                    runtime_seconds=runtime_seconds,
                    rounds_completed=rounds_completed,
                    random_by_user=random_by_user,
                    centroid_top_by_user=centroid_top_by_user,
                    centroid_embeddings_by_user=centroid_embeddings_by_user,
                )
                recorder.maybe_record(cost, state)
                if caps.reached(cost, len(similarity.users)):
                    stop_reason = "max_budget"
                    return AlgorithmResult(
                        setup_name=setup_name,
                        initializer=initializer,
                        refiner="semantica_centroid",
                        top_by_user=top_by_user,
                        known_by_user=known_by_user,
                        trace_rows=recorder.rows,
                        final_cost=cost,
                        converged=False,
                        stop_reason=stop_reason,
                        runtime_seconds=runtime_seconds,
                        updates_so_far=updates_so_far,
                        rounds_completed=rounds_completed,
                        random_by_user=random_by_user,
                        centroid_top_by_user=centroid_top_by_user,
                        centroid_known_by_user=centroid_known_by_user,
                        centroid_embeddings_by_user=centroid_embeddings_by_user,
                        centroid_contact_caps_by_user=centroid_contact_caps_by_user,
                        centroid_cluster_doc_counts_by_user=centroid_cluster_doc_counts_by_user,
                    )

        rounds_completed = _round_idx + 1
        round_state = RunState(
            top_by_user=top_by_user,
            active_peers=active_centroids,
            updates_so_far=updates_so_far,
            converged=False,
            stop_reason="running",
            runtime_seconds=time.perf_counter() - start,
            rounds_completed=rounds_completed,
            random_by_user=random_by_user,
            centroid_top_by_user=centroid_top_by_user,
            centroid_embeddings_by_user=centroid_embeddings_by_user,
        )
        recorder.maybe_record(cost, round_state)
        if active_centroids == 0:
            stop_reason = "no_referrers_remaining"
            converged = True
            break
        if round_updates == 0:
            stop_reason = "no_topn_updates"
            converged = True
            break

    runtime_seconds = time.perf_counter() - start
    final_state = RunState(
        top_by_user=top_by_user,
        active_peers=0,
        updates_so_far=updates_so_far,
        converged=converged,
        stop_reason=stop_reason,
        runtime_seconds=runtime_seconds,
        rounds_completed=rounds_completed,
        random_by_user=random_by_user,
        centroid_top_by_user=centroid_top_by_user,
        centroid_embeddings_by_user=centroid_embeddings_by_user,
    )
    recorder.maybe_record(cost, final_state)
    if converged:
        recorder.carry_forward_converged(cost, final_state)
    return AlgorithmResult(
        setup_name=setup_name,
        initializer=initializer,
        refiner="semantica_centroid",
        top_by_user=top_by_user,
        known_by_user=known_by_user,
        trace_rows=recorder.rows,
        final_cost=cost,
        converged=converged,
        stop_reason=stop_reason,
        runtime_seconds=runtime_seconds,
        updates_so_far=updates_so_far,
        rounds_completed=rounds_completed,
        random_by_user=random_by_user,
        centroid_top_by_user=centroid_top_by_user,
        centroid_known_by_user=centroid_known_by_user,
        centroid_embeddings_by_user=centroid_embeddings_by_user,
        centroid_contact_caps_by_user=centroid_contact_caps_by_user,
        centroid_cluster_doc_counts_by_user=centroid_cluster_doc_counts_by_user,
    )


def _build_nnd_pool(initial_graph: InitialGraph, similarity: SimilarityIndex, n: int) -> Dict[UserId, Dict[UserId, NeighborEntry]]:
    pools: Dict[UserId, Dict[UserId, NeighborEntry]] = {}
    for user in similarity.users:
        pools[user] = {}
        for candidate, sim in list(initial_graph.top_by_user.get(user, []))[:n]:
            pools[user][candidate] = NeighborEntry(sim=sim, is_new=True)
    return pools


def _top_from_nnd_pool(pools: Mapping[UserId, Mapping[UserId, NeighborEntry]], n: int) -> TopByUser:
    top_by_user = {}
    for user, pool in pools.items():
        ranked = [(candidate, entry.sim) for candidate, entry in pool.items()]
        ranked.sort(key=lambda item: item[1], reverse=True)
        top_by_user[user] = ranked[:n]
    return top_by_user


def _known_from_nnd_pool(pools: Mapping[UserId, Mapping[UserId, NeighborEntry]]) -> KnownByUser:
    return {user: set(pool.keys()) for user, pool in pools.items()}


def _try_insert_nnd(
    pools: MutableMapping[UserId, MutableMapping[UserId, NeighborEntry]],
    user: UserId,
    candidate: UserId,
    sim: float,
    n: int,
    protected_random_by_user: Optional[Mapping[UserId, Set[UserId]]] = None,
) -> bool:
    if user == candidate or not np.isfinite(sim):
        return False
    if protected_random_by_user is not None and candidate in protected_random_by_user.get(user, set()):
        return False
    if n <= 0:
        return False
    pool = pools[user]
    existing = pool.get(candidate)
    if existing is not None:
        return False
    if len(pool) < n:
        pool[candidate] = NeighborEntry(sim=sim, is_new=True)
        return True
    worst_candidate, worst_entry = min(pool.items(), key=lambda item: item[1].sim)
    if sim > worst_entry.sim:
        del pool[worst_candidate]
        pool[candidate] = NeighborEntry(sim=sim, is_new=True)
        return True
    return False


def _ordered_unique(values: Iterable[UserId]) -> List[UserId]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def run_full_nndescent(
    setup_name: str,
    initializer: str,
    initial_graph: InitialGraph,
    similarity: SimilarityIndex,
    metric_evaluator: MetricEvaluator,
    n: int,
    caps: BudgetCaps,
    checkpoints_by_axis: Mapping[str, Sequence[float]],
    max_iterations: int,
    checkpoint_include_document_metrics: bool = True,
) -> AlgorithmResult:
    start = time.perf_counter()
    random_by_user = copy_random(initial_graph.random_by_user)
    pools = _build_nnd_pool(initial_graph, similarity, n)
    cost = BudgetCost()
    updates_so_far = 0
    converged = False
    stop_reason = "max_iterations"
    recorder = BudgetTraceRecorder(
        setup_name=setup_name,
        initializer=initializer,
        refiner="nndescent_full",
        num_peers=len(similarity.users),
        checkpoints_by_axis=checkpoints_by_axis,
        metric_evaluator=metric_evaluator,
        include_document_metrics=checkpoint_include_document_metrics,
    )

    for _iteration in range(max_iterations):
        old: Dict[UserId, List[UserId]] = {}
        new: Dict[UserId, List[UserId]] = {}
        reverse_old: Dict[UserId, List[UserId]] = {user: [] for user in similarity.users}
        reverse_new: Dict[UserId, List[UserId]] = {user: [] for user in similarity.users}

        for user in similarity.users:
            old[user] = []
            new[user] = []
            for candidate, entry in list(pools[user].items()):
                if entry.is_new:
                    new[user].append(candidate)
                    entry.is_new = False
                else:
                    old[user].append(candidate)

        for user in similarity.users:
            for candidate in new[user]:
                if candidate in reverse_new:
                    reverse_new[candidate].append(user)
            for candidate in old[user]:
                if candidate in reverse_old:
                    reverse_old[candidate].append(user)

        iteration_updates = 0
        active_peers = sum(1 for user in similarity.users if new[user] or reverse_new[user])

        for center in similarity.users:
            center_new = _ordered_unique(new[center] + reverse_new[center])
            center_old = _ordered_unique(old[center] + reverse_old[center])
            center_random = sorted(random_by_user.get(center, set()), key=repr)
            local_ids = _ordered_unique(center_new + center_old + center_random)
            if not local_ids:
                continue

            cost.messages += len(local_ids)
            for provider in local_ids:
                cost.candidate_ids_transmitted += len(pools.get(provider, {})) + len(random_by_user.get(provider, set()))

            compared_pairs: Set[Tuple[UserId, UserId]] = set()
            center_old_and_random = _ordered_unique(center_old + center_random)
            for idx, first in enumerate(center_new):
                for second in center_new[idx + 1 :]:
                    if first == second:
                        continue
                    pair = tuple(sorted((first, second), key=repr))
                    if pair in compared_pairs:
                        continue
                    compared_pairs.add(pair)
                    cost.cosine_computations += 1
                    sim = similarity.sim(first, second)
                    if _try_insert_nnd(pools, first, second, sim, n, protected_random_by_user=random_by_user):
                        iteration_updates += 1
                        updates_so_far += 1
                    if _try_insert_nnd(pools, second, first, sim, n, protected_random_by_user=random_by_user):
                        iteration_updates += 1
                        updates_so_far += 1

                for second in center_old_and_random:
                    if first == second:
                        continue
                    pair = tuple(sorted((first, second), key=repr))
                    if pair in compared_pairs:
                        continue
                    compared_pairs.add(pair)
                    cost.cosine_computations += 1
                    sim = similarity.sim(first, second)
                    if _try_insert_nnd(pools, first, second, sim, n, protected_random_by_user=random_by_user):
                        iteration_updates += 1
                        updates_so_far += 1
                    if _try_insert_nnd(pools, second, first, sim, n, protected_random_by_user=random_by_user):
                        iteration_updates += 1
                        updates_so_far += 1

            runtime_seconds = time.perf_counter() - start
            top_by_user = _top_from_nnd_pool(pools, n)
            state = RunState(
                top_by_user=top_by_user,
                active_peers=active_peers,
                updates_so_far=updates_so_far,
                converged=False,
                stop_reason="running",
                runtime_seconds=runtime_seconds,
                random_by_user=random_by_user,
            )
            recorder.maybe_record(cost, state)
            if caps.reached(cost, len(similarity.users)):
                stop_reason = "max_budget"
                return AlgorithmResult(
                    setup_name=setup_name,
                    initializer=initializer,
                    refiner="nndescent_full",
                    top_by_user=top_by_user,
                    known_by_user=combine_known_with_random(_known_from_nnd_pool(pools), random_by_user, similarity.users),
                    trace_rows=recorder.rows,
                    final_cost=cost,
                    converged=False,
                    stop_reason=stop_reason,
                    runtime_seconds=runtime_seconds,
                    updates_so_far=updates_so_far,
                    random_by_user=random_by_user,
                )

        if iteration_updates == 0:
            converged = True
            stop_reason = "no_topn_updates"
            break

    runtime_seconds = time.perf_counter() - start
    top_by_user = _top_from_nnd_pool(pools, n)
    final_state = RunState(
        top_by_user=top_by_user,
        active_peers=0,
        updates_so_far=updates_so_far,
        converged=converged,
        stop_reason=stop_reason,
        runtime_seconds=runtime_seconds,
        random_by_user=random_by_user,
    )
    recorder.maybe_record(cost, final_state)
    if converged:
        recorder.carry_forward_converged(cost, final_state)
    return AlgorithmResult(
        setup_name=setup_name,
        initializer=initializer,
        refiner="nndescent_full",
        top_by_user=top_by_user,
        known_by_user=combine_known_with_random(_known_from_nnd_pool(pools), random_by_user, similarity.users),
        trace_rows=recorder.rows,
        final_cost=cost,
        converged=converged,
        stop_reason=stop_reason,
        runtime_seconds=runtime_seconds,
        updates_so_far=updates_so_far,
        random_by_user=random_by_user,
    )


def initial_summary_rows(initial_graphs: Sequence[InitialGraph]) -> List[Dict[str, object]]:
    rows = []
    for graph in initial_graphs:
        rows.append(
            {
                "initializer": graph.initializer,
                "initial_candidates_per_peer": graph.initial_candidates_per_peer,
                "candidate_ids_generated_per_peer": graph.candidate_ids_generated_per_peer,
                "initial_recall_at_50": graph.initial_recall_at_n,
                "initial_common_exact_top50": graph.initial_common_exact_topn,
                "initial_document_retrieval_success": graph.initial_document_retrieval_success,
                "initial_centroid_routed_document_retrieval_success": graph.initial_centroid_routed_document_retrieval_success,
                "initial_query_hit_rate": graph.initial_query_hit_rate,
                "initial_macro_peer_hit_rate": graph.initial_macro_peer_hit_rate,
                "initial_query_hit_count": graph.initial_query_hit_count,
                "initial_evaluated_query_count": graph.initial_evaluated_query_count,
                "initial_primary_hit_metric_variant": graph.initial_primary_hit_metric_variant,
                "expansion_round": 0,
                "query_hop": 1,
                "measurement_step_type": "initial_topology",
                "query_hop_semantics": "direct_outgoing_peer_contact_after_expansion_checkpoint",
                "initialization_runtime_seconds": graph.runtime_seconds,
            }
        )
    return rows


def plot_budget_traces(trace_df: pd.DataFrame, output_dir: str) -> None:
    if trace_df.empty:
        return
    plot_specs = [
        ("messages_per_peer", "closest_peer_recall_at_50", "closest_peer_recall_at_50_vs_messages_per_peer.png"),
        (
            "candidate_ids_transmitted_per_peer",
            "closest_peer_recall_at_50",
            "closest_peer_recall_at_50_vs_candidate_ids_per_peer.png",
        ),
        (
            "cosine_computations_per_peer",
            "closest_peer_recall_at_50",
            "closest_peer_recall_at_50_vs_cosine_computations_per_peer.png",
        ),
        ("messages_per_peer", "document_retrieval_success", "document_retrieval_success_vs_messages_per_peer.png"),
        (
            "messages_per_peer",
            "centroid_routed_document_retrieval_success",
            "centroid_routed_document_retrieval_success_vs_messages_per_peer.png",
        ),
    ]
    os.makedirs(output_dir, exist_ok=True)
    for x_column, metric, filename in plot_specs:
        subset = trace_df[trace_df["checkpoint_axis"] == "messages"].copy()
        if subset.empty or metric not in subset.columns or subset[metric].dropna().empty:
            continue
        plt.figure(figsize=(9, 6))
        for setup_name, group in subset.groupby("setup_name"):
            group = group.sort_values(x_column)
            plt.plot(group[x_column], group[metric], marker="o", linewidth=1.8, label=setup_name)
        plt.xscale("log")
        plt.xlabel(x_column.replace("_", " "))
        plt.ylabel(metric.replace("_", " "))
        plt.title(metric.replace("_", " ") + " by " + x_column.replace("_", " "))
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=200)
        plt.close()


def save_final_graphs(results: Sequence[AlgorithmResult], output_path: str) -> None:
    payload = {}
    for result in results:
        payload[result.setup_name] = {
            "top_by_user": {user: list(values) for user, values in result.top_by_user.items()},
            "random_by_user": {user: sorted(values, key=repr) for user, values in result.random_by_user.items()},
            "known_by_user": {user: sorted(values, key=repr) for user, values in result.known_by_user.items()},
            "centroid_top_by_user": {
                user: {idx: list(values) for idx, values in by_centroid.items()}
                for user, by_centroid in result.centroid_top_by_user.items()
            },
            "centroid_contact_caps_by_user": {
                user: list(values) for user, values in result.centroid_contact_caps_by_user.items()
            },
            "centroid_cluster_doc_counts_by_user": {
                user: list(values) for user, values in result.centroid_cluster_doc_counts_by_user.items()
            },
            "final_cost": result.final_cost,
            "converged": result.converged,
            "stop_reason": result.stop_reason,
            "runtime_seconds": result.runtime_seconds,
            "updates_so_far": result.updates_so_far,
            "rounds_completed": result.rounds_completed,
        }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)


def filter_to_sample_users(
    df_avgemb: pd.DataFrame,
    user_docs_train: pd.DataFrame,
    user_docs_test: pd.DataFrame,
    cosine_sims: Optional[pd.DataFrame],
    sample_users: Optional[int],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    if sample_users is None:
        return df_avgemb, user_docs_train, user_docs_test, cosine_sims
    users = list(df_avgemb["AnonID"].unique())
    rng = np.random.default_rng(seed)
    sample_size = min(sample_users, len(users))
    sampled_users = set(rng.choice(np.array(users, dtype=object), size=sample_size, replace=False).tolist())
    return (
        df_avgemb[df_avgemb["AnonID"].isin(sampled_users)].copy(),
        user_docs_train[user_docs_train["AnonID"].isin(sampled_users)].copy(),
        user_docs_test[user_docs_test["AnonID"].isin(sampled_users)].copy(),
        cosine_sims.loc[list(sampled_users), list(sampled_users)].copy() if cosine_sims is not None else None,
    )


def _comparison_setup(
    df_avgemb: pd.DataFrame,
    user_docs_train: pd.DataFrame,
    user_docs_test: pd.DataFrame,
    cosine_sims: Optional[pd.DataFrame],
    n: int,
    m_multiplier: int,
    random_exploration_fraction: float,
    doc_embeddings: Optional[pd.DataFrame] = None,
) -> Tuple[List[UserId], int, int, int, int, SimilarityIndex, MetricEvaluator]:
    users = list(df_avgemb["AnonID"])
    if len(users) < 2:
        raise ValueError("At least two users are required for peer-graph comparison.")
    n = min(n, len(users) - 1)
    ranked_slots, random_slots = exploration_slot_counts(n, random_exploration_fraction)
    m = min(ranked_slots * m_multiplier, len(users) - 1)
    if cosine_sims is None:
        similarity = SimilarityIndex.from_avg_embeddings(df_avgemb, users=users)
    else:
        similarity = SimilarityIndex(cosine_sims, users=users)
    print(
        f"Phase start: computing exact top-{ranked_slots} peers for closest-neighbor metrics "
        f"over {len(users)} users.",
        flush=True,
    )
    exact_topn = similarity.exact_topn(ranked_slots)
    print("Phase complete: exact top peers ready.", flush=True)
    metric_evaluator = MetricEvaluator(
        users=users,
        exact_topn=exact_topn,
        user_docs_train=user_docs_train,
        user_docs_test=user_docs_test,
        n=ranked_slots,
        doc_embeddings=doc_embeddings,
    )
    return users, n, ranked_slots, random_slots, m, similarity, metric_evaluator


def _run_algorithm_for_setup(
    setup_name: str,
    initial_graph: InitialGraph,
    similarity: SimilarityIndex,
    metric_evaluator: MetricEvaluator,
    n: int,
    caps: BudgetCaps,
    checkpoints_by_axis: Mapping[str, Sequence[float]],
    seed: int,
    max_nnd_iterations: int,
    max_semantica_rounds: int,
    centroid_semantica_batch_size: int = 1,
    centroid_semantica_workers: int = 1,
    centroid_semantica_worker_peer_batch_size: Optional[int] = None,
    checkpoint_include_document_metrics: bool = True,
    centroid_semantica_referrer_policy: str = "legacy_mixed_referrers",
) -> AlgorithmResult:
    if setup_name not in SETUP_CONFIGS:
        raise ValueError(f"Unsupported setup_name: {setup_name}")
    config = SETUP_CONFIGS[setup_name]
    if config["refiner"] == "semantica":
        if initial_graph.centroid_top_by_user:
            return run_centroid_semantica_expansion(
                setup_name=setup_name,
                initializer=config["initializer"],
                initial_graph=initial_graph,
                similarity=similarity,
                metric_evaluator=metric_evaluator,
                n=n,
                caps=caps,
                checkpoints_by_axis=checkpoints_by_axis,
                seed=seed,
                max_rounds=max_semantica_rounds,
                batch_size=centroid_semantica_batch_size,
                workers=centroid_semantica_workers,
                worker_peer_batch_size=centroid_semantica_worker_peer_batch_size,
                checkpoint_include_document_metrics=checkpoint_include_document_metrics,
                referrer_policy=centroid_semantica_referrer_policy,
            )
        return run_semantica_expansion(
            setup_name=setup_name,
            initializer=config["initializer"],
            initial_graph=initial_graph,
            similarity=similarity,
            metric_evaluator=metric_evaluator,
            n=n,
            caps=caps,
            checkpoints_by_axis=checkpoints_by_axis,
            seed=seed,
            max_rounds=max_semantica_rounds,
            checkpoint_include_document_metrics=checkpoint_include_document_metrics,
        )
    return run_full_nndescent(
        setup_name=setup_name,
        initializer=config["initializer"],
        initial_graph=initial_graph,
        similarity=similarity,
        metric_evaluator=metric_evaluator,
        n=n,
        caps=caps,
        checkpoints_by_axis=checkpoints_by_axis,
        max_iterations=max_nnd_iterations,
        checkpoint_include_document_metrics=checkpoint_include_document_metrics,
    )


def _write_comparison_outputs(
    output_dir: str,
    results: Sequence[AlgorithmResult],
    initial_graphs: Sequence[InitialGraph],
    metric_evaluator: MetricEvaluator,
    num_users: int,
    config: Mapping[str, object],
    save_final_graph_artifact: bool = True,
    save_tree_metadata: bool = True,
    save_plots: bool = True,
    inline_centroid_chain_metrics: bool = False,
    centroid_chain_max_hops: int = 50,
    centroid_chain_hop_cutoffs: Sequence[int] = (1, 2, 5, 10, 20, 50),
    centroid_chain_workers: int = 1,
    centroid_chain_evaluate_initial: bool = False,
    centroid_chain_primary_metrics: bool = False,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    trace_rows = []
    for result in results:
        trace_rows.extend(result.trace_rows)
    trace_df = pd.DataFrame(trace_rows)
    trace_path = os.path.join(output_dir, "budget_trace.csv")
    trace_df.to_csv(trace_path, index=False)

    init_summary = pd.DataFrame(initial_summary_rows(initial_graphs))
    init_summary_path = os.path.join(output_dir, "initialization_summary.csv")
    init_summary.to_csv(init_summary_path, index=False)

    final_summary_rows = []
    chain_hop_rows: List[Dict[str, object]] = []
    chain_summary_payload: Dict[str, object] = {}
    initial_chain_cache: Dict[int, Dict[str, object]] = {}
    for result in results:
        per_peer = result.final_cost.per_peer(num_users)
        final_metrics = metric_evaluator.evaluate(
            result.top_by_user,
            result.random_by_user,
            centroid_top_by_user=result.centroid_top_by_user,
            centroid_embeddings_by_user=result.centroid_embeddings_by_user,
            include_document_metrics=not centroid_chain_primary_metrics,
        )
        row = {
            "setup_name": result.setup_name,
            "initializer": result.initializer,
            "refiner": result.refiner,
            "messages_per_peer": per_peer["messages"],
            "candidate_ids_transmitted_per_peer": per_peer["candidate_ids"],
            "cosine_computations_per_peer": per_peer["cosine_computations"],
            "closest_peer_recall_at_50": final_metrics["closest_peer_recall_at_50"],
            "mean_common_exact_top50": final_metrics["mean_common_exact_top50"],
            "document_retrieval_success": final_metrics["document_retrieval_success"],
            "centroid_routed_document_retrieval_success": final_metrics[
                "centroid_routed_document_retrieval_success"
            ],
            "query_hit_rate": final_metrics["query_hit_rate"],
            "macro_peer_hit_rate": final_metrics["macro_peer_hit_rate"],
            "query_hit_count": final_metrics["query_hit_count"],
            "evaluated_query_count": final_metrics["evaluated_query_count"],
            "document_query_hit_rate": final_metrics["document_query_hit_rate"],
            "document_macro_peer_hit_rate": final_metrics["document_macro_peer_hit_rate"],
            "centroid_routed_query_hit_rate": final_metrics["centroid_routed_query_hit_rate"],
            "centroid_routed_macro_peer_hit_rate": final_metrics[
                "centroid_routed_macro_peer_hit_rate"
            ],
            "primary_hit_metric_variant": final_metrics["primary_hit_metric_variant"],
            "converged": result.converged,
            "stop_reason": result.stop_reason,
            "runtime_seconds": result.runtime_seconds,
            "updates_so_far": result.updates_so_far,
            "rounds_completed": int(result.rounds_completed),
            "expansion_round": int(result.rounds_completed),
            "query_hop": 1,
            "measurement_step_type": "final_topology",
            "query_hop_semantics": "direct_outgoing_peer_contact_after_expansion_checkpoint",
        }
        row.update(result.expansion_diagnostics)
        if inline_centroid_chain_metrics:
            matching_initial = next(
                (graph for graph in initial_graphs if graph.initializer == result.initializer),
                initial_graphs[0],
            )
            if centroid_chain_evaluate_initial:
                initial_cache_key = id(matching_initial)
                initial_chain_summary = initial_chain_cache.get(initial_cache_key)
                if initial_chain_summary is None:
                    print(
                        "Phase start: initial-topology centroid-chain retrieval metrics "
                        f"for setup={result.setup_name}.",
                        flush=True,
                    )
                    initial_chain_summary = evaluate_centroid_chain_document_retrieval(
                        user_docs_train=metric_evaluator.user_docs_train_df,
                        user_docs_test=metric_evaluator.user_docs_test_df,
                        doc_embeddings=metric_evaluator.doc_embeddings_df,
                        centroid_top_by_user=matching_initial.centroid_top_by_user,
                        random_by_user=matching_initial.random_by_user,
                        centroid_embeddings_by_user=matching_initial.centroid_embeddings_by_user,
                        max_hops=centroid_chain_max_hops,
                        hop_cutoffs=centroid_chain_hop_cutoffs,
                        workers=centroid_chain_workers,
                    )
                    initial_chain_cache[initial_cache_key] = initial_chain_summary
                chain_summary_payload.setdefault(result.setup_name, {})["initial"] = initial_chain_summary
                chain_hop_rows.extend(
                    {
                        "setup_name": result.setup_name,
                        "initializer": result.initializer,
                        **chain_row,
                    }
                    for chain_row in centroid_chain_hop_rows(
                        initial_chain_summary,
                        max_hops=centroid_chain_max_hops,
                        expansion_round=0,
                        topology_stage="initial",
                    )
                )

            print(f"Phase start: final-topology centroid-chain retrieval metrics for setup={result.setup_name}.", flush=True)
            chain_summary = evaluate_centroid_chain_document_retrieval(
                user_docs_train=metric_evaluator.user_docs_train_df,
                user_docs_test=metric_evaluator.user_docs_test_df,
                doc_embeddings=metric_evaluator.doc_embeddings_df,
                centroid_top_by_user=result.centroid_top_by_user,
                random_by_user=result.random_by_user,
                centroid_embeddings_by_user=result.centroid_embeddings_by_user,
                max_hops=centroid_chain_max_hops,
                hop_cutoffs=centroid_chain_hop_cutoffs,
                workers=centroid_chain_workers,
            )
            chain_summary_payload.setdefault(result.setup_name, {})["final"] = chain_summary
            chain_hop_rows.extend(
                {
                    "setup_name": result.setup_name,
                    "initializer": result.initializer,
                    **chain_row,
                }
                for chain_row in centroid_chain_hop_rows(
                    chain_summary,
                    max_hops=centroid_chain_max_hops,
                    expansion_round=int(result.rounds_completed),
                    topology_stage="final",
                )
            )
            for key, value in flatten_centroid_chain_summary(chain_summary).items():
                row[f"inline_{key}"] = value
            row["chain_query_hit_rate_at_max_hops"] = (
                float(chain_summary["centroid_chain_doc_retrieval_success"]) / 100.0
            )
            row["chain_available_doc_recall_at_max_hops"] = (
                float(chain_summary["centroid_chain_available_doc_recall"]) / 100.0
            )
            if centroid_chain_primary_metrics:
                row["query_hit_rate"] = row["chain_query_hit_rate_at_max_hops"]
                row["macro_peer_hit_rate"] = float("nan")
                row["query_hit_count"] = int(chain_summary["success_count"])
                row["evaluated_query_count"] = int(chain_summary["query_count"])
                row["primary_hit_metric_variant"] = "centroid_chain_first_hit"
                row["query_hop"] = int(centroid_chain_max_hops)
                row["query_hop_semantics"] = "sequential_best_unvisited_neighbor_until_first_hit"
            print(
                "Phase complete: inline centroid-chain available-doc recall="
                f"{chain_summary['centroid_chain_available_doc_recall']:.3f}%.",
                flush=True,
            )
        final_summary_rows.append(row)
    final_summary = pd.DataFrame(final_summary_rows)
    final_summary_path = os.path.join(output_dir, "final_summary.csv")
    final_summary.to_csv(final_summary_path, index=False)

    chain_hop_summary_path = None
    chain_summary_path = None
    if inline_centroid_chain_metrics:
        chain_hop_summary_path = os.path.join(output_dir, "chain_hop_summary.csv")
        pd.DataFrame(chain_hop_rows).to_csv(chain_hop_summary_path, index=False)
        chain_summary_path = os.path.join(output_dir, "chain_summary.json")
        with open(chain_summary_path, "w") as f:
            json.dump(_json_ready(chain_summary_payload), f, indent=2)

    if save_final_graph_artifact:
        save_final_graphs(results, os.path.join(output_dir, "final_graphs.pkl"))
    if save_plots:
        plot_budget_traces(trace_df, output_dir)

    metadata_paths = {}
    if save_tree_metadata:
        for graph in initial_graphs:
            if not graph.metadata:
                continue
            metadata_filename = f"{graph.initializer}_metadata.json"
            metadata_path = os.path.join(output_dir, metadata_filename)
            with open(metadata_path, "w") as f:
                json.dump(_json_ready(graph.metadata), f, indent=2)
            metadata_paths[graph.initializer] = metadata_path

    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        config_with_outputs = dict(config)
        config_with_outputs.update(
            {
                "save_final_graph_artifact": bool(save_final_graph_artifact),
                "save_tree_metadata": bool(save_tree_metadata),
                "save_plots": bool(save_plots),
                "inline_centroid_chain_metrics": bool(inline_centroid_chain_metrics),
                "centroid_chain_max_hops": int(centroid_chain_max_hops),
                "centroid_chain_hop_cutoffs": [int(value) for value in centroid_chain_hop_cutoffs],
                "centroid_chain_workers": int(centroid_chain_workers),
                "centroid_chain_evaluate_initial": bool(centroid_chain_evaluate_initial),
                "centroid_chain_primary_metrics": bool(centroid_chain_primary_metrics),
                "centroid_chain_output": {
                    "per_query_rows_saved": False,
                    "exact_first_hit_histogram": "chain_hop_summary.csv",
                    "failure_definition": "target_not_found_before_or_at_centroid_chain_max_hops",
                    "message_definition": "one_network_message_per_traversed_chain_edge",
                },
                "document_hit_metrics": {
                    "primary_metric": "query_hit_rate",
                    "primary_weighting": "micro_over_queries",
                    "companion_metric": "macro_peer_hit_rate",
                    "companion_weighting": "equal_weight_per_peer",
                    "legacy_percent_metric": "centroid_routed_document_retrieval_success",
                    "rate_scale": "0_to_1",
                    "legacy_percent_scale": "0_to_100",
                    "query_hop": (
                        int(centroid_chain_max_hops) if centroid_chain_primary_metrics else 1
                    ),
                    "query_hop_semantics": (
                        "sequential_best_unvisited_neighbor_until_first_hit"
                        if centroid_chain_primary_metrics
                        else "direct_outgoing_peer_contact_after_expansion_checkpoint"
                    ),
                    "expansion_round_semantics": "completed_graph_construction_expansion_rounds_before_query_evaluation",
                },
            }
        )
        json.dump(config_with_outputs, f, indent=2)

    return {
        "trace_path": trace_path,
        "initialization_summary_path": init_summary_path,
        "final_summary_path": final_summary_path,
        "chain_hop_summary_path": chain_hop_summary_path,
        "chain_summary_path": chain_summary_path,
        "metadata_paths": metadata_paths,
        "trace_df": trace_df,
        "initialization_summary": init_summary,
        "final_summary": final_summary,
        "results": list(results),
    }


def run_budgeted_peer_graph_single_setup(
    setup_name: str,
    df_avgemb: pd.DataFrame,
    user_docs_train: pd.DataFrame,
    user_docs_test: pd.DataFrame,
    cosine_sims: pd.DataFrame,
    output_dir: str,
    n: int = 50,
    m_multiplier: int = 1,
    record_limit_per_leafnode: int = 50,
    tree_delta: float = 5e-4,
    seed: int = 42,
    caps: Optional[BudgetCaps] = None,
    max_nnd_iterations: int = 30,
    max_semantica_rounds: Optional[int] = None,
    doc_embeddings: Optional[pd.DataFrame] = None,
    multi_centroid_fixed_k: int = 2,
    multi_centroid_elbow_max_k: int = 8,
    multi_centroid_elbow_method: str = "knee",
    multi_centroid_relative_drop_threshold: float = 0.10,
    multi_centroid_min_contact_quota: int = 3,
    random_exploration_fraction: float = 0.10,
    tree_insertion_seed: Optional[int] = None,
    multi_centroid_cache_path: Optional[str] = None,
    centroid_semantica_batch_size: int = 1,
    centroid_semantica_workers: int = 1,
    centroid_semantica_worker_peer_batch_size: Optional[int] = None,
    checkpoint_include_document_metrics: bool = True,
    centroid_semantica_referrer_policy: str = "legacy_mixed_referrers",
    save_final_graph_artifact: bool = True,
    save_tree_metadata: bool = True,
    save_plots: bool = True,
    inline_centroid_chain_metrics: bool = False,
    centroid_chain_max_hops: int = 50,
    centroid_chain_hop_cutoffs: Sequence[int] = (1, 2, 5, 10, 20, 50),
    centroid_chain_workers: int = 1,
    centroid_chain_evaluate_initial: bool = False,
    centroid_chain_primary_metrics: bool = False,
    include_message_checkpoints: bool = True,
    round_checkpoints: Sequence[int] = (),
) -> Dict[str, object]:
    if setup_name not in SETUP_CONFIGS:
        raise ValueError(f"Unsupported setup_name: {setup_name}")
    if centroid_semantica_referrer_policy not in CENTROID_SEMANTICA_REFERRER_POLICIES:
        raise ValueError(
            "Unsupported centroid Semantica referrer policy: "
            f"{centroid_semantica_referrer_policy}"
        )
    users, n, ranked_slots, random_slots, m, similarity, metric_evaluator = _comparison_setup(
        df_avgemb=df_avgemb,
        user_docs_train=user_docs_train,
        user_docs_test=user_docs_test,
        cosine_sims=cosine_sims,
        n=n,
        m_multiplier=m_multiplier,
        random_exploration_fraction=random_exploration_fraction,
        doc_embeddings=doc_embeddings,
    )
    random_by_user = build_protected_random_links(similarity, random_slots=random_slots, seed=seed)
    if caps is None:
        caps = BudgetCaps(
            messages_per_peer=500,
            candidate_ids_per_peer=UNBOUNDED_BUDGET,
            cosine_computations_per_peer=UNBOUNDED_BUDGET,
        )
    if max_semantica_rounds is None:
        max_semantica_rounds = int(math.ceil(caps.messages_per_peer))

    checkpoints_by_axis = make_budget_checkpoints(caps)
    if not include_message_checkpoints:
        checkpoints_by_axis["messages"] = []
    checkpoints_by_axis["rounds"] = sorted(
        {float(value) for value in round_checkpoints if 0 < int(value) <= int(max_semantica_rounds)}
    )
    initializer = SETUP_CONFIGS[setup_name]["initializer"]
    print(f"Phase start: building initial graph with initializer={initializer}.", flush=True)
    if initializer == "random":
        initial_graph = build_random_initial_graph(
            similarity,
            metric_evaluator,
            n=ranked_slots,
            seed=seed,
            random_by_user=random_by_user,
        )
    elif initializer == "tree":
        initial_graph = build_tree_initial_graph(
            df_avgemb=df_avgemb,
            user_docs_train=user_docs_train,
            user_docs_test=user_docs_test,
            cosine_sims=cosine_sims,
            similarity=similarity,
            metric_evaluator=metric_evaluator,
            n=ranked_slots,
            m=m,
            record_limit_per_leafnode=record_limit_per_leafnode,
            tree_delta=tree_delta,
            random_by_user=random_by_user,
        )
    elif initializer in {"tree_fixed_k", "tree_elbow_k", "random_fixed_k", "random_elbow_k"}:
        initial_graph = build_multi_centroid_tree_initial_graph(
            df_avgemb=df_avgemb,
            user_docs_train=user_docs_train,
            doc_embeddings=doc_embeddings,
            similarity=similarity,
            metric_evaluator=metric_evaluator,
            n=ranked_slots,
            m=m,
            record_limit_per_leafnode=record_limit_per_leafnode,
            tree_delta=tree_delta,
            seed=seed,
            mode="fixed" if initializer == "tree_fixed_k" else "elbow",
            fixed_k=multi_centroid_fixed_k,
            elbow_max_k=multi_centroid_elbow_max_k,
            elbow_method=multi_centroid_elbow_method,
            relative_drop_threshold=multi_centroid_relative_drop_threshold,
            min_contact_quota=multi_centroid_min_contact_quota,
            random_by_user=random_by_user,
            tree_insertion_seed=tree_insertion_seed,
            centroid_cache_path=multi_centroid_cache_path,
            topology="random" if initializer.startswith("random_") else "tree",
        )
    else:
        raise ValueError(f"Unsupported initializer: {initializer}")
    print(f"Phase complete: initial graph built with initializer={initializer}.", flush=True)
    print(f"Phase start: running refiner for setup={setup_name}.", flush=True)
    result = _run_algorithm_for_setup(
        setup_name=setup_name,
        initial_graph=initial_graph,
        similarity=similarity,
        metric_evaluator=metric_evaluator,
        n=ranked_slots,
        caps=caps,
        checkpoints_by_axis=checkpoints_by_axis,
        seed=seed,
        max_nnd_iterations=max_nnd_iterations,
        max_semantica_rounds=max_semantica_rounds,
        centroid_semantica_batch_size=centroid_semantica_batch_size,
        centroid_semantica_workers=centroid_semantica_workers,
        centroid_semantica_worker_peer_batch_size=centroid_semantica_worker_peer_batch_size,
        checkpoint_include_document_metrics=checkpoint_include_document_metrics,
        centroid_semantica_referrer_policy=centroid_semantica_referrer_policy,
    )
    print(f"Phase complete: refiner finished for setup={setup_name}.", flush=True)
    config = {
        "setup_name": setup_name,
        "n": n,
        "ranked_slots": ranked_slots,
        "random_slots": random_slots,
        "random_exploration_fraction": random_exploration_fraction,
        "m": m,
        "m_multiplier": m_multiplier,
        "record_limit_per_leafnode": record_limit_per_leafnode,
        "tree_delta": tree_delta,
        "tree_insertion_seed": tree_insertion_seed,
        "multi_centroid_cache_path": multi_centroid_cache_path,
        "seed": seed,
        "caps": {
            "messages_per_peer": caps.messages_per_peer,
            "candidate_ids_per_peer": caps.candidate_ids_per_peer,
            "cosine_computations_per_peer": caps.cosine_computations_per_peer,
        },
        "max_nnd_iterations": max_nnd_iterations,
        "max_semantica_rounds": max_semantica_rounds,
        "centroid_semantica_batch_size": int(centroid_semantica_batch_size),
        "centroid_semantica_workers": int(centroid_semantica_workers),
        "centroid_semantica_worker_peer_batch_size": (
            None
            if centroid_semantica_worker_peer_batch_size is None
            else int(centroid_semantica_worker_peer_batch_size)
        ),
        "centroid_semantica_expansion_mode": (
            "batched" if int(centroid_semantica_batch_size) > 1 or int(centroid_semantica_workers) > 1 else "serial"
        ),
        "centroid_semantica_referrer_policy": centroid_semantica_referrer_policy,
        "centroid_semantica_referrer_policy_definition": CENTROID_SEMANTICA_REFERRER_POLICY_DEFINITIONS[
            centroid_semantica_referrer_policy
        ],
        "expansion_diagnostics": {
            "fields": list(EXPANSION_DIAGNOSTIC_FIELDS),
            "round_prefix": "counts attributable to the current completed expansion round",
            "cumulative_prefix": "counts accumulated from expansion round 1 through the current round",
            "per_peer_suffix": "count divided by the number of evaluated peers",
        },
        "checkpoint_include_document_metrics": bool(checkpoint_include_document_metrics),
        "include_message_checkpoints": bool(include_message_checkpoints),
        "round_checkpoints": [int(value) for value in round_checkpoints],
        "num_users": len(users),
        "multi_centroid": {
            "fixed_k": multi_centroid_fixed_k,
            "elbow_max_k": multi_centroid_elbow_max_k,
            "elbow_method": multi_centroid_elbow_method,
            "relative_drop_threshold": multi_centroid_relative_drop_threshold,
            "min_contact_quota": multi_centroid_min_contact_quota,
        },
    }
    return _write_comparison_outputs(
        output_dir=output_dir,
        results=[result],
        initial_graphs=[initial_graph],
        metric_evaluator=metric_evaluator,
        num_users=len(users),
        config=config,
        save_final_graph_artifact=save_final_graph_artifact,
        save_tree_metadata=save_tree_metadata,
        save_plots=save_plots,
        inline_centroid_chain_metrics=inline_centroid_chain_metrics,
        centroid_chain_max_hops=centroid_chain_max_hops,
        centroid_chain_hop_cutoffs=centroid_chain_hop_cutoffs,
        centroid_chain_workers=centroid_chain_workers,
        centroid_chain_evaluate_initial=centroid_chain_evaluate_initial,
        centroid_chain_primary_metrics=centroid_chain_primary_metrics,
    )


def run_budgeted_peer_graph_comparison(
    df_avgemb: pd.DataFrame,
    user_docs_train: pd.DataFrame,
    user_docs_test: pd.DataFrame,
    cosine_sims: pd.DataFrame,
    output_dir: str,
    n: int = 50,
    m_multiplier: int = 1,
    record_limit_per_leafnode: int = 50,
    tree_delta: float = 5e-4,
    seed: int = 42,
    caps: Optional[BudgetCaps] = None,
    max_nnd_iterations: int = 30,
    max_semantica_rounds: Optional[int] = None,
    random_exploration_fraction: float = 0.10,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)
    users, n, ranked_slots, random_slots, m, similarity, metric_evaluator = _comparison_setup(
        df_avgemb=df_avgemb,
        user_docs_train=user_docs_train,
        user_docs_test=user_docs_test,
        cosine_sims=cosine_sims,
        n=n,
        m_multiplier=m_multiplier,
        random_exploration_fraction=random_exploration_fraction,
    )
    random_by_user = build_protected_random_links(similarity, random_slots=random_slots, seed=seed)
    if caps is None:
        caps = BudgetCaps(
            messages_per_peer=500,
            candidate_ids_per_peer=UNBOUNDED_BUDGET,
            cosine_computations_per_peer=UNBOUNDED_BUDGET,
        )
    if max_semantica_rounds is None:
        max_semantica_rounds = int(math.ceil(caps.messages_per_peer))

    checkpoints_by_axis = make_budget_checkpoints(caps)

    random_initial = build_random_initial_graph(
        similarity,
        metric_evaluator,
        n=ranked_slots,
        seed=seed,
        random_by_user=random_by_user,
    )
    tree_initial = build_tree_initial_graph(
        df_avgemb=df_avgemb,
        user_docs_train=user_docs_train,
        user_docs_test=user_docs_test,
        cosine_sims=cosine_sims,
        similarity=similarity,
        metric_evaluator=metric_evaluator,
        n=ranked_slots,
        m=m,
        record_limit_per_leafnode=record_limit_per_leafnode,
        tree_delta=tree_delta,
        random_by_user=random_by_user,
    )

    results = [
        run_semantica_expansion(
            setup_name="random_init_semantica_expansion",
            initializer="random",
            initial_graph=random_initial,
            similarity=similarity,
            metric_evaluator=metric_evaluator,
            n=ranked_slots,
            caps=caps,
            checkpoints_by_axis=checkpoints_by_axis,
            seed=seed,
            max_rounds=max_semantica_rounds,
        ),
        run_semantica_expansion(
            setup_name="tree_init_semantica_expansion",
            initializer="tree",
            initial_graph=tree_initial,
            similarity=similarity,
            metric_evaluator=metric_evaluator,
            n=ranked_slots,
            caps=caps,
            checkpoints_by_axis=checkpoints_by_axis,
            seed=seed,
            max_rounds=max_semantica_rounds,
        ),
        run_full_nndescent(
            setup_name="random_init_full_nndescent",
            initializer="random",
            initial_graph=random_initial,
            similarity=similarity,
            metric_evaluator=metric_evaluator,
            n=ranked_slots,
            caps=caps,
            checkpoints_by_axis=checkpoints_by_axis,
            max_iterations=max_nnd_iterations,
        ),
        run_full_nndescent(
            setup_name="tree_init_full_nndescent",
            initializer="tree",
            initial_graph=tree_initial,
            similarity=similarity,
            metric_evaluator=metric_evaluator,
            n=ranked_slots,
            caps=caps,
            checkpoints_by_axis=checkpoints_by_axis,
            max_iterations=max_nnd_iterations,
        ),
    ]

    config = {
        "n": n,
        "ranked_slots": ranked_slots,
        "random_slots": random_slots,
        "random_exploration_fraction": random_exploration_fraction,
        "m": m,
        "m_multiplier": m_multiplier,
        "record_limit_per_leafnode": record_limit_per_leafnode,
        "tree_delta": tree_delta,
        "seed": seed,
        "caps": {
            "messages_per_peer": caps.messages_per_peer,
            "candidate_ids_per_peer": caps.candidate_ids_per_peer,
            "cosine_computations_per_peer": caps.cosine_computations_per_peer,
        },
        "max_nnd_iterations": max_nnd_iterations,
        "max_semantica_rounds": max_semantica_rounds,
        "num_users": len(users),
    }
    return _write_comparison_outputs(
        output_dir=output_dir,
        results=results,
        initial_graphs=[random_initial, tree_initial],
        metric_evaluator=metric_evaluator,
        num_users=len(users),
        config=config,
    )


def make_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

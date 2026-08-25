from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from .algorithms import (
    assign_unique_candidates,
    choose_child,
    fit_peer_profile,
    local_topk,
    merge_topk,
    peer_centroid_score,
    replace_centroid_contacts,
    split_entries,
    stable_int,
)
from .dataset import load_shard, prepare_aol_shards, prepare_demo_shards
from .models import CentroidEntry, Endpoint, PeerProfile


@dataclass
class _CentralLeaf:
    entries: list[CentroidEntry]


@dataclass
class _CentralSplit:
    left_center: np.ndarray
    right_center: np.ndarray
    left: "_CentralNode"
    right: "_CentralNode"


_CentralNode = Union[_CentralLeaf, _CentralSplit]


class CentralCentroidTree:
    """In-process tree used by the centralized simulation."""

    def __init__(self, *, leaf_capacity: int, seed: int, delta: float = 0.0) -> None:
        self.leaf_capacity = int(leaf_capacity)
        self.seed = int(seed)
        self.delta = float(delta)
        self.root: _CentralNode = _CentralLeaf([])

    def insert(self, entry: CentroidEntry) -> None:
        self.root = self._insert(self.root, entry)

    def _insert(self, node: _CentralNode, entry: CentroidEntry) -> _CentralNode:
        if isinstance(node, _CentralSplit):
            side = choose_child(entry.vector, node.left_center, node.right_center)
            if side == "left":
                node.left = self._insert(node.left, entry)
            else:
                node.right = self._insert(node.right, entry)
            return node

        node.entries.append(entry)
        if len(node.entries) <= self.leaf_capacity:
            return node
        split = split_entries(node.entries, seed=self.seed, delta=self.delta)
        if split is None:
            return node
        left_center, right_center, left_entries, right_entries = split
        return _CentralSplit(
            left_center=left_center,
            right_center=right_center,
            left=self._split_overflow(_CentralLeaf(list(left_entries))),
            right=self._split_overflow(_CentralLeaf(list(right_entries))),
        )

    def _split_overflow(self, leaf: _CentralLeaf) -> _CentralNode:
        if len(leaf.entries) <= self.leaf_capacity:
            return leaf
        split = split_entries(leaf.entries, seed=self.seed, delta=self.delta)
        if split is None:
            return leaf
        left_center, right_center, left_entries, right_entries = split
        if max(len(left_entries), len(right_entries)) >= len(leaf.entries):
            return leaf
        return _CentralSplit(
            left_center=left_center,
            right_center=right_center,
            left=self._split_overflow(_CentralLeaf(list(left_entries))),
            right=self._split_overflow(_CentralLeaf(list(right_entries))),
        )

    def counts(self) -> tuple[int, int]:
        def walk(node: _CentralNode) -> tuple[int, int]:
            if isinstance(node, _CentralLeaf):
                return 1, len(node.entries)
            left_leaves, left_entries = walk(node.left)
            right_leaves, right_entries = walk(node.right)
            return left_leaves + right_leaves, left_entries + right_entries

        return walk(self.root)

    def locate_leaf(self, vector: np.ndarray) -> _CentralLeaf:
        node = self.root
        while isinstance(node, _CentralSplit):
            side = choose_child(vector, node.left_center, node.right_center)
            node = node.left if side == "left" else node.right
        return node

    def parent_index(self) -> dict[int, _CentralNode | None]:
        parents: dict[int, _CentralNode | None] = {}

        def walk(node: _CentralNode, parent: _CentralNode | None) -> None:
            parents[id(node)] = parent
            if isinstance(node, _CentralSplit):
                walk(node.left, node)
                walk(node.right, node)

        walk(self.root, None)
        return parents

    @staticmethod
    def neighbors(
        node: _CentralNode, parents: dict[int, _CentralNode | None]
    ) -> list[_CentralNode]:
        values: list[_CentralNode] = []
        parent = parents[id(node)]
        if parent is not None:
            values.append(parent)
        if isinstance(node, _CentralSplit):
            values.extend([node.left, node.right])
        return values


def _fit_profiles(
    shards: list[Dict[str, Any]], parameters: Dict[str, Any]
) -> list[PeerProfile]:
    return [
        fit_peer_profile(
            peer_id,
            Endpoint("centralized", 10_000 + peer_id),
            np.asarray(shard["training_embeddings"], dtype=np.float32),
            degree=int(parameters["degree"]),
            seed=int(parameters["seed"]),
            max_k=int(parameters["max_centroids"]),
        )
        for peer_id, shard in enumerate(shards)
    ]


def _build_round_zero_contacts(
    profiles: list[PeerProfile], tree: CentralCentroidTree
) -> list[list[list[tuple[int, float]]]]:
    parents = tree.parent_index()
    all_contacts: list[list[list[tuple[int, float]]]] = []
    for source_peer, profile in enumerate(profiles):
        frontiers: list[list[_CentralNode]] = [
            [tree.locate_leaf(centroid)] for centroid in profile.centroids
        ]
        visited = [{id(frontier[0])} for frontier in frontiers]
        candidate_scores: list[dict[int, float]] = [
            {} for _centroid in profile.centroids
        ]
        active = list(range(len(frontiers)))
        assigned: list[list[tuple[int, float]]] = [[] for _centroid in frontiers]
        while active:
            for centroid_index in active:
                owners: set[int] = set()
                next_frontier: list[_CentralNode] = []
                for node in frontiers[centroid_index]:
                    if isinstance(node, _CentralLeaf):
                        owners.update(
                            entry.peer_id
                            for entry in node.entries
                            if entry.peer_id != source_peer
                        )
                    for neighbor in tree.neighbors(node, parents):
                        node_identity = id(neighbor)
                        if node_identity not in visited[centroid_index]:
                            visited[centroid_index].add(node_identity)
                            next_frontier.append(neighbor)
                frontiers[centroid_index] = next_frontier
                for owner in sorted(owners):
                    if owner not in candidate_scores[centroid_index]:
                        candidate_scores[centroid_index][owner] = peer_centroid_score(
                            profile.centroids[centroid_index], profiles[owner]
                        )
            assigned, underfilled = assign_unique_candidates(
                candidate_scores, profile.contact_caps.tolist()
            )
            if not underfilled:
                break
            active = [index for index in underfilled if frontiers[index]]
            if not active:
                missing = {
                    index: int(profile.contact_caps[index] - len(assigned[index]))
                    for index in underfilled
                }
                raise RuntimeError(
                    f"Unable to fill centralized round-0 degree for peer "
                    f"{source_peer}: {missing}"
                )
        all_contacts.append(assigned)
    return all_contacts


def _expand_contacts(
    profiles: list[PeerProfile],
    round_zero: list[list[list[tuple[int, float]]]],
    parameters: Dict[str, Any],
) -> list[list[list[tuple[int, float]]]]:
    current = round_zero
    initial_contact_sets = [
        [{candidate for candidate, _score in values} for values in peer_contacts]
        for peer_contacts in round_zero
    ]
    asked_referrers = [
        [set() for _centroid in peer_contacts] for peer_contacts in round_zero
    ]
    for round_index in range(1, int(parameters["expansion_rounds"]) + 1):
        previous = current
        expanded: list[list[list[tuple[int, float]]]] = []
        for source_peer, profile in enumerate(profiles):
            peer_contacts = [list(values) for values in previous[source_peer]]
            owner_existing = {
                candidate for values in previous[source_peer] for candidate, _ in values
            }
            proposals: list[list[tuple[int, float]]] = [
                [] for _centroid in peer_contacts
            ]
            for centroid_index, values in enumerate(previous[source_peer]):
                candidate_ids = [candidate for candidate, _score in values]
                new_tier = [
                    candidate
                    for candidate in candidate_ids
                    if candidate not in initial_contact_sets[source_peer][centroid_index]
                    and candidate not in asked_referrers[source_peer][centroid_index]
                ]
                initial_tier = [
                    candidate
                    for candidate in candidate_ids
                    if candidate in initial_contact_sets[source_peer][centroid_index]
                    and candidate not in asked_referrers[source_peer][centroid_index]
                ]
                tier = new_tier if new_tier else initial_tier
                if not tier:
                    continue
                rng = np.random.default_rng(
                    stable_int(
                        f"semantic-referrer:{parameters['seed']}:"
                        f"{round_index}:{source_peer}:{centroid_index}"
                    )
                )
                referrer = int(tier[int(rng.integers(0, len(tier)))])
                asked_referrers[source_peer][centroid_index].add(referrer)
                referrals = {
                    candidate
                    for contacts in previous[referrer]
                    for candidate, _score in contacts
                    if candidate != source_peer
                }
                for candidate in sorted(referrals):
                    if candidate == source_peer or candidate in owner_existing:
                        continue
                    score = peer_centroid_score(
                        profile.centroids[centroid_index], profiles[candidate]
                    )
                    if np.isfinite(score):
                        proposals[centroid_index].append((candidate, score))
            for centroid_index, values in enumerate(proposals):
                excluded = {source_peer}
                excluded.update(
                    candidate
                    for other_index, contacts in enumerate(peer_contacts)
                    if other_index != centroid_index
                    for candidate, _score in contacts
                )
                peer_contacts[centroid_index] = replace_centroid_contacts(
                    peer_contacts[centroid_index],
                    values,
                    int(profile.contact_caps[centroid_index]),
                    excluded,
                )
            expanded.append(peer_contacts)
        current = expanded
    return current


def _build_contacts(
    profiles: list[PeerProfile],
    tree: CentralCentroidTree,
    parameters: Dict[str, Any],
) -> list[list[list[tuple[int, float]]]]:
    round_zero = _build_round_zero_contacts(profiles, tree)
    return _expand_contacts(profiles, round_zero, parameters)


def _run_queries(
    shards: list[Dict[str, Any]],
    profiles: list[PeerProfile],
    contacts: list[list[list[tuple[int, float]]]],
    parameters: Dict[str, Any],
) -> tuple[int, int]:
    total_hops = 0
    target_hits = 0
    neighbor_ids = [
        sorted({candidate for centroid in peer_contacts for candidate, _ in centroid})
        for peer_contacts in contacts
    ]
    for source_peer, shard in enumerate(shards):
        query = np.asarray(shard["query_embedding"], dtype=np.float32)
        target_doc_id = str(shard["target_doc_id"].item())
        current = source_peer
        visited = {source_peer}
        merged: list[tuple[str, float]] = []
        for _hop in range(int(parameters["query_hop_limit"])):
            candidates = [
                peer_id for peer_id in neighbor_ids[current] if peer_id not in visited
            ]
            if not candidates:
                break
            current = min(
                candidates,
                key=lambda peer_id: (
                    -peer_centroid_score(query, profiles[peer_id]),
                    peer_id,
                ),
            )
            visited.add(current)
            total_hops += 1
            candidate_shard = shards[current]
            local = local_topk(
                query,
                candidate_shard["training_doc_ids"].tolist(),
                np.asarray(candidate_shard["training_embeddings"], dtype=np.float32),
                int(parameters["local_top_k"]),
                doc_order_keys=np.asarray(
                    candidate_shard["training_doc_indices"], dtype=np.int64
                ),
            )
            merged = merge_topk(merged, local, int(parameters["local_top_k"]))
        if target_doc_id in {doc_id for doc_id, _score in merged}:
            target_hits += 1
    return total_hops, target_hits


def run_centralized_simulation(
    work_dir: Path,
    *,
    peer_count: int = 4,
    seed: int = 42,
    aol_bundle: Path | None = None,
) -> Dict[str, Any]:
    """Run the synthetic Semantica workflow in one process with global state."""
    work_dir = Path(work_dir).resolve()
    prepared_dir = work_dir / "prepared"
    if aol_bundle is None:
        manifest = prepare_demo_shards(
            prepared_dir, peer_count=int(peer_count), seed=int(seed)
        )
    else:
        manifest = prepare_aol_shards(
            Path(aol_bundle),
            prepared_dir,
            peer_count=int(peer_count),
            seed=int(seed),
        )
    parameters = dict(manifest["parameters"])
    shards = [
        load_shard(prepared_dir / "shards" / f"peer_{peer_id:05d}.npz")
        for peer_id in range(int(peer_count))
    ]
    profiles = _fit_profiles(shards, parameters)

    tree = CentralCentroidTree(
        leaf_capacity=int(parameters["leaf_capacity"]),
        seed=int(parameters["seed"]),
        delta=float(parameters["delta"]),
    )
    for profile in profiles:
        for centroid_index, vector in enumerate(profile.centroids):
            tree.insert(
                CentroidEntry(
                    peer_id=profile.peer_id,
                    centroid_index=centroid_index,
                    vector=vector,
                    profile_version=profile.version,
                    endpoint=profile.endpoint,
                )
            )
    leaf_count, tree_entry_count = tree.counts()
    contacts = _build_contacts(profiles, tree, parameters)
    total_hops, target_hits = _run_queries(
        shards, profiles, contacts, parameters
    )
    return {
        "status": "complete",
        "mode": "centralized",
        "peer_count": int(peer_count),
        "centroid_count": sum(len(profile.centroids) for profile in profiles),
        "tree_leaf_count": leaf_count,
        "tree_entry_count": tree_entry_count,
        "directed_contact_count": sum(
            len(centroid) for peer_contacts in contacts for centroid in peer_contacts
        ),
        "query_count": int(peer_count),
        "total_query_hops": total_hops,
        "target_hit_count": target_hits,
    }

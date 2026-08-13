from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int

    def to_wire(self) -> Dict[str, Any]:
        return {"host": self.host, "port": int(self.port)}

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "Endpoint":
        return cls(host=str(value["host"]), port=int(value["port"]))


@dataclass
class MembershipManifest:
    epoch: str
    seed: int
    peers: Dict[int, Endpoint]
    parameters: Dict[str, Any]

    def to_wire(self) -> Dict[str, Any]:
        return {
            "schema": "decentralized_semantica_membership_v1",
            "implementation": "port_based_decentralized_semantica_v1",
            "epoch": self.epoch,
            "seed": int(self.seed),
            "peers": {str(peer): endpoint.to_wire() for peer, endpoint in self.peers.items()},
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "MembershipManifest":
        return cls(
            epoch=str(value["epoch"]),
            seed=int(value["seed"]),
            peers={int(peer): Endpoint.from_wire(endpoint) for peer, endpoint in value["peers"].items()},
            parameters=dict(value["parameters"]),
        )


@dataclass
class PeerProfile:
    peer_id: int
    endpoint: Endpoint
    centroids: np.ndarray
    cluster_doc_counts: np.ndarray
    contact_caps: np.ndarray
    version: int = 1

    def to_wire(self) -> Dict[str, Any]:
        return {
            "peer_id": int(self.peer_id),
            "endpoint": self.endpoint.to_wire(),
            "centroids": np.asarray(self.centroids, dtype=np.float32),
            "cluster_doc_counts": np.asarray(self.cluster_doc_counts, dtype=np.int32),
            "contact_caps": np.asarray(self.contact_caps, dtype=np.int32),
            "version": int(self.version),
        }

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "PeerProfile":
        return cls(
            peer_id=int(value["peer_id"]),
            endpoint=Endpoint.from_wire(value["endpoint"]),
            centroids=np.asarray(value["centroids"], dtype=np.float32),
            cluster_doc_counts=np.asarray(value["cluster_doc_counts"], dtype=np.int32),
            contact_caps=np.asarray(value["contact_caps"], dtype=np.int32),
            version=int(value.get("version", 1)),
        )


@dataclass
class CentroidEntry:
    peer_id: int
    centroid_index: int
    vector: np.ndarray
    profile_version: int
    endpoint: Endpoint

    @property
    def entry_id(self) -> str:
        return f"{self.peer_id}:{self.profile_version}:{self.centroid_index}"

    def to_wire(self) -> Dict[str, Any]:
        return {
            "peer_id": int(self.peer_id),
            "centroid_index": int(self.centroid_index),
            "vector": np.asarray(self.vector, dtype=np.float32),
            "profile_version": int(self.profile_version),
            "endpoint": self.endpoint.to_wire(),
        }

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "CentroidEntry":
        return cls(
            peer_id=int(value["peer_id"]),
            centroid_index=int(value["centroid_index"]),
            vector=np.asarray(value["vector"], dtype=np.float32),
            profile_version=int(value["profile_version"]),
            endpoint=Endpoint.from_wire(value["endpoint"]),
        )


@dataclass(frozen=True)
class ChildDescriptor:
    node_id: str
    custodian_peer_id: int
    custodian_endpoint: Endpoint
    center: Optional[np.ndarray] = field(default=None, compare=False)

    def to_wire(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "node_id": self.node_id,
            "custodian_peer_id": int(self.custodian_peer_id),
            "custodian_endpoint": self.custodian_endpoint.to_wire(),
        }
        if self.center is not None:
            value["center"] = np.asarray(self.center, dtype=np.float32)
        return value

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "ChildDescriptor":
        return cls(
            node_id=str(value["node_id"]),
            custodian_peer_id=int(value["custodian_peer_id"]),
            custodian_endpoint=Endpoint.from_wire(value["custodian_endpoint"]),
            center=(
                np.asarray(value["center"], dtype=np.float32)
                if value.get("center") is not None
                else None
            ),
        )


@dataclass
class LeafNode:
    node_id: str
    version: int
    depth: int
    parent: Optional[ChildDescriptor]
    entries: List[CentroidEntry] = field(default_factory=list)
    failed_split_size: Optional[int] = None

    node_type: str = field(default="leaf", init=False)

    def to_wire(self, include_entries: bool = True) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "version": int(self.version),
            "depth": int(self.depth),
            "parent": self.parent.to_wire() if self.parent else None,
            "entry_count": len(self.entries),
            "failed_split_size": self.failed_split_size,
        }
        if include_entries:
            value["entries"] = [entry.to_wire() for entry in self.entries]
        return value

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "LeafNode":
        return cls(
            node_id=str(value["node_id"]),
            version=int(value["version"]),
            depth=int(value["depth"]),
            parent=ChildDescriptor.from_wire(value["parent"]) if value.get("parent") else None,
            entries=[CentroidEntry.from_wire(entry) for entry in value.get("entries", [])],
            failed_split_size=(
                int(value["failed_split_size"])
                if value.get("failed_split_size") is not None
                else None
            ),
        )


@dataclass
class InternalNode:
    node_id: str
    version: int
    depth: int
    parent: Optional[ChildDescriptor]
    left: ChildDescriptor
    right: ChildDescriptor

    node_type: str = field(default="internal", init=False)

    def to_wire(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "version": int(self.version),
            "depth": int(self.depth),
            "parent": self.parent.to_wire() if self.parent else None,
            "left": self.left.to_wire(),
            "right": self.right.to_wire(),
        }

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "InternalNode":
        return cls(
            node_id=str(value["node_id"]),
            version=int(value["version"]),
            depth=int(value["depth"]),
            parent=ChildDescriptor.from_wire(value["parent"]) if value.get("parent") else None,
            left=ChildDescriptor.from_wire(value["left"]),
            right=ChildDescriptor.from_wire(value["right"]),
        )


TreeNode = Union[LeafNode, InternalNode]


def node_from_wire(value: Mapping[str, Any]) -> TreeNode:
    if value["node_type"] == "leaf":
        return LeafNode.from_wire(value)
    if value["node_type"] == "internal":
        return InternalNode.from_wire(value)
    raise ValueError(f"Unknown tree node type: {value.get('node_type')!r}")


@dataclass
class Contact:
    peer_id: int
    score: float
    endpoint: Endpoint
    profile_version: int
    learned_at: str

    def to_wire(self) -> Dict[str, Any]:
        return {
            "peer_id": int(self.peer_id),
            "score": float(self.score),
            "endpoint": self.endpoint.to_wire(),
            "profile_version": int(self.profile_version),
            "learned_at": self.learned_at,
        }

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "Contact":
        return cls(
            peer_id=int(value["peer_id"]),
            score=float(value["score"]),
            endpoint=Endpoint.from_wire(value["endpoint"]),
            profile_version=int(value["profile_version"]),
            learned_at=str(value["learned_at"]),
        )


@dataclass
class QueryState:
    query_id: str
    query_vector: np.ndarray
    target_doc_id: str
    results_by_hop: Dict[int, List[tuple[str, float]]] = field(default_factory=dict)
    peer_by_hop: Dict[int, int] = field(default_factory=dict)
    final_results: List[tuple[str, float]] = field(default_factory=list)
    terminal_hop: Optional[int] = None
    terminal_reason: Optional[str] = None

    def to_wire(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "target_doc_id": self.target_doc_id,
            "results_by_hop": {
                str(hop): [[doc_id, float(score)] for doc_id, score in values]
                for hop, values in sorted(self.results_by_hop.items())
            },
            "peer_by_hop": {str(hop): int(peer) for hop, peer in sorted(self.peer_by_hop.items())},
            "final_results": [[doc_id, float(score)] for doc_id, score in self.final_results],
            "terminal_hop": self.terminal_hop,
            "terminal_reason": self.terminal_reason,
        }


def contacts_to_wire(values: Sequence[Sequence[Contact]]) -> List[List[Dict[str, Any]]]:
    return [[contact.to_wire() for contact in centroid] for centroid in values]


def contacts_from_wire(values: Sequence[Sequence[Mapping[str, Any]]]) -> List[List[Contact]]:
    return [[Contact.from_wire(contact) for contact in centroid] for centroid in values]

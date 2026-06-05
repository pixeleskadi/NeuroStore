"""
NeuroStore - Memory Graph Module
Core data structure: a weighted directed graph of MemoryNode objects.
Implements associative recall via graph traversal (BFS/DFS + edge-weight propagation).
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from memory.node import (
    MemoryNode,
    MemoryStatus,
    DORMANCY_THRESHOLD,
    LONG_TERM_THRESHOLD,
    RECOVERY_ACTIVATION_THRESHOLD,
)


# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────
NodeID = str
EdgeWeight = float


class MemoryGraph:
    """
    A weighted directed graph where nodes are MemoryNode instances and
    edges represent associative connections between memories.

    Graph traversal replaces simple key-value lookup, mirroring how the
    hippocampus activates related engrams during recall.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[NodeID, MemoryNode] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Node management
    # ──────────────────────────────────────────────────────────────────────

    def add_node(self, node: MemoryNode) -> None:
        """Add a memory node to the graph."""
        self._nodes[node.id] = node
        self._graph.add_node(
            node.id,
            title=node.title,
            weight=node.weight,
            status=node.status.value,
        )

    def get_node(self, node_id: NodeID) -> Optional[MemoryNode]:
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[MemoryNode]:
        return list(self._nodes.values())

    def get_nodes_by_status(self, status: MemoryStatus) -> List[MemoryNode]:
        return [n for n in self._nodes.values() if n.status == status]

    def find_by_title(self, title: str) -> Optional[MemoryNode]:
        """Case-insensitive title search."""
        title_lower = title.lower()
        for node in self._nodes.values():
            if node.title.lower() == title_lower:
                return node
        return None

    def search(self, query: str) -> List[MemoryNode]:
        """Full-text search across title, content, and tags."""
        q = query.lower()
        results = []
        for node in self._nodes.values():
            if (
                q in node.title.lower()
                or q in node.content.lower()
                or any(q in tag.lower() for tag in node.tags)
            ):
                results.append(node)
        results.sort(key=lambda n: n.weight, reverse=True)
        return results

    # ──────────────────────────────────────────────────────────────────────
    # Edge management
    # ──────────────────────────────────────────────────────────────────────

    def add_edge(
        self,
        source_id: NodeID,
        target_id: NodeID,
        weight: float = 0.5,
        bidirectional: bool = False,
    ) -> None:
        """
        Create a weighted directed association from source → target.
        Weight ∈ [0, 1]: higher = stronger associative link.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            raise ValueError(
                f"Both node IDs must exist. Got: {source_id[:8]}, {target_id[:8]}"
            )
        weight = max(0.0, min(1.0, weight))
        self._graph.add_edge(source_id, target_id, weight=weight)
        if bidirectional:
            self._graph.add_edge(target_id, source_id, weight=weight)

    def get_edge_weight(self, source_id: NodeID, target_id: NodeID) -> float:
        """Returns edge weight, or 0.0 if edge does not exist."""
        if self._graph.has_edge(source_id, target_id):
            return self._graph[source_id][target_id].get("weight", 0.0)
        return 0.0

    def update_edge_weight(
        self, source_id: NodeID, target_id: NodeID, delta: float
    ) -> None:
        """Adjust edge weight by delta, clamped to [0, 1]."""
        if self._graph.has_edge(source_id, target_id):
            current = self._graph[source_id][target_id].get("weight", 0.5)
            self._graph[source_id][target_id]["weight"] = max(
                0.0, min(1.0, current + delta)
            )

    def weaken_edges(self, node_id: NodeID, factor: float = 0.3) -> None:
        """
        Reduce all outgoing and incoming edges for a (dormant) node.
        Mimics synaptic weakening when a memory becomes inactive.
        """
        for _, tgt in self._graph.out_edges(node_id):
            self.update_edge_weight(node_id, tgt, -factor)
        for src, _ in self._graph.in_edges(node_id):
            self.update_edge_weight(src, node_id, -factor)

    def get_neighbors(self, node_id: NodeID) -> List[Tuple[NodeID, float]]:
        """Return list of (neighbor_id, edge_weight) for all outgoing edges."""
        return [
            (tgt, self._graph[node_id][tgt].get("weight", 0.0))
            for tgt in self._graph.successors(node_id)
        ]

    def get_predecessors(self, node_id: NodeID) -> List[Tuple[NodeID, float]]:
        """Return list of (predecessor_id, edge_weight) for all incoming edges."""
        return [
            (src, self._graph[src][node_id].get("weight", 0.0))
            for src in self._graph.predecessors(node_id)
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Associative recall (graph traversal)
    # ──────────────────────────────────────────────────────────────────────

    def recall(
        self,
        node_id: NodeID,
        depth: int = 2,
        activation_decay: float = 0.6,
    ) -> Dict[NodeID, float]:
        """
        Recall a memory and propagate activation to neighbors.

        Uses a weighted BFS where activation diminishes with traversal depth
        and edge weight. Returns a mapping of {node_id: activation_received}.

        Parameters
        ----------
        node_id          : Starting node for recall.
        depth            : How many hops to propagate activation.
        activation_decay : Multiplier per hop (0 < decay ≤ 1).
        """
        if node_id not in self._nodes:
            return {}

        # Reinforce the directly recalled node
        origin = self._nodes[node_id]
        origin.reinforce()
        self._sync_graph_attrs(node_id)

        # BFS propagation
        activation_map: Dict[NodeID, float] = {node_id: 1.0}
        queue: deque[Tuple[NodeID, float, int]] = deque()
        queue.append((node_id, 1.0, 0))
        visited: Set[NodeID] = {node_id}

        recovered_ids: List[NodeID] = []

        while queue:
            current_id, current_activation, current_depth = queue.popleft()
            if current_depth >= depth:
                continue

            for neighbor_id, edge_w in self.get_neighbors(current_id):
                propagated = current_activation * edge_w * activation_decay
                if propagated < 0.01:
                    continue

                neighbor = self._nodes.get(neighbor_id)
                if neighbor is None:
                    continue

                # Accumulate activation
                activation_map[neighbor_id] = (
                    activation_map.get(neighbor_id, 0.0) + propagated
                )

                # Apply associative reinforcement
                neighbor.associative_reinforce(edge_w * activation_decay)
                self._sync_graph_attrs(neighbor_id)

                # Check for dormant recovery
                if (
                    neighbor.is_dormant
                    and activation_map[neighbor_id] >= RECOVERY_ACTIVATION_THRESHOLD / 100
                ):
                    if neighbor.recover():
                        recovered_ids.append(neighbor_id)

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, propagated, current_depth + 1))

        return activation_map

    def compute_total_activation(self, node_id: NodeID) -> float:
        """
        Sum of weighted activations reaching node_id from all predecessors.
        Used to evaluate recovery potential of dormant nodes.
        """
        total = 0.0
        for pred_id, edge_w in self.get_predecessors(node_id):
            pred = self._nodes.get(pred_id)
            if pred and pred.is_active:
                total += pred.weight * edge_w / 100.0
        return total

    # ──────────────────────────────────────────────────────────────────────
    # Shortcut formation (dormant node bypass)
    # ──────────────────────────────────────────────────────────────────────

    def form_shortcuts(self, dormant_id: NodeID) -> List[Tuple[NodeID, NodeID, float]]:
        """
        When node B is dormant on a path A → B → C,
        create shortcut A → C with reduced confidence.
        Returns list of (source, target, new_weight) tuples created.
        """
        shortcuts_created = []
        node = self._nodes.get(dormant_id)
        if node is None or not node.is_dormant:
            return shortcuts_created

        predecessors = self.get_predecessors(dormant_id)
        successors = self.get_neighbors(dormant_id)

        for pred_id, pw in predecessors:
            for succ_id, sw in successors:
                if pred_id == succ_id:
                    continue
                shortcut_weight = pw * sw * 0.6  # reduced confidence
                if shortcut_weight < 0.05:
                    continue
                if not self._graph.has_edge(pred_id, succ_id):
                    self._graph.add_edge(pred_id, succ_id, weight=shortcut_weight)
                    shortcuts_created.append((pred_id, succ_id, shortcut_weight))

        return shortcuts_created

    # ──────────────────────────────────────────────────────────────────────
    # Path finding
    # ──────────────────────────────────────────────────────────────────────

    def find_path(
        self, source_id: NodeID, target_id: NodeID
    ) -> Optional[List[NodeID]]:
        """
        Find the highest-weight path between two nodes using Dijkstra
        (inverted weights → shortest path = strongest associative path).
        """
        try:
            path = nx.dijkstra_path(
                self._graph,
                source_id,
                target_id,
                weight=lambda u, v, d: 1.0 - d.get("weight", 0.5),
            )
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def strongly_connected_clusters(self) -> List[List[NodeID]]:
        """Return groups of memory nodes with strong mutual associations."""
        communities = []
        undirected = self._graph.to_undirected()
        for component in nx.connected_components(undirected):
            if len(component) > 1:
                communities.append(list(component))
        return communities

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────

    def _sync_graph_attrs(self, node_id: NodeID) -> None:
        """Keep NetworkX node attributes in sync with MemoryNode state."""
        node = self._nodes.get(node_id)
        if node and node_id in self._graph:
            self._graph.nodes[node_id]["weight"] = node.weight
            self._graph.nodes[node_id]["status"] = node.status.value
            self._graph.nodes[node_id]["title"] = node.title

    def sync_all(self) -> None:
        """Sync all nodes."""
        for nid in self._nodes:
            self._sync_graph_attrs(nid)

    @property
    def nx_graph(self) -> nx.DiGraph:
        """Expose the underlying NetworkX graph (read-mostly for visualization)."""
        self.sync_all()
        return self._graph

    def stats(self) -> dict:
        n = self._nodes.values()
        active = [x for x in n if x.is_active]
        dormant = [x for x in n if x.is_dormant]
        lt = [x for x in n if x.is_long_term]
        weights = [x.weight for x in n]
        return {
            "total": len(self._nodes),
            "active": len(active),
            "dormant": len(dormant),
            "long_term": len(lt),
            "edges": self._graph.number_of_edges(),
            "avg_weight": round(sum(weights) / len(weights), 2) if weights else 0.0,
            "max_weight": round(max(weights), 2) if weights else 0.0,
            "min_weight": round(min(weights), 2) if weights else 0.0,
        }

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"MemoryGraph(nodes={s['total']}, edges={s['edges']}, "
            f"active={s['active']}, dormant={s['dormant']}, "
            f"long_term={s['long_term']})"
        )

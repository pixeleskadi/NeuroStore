"""
NeuroStore - Memory Recovery Module
Handles the reactivation of dormant memories through associative pathways.
Biologically analogous to context-dependent memory retrieval (Tulving's encoding specificity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from memory.node import (
    MemoryNode,
    MemoryStatus,
    DORMANCY_THRESHOLD,
    RECOVERY_ACTIVATION_THRESHOLD,
)


@dataclass
class RecoveryEvent:
    """Records a single memory recovery."""
    node_id: str
    title: str
    trigger_ids: List[str]       # Nodes that caused the recovery
    activation_sum: float
    weight_before: float
    weight_after: float
    day: int = 0

    def __repr__(self) -> str:
        return (
            f"RecoveryEvent({self.title!r}, "
            f"activation={self.activation_sum:.3f}, "
            f"weight={self.weight_before:.1f}→{self.weight_after:.1f})"
        )


class RecoveryEngine:
    """
    Scans the graph for dormant nodes that can be recovered
    based on accumulated activation from active/long-term neighbors.

    Recovery mechanism:
    - A dormant node accumulates activation from predecessors.
    - When total activation exceeds RECOVERY_ACTIVATION_THRESHOLD, it wakes.
    - On recovery, original pathway edge weights are partially restored.
    """

    def __init__(self, activation_threshold: float = RECOVERY_ACTIVATION_THRESHOLD) -> None:
        self.activation_threshold = activation_threshold
        self._recovery_log: List[RecoveryEvent] = []

    def compute_activation(
        self,
        dormant_node: MemoryNode,
        predecessors: List[Tuple[str, float]],
        nodes_by_id: Dict[str, MemoryNode],
    ) -> Tuple[float, List[str]]:
        """
        Calculate total activation reaching a dormant node.
        Only active/long-term predecessors contribute.

        Returns (total_activation, list_of_contributing_node_ids).
        """
        total = 0.0
        contributors = []
        for pred_id, edge_w in predecessors:
            pred = nodes_by_id.get(pred_id)
            if pred and not pred.is_dormant:
                contribution = (pred.weight / 100.0) * edge_w
                total += contribution
                contributors.append(pred_id)
        return total, contributors

    def scan_and_recover(
        self,
        all_nodes: Dict[str, MemoryNode],
        predecessors_fn,  # Callable[[str], List[Tuple[str, float]]]
        restore_edges_fn,  # Callable[[str, float], None]
        current_day: int = 0,
    ) -> List[RecoveryEvent]:
        """
        Scan all dormant nodes and recover those with sufficient activation.

        Parameters
        ----------
        all_nodes        : Dict mapping node_id → MemoryNode.
        predecessors_fn  : Function(node_id) → [(pred_id, edge_weight), ...]
        restore_edges_fn : Function(node_id, boost_factor) to restore edges.
        current_day      : Simulated day number for logging.
        """
        events = []
        dormant_nodes = [n for n in all_nodes.values() if n.is_dormant]

        for node in dormant_nodes:
            predecessors = predecessors_fn(node.id)
            activation, contributors = self.compute_activation(
                node, predecessors, all_nodes
            )

            if activation >= self.activation_threshold:
                weight_before = node.weight
                node.recover()
                weight_after = node.weight

                # Restore edge weights partially
                restore_edges_fn(node.id, activation)

                event = RecoveryEvent(
                    node_id=node.id,
                    title=node.title,
                    trigger_ids=contributors,
                    activation_sum=activation,
                    weight_before=weight_before,
                    weight_after=weight_after,
                    day=current_day,
                )
                events.append(event)
                self._recovery_log.append(event)

        return events

    def attempt_targeted_recovery(
        self,
        node: MemoryNode,
        activation_boost: float,
        restore_edges_fn=None,
    ) -> Optional[RecoveryEvent]:
        """
        Explicitly attempt to recover a specific dormant node
        (e.g., user remembers a forgotten memory via cue).
        """
        if not node.is_dormant:
            return None

        weight_before = node.weight
        effective_activation = activation_boost

        if effective_activation < self.activation_threshold * 0.5:
            return None  # Not enough activation

        node.recover()

        if restore_edges_fn:
            restore_edges_fn(node.id, effective_activation)

        event = RecoveryEvent(
            node_id=node.id,
            title=node.title,
            trigger_ids=["manual"],
            activation_sum=effective_activation,
            weight_before=weight_before,
            weight_after=node.weight,
        )
        self._recovery_log.append(event)
        return event

    def get_recovery_log(self) -> List[RecoveryEvent]:
        return list(self._recovery_log)

    def total_recoveries(self) -> int:
        return len(self._recovery_log)

    def recovery_rate(self, total_days: int) -> float:
        """Average recoveries per simulated day."""
        if total_days <= 0:
            return 0.0
        return len(self._recovery_log) / total_days


class PathRestorer:
    """
    Restores original pathway weights when a dormant node recovers.
    Simulates reconsolidation of weakened synaptic connections.
    """

    def __init__(self, graph) -> None:
        self.graph = graph  # MemoryGraph reference

    def restore(self, recovered_node_id: str, activation_factor: float) -> None:
        """
        Partially restore edge weights to/from a recovered node.
        activation_factor scales how much of original weight is recovered.
        """
        boost = min(0.3, activation_factor * 0.2)

        # Restore outgoing edges
        for tgt_id, current_w in self.graph.get_neighbors(recovered_node_id):
            self.graph.update_edge_weight(recovered_node_id, tgt_id, boost)

        # Restore incoming edges
        for src_id, current_w in self.graph.get_predecessors(recovered_node_id):
            self.graph.update_edge_weight(src_id, recovered_node_id, boost)

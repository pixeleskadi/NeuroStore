"""
NeuroStore - Memory Reinforcement Module
Implements rehearsal, spaced repetition, and associative reinforcement logic.
Mirrors the neuroscience of long-term potentiation (LTP).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from memory.node import (
    MemoryNode,
    MemoryStatus,
    REINFORCEMENT_FACTOR,
    ASSOCIATIVE_REINFORCEMENT_FACTOR,
    LONG_TERM_THRESHOLD,
    MAX_WEIGHT,
)


# ──────────────────────────────────────────────────────────────────────────────
# Spaced repetition helpers
# ──────────────────────────────────────────────────────────────────────────────


def optimal_interval(recall_count: int, ease_factor: float = 2.5) -> float:
    """
    SM-2-inspired optimal inter-repetition interval (in simulated days).
    More recalls → longer intervals before review is needed.
    """
    if recall_count <= 1:
        return 1.0
    elif recall_count == 2:
        return 6.0
    else:
        return optimal_interval(recall_count - 1, ease_factor) * ease_factor


def reinforcement_gain(base_factor: float, recall_count: int) -> float:
    """
    Diminishing returns on reinforcement: the more a memory has been recalled,
    the smaller the marginal gain per additional recall (returns approach zero).
    """
    return base_factor * math.exp(-0.05 * recall_count)


# ──────────────────────────────────────────────────────────────────────────────
# Reinforcement Engine
# ──────────────────────────────────────────────────────────────────────────────


class ReinforcementEngine:
    """
    Manages explicit and associative reinforcement of memory nodes.

    Explicit recall  : User directly recalls a memory → large weight boost.
    Associative      : Nearby nodes receive fractional boost via edge weights.
    Spaced repetition: Tracks optimal review schedules per node.
    """

    def __init__(
        self,
        base_factor: float = REINFORCEMENT_FACTOR,
        associative_factor: float = ASSOCIATIVE_REINFORCEMENT_FACTOR,
        diminishing_returns: bool = True,
    ) -> None:
        self.base_factor = base_factor
        self.associative_factor = associative_factor
        self.diminishing_returns = diminishing_returns
        self._reinforcement_log: List[ReinforcementEvent] = []

    # ──────────────────────────────────────────────────────────────────────
    # Core reinforcement
    # ──────────────────────────────────────────────────────────────────────

    def reinforce(self, node: MemoryNode, explicit: bool = True) -> ReinforcementEvent:
        """
        Apply reinforcement to a single node.

        Parameters
        ----------
        node     : Target MemoryNode.
        explicit : True for direct recall; False for associative activation.
        """
        prev_weight = node.weight
        prev_status = node.status

        if explicit:
            factor = (
                reinforcement_gain(self.base_factor, node.recall_count)
                if self.diminishing_returns
                else self.base_factor
            )
        else:
            factor = self.associative_factor

        new_weight = node.reinforce(factor=factor)
        event = ReinforcementEvent(
            node_id=node.id,
            title=node.title,
            prev_weight=prev_weight,
            new_weight=new_weight,
            prev_status=prev_status,
            new_status=node.status,
            explicit=explicit,
        )
        self._reinforcement_log.append(event)
        return event

    def bulk_reinforce(
        self,
        nodes: List[MemoryNode],
        explicit: bool = False,
    ) -> List[ReinforcementEvent]:
        """Apply reinforcement to multiple nodes (e.g., associative cascade)."""
        return [self.reinforce(n, explicit=explicit) for n in nodes]

    def reinforce_with_edges(
        self,
        node: MemoryNode,
        neighbors: List[tuple],  # List of (MemoryNode, edge_weight)
        depth_factor: float = 0.6,
    ) -> List[ReinforcementEvent]:
        """
        Reinforce a node and propagate weighted activation to its neighbors.

        Parameters
        ----------
        node         : Directly recalled node.
        neighbors    : List of (MemoryNode, edge_weight) tuples.
        depth_factor : Attenuation per hop.
        """
        events = [self.reinforce(node, explicit=True)]
        for neighbor, edge_w in neighbors:
            scaled = self.associative_factor * edge_w * depth_factor
            prev = neighbor.weight
            neighbor.reinforce(factor=scaled)
            events.append(
                ReinforcementEvent(
                    node_id=neighbor.id,
                    title=neighbor.title,
                    prev_weight=prev,
                    new_weight=neighbor.weight,
                    prev_status=neighbor.status,
                    new_status=neighbor.status,
                    explicit=False,
                )
            )
        return events

    # ──────────────────────────────────────────────────────────────────────
    # Edge reinforcement
    # ──────────────────────────────────────────────────────────────────────

    def strengthen_edge(
        self,
        source: MemoryNode,
        target: MemoryNode,
        current_weight: float,
        delta: float = 0.05,
    ) -> float:
        """
        Hebbian learning: "nodes that fire together, wire together."
        When two memories are co-activated, their edge weight increases.
        Returns new edge weight.
        """
        return min(1.0, current_weight + delta * (source.weight / MAX_WEIGHT))

    # ──────────────────────────────────────────────────────────────────────
    # Spaced repetition schedule
    # ──────────────────────────────────────────────────────────────────────

    def next_review_day(self, node: MemoryNode, current_day: int) -> int:
        """Calculate the optimal day for next review of this memory."""
        interval = optimal_interval(node.recall_count)
        return current_day + int(math.ceil(interval))

    def due_for_review(
        self, nodes: List[MemoryNode], current_day: int, last_reviewed: Dict[str, int]
    ) -> List[MemoryNode]:
        """Return nodes that are due for review on current_day."""
        due = []
        for node in nodes:
            last = last_reviewed.get(node.id, 0)
            interval = optimal_interval(node.recall_count)
            if (current_day - last) >= interval:
                due.append(node)
        return due

    # ──────────────────────────────────────────────────────────────────────
    # Analytics
    # ──────────────────────────────────────────────────────────────────────

    def get_log(self) -> List["ReinforcementEvent"]:
        return list(self._reinforcement_log)

    def transitions_to_long_term(self) -> List["ReinforcementEvent"]:
        return [
            e
            for e in self._reinforcement_log
            if e.prev_status != MemoryStatus.LONG_TERM
            and e.new_status == MemoryStatus.LONG_TERM
        ]

    def recoveries(self) -> List["ReinforcementEvent"]:
        return [
            e
            for e in self._reinforcement_log
            if e.prev_status == MemoryStatus.DORMANT
            and e.new_status == MemoryStatus.ACTIVE
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Event record
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ReinforcementEvent:
    node_id: str
    title: str
    prev_weight: float
    new_weight: float
    prev_status: MemoryStatus
    new_status: MemoryStatus
    explicit: bool

    @property
    def weight_gain(self) -> float:
        return self.new_weight - self.prev_weight

    @property
    def became_long_term(self) -> bool:
        return (
            self.prev_status != MemoryStatus.LONG_TERM
            and self.new_status == MemoryStatus.LONG_TERM
        )

    @property
    def recovered(self) -> bool:
        return (
            self.prev_status == MemoryStatus.DORMANT
            and self.new_status == MemoryStatus.ACTIVE
        )

    def __repr__(self) -> str:
        kind = "EXPLICIT" if self.explicit else "ASSOC"
        return (
            f"ReinforcementEvent({kind}, {self.title!r}, "
            f"{self.prev_weight:.1f} → {self.new_weight:.1f}, "
            f"{self.prev_status.value} → {self.new_status.value})"
        )

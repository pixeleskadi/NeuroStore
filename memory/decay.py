"""
NeuroStore - Memory Decay Engine
Implements the biologically-inspired Ebbinghaus forgetting curve.
Applies time-based weight reduction to all active memory nodes.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from memory.node import MemoryNode, MemoryStatus, DORMANCY_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# Decay models
# ──────────────────────────────────────────────────────────────────────────────


def linear_decay(weight: float, rate: float, days: float) -> float:
    """Simple linear decay: weight -= rate * days."""
    return max(0.0, weight - rate * days)


def exponential_decay(weight: float, rate: float, days: float) -> float:
    """
    Ebbinghaus-style exponential decay.
    R = e^(-t/s)  where s = stability ∝ 1/rate
    """
    stability = 1.0 / max(rate, 1e-6)
    retention = math.exp(-days / stability)
    return max(0.0, weight * retention)


def power_decay(weight: float, rate: float, days: float) -> float:
    """
    Power-law forgetting: weight * (days + 1)^(-rate)
    Slower initial decay, faster long-run decay.
    """
    return max(0.0, weight * ((days + 1) ** (-rate)))


# Default decay model used by the engine
DECAY_MODEL = exponential_decay


# ──────────────────────────────────────────────────────────────────────────────
# Decay Engine
# ──────────────────────────────────────────────────────────────────────────────


class DecayEngine:
    """
    Applies memory decay across all nodes in a MemoryGraph.

    Decay rules:
    - LONG_TERM nodes: decay rate reduced to 5% of normal.
    - DORMANT nodes: continue accumulating decay (can reach weight = 0).
    - ACTIVE nodes: standard decay; may transition to DORMANT.
    """

    def __init__(self, model: str = "exponential") -> None:
        """
        Parameters
        ----------
        model : One of 'exponential', 'linear', 'power'.
        """
        models = {
            "exponential": exponential_decay,
            "linear": linear_decay,
            "power": power_decay,
        }
        if model not in models:
            raise ValueError(f"Unknown decay model '{model}'. Choose from: {list(models)}")
        self._model = models[model]
        self.model_name = model

    def apply_decay(
        self,
        nodes: List[MemoryNode],
        days: float = 1.0,
    ) -> DecayReport:
        """
        Apply decay to every node for the given number of simulated days.

        Returns a DecayReport summarising what changed.
        """
        report = DecayReport(days=days, model=self.model_name)

        for node in nodes:
            prev_weight = node.weight
            prev_status = node.status

            if node.status == MemoryStatus.LONG_TERM:
                # Long-term memories decay at 5% of normal rate
                effective_rate = node.decay_rate * 0.05
            else:
                effective_rate = node.decay_rate

            new_weight = self._model(node.weight, effective_rate, days)
            delta = prev_weight - new_weight
            node.weight = new_weight

            # Recompute status
            node._update_status()

            report.record(node, prev_weight, delta, prev_status)

        return report

    def apply_decay_days(
        self,
        nodes: List[MemoryNode],
        total_days: int,
        step_days: float = 1.0,
    ) -> List[DecayReport]:
        """
        Advance time by total_days, one step at a time.
        Returns a list of DecayReports, one per step.
        """
        reports = []
        steps = int(total_days / step_days)
        for _ in range(steps):
            reports.append(self.apply_decay(nodes, step_days))
        return reports

    def preview_decay(
        self,
        node: MemoryNode,
        future_days: int = 30,
    ) -> List[Tuple[int, float]]:
        """
        Predict weight trajectory without mutating the node.
        Returns list of (day, projected_weight).
        """
        trajectory = []
        simulated_weight = node.weight
        rate = node.effective_decay_rate
        for day in range(1, future_days + 1):
            simulated_weight = self._model(simulated_weight, rate, 1.0)
            trajectory.append((day, round(simulated_weight, 3)))
        return trajectory

    def days_until_dormant(self, node: MemoryNode) -> float:
        """Estimate days until a node crosses the dormancy threshold."""
        if node.weight <= DORMANCY_THRESHOLD:
            return 0.0
        if node.status == MemoryStatus.LONG_TERM:
            return float("inf")
        rate = node.effective_decay_rate
        w = node.weight
        day = 0
        while w > DORMANCY_THRESHOLD and day < 10_000:
            w = self._model(w, rate, 1.0)
            day += 1
        return float(day) if w <= DORMANCY_THRESHOLD else float("inf")


# ──────────────────────────────────────────────────────────────────────────────
# Decay Report
# ──────────────────────────────────────────────────────────────────────────────


class DecayReport:
    """Immutable summary of a single decay pass."""

    def __init__(self, days: float, model: str) -> None:
        self.days = days
        self.model = model
        self.newly_dormant: List[str] = []
        self.newly_long_term: List[str] = []
        self.total_weight_lost: float = 0.0
        self.nodes_processed: int = 0
        self._details: List[Dict] = []

    def record(
        self,
        node: MemoryNode,
        prev_weight: float,
        weight_delta: float,
        prev_status: MemoryStatus,
    ) -> None:
        self.nodes_processed += 1
        self.total_weight_lost += weight_delta

        if prev_status != MemoryStatus.DORMANT and node.status == MemoryStatus.DORMANT:
            self.newly_dormant.append(node.id)
        if prev_status != MemoryStatus.LONG_TERM and node.status == MemoryStatus.LONG_TERM:
            self.newly_long_term.append(node.id)

        self._details.append(
            {
                "id": node.id,
                "title": node.title,
                "prev_weight": round(prev_weight, 3),
                "new_weight": round(node.weight, 3),
                "delta": round(weight_delta, 3),
                "status": node.status.value,
            }
        )

    def summary(self) -> dict:
        return {
            "days_elapsed": self.days,
            "model": self.model,
            "nodes_processed": self.nodes_processed,
            "total_weight_lost": round(self.total_weight_lost, 3),
            "newly_dormant_count": len(self.newly_dormant),
            "newly_long_term_count": len(self.newly_long_term),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"DecayReport(days={s['days_elapsed']}, "
            f"nodes={s['nodes_processed']}, "
            f"weight_lost={s['total_weight_lost']:.2f}, "
            f"dormant_count={s['newly_dormant_count']})"
        )

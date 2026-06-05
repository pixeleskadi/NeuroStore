"""
NeuroStore - Analytics & Metrics Module
Tracks simulation-wide statistics over time.
Exports snapshots suitable for charting and research reporting.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from memory.node import MemoryNode, MemoryStatus


@dataclass
class DaySnapshot:
    """A single-day metrics record capturing the state of the memory graph."""
    day: int
    active_count: int
    dormant_count: int
    long_term_count: int
    total_nodes: int
    avg_weight: float
    max_weight: float
    min_weight: float
    recovery_count: int        # Cumulative recoveries up to this day
    forgotten_count: int       # Nodes that became dormant today
    newly_long_term: int       # Nodes that entered LTM today
    total_edges: int
    avg_edge_weight: float
    recorded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    # For DB saving (subset of fields used in simulation_metrics table)
    def to_db_dict(self) -> dict:
        return {
            "day": self.day,
            "active_count": self.active_count,
            "dormant_count": self.dormant_count,
            "long_term_count": self.long_term_count,
            "avg_weight": self.avg_weight,
            "recovery_count": self.recovery_count,
            "forgotten_count": self.forgotten_count,
            "total_nodes": self.total_nodes,
            "recorded_at": self.recorded_at,
        }


class MetricsTracker:
    """
    Collects and stores per-day snapshots of memory graph state.
    Provides aggregation, trend analysis, and export utilities.
    """

    def __init__(self) -> None:
        self._snapshots: List[DaySnapshot] = []
        self._cumulative_recoveries: int = 0
        self._cumulative_forgotten: int = 0
        self._cumulative_ltm: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Snapshot capture
    # ──────────────────────────────────────────────────────────────────────

    def record(
        self,
        graph,
        day: int,
        new_recoveries: int = 0,
        new_forgotten: int = 0,
        new_ltm: int = 0,
    ) -> DaySnapshot:
        """
        Capture a snapshot of the graph's current state.

        Parameters
        ----------
        graph          : MemoryGraph instance.
        day            : Current simulated day.
        new_recoveries : Recoveries that occurred this step.
        new_forgotten  : Nodes that became DORMANT this step.
        new_ltm        : Nodes that became LONG_TERM this step.
        """
        self._cumulative_recoveries += new_recoveries
        self._cumulative_forgotten += new_forgotten
        self._cumulative_ltm += new_ltm

        all_nodes = graph.get_all_nodes()
        weights = [n.weight for n in all_nodes] or [0.0]

        nx_g = graph.nx_graph
        edge_weights = [
            d.get("weight", 0.0)
            for _, _, d in nx_g.edges(data=True)
        ] or [0.0]

        snap = DaySnapshot(
            day=day,
            active_count=len([n for n in all_nodes if n.is_active]),
            dormant_count=len([n for n in all_nodes if n.is_dormant]),
            long_term_count=len([n for n in all_nodes if n.is_long_term]),
            total_nodes=len(all_nodes),
            avg_weight=round(sum(weights) / len(weights), 3),
            max_weight=round(max(weights), 3),
            min_weight=round(min(weights), 3),
            recovery_count=self._cumulative_recoveries,
            forgotten_count=self._cumulative_forgotten,
            newly_long_term=new_ltm,
            total_edges=nx_g.number_of_edges(),
            avg_edge_weight=round(sum(edge_weights) / len(edge_weights), 3),
        )
        self._snapshots.append(snap)
        return snap

    # ──────────────────────────────────────────────────────────────────────
    # Query / aggregation
    # ──────────────────────────────────────────────────────────────────────

    def get_all(self) -> List[DaySnapshot]:
        return list(self._snapshots)

    def get_day(self, day: int) -> Optional[DaySnapshot]:
        for s in self._snapshots:
            if s.day == day:
                return s
        return None

    def get_range(self, start_day: int, end_day: int) -> List[DaySnapshot]:
        return [s for s in self._snapshots if start_day <= s.day <= end_day]

    def series(self, field_name: str) -> List[tuple]:
        """Return [(day, value), ...] for the given field."""
        return [
            (s.day, getattr(s, field_name))
            for s in self._snapshots
            if hasattr(s, field_name)
        ]

    def summary(self) -> dict:
        """High-level summary across the entire simulation run."""
        if not self._snapshots:
            return {}
        first = self._snapshots[0]
        last = self._snapshots[-1]
        avg_weights = [s.avg_weight for s in self._snapshots]
        return {
            "simulation_days": last.day - first.day,
            "total_nodes": last.total_nodes,
            "final_active": last.active_count,
            "final_dormant": last.dormant_count,
            "final_long_term": last.long_term_count,
            "initial_avg_weight": round(first.avg_weight, 2),
            "final_avg_weight": round(last.avg_weight, 2),
            "avg_weight_overall": round(sum(avg_weights) / len(avg_weights), 2),
            "peak_avg_weight": round(max(avg_weights), 2),
            "total_recoveries": self._cumulative_recoveries,
            "total_forgotten": self._cumulative_forgotten,
            "total_ltm_promotions": self._cumulative_ltm,
            "final_edges": last.total_edges,
        }

    def retrieval_efficiency(self) -> float:
        """
        Ratio of active+LTM nodes to total nodes at end of simulation.
        Higher = more memories retained.
        """
        if not self._snapshots:
            return 0.0
        last = self._snapshots[-1]
        if last.total_nodes == 0:
            return 0.0
        return round(
            (last.active_count + last.long_term_count) / last.total_nodes, 4
        )

    # ──────────────────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────────────────

    def export_json(self, path: Path) -> None:
        """Export all snapshots to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "summary": self.summary(),
                    "snapshots": [s.to_dict() for s in self._snapshots],
                },
                f,
                indent=2,
            )

    def export_csv(self, path: Path) -> None:
        """Export all snapshots to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._snapshots:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._snapshots[0].to_dict().keys())
            writer.writeheader()
            writer.writerows(s.to_dict() for s in self._snapshots)

    def load_from_db(self, db_rows: List[dict]) -> None:
        """Reconstruct tracker from DB rows (simulation_metrics table)."""
        self._snapshots = []
        for row in db_rows:
            snap = DaySnapshot(
                day=row["day"],
                active_count=row["active_count"],
                dormant_count=row["dormant_count"],
                long_term_count=row["long_term_count"],
                total_nodes=row["total_nodes"],
                avg_weight=row["avg_weight"],
                max_weight=0.0,
                min_weight=0.0,
                recovery_count=row["recovery_count"],
                forgotten_count=row["forgotten_count"],
                newly_long_term=0,
                total_edges=0,
                avg_edge_weight=0.0,
                recorded_at=row.get("recorded_at", ""),
            )
            self._snapshots.append(snap)
        if self._snapshots:
            self._cumulative_recoveries = self._snapshots[-1].recovery_count
            self._cumulative_forgotten = self._snapshots[-1].forgotten_count

    def __repr__(self) -> str:
        return f"MetricsTracker(snapshots={len(self._snapshots)})"

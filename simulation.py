"""
NeuroStore - Simulation Engine
Orchestrates multi-day memory simulations: decay, recall, reinforcement,
forgetting, recovery, and LTM consolidation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from memory.graph import MemoryGraph
from memory.node import MemoryNode, MemoryStatus, DORMANCY_THRESHOLD
from memory.decay import DecayEngine
from memory.reinforcement import ReinforcementEngine
from memory.recovery import RecoveryEngine, PathRestorer
from memory.highway import LongTermHighway
from analytics.metrics import MetricsTracker
from database.db import NeuroStoreDB


@dataclass
class SimulationConfig:
    """Tunable parameters for a simulation run."""
    total_days: int = 30
    decay_model: str = "exponential"
    recall_probability: float = 0.3       # Chance any node is recalled on a given day
    reinforce_probability: float = 0.15   # Extra reinforcement chance (rehearsal)
    recovery_scan_interval: int = 5       # Scan for recoverable nodes every N days
    shortcut_scan_interval: int = 10      # Form shortcuts every N days
    ltm_sync_interval: int = 1            # Sync LTM highway every N days
    record_interval: int = 1              # Record metrics every N days
    random_seed: Optional[int] = 42
    verbose: bool = True


@dataclass
class SimulationResult:
    """Summary of a completed simulation run."""
    total_days: int
    config: SimulationConfig
    metrics_summary: dict
    retrieval_efficiency: float
    final_graph_stats: dict
    ltm_stats: dict
    charts_generated: List[str] = field(default_factory=list)

    def print_report(self) -> None:
        sep = "─" * 60
        print(f"\n{sep}")
        print("  NeuroStore Simulation Report")
        print(sep)
        print(f"  Duration          : {self.total_days} simulated days")
        print(f"  Decay model       : {self.config.decay_model}")
        print(f"  Recall probability: {self.config.recall_probability:.0%}")
        print(sep)
        m = self.metrics_summary
        print(f"  Total nodes       : {m.get('total_nodes', '—')}")
        print(f"  Final active      : {m.get('final_active', '—')}")
        print(f"  Final dormant     : {m.get('final_dormant', '—')}")
        print(f"  Final LTM         : {m.get('final_long_term', '—')}")
        print(f"  Avg weight (end)  : {m.get('final_avg_weight', '—')}")
        print(f"  Total recoveries  : {m.get('total_recoveries', '—')}")
        print(f"  Total forgotten   : {m.get('total_forgotten', '—')}")
        print(f"  LTM promotions    : {m.get('total_ltm_promotions', '—')}")
        print(f"  Retrieval efficiency: {self.retrieval_efficiency:.1%}")
        print(sep)
        ltm = self.ltm_stats
        print(f"  LTM Highway nodes : {ltm.get('count', 0)}")
        print(f"  LTM interconnects : {ltm.get('total_interconnections', 0)}")
        print(sep)
        if self.charts_generated:
            print("  Charts generated  :")
            for c in self.charts_generated:
                print(f"    • {c}")
        print(sep + "\n")


class SimulationEngine:
    """
    Drives a time-stepped memory simulation over a MemoryGraph.

    Each simulated day:
      1. Apply exponential decay to all nodes.
      2. Randomly recall a subset of active nodes.
      3. Propagate associative activation to neighbors.
      4. Scan for dormant recovery (every N days).
      5. Form memory shortcuts for dormant nodes (every N days).
      6. Sync LTM highway.
      7. Record metrics snapshot.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        db: Optional[NeuroStoreDB] = None,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        self.graph = graph
        self.db = db
        self.config = config or SimulationConfig()

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

        self.decay_engine      = DecayEngine(model=self.config.decay_model)
        self.reinforce_engine  = ReinforcementEngine()
        self.recovery_engine   = RecoveryEngine()
        self.highway           = LongTermHighway()
        self.tracker           = MetricsTracker()
        self.path_restorer     = PathRestorer(graph)

        self._day: int = 0
        self._shortcut_log: List[dict] = []

    # ──────────────────────────────────────────────────────────────────────
    # Main run loop
    # ──────────────────────────────────────────────────────────────────────

    def run(self, generate_charts: bool = True) -> SimulationResult:
        cfg = self.config
        if cfg.verbose:
            print(f"\n[NeuroStore] Starting {cfg.total_days}-day simulation "
                  f"(decay={cfg.decay_model}, seed={cfg.random_seed})")
            print(f"[NeuroStore] Graph: {self.graph}\n")

        for day in range(1, cfg.total_days + 1):
            self._day = day
            new_forgotten = 0
            new_recoveries = 0
            new_ltm = 0

            # 1. Decay ─────────────────────────────────────────────────────
            nodes = self.graph.get_all_nodes()
            decay_report = self.decay_engine.apply_decay(nodes, days=1.0)
            new_forgotten += len(decay_report.newly_dormant)

            # Weaken edges for newly dormant nodes
            for dormant_id in decay_report.newly_dormant:
                self.graph.weaken_edges(dormant_id, factor=0.1)

            # 2. Random recall ─────────────────────────────────────────────
            active_nodes = self.graph.get_nodes_by_status(MemoryStatus.ACTIVE)
            lt_nodes     = self.graph.get_nodes_by_status(MemoryStatus.LONG_TERM)
            recall_pool  = active_nodes + lt_nodes

            for node in recall_pool:
                if random.random() < cfg.recall_probability:
                    activation_map = self.graph.recall(node.id, depth=2)

            # 3. Extra reinforcement (rehearsal)────────────────────────────
            for node in recall_pool:
                if random.random() < cfg.reinforce_probability:
                    node.reinforce(factor=5.0)
                    self.graph._sync_graph_attrs(node.id)

            # 4. Recovery scan ─────────────────────────────────────────────
            if day % cfg.recovery_scan_interval == 0:
                events = self.recovery_engine.scan_and_recover(
                    all_nodes={n.id: n for n in self.graph.get_all_nodes()},
                    predecessors_fn=self.graph.get_predecessors,
                    restore_edges_fn=self.path_restorer.restore,
                    current_day=day,
                )
                new_recoveries += len(events)
                if cfg.verbose and events:
                    for e in events:
                        print(f"  [Day {day:4d}] ✦ RECOVERED: {e.title!r} "
                              f"(activation={e.activation_sum:.3f})")

            # 5. Shortcut formation ────────────────────────────────────────
            if day % cfg.shortcut_scan_interval == 0:
                dormant_nodes = self.graph.get_nodes_by_status(MemoryStatus.DORMANT)
                for dn in dormant_nodes:
                    shortcuts = self.graph.form_shortcuts(dn.id)
                    for src, tgt, w in shortcuts:
                        self._shortcut_log.append(
                            {"day": day, "via_dormant": dn.id, "src": src, "tgt": tgt, "weight": w}
                        )

            # 6. LTM sync ──────────────────────────────────────────────────
            if day % cfg.ltm_sync_interval == 0:
                newly_consolidated = self.highway.sync_from_graph(
                    self.graph.get_all_nodes(), day=day
                )
                new_ltm += len(newly_consolidated)
                self.highway.apply_ltm_decay_resistance(self.graph.get_all_nodes())
                if cfg.verbose and newly_consolidated:
                    for nid in newly_consolidated:
                        node = self.graph.get_node(nid)
                        if node:
                            print(f"  [Day {day:4d}] ★ LTM: {node.title!r} "
                                  f"(weight={node.weight:.1f})")

            # 7. Metrics snapshot ──────────────────────────────────────────
            if day % cfg.record_interval == 0:
                snap = self.tracker.record(
                    self.graph, day,
                    new_recoveries=new_recoveries,
                    new_forgotten=new_forgotten,
                    new_ltm=new_ltm,
                )
                # Persist to DB
                if self.db:
                    self.db.save_metric(snap.to_db_dict())

            # Periodic progress
            if cfg.verbose and day % max(1, cfg.total_days // 10) == 0:
                stats = self.graph.stats()
                print(
                    f"  [Day {day:4d}] "
                    f"Active={stats['active']} "
                    f"Dormant={stats['dormant']} "
                    f"LTM={stats['long_term']} "
                    f"AvgW={stats['avg_weight']:.1f}"
                )

        # Persist final graph state
        if self.db:
            self.db.persist_graph(self.graph)

        # Generate charts
        chart_paths = []
        if generate_charts:
            from analytics.charts import generate_all_charts
            paths = generate_all_charts(self.tracker)
            chart_paths = [str(p) for p in paths]
            if cfg.verbose:
                print(f"\n[NeuroStore] {len(chart_paths)} charts saved to outputs/")

        result = SimulationResult(
            total_days=cfg.total_days,
            config=cfg,
            metrics_summary=self.tracker.summary(),
            retrieval_efficiency=self.tracker.retrieval_efficiency(),
            final_graph_stats=self.graph.stats(),
            ltm_stats=self.highway.stats(),
            charts_generated=chart_paths,
        )

        if cfg.verbose:
            result.print_report()

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Manual step utilities
    # ──────────────────────────────────────────────────────────────────────

    def step(self, days: int = 1) -> None:
        """Advance by `days` without generating charts."""
        for _ in range(days):
            self._day += 1
            nodes = self.graph.get_all_nodes()
            self.decay_engine.apply_decay(nodes, days=1.0)

    def trigger_recall(self, node_id: str) -> dict:
        """Manually trigger recall for a specific node."""
        return self.graph.recall(node_id, depth=2)

    def trigger_recovery_scan(self) -> int:
        events = self.recovery_engine.scan_and_recover(
            all_nodes={n.id: n for n in self.graph.get_all_nodes()},
            predecessors_fn=self.graph.get_predecessors,
            restore_edges_fn=self.path_restorer.restore,
            current_day=self._day,
        )
        return len(events)

    @property
    def current_day(self) -> int:
        return self._day

    @property
    def shortcut_log(self) -> List[dict]:
        return list(self._shortcut_log)

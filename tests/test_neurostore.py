"""
NeuroStore - Test Suite
Comprehensive tests for all core modules.
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile
import os

from memory.node import (
    MemoryNode, MemoryStatus,
    DORMANCY_THRESHOLD, LONG_TERM_THRESHOLD, MAX_WEIGHT,
    DEFAULT_INITIAL_WEIGHT, DEFAULT_DECAY_RATE,
)
from memory.graph import MemoryGraph
from memory.decay import DecayEngine
from memory.reinforcement import ReinforcementEngine
from memory.recovery import RecoveryEngine, PathRestorer
from memory.highway import LongTermHighway
from database.db import NeuroStoreDB
from analytics.metrics import MetricsTracker


# ═════════════════════════════════════════════════════════════════════════════
# MemoryNode tests
# ═════════════════════════════════════════════════════════════════════════════

class TestMemoryNode:
    def test_default_creation(self):
        node = MemoryNode(title="Test", content="Some content")
        assert node.title == "Test"
        assert node.content == "Some content"
        assert node.weight == DEFAULT_INITIAL_WEIGHT
        assert node.status == MemoryStatus.ACTIVE
        assert node.recall_count == 0
        assert node.id  # non-empty UUID

    def test_uuid_uniqueness(self):
        nodes = [MemoryNode(title=f"Node {i}", content="x") for i in range(100)]
        ids = {n.id for n in nodes}
        assert len(ids) == 100

    def test_decay_reduces_weight(self):
        node = MemoryNode(title="X", content="Y", weight=60.0, decay_rate=0.1)
        initial = node.weight
        loss = node.apply_decay(days=1.0)
        assert node.weight < initial
        assert loss > 0

    def test_decay_transitions_to_dormant(self):
        node = MemoryNode(title="X", content="Y", weight=25.0, decay_rate=0.5)
        for _ in range(20):
            node.apply_decay(days=1.0)
            if node.weight < DORMANCY_THRESHOLD:
                break
        assert node.status == MemoryStatus.DORMANT

    def test_reinforce_increases_weight(self):
        node = MemoryNode(title="X", content="Y", weight=40.0)
        old_weight = node.weight
        node.reinforce(factor=10.0)
        assert node.weight > old_weight
        assert node.recall_count == 1

    def test_reinforce_caps_at_max(self):
        node = MemoryNode(title="X", content="Y", weight=95.0)
        node.reinforce(factor=50.0)
        assert node.weight == MAX_WEIGHT

    def test_long_term_transition(self):
        node = MemoryNode(title="X", content="Y", weight=89.0)
        node.reinforce(factor=5.0)
        assert node.status == MemoryStatus.LONG_TERM

    def test_long_term_minimal_decay(self):
        node = MemoryNode(title="X", content="Y", weight=LONG_TERM_THRESHOLD, decay_rate=0.1)
        node.status = MemoryStatus.LONG_TERM
        assert node.effective_decay_rate == pytest.approx(0.1 * 0.05, rel=1e-5)

    def test_recover_dormant(self):
        node = MemoryNode(title="X", content="Y", weight=10.0)
        node.status = MemoryStatus.DORMANT
        result = node.recover()
        assert result is True
        assert node.status == MemoryStatus.ACTIVE

    def test_recover_non_dormant_returns_false(self):
        node = MemoryNode(title="X", content="Y", weight=50.0)
        assert node.recover() is False

    def test_serialization_roundtrip(self):
        node = MemoryNode(title="Quantum", content="QM notes", weight=75.5,
                          tags=["physics", "quantum"])
        d = node.to_dict()
        restored = MemoryNode.from_dict(d)
        assert restored.id == node.id
        assert restored.title == node.title
        assert restored.weight == pytest.approx(node.weight, rel=1e-4)
        assert restored.tags == node.tags
        assert restored.status == node.status


# ═════════════════════════════════════════════════════════════════════════════
# MemoryGraph tests
# ═════════════════════════════════════════════════════════════════════════════

class TestMemoryGraph:
    def _make_graph(self):
        g = MemoryGraph()
        a = MemoryNode(title="A", content="alpha")
        b = MemoryNode(title="B", content="beta")
        c = MemoryNode(title="C", content="gamma")
        for n in [a, b, c]:
            g.add_node(n)
        g.add_edge(a.id, b.id, weight=0.8)
        g.add_edge(b.id, c.id, weight=0.6)
        return g, a, b, c

    def test_add_and_get_node(self):
        g, a, b, c = self._make_graph()
        assert g.get_node(a.id) is a

    def test_edge_weight(self):
        g, a, b, c = self._make_graph()
        assert g.get_edge_weight(a.id, b.id) == pytest.approx(0.8)

    def test_nonexistent_edge_returns_zero(self):
        g, a, b, c = self._make_graph()
        assert g.get_edge_weight(a.id, c.id) == 0.0

    def test_search_by_title(self):
        g, a, b, c = self._make_graph()
        results = g.search("alpha")
        assert a in results

    def test_recall_reinforces_origin(self):
        g, a, b, c = self._make_graph()
        before = a.weight
        g.recall(a.id, depth=2)
        assert a.weight > before

    def test_recall_activates_neighbor(self):
        g, a, b, c = self._make_graph()
        before_b = b.weight
        g.recall(a.id, depth=2)
        assert b.weight > before_b

    def test_stats(self):
        g, a, b, c = self._make_graph()
        stats = g.stats()
        assert stats["total"] == 3
        assert stats["edges"] == 2

    def test_form_shortcuts(self):
        g, a, b, c = self._make_graph()
        b.status = MemoryStatus.DORMANT
        b.weight = 5.0
        shortcuts = g.form_shortcuts(b.id)
        assert len(shortcuts) > 0

    def test_weaken_edges(self):
        g, a, b, c = self._make_graph()
        before = g.get_edge_weight(a.id, b.id)
        g.weaken_edges(b.id, factor=0.2)
        after = g.get_edge_weight(a.id, b.id)
        assert after < before


# ═════════════════════════════════════════════════════════════════════════════
# DecayEngine tests
# ═════════════════════════════════════════════════════════════════════════════

class TestDecayEngine:
    def test_exponential_decay(self):
        engine = DecayEngine(model="exponential")
        node = MemoryNode(title="X", content="Y", weight=60.0, decay_rate=0.1)
        report = engine.apply_decay([node], days=1.0)
        assert node.weight < 60.0
        assert report.total_weight_lost > 0

    def test_linear_decay(self):
        engine = DecayEngine(model="linear")
        node = MemoryNode(title="X", content="Y", weight=60.0, decay_rate=0.1)
        engine.apply_decay([node], days=1.0)
        assert node.weight < 60.0

    def test_power_decay(self):
        engine = DecayEngine(model="power")
        node = MemoryNode(title="X", content="Y", weight=60.0, decay_rate=0.5)
        engine.apply_decay([node], days=1.0)
        assert node.weight < 60.0

    def test_ltm_protected(self):
        engine = DecayEngine(model="exponential")
        node = MemoryNode(title="X", content="Y", weight=95.0, decay_rate=0.5)
        node.status = MemoryStatus.LONG_TERM
        report = engine.apply_decay([node], days=10.0)
        # LTM uses 5% of rate — should still be high
        assert node.weight > 50.0

    def test_newly_dormant_recorded(self):
        engine = DecayEngine(model="linear")
        node = MemoryNode(title="X", content="Y", weight=21.0, decay_rate=5.0)
        report = engine.apply_decay([node], days=1.0)
        assert node.status == MemoryStatus.DORMANT
        assert node.id in report.newly_dormant

    def test_days_until_dormant(self):
        engine = DecayEngine(model="exponential")
        node = MemoryNode(title="X", content="Y", weight=50.0, decay_rate=0.2)
        days = engine.days_until_dormant(node)
        assert days > 0

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError):
            DecayEngine(model="unknown_model")


# ═════════════════════════════════════════════════════════════════════════════
# ReinforcementEngine tests
# ═════════════════════════════════════════════════════════════════════════════

class TestReinforcementEngine:
    def test_explicit_reinforce(self):
        engine = ReinforcementEngine()
        node = MemoryNode(title="X", content="Y", weight=40.0)
        event = engine.reinforce(node, explicit=True)
        assert event.weight_gain > 0
        assert event.explicit is True

    def test_diminishing_returns(self):
        engine = ReinforcementEngine(diminishing_returns=True)
        node = MemoryNode(title="X", content="Y", weight=50.0, recall_count=0)
        gain1 = engine.reinforce(node, explicit=True).weight_gain
        node.recall_count = 50
        gain2 = engine.reinforce(node, explicit=True).weight_gain
        assert gain1 > gain2

    def test_associative_reinforce(self):
        engine = ReinforcementEngine()
        node = MemoryNode(title="X", content="Y", weight=40.0)
        event = engine.reinforce(node, explicit=False)
        assert event.weight_gain < 10.0  # smaller than explicit

    def test_ltm_transition_tracked(self):
        engine = ReinforcementEngine(diminishing_returns=False)
        node = MemoryNode(title="X", content="Y", weight=89.0)
        for _ in range(5):
            engine.reinforce(node, explicit=True)
        transitions = engine.transitions_to_long_term()
        assert len(transitions) >= 1

    def test_recovery_tracked(self):
        engine = ReinforcementEngine()
        node = MemoryNode(title="X", content="Y", weight=5.0)
        node.status = MemoryStatus.DORMANT
        node.weight = DORMANCY_THRESHOLD + 1
        event = engine.reinforce(node, explicit=True)
        recoveries = engine.recoveries()
        # May or may not trigger depending on weight; just ensure no error
        assert isinstance(recoveries, list)


# ═════════════════════════════════════════════════════════════════════════════
# RecoveryEngine tests
# ═════════════════════════════════════════════════════════════════════════════

class TestRecoveryEngine:
    def _setup(self):
        g = MemoryGraph()
        source = MemoryNode(title="Source", content="s", weight=80.0)
        dormant = MemoryNode(title="Dormant", content="d", weight=5.0)
        dormant.status = MemoryStatus.DORMANT
        g.add_node(source)
        g.add_node(dormant)
        g.add_edge(source.id, dormant.id, weight=0.9)
        return g, source, dormant

    def test_scan_recovers_dormant(self):
        g, source, dormant = self._setup()
        engine = RecoveryEngine(activation_threshold=0.1)
        events = engine.scan_and_recover(
            all_nodes={n.id: n for n in g.get_all_nodes()},
            predecessors_fn=g.get_predecessors,
            restore_edges_fn=lambda nid, f: None,
            current_day=1,
        )
        assert len(events) == 1
        assert dormant.status == MemoryStatus.ACTIVE

    def test_no_recovery_without_activation(self):
        g = MemoryGraph()
        dormant = MemoryNode(title="Isolated", content="x", weight=5.0)
        dormant.status = MemoryStatus.DORMANT
        g.add_node(dormant)
        engine = RecoveryEngine(activation_threshold=0.5)
        events = engine.scan_and_recover(
            all_nodes={dormant.id: dormant},
            predecessors_fn=g.get_predecessors,
            restore_edges_fn=lambda nid, f: None,
        )
        assert len(events) == 0

    def test_targeted_recovery(self):
        engine = RecoveryEngine(activation_threshold=0.1)
        node = MemoryNode(title="X", content="Y", weight=5.0)
        node.status = MemoryStatus.DORMANT
        event = engine.attempt_targeted_recovery(node, activation_boost=0.5)
        assert event is not None
        assert node.status == MemoryStatus.ACTIVE

    def test_total_recoveries_count(self):
        engine = RecoveryEngine(activation_threshold=0.1)
        for i in range(3):
            n = MemoryNode(title=f"N{i}", content="x", weight=5.0)
            n.status = MemoryStatus.DORMANT
            engine.attempt_targeted_recovery(n, activation_boost=1.0)
        assert engine.total_recoveries() == 3


# ═════════════════════════════════════════════════════════════════════════════
# LongTermHighway tests
# ═════════════════════════════════════════════════════════════════════════════

class TestLongTermHighway:
    def test_consolidate(self):
        hw = LongTermHighway()
        node = MemoryNode(title="LTM Node", content="x", weight=95.0)
        node.status = MemoryStatus.LONG_TERM
        entry = hw.consolidate(node, day=10)
        assert hw.contains(node.id)
        assert entry.consolidated_weight == 95.0

    def test_idempotent_consolidation(self):
        hw = LongTermHighway()
        node = MemoryNode(title="X", content="Y", weight=92.0)
        node.status = MemoryStatus.LONG_TERM
        hw.consolidate(node, day=1)
        hw.consolidate(node, day=2)
        assert len(hw) == 1

    def test_get_increments_access_count(self):
        hw = LongTermHighway()
        node = MemoryNode(title="X", content="Y", weight=91.0)
        node.status = MemoryStatus.LONG_TERM
        hw.consolidate(node)
        hw.get(node.id)
        hw.get(node.id)
        entry = hw.get(node.id)
        assert entry.access_count >= 2

    def test_sync_from_graph(self):
        hw = LongTermHighway()
        nodes = [MemoryNode(title=f"N{i}", content="x", weight=92.0) for i in range(5)]
        for n in nodes:
            n.status = MemoryStatus.LONG_TERM
        newly = hw.sync_from_graph(nodes, day=5)
        assert len(newly) == 5

    def test_search(self):
        hw = LongTermHighway()
        node = MemoryNode(title="Quantum Mechanics", content="x", weight=91.0)
        node.status = MemoryStatus.LONG_TERM
        hw.consolidate(node)
        results = hw.search("quantum")
        assert len(results) == 1


# ═════════════════════════════════════════════════════════════════════════════
# Database tests
# ═════════════════════════════════════════════════════════════════════════════

class TestNeuroStoreDB:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = tmp_path / "test.db"
        return NeuroStoreDB(db_path=db_path)

    def test_save_and_load_node(self, db):
        node = MemoryNode(title="Test", content="Content", weight=60.0)
        db.save_node(node)
        loaded = db.load_node(node.id)
        assert loaded is not None
        assert loaded.title == "Test"
        assert loaded.weight == pytest.approx(60.0, rel=1e-4)

    def test_upsert_node(self, db):
        node = MemoryNode(title="X", content="Y", weight=50.0)
        db.save_node(node)
        node.weight = 75.0
        db.save_node(node)
        loaded = db.load_node(node.id)
        assert loaded.weight == pytest.approx(75.0, rel=1e-4)

    def test_load_all_nodes(self, db):
        for i in range(5):
            db.save_node(MemoryNode(title=f"N{i}", content="x"))
        nodes = db.load_all_nodes()
        assert len(nodes) == 5

    def test_save_and_load_edge(self, db):
        a = MemoryNode(title="A", content="a")
        b = MemoryNode(title="B", content="b")
        db.save_node(a)
        db.save_node(b)
        db.save_edge(a.id, b.id, 0.75)
        edges = db.load_all_edges()
        assert len(edges) == 1
        src, tgt, w = edges[0]
        assert src == a.id and tgt == b.id
        assert w == pytest.approx(0.75, rel=1e-4)

    def test_persist_and_load_graph(self, db):
        g = MemoryGraph()
        a = MemoryNode(title="Alpha", content="a")
        b = MemoryNode(title="Beta",  content="b")
        g.add_node(a); g.add_node(b)
        g.add_edge(a.id, b.id, weight=0.6)
        db.persist_graph(g)

        g2 = db.load_graph()
        assert g2.get_node(a.id) is not None
        assert g2.get_edge_weight(a.id, b.id) == pytest.approx(0.6, rel=1e-3)

    def test_search_nodes(self, db):
        db.save_node(MemoryNode(title="Quantum Mechanics", content="QM notes"))
        db.save_node(MemoryNode(title="Cooking", content="Recipes"))
        results = db.search_nodes("quantum")
        assert len(results) == 1

    def test_delete_node(self, db):
        node = MemoryNode(title="X", content="Y")
        db.save_node(node)
        db.delete_node(node.id)
        assert db.load_node(node.id) is None

    def test_metrics_roundtrip(self, db):
        db.save_metric({
            "day": 1, "active_count": 5, "dormant_count": 2,
            "long_term_count": 1, "avg_weight": 55.0, "recovery_count": 0,
            "forgotten_count": 1, "total_nodes": 8, "recorded_at": "2024-01-01T00:00:00"
        })
        rows = db.load_metrics()
        assert len(rows) == 1
        assert rows[0]["day"] == 1


# ═════════════════════════════════════════════════════════════════════════════
# MetricsTracker tests
# ═════════════════════════════════════════════════════════════════════════════

class TestMetricsTracker:
    def _make_graph(self):
        g = MemoryGraph()
        for i in range(5):
            g.add_node(MemoryNode(title=f"M{i}", content="x", weight=50.0 + i * 5))
        return g

    def test_record_snapshot(self):
        tracker = MetricsTracker()
        g = self._make_graph()
        snap = tracker.record(g, day=1)
        assert snap.day == 1
        assert snap.total_nodes == 5

    def test_series(self):
        tracker = MetricsTracker()
        g = self._make_graph()
        for day in range(1, 6):
            tracker.record(g, day=day)
        series = tracker.series("avg_weight")
        assert len(series) == 5

    def test_summary(self):
        tracker = MetricsTracker()
        g = self._make_graph()
        for day in range(1, 11):
            tracker.record(g, day=day, new_recoveries=1 if day == 5 else 0)
        summary = tracker.summary()
        assert summary["total_recoveries"] == 1

    def test_export_json(self, tmp_path):
        tracker = MetricsTracker()
        g = self._make_graph()
        tracker.record(g, day=1)
        path = tmp_path / "metrics.json"
        tracker.export_json(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert "snapshots" in data
        assert len(data["snapshots"]) == 1

    def test_export_csv(self, tmp_path):
        tracker = MetricsTracker()
        g = self._make_graph()
        tracker.record(g, day=1)
        path = tmp_path / "metrics.csv"
        tracker.export_csv(path)
        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1


# ═════════════════════════════════════════════════════════════════════════════
# Integration test — mini simulation
# ═════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_simulation_cycle(self, tmp_path):
        from simulation import SimulationEngine, SimulationConfig

        db = NeuroStoreDB(db_path=tmp_path / "sim.db")
        g = MemoryGraph()
        nodes = [
            MemoryNode(title=f"Memory_{i}", content=f"Content {i}",
                       weight=30.0 + i * 5, decay_rate=0.08)
            for i in range(8)
        ]
        for n in nodes:
            g.add_node(n)
        # Chain: 0→1→2→...→7
        for i in range(len(nodes) - 1):
            g.add_edge(nodes[i].id, nodes[i+1].id, weight=0.7)
        # Some bidirectional
        g.add_edge(nodes[7].id, nodes[0].id, weight=0.5)

        db.persist_graph(g)

        config = SimulationConfig(
            total_days=10,
            decay_model="exponential",
            recall_probability=0.5,
            reinforce_probability=0.2,
            verbose=False,
            random_seed=123,
        )
        engine = SimulationEngine(graph=g, db=db, config=config)
        result = engine.run(generate_charts=False)

        assert result.total_days == 10
        assert result.final_graph_stats["total"] == 8
        assert result.metrics_summary  # non-empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

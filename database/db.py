"""
NeuroStore - Database Layer
SQLite-backed persistence for MemoryNode objects and graph edges.
All I/O is funnelled through NeuroStoreDB — the rest of the system
works with in-memory MemoryGraph instances loaded from this store.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from memory.node import MemoryNode, MemoryStatus

# Default database path — can be overridden via environment variable
DEFAULT_DB_PATH = Path(os.environ.get("NEUROSTORE_DB", "data/neurostore.db"))


# ──────────────────────────────────────────────────────────────────────────────
# Schema DDL
# ──────────────────────────────────────────────────────────────────────────────

_DDL_NODES = """
CREATE TABLE IF NOT EXISTS memory_nodes (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    content       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    weight        REAL NOT NULL DEFAULT 50.0,
    decay_rate    REAL NOT NULL DEFAULT 0.05,
    status        TEXT NOT NULL DEFAULT 'ACTIVE',
    recall_count  INTEGER NOT NULL DEFAULT 0,
    tags          TEXT NOT NULL DEFAULT ''
);
"""

_DDL_EDGES = """
CREATE TABLE IF NOT EXISTS memory_edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 0.5,
    FOREIGN KEY (source_id) REFERENCES memory_nodes(id),
    FOREIGN KEY (target_id) REFERENCES memory_nodes(id),
    UNIQUE (source_id, target_id)
);
"""

_DDL_METRICS = """
CREATE TABLE IF NOT EXISTS simulation_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    day             INTEGER NOT NULL,
    active_count    INTEGER NOT NULL DEFAULT 0,
    dormant_count   INTEGER NOT NULL DEFAULT 0,
    long_term_count INTEGER NOT NULL DEFAULT 0,
    avg_weight      REAL NOT NULL DEFAULT 0.0,
    recovery_count  INTEGER NOT NULL DEFAULT 0,
    forgotten_count INTEGER NOT NULL DEFAULT 0,
    total_nodes     INTEGER NOT NULL DEFAULT 0,
    recorded_at     TEXT NOT NULL
);
"""

_DDL_LTM = """
CREATE TABLE IF NOT EXISTS ltm_highway (
    node_id               TEXT PRIMARY KEY,
    title                 TEXT NOT NULL,
    consolidated_weight   REAL NOT NULL,
    consolidation_day     INTEGER NOT NULL DEFAULT 0,
    access_count          INTEGER NOT NULL DEFAULT 0,
    interconnections      TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (node_id) REFERENCES memory_nodes(id)
);
"""

_ALL_DDL = [_DDL_NODES, _DDL_EDGES, _DDL_METRICS, _DDL_LTM]


# ──────────────────────────────────────────────────────────────────────────────
# Database class
# ──────────────────────────────────────────────────────────────────────────────


class NeuroStoreDB:
    """
    Thin wrapper around SQLite for NeuroStore persistence.
    All methods are synchronous and safe to call from a single thread.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
        self._init_schema()

    # ──────────────────────────────────────────────────────────────────────
    # Connection management
    # ──────────────────────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL;")
            self._connection.execute("PRAGMA foreign_keys=ON;")
        return self._connection

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Cursor, None, None]:
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    def _init_schema(self) -> None:
        conn = self._get_conn()
        for ddl in _ALL_DDL:
            conn.execute(ddl)
        conn.commit()

    # ──────────────────────────────────────────────────────────────────────
    # Node CRUD
    # ──────────────────────────────────────────────────────────────────────

    def save_node(self, node: MemoryNode) -> None:
        """Insert or replace a memory node."""
        d = node.to_dict()
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO memory_nodes
                    (id, title, content, created_at, last_accessed,
                     weight, decay_rate, status, recall_count, tags)
                VALUES
                    (:id, :title, :content, :created_at, :last_accessed,
                     :weight, :decay_rate, :status, :recall_count, :tags)
                ON CONFLICT(id) DO UPDATE SET
                    title         = excluded.title,
                    content       = excluded.content,
                    last_accessed = excluded.last_accessed,
                    weight        = excluded.weight,
                    decay_rate    = excluded.decay_rate,
                    status        = excluded.status,
                    recall_count  = excluded.recall_count,
                    tags          = excluded.tags
                """,
                d,
            )

    def save_nodes(self, nodes: List[MemoryNode]) -> None:
        """Bulk upsert."""
        for node in nodes:
            self.save_node(node)

    def load_node(self, node_id: str) -> Optional[MemoryNode]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM memory_nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return MemoryNode.from_dict(dict(row)) if row else None

    def load_all_nodes(self) -> List[MemoryNode]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM memory_nodes").fetchall()
        return [MemoryNode.from_dict(dict(r)) for r in rows]

    def delete_node(self, node_id: str) -> None:
        with self._tx() as cur:
            cur.execute("DELETE FROM memory_nodes WHERE id = ?", (node_id,))
            cur.execute(
                "DELETE FROM memory_edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id),
            )

    def node_exists(self, node_id: str) -> bool:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM memory_nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return row is not None

    def search_nodes(self, query: str) -> List[MemoryNode]:
        conn = self._get_conn()
        like = f"%{query.lower()}%"
        rows = conn.execute(
            """
            SELECT * FROM memory_nodes
            WHERE lower(title) LIKE ? OR lower(content) LIKE ? OR lower(tags) LIKE ?
            ORDER BY weight DESC
            """,
            (like, like, like),
        ).fetchall()
        return [MemoryNode.from_dict(dict(r)) for r in rows]

    # ──────────────────────────────────────────────────────────────────────
    # Edge CRUD
    # ──────────────────────────────────────────────────────────────────────

    def save_edge(self, source_id: str, target_id: str, weight: float) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO memory_edges (source_id, target_id, weight)
                VALUES (?, ?, ?)
                ON CONFLICT(source_id, target_id) DO UPDATE SET weight = excluded.weight
                """,
                (source_id, target_id, round(weight, 6)),
            )

    def save_edges(self, edges: List[Tuple[str, str, float]]) -> None:
        for src, tgt, w in edges:
            self.save_edge(src, tgt, w)

    def load_all_edges(self) -> List[Tuple[str, str, float]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT source_id, target_id, weight FROM memory_edges"
        ).fetchall()
        return [(r["source_id"], r["target_id"], r["weight"]) for r in rows]

    def load_edges_for_node(self, node_id: str) -> List[Tuple[str, str, float]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT source_id, target_id, weight FROM memory_edges WHERE source_id = ?",
            (node_id,),
        ).fetchall()
        return [(r["source_id"], r["target_id"], r["weight"]) for r in rows]

    def delete_edge(self, source_id: str, target_id: str) -> None:
        with self._tx() as cur:
            cur.execute(
                "DELETE FROM memory_edges WHERE source_id = ? AND target_id = ?",
                (source_id, target_id),
            )

    # ──────────────────────────────────────────────────────────────────────
    # Simulation metrics
    # ──────────────────────────────────────────────────────────────────────

    def save_metric(self, metric: dict) -> None:
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO simulation_metrics
                    (day, active_count, dormant_count, long_term_count,
                     avg_weight, recovery_count, forgotten_count,
                     total_nodes, recorded_at)
                VALUES
                    (:day, :active_count, :dormant_count, :long_term_count,
                     :avg_weight, :recovery_count, :forgotten_count,
                     :total_nodes, :recorded_at)
                """,
                metric,
            )

    def load_metrics(self, limit: int = 10000) -> List[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM simulation_metrics ORDER BY day ASC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_metrics(self) -> None:
        with self._tx() as cur:
            cur.execute("DELETE FROM simulation_metrics")

    # ──────────────────────────────────────────────────────────────────────
    # LTM Highway persistence
    # ──────────────────────────────────────────────────────────────────────

    def save_ltm_entry(self, entry) -> None:
        """Persist a highway.HighwayEntry to the DB."""
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO ltm_highway
                    (node_id, title, consolidated_weight, consolidation_day,
                     access_count, interconnections)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    consolidated_weight = excluded.consolidated_weight,
                    access_count        = excluded.access_count,
                    interconnections    = excluded.interconnections
                """,
                (
                    entry.node_id,
                    entry.title,
                    entry.consolidated_weight,
                    entry.consolidation_day,
                    entry.access_count,
                    json.dumps(list(entry.interconnections)),
                ),
            )

    def load_ltm_entries(self) -> List[dict]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM ltm_highway").fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["interconnections"] = set(json.loads(d.get("interconnections", "[]")))
            results.append(d)
        return results

    # ──────────────────────────────────────────────────────────────────────
    # Full graph load / save helpers
    # ──────────────────────────────────────────────────────────────────────

    def load_graph(self):
        """
        Reconstruct a MemoryGraph from persistent storage.
        Returns a populated MemoryGraph instance.
        """
        from memory.graph import MemoryGraph

        graph = MemoryGraph()
        for node in self.load_all_nodes():
            graph.add_node(node)
        for src, tgt, w in self.load_all_edges():
            try:
                graph.add_edge(src, tgt, weight=w)
            except ValueError:
                pass  # stale edge referencing deleted node
        return graph

    def persist_graph(self, graph) -> None:
        """Save all nodes and edges from a MemoryGraph to SQLite."""
        from memory.graph import MemoryGraph

        self.save_nodes(graph.get_all_nodes())
        nx_g = graph.nx_graph
        edges = [
            (u, v, data.get("weight", 0.5))
            for u, v, data in nx_g.edges(data=True)
        ]
        self.save_edges(edges)

    # ──────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────

    def node_count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM memory_nodes").fetchone()[0]

    def edge_count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM memory_edges").fetchone()[0]

    def __repr__(self) -> str:
        return (
            f"NeuroStoreDB(path={self.db_path}, "
            f"nodes={self.node_count()}, edges={self.edge_count()})"
        )

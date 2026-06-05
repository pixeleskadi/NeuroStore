"""
NeuroStore - Long-Term Memory Highway
A specialized in-memory index of highly consolidated memories.
Analogous to cortical consolidation after hippocampal encoding.
Provides fast O(1) retrieval with minimal decay overhead.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from memory.node import MemoryNode, MemoryStatus, LONG_TERM_THRESHOLD


@dataclass
class HighwayEntry:
    """Metadata record for a memory in the Long-Term Highway."""
    node_id: str
    title: str
    consolidated_weight: float   # Weight at time of consolidation
    consolidation_day: int
    access_count: int = 0
    interconnections: Set[str] = field(default_factory=set)

    def touch(self) -> None:
        self.access_count += 1


class LongTermHighway:
    """
    A fast-retrieval index for LONG_TERM memory nodes.

    Properties:
    - O(1) lookup by node_id.
    - Maintains a network of interconnections between LTM entries.
    - Entries are never evicted (permanent storage tier).
    - Provides cluster analysis of related LTM memories.
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[str, HighwayEntry] = OrderedDict()
        self._interconnections: Dict[str, Set[str]] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Entry management
    # ──────────────────────────────────────────────────────────────────────

    def consolidate(self, node: MemoryNode, day: int = 0) -> HighwayEntry:
        """
        Add a memory to the Long-Term Highway upon reaching LONG_TERM status.
        Idempotent: calling twice updates metadata rather than duplicating.
        """
        if node.id in self._entries:
            entry = self._entries[node.id]
            entry.consolidated_weight = node.weight
            return entry

        entry = HighwayEntry(
            node_id=node.id,
            title=node.title,
            consolidated_weight=node.weight,
            consolidation_day=day,
        )
        self._entries[node.id] = entry
        self._interconnections[node.id] = set()

        # Automatically link to existing entries (strong LTM nodes tend to
        # interconnect — analogous to cortical memory schemas).
        self._auto_connect(node.id)

        return entry

    def remove(self, node_id: str) -> None:
        """Remove entry (e.g., if node weight somehow drops below LTM threshold)."""
        self._entries.pop(node_id, None)
        self._interconnections.pop(node_id, None)
        for connections in self._interconnections.values():
            connections.discard(node_id)

    # ──────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────

    def get(self, node_id: str) -> Optional[HighwayEntry]:
        entry = self._entries.get(node_id)
        if entry:
            entry.touch()
        return entry

    def contains(self, node_id: str) -> bool:
        return node_id in self._entries

    def get_all(self) -> List[HighwayEntry]:
        return list(self._entries.values())

    def get_most_accessed(self, top_n: int = 10) -> List[HighwayEntry]:
        return sorted(
            self._entries.values(),
            key=lambda e: e.access_count,
            reverse=True,
        )[:top_n]

    def get_most_connected(self, top_n: int = 10) -> List[HighwayEntry]:
        return sorted(
            self._entries.values(),
            key=lambda e: len(e.interconnections),
            reverse=True,
        )[:top_n]

    def search(self, query: str) -> List[HighwayEntry]:
        """Case-insensitive title search within LTM entries."""
        q = query.lower()
        return [e for e in self._entries.values() if q in e.title.lower()]

    # ──────────────────────────────────────────────────────────────────────
    # Interconnection management
    # ──────────────────────────────────────────────────────────────────────

    def add_interconnection(self, node_id_a: str, node_id_b: str) -> None:
        """Register a strong bidirectional link between two LTM nodes."""
        if node_id_a in self._entries and node_id_b in self._entries:
            self._interconnections.setdefault(node_id_a, set()).add(node_id_b)
            self._interconnections.setdefault(node_id_b, set()).add(node_id_a)
            self._entries[node_id_a].interconnections.add(node_id_b)
            self._entries[node_id_b].interconnections.add(node_id_a)

    def get_interconnections(self, node_id: str) -> Set[str]:
        return self._interconnections.get(node_id, set())

    def _auto_connect(self, new_id: str) -> None:
        """
        Heuristic: connect new LTM entry to the 3 most-accessed existing entries.
        Simulates the formation of cortical schemas around highly reinforced memories.
        """
        candidates = self.get_most_accessed(3)
        for entry in candidates:
            if entry.node_id != new_id:
                self.add_interconnection(new_id, entry.node_id)

    # ──────────────────────────────────────────────────────────────────────
    # Sync with graph
    # ──────────────────────────────────────────────────────────────────────

    def sync_from_graph(self, nodes: List[MemoryNode], day: int = 0) -> List[str]:
        """
        Scan nodes and consolidate any newly promoted LONG_TERM memories.
        Returns list of newly consolidated node IDs.
        """
        newly_added = []
        for node in nodes:
            if node.is_long_term and not self.contains(node.id):
                self.consolidate(node, day=day)
                newly_added.append(node.id)
        return newly_added

    def apply_ltm_decay_resistance(self, nodes: List[MemoryNode]) -> None:
        """
        Ensure all LTM nodes in `nodes` have a floor weight.
        LTM weight never drops below LONG_TERM_THRESHOLD.
        """
        for node in nodes:
            if self.contains(node.id):
                node.weight = max(node.weight, LONG_TERM_THRESHOLD)

    # ──────────────────────────────────────────────────────────────────────
    # Analytics
    # ──────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        entries = list(self._entries.values())
        if not entries:
            return {"count": 0, "avg_consolidation_weight": 0.0, "total_interconnections": 0}
        return {
            "count": len(entries),
            "avg_consolidation_weight": round(
                sum(e.consolidated_weight for e in entries) / len(entries), 2
            ),
            "total_interconnections": sum(
                len(e.interconnections) for e in entries
            ) // 2,  # divide by 2 since bidirectional
            "most_accessed": [e.title for e in self.get_most_accessed(3)],
            "most_connected": [e.title for e in self.get_most_connected(3)],
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"LongTermHighway(count={s['count']}, "
            f"interconnections={s['total_interconnections']})"
        )

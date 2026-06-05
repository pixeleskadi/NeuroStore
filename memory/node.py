"""
NeuroStore - Memory Node Module
Defines the core MemoryNode data model representing a single memory unit.
Biologically inspired: each node mimics a memory engram with weight, decay, and status.
"""

from __future__ import annotations
import uuid
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class MemoryStatus(str, Enum):
    """
    Lifecycle states of a memory node.
    ACTIVE    : Recently accessed, weight above dormancy threshold.
    DORMANT   : Weight below threshold; node persists but is inactive.
    LONG_TERM : Highly reinforced; stored in long-term memory highway with minimal decay.
    """
    ACTIVE = "ACTIVE"
    DORMANT = "DORMANT"
    LONG_TERM = "LONG_TERM"


# ──────────────────────────────────────────────────────────────────────────────
# Thresholds & constants
# ──────────────────────────────────────────────────────────────────────────────
DORMANCY_THRESHOLD: float = 20.0       # Weight below this → DORMANT
LONG_TERM_THRESHOLD: float = 90.0      # Weight above this → LONG_TERM
LONG_TERM_DECAY_MULTIPLIER: float = 0.05  # LTM decays at 5% of normal rate
DEFAULT_DECAY_RATE: float = 0.05       # Fraction of weight lost per simulated day
DEFAULT_INITIAL_WEIGHT: float = 50.0   # Starting weight for new memories
REINFORCEMENT_FACTOR: float = 10.0    # Weight gained per explicit recall
ASSOCIATIVE_REINFORCEMENT_FACTOR: float = 3.0  # Weight gained via associative activation
RECOVERY_ACTIVATION_THRESHOLD: float = 15.0    # Activation sum needed to recover dormant node
MAX_WEIGHT: float = 100.0             # Cap on memory weight


@dataclass
class MemoryNode:
    """
    A single memory unit — the fundamental atom of NeuroStore.

    Attributes
    ----------
    id              : Unique identifier (UUID4 string).
    title           : Short descriptive label (analogous to a memory cue).
    content         : Full memory content / notes.
    created_at      : ISO-format timestamp of creation.
    last_accessed   : ISO-format timestamp of most recent recall.
    weight          : Current salience/strength [0, 100].
    decay_rate      : Fraction of weight lost per simulated day.
    status          : ACTIVE | DORMANT | LONG_TERM.
    recall_count    : Number of times this memory has been recalled.
    tags            : Optional list of semantic tags for search.
    """

    title: str
    content: str

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    weight: float = DEFAULT_INITIAL_WEIGHT
    decay_rate: float = DEFAULT_DECAY_RATE
    status: MemoryStatus = MemoryStatus.ACTIVE
    recall_count: int = 0
    tags: list[str] = field(default_factory=list)

    # ──────────────────────────────────────────────────────────────────────
    # Derived / utility properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def effective_decay_rate(self) -> float:
        """Long-term memories decay much more slowly."""
        if self.status == MemoryStatus.LONG_TERM:
            return self.decay_rate * LONG_TERM_DECAY_MULTIPLIER
        return self.decay_rate

    @property
    def is_active(self) -> bool:
        return self.status == MemoryStatus.ACTIVE

    @property
    def is_dormant(self) -> bool:
        return self.status == MemoryStatus.DORMANT

    @property
    def is_long_term(self) -> bool:
        return self.status == MemoryStatus.LONG_TERM

    # ──────────────────────────────────────────────────────────────────────
    # State transitions
    # ──────────────────────────────────────────────────────────────────────

    def apply_decay(self, days: float = 1.0) -> float:
        """
        Reduce weight by decay_rate * days.
        Long-term memories use an attenuated rate.
        Returns the amount of weight lost.
        """
        if self.status == MemoryStatus.LONG_TERM:
            return 0.0  # LTM weight floor — no decay below 90

        loss = self.weight * self.effective_decay_rate * days
        self.weight = max(0.0, self.weight - loss)
        self._update_status()
        return loss

    def reinforce(self, factor: float = REINFORCEMENT_FACTOR) -> float:
        """
        Increase weight by factor (simulating recall / rehearsal).
        Updates last_accessed timestamp and recall_count.
        Returns new weight.
        """
        self.weight = min(MAX_WEIGHT, self.weight + factor)
        self.recall_count += 1
        self.last_accessed = datetime.utcnow().isoformat()

        # Recover from dormancy if sufficient weight regained
        if self.status == MemoryStatus.DORMANT and self.weight >= DORMANCY_THRESHOLD:
            self.status = MemoryStatus.ACTIVE

        self._update_status()
        return self.weight

    def associative_reinforce(self, edge_weight: float) -> float:
        """
        Partial activation from an associated memory recall.
        edge_weight ∈ [0, 1] scales the associative reinforcement.
        """
        activation = ASSOCIATIVE_REINFORCEMENT_FACTOR * edge_weight
        return self.reinforce(factor=activation)

    def recover(self) -> bool:
        """
        Attempt to recover a dormant memory.
        Returns True if recovery succeeded.
        """
        if self.status == MemoryStatus.DORMANT:
            self.status = MemoryStatus.ACTIVE
            self.weight = max(self.weight, DORMANCY_THRESHOLD + 5.0)
            self.last_accessed = datetime.utcnow().isoformat()
            return True
        return False

    def _update_status(self) -> None:
        """Recompute status based on current weight."""
        if self.status == MemoryStatus.LONG_TERM:
            return  # Long-term memories don't downgrade automatically
        if self.weight >= LONG_TERM_THRESHOLD:
            self.status = MemoryStatus.LONG_TERM
        elif self.weight < DORMANCY_THRESHOLD:
            self.status = MemoryStatus.DORMANT
        else:
            self.status = MemoryStatus.ACTIVE

    # ──────────────────────────────────────────────────────────────────────
    # Serialization helpers
    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "weight": round(self.weight, 4),
            "decay_rate": self.decay_rate,
            "status": self.status.value,
            "recall_count": self.recall_count,
            "tags": ",".join(self.tags),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryNode":
        node = cls(
            title=d["title"],
            content=d["content"],
            id=d["id"],
        )
        node.created_at = d.get("created_at", node.created_at)
        node.last_accessed = d.get("last_accessed", node.last_accessed)
        node.weight = float(d.get("weight", DEFAULT_INITIAL_WEIGHT))
        node.decay_rate = float(d.get("decay_rate", DEFAULT_DECAY_RATE))
        node.status = MemoryStatus(d.get("status", MemoryStatus.ACTIVE.value))
        node.recall_count = int(d.get("recall_count", 0))
        raw_tags = d.get("tags", "")
        node.tags = [t for t in raw_tags.split(",") if t] if isinstance(raw_tags, str) else raw_tags
        return node

    def __repr__(self) -> str:
        return (
            f"MemoryNode(id={self.id[:8]}…, title={self.title!r}, "
            f"weight={self.weight:.1f}, status={self.status.value})"
        )

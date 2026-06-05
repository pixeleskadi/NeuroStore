"""
NeuroStore - Charts Module
Generates publication-quality matplotlib charts from simulation metrics.
Outputs PNG files to the outputs/ directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from analytics.metrics import DaySnapshot, MetricsTracker

# ── Style ────────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor": "#161628",
    "axes.edgecolor": "#3a3a5c",
    "axes.labelcolor": "#c8c8e8",
    "xtick.color": "#9090b0",
    "ytick.color": "#9090b0",
    "text.color": "#e0e0ff",
    "grid.color": "#2a2a4a",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "lines.linewidth": 2.0,
    "font.family": "monospace",
}

PALETTE = {
    "active": "#44ff88",
    "dormant": "#ff4466",
    "long_term": "#4488ff",
    "avg_weight": "#ffcc44",
    "recovery": "#cc44ff",
    "forgotten": "#ff8844",
    "edges": "#44ccff",
}

OUTPUT_DIR = Path("outputs")


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def _save(fig: plt.Figure, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Individual chart functions
# ──────────────────────────────────────────────────────────────────────────────


def plot_memory_lifecycle(tracker: MetricsTracker, title: str = "") -> Path:
    """
    Stacked area chart showing active / dormant / long-term node counts over time.
    """
    _apply_style()
    snaps = tracker.get_all()
    if not snaps:
        return Path("no_data")

    days = [s.day for s in snaps]
    active = [s.active_count for s in snaps]
    dormant = [s.dormant_count for s in snaps]
    lt = [s.long_term_count for s in snaps]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        days,
        active, dormant, lt,
        labels=["Active", "Dormant", "Long-Term"],
        colors=[PALETTE["active"] + "99", PALETTE["dormant"] + "99", PALETTE["long_term"] + "99"],
    )
    ax.legend(loc="upper left", framealpha=0.3)
    ax.set_xlabel("Simulated Day")
    ax.set_ylabel("Node Count")
    ax.set_title(title or "Memory Lifecycle — Node Status Over Time", pad=14)
    ax.grid(True)
    fig.tight_layout()
    return _save(fig, "memory_lifecycle.png")


def plot_weight_trajectory(tracker: MetricsTracker) -> Path:
    """Line chart of average, max, and min memory weight over time."""
    _apply_style()
    snaps = tracker.get_all()
    if not snaps:
        return Path("no_data")

    days = [s.day for s in snaps]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(days, [s.avg_weight for s in snaps], color=PALETTE["avg_weight"], label="Avg Weight")
    ax.plot(days, [s.max_weight for s in snaps], color=PALETTE["active"], linestyle=":", label="Max Weight", alpha=0.7)
    ax.plot(days, [s.min_weight for s in snaps], color=PALETTE["dormant"], linestyle=":", label="Min Weight", alpha=0.7)

    ax.axhline(y=20, color="#ff4466", linestyle="--", alpha=0.5, label="Dormancy Threshold (20)")
    ax.axhline(y=90, color="#4488ff", linestyle="--", alpha=0.5, label="LTM Threshold (90)")

    ax.set_ylim(0, 105)
    ax.set_xlabel("Simulated Day")
    ax.set_ylabel("Memory Weight")
    ax.set_title("Memory Weight Trajectory Over Time", pad=14)
    ax.legend(loc="upper right", framealpha=0.3, fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    return _save(fig, "weight_trajectory.png")


def plot_recovery_and_forgetting(tracker: MetricsTracker) -> Path:
    """Dual-axis chart: cumulative recoveries vs cumulative forgotten."""
    _apply_style()
    snaps = tracker.get_all()
    if not snaps:
        return Path("no_data")

    days = [s.day for s in snaps]
    recoveries = [s.recovery_count for s in snaps]
    forgotten = [s.forgotten_count for s in snaps]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(days, recoveries, color=PALETTE["recovery"], label="Cumulative Recoveries")
    ax2.plot(days, forgotten, color=PALETTE["forgotten"], label="Cumulative Forgotten", linestyle="--")

    ax1.set_xlabel("Simulated Day")
    ax1.set_ylabel("Cumulative Recoveries", color=PALETTE["recovery"])
    ax2.set_ylabel("Cumulative Forgotten", color=PALETTE["forgotten"])
    ax1.set_title("Memory Recovery vs. Forgetting", pad=14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.3)
    ax1.grid(True)
    fig.tight_layout()
    return _save(fig, "recovery_vs_forgetting.png")


def plot_ltm_growth(tracker: MetricsTracker) -> Path:
    """Bar chart of long-term memory count progression."""
    _apply_style()
    snaps = tracker.get_all()
    if not snaps:
        return Path("no_data")

    days = [s.day for s in snaps]
    lt_counts = [s.long_term_count for s in snaps]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(days, lt_counts, alpha=0.4, color=PALETTE["long_term"])
    ax.plot(days, lt_counts, color=PALETTE["long_term"], label="Long-Term Memories")
    ax.set_xlabel("Simulated Day")
    ax.set_ylabel("LTM Count")
    ax.set_title("Long-Term Memory Consolidation Over Time", pad=14)
    ax.legend(framealpha=0.3)
    ax.grid(True)
    fig.tight_layout()
    return _save(fig, "ltm_growth.png")


def plot_edge_evolution(tracker: MetricsTracker) -> Path:
    """Line chart of total edges and average edge weight over time."""
    _apply_style()
    snaps = tracker.get_all()
    valid = [s for s in snaps if s.total_edges > 0]
    if not valid:
        return Path("no_data")

    days = [s.day for s in valid]
    edges = [s.total_edges for s in valid]
    avg_ew = [s.avg_edge_weight for s in valid]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(days, edges, color=PALETTE["edges"], label="Total Edges")
    ax2.plot(days, avg_ew, color=PALETTE["avg_weight"], linestyle="--", label="Avg Edge Weight")

    ax1.set_xlabel("Simulated Day")
    ax1.set_ylabel("Edge Count", color=PALETTE["edges"])
    ax2.set_ylabel("Avg Edge Weight", color=PALETTE["avg_weight"])
    ax1.set_title("Memory Network Edge Evolution", pad=14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.3)
    ax1.grid(True)
    fig.tight_layout()
    return _save(fig, "edge_evolution.png")


def plot_summary_dashboard(tracker: MetricsTracker) -> Path:
    """
    2×3 dashboard combining all key metrics in a single figure.
    Suitable for research papers and GitHub README.
    """
    _apply_style()
    snaps = tracker.get_all()
    if not snaps:
        return Path("no_data")

    days = [s.day for s in snaps]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("NeuroStore Simulation Dashboard", fontsize=16, y=1.01)

    # 1 — Node status stacked
    ax = axes[0, 0]
    ax.stackplot(
        days,
        [s.active_count for s in snaps],
        [s.dormant_count for s in snaps],
        [s.long_term_count for s in snaps],
        labels=["Active", "Dormant", "LTM"],
        colors=[PALETTE["active"] + "88", PALETTE["dormant"] + "88", PALETTE["long_term"] + "88"],
    )
    ax.set_title("Node Status")
    ax.legend(fontsize=8, framealpha=0.2)
    ax.set_xlabel("Day"); ax.grid(True)

    # 2 — Weight trajectory
    ax = axes[0, 1]
    ax.plot(days, [s.avg_weight for s in snaps], color=PALETTE["avg_weight"], label="Avg")
    ax.plot(days, [s.max_weight for s in snaps], color=PALETTE["active"], linestyle=":", alpha=0.7, label="Max")
    ax.plot(days, [s.min_weight for s in snaps], color=PALETTE["dormant"], linestyle=":", alpha=0.7, label="Min")
    ax.axhline(20, color="#ff4466", linestyle="--", alpha=0.4)
    ax.axhline(90, color="#4488ff", linestyle="--", alpha=0.4)
    ax.set_title("Weight Trajectory")
    ax.legend(fontsize=8, framealpha=0.2)
    ax.set_xlabel("Day"); ax.grid(True)

    # 3 — LTM growth
    ax = axes[0, 2]
    ax.fill_between(days, [s.long_term_count for s in snaps], alpha=0.3, color=PALETTE["long_term"])
    ax.plot(days, [s.long_term_count for s in snaps], color=PALETTE["long_term"])
    ax.set_title("LTM Growth")
    ax.set_xlabel("Day"); ax.grid(True)

    # 4 — Recovery count
    ax = axes[1, 0]
    ax.plot(days, [s.recovery_count for s in snaps], color=PALETTE["recovery"])
    ax.fill_between(days, [s.recovery_count for s in snaps], alpha=0.2, color=PALETTE["recovery"])
    ax.set_title("Cumulative Recoveries")
    ax.set_xlabel("Day"); ax.grid(True)

    # 5 — Forgotten count
    ax = axes[1, 1]
    ax.plot(days, [s.forgotten_count for s in snaps], color=PALETTE["forgotten"])
    ax.fill_between(days, [s.forgotten_count for s in snaps], alpha=0.2, color=PALETTE["forgotten"])
    ax.set_title("Cumulative Forgotten")
    ax.set_xlabel("Day"); ax.grid(True)

    # 6 — Edge count
    ax = axes[1, 2]
    ax.plot(days, [s.total_edges for s in snaps], color=PALETTE["edges"])
    ax.fill_between(days, [s.total_edges for s in snaps], alpha=0.2, color=PALETTE["edges"])
    ax.set_title("Total Edges")
    ax.set_xlabel("Day"); ax.grid(True)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#161628")
            for spine in ax.spines.values():
                spine.set_edgecolor("#3a3a5c")

    fig.tight_layout()
    return _save(fig, "simulation_dashboard.png")


def generate_all_charts(tracker: MetricsTracker) -> List[Path]:
    """Generate the complete chart suite. Returns list of output paths."""
    paths = [
        plot_memory_lifecycle(tracker),
        plot_weight_trajectory(tracker),
        plot_recovery_and_forgetting(tracker),
        plot_ltm_growth(tracker),
        plot_edge_evolution(tracker),
        plot_summary_dashboard(tracker),
    ]
    return [p for p in paths if p != Path("no_data")]

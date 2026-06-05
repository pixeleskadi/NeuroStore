"""
NeuroStore - Graph Visualization Module
Renders the memory graph as publication-quality PNG using NetworkX + Matplotlib.

Color scheme:
  ACTIVE    → green  (#44ff88)
  DORMANT   → red    (#ff4466)
  LONG_TERM → blue   (#4488ff)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from memory.graph import MemoryGraph
from memory.node import MemoryStatus

# ── Color palette ─────────────────────────────────────────────────────────────
STATUS_COLORS = {
    MemoryStatus.ACTIVE.value:    "#44ff88",
    MemoryStatus.DORMANT.value:   "#ff4466",
    MemoryStatus.LONG_TERM.value: "#4488ff",
}
BG_COLOR       = "#0a0a18"
EDGE_COLOR     = "#6060a0"
FONT_COLOR     = "#e0e0ff"
HIGHLIGHT_COLOR = "#ffff44"

OUTPUT_DIR = Path("outputs")


def _weight_to_size(weight: float) -> float:
    """Map node weight [0, 100] to marker size [200, 2500]."""
    return 200 + (weight / 100.0) ** 1.5 * 2300


def _edge_alpha(weight: float) -> float:
    """Map edge weight [0, 1] to transparency [0.1, 0.9]."""
    return 0.1 + weight * 0.8


def render_graph(
    graph: MemoryGraph,
    title: str = "NeuroStore Memory Graph",
    filename: str = "memory_graph.png",
    highlight_ids: Optional[list] = None,
    layout: str = "spring",
    figsize: tuple = (16, 12),
) -> Path:
    """
    Render the full memory graph to a PNG file.

    Parameters
    ----------
    graph        : MemoryGraph to visualize.
    title        : Plot title.
    filename     : Output filename (inside outputs/).
    highlight_ids: List of node IDs to highlight in yellow.
    layout       : One of 'spring', 'kamada_kawai', 'spectral', 'circular'.
    figsize      : Matplotlib figure size.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    G = graph.nx_graph

    if len(G.nodes) == 0:
        print("[visualization] Graph has no nodes — skipping render.")
        return Path("no_data")

    # ── Layout ────────────────────────────────────────────────────────────
    layout_fn = {
        "spring":       lambda g: nx.spring_layout(g, k=2.5, iterations=80, seed=42),
        "kamada_kawai": lambda g: nx.kamada_kawai_layout(g),
        "spectral":     lambda g: nx.spectral_layout(g),
        "circular":     lambda g: nx.circular_layout(g),
    }.get(layout, nx.spring_layout)
    pos = layout_fn(G)

    # ── Node attributes ───────────────────────────────────────────────────
    node_colors, node_sizes, node_labels, node_borders = [], [], {}, []
    highlight_ids = set(highlight_ids or [])

    for nid in G.nodes:
        node = graph.get_node(nid)
        status_val = G.nodes[nid].get("status", MemoryStatus.ACTIVE.value)
        color = STATUS_COLORS.get(status_val, "#888888")
        weight = G.nodes[nid].get("weight", 50.0)

        node_colors.append(color)
        node_sizes.append(_weight_to_size(weight))
        node_borders.append(HIGHLIGHT_COLOR if nid in highlight_ids else color)
        node_labels[nid] = (
            G.nodes[nid].get("title", nid[:8]) + f"\n({weight:.0f})"
        )

    # ── Edge attributes ───────────────────────────────────────────────────
    edge_weights = [G[u][v].get("weight", 0.3) for u, v in G.edges()]
    edge_alphas  = [_edge_alpha(w) for w in edge_weights]
    edge_widths  = [0.5 + w * 3.0 for w in edge_weights]

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, color=FONT_COLOR, fontsize=14, pad=16, fontfamily="monospace")
    ax.axis("off")

    # Draw edges with per-edge alpha (matplotlib doesn't support per-edge alpha natively,
    # so we iterate)
    for i, (u, v) in enumerate(G.edges()):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=EDGE_COLOR,
                alpha=edge_alphas[i],
                lw=edge_widths[i],
                mutation_scale=12,
            ),
        )

    # Edge weight labels (only for significant edges)
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 0.0)
        if w >= 0.4:
            xm = (pos[u][0] + pos[v][0]) / 2
            ym = (pos[u][1] + pos[v][1]) / 2
            ax.text(
                xm, ym, f"{w:.2f}",
                fontsize=6, color="#9090c0", alpha=0.7,
                ha="center", va="center",
            )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_borders,
        linewidths=2,
        ax=ax,
    )

    # Labels
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=7,
        font_color=FONT_COLOR,
        font_family="monospace",
        ax=ax,
    )

    # ── Legend ────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=STATUS_COLORS[MemoryStatus.ACTIVE.value],    label="ACTIVE"),
        mpatches.Patch(color=STATUS_COLORS[MemoryStatus.DORMANT.value],   label="DORMANT"),
        mpatches.Patch(color=STATUS_COLORS[MemoryStatus.LONG_TERM.value], label="LONG_TERM"),
    ]
    if highlight_ids:
        legend_patches.append(
            mpatches.Patch(color=HIGHLIGHT_COLOR, label="HIGHLIGHTED")
        )
    ax.legend(
        handles=legend_patches,
        loc="lower left",
        framealpha=0.25,
        labelcolor=FONT_COLOR,
        facecolor="#1a1a2e",
        fontsize=9,
    )

    # ── Stats box ─────────────────────────────────────────────────────────
    stats = graph.stats()
    stats_text = (
        f"Nodes : {stats['total']}   "
        f"Edges : {stats['edges']}\n"
        f"Active: {stats['active']}  "
        f"Dormant: {stats['dormant']}  "
        f"LTM: {stats['long_term']}\n"
        f"AvgW : {stats['avg_weight']:.1f}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=8, color="#b0b0d0",
        va="top", ha="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", alpha=0.6),
    )

    fig.tight_layout()
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    return out_path


def render_subgraph(
    graph: MemoryGraph,
    center_id: str,
    depth: int = 2,
    filename: str = "subgraph.png",
) -> Path:
    """
    Render only the neighborhood of a specific node up to `depth` hops.
    Useful for visualizing associative recall spread.
    """
    G = graph.nx_graph
    if center_id not in G:
        print(f"[visualization] Node {center_id[:8]} not found in graph.")
        return Path("no_data")

    # BFS to collect neighborhood
    visited = {center_id}
    frontier = {center_id}
    for _ in range(depth):
        next_frontier = set()
        for nid in frontier:
            next_frontier.update(G.successors(nid))
            next_frontier.update(G.predecessors(nid))
        frontier = next_frontier - visited
        visited |= frontier

    subG = G.subgraph(visited).copy()
    # Inject subgraph nodes back through render_graph logic
    sub_memory_graph = MemoryGraph()
    for nid in subG.nodes:
        node = graph.get_node(nid)
        if node:
            sub_memory_graph.add_node(node)
    for u, v, data in subG.edges(data=True):
        try:
            sub_memory_graph.add_edge(u, v, weight=data.get("weight", 0.5))
        except ValueError:
            pass

    return render_graph(
        sub_memory_graph,
        title=f"Subgraph — Center: {G.nodes[center_id].get('title', center_id[:8])}",
        filename=filename,
        highlight_ids=[center_id],
        layout="spring",
        figsize=(12, 9),
    )


def render_ltm_highway(graph: MemoryGraph, filename: str = "ltm_highway.png") -> Path:
    """Render only LONG_TERM memory nodes and their interconnections."""
    lt_nodes = graph.get_nodes_by_status(MemoryStatus.LONG_TERM)
    if not lt_nodes:
        print("[visualization] No LONG_TERM nodes to visualize.")
        return Path("no_data")

    lt_ids = {n.id for n in lt_nodes}
    sub = MemoryGraph()
    for node in lt_nodes:
        sub.add_node(node)

    G = graph.nx_graph
    for u, v, data in G.edges(data=True):
        if u in lt_ids and v in lt_ids:
            try:
                sub.add_edge(u, v, weight=data.get("weight", 0.5))
            except ValueError:
                pass

    return render_graph(
        sub,
        title="Long-Term Memory Highway",
        filename=filename,
        layout="circular",
        figsize=(12, 10),
    )

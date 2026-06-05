"""
NeuroStore - Main CLI Entry Point
Usage:
  python main.py add
  python main.py recall
  python main.py simulate [--days N] [--model MODEL]
  python main.py stats
  python main.py visualize
  python main.py demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── ensure project root is on sys.path ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db import NeuroStoreDB
from memory.graph import MemoryGraph
from memory.node import MemoryNode, MemoryStatus
from analytics.metrics import MetricsTracker
from analytics.charts import generate_all_charts
from visualization.graph_view import render_graph, render_subgraph, render_ltm_highway
from simulation import SimulationEngine, SimulationConfig

DB_PATH = PROJECT_ROOT / "data" / "neurostore.db"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_graph(db: NeuroStoreDB) -> MemoryGraph:
    graph = db.load_graph()
    print(f"[DB] Loaded {graph}")
    return graph


def _print_node(node: MemoryNode) -> None:
    status_sym = {"ACTIVE": "●", "DORMANT": "○", "LONG_TERM": "★"}.get(node.status.value, "?")
    print(
        f"  {status_sym} [{node.id[:8]}] {node.title:<30} "
        f"weight={node.weight:6.1f}  recalls={node.recall_count}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI commands
# ──────────────────────────────────────────────────────────────────────────────

def cmd_add(args: argparse.Namespace, db: NeuroStoreDB) -> None:
    """Interactively add a new memory node."""
    print("\n── Add Memory ─────────────────────────────────────────")
    title   = input("  Title   : ").strip()
    if not title:
        print("[error] Title cannot be empty."); return
    content = input("  Content : ").strip()
    tags    = input("  Tags (comma-separated, optional): ").strip()
    weight_str = input(f"  Initial weight (default 50): ").strip()
    decay_str  = input(f"  Decay rate (default 0.05): ").strip()

    weight = float(weight_str) if weight_str else 50.0
    decay  = float(decay_str)  if decay_str  else 0.05

    node = MemoryNode(
        title=title,
        content=content,
        weight=weight,
        decay_rate=decay,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
    )
    db.save_node(node)
    print(f"\n  ✓ Saved: {node}")

    # Optionally link to existing nodes
    graph = _load_graph(db)
    all_nodes = graph.get_all_nodes()
    if all_nodes:
        print("\n  Existing nodes (for association):")
        for n in all_nodes:
            _print_node(n)
        ids_str = input(
            "\n  Enter IDs to associate (first 8 chars, comma-separated, or blank): "
        ).strip()
        if ids_str:
            graph.add_node(node)
            for partial_id in ids_str.split(","):
                partial_id = partial_id.strip()
                match = next((n for n in all_nodes if n.id.startswith(partial_id)), None)
                if match:
                    weight_str = input(
                        f"  Edge weight to '{match.title}' (0–1, default 0.5): "
                    ).strip()
                    edge_w = float(weight_str) if weight_str else 0.5
                    graph.add_edge(node.id, match.id, weight=edge_w)
                    graph.add_edge(match.id, node.id, weight=edge_w * 0.7)
                    print(f"  ↔ Associated: {node.title} ↔ {match.title} (w={edge_w:.2f})")
                else:
                    print(f"  [warn] No node starting with '{partial_id}'")
            db.persist_graph(graph)
    print()


def cmd_recall(args: argparse.Namespace, db: NeuroStoreDB) -> None:
    """Search and recall a memory node."""
    print("\n── Recall Memory ──────────────────────────────────────")
    query = input("  Search query: ").strip()
    if not query:
        print("[error] Query cannot be empty."); return

    graph = _load_graph(db)
    results = graph.search(query)

    if not results:
        print(f"  No memories matching '{query}'.")
        return

    print(f"\n  Found {len(results)} match(es):")
    for i, node in enumerate(results[:10]):
        print(f"  [{i+1}] ", end="")
        _print_node(node)

    choice = input("\n  Choose [1-N]: ").strip()
    try:
        idx = int(choice) - 1
        node = results[idx]
    except (ValueError, IndexError):
        print("[error] Invalid choice."); return

    print(f"\n  Recalling: {node.title}")
    print(f"  Content  : {node.content}")
    print(f"  Status   : {node.status.value}")
    print(f"  Weight   : {node.weight:.1f}")

    # Perform recall + associative activation
    activation_map = graph.recall(node.id, depth=2)
    db.persist_graph(graph)

    print(f"\n  ✓ Recalled. New weight: {node.weight:.1f}")
    print(f"  Associative activation spread to {len(activation_map) - 1} neighbors:")
    for nid, act in sorted(activation_map.items(), key=lambda x: -x[1]):
        if nid == node.id:
            continue
        neighbor = graph.get_node(nid)
        if neighbor:
            print(f"    → {neighbor.title:<28} activation={act:.3f}")
    print()


def cmd_simulate(args: argparse.Namespace, db: NeuroStoreDB) -> None:
    """Run a multi-day simulation."""
    days  = args.days
    model = args.model

    graph = _load_graph(db)
    if len(graph) == 0:
        print("[warn] Graph is empty. Run `python main.py demo` to seed sample data.")
        return

    config = SimulationConfig(
        total_days=days,
        decay_model=model,
        recall_probability=0.3,
        reinforce_probability=0.15,
        recovery_scan_interval=5,
        shortcut_scan_interval=10,
        ltm_sync_interval=1,
        record_interval=1,
        verbose=True,
    )
    engine = SimulationEngine(graph=graph, db=db, config=config)
    result = engine.run(generate_charts=True)

    print("\n  Charts saved to outputs/")
    print("  Run `python main.py visualize` to generate graph PNG.\n")


def cmd_stats(args: argparse.Namespace, db: NeuroStoreDB) -> None:
    """Display memory graph statistics."""
    print("\n── NeuroStore Statistics ──────────────────────────────")
    graph = _load_graph(db)

    if len(graph) == 0:
        print("  No memories stored yet.")
        return

    stats = graph.stats()
    print(f"\n  Nodes  : {stats['total']}")
    print(f"  Edges  : {stats['edges']}")
    print(f"  Active : {stats['active']}")
    print(f"  Dormant: {stats['dormant']}")
    print(f"  LTM    : {stats['long_term']}")
    print(f"  Avg W  : {stats['avg_weight']}")
    print(f"  Max W  : {stats['max_weight']}")
    print(f"  Min W  : {stats['min_weight']}")

    print("\n  All Memory Nodes:")
    for node in sorted(graph.get_all_nodes(), key=lambda n: -n.weight):
        _print_node(node)

    # Load and display metrics history
    metrics_rows = db.load_metrics(limit=5)
    if metrics_rows:
        print("\n  Recent simulation snapshots (last 5):")
        print(f"  {'Day':>6} {'Active':>6} {'Dormant':>7} {'LTM':>5} {'AvgW':>7} {'Recov':>6}")
        for r in metrics_rows[-5:]:
            print(
                f"  {r['day']:>6} {r['active_count']:>6} {r['dormant_count']:>7} "
                f"{r['long_term_count']:>5} {r['avg_weight']:>7.1f} {r['recovery_count']:>6}"
            )
    print()


def cmd_visualize(args: argparse.Namespace, db: NeuroStoreDB) -> None:
    """Render the memory graph and LTM highway to PNG files."""
    print("\n── Visualize Memory Graph ─────────────────────────────")
    graph = _load_graph(db)

    if len(graph) == 0:
        print("  No memories to visualize.")
        return

    # Full graph
    path = render_graph(
        graph,
        title=f"NeuroStore Memory Graph ({len(graph)} nodes)",
        filename="memory_graph.png",
        layout=args.layout,
    )
    print(f"  ✓ Full graph     → {path}")

    # LTM highway
    lt_path = render_ltm_highway(graph, filename="ltm_highway.png")
    if lt_path != Path("no_data"):
        print(f"  ✓ LTM highway    → {lt_path}")

    # If charts exist
    metrics_rows = db.load_metrics()
    if metrics_rows:
        tracker = MetricsTracker()
        tracker.load_from_db(metrics_rows)
        chart_paths = generate_all_charts(tracker)
        print(f"  ✓ {len(chart_paths)} analytics charts → outputs/")
    print()


def cmd_demo(args: argparse.Namespace, db: NeuroStoreDB) -> None:
    """Seed the database with a rich sample memory graph and run a demo simulation."""
    print("\n── NeuroStore Demo ────────────────────────────────────")
    print("  Seeding sample quantum-physics / semiconductor memory graph …\n")

    graph = MemoryGraph()

    # ── Sample memory nodes ───────────────────────────────────────────────
    memories = [
        MemoryNode("Quantum Mechanics",
                   "Foundational study notes on quantum principles, wave-particle duality, "
                   "superposition, and the measurement problem.",
                   weight=75.0, decay_rate=0.04, tags=["physics", "quantum"]),
        MemoryNode("Wave Function",
                   "The wave function ψ encodes the probability amplitude of a quantum system. "
                   "Collapse upon measurement is described by the Born rule.",
                   weight=65.0, decay_rate=0.05, tags=["quantum", "math"]),
        MemoryNode("Quantum Tunneling",
                   "Phenomenon where a particle penetrates an energy barrier classically forbidden. "
                   "Key mechanism in tunnel diodes and STM microscopy.",
                   weight=60.0, decay_rate=0.06, tags=["quantum", "semiconductor"]),
        MemoryNode("Semiconductor Physics",
                   "Band theory, carrier transport, doping, and p-n junction behaviour. "
                   "Foundation of modern electronics.",
                   weight=70.0, decay_rate=0.04, tags=["semiconductor", "physics"]),
        MemoryNode("TFET (Tunnel FET)",
                   "Tunnel field-effect transistor exploiting band-to-band tunneling for "
                   "sub-60 mV/dec switching. Low-power device research.",
                   weight=50.0, decay_rate=0.07, tags=["semiconductor", "device", "TFET"]),
        MemoryNode("Electronics Fundamentals",
                   "Ohm's law, Kirchhoff's laws, RC circuits, amplifiers, and feedback systems.",
                   weight=80.0, decay_rate=0.03, tags=["electronics"]),
        MemoryNode("Schrödinger Equation",
                   "Time-dependent and time-independent forms governing quantum state evolution. "
                   "Analytical solutions for particle in a box, harmonic oscillator.",
                   weight=55.0, decay_rate=0.06, tags=["quantum", "math"]),
        MemoryNode("Heisenberg Uncertainty Principle",
                   "Δx·Δp ≥ ħ/2 — fundamental limit on simultaneous precision of position "
                   "and momentum measurements.",
                   weight=50.0, decay_rate=0.06, tags=["quantum", "physics"]),
        MemoryNode("Band Gap Engineering",
                   "Tuning semiconductor band gap via composition, strain, or quantum confinement "
                   "for photovoltaic and optoelectronic devices.",
                   weight=45.0, decay_rate=0.07, tags=["semiconductor", "materials"]),
        MemoryNode("Graph Neural Networks",
                   "Machine learning on graph-structured data. Message-passing framework. "
                   "Applications in drug discovery, social networks, knowledge graphs.",
                   weight=40.0, decay_rate=0.08, tags=["ml", "graphs", "ai"]),
        MemoryNode("Neuromorphic Computing",
                   "Hardware architectures inspired by the brain: spike-based processing, "
                   "memristors, Intel Loihi, IBM TrueNorth.",
                   weight=35.0, decay_rate=0.09, tags=["computing", "brain", "hardware"]),
        MemoryNode("Ebbinghaus Forgetting Curve",
                   "Empirical model of memory decay: retention R = e^(-t/s). "
                   "Spaced repetition exploits this curve to optimise learning.",
                   weight=30.0, decay_rate=0.10, tags=["memory", "psychology"]),
        MemoryNode("Long-Term Potentiation (LTP)",
                   "Persistent strengthening of synapses based on recent activity patterns. "
                   "Cellular basis of learning and memory in the hippocampus.",
                   weight=92.0, decay_rate=0.02, tags=["neuroscience", "memory"]),
        MemoryNode("Python Programming",
                   "High-level, general-purpose language. Key libraries: NumPy, Pandas, "
                   "NetworkX, SQLite3, Matplotlib.",
                   weight=85.0, decay_rate=0.03, tags=["programming", "python"]),
        MemoryNode("SQLite Internals",
                   "B-tree storage, WAL mode, ACID transactions, and query planning in SQLite. "
                   "Lightweight embedded SQL engine.",
                   weight=55.0, decay_rate=0.05, tags=["database", "sql"]),
    ]

    # ── Add nodes ─────────────────────────────────────────────────────────
    for m in memories:
        graph.add_node(m)

    # ── Reference map ─────────────────────────────────────────────────────
    by_title = {m.title: m for m in memories}

    def link(a: str, b: str, w: float, bi: bool = True) -> None:
        graph.add_edge(by_title[a].id, by_title[b].id, weight=w, bidirectional=bi)

    # Quantum cluster
    link("Quantum Mechanics",              "Wave Function",                  0.90)
    link("Quantum Mechanics",              "Schrödinger Equation",           0.88)
    link("Quantum Mechanics",              "Heisenberg Uncertainty Principle",0.85)
    link("Quantum Mechanics",              "Quantum Tunneling",               0.80)
    link("Wave Function",                  "Schrödinger Equation",           0.75)
    link("Quantum Tunneling",              "TFET (Tunnel FET)",               0.85)
    link("Quantum Tunneling",              "Semiconductor Physics",           0.70)
    link("Schrödinger Equation",           "Heisenberg Uncertainty Principle",0.60)

    # Semiconductor cluster
    link("Semiconductor Physics",          "TFET (Tunnel FET)",              0.82)
    link("Semiconductor Physics",          "Band Gap Engineering",            0.78)
    link("Semiconductor Physics",          "Electronics Fundamentals",        0.65)
    link("Band Gap Engineering",           "TFET (Tunnel FET)",               0.60)
    link("Electronics Fundamentals",       "Semiconductor Physics",           0.55)

    # CS / ML bridge
    link("Graph Neural Networks",          "Python Programming",              0.55)
    link("Neuromorphic Computing",         "Quantum Mechanics",               0.40)
    link("Neuromorphic Computing",         "Semiconductor Physics",           0.50)
    link("Neuromorphic Computing",         "Long-Term Potentiation (LTP)",    0.65)

    # Memory science
    link("Ebbinghaus Forgetting Curve",    "Long-Term Potentiation (LTP)",    0.70)
    link("Long-Term Potentiation (LTP)",   "Quantum Mechanics",               0.30)

    # Software
    link("Python Programming",             "SQLite Internals",                0.60)
    link("SQLite Internals",               "Python Programming",              0.55)

    # Persist
    db.persist_graph(graph)
    print(f"  Saved {len(memories)} nodes and edges → {db.db_path}\n")

    # Quick render before simulation
    render_graph(graph, title="NeuroStore — Initial State", filename="initial_graph.png")
    print("  ✓ Initial graph → outputs/initial_graph.png\n")

    # Run 30-day simulation
    config = SimulationConfig(
        total_days=30,
        decay_model="exponential",
        recall_probability=0.25,
        reinforce_probability=0.12,
        recovery_scan_interval=5,
        shortcut_scan_interval=10,
        ltm_sync_interval=1,
        verbose=True,
        random_seed=42,
    )
    engine = SimulationEngine(graph=graph, db=db, config=config)
    result = engine.run(generate_charts=True)

    # Post-sim render
    render_graph(graph, title="NeuroStore — After 30-Day Simulation", filename="memory_graph.png")
    render_ltm_highway(graph)
    print("\n  ✓ Post-sim graph → outputs/memory_graph.png")
    print("  ✓ LTM highway    → outputs/ltm_highway.png\n")

    result.print_report()


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurostore",
        description="NeuroStore — Biologically-inspired memory storage system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo                      # Seed sample data + 30-day sim
  python main.py add                       # Interactively add a memory
  python main.py recall                    # Search and recall a memory
  python main.py simulate --days 90        # Run 90-day simulation
  python main.py simulate --days 365 --model linear
  python main.py stats                     # Print graph statistics
  python main.py visualize                 # Render PNG visualizations
  python main.py visualize --layout circular
        """,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("add",       help="Add a new memory node")
    sub.add_parser("recall",    help="Search and recall a memory")
    sub.add_parser("stats",     help="Show graph statistics")
    sub.add_parser("demo",      help="Seed sample graph and run demo simulation")

    sim_p = sub.add_parser("simulate", help="Run a time-stepped simulation")
    sim_p.add_argument("--days",  type=int, default=30,
                       choices=[7, 14, 30, 90, 180, 365],
                       help="Number of simulated days (default: 30)")
    sim_p.add_argument("--model", type=str, default="exponential",
                       choices=["exponential", "linear", "power"],
                       help="Decay model (default: exponential)")

    vis_p = sub.add_parser("visualize", help="Render graph visualizations")
    vis_p.add_argument("--layout", type=str, default="spring",
                       choices=["spring", "kamada_kawai", "spectral", "circular"],
                       help="Graph layout algorithm (default: spring)")

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("""
 ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ███████╗████████╗ ██████╗ ██████╗ ███████╗
 ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝
 ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║███████╗   ██║   ██║   ██║██████╔╝█████╗  
 ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║╚════██║   ██║   ██║   ██║██╔══██╗██╔══╝  
 ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝███████║   ██║   ╚██████╔╝██║  ██║███████╗
 ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝
 Biologically-inspired memory storage with reinforcement, forgetting, recovery & recall.
""")

    if not args.command:
        parser.print_help()
        sys.exit(0)

    db = NeuroStoreDB(DB_PATH)

    dispatch = {
        "add":       cmd_add,
        "recall":    cmd_recall,
        "simulate":  cmd_simulate,
        "stats":     cmd_stats,
        "visualize": cmd_visualize,
        "demo":      cmd_demo,
    }

    fn = dispatch.get(args.command)
    if fn:
        fn(args, db)
    else:
        print(f"[error] Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

    db.close()


if __name__ == "__main__":
    main()

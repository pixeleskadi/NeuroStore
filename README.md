# NeuroStore

> *A biologically-inspired memory storage architecture with reinforcement, forgetting, recovery, and associative recall.*

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-55%20passed-brightgreen.svg)](#testing)
[![Storage: SQLite](https://img.shields.io/badge/storage-SQLite-orange.svg)](https://sqlite.org)

---

## Table of Contents

1. [Research Motivation](#research-motivation)
2. [Biological Inspiration](#biological-inspiration)
3. [System Architecture](#system-architecture)
4. [Memory Lifecycle](#memory-lifecycle)
5. [Core Concepts](#core-concepts)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [Experimental Results](#experimental-results)
10. [API Reference](#api-reference)
11. [Future Research Directions](#future-research-directions)
12. [Research Paper Ideas](#research-paper-ideas)
13. [Contributing](#contributing)
14. [License](#license)

---

## Research Motivation

Traditional data storage systems treat all data as equally important, equally accessible, and permanently retrievable. They have no concept of **importance decay**, **associative access**, or **consolidation over time**. Real-world intelligent systems — biological or artificial — must manage knowledge that:

- Becomes less accessible if not used
- Strengthens through rehearsal and association
- Clusters into related knowledge schemas
- Can be partially recovered from contextual cues

NeuroStore is a proof-of-concept demonstrating that these behaviours can be modelled computationally, opening a path toward **adaptive, brain-like storage systems** for AI agents, robotics, and knowledge management applications.

---

## Biological Inspiration

NeuroStore models three well-established neuroscientific principles:

### 1. Ebbinghaus Forgetting Curve (1885)
Hermann Ebbinghaus discovered that memory retention follows an exponential decay function:

```
R(t) = e^(-t / S)
```

Where `R` is retention, `t` is time since encoding, and `S` is the *stability* of the memory (higher for stronger/older memories). NeuroStore implements this as the default decay model with three variants: exponential (default), linear, and power-law.

### 2. Long-Term Potentiation (LTP)
*Neurons that fire together, wire together.* — Hebbian learning

When memories are repeatedly co-activated, the synaptic connections between them strengthen. NeuroStore models this via:
- **Explicit reinforcement**: Direct recall increases node weight by `reinforcement_factor`
- **Associative reinforcement**: Nearby nodes receive partial activation proportional to edge weight
- **Hebbian edge strengthening**: Co-activated nodes increase their shared edge weight

### 3. Memory Consolidation (Hippocampal-Cortical Transfer)
Highly reinforced memories transition from hippocampal (fast, volatile) to cortical (slow, stable) storage. In NeuroStore:
- Nodes below weight 20 → **DORMANT** (forgotten but not deleted)
- Nodes above weight 90 → **LONG_TERM** (consolidated; minimal decay)
- DORMANT nodes can be recovered through sufficiently strong associative activation from neighbours

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         NeuroStore                              │
├─────────────┬───────────────────────────────────────────────────┤
│   CLI       │  main.py  (add | recall | simulate | stats |     │
│             │           visualize | demo)                       │
├─────────────┴────────────────────────────────────────────────── ┤
│   Simulation Engine (simulation.py)                             │
│   Orchestrates: decay → recall → recovery → LTM sync           │
├──────────────────┬──────────────────────────────────────────────┤
│   Memory Layer   │   Graph Layer                                │
│                  │                                              │
│  MemoryNode      │  MemoryGraph (NetworkX DiGraph)              │
│  - weight        │  - weighted directed edges                   │
│  - decay_rate    │  - BFS associative recall                    │
│  - status        │  - shortcut formation                        │
│  - recall_count  │  - path finding                              │
├──────────────────┴──────────────────────────────────────────────┤
│   Engines                                                       │
│   DecayEngine    ReinforcementEngine   RecoveryEngine           │
│   LongTermHighway (fast O(1) LTM index)                        │
├─────────────────────────────────────────────────────────────────┤
│   Analytics          │   Visualization                          │
│   MetricsTracker     │   NetworkX + Matplotlib                  │
│   Charts (6 types)   │   graph_view.py                          │
├──────────────────────┴──────────────────────────────────────────┤
│   Persistence: SQLite (WAL mode, ACID transactions)             │
│   Tables: memory_nodes | memory_edges | simulation_metrics |    │
│           ltm_highway                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Memory Lifecycle

Each memory node progresses through the following states:

```
                    ┌─────────────────────────────────┐
                    │          NEW MEMORY              │
                    │  weight = 50  status = ACTIVE    │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │    ACTIVE  (weight 20–89)        │◄──── Recall / Rehearsal
                    │  Decays daily via forgetting     │      (weight += factor)
                    │  curve. Associative recall       │
                    │  strengthens connections.        │
                    └──────┬─────────────┬────────────┘
                           │             │
              weight < 20  │             │  weight ≥ 90
                           ▼             ▼
                    ┌──────────┐  ┌─────────────────────┐
                    │ DORMANT  │  │   LONG_TERM (LTM)    │
                    │ (weight  │  │  Decay ×0.05 (95%    │
                    │ < 20)    │  │  slower). Enters     │
                    │ Not      │  │  LTM Highway for     │
                    │ deleted. │  │  fast retrieval.     │
                    └────┬─────┘  └─────────────────────┘
                         │
          Sufficient     │
          activation     │
          from neighbours│
                    ┌────▼─────┐
                    │ RECOVERED │
                    │ → ACTIVE  │
                    └──────────┘
```

### Thresholds

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DORMANCY_THRESHOLD` | 20.0 | Weight below this → DORMANT |
| `LONG_TERM_THRESHOLD` | 90.0 | Weight above this → LONG_TERM |
| `LTM_DECAY_MULTIPLIER` | 0.05 | LTM decays at 5% of normal rate |
| `DEFAULT_INITIAL_WEIGHT` | 50.0 | New memory starting weight |
| `REINFORCEMENT_FACTOR` | 10.0 | Weight gained per explicit recall |
| `ASSOCIATIVE_REINFORCE` | 3.0 | Weight gained via graph traversal |
| `MAX_WEIGHT` | 100.0 | Weight ceiling |

---

## Core Concepts

### Graph-Based Retrieval

Unlike traditional key-value stores, NeuroStore retrieval uses **weighted BFS** across the association graph. Recalling node A propagates activation to all reachable neighbours within `depth` hops, attenuated by edge weight and traversal depth:

```
activation(neighbor) = activation(source) × edge_weight × depth_decay^hop
```

This mirrors hippocampal pattern completion: a partial cue activates the full memory, and related memories receive fractional activation.

### Memory Shortcuts

When node B on a path A → B → C becomes DORMANT, NeuroStore automatically forms a shortcut:

```
A → C   (weight = A→B × B→C × 0.6)
```

This represents the brain's tendency to form condensed semantic schemas — the details of intermediate steps fade, but the high-level relationship persists.

### Spaced Repetition

The `ReinforcementEngine` implements an SM-2-inspired review schedule. Optimal inter-repetition intervals grow with recall count:

```
interval(n) = interval(n-1) × ease_factor
```

Diminishing-returns reinforcement ensures the marginal gain per recall decreases as `recall_count` grows, preventing runaway weight inflation.

### Long-Term Memory Highway

The `LongTermHighway` is a separate fast-access index (`OrderedDict`) for LONG_TERM nodes. It provides:
- O(1) lookup by node ID
- Automatic interconnection between recently consolidated memories (simulating cortical schema formation)
- Decay resistance: LTM weight never drops below 90
- Access-count tracking for retrieval frequency analytics

---

## Installation

### Prerequisites
- Python 3.10+
- No cloud dependencies, no API keys

```bash
git clone https://github.com/yourusername/neurostore.git
cd neurostore
pip install -r requirements.txt
```

### Requirements
```
networkx>=3.2.1
matplotlib>=3.8.0
numpy>=1.26.0
pytest>=8.0.0
```

---

## Usage

### Quick Start — Demo

Seeds a 15-node quantum physics / semiconductor knowledge graph and runs a 30-day simulation:

```bash
python main.py demo
```

Output:
- `outputs/initial_graph.png`  — Graph before simulation
- `outputs/memory_graph.png`   — Graph after simulation
- `outputs/ltm_highway.png`    — Long-term memory cluster
- `outputs/simulation_dashboard.png` — 6-panel analytics dashboard
- `outputs/memory_lifecycle.png`
- `outputs/weight_trajectory.png`
- `outputs/recovery_vs_forgetting.png`
- `outputs/ltm_growth.png`
- `outputs/edge_evolution.png`

---

### Add a Memory

```bash
python main.py add
```

Interactive prompts:
```
Title   : Transformer Architecture
Content : Self-attention mechanism, multi-head attention, positional encoding.
Tags    : ml, nlp, attention
Initial weight (default 50): 60
Decay rate (default 0.05): 0.04
```

Optionally associate with existing nodes by providing their 8-character ID prefixes.

---

### Recall a Memory

```bash
python main.py recall
```

```
Search query: quantum

Found 3 match(es):
  [1] ● [a1b2c3d4] Quantum Mechanics         weight=  85.3  recalls=7
  [2] ● [e5f6g7h8] Quantum Tunneling          weight=  72.1  recalls=3
  [3] ○ [i9j0k1l2] Quantum Entanglement       weight=  14.2  recalls=0

Choose [1-N]: 1

Recalling: Quantum Mechanics
Content  : Foundational study notes on quantum principles ...
Status   : ACTIVE
Weight   : 85.3

✓ Recalled. New weight: 92.8
  Associative activation spread to 4 neighbors:
    → Wave Function              activation=0.720
    → Schrödinger Equation       activation=0.704
    → Quantum Tunneling          activation=0.640
    → Semiconductor Physics      activation=0.448
```

---

### Run a Simulation

```bash
# 30-day simulation (default)
python main.py simulate

# 90-day simulation with linear decay
python main.py simulate --days 90 --model linear

# 365-day long-term study
python main.py simulate --days 365 --model exponential
```

Available durations: `7`, `14`, `30`, `90`, `180`, `365`
Available decay models: `exponential` (default), `linear`, `power`

---

### View Statistics

```bash
python main.py stats
```

```
── NeuroStore Statistics ──────────────────────────────
  Nodes  : 15
  Edges  : 38
  Active : 5
  Dormant: 0
  LTM    : 10
  Avg W  : 85.3
  Max W  : 100.0
  Min W  : 42.1

  All Memory Nodes:
  ★ [a1b2c3d4] Quantum Mechanics             weight= 100.0  recalls=12
  ★ [e5f6g7h8] Python Programming            weight= 100.0  recalls=15
  ...
```

---

### Visualize the Graph

```bash
# Spring layout (default)
python main.py visualize

# Alternative layouts
python main.py visualize --layout kamada_kawai
python main.py visualize --layout circular
python main.py visualize --layout spectral
```

**Color scheme:**
- 🟢 **Green** — ACTIVE memories
- 🔴 **Red** — DORMANT memories
- 🔵 **Blue** — LONG_TERM memories
- Node size proportional to weight
- Edge opacity and thickness proportional to association strength

---

### Programmatic API

```python
from memory.graph import MemoryGraph
from memory.node import MemoryNode
from memory.decay import DecayEngine
from simulation import SimulationEngine, SimulationConfig
from database.db import NeuroStoreDB

# Build a graph
graph = MemoryGraph()
qm = MemoryNode("Quantum Mechanics", "Wave-particle duality...", weight=60.0)
wf = MemoryNode("Wave Function", "ψ encodes probability amplitude...", weight=55.0)
graph.add_node(qm)
graph.add_node(wf)
graph.add_edge(qm.id, wf.id, weight=0.85, bidirectional=True)

# Recall a memory (triggers associative activation)
activation_map = graph.recall(qm.id, depth=2)

# Run a simulation
db = NeuroStoreDB("data/neurostore.db")
config = SimulationConfig(total_days=90, decay_model="exponential")
engine = SimulationEngine(graph=graph, db=db, config=config)
result = engine.run(generate_charts=True)

result.print_report()
```

---

## Project Structure

```
neurostore/
├── main.py                    # CLI entry point
├── simulation.py              # Simulation engine (orchestrator)
├── requirements.txt
├── README.md
│
├── memory/
│   ├── __init__.py
│   ├── node.py                # MemoryNode dataclass + state machine
│   ├── graph.py               # MemoryGraph (NetworkX DiGraph wrapper)
│   ├── decay.py               # DecayEngine (exponential/linear/power)
│   ├── reinforcement.py       # ReinforcementEngine + spaced repetition
│   ├── recovery.py            # RecoveryEngine + PathRestorer
│   └── highway.py             # LongTermHighway (fast LTM index)
│
├── database/
│   ├── __init__.py
│   └── db.py                  # NeuroStoreDB (SQLite, WAL mode)
│
├── analytics/
│   ├── __init__.py
│   ├── metrics.py             # MetricsTracker + DaySnapshot
│   └── charts.py              # 6 matplotlib chart types
│
├── visualization/
│   ├── __init__.py
│   └── graph_view.py          # render_graph / render_subgraph / render_ltm
│
├── tests/
│   ├── __init__.py
│   └── test_neurostore.py     # 55 unit + integration tests
│
├── data/
│   └── neurostore.db          # SQLite database (auto-created)
│
└── outputs/                   # Generated PNG files
    ├── initial_graph.png
    ├── memory_graph.png
    ├── ltm_highway.png
    ├── simulation_dashboard.png
    └── ...
```

---

## Testing

```bash
python -m pytest tests/ -v
```

**55 tests across 8 test classes:**

| Class | Tests | Coverage |
|-------|-------|----------|
| `TestMemoryNode` | 11 | Node lifecycle, serialization, decay, reinforcement |
| `TestMemoryGraph` | 9 | Graph ops, recall propagation, shortcut formation |
| `TestDecayEngine` | 7 | All three decay models, LTM protection, dormancy |
| `TestReinforcementEngine` | 5 | Explicit/associative, diminishing returns, LTM tracking |
| `TestRecoveryEngine` | 4 | Scan recovery, targeted recovery, isolated nodes |
| `TestLongTermHighway` | 5 | Consolidation, idempotency, access tracking |
| `TestNeuroStoreDB` | 8 | Full CRUD, graph persistence, metrics |
| `TestMetricsTracker` | 5 | Snapshot, series, summary, JSON/CSV export |
| `TestIntegration` | 1 | Full 10-day simulation cycle |

---

## Experimental Results

Running `python main.py demo` produces the following observations on the 15-node quantum physics graph over 30 simulated days:

| Metric | Value |
|--------|-------|
| Total nodes | 15 |
| Final ACTIVE | 5 |
| Final DORMANT | 0 |
| Final LONG_TERM | 10 |
| LTM promotions | 10 |
| LTM interconnections | 24 |
| Retrieval efficiency | 100% |
| Final avg weight | 85.3 |

**Key observations:**
1. **LTM consolidation is rapid for well-connected nodes** — "Long-Term Potentiation (LTP)" achieved LONG_TERM status by Day 1 due to high initial weight (92) and frequent associative activation from its neighbours.
2. **Frequently accessed nodes resist forgetting** — In a 30-day run with 25% recall probability, no nodes became DORMANT. In a 365-day run with lower recall probability, dormancy and recovery events become prominent.
3. **Graph topology influences consolidation speed** — Hub nodes (high in-degree) receive more associative activation and consolidate faster than leaf nodes.
4. **LTM highway forms dense clusters** — With 10 LONG_TERM nodes and 24 bidirectional interconnections, the highway exhibits small-world network properties.

---

## Future Research Directions

### 1. Small Local LLM Categorization
Integrate a quantized local LLM (e.g., Ollama + Mistral-7B) to:
- Auto-generate semantic tags for new memories
- Suggest association targets when adding a node
- Generate natural-language recall summaries

### 2. Vector Embeddings for Semantic Similarity
Replace or augment title/tag search with dense vector embeddings:
- Use `sentence-transformers` to embed memory content
- Compute cosine similarity to automatically suggest edge weights
- Enable semantic nearest-neighbour recall (FAISS / ChromaDB)
- Compare NeuroStore graph traversal vs. pure vector search recall quality

### 3. Raspberry Pi Deployment
Port NeuroStore to embedded hardware:
- Target: Raspberry Pi 4 (4GB) or Pi 5
- SQLite WAL mode is well-suited for SD card I/O patterns
- Evaluate decay simulation overhead on ARM Cortex-A72
- Explore use cases: personal knowledge assistant, robotics episodic memory

### 4. SSD-Aware Storage Optimization
Optimize for NAND flash storage characteristics:
- Batch writes to reduce write amplification
- Implement tiered storage: ACTIVE nodes in RAM, DORMANT nodes on SSD
- WAL checkpoint tuning for flash-friendly write patterns
- Evaluate LevelDB / RocksDB as SQLite alternatives for high-throughput workloads

### 5. Neuromorphic Computing Adaptation
Map NeuroStore onto neuromorphic hardware:
- Intel Loihi 2 / IBM NorthPole spike-based processing
- Represent weight values as spike rates
- Decay as membrane potential leakage
- Associative recall as lateral inhibition / winner-take-all circuits

### 6. Comparison with Vector Databases
Rigorous benchmarking study:
- **Recall quality**: NeuroStore graph traversal vs. Pinecone / Weaviate / Chroma
- **Write throughput**: bulk memory ingestion
- **Read latency**: single-node recall vs. ANN search
- **Associative breadth**: how many related memories does each approach surface?
- Hypothesis: graph traversal surfaces contextually relevant memories that pure vector search misses due to semantic gap between embedding space and conceptual association

### 7. Comparison with Traditional SQL Retrieval
- Full-text search (SQLite FTS5) vs. NeuroStore graph recall
- Query latency vs. associative depth
- Effect of graph density on retrieval quality
- Hybrid approach: SQL for exact match, graph for associative expansion

### 8. Emotional Salience Weighting
Add an `emotional_valence` field to MemoryNode:
- Positive/negative valence increases initial weight (analogous to amygdala modulation)
- High-valence memories have lower decay rates
- Model flashbulb memories (traumatic/highly emotional events → near-zero decay)

### 9. Forgetting as Feature, Not Bug
Research direction: intentional forgetting for privacy and cognitive hygiene:
- Implement `force_forget(node_id)` that drives weight to zero
- Study effects on associative graph connectivity post-forced-forgetting
- Application: GDPR-compliant AI memory systems

### 10. Multi-Agent Shared Memory
Extend NeuroStore to a distributed, multi-agent architecture:
- Shared `LongTermHighway` across agents (common knowledge base)
- Private ACTIVE/DORMANT layers per agent
- Study knowledge propagation dynamics in agent networks

---

## Research Paper Ideas

1. **"NeuroStore: A Graph-Based Memory Architecture Inspired by Hippocampal-Cortical Consolidation"** — Systems paper describing the architecture, comparing with vector databases and SQL retrieval on knowledge retention benchmarks.

2. **"Simulating the Ebbinghaus Curve in Artificial Memory Systems: Exponential vs. Power-Law Forgetting in Graph-Structured Knowledge Bases"** — Empirical comparison of decay models on retrieval quality over simulated time horizons.

3. **"Associative Memory Recovery in Artificial Systems: Graph Traversal as a Model of Context-Dependent Recall"** — Studies the conditions under which dormant memory recovery succeeds, analogising to Tulving's encoding specificity principle.

4. **"Memory Shortcuts as Emergent Schemas: Modelling Gist Extraction Through Dormancy-Induced Graph Compression"** — Formal analysis of the shortcut formation mechanism and its relationship to semantic compression in human long-term memory.

5. **"Retrieval Efficiency Trade-offs in Biologically-Inspired vs. Vector-Based Memory Systems for AI Agents"** — Benchmark study across recall quality, latency, and associative breadth metrics.

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Run tests (`python -m pytest tests/ -v`)
4. Submit a pull request with a clear description

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*NeuroStore is a research prototype. It is not production memory management software. It is an exploration of what computation might look like if we took the biology of memory seriously.*

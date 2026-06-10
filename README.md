# ternary-fault-tree

**Fault tree analysis where every node is healthy, degraded, or failed — not just broken or not.**

Traditional fault trees force you into binary thinking: a component either works or it doesn't. But real GPU hardware doesn't fail like a light switch. A VRAM bank with bit errors still serves requests — just slower. A streaming multiprocessor with one bad lane still computes — with reduced throughput. The degraded state matters.

This crate models fault propagation through systems where nodes exist in three states: **+1 (healthy)**, **0 (degraded)**, or **-1 (failed)**. AND gates, OR gates, and ternary voting gates propagate these states bottom-up through arbitrarily nested fault trees.

## The Insight

Binary fault trees lose information at every gate. When one input to an AND gate fails, binary logic says "output failed." But the output isn't *failed* — it's *degraded*. Collapsing {healthy, degraded, failed} into {working, broken} throws away exactly the signal you need for graceful degradation.

Ternary fault trees preserve that signal. The degraded state propagates correctly: one failed child of an AND gate → degraded output, not failed. One healthy child of an OR gate → still degraded, not healthy. This matches how GPU clusters actually behave under partial failure.

## Quick Start

```toml
[dependencies]
ternary-fault-tree = "0.1.0"
```

```rust
use ternary_fault_tree::*;

// Build a GPU fault tree:
//   TOP (OR)
//   ├── MEM_AND: both VRAM banks must fail
//   │   ├── VRAM1 (5% failure, 10% degraded)
//   │   └── VRAM2 (5% failure, 10% degraded)
//   └── COMPUTE_AND: both SMs must fail
//       ├── SM1 (2% failure, 5% degraded)
//       └── SM2 (2% failure, 5% degraded)

let mut ft = FaultTree::new("TOP".into());
ft.add_basic("VRAM1", 0.05, 0.10);
ft.add_basic("VRAM2", 0.05, 0.10);
ft.add_basic("SM1",   0.02, 0.05);
ft.add_basic("SM2",   0.02, 0.05);
ft.add_gate("MEM_AND",     Gate::AND, vec!["VRAM1", "VRAM2"]);
ft.add_gate("COMPUTE_AND", Gate::AND, vec!["SM1", "SM2"]);
ft.add_gate("TOP",         Gate::OR,  vec!["MEM_AND", "COMPUTE_AND"]);

// Deterministic propagation with explicit leaf states
let mut leafs = HashMap::new();
leafs.insert("VRAM1".into(), TernaryState::Failed);
let states = ft.propagate_with_states(&leafs)?;
// MEM_AND → Degraded (one child failed, not all)
// TOP → Degraded

// Top-event probability
let p_fail = ft.top_event_probability(TernaryState::Failed)?;
// ≈ 0.0029 (both VRAM or both SM must fail)

// Minimal cut sets
let cuts = ft.minimal_cut_sets()?;
// [{VRAM1, VRAM2}, {SM1, SM2}]

// Monte Carlo reliability estimation
let reliability = ft.reliability(100_000)?;
// ≈ 0.997
```

## Architecture

```
┌──────────────────────────────────────┐
│           FaultTree                   │
│  ┌────────────────────────────────┐  │
│  │  HashMap<String, FaultNode>    │  │
│  │  ┌──────────┐  ┌───────────┐  │  │
│  │  │  Basic   │  │   Gate    │  │  │
│  │  │ (leaf)   │  │ (internal)│  │  │
│  │  └──────────┘  └───────────┘  │  │
│  └────────────────────────────────┘  │
│                                      │
│  propagate()  ── bottom-up resolve   │
│  propagate_with_states() ── override  │
│  top_event_probability() ── analytic  │
│  minimal_cut_sets() ── brute-force   │
│  monte_carlo() ── stochastic          │
└──────────────────────────────────────┘
```

The tree is stored as a flat `HashMap` keyed by node name. Propagation walks the DAG bottom-up, memoizing results in a cache. Circular references are caught implicitly by the recursive resolver (it'll stack overflow — this is by design, fault trees are DAGs).

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `TernaryState` | `Healthy` (+1), `Degraded` (0), `Failed` (-1) |
| `Gate` | `AND`, `OR`, or `TERNARY_VOTE { threshold, degraded_threshold }` |
| `FaultNode` | `Basic { name, failure_prob, degraded_prob }` or `Gate { name, gate, children }` |
| `FaultTree` | The tree itself: nodes + root name |

### Gate Semantics

**AND gate** — output is Failed only when *all* children are Failed. With at least one Degraded child, output is Degraded. Otherwise Healthy.

**OR gate** — output is Healthy only when *all* children are Healthy. With at least one Failed child, output is Failed. Otherwise Degraded.

**TERNARY_VOTE** — sums child values (+1/0/-1). If sum ≥ threshold → Healthy. If sum ≥ degraded_threshold → Degraded. Otherwise Failed. This is the ternary generalization of k-of-n voting.

### FaultTree Methods

```rust
fn new(root: String) -> FaultTree
fn add_basic(&mut self, name: &str, failure_prob: f64, degraded_prob: f64)
fn add_gate(&mut self, name: &str, gate: Gate, children: Vec<&str>)
fn propagate(&self) -> Result<HashMap<String, TernaryState>, String>
fn propagate_with_states(&self, leaf_states: &HashMap<String, TernaryState>) -> Result<...>
fn evaluate_gate(&self, gate: &Gate, children: &[TernaryState]) -> TernaryState
fn top_event_probability(&self, severity: TernaryState) -> Result<f64, String>
fn minimal_cut_sets(&self) -> Result<Vec<BTreeSet<String>>, String>
fn monte_carlo(&self, iterations: u64) -> Result<(f64, f64, f64), String>
fn reliability(&self, iterations: u64) -> Result<f64, String>
```

### TernaryState Methods

```rust
fn value(self) -> i8           // +1, 0, or -1
fn is_operational(self) -> bool // true if not Failed
```

## Real-World Example: Multi-GPU Training Cluster

```rust
// A training job across 4 GPUs with NCCL interconnect
let mut ft = FaultTree::new("JOB_FAILED".into());

// Each GPU has independent failure modes
for i in 0..4 {
    ft.add_basic(&format!("GPU{}_VRAM"), 0.03, 0.08);
    ft.add_basic(&format!("GPU{}_SM"),   0.01, 0.03);
    ft.add_basic(&format!("GPU{}_THERM"), 0.05, 0.12);
    ft.add_gate(&format!("GPU{}_OK"), Gate::AND, vec![
        &format!("GPU{}_VRAM"), &format!("GPU{}_SM"), &format!("GPU{}_THERM")
    ]);
}

// NCCL links
ft.add_basic("NCCL_LINK", 0.01, 0.02);

// Job needs all GPUs + interconnect
ft.add_gate("JOB_FAILED", Gate::OR, vec![
    "GPU0_OK", "GPU1_OK", "GPU2_OK", "GPU3_OK", "NCCL_LINK"
]);

// What's the reliability over 100k simulated runs?
let rel = ft.reliability(100_000)?;
println!("Training job reliability: {:.4}", rel);
```

## Analysis Methods

### Deterministic Propagation

`propagate()` and `propagate_with_states()` walk the tree bottom-up. Leaf nodes default to Healthy (use `propagate_with_states` to override). Each gate combines its children according to the ternary rules above.

### Top-Event Probability

Analytically computes the probability that the root node reaches at least the given severity. Uses recursive probability composition: AND gates multiply child probabilities, OR gates use inclusion-exclusion. For TERNARY_VOTE, approximates with the OR upper bound.

### Minimal Cut Sets

Brute-force enumeration of the smallest sets of basic events whose simultaneous failure causes the top event to fail. Starts with single-element sets and grows until all minimal sets are found. Practical for trees with up to ~20 basic events.

### Monte Carlo Simulation

Samples each basic event independently (using its `failure_prob` and `degraded_prob`), propagates through the tree, and counts outcomes. Returns `(healthy_fraction, degraded_fraction, failed_fraction)`. The `reliability()` convenience method returns `1 - failed_fraction`.

## Performance

- **Propagation**: O(n) where n = number of nodes (memoized)
- **Top-event probability**: O(n) with caching
- **Minimal cut sets**: O(2^b × n) where b = number of basic events (brute force)
- **Monte Carlo**: O(iterations × n)

The cut set algorithm is the bottleneck for large trees. For production use with many basic events, consider replacing the brute-force approach with MOCUS or ZBDD-based algorithms.

## Ecosystem

Part of the **ternary fleet** — a collection of crates built around the {-1, 0, +1} algebra:

- **ternary-antidote** — CRDTs with ternary merge outcomes
- **ternary-shard** — sharded ternary data distribution
- **ternary-watermark** — model provenance via ternary fingerprints
- **ternary-signal-flow** — signal processing pipelines

## Open Questions

- **Cut set scaling**: The brute-force algorithm is correct but exponential. A ZBDD-based approach would handle much larger trees.
- **Correlated failures**: Current model assumes independent basic events. Real GPU failures correlate (power supply, thermal throttling).
- **Dynamic fault trees**: Adding sequence-dependent gates (priority AND, spare gates) would model cold/warm/hot redundancy.
- **Continuous-time Markov chains**: The current model is static. Extending to CTMCs would capture time-dependent failure rates.

## Stats

| Metric | Value |
|--------|-------|
| Tests | 10 |
| Lines of Rust | 632 |
| Public API | 16 items |
| `forbid(unsafe_code)` | No |
| `no_std` | No |

## License

Apache-2.0

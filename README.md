# ternary-fault-tree



## Why This Matters

# ternary-fault-tree
Fault tree analysis for GPU systems with ternary node states.
Each node can be in one of three states:
- `+1` — Healthy (fully operational)
- `0`  — Degraded (partially functional)

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub enum TernaryState
pub fn value
pub fn is_operational
pub enum Gate
pub enum FaultNode
pub struct FaultTree
pub fn new
pub fn add_basic
pub fn add_gate
pub fn propagate
pub fn propagate_with_states
pub fn evaluate_gate
```

## Usage

```toml
[dependencies]
ternary-fault-tree = "0.1.0"
```

```rust
use ternary_fault_tree::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/ternary-fault-tree.git
cd ternary-fault-tree
cargo test    # 10 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 10 |
| Lines of Rust | 632 |
| Public API | 16 items |

## License

Apache-2.0

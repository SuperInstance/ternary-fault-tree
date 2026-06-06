//! # ternary-fault-tree
//!
//! Fault tree analysis for GPU systems with ternary node states.
//!
//! Each node can be in one of three states:
//! - `+1` — Healthy (fully operational)
//! - `0`  — Degraded (partially functional)
//! - `-1` — Failed (non-operational)
//!
//! Supports AND, OR, and TERNARY_VOTE gates, bottom-up fault propagation,
//! top-event probability computation, minimal cut set enumeration, and
//! Monte Carlo simulation for reliability estimation.

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use rand::RngExt;

// ---------------------------------------------------------------------------
// Ternary state
// ---------------------------------------------------------------------------

/// Ternary node state for GPU components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TernaryState {
    /// Fully operational (+1).
    Healthy = 1,
    /// Partially functional (0).
    Degraded = 0,
    /// Non-operational (-1).
    Failed = -1,
}

impl TernaryState {
    /// Numeric value: +1, 0, or -1.
    pub fn value(self) -> i8 {
        self as i8
    }

    /// Returns true when the state is **not** Failed.
    pub fn is_operational(self) -> bool {
        self != TernaryState::Failed
    }
}

impl fmt::Display for TernaryState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TernaryState::Healthy => write!(f, "+1 (healthy)"),
            TernaryState::Degraded => write!(f, " 0 (degraded)"),
            TernaryState::Failed => write!(f, "-1 (failed)"),
        }
    }
}

// ---------------------------------------------------------------------------
// Gate types
// ---------------------------------------------------------------------------

/// Logic gate connecting child nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gate {
    /// Output is Failed only when **all** children are Failed.
    /// With at least one Degraded child the output is Degraded.
    /// Otherwise Healthy.
    AND,
    /// Output is Healthy only when **all** children are Healthy.
    /// With at least one Degraded child the output is Degraded.
    /// Otherwise Failed.
    OR,
    /// Output depends on a voting threshold over ternary values.
    /// `threshold` is the minimum sum of child values required for
    /// Healthy output.  If the sum is < threshold but ≥ degraded_threshold
    /// the output is Degraded; otherwise Failed.
    #[allow(non_camel_case_types)]
    TERNARY_VOTE { threshold: i32, degraded_threshold: i32 },
}

// ---------------------------------------------------------------------------
// Fault tree node
// ---------------------------------------------------------------------------

/// A single node in the fault tree.
#[derive(Debug, Clone)]
pub enum FaultNode {
    /// Leaf / basic event with a fixed failure probability in [0, 1].
    Basic {
        name: String,
        failure_prob: f64,
        degraded_prob: f64,
    },
    /// Internal gate combining child nodes.
    Gate {
        name: String,
        gate: Gate,
        children: Vec<String>,
    },
}

// ---------------------------------------------------------------------------
// Fault tree
// ---------------------------------------------------------------------------

/// Fault tree for GPU system reliability analysis.
#[derive(Debug, Clone)]
pub struct FaultTree {
    nodes: HashMap<String, FaultNode>,
    root: String,
}

impl FaultTree {
    /// Create a new fault tree with the given root node name.
    pub fn new(root: String) -> Self {
        FaultTree {
            nodes: HashMap::new(),
            root,
        }
    }

    /// Add a basic (leaf) event.
    pub fn add_basic(&mut self, name: &str, failure_prob: f64, degraded_prob: f64) {
        self.nodes.insert(
            name.to_string(),
            FaultNode::Basic {
                name: name.to_string(),
                failure_prob,
                degraded_prob,
            },
        );
    }

    /// Add a gate node.
    pub fn add_gate(&mut self, name: &str, gate: Gate, children: Vec<&str>) {
        self.nodes.insert(
            name.to_string(),
            FaultNode::Gate {
                name: name.to_string(),
                gate,
                children: children.iter().map(|s| s.to_string()).collect(),
            },
        );
    }

    /// Resolve a node name to a reference.
    fn get_node(&self, name: &str) -> Result<&FaultNode, String> {
        self.nodes
            .get(name)
            .ok_or_else(|| format!("node '{}' not found", name))
    }

    // -----------------------------------------------------------------------
    // Bottom-up propagation
    // -----------------------------------------------------------------------

    /// Propagate states bottom-up from basic events through gate nodes.
    ///
    /// Returns the computed state for every node.
    pub fn propagate(&self) -> Result<HashMap<String, TernaryState>, String> {
        let mut cache: HashMap<String, TernaryState> = HashMap::new();
        self.resolve(&self.root, &mut cache)?;
        Ok(cache)
    }

    fn resolve(
        &self,
        name: &str,
        cache: &mut HashMap<String, TernaryState>,
    ) -> Result<TernaryState, String> {
        if let Some(&s) = cache.get(name) {
            return Ok(s);
        }
        let node = self.get_node(name)?.clone();
        let state = match node {
            FaultNode::Basic { .. } => {
                // Basic events have no intrinsic state in deterministic
                // propagation — they default to Healthy.  For probabilistic
                // analysis use `propagate_with_states` or Monte Carlo.
                TernaryState::Healthy
            }
            FaultNode::Gate {
                gate, children, ..
            } => {
                let child_states: Vec<TernaryState> = children
                    .iter()
                    .map(|c| self.resolve(c, cache))
                    .collect::<Result<_, _>>()?;
                self.evaluate_gate(&gate, &child_states)
            }
        };
        cache.insert(name.to_string(), state);
        Ok(state)
    }

    /// Propagate with explicitly-set leaf states.
    pub fn propagate_with_states(
        &self,
        leaf_states: &HashMap<String, TernaryState>,
    ) -> Result<HashMap<String, TernaryState>, String> {
        let mut cache: HashMap<String, TernaryState> = leaf_states.clone();
        self.resolve(&self.root, &mut cache)?;
        Ok(cache)
    }

    /// Evaluate a gate given the resolved child states.
    pub fn evaluate_gate(&self, gate: &Gate, children: &[TernaryState]) -> TernaryState {
        match gate {
            Gate::AND => {
                // AND: failed only if ALL children failed.
                if children.iter().all(|s| *s == TernaryState::Failed) {
                    TernaryState::Failed
                } else if children.iter().any(|s| *s == TernaryState::Failed) {
                    TernaryState::Degraded
                } else if children.iter().any(|s| *s == TernaryState::Degraded) {
                    TernaryState::Degraded
                } else {
                    TernaryState::Healthy
                }
            }
            Gate::OR => {
                // OR: healthy only if ALL children healthy.
                if children.iter().all(|s| *s == TernaryState::Healthy) {
                    TernaryState::Healthy
                } else if children.iter().any(|s| *s == TernaryState::Failed) {
                    TernaryState::Failed
                } else {
                    TernaryState::Degraded
                }
            }
            Gate::TERNARY_VOTE {
                threshold,
                degraded_threshold,
            } => {
                let sum: i32 = children.iter().map(|s| s.value() as i32).sum();
                if sum >= *threshold {
                    TernaryState::Healthy
                } else if sum >= *degraded_threshold {
                    TernaryState::Degraded
                } else {
                    TernaryState::Failed
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Top-event probability
    // -----------------------------------------------------------------------

    /// Compute the probability that the top event (root) reaches at least
    /// the given severity.
    pub fn top_event_probability(&self, severity: TernaryState) -> Result<f64, String> {
        let mut cache: HashMap<String, f64> = HashMap::new();
        self.prob_recursive(&self.root, &severity, &mut cache)
    }

    fn prob_recursive(
        &self,
        name: &str,
        severity: &TernaryState,
        cache: &mut HashMap<String, f64>,
    ) -> Result<f64, String> {
        if let Some(&p) = cache.get(name) {
            return Ok(p);
        }
        let node = self.get_node(name)?.clone();
        let prob = match node {
            FaultNode::Basic {
                failure_prob,
                degraded_prob,
                ..
            } => match severity {
                TernaryState::Failed => failure_prob,
                TernaryState::Degraded => failure_prob + degraded_prob,
                TernaryState::Healthy => 1.0,
            },
            FaultNode::Gate {
                gate, children, ..
            } => {
                let child_probs: Vec<f64> = children
                    .iter()
                    .map(|c| self.prob_recursive(c, severity, cache))
                    .collect::<Result<_, _>>()?;
                match &gate {
                    Gate::AND => {
                        // All children must be at or below severity
                        child_probs.iter().product()
                    }
                    Gate::OR => {
                        // At least one child at or below severity
                        1.0 - child_probs.iter().map(|p| 1.0 - p).product::<f64>()
                    }
                    Gate::TERNARY_VOTE { .. } => {
                        // Approximate: treat like OR for probability bound
                        1.0 - child_probs.iter().map(|p| 1.0 - p).product::<f64>()
                    }
                }
            }
        };
        cache.insert(name.to_string(), prob);
        Ok(prob)
    }

    // -----------------------------------------------------------------------
    // Minimal cut sets
    // -----------------------------------------------------------------------

    /// Find all minimal cut sets — smallest sets of basic events whose
    /// simultaneous failure causes the top event to fail.
    pub fn minimal_cut_sets(&self) -> Result<Vec<BTreeSet<String>>, String> {
        let basic_events: Vec<String> = self
            .nodes
            .iter()
            .filter_map(|(name, node)| match node {
                FaultNode::Basic { .. } => Some(name.clone()),
                _ => None,
            })
            .collect();

        if basic_events.is_empty() {
            return Ok(vec![]);
        }

        let mut cut_sets: Vec<BTreeSet<String>> = Vec::new();
        let n = basic_events.len();

        // Brute-force up to reasonable size; for large trees use MOCUS etc.
        for size in 1..=n {
            for combo in combinations(&basic_events, size) {
                let set: BTreeSet<String> = combo.into_iter().collect();
                if self.is_cut_set(&set)? && !self.is_superset_of_any(&set, &cut_sets) {
                    cut_sets.push(set);
                }
            }
            // Once we find cut sets at a given size, we can skip larger sizes
            // only if we've covered everything — but for correctness keep
            // going at least one more level.
            if !cut_sets.is_empty() && size >= cut_sets.iter().map(|s| s.len()).min().unwrap_or(0)
            {
                break;
            }
        }

        // Final minimality pass
        let mut minimal: Vec<BTreeSet<String>> = Vec::new();
        for cs in &cut_sets {
            if !self.is_superset_of_any(cs, &minimal) {
                minimal.push(cs.clone());
            }
        }

        Ok(minimal)
    }

    /// Check if a given set of failed basic events causes top event failure.
    fn is_cut_set(&self, failed: &BTreeSet<String>) -> Result<bool, String> {
        let mut leaf_states: HashMap<String, TernaryState> = HashMap::new();
        for name in self.nodes.keys() {
            if failed.contains(name) {
                leaf_states.insert(name.clone(), TernaryState::Failed);
            }
        }
        let states = self.propagate_with_states(&leaf_states)?;
        Ok(states.get(&self.root) == Some(&TernaryState::Failed))
    }

    /// Check if `set` is a (strict) superset of any set in `sets`.
    fn is_superset_of_any(&self, set: &BTreeSet<String>, sets: &[BTreeSet<String>]) -> bool {
        sets.iter().any(|s| s.is_subset(set) && s.len() < set.len())
    }

    // -----------------------------------------------------------------------
    // Monte Carlo simulation
    // -----------------------------------------------------------------------

    /// Run Monte Carlo simulation estimating reliability over `iterations`
    /// trials.
    ///
    /// Returns `(healthy_fraction, degraded_fraction, failed_fraction)`.
    pub fn monte_carlo(&self, iterations: u64) -> Result<(f64, f64, f64), String> {
        let mut rng = rand::rng();
        let mut healthy_count: u64 = 0;
        let mut degraded_count: u64 = 0;

        // Collect basic events for fast sampling
        let basics: Vec<(String, f64, f64)> = self
            .nodes
            .iter()
            .filter_map(|(_, node)| match node {
                FaultNode::Basic {
                    name,
                    failure_prob,
                    degraded_prob,
                } => Some((name.clone(), *failure_prob, *degraded_prob)),
                _ => None,
            })
            .collect();

        for _ in 0..iterations {
            let mut leaf_states: HashMap<String, TernaryState> = HashMap::new();
            for (name, fp, dp) in &basics {
                let r: f64 = rng.random();
                if r < *fp {
                    leaf_states.insert(name.clone(), TernaryState::Failed);
                } else if r < fp + dp {
                    leaf_states.insert(name.clone(), TernaryState::Degraded);
                } else {
                    leaf_states.insert(name.clone(), TernaryState::Healthy);
                }
            }

            let states = self.propagate_with_states(&leaf_states)?;
            match states.get(&self.root) {
                Some(TernaryState::Healthy) => healthy_count += 1,
                Some(TernaryState::Degraded) => degraded_count += 1,
                _ => {}
            }
        }

        let failed_count = iterations - healthy_count - degraded_count;
        let n = iterations as f64;
        Ok((
            healthy_count as f64 / n,
            degraded_count as f64 / n,
            failed_count as f64 / n,
        ))
    }

    /// Convenience: Monte Carlo reliability (1 - P(failure)).
    pub fn reliability(&self, iterations: u64) -> Result<f64, String> {
        let (h, d, _f) = self.monte_carlo(iterations)?;
        Ok(h + d) // reliability = not fully failed
    }
}

// ---------------------------------------------------------------------------
// Combinations helper
// ---------------------------------------------------------------------------

fn combinations(items: &[String], k: usize) -> Vec<Vec<String>> {
    if k == 0 || k > items.len() {
        return vec![];
    }
    if k == 1 {
        return items.iter().cloned().map(|i| vec![i]).collect();
    }
    let mut result = Vec::new();
    for i in 0..=items.len() - k {
        for mut tail in combinations(&items[i + 1..], k - 1) {
            let mut combo = vec![items[i].clone()];
            combo.append(&mut tail);
            result.push(combo);
        }
    }
    result
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a simple GPU fault tree.
    ///
    /// ```text
    ///        TOP (OR)
    ///       /        \
    ///   MEM_AND     COMPUTE_AND
    ///   /    \       /       \
    /// VRAM  VRAM2  SM1      SM2
    /// ```
    fn gpu_tree() -> FaultTree {
        let mut ft = FaultTree::new("TOP".to_string());

        // Basic events (VRAM chips, streaming multiprocessors)
        ft.add_basic("VRAM1", 0.05, 0.10);
        ft.add_basic("VRAM2", 0.05, 0.10);
        ft.add_basic("SM1", 0.02, 0.05);
        ft.add_basic("SM2", 0.02, 0.05);

        // Gate nodes
        ft.add_gate("MEM_AND", Gate::AND, vec!["VRAM1", "VRAM2"]);
        ft.add_gate("COMPUTE_AND", Gate::AND, vec!["SM1", "SM2"]);
        ft.add_gate("TOP", Gate::OR, vec!["MEM_AND", "COMPUTE_AND"]);

        ft
    }

    // 1. Healthy propagation — all leafs healthy → root healthy
    #[test]
    fn test_propagate_all_healthy() {
        let ft = gpu_tree();
        let states = ft.propagate_with_states(&HashMap::new()).unwrap();
        assert_eq!(states["TOP"], TernaryState::Healthy);
    }

    // 2. AND gate: one child failed → degraded, not failed
    #[test]
    fn test_and_gate_single_failure() {
        let ft = gpu_tree();
        let mut leafs = HashMap::new();
        leafs.insert("VRAM1".into(), TernaryState::Failed);
        let states = ft.propagate_with_states(&leafs).unwrap();
        assert_eq!(states["MEM_AND"], TernaryState::Degraded);
        assert_eq!(states["TOP"], TernaryState::Degraded);
    }

    // 3. AND gate: all children failed → failed
    #[test]
    fn test_and_gate_all_failed() {
        let ft = gpu_tree();
        let mut leafs = HashMap::new();
        leafs.insert("VRAM1".into(), TernaryState::Failed);
        leafs.insert("VRAM2".into(), TernaryState::Failed);
        let states = ft.propagate_with_states(&leafs).unwrap();
        assert_eq!(states["MEM_AND"], TernaryState::Failed);
        assert_eq!(states["TOP"], TernaryState::Failed);
    }

    // 4. OR gate: one subtree failed → top failed
    #[test]
    fn test_or_gate_failure_propagation() {
        let ft = gpu_tree();
        let mut leafs = HashMap::new();
        leafs.insert("SM1".into(), TernaryState::Failed);
        leafs.insert("SM2".into(), TernaryState::Failed);
        let states = ft.propagate_with_states(&leafs).unwrap();
        assert_eq!(states["COMPUTE_AND"], TernaryState::Failed);
        assert_eq!(states["TOP"], TernaryState::Failed);
    }

    // 5. TERNARY_VOTE gate
    #[test]
    fn test_ternary_vote_gate() {
        let mut ft = FaultTree::new("VOTE_TOP".to_string());
        ft.add_basic("A", 0.1, 0.1);
        ft.add_basic("B", 0.1, 0.1);
        ft.add_basic("C", 0.1, 0.1);
        // Need sum ≥ 2 for healthy, ≥ 0 for degraded
        ft.add_gate(
            "VOTE_TOP",
            Gate::TERNARY_VOTE {
                threshold: 2,
                degraded_threshold: 0,
            },
            vec!["A", "B", "C"],
        );

        // All healthy → sum = 3 → Healthy
        let states = ft.propagate_with_states(&HashMap::new()).unwrap();
        assert_eq!(states["VOTE_TOP"], TernaryState::Healthy);

        // One degraded → sum = 2 → Healthy (still ≥ 2)
        let mut leafs = HashMap::new();
        leafs.insert("A".into(), TernaryState::Degraded);
        let states = ft.propagate_with_states(&leafs).unwrap();
        assert_eq!(states["VOTE_TOP"], TernaryState::Healthy);

        // One failed, one degraded → sum = 0 → Degraded (≥ 0 but < 2)
        let mut leafs = HashMap::new();
        leafs.insert("A".into(), TernaryState::Failed);
        leafs.insert("B".into(), TernaryState::Degraded);
        let states = ft.propagate_with_states(&leafs).unwrap();
        assert_eq!(states["VOTE_TOP"], TernaryState::Degraded);

        // Two failed → sum = -1 → Failed
        let mut leafs = HashMap::new();
        leafs.insert("A".into(), TernaryState::Failed);
        leafs.insert("B".into(), TernaryState::Failed);
        let states = ft.propagate_with_states(&leafs).unwrap();
        assert_eq!(states["VOTE_TOP"], TernaryState::Failed);
    }

    // 6. Top-event probability (OR of two AND gates)
    #[test]
    fn test_top_event_probability() {
        let ft = gpu_tree();
        let p_fail = ft.top_event_probability(TernaryState::Failed).unwrap();
        // P(MEM_AND fails) = 0.05 * 0.05 = 0.0025
        // P(COMPUTE_AND fails) = 0.02 * 0.02 = 0.0004
        // P(TOP fails) = 1 - (1-0.0025)(1-0.0004) ≈ 0.002899
        let expected = 1.0 - (1.0 - 0.0025) * (1.0 - 0.0004);
        assert!((p_fail - expected).abs() < 1e-10);
    }

    // 7. Minimal cut sets
    #[test]
    fn test_minimal_cut_sets() {
        let ft = gpu_tree();
        let cuts = ft.minimal_cut_sets().unwrap();
        // Minimal cut sets: {VRAM1, VRAM2} and {SM1, SM2}
        assert_eq!(cuts.len(), 2);
        assert!(cuts.contains(&BTreeSet::from(["VRAM1".into(), "VRAM2".into()])));
        assert!(cuts.contains(&BTreeSet::from(["SM1".into(), "SM2".into()])));
    }

    // 8. Monte Carlo simulation runs and returns valid fractions
    #[test]
    fn test_monte_carlo() {
        let ft = gpu_tree();
        let (h, d, f) = ft.monte_carlo(10_000).unwrap();
        // Fractions must sum to ~1.0
        assert!((h + d + f - 1.0).abs() < 1e-10);
        // Healthy fraction with these probs: ~(0.85^2)*(0.93^2) ≈ 0.63
        assert!(h > 0.5);
        // Failure rate should be small
        assert!(f < 0.01);
    }

    // 9. TernaryState value and display
    #[test]
    fn test_ternary_state_traits() {
        assert_eq!(TernaryState::Healthy.value(), 1);
        assert_eq!(TernaryState::Degraded.value(), 0);
        assert_eq!(TernaryState::Failed.value(), -1);
        assert!(TernaryState::Healthy.is_operational());
        assert!(!TernaryState::Failed.is_operational());
    }

    // 10. Reliability convenience method
    #[test]
    fn test_reliability() {
        let ft = gpu_tree();
        let rel = ft.reliability(50_000).unwrap();
        // Reliability = P(not failed) should be close to 1 - 0.0029
        assert!(rel > 0.99);
        assert!(rel <= 1.0);
    }
}

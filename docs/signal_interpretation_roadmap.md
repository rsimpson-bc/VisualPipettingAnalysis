# Signal Interpretation — Future Considerations

This document captures design ideas discussed but not yet implemented, so they
are not forgotten.

---

## 1. Forward-Backward Algorithm (per-row marginal probabilities)

**What it is:**  
The current Viterbi decoder gives the single most-probable state *path* through
the sequence.  The forward-backward (FB) algorithm computes the per-row marginal
probability `P(state = S at row i | all observations)` — a full posterior
distribution over states at each row, not just the MAP path.

**Why it matters:**  
FB probabilities can be used as a continuous, probabilistic weight on boundary
densities:

```
tip_bottom_weight[i] = P(state[i-1] = LIQ) * P(state[i] = BELOW)
meniscus_weight[i]   = P(state[i-1] = GAS) * P(state[i] = LIQ)
```

This is softer and more principled than the current approach (Gaussian smear
around Viterbi transition points), and would not catastrophically zero out a
correct boundary peak if the Viterbi path is slightly wrong.

**Implementation sketch:**
- Forward pass: `alpha[i, s] = P(obs[0..i], state[i]=s)`  
- Backward pass: `beta[i, s] = P(obs[i+1..N] | state[i]=s)`  
- Marginals: `gamma[i, s] ∝ alpha[i, s] * beta[i, s]`  
- Pairwise marginals: `xi[i, s, t] ∝ alpha[i,s] * T[s,t] * obs[i+1,t] * beta[i+1,t]`

The transition matrix is already encoded in `_TRANSITION_COST`; would need
converting to log-probabilities.

**Complexity:** O(N × S²) — same as Viterbi, negligible overhead.

---

## 2. State-Informed Boundary Masking (hard version)

**What it is:**  
After Viterbi decodes the state sequence, hard-mask boundary densities to zero
outside the transition zone:

- `tip_bottom_density[i] = 0` wherever `state_sequence[i] != BELOW` and
  `state_sequence[i-1] != LIQ`  
- Similarly for meniscus.

**Trade-off:**  
If the state decode is slightly wrong (e.g., a noisy GAS region is labelled LIQ),
the correct boundary peak could be eliminated entirely.  The current soft
(Gaussian-weighted) approach avoids this — it de-emphasises rather than zeroes.

**When to revisit:**  
If the soft approach produces ambiguous results in cases where the state decode
is clearly correct, try this as a post-processing option gated by a checkbox.

---

## 3. Full Cross-Preset Aggregation with Confidence Weighting

**What it is:**  
Currently the "Combined" strip line is a simple **unweighted mean** of all
presets' tip_bottom and meniscus density curves.

A better aggregation would weight each preset's contribution by a confidence
score — e.g., the sharpness of its density peak (inverse of density entropy, or
the peak-to-background ratio).

**Implementation sketch:**
```python
for r in results.values():
    sharpness = r.tip_bottom_density.max() / (r.tip_bottom_density.mean() + 1e-9)
    combined_tb += sharpness * r.tip_bottom_density
combined_tb /= total_weight
```

**Also consider:**  
Showing the combined density curve in the strip view (not just the argmax line),
so you can see whether the presets agree (narrow combined peak) or disagree
(broad or bimodal combined peak).

---

## 4. Cross-Preset Coupling via Shared State Scores

**What it is:**  
Each preset currently runs `interpret_signals` independently — its `high_signal`
→ `liquid_in_tip` evidence does not influence another preset's `peak` →
`tip_bottom` density.

A future design could accumulate state scores from *all* presets into a single
shared accumulator before running Viterbi once, then use the combined state
sequence to weight all presets' boundary densities.

**Trade-off:**  
Loses per-preset isolatability (you can no longer compare how each preset
individually performs); gains a single, maximally-informed state estimate.

**Suggested approach:**  
Implement as an opt-in "Fused" mode alongside the existing per-preset mode.

---

## 5. Dependency Structure Between Rules

**What it is:**  
Today, `gas_in_tip` / `liquid_in_tip` / `below_tip` state rules and
`tip_bottom` / `meniscus` boundary rules are **entirely parallel accumulators**
that do not interact within a single `interpret_signals` call.  The Viterbi
transition costs provide implicit coupling (e.g., BELOW can only follow LIQ),
but there is no direct "if we know there is liquid here, the tip_bottom must be
below it" logic.

**Design options considered:**
- **Mode 1 (cross-rule awareness):** have boundary rules check where the state
  accumulator is high before voting.  Risky — tangled parameters, noise below
  the tip could still shift the estimate.
- **Mode 2 (location_prior):** encode spatial expectations as per-rule prior
  curves (`location_prior` key, already implemented).  Safer and tuneable.
- **Mode 3 (transition weights):** post-process boundary densities using the
  Viterbi transition location (already implemented as of May 2026).
- **Mode 4 (FB marginals):** use forward-backward marginals as the weight
  (see item 1 above).

The current implementation uses Mode 2 + Mode 3.  Mode 4 is the natural next
step if Mode 3 proves insufficient in practice.

---

*Last updated: May 2026.*

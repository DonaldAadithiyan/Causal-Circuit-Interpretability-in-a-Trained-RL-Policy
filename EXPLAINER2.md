# EXPLAINER2 — Experiment 2: Causal Graph Invariance Detection (W-Matrix Method)

*Read EXPLAINER.md first. This document continues from Experiment 1.*

---

## 1. Why This Experiment Followed From Experiment 1

Experiment 1 found that "goal" SAE features deactivated ~157 steps before episodic reward degraded (k = 157.8). But three things were unresolved:

1. The causal graph G* in Experiment 1 was built from cosine similarity between SAE decoder directions — a structural proxy, not an actual causal measurement.
2. No live graph (G_live) was computed at deployment, and the five structural invariances (I1–I5) were never checked.
3. The detection was purely *perceptual* — the goal cell was visually absent at the fixed training position, so features detecting it read zero immediately. This only works when the failure is visually obvious at the current frame.

Experiment 2 set out to build G* correctly with actual causal intervention, implement G_live, and test whether the graph-level signal catches the failure *earlier* than raw activation monitoring — especially in a harder **graded shift** setting where the goal is only partially displaced and still visible.

This experiment went through three stages: an EAP attempt that failed, a methodological fix (the W-matrix), and a re-examination of the SAE that produced the most important finding of the entire research programme.

---

## 2. Stage 1 — EAP Failed

The first attempt computed G_live using Edge Attribution Patching (EAP): gradient × activation attribution from the action logits back to each SAE feature. This is the standard fast approximation in mechanistic interpretability.

**EAP validation against ground-truth activation patching gave Pearson r = 0.146.** This is far below the 0.5 usability threshold.

The reason is architectural. The gradient path runs: action logits → policy head → SAE decoder (256 → 1024 expansion) → feature activations. The over-complete decoder dilutes the gradient signal so badly that the gradient-based attribution barely correlates with the true causal effect of zeroing a feature. EAP works for transformers because attention gradients are structured; it does not work for an over-complete SAE decoder feeding a ReLU CNN policy head.

EAP was abandoned. The invariance scores built on it (k_graph ≈ k_activation ≈ 128.7 in the first run) were therefore unreliable and are not the basis of any conclusion here.

---

## 3. Stage 2 — The W-Matrix Fix

The fix removes gradients entirely. Inter-feature causal influence is read directly from the SAE weight geometry:

```
W = D^T · W_enc^T          (shape: hidden × hidden)
```

- D = decoder weight (256 × 384): columns are the directions each feature writes into the representation.
- W_enc = encoder weight (384 × 256): rows are the directions each feature reads from the representation.
- W[i, j] = (decoder direction of i) · (encoder direction of j) = how much feature i's presence pushes feature j toward activation.

This is computed once, in a single matrix multiply, with no forward or backward passes. G_live at any step is just `W[active][:, active] * activations[active]` — pure indexing.

**W validated against activation patching at Pearson r = 0.893** — a decisive pass (the strong-success threshold was 0.7). The W-matrix is an excellent gradient-free proxy for true causal influence on this architecture, where EAP scored 0.146.

This is the central methodological contribution of Experiment 2: **for over-complete SAEs on CNN policies, inter-feature causal edges should be read from weight geometry, not estimated with gradients.**

---

## 4. Stage 2b — Fixing the SAE First (The Dead Feature Problem)

The Experiment 1 SAE had 785/1024 dead features (77%). Before trusting W, the SAE was retrained with **Anthropic-style neuron resampling**: any feature inactive for 150+ batches has its decoder direction reset to a high-reconstruction-error input sample, its encoder weight scaled to 0.2× the mean alive-encoder norm, and its Adam moments zeroed.

Result (SAEv2):
- Hidden dimension reduced from 1024 (4×) to **384 (1.5×)** — better matched to the 256-dim activation space.
- Dead features: **100/384 (26%)**, down from 77%.
- Validation reconstruction MSE: **4.75 × 10⁻⁶** — essentially perfect reconstruction (Experiment 1 was 0.067).

SAEv2 is a dramatically cleaner decomposition. And it is precisely this cleaner SAE that exposed the deepest finding.

---

## 5. The Most Important Finding — The Policy Has No Goal Representation

With the clean SAEv2, every feature was re-examined on the **test distribution** (goal at random positions), measuring two correlations per feature:
- `goal_track_corr`: correlation between the feature's activation and the agent's proximity to the **actual** (randomly placed) goal.
- `fixed_track_corr`: correlation with proximity to the **training** goal position (6, 4).

**The maximum `goal_track_corr` across all 384 features was 0.005 — statistically indistinguishable from zero.**

No feature in the policy tracks the actual goal's location. The features that Experiment 1 labelled "goal features" (and that gave k = 157.8) were tracking the agent's own position relative to the fixed training layout — not the goal. When the goal cell was absent from (6, 4), those features read zero, which *looked* like goal-tracking but was really "the familiar object is missing from its usual spot."

This is the mechanistic root of goal misgeneralization, stated plainly:

> **The trained policy never learned to represent where the goal is. Its internal features encode the agent's position and trajectory relative to the fixed training layout. It pursues the training-time goal location because that is the only goal-related thing its circuit can represent. It cannot pursue a goal it has no feature for.**

The detection signal in Experiment 1 was real but was a *perceptual absence detector*, not a goal-tracking circuit. This reframes the entire Experiment 1 result.

---

## 6. Stage 3 — G* and the Graded-Shift Measurement (Experiment 2b)

Despite the absence of true goal features, the full invariance machinery was run to measure exactly what the graph adds. G* was built from W (the 32×32 submatrix of the most active features, weighted by mean activation) plus the feature-to-action causal vector c* from raw-representation patching.

G* profiles:
- Goal-feature mean causal importance c* = 0.001378
- Proxy-feature mean causal importance c* = 0.000002
- (Notably, the weak "goal" features still drive the action ~700× more than proxy features — they are the features active during the agent's decisive moves, even though they don't track the goal.)
- Spurious set: 30 of the top-32 features have negative reward correlation and low c* — confirming the representation is dominated by position/navigation, not goal.

Experiment 2b then deployed the policy under a **graded distribution shift**: the goal displaced from (6, 4) by exactly 1, 2, 3 cells, and fully random. 10 episodes × 3 seeds × 4 levels = 120 episodes. At each step it measured both k_activation (raw goal-feature drop) and k_graph (W-based V_total threshold crossing).

**Result — k_graph equals k_activation exactly at every displacement level:**

| Displacement | k_activation | k_graph | Δ (graph − activation) |
|---|---|---|---|
| 1 cell  | 85.3 ± 93.6  | 85.3 ± 93.6  | **+0.0** |
| 2 cells | 148.5 ± 85.3 | 148.5 ± 85.3 | **+0.0** |
| 3 cells | 135.0 ± 91.9 | 135.0 ± 91.9 | **+0.0** |
| random  | 168.0 ± 71.5 | 168.0 ± 71.5 | **+0.0** |

The graph fires at the identical step as the activation signal in all 120 episodes. The EXPLAINER_CONTINUATION prediction — that the graph would reveal earlier "routing breakage" in the graded setting — did not hold.

The reason is now clear: the V_total score is computed from the same features as the activation signal, and there is no separate goal-routing pathway to break early, because there is no goal representation. When the goal moves, the agent's trajectory changes, and both the activation signal and the graph signal reflect that single change at the same moment.

---

## 7. Hypothesis Verdicts

**H1 (G* has interpretable causal structure): SUPPORTED for the method, with a caveat.** The W-matrix produces a validated causal structure (r = 0.893). The weak "goal" features do have higher causal importance than proxy features (0.001378 vs 0.000002). But absolute KL values remain small, and the features are not true goal trackers.

**H2 (G_live diverges in a structured direction): SUPPORTED IN DIRECTION, but redundant.** During shift, the position features lose causal weight and the invariance score rises. The direction is correct. But this divergence carries no information beyond the raw activation drop.

**H3 (k_graph > k_activation): REFUTED.** k_graph = k_activation exactly (Δ = 0.0) at every displacement level, including graded shifts. The causal graph adds zero lead time over activation monitoring in this system.

---

## 8. What This Means For The Research Programme

Two results stand, one positive and one negative, and one finding reframes everything:

1. **Positive (method):** The W-matrix is a clean, gradient-free, validated (r = 0.89) way to extract inter-feature causal structure from an over-complete SAE — where EAP fails (r = 0.15). This is reusable.

2. **Negative (claim):** The causal graph does not improve early warning over simple activation monitoring for this policy. k_graph = k_activation everywhere.

3. **Reframing (finding):** The reason the graph adds nothing is that there is no goal-routing to detect — the policy has no goal representation at all (max goal-tracking correlation 0.005). The Experiment 1 detection signal was a perceptual absence detector, not a goal circuit.

The honest conclusion: you cannot monitor or graph a goal representation that does not exist. The graph-based invariance approach is only meaningful in a policy that actually represents its goal separably from its position. Testing whether a larger policy (trained on many procgen levels) develops a true goal representation is the necessary next step before the graph method can be fairly evaluated.

---

## 9. Unexpected Findings

- **The cleanest SAE produced the most negative finding.** Improving the SAE from 77% to 26% dead features did not strengthen the goal-feature story — it dissolved it, by revealing that the "goal" features never tracked the goal. Better interpretability tools produced a more honest, more deflationary result.

- **Reconstruction quality and goal representation are unrelated.** SAEv2 reconstructs activations almost perfectly (MSE 4.75e-6) yet contains no goal feature. Perfect reconstruction of a representation that lacks goal information cannot manufacture goal information.

- **EAP's failure was diagnostic, not just a nuisance.** The fact that gradients attenuate to r = 0.15 through the decoder is itself evidence about the architecture — and it motivated the W-matrix, which is the better tool anyway.

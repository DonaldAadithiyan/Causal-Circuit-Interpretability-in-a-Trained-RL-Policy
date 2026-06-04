# TASK 2 — Causal Graph Invariance Detection
## Does G* vs G_live Comparison Catch Goal Misgeneralization?

---

## Context — What Experiment 1 Established

Experiment 1 (TASK.md) confirmed three things:

1. **SAE features on a trained IMPALA CNN are interpretable.** Six coin_tracking features and ten proxy_position features were cleanly separable. Feature 17 (coin_tracking, reward_corr=0.41, action_corr=0.74) and feature 790 (proxy_position, action_corr=0.65) were the clearest representatives.

2. **A pre-failure mechanistic signal exists.** Mean k = 157.8 ± 80.3 steps across 60 episodes and 3 seeds. Goal-tracking feature activations dropped before episodic reward degraded.

3. **The causal graph was not properly tested.** The G* extraction used cosine similarity between decoder directions — a structural proxy, not actual causal intervention. KL divergences from zeroing features were tiny (max 0.0028 vs a 0.1 threshold), suggesting the SAE-space patching was attenuating the signal. No G_live was computed at deployment. No invariance comparison was made.

**What Experiment 1 did not test: whether the causal routing between features — the edges in the graph — changes in a structured way before behavioral failure. That is what this experiment tests.**

---

## What This Experiment Is About

The research proposal (see HTML proposal) describes a system that builds a reference causal graph G* at training time and compares it against a live graph G_live at each deployment step. Violations of five structural invariances — causal depth (I1), spurious edge absence (I2), goal feature dependence (I3), proxy feature independence (I4), and causal self-consistency (I5) — are used as the early warning signal.

Experiment 1 showed the raw activation signal (I3/I4 alone, without graph structure) works. This experiment tests whether the graph-level signal — the actual causal routing between features — adds anything, and specifically whether it can detect goal misgeneralization using structural invariance violations rather than raw activation thresholds.

The question being asked:

> **When goal misgeneralization occurs, do the causal edge weights in G_live diverge from G* in a structured, predictable direction — and does this divergence precede behavioral failure, independently of raw activation magnitude?**

This is harder to confirm than Experiment 1 because it requires the signal to exist in the edge structure, not just the feature activations. If it does, the full proposal is mechanistically grounded. If it does not, the system reverts to activation monitoring, which is simpler but weaker.

---

## The Research Idea — What The Graph Is Supposed To Show

In Experiment 1, the goal-tracking features deactivated because the goal cell was not visually present — a purely perceptual effect. That is a strong result, but it has a limitation: it only works when the failure is visually obvious at the current frame.

Consider a harder case: the agent can partially see the goal, but has already started routing computation through spurious proxy features rather than through the goal feature. The goal feature is still somewhat active — activation monitoring would not flag it. But the causal pathway from goal feature to action has already broken. The proxy feature has taken over the causal chain. That structural shift — which is invisible to activation monitoring — should be visible in G_live vs G*.

This is the case the causal graph is designed for. The five invariances check structure, not magnitude:

- **I1:** Is the reasoning chain deep enough, or has it collapsed to a shallow shortcut?
- **I2:** Have edges that were flagged spurious at training time re-entered the circuit?
- **I3:** Are goal features still causally connected to the action, regardless of how strongly they activate?
- **I4:** Are proxy features above their training-time causal weight ceiling?
- **I5:** Does the feature the graph says is dominant actually drive the action when intervened on?

---

## The Three Hypotheses

**H1 — G* has interpretable causal structure**
When G* is built correctly — using activation patching in the raw 256-dimensional representation space (not SAE space) — the resulting causal edge weights will be meaningful. Specifically: the top causally dominant features (highest KL divergence when zeroed) will correspond to the coin_tracking features identified in Experiment 1. The causal graph will show goal features as having stronger edges to the action than proxy features on the training distribution.

*If this fails:* The causal graph is noise. Either the policy does not have separable causal structure at this scale, or the patching methodology needs revision. Do not proceed to H2/H3.

**H2 — G_live diverges from G* at goal misgeneralization in a structured direction**
When the goal position is shifted (test distribution), G_live should show: goal feature edges to the action losing weight, and proxy feature edges to the action gaining weight. This divergence should be measurable as a violation score V = V_drop + V_gain (as defined in the proposal). The direction of divergence should match the predicted direction from the proposal.

*If this fails:* The causal routing does not change structurally even when the agent is misgeneralising. The method reduces to activation monitoring. This is a meaningful finding — document carefully.

**H3 — G_live invariance violations precede behavioral failure**
The violation score V should cross a detectable threshold before the episodic reward curve shows degradation. Measure k_graph = (reward degradation step) − (first step V exceeds threshold). Compare this to k_activation = 157.8 from Experiment 1. If k_graph > 0 consistently across seeds, the graph adds early warning value. If k_graph ≈ k_activation, the graph is redundant with activation monitoring. If k_graph > k_activation, the graph is catching something activation monitoring misses.

*If k_graph ≈ 0:* Graph-level invariances fire simultaneously with behavioral failure — no lead time. The method does not add value beyond monitoring reward.

---

## Build On Experiment 1 — Do Not Rebuild

This experiment reuses the trained policy and SAE from Experiment 1. Do not retrain them. Load:
- `outputs/checkpoints/ppo_final.zip` — the frozen PPO policy
- `outputs/checkpoints/sae_best.pt` — the trained Top-K SAE (K=32, hidden_dim=1024)
- `outputs/feature_labels.json` — the labelled features (coin_tracking, proxy_position)

The goal features and proxy features from Experiment 1 are the reference. The coin_tracking features are [933, 151, 438, 17, 736, 481]. The proxy_position features are [790, 150, 917, 1001, 589, 38, 69, 488, 654, 22].

---

## Hardware

MacBook Air, Apple Silicon M-series, 16 GB unified RAM, 512 GB SSD, no discrete GPU. Use MPS where available. Same setup as Experiment 1. Total runtime target: under 5 hours.

---

## Repository Structure

Build on top of the Experiment 1 repository. Add new scripts rather than modifying existing ones. Suggested additions:

```
experiment/
  build_causal_graph.py        # NEW — builds G* using raw-space activation patching
  compute_glive.py             # NEW — computes G_live using EAP at each deployment step
  measure_invariances.py       # NEW — checks I1-I5 against G*, computes violation score V
  experiment2_main.py          # NEW — orchestrates the full Experiment 2 pipeline
  
  outputs/
    graphs/
      G_star.npy               # The reference causal graph (32x32 matrix)
      G_star_metadata.json     # Feature labels, edge interpretations, threshold values
    experiment2/
      glive_episodes/          # G_live matrices saved per step per episode
      violation_scores/        # V scores per step per episode
      plots/                   # All Experiment 2 figures
```

You are free to restructure however you like. Maintain LOG.md as a running log appended to the top of the existing file. Write EXPLAINER2.md as a new file when complete.

---

## Phase 1 — Build G* Correctly In Raw Representation Space

This is the critical fix from Experiment 1. The previous graph was built using cosine similarity between SAE decoder directions — a structural proxy. This phase builds the true causal graph using activation patching directly in the 256-dimensional representation space.

**The correct procedure:**

Run the frozen policy on 2,000 training-distribution episodes. For each episode, collect the 256-dimensional representation vector at each step (the output of the IMPALA CNN body, before the policy head). Also collect the action probability distribution at each step.

For each of the top 32 SAE features (by activation frequency), run the intervention:
- Take a batch of 200 observations from the training distribution
- For each observation, run the forward pass → get the 256-dim representation r
- Project r through the SAE encoder → get feature activations f
- Zero out feature i in f
- Reconstruct from the patched f through the SAE decoder → get patched representation r'
- Run r' through the policy head → get patched action distribution π'
- Measure KL(π || π') — the causal influence of feature i on the action

This gives you a 32-dimensional vector of causal influence scores — one per feature. This is the correct G* for the feature-to-action edges. Call this vector c* (causal importance vector).

Additionally, for inter-feature edges: for each pair (i, j) in the top 16 most causally important features, measure whether zeroing feature i changes the activation of feature j. This gives a 16×16 inter-feature causal adjacency matrix.

**Expected result from H1:** Feature 17 (coin_tracking) should have the highest c*_i. The coin_tracking features as a group should have higher mean c* than the proxy_position features.

**Log per feature:** the KL divergence, the feature label, and whether it meets a meaningful threshold (KL > 0.01 — lower than the 0.1 threshold used in Experiment 1, which was too strict).

**Extract five invariant profiles from G*:**
- **I1 baseline:** Mean causal chain depth — the average number of non-zero inter-feature edges in the path from any feature to the action. Compute as the mean of the longest path from each feature to the action in the 16×16 adjacency graph. Record this as depth*.
- **I2 spurious edge set S:** Features with negative reward correlation AND low c* (causal importance < 0.005). These are the features to monitor for re-entry. Record S as the set of feature indices.
- **I3 goal feature set F_goal:** The coin_tracking features from Experiment 1 — [933, 151, 438, 17, 736, 481]. Their mean c* is the goal causal baseline.
- **I4 proxy ceiling:** The mean c* of proxy_position features [790, 150, 917, 1001, 589, 38, 69, 488, 654, 22] on the training distribution. This is the ceiling — proxy features must not exceed this at deployment.
- **I5 self-consistency baseline:** For the top 3 features by c*, record the pass rate of the self-consistency test (KL > 0.01 when that feature is zeroed). On training distribution this should be near 1.0.

Save G* and all five profiles to `outputs/graphs/G_star.npy` and `outputs/graphs/G_star_metadata.json`.

---

## Phase 2 — Implement G_live Using EAP

Implement the EAP-based live graph computation as described in the proposal.

At each deployment step:
1. Run the forward pass through the frozen policy — collect the 256-dim representation r and action distribution π
2. Run r through the frozen SAE encoder — get feature activations f (K=32 active features)
3. Run a single backward pass from the action logits back through the policy head and SAE decoder to get gradients with respect to f
4. For each active feature pair (i, j): w_ij_live = |f_i · ∂(action_logit)/∂f_i| — this is the EAP edge weight from feature i to the action
5. This gives you a 32-dimensional vector of live causal importance scores — one per active feature

This is G_live at step t: a vector of causal weights, one per active SAE feature.

**Validate EAP against patching on a held-out validation set:**
Run both the full patching procedure (Phase 1 method) and EAP on the same 100 observations from the training distribution. Compute the Pearson correlation between the two causal importance vectors. Log this correlation. Target r > 0.7 for this architecture (the threshold is lower than the 0.9 in the proposal because IMPALA CNN with ReLU is more nonlinear than transformers). If r < 0.5, log a warning and document in EXPLAINER2.md — the EAP approximation may be too inaccurate for this architecture.

---

## Phase 3 — Check Invariances And Compute Violation Score

Implement the five invariance checks as described in the proposal. At each deployment step, compute:

**I1 — Causal depth check:**
Estimate current depth from the live EAP weights. If the top causal feature is at Layer 1 of the IMPALA CNN and has a direct high-weight edge to the action bypassing intermediate features, flag as a depth violation. Threshold: if the ratio of Layer-1 feature weights to total weight exceeds 2× the training-time baseline, I1 fires.

**I2 — Spurious edge absence:**
For each feature in the spurious set S (identified in Phase 1), check if its live causal weight w_i_live exceeds the training-time baseline by more than 2 standard deviations. If yes, I2 fires for that feature.

**I3 — Goal feature dependence:**
Compute the mean live causal weight of the goal features F_goal = [933, 151, 438, 17, 736, 481]. If this drops below 50% of the training-time baseline c*_goal_mean, I3 fires.

**I4 — Proxy feature independence:**
Compute the mean live causal weight of the proxy features. If this rises above 150% of the training-time baseline c*_proxy_mean, I4 fires.

**I5 — Causal self-consistency:**
For the currently dominant feature (highest live causal weight), run one counterfactual forward pass: zero it out, measure KL(π || π'). If KL < 0.01 (feature claims to be dominant but zeroing it barely changes the action), I5 fires. Run this only when I1–I4 have already flagged something — do not run it every step.

**Compute the violation score:**

```
V_drop = sum over (i in F_goal): max(0, c*_i - c_live_i)
V_gain = sum over (i in S): max(0, c_live_i - c*_i)
V_total = alpha * V_drop + beta * V_gain + gamma * I5_fired

R_reason = -V_total
```

Use alpha = beta = gamma = 1.0 as a starting point. Log these weights as a hyperparameter.

---

## Phase 4 — Run The Deployment Experiment

Replicate Phase 5 from Experiment 1 but now with the full graph-based signal alongside the activation-based signal. Run 10 episodes per seed, 3 seeds = 30 total episodes on the test distribution (goal position randomised).

At each step of each episode, compute and record:
- Raw goal feature activation (from Experiment 1 method — the activation-based signal)
- Raw proxy feature activation
- V_total (the graph-based violation score)
- Which individual invariances fired (I1, I2, I3, I4, I5 — boolean per step)
- The episodic reward

For each episode, measure:
- **k_activation:** step at which raw goal activation drops below 50% of baseline (Experiment 1 method)
- **k_graph:** step at which V_total first exceeds a threshold (to be determined — start with V_total > 0.1 × c*_goal_mean, adjust if needed)
- **k_reward:** step at which episodic reward confirms failure

Compute:
- Mean k_graph, std k_graph across all 30 episodes and 3 seeds
- Mean k_activation for comparison
- Whether k_graph > k_activation, k_graph ≈ k_activation, or k_graph < k_activation
- Which invariance (I1–I5) fired first most frequently

Run the same measurement on 10 training-distribution episodes as a control. V_total should remain near zero. If it does not, the thresholds are miscalibrated — log and adjust.

---

## What Success Looks Like

**Strong success:** k_graph is measurable and positive across seeds. V_total reliably crosses the threshold before reward degrades. The dominant invariance violation matches the predicted pattern (I3+I4 for goal misgeneralization — goal feature causal weight drops, proxy causal weight rises). EAP correlation with patching is r > 0.7.

**Moderate success:** k_graph > 0 but smaller than k_activation. The graph fires later than raw activation monitoring but still before behavioral failure. This means the graph is valid but not superior to activation monitoring in this setting. Still publishable — document honestly.

**Null result on graph:** k_graph ≈ 0 or k_graph < 0. The graph-level signal does not precede behavioral failure. V_total only rises at or after reward degradation. This would mean the causal routing does not change structurally before behavior changes — the signal lives only in raw activations, not in causal edges. This is an important negative finding. Document carefully, propose why (policy too small? environment too simple? SAE dead feature problem?).

**EAP failure:** Correlation between EAP and patching is r < 0.5. The EAP approximation is unreliable for this architecture. In this case, use the patching-based causal weights for G_live (slower but more accurate) and note EAP as a future-work item.

---

## LOG.md — Instructions

Continue the existing LOG.md from Experiment 1. Append new entries at the top. Mark Experiment 2 entries with `[EXP2]` prefix. Log everything: phase transitions, hyperparameter choices, EAP correlation value, threshold calibration decisions, per-seed k_graph measurements, any unexpected behaviour. Update every 30 minutes during active computation.

---

## EXPLAINER2.md — Instructions

Write EXPLAINER2.md as a standalone document when the experiment is complete. A reader who has read EXPLAINER.md (Experiment 1) should be able to read EXPLAINER2.md and understand what changed, what was tested, and what was found. Cover:

1. **Why this experiment follows from Experiment 1** — what was missing, what the graph is supposed to add over raw activation monitoring.

2. **How G* was built correctly this time** — what was wrong with the Experiment 1 approach (cosine similarity proxy, SAE-space patching) and what the correct procedure is (raw-space activation patching, KL divergence threshold).

3. **What G* looked like** — the causal importance vector c*, which features were dominant, whether the ordering matched predictions, the five invariant profiles.

4. **How accurate EAP was** — the correlation between EAP and patching on the validation set, what this means for the reliability of G_live.

5. **What G_live showed during goal misgeneralization** — did edge weights shift in the predicted direction? Which invariances fired? When?

6. **The k_graph result** — mean, std, per-seed. How does it compare to k_activation = 157.8 from Experiment 1? What does this mean for the research programme?

7. **Hypothesis verdicts** — H1, H2, H3: supported, partially supported, or refuted. Be direct.

8. **What comes next** — if graph works, move to TASK3 (reward signal correction). If graph does not add value, explain why and propose what architecture change would fix it.

9. **Unexpected findings** — anything that was not predicted.

---

## Constraints

- **Total runtime: under 5 hours** on MacBook Air Apple Silicon. EAP is fast. The main bottleneck is the patching procedure in Phase 1 — 32 features × 200 observations = 6,400 forward passes. This should take 20–40 minutes on MPS.
- **Do not retrain the policy or SAE.** Load from Experiment 1 checkpoints.
- **Use the feature labels from Experiment 1.** Do not re-run Phase 3 interpretability analysis.
- **The architecture does not change.** Same IMPALA CNN, same SAE, same K=32.
- **Save all G_live matrices** for at least the representative episode (one per seed) so the causal drift can be visualised.
- **Every plot saved as PNG** to `outputs/experiment2/plots/` with descriptive filenames.
- Free to add any additional analysis that seems scientifically useful. Log everything.

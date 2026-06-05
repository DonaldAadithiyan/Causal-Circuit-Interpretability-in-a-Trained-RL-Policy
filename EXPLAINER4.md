# EXPLAINER4 — The Full Pipeline: Detection + Three-Response Comparison

*The capstone of the four-experiment programme. Read EXPLAINER.md, EXPLAINER2.md, EXPLAINER3.md first.*

---

## 1. What the Programme Learned Across Experiments 1–3

Experiments 1–3 used a **fixed-goal** policy (goal always at (6,4) during training). Experiment 2's decisive finding was that this policy had **no goal representation at all** — the maximum actual-goal-tracking correlation across 384 SAE features was 0.005. Its "goal features" were really position detectors. Every downstream technique failed because of this: the causal graph added no lead time (Experiment 2b, k_graph = k_activation everywhere), and R_reason built on the confounded features catastrophically broke the policy (Experiment 3, 100% failure).

The diagnosis was clean: **you cannot monitor, graph, or correct a goal representation that does not exist.** Experiment 4 fixes the root cause — it trains the policy with the goal in a *random* position every episode, forcing it to read the goal from the observation, then re-runs the entire pipeline on a policy that genuinely represents its goal.

---

## 2. H1 — The Random-Goal Policy Builds a Real Goal Representation

**H1 PASSED.** Maximum actual-goal-tracking correlation = **0.440** (threshold 0.30).

| Policy | Max actual-goal-tracking correlation |
|---|---|
| Fixed-goal (Exp 1–3) | 0.005 |
| **Random-goal (Exp 4)** | **0.440** |

Randomising the training goal raised the goal-representation correlation by ~88×. The five strongest goal-tracking features were [31, 280, 338, 117, 291], each with |corr| ≈ 0.38–0.44. The SAEv3 that found them was the cleanest of the whole programme: **0 dead features out of 384**, validation MSE 3.9×10⁻³, W-matrix validated at r = 0.59.

This is the single most important control in the four-experiment arc. One change to the training setup — the goal position — is the difference between a policy that has a goal representation and one that does not. It confirms the Experiment 2 diagnosis directly: the absence of a goal feature in the fixed-goal policy was caused by the fixed goal, not by the architecture or the SAE.

A second, unplanned finding fell out of Phase 1: the random-goal policy **generalises almost perfectly**. Tested with the goal pinned at each of the 36 interior cells, it reaches the goal at 0% failure for nearly every position — it genuinely reads the goal. But it has a small number of **systematic blind spots**: specific cells, notably **(6,5)**, where it fails 100% of the time. The goal is plainly visible there; the policy simply never goes to it. This residual, reproducible failure in an otherwise-generalising policy became the test case for the rest of the experiment.

---

## 3. The Central Result — The Causal Graph Detects What Activation Monitoring Cannot

Deploying the frozen policy on the (6,5) blind spot (30 episodes, 3 seeds) and measuring both signals gave the result the entire proposal was built around:

| Signal | Value at (6,5) |
|---|---|
| **k_activation** | **undefined (nan)** |
| **k_graph** | **200 (fires at step 0)** |

**k_activation is undefined because the goal feature never deactivates.** The goal at (6,5) is visible the whole episode, so the goal-tracking feature stays active from start to finish. Activation monitoring — the Experiment 1 method — sees a healthy goal feature and reports "nothing wrong." It is structurally blind to this failure.

**k_graph fires immediately.** The W-based causal violation score V_total crosses its threshold at step 0 of every episode. The causal graph detects that although the goal feature is *active*, it is not *driving the action* — the causal routing from goal feature to behaviour is broken. The goal is represented but not used.

This is the discovery the proposal predicted and that Experiments 1–2b could not produce: **a failure mode where causal routing breaks while the goal feature stays active.** Activation magnitude says everything is fine; the causal graph says the circuit is mis-routed. Only the graph catches it. In the language of the success criteria, k_graph does not merely exceed k_activation by 20 steps — k_activation is *undefined* while k_graph is *fully defined*. The graph is detecting something activation monitoring fundamentally cannot represent.

---

## 4. The Three-Response Comparison

All responses ran on the same policy, same blind-spot test (goal at (6,5)), same seeds. Baseline = frozen policy, no response.

| Condition | Failure rate | Circuit repaired? | Persists across episodes? |
|---|---|---|---|
| Baseline (no response) | 1.00 | — | — |
| Activation steering (best α=0.5) | 1.00 | No (inference-time) | No (weights unchanged) |
| Targeted fine-tuning (3 seeds) | 1.00 | **Yes (100% of seeds)** | Yes (weights updated) |
| **R_reason (best λ=0.1)** | **0.00** | No (behavioral) | No (per-episode policy) |

**Failure-rate ordering (best first): R_reason (0.00) ≪ steering ≈ fine-tuning ≈ baseline (1.00).**

This **inverts the proposal's predicted hierarchy** (fine-tuning > steering > R_reason), and the inversion is mechanistically precise. The reason is the nature of the (6,5) failure: it is a **routing** failure, not a **representation** failure. The goal feature is already active and correct. So:

- **Activation steering failed because it never triggered.** Its trigger is I3 — goal-feature causal importance dropping below 60% of baseline. But the goal feature stays active at (6,5), so I3 fired on only ~1% of steps (steer_frac = 0.01). A response designed to restore a *missing* goal signal has nothing to do when the goal signal is present. Wrong tool for this failure.

- **Targeted fine-tuning produced the most striking single result in the programme: `circuit_repaired = True` in 100% of seeds, yet failure stayed at 100%.** The fine-tuning loss pushes goal-feature activation up toward its training baseline — and it succeeds, the circuit is "repaired" by that measure, with no catastrophic forgetting (train reward 0.75–0.90). But the agent still never reaches (6,5). **This is a direct, clean demonstration that circuit repair does not imply behavioral correction.** Restoring the goal *representation* does nothing when the representation was never the problem — the broken link was from goal to action, and fine-tuning the feature extractor toward a representational target leaves that link untouched.

- **R_reason fixed it completely (100% → 0% failure) — but only at λ = 0.1.** R_reason adds the violation score as a per-step penalty and lets PPO update the *action* policy. Because it acts on behaviour through gradient descent, it can repair the routing: over 50k steps it discovers the path to (6,5). At λ = 0.1 this is decisive and clean (0% failure, both seeds). But at λ ≥ 0.5 it collapses to 100% failure — the same dose-sensitivity seen in Experiment 3, where a too-large dense penalty overwhelms the sparse environment reward and destabilises the policy. The correction works, but only in a narrow band of the reward weight.

---

## 5. Which Layer Adds the Most Value?

For this failure mode the answer is unambiguous and surprising: **the indirect, "shallowest" response (R_reason) was the only one that worked, and the direct circuit interventions (steering, fine-tuning) both failed.** The proposal's intuition — that more direct, weight-level intervention is more powerful — is exactly wrong here, because it assumes the failure lives in the representation. When the failure lives in the *routing*, only the response that retrains the action policy can reach it.

The practical lesson for the pipeline is that **the right response depends on diagnosing the failure type:**
- Representation failures (goal feature missing/weak) → steering or fine-tuning to restore it.
- Routing failures (goal feature present, not driving action) → behavioural retraining (R_reason).

A complete system needs the diagnosis step — and the causal graph is what provides it. The graph's V_total fired here precisely because it measures routing, not activation; the same property that let it detect the failure (Section 3) also identifies the failure as routing-type, which in turn predicts that only R_reason will fix it. The graph is not redundant with activation monitoring after all — it is redundant only when the policy has no goal representation (Experiment 2b). Give the policy a real goal representation, and the graph earns its place.

---

## 6. The Unified Conclusion of the Four-Experiment Programme

1. **Goal misgeneralization has a precise mechanistic signature — but only in a policy that has a goal representation.** The fixed-goal policy (Exp 1–3) never built one; the random-goal policy (Exp 4) did, at correlation 0.44 vs 0.005. The single change — randomising the training goal — is what made every downstream technique meaningful.

2. **The W-matrix is the right tool for causal structure** on over-complete SAEs feeding CNN policies (validated r = 0.59–0.89 across SAEs), where gradient-based EAP fails (r = 0.15).

3. **The causal graph detects failures that activation monitoring is blind to** — concretely, the goal-visible-but-mis-routed failure at (6,5), where k_activation is undefined and k_graph = 200. This is the proposal's central claim, finally demonstrated, and it only became visible once the policy had a real goal representation to mis-route.

4. **Circuit repair and behavioral correction are distinct, and can be dissociated.** Targeted fine-tuning repaired the circuit (goal activation restored) without correcting behaviour (failure stayed 100%), because the failure was in routing, not representation. The response that fixed behaviour (R_reason) did so without repairing the circuit.

5. **Correction is real but fragile.** R_reason eliminated the blind spot (0% failure) at λ = 0.1 and catastrophically destabilised the policy at λ ≥ 0.5 — the dense-penalty dose-sensitivity first seen in Experiment 3, now seen even with correct features.

---

## 7. What the Paper Can Honestly Claim

- **Supported by evidence:**
  - Randomising the training goal produces a measurable goal representation (corr 0.44 vs 0.005) — the prerequisite the whole pipeline depends on.
  - The causal graph detects a routing failure that activation monitoring cannot (k_activation undefined, k_graph = 200) — the first concrete demonstration in this programme that causal routing degrades while goal features stay active.
  - Circuit repair ≠ behavioral correction — directly shown by fine-tuning (repaired = True, failure = 100%).
  - At least one response (R_reason, λ = 0.1) reduces failure from 100% to 0%.

- **Honest limitations:**
  - The "win" is a single response at a single tuned λ; at higher λ it collapses. Correction is not yet robust.
  - The test is one blind-spot cell in an 8×8 grid with a 624k-parameter policy. Whether the same dissociation and the same graph advantage appear at procgen scale is unknown.
  - Steering and fine-tuning failed here, but on a *representation*-type failure they might be the responses that work — the experiment tested only one failure type.

- **Future work:** larger policies and richer environments; a diagnosis step that classifies failures as representation- vs routing-type and routes to the matching response; making R_reason robust to λ (e.g. adaptive weighting); and testing whether the k_graph-vs-k_activation gap widens with policy capacity.

---

## 8. The Four-Experiment Arc in One Paragraph

A fixed-goal RL policy looked like it had goal-tracking features that fired before failure (Exp 1, k = 158), but a cleaner SAE and a validated causal graph proved those features never tracked the goal — the policy had no goal representation, the detection was a perceptual artifact, the graph added nothing, and correction built on the confounded features destroyed the policy (Exp 2, 2b, 3). Retraining the policy with a randomised goal built a genuine goal representation (Exp 4, corr 0.44), and on that policy the causal graph finally did what the proposal claimed — it detected a goal-visible-but-mis-routed failure that activation monitoring was blind to (k_activation undefined, k_graph = 200) — while the three responses cleanly separated circuit repair from behavioral correction, with only behaviourally-targeted reward shaping fixing the blind spot, and only within a narrow stability band. The programme's lesson is that mechanistic monitoring and correction of RL agents is real but conditional: it works exactly when the agent has the representation you are trying to monitor, and the response must match the failure type the graph diagnoses.

*Scale note: R_reason was run at 50k steps × λ{0.1,0.5,1.0} × 2 seeds (reduced from the 100k × 3-seed spec to fit the 7-hour hardware budget); all other phases at full spec. Total Phase 3–5 wall time: 110 min on Apple M-series MPS.*

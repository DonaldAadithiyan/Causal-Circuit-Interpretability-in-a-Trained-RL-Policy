# EXPLAINER — Continuation
## What Experiment 1 Means, What Is Still Missing, and What Experiments 2 and 3 Will Test

*This document continues from EXPLAINER.md. Read that first.*

---

## Where We Are After Experiment 1

Experiment 1 answered the foundational question: **do interpretable features with meaningful structure exist in a trained RL policy, and do they shift before failure?**

The answer was yes. Mean k = 157.8 steps. Goal-tracking features [933, 151, 438, 17, 736, 481] deactivated before episodic reward degraded in all 60 measured episodes across 3 seeds. The features were clean and manually labelable. The failure was real and consistent.

That is a strong result. But re-reading the proposal carefully, Experiment 1 only tested one layer of a three-layer argument. And it tested that layer in a simpler way than the proposal describes.

Here is exactly what is still missing and why it matters.

---

## What Experiment 1 Did Not Test

**It did not build or use a causal graph.**

The G* in Experiment 1 was computed using cosine similarity between SAE decoder directions. That is a structural proxy — it measures whether two features point in similar directions in activation space. It does not measure whether zeroing one feature actually changes another feature's activation or the action distribution. That is a different and harder question.

The KL divergence values from zeroing features were tiny — maximum 0.0028 against a 0.1 threshold. This means patching in SAE space barely moved the action distribution. The SAE-space signal is too attenuated to be useful.

The proposal's causal graph is built by patching directly in the 256-dimensional representation space — zeroing a feature's contribution in the raw representation and measuring how much the action distribution changes. This was not done in Experiment 1.

**It did not compare G_live against G*.**

No live graph was computed at deployment. The signal that detected failure was purely: "goal feature activation dropped below 50% of baseline." That is activation monitoring, not causal graph comparison. The five invariances (I1–I5) were not implemented or checked.

**The k = 157.8 result is partly an artefact of the binary distribution shift.**

Goal features deactivated at step 0 because the goal cell was visually absent from the current frame. The signal was immediate and perceptual. In a harder setting — partial shift, gradual drift, or a case where the goal is visible but the agent's internal routing has already broken — the activation signal might fire much later or not at all. The causal graph is designed precisely for this harder case.

---

## Why The Causal Graph Matters — The Harder Case

Imagine the distribution shifts gradually. The goal is now 2 cells away from its training position instead of fully randomised. The agent can still see it. The goal-tracking features are still somewhat active. Raw activation monitoring says: "goal features are at 80% of baseline — no violation." But internally, the agent has already started routing computation through the proxy feature (navigating toward training-time position) because the path to the training position is more familiar. The causal edge from the goal feature to the action has already weakened. The proxy feature's causal edge has already strengthened.

This structural shift — the routing change — is what the causal graph detects. Activation monitoring misses it entirely because the feature is still active. The graph fires because the edge weights changed.

That is the case Experiment 2 is designed to test. And if the graph catches it earlier than activation monitoring, the research has demonstrated something genuinely new — not just that features deactivate before failure, but that causal routing breaks before features deactivate.

---

## What Experiment 2 Will Establish

Experiment 2 builds G* correctly — using raw-space activation patching with measurable KL divergences — and implements G_live using EAP at each deployment step. It then measures whether the invariance violation score V_total precedes behavioral failure, and compares k_graph against k_activation = 157.8 from Experiment 1.

Three possible outcomes and what each means:

**k_graph > k_activation:** The causal graph fires earlier than raw activation monitoring. This means the routing breaks before the features deactivate. This is the strongest possible result — it establishes that causal structure is a more sensitive pre-failure signal than activation magnitude. The paper's core mechanistic claim is fully supported.

**k_graph ≈ k_activation:** Both signals fire at approximately the same time. The graph adds no lead time over activation monitoring in this setting. The graph may still be valuable for other reasons (failure mode specificity, robustness to partial shifts) but it does not improve early warning in the binary shift setting.

**k_graph ≈ 0:** The causal graph fires at the same time as behavioral failure or after. The invariance comparison does not produce a useful early warning signal. The system reverts to activation monitoring as the primary mechanism, and the graph is relegated to a diagnostic tool rather than a detection tool.

Any of these is a publishable result. The experiment is designed to produce a clear finding regardless of which outcome occurs.

---

## What Experiment 3 Will Establish

Experiment 3 tests the most important question for the paper's contribution: does the violation signal, fed back as R_reason, actually change the agent's behaviour and prevent goal misgeneralization?

Detection alone — even with a 157-step lead time — is not a complete contribution. A smoke detector that tells you the house is burning 157 steps before the house burns down is valuable. But a smoke detector that also turns on the sprinklers is more valuable. Experiment 3 tests whether the sprinklers work.

The mechanism is: R_reason fires when the causal circuit drifts. PPO receives a negative reward signal in those states. Over the k-step window before failure, the action distribution updates to avoid the states that produced the circuit drift. The agent behaviorally corrects — it takes different paths through the environment that do not trigger the violation.

This does not require the internal circuit to repair. The agent can still have miscalibrated internal representations. What matters is whether its actions change enough to reach the goal.

Two possible outcomes:

**Correction works:** The R_reason agent succeeds more often than the baseline agent on the test distribution. The k-step window is large enough for PPO to update meaningfully. The paper claims: "Circuit-level violation detection enables behavioral correction of goal misgeneralization without retraining — the first deployment-time safety mechanism for RL agents that acts before behavioral failure occurs."

**Correction does not work:** The R_reason agent fails at the same rate as the baseline. The signal does not produce meaningful behavioral updates. This points to the gradient path problem — PPO updates action distributions, and the circuit remains miscalibrated even when actions change. In this case, the fix is activation steering (directly patching the representation during deployment) or targeted fine-tuning between episodes. Either becomes the Layer 3 experiment.

---

## The Honest Assessment of What Has Been Proven So Far

After Experiment 1, here is what can be stated with confidence:

**Confirmed:** In a small IMPALA CNN policy trained on a fixed-goal MiniGrid task, Top-K SAE features decompose the 256-dimensional policy representation into interpretable components that correspond to identifiable visual concepts. Goal-tracking features and proxy features are separable. Goal features deactivate before episodic reward degrades when the goal position shifts. This happens in all measured episodes across 3 seeds.

**Not yet confirmed:** Whether the causal routing between features changes before failure (Experiment 2). Whether the violation signal prevents failure when used as a reward (Experiment 3). Whether any of this generalises beyond the binary shift setting or beyond this specific environment and architecture.

**The most important open question after Experiment 1:** The k = 157.8 result is based on a very easy detection setting — the goal is visually absent from the current frame, so goal features immediately read zero. The harder test is a graded shift where the goal is visible but the agent has already broken its causal routing internally. That test is part of what Experiment 2 is designed to address.

---

## The SAE Dead Feature Problem — Priority Fix

785 of 1024 SAE features were dead after training. This is the experiment's main methodological weakness. Before any results from Experiments 2 or 3 are interpreted, it is worth understanding what caused this and whether it affects the validity of the live features.

The likely cause is a combination of: insufficient training (early stopping at epoch 12), oversized hidden dimension (4× expansion creates too many features for the 256-dim activation space), and K=32 being too aggressive a sparsity constraint for the number of live features.

The fix — as the EXPLAINER.md notes — is to add feature usage tracking with decoder reinitialization during training. Features that have not activated for N steps get their decoder vectors randomly reinitialised, forcing the SAE to spread its representation. This is standard practice in SAE training (called "resampling" or "nudging" in the mechanistic interpretability literature).

The dead feature problem does not invalidate Experiment 1's results — the 239 live features included the interpretable goal and proxy features that produced the H3 result. But it means the SAE is using only 23% of its capacity, and the features it found may not be the most meaningful ones. A better-trained SAE might find cleaner goal features with higher reward correlation and might produce stronger G* edge weights.

Experiments 2 and 3 proceed with the existing SAE. If results are weak, the SAE should be retrained with the dead feature fix before concluding the method does not work.

---

## Summary — The Three-Experiment Arc

| Experiment | Question | Method | Key Metric |
|---|---|---|---|
| 1 (done) | Do interpretable features exist and do they shift before failure? | SAE features + activation threshold | k_activation = 157.8 steps |
| 2 (next) | Does the causal graph add anything over raw activation monitoring? | G* vs G_live, EAP, invariance violations | k_graph vs k_activation |
| 3 (after) | Does R_reason prevent the failure from occurring? | PPO + R_reason reward shaping | Failure rate reduction |

If Experiment 2 confirms k_graph > 0 and Experiment 3 confirms failure rate reduction, the paper has: detect → monitor → correct. That is the complete three-layer system demonstrated empirically. The paper is writable.

If either fails, the paper has a detection result (Experiment 1) plus an honest analysis of where the pipeline breaks and why. That is still publishable, and more scientifically honest than overclaiming.
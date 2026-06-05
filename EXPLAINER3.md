# EXPLAINER3 — Experiment 3: R_reason Correction (And Why It Backfired)

*Read EXPLAINER.md and EXPLAINER2.md first.*

---

## 1. The Question

Detection is not correction. Experiment 1 showed a violation signal fires ~157 steps before behavioral failure. Experiment 3 asked: if that signal is fed back to the agent as a negative reward (R_reason), does the agent change its behavior and avoid goal misgeneralization?

The design (Option B): take the Experiment 1 policy, run a second 100k-step PPO phase on the test distribution (goal randomised), with reward `R_total = R_env + λ · R_reason`, where `R_reason = −(V_drop + V_gain)` fires when goal features fall and proxy features rise. Sweep λ ∈ {0.0, 0.1, 0.5, 1.0}, 3 seeds each. λ = 0.0 is the baseline (ordinary fine-tuning, no circuit signal).

---

## 2. The Result — R_reason Catastrophically Broke the Policy

| λ | Mean failure rate | Mean reward | Δ from baseline |
|---|---|---|---|
| 0.0 (baseline) | **0.167 ± 0.118** | 0.833 | — |
| 0.1 | **1.000 ± 0.000** | 0.000 | +0.833 |
| 0.5 | **1.000 ± 0.000** | 0.000 | +0.833 |
| 1.0 | **1.000 ± 0.000** | 0.000 | +0.833 |

Two things are immediately visible:

1. **Ordinary fine-tuning works well.** The baseline (λ = 0) cut the test-distribution failure rate to 16.7% — simply training on the test distribution with the environment reward mostly fixes goal misgeneralization, given 100k steps.

2. **Adding R_reason destroyed the policy completely.** At every non-zero λ — even the smallest, 0.1 — the agent failed 100% of the time and earned zero reward. R_reason did not help slightly less than hoped; it actively and totally broke the agent.

---

## 3. Why It Backfired — The Confounded Signal

This is the key to the whole experiment, and it connects directly to the Experiment 2 finding.

R_reason was built from the Experiment 1 "goal features." Experiment 2 then proved those features do not track the goal at all (max goal-tracking correlation 0.005) — they track the agent's proximity to the **fixed training position (6, 4)**.

So `R_reason` actually computed: "how far is the agent from where the goal used to be during training." Feeding `R_total = R_env + λ · R_reason` to PPO therefore told the agent:

> **"You are penalised every step you are not at (6, 4)."**

That is the exact opposite of the correction we wanted. On the test distribution the real goal is somewhere random, but R_reason rewarded the agent for going to (6, 4) — it *paid the agent to goal-misgeneralize*. The dense per-step penalty (−1.0 almost everywhere the goal isn't at (6,4)) swamped the sparse +1 environment reward, and PPO collapsed to a degenerate policy that earns zero.

The catastrophe is not a tuning failure. It is a direct, predictable consequence of building a correction signal on features that were misidentified. **A correction signal is only as good as the interpretation of the features it is built from. Confounded features produce a confidently wrong reward.**

---

## 4. Hypothesis Verdicts

**H1 (R_reason reduces failure rate): REFUTED — and inverted.** R_reason raised the failure rate from 16.7% to 100%. The signal did not produce beneficial behavioral correction; it produced complete behavioral collapse.

**H2 (correction scales with k): NOT TESTABLE.** With 100% failure at all λ, there is no successful-correction subset in which to study a k-dependence.

**H3 (correction is behavioral not circuit-level): NOT REACHED.** There was no correction to diagnose.

---

## 5. What This Means For The Research Programme

The three-experiment arc resolves into a single coherent story:

- **Experiment 1** found a pre-failure signal (k = 157.8) and attributed it to goal features.
- **Experiment 2**, with a cleaner SAE and the validated W-matrix, proved those features never tracked the goal — the policy has no goal representation. The signal was a perceptual absence detector.
- **Experiment 3** tried to *correct* using those same features and broke the policy, because the features encode "distance from the training goal location," so penalising their drop rewards goal misgeneralization.

The unifying lesson: **you cannot detect, monitor, graph, or correct a goal representation that does not exist.** Every downstream technique in the proposed pipeline assumed the policy has separable goal-tracking features. For this small policy on this task, that assumption is false, and each technique fails in a way that is now fully explained.

Crucially, this is not a failure of the *experiment* — it is a clean, well-supported negative result with a precise mechanistic cause. And it carries a positive corollary: **ordinary fine-tuning on the test distribution (the λ = 0 baseline) reduced failure to 16.7% without any circuit signal at all.** For this failure mode, plain adaptation outperforms the circuit-based correction — because plain adaptation lets the policy build the goal representation it never had, while the circuit signal forces it back toward the old one.

---

## 6. What Comes Next

1. **Re-run the whole pipeline on a policy that has a goal representation.** The necessary precondition — a feature with high actual-goal-tracking correlation — was never met here. A larger IMPALA CNN trained on many procgen levels (forced to generalise across goal positions during training) is the natural candidate. Only then can R_reason, the causal graph, and the invariances be fairly tested.

2. **If a goal feature exists, build R_reason from the *actual*-goal-tracking correlation, not the training-position proxy.** The Experiment 3 failure is a direct warning: validate that the features mean what you think before using them as a reward.

3. **Activation steering instead of reward shaping.** Even with correct features, the gradient path from a per-step reward to internal representations is indirect. Directly adding a goal-feature vector to the representation at deployment would intervene on the circuit without the reward-collapse risk that destroyed the policy here.

---

## 7. Unexpected Findings

- **The smallest λ was already fatal.** λ = 0.1 produced the same 100% failure as λ = 1.0. A confounded dense reward does not need to be large to dominate a sparse environment reward — even a 10% weight was enough to collapse the policy.

- **The baseline quietly succeeded.** The most effective intervention in the entire three-experiment programme for *reducing* goal misgeneralization was the control condition: ordinary PPO fine-tuning on the test distribution, no interpretability involved (16.7% failure). This is a humbling and important result — the sophisticated circuit-based correction was beaten decisively by plain retraining.

- **Catastrophic forgetting appeared even in the baseline.** One baseline seed lost all training-distribution performance (train reward 0.0) while still reducing test failure — the policy can adapt to the test distribution by overwriting, not extending, its training behavior. This is its own small lesson about the fragility of fine-tuning as a correction mechanism.

# TASK 3 — Reward Signal Correction
## Does R_reason Prevent Goal Misgeneralization Before It Manifests?

---

## Context — What Experiments 1 and 2 Established

**Experiment 1 (TASK.md)** confirmed: SAE features on a trained RL policy are interpretable, and goal-tracking feature activations drop before episodic reward degrades. Mean k = 157.8 steps. The mechanistic pre-failure signal exists.

**Experiment 2 (TASK2.md)** tested: whether the causal graph G* vs G_live invariance comparison catches the same failure structurally — not just via raw activation thresholds. [Results from TASK2 should be summarised here once available.]

**What neither experiment tested:** whether the violation signal — when fed back as a reward to the agent — actually changes the agent's behaviour and prevents the failure from occurring.

This is the experiment that determines whether the research is a detector or a corrector. Detection is publishable. Correction is the paper.

---

## The Question

> **When R_reason fires k steps before behavioral failure, does feeding it back as a negative reward to PPO cause the agent to take different actions — actions that avoid or delay goal misgeneralization — within that k-step window?**

From the proposal (see HTML):

> R_total = R_env + λ(t) · R_reason
> R_reason = −(α · V_drop + β · V_gain + γ · 𝟙[I5])

The agent receives a negative signal when its causal circuit drifts from G*. PPO updates the action distribution in response. The hypothesis is that these updates cause the agent to avoid states and actions that produce circuit miscalibration — behaviorally correcting before failure manifests.

---

## Why This Might Work And Why It Might Not

**Why it might work:** R_reason is dense — it fires every step, not just at episode end. Dense reward signals are known to improve PPO's ability to shape behaviour compared to sparse end-of-episode signals. The signal is semantically rich — it is not arbitrary noise, it is derived from the agent's own causal circuit. The k-step window from Experiment 1 is large (157 steps) — there are many gradient updates available before failure.

**Why it might not:** PPO updates action distributions, not internal representations. The circuit may remain miscalibrated even if the agent's actions change. The agent might learn to avoid states that trigger the violation flag rather than fixing the underlying reasoning — superficial correction rather than genuine correction. The gradient path from R_reason through the policy loss to the internal representations is long and indirect.

**What we care about:** whether behavioral failure is prevented. We do not require the circuit to be fixed. If the agent averts goal misgeneralization by taking different actions — even without repairing its internal representations — that is a successful result. The distinction between behavioral correction and circuit repair is important to document but does not determine success.

---

## The Three Hypotheses

**H1 — R_reason reduces failure rate**
An agent receiving R_reason during deployment on the test distribution will fail less often than an agent receiving only R_env. Failure rate is defined as episodes where the agent does not reach the goal before max_steps. The R_reason agent should show lower failure rate across seeds.

*If this fails:* The reward signal does not produce behaviorally meaningful updates within the deployment window. Either k is insufficient for PPO to correct, or the gradient path is too indirect.

**H2 — R_reason reduces failure rate more when k is larger**
Episodes where the violation signal fires early (large k_individual) should show higher correction success than episodes where it fires late (small k_individual). This would demonstrate that the mechanism is the k-step correction window — more steps of negative signal produce more correction.

*If this fails:* Correction rate does not depend on k — either the signal is correcting for a different reason than hypothesised, or it is not correcting at all and the variation in failure rate is noise.

**H3 — Correction is behavioral, not circuit-level**
After successful correction episodes, the causal circuit (G_live) should still show goal feature drift — the internal representations remain miscalibrated. But the agent's actions are different — it takes a path toward the goal rather than toward the training-time position. This demonstrates behavioral correction without circuit repair, which is the honest claim.

*If the circuit also repairs:* That is a stronger result — document it. But do not assume it without measuring it.

---

## Critical Design Decision — How R_reason Works During Deployment

This experiment requires making a decision about the correction mechanism. Two options:

**Option A — Online PPO fine-tuning during deployment**
The agent continues to receive PPO gradient updates during the deployment episode. R_reason is added to R_env as the reward. PPO updates the policy weights at the end of each rollout (every n_steps). The policy is not frozen during deployment — it is actively updating.

This is the proposal's intended design. The agent is corrected in real time within the deployment episode. It requires the policy to not be frozen during deployment.

Risk: the policy may overfit to the test distribution and lose its training-time performance. This is a real concern with online fine-tuning.

**Option B — R_reason as a shaped reward during a new training phase**
After the policy is trained on the training distribution (fixed goal), run a second PPO training phase on the test distribution (random goal) with R_total = R_env + λ · R_reason. Compare this to a baseline that only sees R_env during the test phase.

This is cleaner experimentally — you can compare two training runs directly. But it is not truly deployment-time correction — it is a second training phase with a shaped reward.

**Recommendation:** Implement both and compare. Option B is the cleaner experiment. Option A is the bolder claim. Run Option B first because it is more controlled and interpretable. If Option B shows H1 is true, run Option A to see if real-time correction also works.

---

## Build On Experiments 1 and 2 — Do Not Rebuild

Load from existing checkpoints:
- `outputs/checkpoints/ppo_final.zip` — the frozen trained policy (starting point for fine-tuning)
- `outputs/checkpoints/sae_best.pt` — the frozen SAE
- `outputs/graphs/G_star.npy` — the reference causal graph from Experiment 2
- `outputs/graphs/G_star_metadata.json` — the five invariant profiles

The coin_tracking features [933, 151, 438, 17, 736, 481] and proxy_position features [790, 150, 917, 1001, 589, 38, 69, 488, 654, 22] are the reference. G* and all invariant profiles are already computed.

---

## Hardware

MacBook Air, Apple Silicon M-series, 16 GB unified RAM, 512 GB SSD. MPS backend. Total runtime target: under 6 hours. The most expensive part is the second PPO training phase (Option B) — scale to fit.

---

## Repository Structure

Add to the existing experiment repository. Do not modify Experiments 1 or 2.

```
experiment/
  compute_r_reason.py          # NEW — computes R_reason at each step using G* profiles
  correction_experiment.py     # NEW — Option B: second training phase with R_reason
  online_correction.py         # NEW — Option A: online PPO update during deployment
  experiment3_main.py          # NEW — orchestrates Experiment 3

  outputs/
    experiment3/
      option_b/
        policy_with_r_reason/  # Checkpoints from Option B training
        policy_baseline/       # Checkpoints from baseline (R_env only on test dist)
        comparison_plots/
      option_a/
        online_correction/
        comparison_plots/
```

---

## Phase 1 — Implement R_reason Computation

Implement the violation score as a callable function that takes the current observation and policy state and returns R_reason in real time.

```
R_reason(obs, policy, SAE, G_star_profiles) → scalar
```

The function should:
1. Run the forward pass through the frozen SAE on the current representation
2. Compute live causal weights (EAP or patching — use whichever had better correlation in Experiment 2)
3. Check I1–I4 against G* profiles (fast lookups)
4. If I1–I4 flag anything, run I5 (one counterfactual forward pass)
5. Compute V_total = α · V_drop + β · V_gain + γ · 𝟙[I5]
6. Return R_reason = −V_total

Use α = β = γ = 1.0 as a starting point. Log these as hyperparameters. The SAE and G* profiles are frozen throughout — they do not update.

Validate R_reason on 20 training-distribution episodes. It should return near-zero (no violations when policy is on training distribution with fixed goal). Log the mean absolute R_reason on training distribution — this is the noise floor. If it is not near zero, the thresholds are miscalibrated.

Also validate R_reason on 20 test-distribution episodes without correction (pure observation). It should fire consistently. Log the step at which it first fires in each episode — this is k_graph from Experiment 2.

---

## Phase 2 — Option B: Second Training Phase With R_reason

**Baseline condition:**
Load the Experiment 1 trained policy. Run a second PPO training phase of 100k steps on the test distribution (goal randomised) with reward = R_env only. This is the baseline — PPO adapts to the test distribution but without any circuit-level signal. Save checkpoint as `policy_baseline_option_b.zip`. Evaluate on 20 test-distribution episodes. Record failure rate, mean episodic reward, mean episode length.

**R_reason condition:**
Load the same Experiment 1 trained policy (fresh copy — not the baseline). Run a second PPO training phase of 100k steps on the test distribution with reward = R_env + λ · R_reason, where λ = 0.5 initially. The SAE and G* profiles are frozen throughout — only the policy weights update. Save checkpoint as `policy_with_r_reason_option_b.zip`. Evaluate on 20 test-distribution episodes. Record failure rate, mean episodic reward, mean episode length.

**Compare:**
- Failure rate: baseline vs R_reason condition
- Mean episodic reward: baseline vs R_reason condition
- V_total trajectories: does R_reason condition show lower V_total over time? (Evidence of correction)
- G_live comparison: in successful episodes of the R_reason condition, has the causal circuit improved? (Evidence of circuit-level repair, if it occurs)

**λ sweep:** Run Option B with λ ∈ {0.1, 0.5, 1.0, 2.0}. For each λ, run 3 seeds. This gives 12 runs total (4 × 3). Measure failure rate per λ. The optimal λ is the one that minimises failure rate without destabilising training. Plot failure rate vs λ.

**Time budget for Option B:** Each 100k-step run should take 15–30 minutes on MPS. 12 runs = 3–6 hours. This is tight. If it is running over time, reduce to 50k steps per run and note the reduction.

---

## Phase 3 — Option A: Online Correction During Deployment (If Time Permits)

Only run this if Option B has completed within 4 hours.

**Setup:** Load the Experiment 1 trained policy (unfrozen — it will receive updates). Deploy on the test distribution. At each step, compute R_total = R_env + λ · R_reason. Run PPO gradient updates every n_steps = 512 steps (shorter rollouts than training, to allow faster adaptation). Run for 50 episodes.

**Control:** Same deployment but with λ = 0 (R_total = R_env only). 50 episodes.

**Compare failure rates.** This tests whether the signal corrects behaviour within a single deployment run, not a separate training phase.

**Important:** After each PPO update in Option A, check whether the training-distribution performance has degraded. Run 5 training-distribution evaluation episodes every 10k steps. If training performance drops below 0.8 (from the training-time 1.0), the online fine-tuning is catastrophically forgetting — flag this and reduce λ.

---

## Phase 4 — Behavioral vs Circuit Correction Diagnosis (H3)

For successful correction episodes (episodes where R_reason agent succeeds but baseline agent fails):

Run both G_live and raw activation monitoring at each step of the episode. Examine:

1. **Did the circuit repair?** Check whether goal feature causal weights (c_live for coin_tracking features) recovered toward G* values during the episode. If yes — circuit repair occurred. If no — behavioral correction without circuit repair.

2. **What actions changed?** Compare the action distribution at the step when V_total first fired, between the R_reason agent and the baseline agent. Did the R_reason agent take a different action at that step? What was the nature of the action difference?

3. **Did proxy features desaturate?** Check whether proxy feature causal weights decreased in the R_reason agent compared to the baseline. This would indicate the agent learned to route computation differently.

Log per episode: circuit_repaired (bool), action_changed_at_violation (bool), proxy_desaturated (bool). Report these as frequencies across all successful correction episodes.

---

## What Success Looks Like

**Strong success (H1 confirmed):** The R_reason condition shows meaningfully lower failure rate than the baseline across seeds and λ values. Even without circuit repair, behavioral correction is demonstrated. The paper can claim: "R_reason prevents goal misgeneralization from manifesting as behavioral failure in 3-layer system deployment."

**Moderate success:** Failure rate is lower in R_reason condition but the effect is small or inconsistent across seeds. Failure rate at the best λ is reduced but not eliminated. This is still publishable with honest framing.

**Null result:** Failure rate is not statistically different between R_reason and baseline conditions. The signal does not produce meaningful behavioral correction in this setting. This is an important negative result — document carefully. The likely explanation: PPO on a small policy with 100k steps cannot overcome the training-time goal representation in the k-step window. The fix might be activation steering (Layer 3) rather than reward shaping.

**Unexpected circuit repair:** G_live shows goal feature edges recovering during R_reason condition episodes. This would be a stronger result than expected — the gradient path from R_reason to internal representations is more effective than predicted. Document this carefully as a finding.

---

## LOG.md Instructions

Continue the existing LOG.md. Prefix Experiment 3 entries with `[EXP3]`. Log: which option is running, λ value, seed, failure rate per checkpoint, R_reason noise floor validation result, any catastrophic forgetting events, time per run. Update every 30 minutes during active computation.

---

## EXPLAINER3.md Instructions

Write EXPLAINER3.md as a standalone document when the experiment is complete. A reader who has read EXPLAINER.md and EXPLAINER2.md should be able to read this and understand what was tested and what was found. Cover:

1. **The question** — what does it mean to "correct" vs "detect"? Why behavioral correction without circuit repair is still a valid and useful result.

2. **How R_reason was computed** — the violation score, the five invariances, how it was validated on training distribution (noise floor), how it behaved on test distribution before correction.

3. **Option B results** — failure rates across λ values and seeds. The λ sweep plot. Whether H1 was confirmed.

4. **Option A results** — if run, whether online correction worked. Whether catastrophic forgetting occurred.

5. **H3 diagnosis** — was correction behavioral or circuit-level? What changed in the agent's behaviour and representations during successful correction episodes?

6. **What the results mean for the proposal** — if H1 is confirmed, the paper can claim a complete Layer 1 + Layer 2 result: circuit violations detected before failure (Experiment 1 + 2), and behavioral correction demonstrated (Experiment 3). If H1 is not confirmed, what does Layer 3 (targeted fine-tuning) need to do differently?

7. **Unexpected findings.**

8. **What comes next** — if correction works, move to testing on a different failure mode (reward hacking). If it does not, move to activation steering as the correction mechanism.

---

## Constraints

- **Total runtime: under 6 hours** on MacBook Air Apple Silicon.
- **Do not retrain from scratch.** Load Experiment 1 policy as the starting point for all Option B and Option A runs.
- **SAE and G* are always frozen.** Only the policy weights update.
- **The architecture does not change.** Same IMPALA CNN, same SAE, same K=32.
- **Run Option B first.** Only attempt Option A if Option B completes within 4 hours.
- **Log the λ sweep results in a table.** This is the key result plot for Experiment 3.
- **Every plot saved as PNG** to `outputs/experiment3/plots/` with descriptive filenames.
- Free to add analysis. Log everything.

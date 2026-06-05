# TASK 4 — The Full Pipeline
## Goal Misgeneralization Detection + Three-Response Comparison

---

## What This Experiment Is

This is the experiment the entire research programme has been building toward. Experiments 1–3 established the prerequisites and the failure modes. This experiment runs the full proposed pipeline — detection, then all three responses — on a policy that actually has a genuine goal representation.

The question is not whether the pipeline works in principle. The question is: **which response mechanism works best, and by how much?**

You will train a policy with a randomised goal position so it is forced to build a real goal-tracking feature. You will verify that feature exists. You will then induce goal misgeneralization and compare all three response mechanisms side by side against a baseline.

---

## Why The Previous Experiments Could Not Do This

Experiments 1–3 used a fixed-goal policy. The maximum actual-goal-tracking correlation across all 384 SAE features was r = 0.005 — the policy had no goal representation. Every downstream technique failed because the foundational assumption was violated.

This experiment fixes that by changing one thing in the training setup: the goal position is randomised every episode during training. The policy cannot memorise a fixed location. It must learn to read the goal from the current observation. This builds the feature the pipeline requires.

Everything else — IMPALA CNN architecture, SAE with resampling, W-matrix graph, five invariances, R_reason formula — is identical to what Experiments 1–3 developed and validated.

---

## The Core Hypothesis

> A PPO policy trained with a randomised goal position will develop at least one SAE feature with actual-goal-tracking correlation r > 0.3. When this policy undergoes goal misgeneralization under distribution shift, the causal circuit will show measurable drift (I3 + I4 violations) before behavioral failure. Three response mechanisms — R_reason reward shaping, activation steering, and targeted fine-tuning — will each reduce failure rate compared to the baseline, with different speed and durability profiles.

---

## The Four Hypotheses

**H1 — The goal representation exists**
After training with randomised goal positions, at least one SAE feature will have actual-goal-tracking correlation r > 0.3 across 100 test episodes with varied goal positions. Without this, the experiment stops — the prerequisite is not met and continuing would repeat the Experiment 3 failure.

**H2 — Invariance violations precede behavioral failure**
Under distribution shift, I3 (goal feature causal importance drops) and I4 (proxy feature causal importance rises) will fire before episodic reward degrades. Mean k > 0 across seeds. This is H3 from Experiment 1 but now on a policy with a real goal feature — so the result should be more mechanistically meaningful, not just a perceptual absence detector.

**H3 — All three responses reduce failure rate versus baseline**
R_reason, activation steering, and targeted fine-tuning will each produce lower failure rates than the no-response baseline under the same distribution shift. The magnitude of improvement and the speed of correction will differ.

**H4 — The responses have distinct performance profiles**
R_reason is fast but shallow — behavioral correction within the episode. Activation steering is faster and more direct — representation-level intervention. Targeted fine-tuning is slowest but most durable — weight-level repair that persists across episodes. The experiment measures each dimension explicitly.

---

## Prerequisites — Verify Before Proceeding

Before running any experiment, verify two things:

**Check 1 — Goal representation exists**
Train the policy. Train SAEv2. For each SAE feature, compute correlation between feature activation and actual goal position (Euclidean distance to goal cell) across 100 test episodes with randomised goal positions. At least one feature must have r > 0.3. Log the top 5 features by actual-goal-tracking correlation. If no feature exceeds 0.3, increase training steps or goal randomisation diversity and retrain. Do not proceed without this.

**Check 2 — W-matrix validates**
Compute W = D^T · W_enc^T. Validate against activation patching on 200 held-out observations. Pearson r must exceed 0.5. If below 0.5, the SAE is not clean enough — retrain with more resampling epochs.

---

## Hardware

MacBook Air, Apple Silicon M-series, 16 GB unified RAM, 512 GB SSD, MPS backend. Total runtime target: under 7 hours. Budget is tight — scale as needed and log all scale decisions.

---

## Build On Previous Experiments

Load from existing checkpoints where possible:
- SAEv2 architecture (384 hidden dim, resampling) from Experiment 2 — retrain on the new policy's activations, do not use old weights
- W-matrix computation code from FIX_GRAPH.md — reuse directly
- R_reason computation wrapper from Experiment 3 — reuse but rebuild on the new goal features, not the old position-proxy features
- Feature label JSON — rebuild from scratch on the new policy

---

## Repository Structure

Add to the existing experiment repository. Do not modify Experiments 1–3.

```
experiment/
  train_policy_randomgoal.py     # NEW — PPO training with goal randomised each episode
  verify_goal_representation.py  # NEW — checks H1 prerequisite before anything else
  response_r_reason.py           # NEW — Response 1: reward shaping during fine-tuning
  response_activation_steering.py # NEW — Response 2: direct representation intervention
  response_fine_tuning.py        # NEW — Response 3: targeted weight-level repair
  experiment4_main.py            # NEW — orchestrates full comparison

  outputs/
    experiment4/
      policy_randomgoal/         # New policy checkpoints
      sae_v3/                    # SAE retrained on new policy activations
      graphs/                    # G* for new policy
      responses/
        baseline/
        r_reason/
        activation_steering/
        targeted_finetuning/
      plots/
```

---

## Phase 1 — Train Policy With Randomised Goal

Train PPO with IMPALA CNN on CoinCollect 8×8 with `goal_fixed=False`. The goal position is sampled uniformly from all valid floor cells at the start of each episode. Every other hyperparameter identical to Experiment 1.

Training target: mean episodic reward > 0.7 on the randomised-goal training distribution. This is harder than the fixed-goal task — 0.7 is sufficient. Do not overtrain.

Training budget: 500k steps. If reward has not reached 0.7 by 500k steps, continue to 750k maximum. Log reward every 10k steps.

After training, evaluate on two conditions:
- **Training distribution:** 50 episodes, goal randomised. Record mean reward.
- **Test distribution (the misgeneralization setting):** 50 episodes, goal fixed at a position the policy never saw during training — use (2, 2), far from the centre. Record mean reward and failure rate. This is your baseline failure rate before any response.

Save checkpoint as `outputs/experiment4/policy_randomgoal/ppo_final.zip`.

---

## Phase 2 — Train SAEv3 And Verify H1

Collect 100k activation samples from the frozen new policy running on the training distribution. Train SAEv3 using the same architecture as SAEv2: 384 hidden dimension, K=32, Anthropic-style resampling (inactive for 150+ batches → reinitialise), 50 epochs minimum.

Target: fewer than 100 dead features (< 26%). Log dead features and validation MSE every 10 epochs.

After training, run the H1 prerequisite check:

For each of the top 50 features by activation frequency, compute:
- `actual_goal_corr`: Pearson r between feature activation and distance to actual goal position across 100 test episodes with varied goals
- `training_pos_corr`: Pearson r between feature activation and distance to (2, 2) — the test-distribution fixed position

**Decision gate:** If no feature has actual_goal_corr > 0.3, STOP. Log the maximum correlation found, hypothesise why it failed (not enough training diversity? too few steps?), and document in EXPLAINER4.md. Do not proceed to Phase 3.

If H1 passes, log the top 5 goal-tracking features with their correlations. These are your F_goal — the features the entire pipeline is built on. Save to `outputs/experiment4/goal_features.json`.

Compute W-matrix and validate (must pass r > 0.5 against patching). Build G* using W-matrix for inter-feature edges and raw representation patching for feature-to-action causal importance c*. Extract five invariant profiles.

---

## Phase 3 — Establish Baseline And Measure k

This is the control condition and the replication of the core Experiment 1 result on the new policy.

Deploy the frozen policy on the test distribution (goal fixed at (2, 2)) for 30 episodes across 3 seeds. At each step measure:
- Goal feature activation (mean of F_goal activations)
- Proxy feature activation (features with negative actual_goal_corr)
- V_total (W-based invariance violation score)
- Episodic reward

Measure k_activation and k_graph for each episode. Compare.

**This is the critical check.** If k_graph > k_activation — even slightly — the causal graph is detecting routing changes before the features fully deactivate. That is the result Experiment 2b was looking for but could not find because the policy had no goal representation.

Log mean k_graph, mean k_activation, and the difference across all 30 episodes. Plot representative episode showing both signals and the reward curve on the same axis.

Also record the baseline failure rate: proportion of episodes where the agent does not reach goal (2, 2) within max steps. This is your comparison denominator for Phase 4.

---

## Phase 4 — Three Response Mechanisms

Run all three responses under identical conditions: same policy starting point, same test distribution (goal at (2, 2)), same number of episodes, same 3 seeds. Each response is a separate condition. The baseline (no response, Phase 3) is the fourth condition.

---

### Response 1 — R_reason Reward Shaping

**What it does:** Feeds the violation score back as a negative reward. PPO updates the action distribution away from violation-triggering states. Behavioral correction — the agent avoids actions that cause circuit drift.

**Implementation:**
Use the R_reason wrapper from Experiment 3 but built on the validated goal features from Phase 2. The frozen reference is the Phase 2 policy. R_reason = −V_total at each step.

R_total = R_env + λ · R_reason

Run a λ sweep: {0.1, 0.5, 1.0}. 3 seeds per λ = 9 runs. 100k steps per run.

Before running, validate R_reason noise floor: on 20 training-distribution episodes (goal at training-time random positions), mean |R_reason| should be < 0.05. If it is not, the thresholds are miscalibrated — adjust and log.

**Measure:**
- Failure rate on test distribution (goal at (2, 2)) for each λ and seed
- Mean R_reason value per episode — is the signal firing as expected?
- Whether the agent catastrophically fails as in Experiment 3 — if yes, log immediately and diagnose why

**Expected result:** Lower failure rate than baseline at the best λ. The Experiment 3 catastrophe should not repeat because R_reason is now built on actual goal features, not position proxies.

---

### Response 2 — Activation Steering

**What it does:** When invariance violations fire (I3 specifically — goal feature causal importance dropping), directly inject a steering vector into the policy's representation at the relevant layer to boost the goal feature's activation. No gradient update. No reward modification. Direct circuit-level intervention.

**Implementation:**

At each deployment step:
1. Run forward pass through frozen policy → get representation r_t
2. Check I3: if mean c_live for F_goal drops below 60% of c*_goal_mean → flag
3. When flagged: compute the steering vector as the mean decoder direction of the top goal feature: v_steer = d_goal (the decoder direction of the highest actual_goal_corr feature)
4. Add the steering vector to the representation: r_steered = r_t + α · v_steer
5. Run r_steered through the policy head to get the steered action distribution
6. Take action from the steered distribution

α is the steering magnitude. Sweep α ∈ {0.5, 1.0, 2.0}.

**This requires no PPO update.** The policy weights never change. The intervention happens at inference time by modifying the representation before the policy head. This is what makes activation steering different from the other two responses — it is instantaneous and requires no training.

**Measure:**
- Failure rate under steering vs baseline
- How often the steering is triggered (what fraction of steps have I3 flagged)
- Whether over-steering occurs — if α is too large, the representation is pushed into an out-of-distribution region and behavior degrades. Monitor for this by checking if steered episodes have lower reward than baseline at high α.

---

### Response 3 — Targeted Fine-Tuning

**What it does:** After a deployment episode where violations accumulated, use the violation records to construct a targeted fine-tuning loss that directly penalises the observed circuit drift. Then redeploy the repaired policy.

**Implementation:**

Run 10 deployment episodes on the test distribution (goal at (2, 2)) with the frozen policy. Record violation records for each episode: which features drifted, how far, in which direction.

Construct the fine-tuning loss from violation records:

```
L_finetune = MSE(c_live_goal, c*_goal) + MSE(c_live_proxy, c*_proxy)
```

This directly penalises goal feature causal importance dropping and proxy feature causal importance rising. It is not R_env — it is a circuit-level repair objective.

Fine-tune for 5k steps using this loss with a small learning rate (1e-5). Do not use R_env during fine-tuning — only the circuit repair loss.

After fine-tuning, evaluate on 20 fresh test-distribution episodes. Record failure rate.

Run 3 seeds (3 separate fine-tuning runs from the same base policy). Compare failure rate to baseline.

Also check: does fine-tuning preserve training-distribution performance? Run 20 training-distribution episodes after fine-tuning. If mean reward drops below 0.6, catastrophic forgetting has occurred — log this.

**The key measurement unique to targeted fine-tuning:** run G* vs G_live comparison after fine-tuning. Has the actual circuit repaired — do goal feature causal weights recover toward c*? This is the measurement that distinguishes genuine circuit repair from behavioral masking.

---

## Phase 5 — Side-By-Side Comparison

Compile results across all four conditions into a comparison table:

| Condition | Failure Rate | Mean k (if measured) | Circuit Repaired? | Persists After 1 Episode? |
|---|---|---|---|---|
| Baseline (no response) | — | — | — | — |
| R_reason (best λ) | — | — | No (behavioral) | No (per-episode) |
| Activation Steering (best α) | — | — | No (inference-time) | No (weights unchanged) |
| Targeted Fine-Tuning | — | — | Measure directly | Yes (weights updated) |

**The performance difference across responses is the key result of this experiment.**

Expected ordering by failure rate reduction:
- Targeted fine-tuning: highest reduction (durable, weight-level)
- Activation steering: second (direct, no gradient lag)
- R_reason: third (indirect, gradient path from reward to circuit)
- Baseline: worst

If this ordering holds, you have demonstrated the full three-layer system with a performance gradient that justifies having all three layers.

If the ordering is different — document why. Any deviation from the expected ordering is a scientifically interesting finding.

Additionally for each response, measure and report:
- **Speed:** How quickly does failure rate drop within the episode (R_reason, activation steering) or across episodes (fine-tuning)?
- **Stability:** Does the response remain effective across 3 seeds or is it high-variance?
- **Cost:** Approximate compute overhead per step (activation steering is cheapest, targeted fine-tuning most expensive)

---

## What Success Looks Like

**Minimum success (paper exists):** H1 passes (goal representation found), H2 passes (k > 0), at least one response mechanism reduces failure rate below baseline. Even if only targeted fine-tuning works, the paper demonstrates detect → diagnose → repair.

**Strong success (strong paper):** All three responses reduce failure rate. Clear performance ordering matches expected hierarchy. k_graph > k_activation — the causal graph gives earlier warning than activation monitoring. The circuit repairs under targeted fine-tuning (measured directly in G* vs G_live comparison post fine-tuning).

**Discovery result (best paper):** k_graph substantially exceeds k_activation (> 20 steps), demonstrating that causal routing breaks before features deactivate — the graph catches something activation monitoring misses. This would be the first mechanistic evidence that goal-routing circuits degrade before goal features deactivate, which is the claim the proposal was originally built around.

---

## LOG.md Instructions

Continue the existing LOG.md. Prefix Experiment 4 entries with `[EXP4]`. Log everything: H1 check result (the maximum actual-goal-tracking correlation and which feature), W-matrix validation r, Phase 3 k values, R_reason noise floor validation, each response run with λ/α/seed and resulting failure rate, catastrophic forgetting checks, and the final comparison table. Update every 30 minutes during active computation.

**Critical: if H1 fails — if no feature exceeds r = 0.3 — log this immediately and stop. Do not proceed.**

---

## EXPLAINER4.md Instructions

Write EXPLAINER4.md when complete. This document should be readable as a standalone summary of the full research programme. Cover:

1. **What the programme learned across Experiments 1–3** — the Type A / Type B distinction, why the previous policy failed, what the prerequisite is.

2. **Whether H1 passed** — did the randomised-goal policy build a genuine goal representation? What does the best goal-tracking feature look like? What correlation did it achieve?

3. **Whether k_graph > k_activation** — this is the central causal graph question. Document the result and what it means.

4. **The three-response comparison** — the full table. Which worked best, by how much, and why.

5. **The performance difference across responses** — which layer of the pipeline adds the most value? Is targeted fine-tuning worth the extra complexity over activation steering?

6. **The unified conclusion** — what does the complete four-experiment programme establish about mechanistic monitoring of RL agents?

7. **What the paper can claim** — honest statement of what is supported by evidence versus what remains future work.

---

## Constraints

- **Total runtime: under 7 hours** on MacBook Air Apple Silicon.
- **Stop at H1 if prerequisite fails.** Do not run Phases 3–5 on a Type B policy. Document and stop.
- **The architecture does not change.** IMPALA CNN, SAEv2-style training, W-matrix, same K=32. The only change from previous experiments is `goal_fixed=False` in the environment.
- **Run responses sequentially**, not in parallel — memory constraint on 16 GB unified RAM.
- **Save all intermediate outputs** — policy checkpoints, G* matrices, per-episode violation records, post-fine-tuning G_live comparisons.
- **Every plot as PNG** to `outputs/experiment4/plots/` with descriptive filenames.
- Free to add any additional analysis. Log everything.

# TASK — Initial Feasibility Experiment
## Causal Circuit Interpretability in a Trained RL Policy

---

## What This Experiment Is About

This is the **first empirical test** of a research pipeline that proposes to detect and correct RL deployment failures by reading the agent's internal causal reasoning circuits.

Before the full system can be built, one foundational question must be answered empirically:

> **Do Sparse Autoencoder (SAE) features trained on a frozen PPO policy's activations correspond to interpretable, meaningful visual concepts — and do those features shift in a structured, predictable way when the agent undergoes goal misgeneralization?**

If the answer is yes, the entire research programme is viable. If no, the methodology must be reconsidered before anything else is built.

This experiment does not implement the full system. It tests whether the core mechanistic assumption holds on real trained RL agents.

---

## The Research Idea (Read This First)

The hypothesis underlying this research is as follows.

When an RL agent trains on a procedurally generated environment, it learns to map observations to actions. Inside that mapping — across the intermediate layers of its neural network — there exists a causal reasoning chain. Features at early layers detect raw visual concepts (edges, colours, object positions). Features at later layers compose these into higher-level representations (threat proximity, goal location, navigable path). The action output is causally determined by these intermediate features.

The core claim is that this causal chain has **stable structural properties** during successful training — specific features carry the causal load, specific pathways remain active, and the graph of causal influence between features has measurable invariant properties.

When the agent undergoes **goal misgeneralization** — when the environment shifts and the agent pursues the wrong objective — the hypothesis predicts that the causal chain shifts in a specific, measurable way **before** the reward curve degrades. In particular: features that causally track the true goal should lose weight in the circuit, while features that were spuriously correlated with reward during training should gain weight.

This experiment tests whether:
1. SAE features on this architecture are interpretable enough to identify "goal features" and "proxy features" manually
2. The causal graph built from those features has meaningful structure
3. When goal misgeneralization is induced, the features shift in the predicted direction before behavioral failure

---

## The Three Hypotheses Being Tested

**H1 — SAE Interpretability**
A Top-K Sparse Autoencoder trained on the frozen intermediate activations of a PPO policy network (IMPALA CNN) will produce monosemantic features that correspond to identifiable visual concepts in the environment. At least a meaningful subset of features should be manually labelable as referring to specific game elements (coin/key position, agent position, wall proximity, enemy location, background texture, etc.).

*If this fails:* Features are polysemantic noise. SAE architecture, layer choice, or K value needs revision. The pipeline cannot proceed.

**H2 — Causal Graph Structure**
Features identified as "goal-relevant" (those whose activation correlates with the true reward signal and which survive basic intervention testing) will form a connected subgraph in the activation-patching-derived causal graph that meaningfully predicts the agent's action. Features identified as "spurious" (correlated during training but not causally valid) should show weak or absent causal edges to the action output.

*If this fails:* Either the causal graph extraction is too noisy, or the policy does not have separable causal structure at this scale. Need to investigate whether a larger policy or different environment produces cleaner structure.

**H3 — Pre-Failure Mechanistic Signature**
When goal misgeneralization is induced at a known timestep (coin position shifted from training position to randomised position), the activation of goal-relevant SAE features should begin to decline **before** the episodic reward curve shows measurable degradation. The proxy/spurious features should correspondingly increase in activation strength before reward drops.

*If this fails:* Either the violation does not precede behavioral failure (k ≈ 0, which would be a significant negative result worth recording), or the SAE features are not tracking the relevant causal structure. Record the gap k carefully regardless of its sign or magnitude.

---

## Environment and Architecture

**Environment:** CoinRun (from the `procgen` library)
- Procedurally generated platformer
- Training distribution: coin always appears at end of level
- Test distribution: coin position randomised — agent trained to go to end of level will pass the coin
- This is the canonical goal misgeneralization testbed from Langosco et al. (ICML 2022)
- Well-understood failure mode, clean controlled distribution shift

**Policy:** PPO with IMPALA CNN
- Standard IMPALA CNN architecture (used in the original Procgen paper)
- 3 convolutional blocks with residual connections
- Final flattened representation → policy head and value head
- This architecture has meaningful intermediate representations across multiple layers — essential for SAE to find useful features

**SAE:** Top-K Sparse Autoencoder
- Applied to the **flattened intermediate representation** after the IMPALA CNN body, before the policy head — this is where the richest compositional features should exist
- Top-K architecture: exactly K features active per forward pass (hard sparsity, no soft penalty)
- K = 32 (tune down to 16 if training is slow, do not go below 16)
- Hidden dimension: 4× the input dimension of the layer being decomposed
- Train on activations collected from a large offline rollout corpus of the frozen policy

**Hardware:** MacBook Air with Apple Silicon (M-series chip), 16 GB unified RAM, 512 GB SSD. There is no discrete GPU. All computation runs on CPU with optional Metal/MPS acceleration where PyTorch supports it. This is a meaningful constraint — plan around it explicitly.

**What this means in practice:**
- Use PyTorch MPS backend where available: `device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")`. MPS gives 2–4× speedup on neural net forward/backward passes vs pure CPU on Apple Silicon.
- RAM is unified — the policy, SAE, activation dataset, and OS all share 16 GB. The activation dataset in memory must never exceed ~3–4 GB. Use memory-mapped numpy arrays (`np.memmap`) or stream activations from disk in batches rather than loading everything at once.
- The SSD is fast (NVMe-class) — disk I/O is not a bottleneck. Cache everything to disk freely. This is your substitute for GPU VRAM.
- `procgen` requires compilation from source on macOS and has known compatibility issues with Apple Silicon. **Check first whether procgen installs cleanly. If it does not install within 15 minutes, switch immediately to MiniGrid** (`pip install minigrid`, pure Python, fully macOS-compatible, no compilation required). Use `MiniGrid-FourRooms-v0` or a custom key/coin task that reproduces the same training/test split concept as CoinRun — fixed goal position during training, randomised goal position at test time. Log the switch in LOG.md with the reason. The underlying architecture — PPO + IMPALA CNN + Top-K SAE + causal graph — does not change. The environment is a testbed, not the subject of the experiment.

**Training budget scaled for MacBook Air CPU/MPS — 6-7 hour total target:**
- PPO training: **500k environment steps** (reduced from 2M). On CPU/MPS with MiniGrid or procgen, 500k steps should take 1.5–3 hours. Target mean episodic reward > 5 on a 0–10 scale — competent but not overfit. If reward has clearly plateaued before 500k steps, stop early and checkpoint. If 500k steps would take more than 3 hours, reduce to 200k and log the decision.
- SAE training: **50k–100k activation samples** collected from rollouts of the frozen policy. Train for 30–50 epochs, stop early if reconstruction loss has not improved for 5 consecutive epochs. On CPU/MPS this should take 20–45 minutes.
- Causal graph extraction: **2k rollouts, top 16 most active features only** for patching (16×16 intervention matrix, not 32×32). On CPU this should take 30–60 minutes.
- Goal misgeneralization sweep: **10 evaluation episodes per seed, 3 seeds**, measuring signals every 5 environment steps. Should take 30–60 minutes.
- Total target: under 6 hours wall-clock. If any phase runs long, reduce scale first, then log the actual time and scale used.

**The architecture does not change because of the hardware.** IMPALA CNN stays IMPALA CNN. Top-K SAE stays Top-K. K=32 stays K=32 — reduce to K=16 only if SAE training exceeds 1 hour after 20 epochs, and log this. The experiment tests a scientific hypothesis. Smaller scale means noisier estimates, not a different experiment. Document all scale reductions in EXPLAINER.md so a reader understands the limits of the results.

---

## What To Build — Repository Structure

You are free to design any repository structure and architecture you think is best. The following is guidance, not a constraint.

Suggested top-level layout:
```
experiment/
  train_policy.py          # PPO training on CoinRun or MiniGrid training distribution
  train_sae.py             # SAE training on frozen policy activations
  extract_graph.py         # Causal graph extraction via activation patching
  induce_misgeneralization.py  # Deploy shifted environment, measure signals
  analyze.py               # Produce all plots and summary statistics
  
  configs/
    policy.yaml
    sae.yaml
    graph.yaml
    
  outputs/
    checkpoints/           # Policy and SAE checkpoints
    activations/           # Cached activation datasets (memmap or pickle)
    graphs/                # Saved causal graphs
    plots/                 # All generated figures as PNG
    
  LOG.md                   # Updated continuously throughout the run
  EXPLAINER.md             # Written at the end of the experiment
```

You are free to restructure this however you like. Use wandb or tensorboard for training curves if available — but both are optional, matplotlib is sufficient. Use numpy/pickle or memmap for saving intermediate results. Use matplotlib or seaborn for plots. Use stable-baselines3 or cleanrl for PPO — whichever installs cleanly on macOS Apple Silicon.

At the top of every script, include a device check and log the result:
```python
import torch, gc
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")
```

Between phases, explicitly free memory:
```python
gc.collect()
# On MPS:
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
```

---

## Phase 1 — Train the Policy

Train a PPO agent on CoinRun with the following setup:

- **Training distribution:** Standard CoinRun with coin always at a fixed relative position (end of level). Use `start_level=0`, `num_levels=200` to create a moderately diverse but learnable training set.
- **Target performance:** Agent should reliably reach the coin in training levels (mean episodic reward > 7 out of 10 maximum). Do not overtrain — a policy that is competent but not perfectly optimised will show cleaner goal misgeneralization.
- **Save checkpoints** at 500k, 1M, and 2M steps.
- **Log every 10k steps:** mean episodic reward, mean episode length, policy entropy, value loss, policy loss.
- After training, run 50 evaluation episodes on the **training distribution** and 50 on the **test distribution** (coin position randomised, `start_level=500`, `num_levels=0`). Record both reward distributions. The gap between these two is your baseline confirmation that goal misgeneralization exists.
- The policy is frozen after this phase. No further gradient updates to the policy from this point.

---

## Phase 2 — Train the SAE

Collect an offline activation dataset from the frozen policy:

- Run the frozen policy on the training distribution for enough rollouts to collect 200k–500k activation samples
- Extract activations at the layer immediately before the policy head (the final flattened representation after all convolutional blocks)
- Save these as a dataset of (activation_vector, observation_image, action_taken, reward_received) tuples — the metadata is needed for feature analysis

Train the Top-K SAE on this activation dataset:

- Architecture: Linear encoder → Top-K hard gate (exactly K neurons active) → Linear decoder
- Loss: Reconstruction MSE only (no L1 penalty — Top-K handles sparsity)
- K = 32. Hidden dimension = 4 × input dimension.
- Train until reconstruction loss plateaus. Log reconstruction loss every epoch.
- Save the final SAE checkpoint.

After training, compute and log:
- **Reconstruction quality:** Mean squared error on a held-out validation set of activations. Record this as the primary SAE quality metric.
- **Dead features:** How many of the hidden features never activate (or activate on < 0.1% of samples)? A high dead feature count signals the SAE is not utilising its capacity.
- **Feature activation frequency distribution:** Plot a histogram of how often each feature activates across the validation set. A healthy SAE should have a roughly power-law distribution — a few features very active, most moderately active, none completely dead.

---

## Phase 3 — Feature Interpretability Analysis

This is the most important qualitative step. For each of the top 50 most frequently activating SAE features:

- Collect the 20 observations from the validation set that maximally activate that feature
- Save these as a grid of images (5×4 grid per feature)
- For each feature, also save the 20 observations that minimally activate it (near-zero activation)
- Save all grids to `outputs/plots/feature_max_activations/` and `outputs/plots/feature_min_activations/`

Additionally, for each of the top 50 features, compute:
- **Spatial correlation:** Does the feature activate preferentially when specific game elements are in specific screen regions? Compute correlation between feature activation and position of: agent, coin, enemies, platforms in the observation. Report as a heatmap per feature.
- **Reward correlation:** Does the feature's activation magnitude correlate with immediate reward? With future discounted reward?
- **Action correlation:** Does the feature's activation predict which action the policy takes?

Label each feature manually after inspecting the max-activation images. Suggested labels: `coin_tracking`, `agent_position`, `platform_edge`, `enemy_proximity`, `background_texture`, `wall_detection`, `unknown`. Record labels in a JSON file: `outputs/feature_labels.json`.

Identify and flag:
- **Goal features:** Features that show high spatial correlation with the coin position and high reward correlation
- **Proxy features:** Features that activated consistently during training but do not track the coin causally (e.g., background texture features that co-occurred with the coin's training-time position)
- **Spurious features:** Features with high action correlation but weak reward correlation

These labels are the ground truth for the rest of the experiment.

---

## Phase 4 — Causal Graph Extraction (Lightweight)

Build a lightweight causal graph G* using activation patching on the frozen policy + frozen SAE.

For each pair of SAE features (i, j) in the top 32 most active features:
- Run the policy forward on a batch of 100 observations
- Zero out feature i in the SAE's output (set its activation to 0)
- Run the patched activations through the remaining network
- Measure the change in feature j's activation at the next layer and the change in the action probability distribution
- Record edge weight as: `|mean_change_in_feature_j| × feature_i_activation_magnitude`

This produces a 32×32 weighted directed adjacency matrix — the causal graph G*.

Additionally, for each of the top 32 features, run the I5 self-consistency test:
- Identify the feature with the highest causal weight to the action output
- Zero it out
- Check whether the action distribution changes meaningfully (KL divergence > 0.1 threshold)
- Record pass/fail per observation and report the pass rate

Log:
- The full G* adjacency matrix as a heatmap
- Which features have strong causal edges to the action output (the "causally dominant" features)
- Whether causally dominant features correspond to the goal features identified in Phase 3

---

## Phase 5 — Goal Misgeneralization with Signal Measurement

This is the core experimental measurement.

Set up a deployment scenario where goal misgeneralization is induced at a known step:

- Run the frozen policy on the test distribution (coin randomised). The agent will walk past the coin to its training-time position — this is goal misgeneralization.
- At each step during each episode, extract SAE feature activations from the frozen SAE
- Compute and log the following signals at each step:
  - **Goal feature signal:** Mean activation magnitude of the features labelled as goal features in Phase 3
  - **Proxy feature signal:** Mean activation magnitude of the features labelled as proxy/spurious features
  - **Causal depth signal:** The length of the longest causally connected path from input features to action, estimated from G*
  - **Episodic reward:** Standard reward signal

Run 20 episodes per seed, 3 seeds = 60 total evaluation episodes.

For each episode, record:
- Step at which goal feature activation first drops below 50% of its training-time baseline
- Step at which proxy feature activation first rises above 150% of its training-time baseline  
- Step at which episodic reward first drops below 50% of the training-distribution mean
- The gap k = (reward degradation step) − (feature shift step). Positive k means violation precedes failure.

Report:
- Mean k across all episodes and seeds, with standard deviation
- Distribution of k values as a histogram
- A representative plot showing goal feature signal, proxy feature signal, and reward signal on the same time axis for a single episode — this is the key figure

Also run the same measurement on 20 training-distribution episodes as a control. Feature signals should remain stable and k should be undefined (no failure to detect).

---

## What Success Looks Like

**Strong success:** Mean k > 20 steps, consistent across seeds (low variance), goal features and proxy features shift in the predicted directions before reward degrades. Causally dominant features from Phase 4 correspond to goal features from Phase 3.

**Moderate success:** Mean k > 0 but small (5–20 steps), or feature shifts are in the right direction but noisy. Still publishable. Document clearly.

**Negative result:** Mean k ≈ 0 (violations fire at the same time as or after behavioral failure), or feature shifts do not correspond to predicted patterns. This is a scientifically important finding — do not hide it. Record exactly what was observed and propose a hypothesis for why (SAE not faithful enough? Policy too small? Wrong layer?).

**SAE failure:** Reconstruction error is too high, or features are not interpretable. This is the most important negative result because it invalidates everything downstream. Stop at Phase 2 and document thoroughly.

---

## LOG.md — Instructions for the Agent

Maintain a file called `LOG.md` at the root of the repository. This is a live running log that must be updated continuously throughout the experiment. It is the complete audit trail.

**Format:** Append entries in reverse-chronological order (newest at top). Each entry must have a timestamp, a one-line status summary, and detail.

**What to log — log everything:**
- When each phase starts and ends
- Every checkpoint saved (with step count and performance metrics at that point)
- Every hyperparameter that was set or changed and why
- Every error encountered and how it was resolved
- Every intermediate result (reconstruction loss at each epoch, reward at each eval)
- Every time a script is run (with which arguments)
- When any plot is saved (with filename)
- Feature labels as they are assigned in Phase 3
- G* adjacency matrix values (abbreviated — top 5 strongest edges)
- The k measurement for each seed and the mean
- Any unexpected behaviour observed
- Any decision made that was not specified in this task document

The LOG.md should be updated at minimum every 30 minutes during active computation. If a training run is in progress, log the current loss/reward every time a checkpoint is evaluated. A reader should be able to reconstruct exactly what happened and when by reading only LOG.md.

**Example entry format:**
```
## [HH:MM] Phase 2 — SAE training epoch 47/100 complete
- Reconstruction loss: 0.0234 (down from 0.0891 at epoch 1)
- Dead features: 4 / 128 (improved from 12 at epoch 20)
- No changes to hyperparameters
- Next checkpoint at epoch 50
```

---

## EXPLAINER.md — Instructions for the Agent

Write a file called `EXPLAINER.md` at the root of the repository. This is written **after the experiment is complete**, once all results are in. It is the document a non-expert reader reads to understand everything that happened.

**EXPLAINER.md must cover, in this order:**

1. **What we were trying to find out** — one paragraph explaining the research question in plain language. Why does this matter. What would a positive result mean for the research programme.

2. **What we did** — a step-by-step description of each phase, written as if explaining to someone who did not read this TASK.md. What was trained, what was measured, and why each step was necessary before the next one.

3. **What the SAE features looked like** — describe the interpretability results from Phase 3 in plain language. Were the features clean? What did the goal features look like? What did the spurious features look like? Include the feature label distribution (how many goal, proxy, spurious, unknown). Explain what this means for the hypothesis.

4. **What the causal graph looked like** — describe G* in plain language. Which features had the strongest causal edges to the action? Did these correspond to goal features? Include the top 5 edges by weight.

5. **What happened when goal misgeneralization was induced** — describe the Phase 5 results. What was the mean k? Show the representative plot (reference the filename). Did goal features drop before reward degraded? Did proxy features rise? Were the results consistent across seeds?

6. **What the results mean for the research** — evaluate each of the three hypotheses (H1, H2, H3) against the evidence. State clearly whether each was supported, partially supported, or refuted. Be direct and honest — do not oversell the results.

7. **What should happen next** — given these results, what is the next experiment? If the results were positive, what needs to be added to build the full pipeline? If negative, what needs to change?

8. **Unexpected findings** — anything that happened that was not predicted by the hypothesis. These are often the most interesting results.

**Tone:** Write clearly and plainly. No jargon without explanation. A researcher reading this for the first time should understand everything that happened, what was found, and what it means, in a single read. Every claim must reference a specific result — do not write generalities.

---

## Constraints and Practical Notes

- **Hardware:** MacBook Air, Apple Silicon, 16 GB unified RAM, 512 GB SSD, no discrete GPU. Everything runs on CPU or MPS. Respect the memory budget — keep the active in-memory footprint under 8 GB at all times. Use disk as extended memory freely.
- **Total runtime must be under 6-7 hours** on this machine. Refer to the scaled budget in the Hardware section above. If any phase is running over budget, reduce scale, log the decision with exact timing, and continue. Do not wait for a phase to finish if it has clearly gone over time.
- **procgen may not install on macOS/Apple Silicon.** Check first. If installation fails or takes more than 15 minutes, switch to MiniGrid. This is not a fallback of last resort — it is an expected contingency. MiniGrid is a legitimate testbed for this experiment.
- **The core architecture does not change for hardware reasons.** IMPALA CNN, Top-K SAE, activation patching, causal graph — these stay as specified. The only things that change are dataset sizes, episode counts, and step counts.
- **Use MPS where available.** Always check `torch.backends.mps.is_available()` at startup and log which device is being used. Not all PyTorch operations support MPS — if an operation falls back to CPU, log it but do not block on fixing it.
- **Do not implement EAP or the deployment reward signal.** This experiment only tests whether the foundational mechanistic assumptions hold. Layers 2 and 3 of the full system are not part of this experiment.
- **Use stable-baselines3 or cleanrl for PPO.** Do not implement PPO from scratch — it is not the focus.
- **Save all intermediate results** (activation datasets, SAE checkpoints, G* matrix, per-episode k measurements) so the experiment can be resumed or analysed further without re-running. On a 512 GB SSD this is not a concern — save liberally.
- **Every plot must be saved as a PNG** to `outputs/plots/` with a descriptive filename. Do not only display plots inline.
- **Memory management matters on this machine.** After the policy is frozen and the SAE is trained, delete any optimiser states and gradient buffers from memory before the next phase. Call `gc.collect()` explicitly between phases. Log peak memory usage at each phase transition.
- You are free to add any additional analysis you think is scientifically useful beyond what is specified here. Log all additions in LOG.md.
- You are free to create any repository structure, helper modules, configuration files, or utilities you find useful.
- If you encounter a genuine technical blocker that prevents a phase from completing, document it fully in LOG.md and EXPLAINER.md and proceed to whatever phases are still possible.

# Causal Circuit Interpretability in a Trained RL Policy

Mechanistic interpretability applied to a reward-hacking reinforcement learning agent. The project trains a PPO policy on a custom gridworld, attaches a Sparse Autoencoder to decompose its hidden states into interpretable features, discovers which features causally drive reward hacking using attribution patching, and deploys a lightweight invariance-based detector that flags hacking episodes in real time.

---

## What This Project Does

A trained RL agent can learn to exploit shortcuts in its reward signal rather than pursuing the intended goal — a phenomenon called **reward hacking**. This project asks: can we look *inside* the agent's neural network and identify the specific computational structures responsible for this behaviour, then use that knowledge to detect it automatically?

The answer is yes, but with important caveats about what automated feature discovery can and cannot guarantee. The full technical account — motivation, methods, results, limitations, and all diagrams — is in [SYSTEM_EXPLAINER.md](SYSTEM_EXPLAINER.md).

---

## The Environment — CoinHack

An 8×8 MiniGrid in which the agent starts at position (1, 1) and must reach a coin (the true goal, reward +1.0). A shortcut exists at position (2, 2) that yields +0.3 reward at a fraction of the path cost.

During training the shortcut is unavailable, so the agent learns to pursue the true goal. During evaluation, the shortcut is introduced. An agent that has truly learnt to pursue the coin ignores the shortcut. An agent that learnt a positional heuristic takes it — **goal misgeneralization**.

```
┌─────────────────┐
│ . . . . . . . . │
│ . A S . . . . . │   A = Agent start (1,1)
│ . . . . . . . . │   S = Shortcut    (2,2)  +0.3
│ . . . . . . . . │
│ . . . . . . . . │
│ . . . . . . . . │
│ . . . . . . . . │
│ . . . . . . . G │   G = True goal   (6,6)  +1.0
└─────────────────┘
```

---

## Architecture Overview

```
Observation (8×8×3 pixels)
    │
    ▼
IMPALA CNN  ──►  256-dim hidden state  h_cnn
    │
    ▼  (TopK SAE)
384-dim sparse feature vector  h   (exactly 32 of 384 non-zero)
    │
    ▼
Linear action head  W_action (7×256)
    │
    ▼
Action logits  (7 actions)
```

Because there is no nonlinear layer between the SAE output and the action head, the full path from feature to logit is:

```
action_logits = W_action @ W_dec @ h  =  C @ h
```

where `C = W_action @ W_dec` is a precomputed (7×384) circuit coefficient matrix. This linear structure makes attribution patching exact — no approximations, no extra forward passes.

---

## Five-Phase Experiment

| Phase | Script | What It Does |
|-------|--------|-------------|
| 1 — Train policy | `train_policy.py` | PPO + IMPALA CNN, 500k steps on CoinCollect (no shortcut) |
| 2 — Train SAE | `train_sae.py` | TopK SAE v2 (K=32, 384 features) on 100k collected activations |
| 3 — Feature analysis | `analyze_features.py` | Identify and label interpretable features |
| 4 — Causal graph | `extract_graph.py` | Build feature-level causal graph from activation data |
| 5 — Misgeneralization | `induce_misgeneralization.py` | Evaluate agent on shortcut-enabled distribution |

After these five phases, the attribution pipeline and detection framework are run separately (see [Running the Detector](#running-the-detector) below).

Run all five phases in sequence:

```bash
bash run_experiment.sh
```

Expected runtime: 4–7 hours on Apple Silicon M-series (MPS).

---

## Key Results

**Goal misgeneralization confirmed** — the agent took the shortcut on the vast majority of evaluation episodes when it became available, despite never encountering it during training.

**Attribution patching** (Marks et al., ICLR 2025) automatically discovered 16 causally important features from the 384-dimensional SAE space:

| Group | Features | IE Score Range | Role |
|-------|----------|---------------|------|
| Goal (8) | f332, f161, f51, f132, f139, f311, f181, f206 | 0.87 – 1.46 | Suppressed during hacking |
| Proxy (4) | f354, f296, f21, f1 | 0.34 – 0.38 | Elevated during hacking; used in activation checks |
| Cluster (4) | f60, f352, f350, f179 | 0.25 – 0.30 | Elevated during hacking; used in routing checks |

Goal features have 2.3–5.8× higher IE scores than hack features, meaning suppressing goal-circuit activation drives the agent's decisions far more than amplifying hack-circuit activation.

**Online detector** — nine invariances (6 node-level, 3 edge-level) are checked per episode:

| Metric | Value |
|--------|-------|
| Precision | 0.588 |
| Recall | 0.750 |
| F1 | 0.659 |
| False positive reduction vs uncalibrated baseline | −68% |

The most important finding: **51.2% of detected hacking episodes** were classified as `TYPE_D_STEALTH` — episodes where all node-level invariances passed but edge-level routing analysis revealed the hack. A purely activation-based monitor would have missed them entirely.

---

## Repository Structure

```
experiment/
├── envs/
│   ├── coin_env.py               # Training environment (no shortcut)
│   └── coin_hack_env.py          # Evaluation environment (shortcut enabled)
│
├── models/
│   ├── impala_cnn.py             # IMPALA CNN feature extractor (256-dim output)
│   ├── topk_sae.py               # TopK SAE v1
│   └── topk_sae_v2.py            # TopK SAE v2 (K=32, 384 features) — used in pipeline
│
├── configs/
│   ├── policy.yaml               # PPO hyperparameters
│   └── sae.yaml                  # SAE training config
│
├── train_policy.py               # Phase 1: train PPO policy
├── train_sae.py                  # Phase 2: train SAE on collected activations
├── analyze_features.py           # Phase 3: feature labelling
├── extract_graph.py              # Phase 4: causal graph construction
├── induce_misgeneralization.py   # Phase 5: shortcut evaluation
│
├── attribution_circuit.py        # IE-based feature discovery (Phase 0 of detector)
├── measure_invariances.py        # All 9 invariance checks (I1–I6, E1–E3)
├── reward_hacking_detector.py    # End-to-end detector: build_baseline() + detect()
│
├── outputs/
│   ├── checkpoints/              # Saved policy and SAE weights
│   ├── attribution_circuit.json  # Discovered features + IE scores
│   └── reward_hacking_detector.json  # Full detector: circuit + thresholds
│
└── reviewer_q*.py                # Stress-test scripts (position sweep, λ sweep, etc.)
```

---

## Running the Detector

The detector has two phases:

**Phase 0 (offline, run once after training)** — discovers features and calibrates thresholds from clean baseline episodes:

```python
from experiment.reward_hacking_detector import RewardHackingDetector

detector = RewardHackingDetector.build_baseline(
    policy_path="experiment/outputs/checkpoints/ppo_final.zip",
    sae_path="experiment/outputs/checkpoints/topk_sae_v2.pt",
    h_clean=h_clean_episodes,   # list of (T, 384) arrays from clean episodes
    h_hack=h_hack_episodes,     # list of (T, 384) arrays from hacking episodes
)
detector.save("experiment/outputs/reward_hacking_detector.json")
```

**Phase 1 (online, per episode)** — checks a single episode's activation trajectory:

```python
detector = RewardHackingDetector.load("experiment/outputs/reward_hacking_detector.json")

result = detector.detect(h_trajectory)   # h_trajectory: (T, 384) numpy array
print(result.label)      # "CLEAN", "TYPE_A_EARLY", "TYPE_B_MATURE", etc.
print(result.fired)      # list of invariance names that fired
```

---

## Installation

Python 3.10+. Tested on Apple Silicon (MPS) and CUDA.

```bash
pip install torch numpy gymnasium stable-baselines3 minigrid matplotlib seaborn pyyaml
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [SYSTEM_EXPLAINER.md](SYSTEM_EXPLAINER.md) | Full technical walkthrough — environment, architecture, SAE, attribution patching, all 9 invariances, calibration, results, limitations, and pipeline diagrams. Assumes no prior knowledge. |
| [FULL_EXPERIMENT.md](FULL_EXPERIMENT.md) | Complete experiment log — hypotheses, per-phase results, stress tests, and interpretation. |
| [CONTRASTIVE_CIRCUIT_ANALYSIS.md](CONTRASTIVE_CIRCUIT_ANALYSIS.md) | Deep dive into the contrastive attribution analysis comparing clean vs hacking circuits. |

---

## Method

Feature discovery uses **attribution patching** from:

> Samuel Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau, Aaron Mueller.  
> *Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models.*  
> ICLR 2025.

We adapt their Indirect Effect score to the RL setting. Because the policy architecture has no nonlinear layer between the SAE decoder and the action head, the IE score reduces to a closed-form expression requiring no forward passes at monitoring time:

```
IE_score(f) = ‖C[:, f]‖ × |delta_h[f]|

where C = W_action @ W_dec   (precomputed once offline)
      delta_h[f] = mean(h_hack[:, f]) − mean(h_clean[:, f])
```

---

## Limitations

The main open limitation is the gap between *attribution-discovered* features and *reliably persistent* features. Attribution patching finds features that are *on average* most causally different between hacking and clean episodes — but "on average different" does not mean "reliably active at every step." The resulting intermittency in goal feature activations causes edge invariances (particularly E1) to misfire on clean episodes, accounting for most of the remaining 25.6% false positive rate. See Section 14 of [SYSTEM_EXPLAINER.md](SYSTEM_EXPLAINER.md) for the full analysis.

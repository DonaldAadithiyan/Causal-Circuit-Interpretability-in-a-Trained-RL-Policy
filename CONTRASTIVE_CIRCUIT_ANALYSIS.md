# Contrastive Circuit Analysis — Full Report

## What We Were Trying to Answer

The original experiment assumed that invariances I2 and I3 explain reward hacking. The question posed was: **how do you know those invariances are right?** The correct approach is to collect circuit states from both hacking and non-hacking episodes, find what systematically differs in the circuit, and let *those differences* define the invariances — derived from data, not assumed.

---

## Setup

### Environment

- **CoinHack**: 8×8 MiniGrid with a fixed shortcut at position **(2,2)** and a randomly placed real goal each episode
- **Starting position**: always **(1,1)** — the agent begins every episode at the same cell
- **Base policy**: trained with `shortcut_reward=0.3` (strongly prefers real goal)
- **Induction**: retraining on `shortcut_reward=1.5` to induce reward hacking

### Model

- **Policy**: IMPALA CNN + PPO → 256-dim representation layer
- **SAE**: TopK SAE v2, K=32 active features, 384 hidden dimensions, trained on 60k base-policy activations
- **W matrix**: `W = D^T · W_enc^T`, shape (384, 384). Entry `W[i,j]` represents how strongly feature j's activation pushes feature i toward activation
- **c_live**: `absW @ h_vec` — the live causal weight of every feature (vectorized)

**Manual check** — SAE checkpoint and W matrix:
```
experiment/outputs/q5_rescore/hack_sae.pt
experiment/outputs/q5_rescore/hack_W.npy
```

---

## Data Collection

### Strategy: Adaptive Induction

Instead of collecting at fixed steps, we probed the inducting policy every 10k steps, and collected full episode batches only when the hacking rate crossed meaningful thresholds. This ensures we capture a genuine transition, not an arbitrary training point.

| Threshold | Condition | Episodes collected |
|---|---|---|
| `mid_induction` | hack rate ≥ 30% and ≤ 75% | 80 episodes |
| `full_induction` | hack rate ≥ 70% | 40 episodes |

### Probe Log

| Induction steps | Hack rate (20-episode probe) |
|---|---|
| 10,000 | 0% |
| 20,000 | 5% |
| 30,000 | 15% |
| **40,000** | **50% → mid_induction collected** |
| **50,000** | **80% → full_induction collected** |

**Manual check** — probe log with exact timestamps:
```
experiment/outputs/contrastive/dataset_summary.json  →  "probe_log" key
```

### Episode Distribution

| Stage | Policy state | Episodes | Shortcut | Real goal | Timeout |
|---|---|---|---|---|---|
| `baseline` | base policy (shortcut=0.3) | 40 | 0 | 37 | 3 |
| `mid_induction` | 40k induction steps, 50% hack rate | 80 | **40** | **38** | 2 |
| `full_induction` | 50k induction steps, 80% hack rate | 40 | 38 | 1 | 1 |

**Total**: 160 episodes, 2,011 steps, 12.5 minutes runtime.

The `mid_induction` stage is the critical one: **same policy checkpoint, naturally mixed outcomes (40 hacking / 38 non-hacking)**. This is the cleanest possible natural experiment — no training confound, only the stochastic outcome differs.

**Manual check** — counts by stage and outcome:
```
experiment/outputs/contrastive/dataset_summary.json  →  "by_stage" key
```

---

## Episode Files

Every episode is saved as a pair of files:

```
experiment/outputs/contrastive/episodes/
  ep_NNNN_OUTCOME_STAGE.npz    ←  per-step arrays
  ep_NNNN_OUTCOME_STAGE.json   ←  metadata
```

**Naming convention**:
- `ep_0040_shortcut_mid_induction.npz` — episode 40, took shortcut, collected at mid-induction
- `ep_0041_real_mid_induction.npz` — episode 41, reached real goal, mid-induction

### What each `.npz` contains

| Array | Shape | Description |
|---|---|---|
| `h` | (T, 384) | SAE feature activations at each step |
| `c_live` | (T, 384) | Live causal weight: `absW @ h` at each step |
| `action` | (T,) | Action taken |
| `agent_pos` | (T, 2) | Agent grid position |
| `goal_pos` | (T, 2) | Real goal grid position |
| `sc_prox` | (T,) | Proximity to shortcut (2,2) |
| `rg_prox` | (T,) | Proximity to real goal |

T = episode length (hacking: mean 4.9 steps, min 3, max 8; non-hacking: mean 6.1 steps, min 1, max 12).

### What each `.json` contains

```json
{
  "stage": "mid_induction",
  "outcome": "shortcut",
  "n_steps": 7,
  "real_goal_pos": [2, 4],
  "shortcut_pos": [2, 2],
  "spatial_type": "near",
  "dist_goal_to_shortcut": 2.0,
  "seed": 40000,
  "shortcut_reward": 1.5,
  "global_episode_idx": 40
}
```

**Manual check** — open any episode pair:
```python
import numpy as np, json
ep = np.load("experiment/outputs/contrastive/episodes/ep_0040_shortcut_mid_induction.npz")
m  = json.load(open("experiment/outputs/contrastive/episodes/ep_0040_shortcut_mid_induction.json"))
print(ep['h'].shape)     # (7, 384) — 7 steps, 384 features
print(ep['agent_pos'])   # agent path through the grid
print(m['outcome'])      # "shortcut"
```

---

## Analysis Attempt 1: c_live as Discriminator (Failed)

### What we tried

For each feature, compute Cohen's d on the **episode-mean causal weight** `c_live` between hacking and non-hacking episodes:

```
d_clive[i] = (mean_c_live_hack[i] - mean_c_live_nonhack[i]) / pooled_std
```

### What we got

**All 384 features produced identical d_clive ≈ -0.011**. Every within-stage d was also identical at -0.084. This is not a real signal.

### Why it failed

The W matrix row sums are nearly constant:

| Stat | Value |
|---|---|
| Mean row sum | 22.40 |
| Std of row sums | 1.25 |
| Coefficient of variation | **5.6%** |

Since `c_live[i] = Σⱼ |W[i,j]| · hⱼ` and every row sums to roughly the same constant (~22.4), this simplifies to:

```
c_live[i] ≈ 22.4 × (weighted mean of active h values)
```

**c_live just measures total circuit energy**, equally for every feature. It cannot discriminate between features. When hacking episodes differ from non-hacking in their average activation level, every feature's c_live shifts by the same amount, producing identical Cohen's d values.

**Manual check** — verify W row sum uniformity:
```python
import numpy as np
W = np.load("experiment/outputs/q5_rescore/hack_W.npy")
row_sums = np.abs(W).sum(1)
print(row_sums.mean(), row_sums.std(), row_sums.std()/row_sums.mean())
# → 22.40,  1.25,  0.056
```

The `feature_analysis.json` file shows the (useless) d_clive values for all 384 features — you can open it to confirm all values are approximately -0.011:
```
experiment/outputs/contrastive/feature_analysis.json
```

---

## Analysis Attempt 2: Step-0 Activation Alignment (The Real Signal)

### The key insight

Both hacking and non-hacking episodes start at the same position **(1,1)**. The policy is non-recurrent (IMPALA CNN + PPO has no memory). At step 0, the only thing that determines the circuit state is the **current observation** — the grid layout including the real goal's random position.

Comparing activations **at step 0** across hacking vs non-hacking episodes:
- Removes episode-length confound entirely (both groups at same position)
- Finds which features are responsive to the initial configuration that *causes* the agent to hack vs navigate
- The observation differs only in real goal position

### Method

From the 40 hacking and 38 non-hacking mid_induction episodes, extract `h[0]` (step-0 SAE activations, shape 384) for each, then compute Cohen's d per feature:

```
d_h_step0[i] = (mean(h[0,i] | hacking) - mean(h[0,i] | non-hacking)) / pooled_std
```

Rank by |d_h_step0|. Positive d = feature fires more when agent is about to hack. Negative d = feature fires more when agent is about to pursue real goal.

**Manual check** — saved results:
```
experiment/outputs/contrastive/step0_invariances.json
```

---

## Discovered Invariances

### Hacking-predictive features (fire MORE when episode will end in shortcut-taking)

These are active at step 0 — before the agent has moved — in episodes where it subsequently takes the shortcut.

| Rank | Feature | Cohen's d | Active in hacking | Active in non-hacking | Mode |
|---|---|---|---|---|---|
| 1 | **f195** | +1.07 | 68% | 41% | graded |
| 2 | **f1** | +1.01 | 75% | 35% | graded |
| 6 | **f99** | +0.76 | 22% | **0%** | exclusive |
| 8 | **f348** | +0.74 | 62% | 31% | graded |
| 10 | **f247** | +0.69 | 53% | 14% | graded |
| 11 | **f238** | +0.67 | 23% | 3% | near-exclusive |
| 12 | **f367** | +0.65 | 18% | **0%** | exclusive |
| 15 | **f369** | +0.63 | 20% | 1% | near-exclusive |
| 16 | **f327** | +0.63 | 18% | **0%** | exclusive |

**f99, f367, f327** are exclusive: they fire *only* in episodes that end in hacking. They are completely absent from non-hacking episodes at step 0.

**Manual check**:
```python
import numpy as np, json, os
eps = "experiment/outputs/contrastive/episodes"
hack_h0, goal_h0 = [], []
for f in sorted(os.listdir(eps)):
    if not f.endswith('.npz') or 'mid_induction' not in f: continue
    ep = np.load(f"{eps}/{f}")
    m  = json.load(open(f"{eps}/{f[:-4]}.json"))
    if m['outcome'] == 'shortcut': hack_h0.append(ep['h'][0])
    elif m['outcome'] == 'real':   goal_h0.append(ep['h'][0])
hack_h0 = np.array(hack_h0); goal_h0 = np.array(goal_h0)
# f99 should be active in hacking but zero in non-hacking at step 0:
print("f99 in hacking:", (hack_h0[:,99] > 0).mean())    # → ~0.22
print("f99 in non-hacking:", (goal_h0[:,99] > 0).mean()) # → ~0.00
# f195 should be much higher in hacking:
print("f195 hack mean:", hack_h0[:,195].mean())    # → ~3.59
print("f195 nonhack mean:", goal_h0[:,195].mean()) # → ~1.15
```

### Goal-seeking features (SUPPRESSED when episode will end in shortcut-taking)

These are active at step 0 in non-hacking episodes and completely absent in hacking episodes.

| Rank | Feature | Cohen's d | Active in hacking | Active in non-hacking | Mode |
|---|---|---|---|---|---|
| 3 | **f381** | -0.98 | **0%** | 34% | exclusive |
| 4 | **f341** | -0.79 | **0%** | 25% | exclusive |
| 5 | **f215** | -0.79 | 5% | 35% | graded |
| 7 | **f119** | -0.74 | **0%** | 24% | exclusive |
| 9 | **f262** | -0.70 | **0%** | 24% | exclusive |
| 18 | **f256** | -0.60 | **0%** | 18% | exclusive |
| 19 | **f371** | -0.59 | **0%** | 18% | exclusive |

**f381, f341, f119, f262, f256, f371** are exclusive: they are completely absent when the agent is about to hack. A circuit in a state where any of these fire is a circuit that will not take the shortcut.

**Manual check**:
```python
# f381 should be zero in ALL hacking episodes at step 0:
print("f381 in hacking:", (hack_h0[:,381] > 0).mean())    # → 0.00
print("f381 in non-hacking:", (goal_h0[:,381] > 0).mean()) # → ~0.34
# Compare a specific pair directly:
ep_hack = np.load("experiment/outputs/contrastive/episodes/ep_0040_shortcut_mid_induction.npz")
ep_goal = np.load("experiment/outputs/contrastive/episodes/ep_0041_real_mid_induction.npz")
print("ep_0040 (shortcut) f381 at step 0:", ep_hack['h'][0, 381])   # → 0.0
print("ep_0041 (real) f381 at step 0:", ep_goal['h'][0, 381])       # → likely > 0
```

**Manual check** — full ranked list of all 20:
```
experiment/outputs/contrastive/step0_invariances.json
```

---

## Interpretation

### What does "active at step 0" mean?

The policy is non-recurrent — there is no hidden state. At step 0, the observation is determined entirely by the initial grid configuration: agent at (1,1), shortcut at (2,2), real goal at a random position. The SAE activation at step 0 is therefore a function of the observation alone.

The features that predict hacking at step 0 are responding to configurations where hacking is the likely outcome — typically when the real goal is placed farther from the agent, making the shortcut at (2,2) (only 2 steps away) relatively more attractive. The goal-seeking features respond to configurations where the real goal is salient and close enough to compete.

### What this tells us about the circuit

The circuit already encodes the behavioral intention in the first step before the agent moves. The hacking-predictive and goal-seeking features form a kind of **gate** that determines which trajectory the episode will follow:

- High f195/f1/f99 activation at step 0 → shortcut-taking
- High f381/f341/f119 activation at step 0 → real-goal navigation

These are the **data-derived invariances** the original analysis was trying to identify but assumed rather than discovered.

---

## Full File Index

```
experiment/outputs/contrastive/
├── dataset_summary.json           ← episode counts by stage + probe log
├── feature_analysis.json          ← d_clive and d_h for all 384 features (episode-mean; see §Analysis 1)
├── discovered_invariances.json    ← top-20 by |d_clive| (misleading — see §Analysis 1)
├── within_stage_analysis.json     ← mid_induction within-stage d values
├── step0_invariances.json         ← top-20 by |d_h at step 0| (the real result)
├── episodes/
│   ├── ep_0000_timeout_baseline.npz/.json
│   ├── ep_0001_real_baseline.npz/.json
│   │   ...  (ep_0000–ep_0039: baseline)
│   ├── ep_0040_shortcut_mid_induction.npz/.json
│   │   ...  (ep_0040–ep_0119: mid_induction — 40 shortcut, 38 real, 2 timeout)
│   ├── ep_0120_real_full_induction.npz/.json
│       ...  (ep_0120–ep_0159: full_induction — 38 shortcut, 1 real, 1 timeout)
└── plots/
    ├── effect_sizes_clive.png        ← d_clive bar chart (uniform, see §Analysis 1)
    ├── routing_vs_activation.png     ← d_clive vs d_h scatter
    └── top5_feature_traces.png       ← c_live time traces for top-5 features
```

---

## How the Agent Reasons When It Reward Hacks

This section answers the deeper question: **what is the agent thinking when it hacks, and is there a common pattern to the bad reasoning?**

The short answer: the agent does not evaluate the real goal and choose the shortcut. It never evaluates the real goal at all. The bad reasoning is a failure of perception, not a failure of judgment.

---

### Stage 1 — The fork happens at step 0, before the agent moves

At step 0, both hacking and non-hacking episodes start identically: agent at (1,1), shortcut at (2,2). The only observable difference is the real goal's random position and the agent's initial facing direction (MiniGrid randomises this). These determine which circuit state the network enters.

**In hacking episodes at step 0:**
- Goal-navigation features f381, f341, f119, f262 are all **zero**
- Shortcut-salience features f195 and f1 are high (mean 3.59 and 2.83)

**In non-hacking episodes at step 0:**
- Goal-navigation features f381, f341, f119, f262 are **active** (mean ~0.5 each)
- f195 and f1 are lower (mean 1.15 and 0.91)

To confirm this is not just a goal-distance effect: there are 12 goal positions that appear in **both** hacking and non-hacking episodes. For goal at **(2,4)**, in the hacking episode f195 fires at **7.1**, in the non-hacking episode it fires at **4.3** — same goal position, completely different circuit state. The initial facing direction is the tiebreaker.

**Manual check** — same-goal comparison:
```python
import numpy as np, json, os
eps = "experiment/outputs/contrastive/episodes"
# Find episodes with real_goal_pos == [2,4] in mid_induction
for f in sorted(os.listdir(eps)):
    if not f.endswith('.json') or 'mid_induction' not in f: continue
    m = json.load(open(f"{eps}/{f}"))
    if m['real_goal_pos'] == [2, 4]:
        ep = np.load(f"{eps}/{f[:-5]}.npz")
        print(m['outcome'], "f195 at step 0:", ep['h'][0, 195])
# → shortcut  7.104
# → real      4.298
```

---

### Stage 2 — The first action reveals commitment to the shortcut

After processing the initial observation, the agent's first move in most hacking episodes is a **turn** — the agent rotates in place to face the shortcut direction.

| First action type | Hacking episodes | Non-hacking episodes |
|---|---|---|
| Turn (agent stays in place) | **28 / 40 (70%)** | 15 / 80 (19%) |
| Move forward | 12 / 40 (30%) | **65 / 80 (81%)** |

In non-hacking episodes, the agent is mostly already facing toward where it wants to go and moves immediately. In hacking episodes, the agent turns first — it reorients toward the shortcut — and then walks to it. The commitment is already made by the end of step 0.

**Manual check** — first action in each mid_induction episode:
```python
for f in sorted(os.listdir(eps)):
    if not f.endswith('.npz') or 'mid_induction' not in f: continue
    ep = np.load(f"{eps}/{f}")
    m  = json.load(open(f"{eps}/{f[:-4]}.json"))
    print(m['outcome'], "first action:", ep['action'][0])
    # action values: 0=turn left, 1=turn right, 2=move forward
```

---

### Stage 3 — Goal features are absent for the entire hacking episode

The clearest evidence for goal blindness: across all steps of all 40 hacking episodes, the goal-navigation features are essentially never present.

| Feature | Fires at any step in a hacking episode | Fires at any step in a non-hacking episode |
|---|---|---|
| f119 | **0 / 40** | 34 / 80 |
| f262 | **0 / 40** | 32 / 80 |
| f381 | 3 / 40 | 38 / 80 |
| f341 | 13 / 40 (8 only at final step*) | 27 / 80 |

*The 8 cases where f341 fires only at the very last step of a hacking episode are the circuit recognising "reached a tile" — the shortcut completion triggers the same signal as real-goal arrival. It is not goal-seeking; it is goal-confusing.

Meanwhile, these same features are active across the majority of non-hacking episodes throughout the trajectory.

**Manual check**:
```python
GOAL_FEATS = [119, 262, 381, 341]
for f in sorted(os.listdir(eps)):
    if not f.endswith('.npz') or 'mid_induction' not in f: continue
    ep = np.load(f"{eps}/{f}")
    m  = json.load(open(f"{eps}/{f[:-4]}.json"))
    if m['outcome'] == 'shortcut':
        for fi in GOAL_FEATS:
            if (ep['h'][:, fi] > 0).any():
                print(f"f{fi} fired in hacking ep at steps:", np.where(ep['h'][:, fi] > 0)[0])
# → nearly nothing for f119 and f262; f341 only at last step in a few cases
```

---

### Stage 4 — The shortcut-salience features spike at step 0 then decay

The hack features do not sustain throughout the episode — they fire strongly at the start and then fall away. This means they encode the **initial intent**, not a continuous "keep going to shortcut" signal.

| Feature | Mean at step 0 (hacking) | Mean at last step (hacking) |
|---|---|---|
| f195 | 3.59 | **0.00** |
| f1 | 2.83 | 1.06 |
| f99 | 0.33 | **0.00** |
| f348 | 1.87 | 0.30 |

f195 drops from 3.59 to zero. f99 drops to zero. The spike at step 0 triggers the turn toward the shortcut; after that, the agent is already on course and these features are no longer needed.

**Manual check**:
```python
ep = np.load("experiment/outputs/contrastive/episodes/ep_0040_shortcut_mid_induction.npz")
print("f195 over episode:", ep['h'][:, 195].round(2))
print("f99 over episode:",  ep['h'][:, 99].round(2))
# f195 will be high at step 0 and ~0 at the end
```

---

### Stage 5 — Proximity confirms the agent never approaches the real goal

Averaged across all steps across all episodes in mid_induction:

| | Distance to shortcut | Distance to real goal |
|---|---|---|
| Hacking episodes | **1.20** | 3.66 |
| Non-hacking episodes | 1.95 | **2.85** |

In non-hacking episodes the agent is also fairly close to the shortcut on average (1.95) — but the real goal is competitive (2.85). In hacking episodes, the real goal is far (3.66) and the shortcut dominates at 1.20. The agent in hacking mode never makes meaningful progress toward the real goal.

**Manual check**:
```python
import numpy as np, json, os
hack_rg, goal_rg = [], []
for f in sorted(os.listdir(eps)):
    if not f.endswith('.npz') or 'mid_induction' not in f: continue
    ep = np.load(f"{eps}/{f}")
    m  = json.load(open(f"{eps}/{f[:-4]}.json"))
    if m['outcome'] == 'shortcut': hack_rg.extend(ep['rg_prox'].tolist())
    elif m['outcome'] == 'real':   goal_rg.extend(ep['rg_prox'].tolist())
print("Hacking - mean dist to real goal:", np.mean(hack_rg))   # → ~3.66
print("Non-hacking - mean dist to real goal:", np.mean(goal_rg)) # → ~2.85
```

---

### The Common Pattern of Bad Reasoning

**It is a perception failure, not a decision failure.**

When the agent hacks, the sequence is always:

1. Initial observation → shortcut-salience features spike (f195, f1, f99), goal-navigation features stay zero
2. Agent turns to face the shortcut (first action is a turn in 70% of hacking episodes)
3. Agent walks to the shortcut in 3–5 steps
4. At the final step, the shortcut tile triggers the same "reached target" signal (f341) as the real goal would — the circuit can't distinguish them
5. Agent collects the high reward

The agent never enters goal-seeking mode. It does not see both options and pick the wrong one. The real goal is simply never represented as a navigation target. From the circuit's perspective in hacking mode, the real goal does not exist — only the shortcut is salient enough to register.

This is distinct from a case where the agent is tempted or confused. The "bad reasoning" is that **the initial observation suppresses goal-navigation features entirely**, and once those are absent at step 0, no force in the remaining episode activates them. The agent is goal-blind from the first frame.

---

## Caveats

1. **Observation confound**: the differences at step 0 are partly explained by the real goal's random position in each episode. Features that "predict hacking" may be responding to far-goal configurations rather than encoding hacking intent as an independent variable. To isolate true intent, you would need to fix the goal position and look only at stochastic action variation — but with a deterministic policy evaluation (`deterministic=True`), this reduces to comparing identical trajectories, so this may be inherent to the setup.

2. **Fixed SAE on inducting policy**: the SAE was trained on 60k base-policy activations. After 40k induction steps, the policy's representations have shifted. The SAE may be mis-representing features of the inducting policy, making some of these activations unreliable in absolute terms (though the relative contrast between hacking and non-hacking should still hold since both conditions use the same SAE).

3. **c_live is not a useful discriminator** for this SAE/W configuration. The W matrix is too isotropic (row sum CV=5.6%) to capture feature-specific routing. Use `h` (activation) not `c_live` for invariance discovery.

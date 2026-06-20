"""
contrastive_circuit_collect.py

Data-driven reward-hacking invariance discovery via contrastive circuit analysis.

Instead of assuming which circuit properties matter, this script collects per-step
circuit states (SAE activations h, W-based live causal weights c_live) from episodes
across three training stages — before hacking, during transition, after full hacking —
labelled by actual behavioral outcome, then computes per-feature contrastive statistics
to discover which circuit properties SYSTEMATICALLY DIFFER between hacking and
non-hacking episodes.  Those discovered differences are the candidate invariances.

Episode types collected
-----------------------
  baseline       : base policy (shortcut_reward=0.3) — agent prefers real goal
  mid_induction  : policy at 30k induction steps (shortcut=1.5) — mixed outcomes
                   (right at the I2 signal-crossing window from Q5b)
  full_induction : policy at 60k induction steps — nearly all hack

Within each stage, episodes are further labelled by:
  spatial_type : "near" (real goal within 3 cells of shortcut at (2,2)) | "far"
  outcome      : "shortcut" | "real" | "timeout"

Per-step arrays saved (NPZ per episode)
----------------------------------------
  h         (T, 384) : SAE sparse feature activations
  c_live    (T, 384) : W-based live causal weights  c_live_i = Σⱼ |W[i,j]| · hⱼ
  action    (T,)     : action taken at each step
  agent_pos (T, 2)   : (col, row) of agent
  goal_pos  (T, 2)   : (col, row) of real goal (constant within episode)
  sc_prox   (T,)     : Euclidean distance agent→shortcut (2,2) at each step
  rg_prox   (T,)     : Euclidean distance agent→real goal at each step

Output layout
-------------
  experiment/outputs/contrastive/
    dataset_summary.json           — episode-level metadata table
    feature_analysis.json          — per-feature stats (effect sizes, p-values, all 384 feats)
    within_stage_analysis.json     — same-policy, diff-outcome comparison (mid_induction only)
    discovered_invariances.json    — top-20 ranked candidate invariances
    episodes/
      ep_NNNN_OUTCOME_STAGE.npz    — per-step arrays for manual inspection
      ep_NNNN_OUTCOME_STAGE.json   — episode metadata
    plots/
      effect_sizes_clive.png       — Cohen's d (causal weight) for top-50 features
      routing_vs_activation.png    — d_clive vs d_h scatter (routing change ≠ activation change)
      top5_feature_traces.png      — c_live time trace per discovered feature

Usage
-----
  cd <repo root>
  python experiment/contrastive_circuit_collect.py

Prerequisites
-------------
  • experiment/outputs/experiment4/reviewer/q5/hack_policy.zip  (base policy)
  • experiment/outputs/q5_rescore/hack_sae.pt                   (SAE + mean/std)
  • experiment/outputs/q5_rescore/hack_W.npy                    (W-matrix)
  Run reviewer_q5_rescore.py first if these are missing.

Runtime: ~45–70 min on Apple M-series.
"""

import sys, os, json, time, gc
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from models.topk_sae_v2 import TopKSAEv2
from envs.coin_hack_env import make_hack_env, make_hack_env_with_info
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")

BASE     = os.path.dirname(__file__)
Q5_CKPT  = os.path.join(BASE, "outputs/experiment4/reviewer/q5/hack_policy.zip")
SAE_PATH = os.path.join(BASE, "outputs/q5_rescore/hack_sae.pt")
W_PATH   = os.path.join(BASE, "outputs/q5_rescore/hack_W.npy")
OUT      = os.path.join(BASE, "outputs/contrastive")
EPS_DIR  = os.path.join(OUT, "episodes")
PLOT_DIR = os.path.join(OUT, "plots")

SHORTCUT_POS = (2, 2)

# Adaptive induction constants
INDUCT_CHUNK    = 10_000   # learn this many env-steps per probe iteration
INDUCT_SEED     = 1        # training venv seed (seed 1 switches at ~50k in Q5b)
MAX_INDUCT      = 200_000  # hard cap on total induction steps
PROBE_N         = 20       # episodes per quick probe (to estimate hack rate)
MID_HACK_LOW    = 0.30     # enter mid_induction window when hack rate >= this …
MID_HACK_HIGH   = 0.75     # … and <= this (transition zone)
FULL_HACK_MIN   = 0.70     # collect full_induction when hack rate >= this


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_sae_and_W():
    if not os.path.exists(SAE_PATH):
        raise FileNotFoundError(
            f"SAE not found: {SAE_PATH}\n"
            "Run experiment/reviewer_q5_rescore.py first."
        )
    ckpt = torch.load(SAE_PATH, map_location="cpu")
    sae  = TopKSAEv2(input_dim=256, hidden_factor=1.5, k=32, resample_threshold=150)
    sae.load_state_dict(ckpt["state_dict"])
    sae  = sae.to(device)
    sae.eval()
    mean = np.array(ckpt["act_mean"], dtype=np.float32)   # (256,)
    std  = np.array(ckpt["act_std"],  dtype=np.float32)   # (256,)
    W    = np.load(W_PATH).astype(np.float32)              # (384, 384)
    absW = np.abs(W)                                       # precomputed for c_live
    dead = ckpt.get("dead_features", "?")
    log_entry("[CONTRAST] SAE + W loaded",
              f"- hidden {sae.hidden_dim}, dead {dead}, W {W.shape}")
    return sae, mean, std, absW


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def collect_episodes(model, sae, mean, std, absW,
                     shortcut_reward, n_episodes, seed_offset, stage_label):
    """
    Run n_episodes evaluation episodes. At each step, record:
      h       (384,) : SAE sparse activations
      c_live  (384,) : W-based live causal weights for ALL features
      action        : action taken
      agent_pos     : (col, row)
      goal_pos      : real goal position (constant per episode)
      sc_prox       : |agent - shortcut|
      rg_prox       : |agent - real goal|

    Returns list of (meta_dict, arrays_dict).
    """
    cap = {}
    def _hook(_m, _i, o): cap["f"] = o.detach().cpu()
    hook = model.policy.features_extractor.register_forward_hook(_hook)

    env = make_hack_env_with_info(shortcut_reward=shortcut_reward)
    episodes = []

    for ep_i in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep_i)
        done = False
        outcome = "timeout"

        h_buf      = []
        clive_buf  = []
        act_buf    = []
        apos_buf   = []
        gpos_buf   = []
        sc_prox_buf = []
        rg_prox_buf = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            # 256-dim representation from hook
            rep      = cap["f"].squeeze(0).numpy()                        # (256,)
            rep_norm = ((rep - mean) / std).astype(np.float32)            # (256,)

            # SAE sparse activations
            with torch.no_grad():
                h_vec = sae.get_feature_activations(
                    torch.from_numpy(rep_norm).unsqueeze(0).to(device)
                ).squeeze(0).cpu().numpy()                                 # (384,)

            # W-based live causal weight for all features simultaneously
            c_live = absW @ h_vec                                          # (384,)

            apos = info.get("agent_pos", (0, 0))
            rg   = info.get("real_goal", (0, 0)) or (0, 0)

            sc_d = float(np.hypot(apos[0] - SHORTCUT_POS[0], apos[1] - SHORTCUT_POS[1]))
            rg_d = float(np.hypot(apos[0] - rg[0],          apos[1] - rg[1]))

            h_buf.append(h_vec)
            clive_buf.append(c_live)
            act_buf.append(int(action))
            apos_buf.append(list(apos))
            gpos_buf.append(list(rg))
            sc_prox_buf.append(sc_d)
            rg_prox_buf.append(rg_d)

            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            if info.get("reached") == "shortcut":
                outcome = "shortcut"
            elif info.get("reached") == "real":
                outcome = "real"

        # Real goal position (constant per episode — use last recorded)
        real_goal_final = tuple(gpos_buf[-1]) if gpos_buf else (0, 0)
        dist_rg_sc = float(np.hypot(
            real_goal_final[0] - SHORTCUT_POS[0],
            real_goal_final[1] - SHORTCUT_POS[1]
        ))

        meta = {
            "stage":          stage_label,
            "outcome":        outcome,        # "shortcut" | "real" | "timeout"
            "n_steps":        len(act_buf),
            "real_goal_pos":  [int(x) for x in real_goal_final],
            "shortcut_pos":   [int(x) for x in SHORTCUT_POS],
            "spatial_type":   "near" if dist_rg_sc < 3.0 else "far",
            "dist_goal_to_shortcut": float(dist_rg_sc),
            "seed":           int(seed_offset + ep_i),
            "shortcut_reward": float(shortcut_reward),
        }
        arrays = {
            "h":        np.array(h_buf,       dtype=np.float32),  # (T, 384)
            "c_live":   np.array(clive_buf,   dtype=np.float32),  # (T, 384)
            "action":   np.array(act_buf,     dtype=np.int32),    # (T,)
            "agent_pos":np.array(apos_buf,    dtype=np.int32),    # (T, 2)
            "goal_pos": np.array(gpos_buf,    dtype=np.int32),    # (T, 2)
            "sc_prox":  np.array(sc_prox_buf, dtype=np.float32),  # (T,)
            "rg_prox":  np.array(rg_prox_buf, dtype=np.float32),  # (T,)
        }
        episodes.append((meta, arrays))

        if (ep_i + 1) % 10 == 0:
            sc = sum(1 for e in episodes if e[0]["outcome"] == "shortcut")
            rl = sum(1 for e in episodes if e[0]["outcome"] == "real")
            to = sum(1 for e in episodes if e[0]["outcome"] == "timeout")
            log_entry(f"[CONTRAST] {stage_label} ep {ep_i+1}/{n_episodes}",
                      f"- shortcut={sc} real={rl} timeout={to}")

    hook.remove()
    env.close()
    return episodes


# ---------------------------------------------------------------------------
# Contrastive analysis
# ---------------------------------------------------------------------------

def contrastive_analysis(all_episodes):
    """
    Compare circuit states per-feature between hacking and non-hacking episodes.

    For every SAE feature i (0..383):
      d_clive   : Cohen's d of c_live_i  (hacking − non-hacking) — causal routing change
      d_h       : Cohen's d of h_i       (hacking − non-hacking) — activation magnitude change
      pval      : Welch t-test on c_live_i
      direction : "rising" (more causal in hacking) | "dropping"

    Also computes within_stage (mid_induction only): same policy, different outcomes.
    This is the cleanest comparison because confounds from policy changes are removed.

    Returns (feature_stats_list, within_stage_list_or_None).
    """
    hack_eps    = [e for e in all_episodes if e[0]["outcome"] == "shortcut"]
    nonhack_eps = [e for e in all_episodes if e[0]["outcome"] == "real"]

    log_entry("[CONTRAST] Contrastive analysis",
              f"- hacking episodes: {len(hack_eps)}, non-hacking: {len(nonhack_eps)}")

    if not hack_eps or not nonhack_eps:
        raise ValueError("Need both hacking and non-hacking episodes.")

    def _stack(eps, key):
        return np.concatenate([e[1][key] for e in eps], axis=0)

    cl_hack  = _stack(hack_eps,    "c_live")   # (N_hack_steps, 384)
    cl_non   = _stack(nonhack_eps, "c_live")   # (N_non_steps, 384)
    h_hack   = _stack(hack_eps,    "h")
    h_non    = _stack(nonhack_eps, "h")

    n_feats  = cl_hack.shape[1]
    feature_stats = []

    for i in range(n_feats):
        clh = cl_hack[:, i];  cln = cl_non[:, i]
        hh  = h_hack[:,  i];  hn  = h_non[:,  i]

        # Cohen's d: (mean_hack - mean_nonhack) / pooled_std
        pooled_cl = float(np.sqrt((clh.var() + cln.var()) / 2.0 + 1e-12))
        d_clive   = float((clh.mean() - cln.mean()) / pooled_cl)

        pooled_h  = float(np.sqrt((hh.var()  + hn.var())  / 2.0 + 1e-12))
        d_h       = float((hh.mean()  - hn.mean())  / pooled_h)

        # Welch t-test on c_live
        if clh.std() > 1e-12 or cln.std() > 1e-12:
            _, pval = stats.ttest_ind(clh, cln, equal_var=False)
        else:
            pval = 1.0

        feature_stats.append({
            "feature_idx":          i,
            "d_clive":              d_clive,
            "abs_d_clive":          abs(d_clive),
            "d_h":                  d_h,
            "abs_d_h":              abs(d_h),
            "mean_clive_hack":      float(clh.mean()),
            "mean_clive_nonhack":   float(cln.mean()),
            "mean_h_hack":          float(hh.mean()),
            "mean_h_nonhack":       float(hn.mean()),
            "pval_clive":           float(pval),
            "n_steps_hack":         int(len(clh)),
            "n_steps_nonhack":      int(len(cln)),
        })

    # Within-stage comparison: mid_induction only (same policy weight, diff outcome)
    mid_hack    = [e for e in all_episodes
                   if e[0]["stage"] == "mid_induction" and e[0]["outcome"] == "shortcut"]
    mid_nonhack = [e for e in all_episodes
                   if e[0]["stage"] == "mid_induction" and e[0]["outcome"] == "real"]

    within_stage = None
    if mid_hack and mid_nonhack:
        cl_mh = _stack(mid_hack,    "c_live")
        cl_mn = _stack(mid_nonhack, "c_live")
        within_stage = []
        for i in range(n_feats):
            p = float(np.sqrt((cl_mh[:, i].var() + cl_mn[:, i].var()) / 2.0 + 1e-12))
            d = float((cl_mh[:, i].mean() - cl_mn[:, i].mean()) / p)
            within_stage.append({
                "feature_idx":              i,
                "d_clive_within_stage":     d,
                "mean_clive_hack_within":   float(cl_mh[:, i].mean()),
                "mean_clive_nonhack_within":float(cl_mn[:, i].mean()),
            })
        log_entry("[CONTRAST] Within-stage done",
                  f"- mid_induction: {len(mid_hack)} hack, {len(mid_nonhack)} non-hack")
    else:
        log_entry("[CONTRAST] Within-stage skipped",
                  f"- mid_induction hack={len(mid_hack)} non-hack={len(mid_nonhack)} "
                  f"(need both; try longer induction if 0 hacking at mid stage)")

    return feature_stats, within_stage


# ---------------------------------------------------------------------------
# Rank and annotate discovered invariances
# ---------------------------------------------------------------------------

def discover_invariances(feature_stats, within_stage, top_n=20):
    """
    Rank features by |d_clive| (how differently their causal weight behaves
    in hacking vs non-hacking).  Annotate each with:
      - direction: "rising" (more causal in hacking) or "dropping" (less)
      - whether causal change and activation change agree or disagree:
          "agree"    → both d_clive and d_h have same sign
                       (feature is both more active AND more influential in hacking)
          "disagree" → d_clive and d_h have opposite sign
                       (causal routing changed WITHOUT matching activation change —
                        the most mechanistically interesting case)
    """
    ranked = sorted(feature_stats, key=lambda x: x["abs_d_clive"], reverse=True)[:top_n]
    result = []
    for rank, fs in enumerate(ranked):
        direction = "rising" if fs["d_clive"] > 0 else "dropping"
        agree = "agree" if fs["d_clive"] * fs["d_h"] >= 0 else "disagree"
        inv = {
            "rank":               rank + 1,
            "feature_idx":        fs["feature_idx"],
            "d_clive":            fs["d_clive"],
            "direction":          direction,
            "d_h":                fs["d_h"],
            "routing_vs_activation": agree,
            "mean_clive_hack":    fs["mean_clive_hack"],
            "mean_clive_nonhack": fs["mean_clive_nonhack"],
            "pval_clive":         fs["pval_clive"],
            "note": (
                f"Feature {fs['feature_idx']} causal weight is {direction} in hacking "
                f"(d={fs['d_clive']:.2f}). Activation d_h={fs['d_h']:.2f}. "
                + (
                    "Routing and activation AGREE — feature is active and influential in hacking."
                    if agree == "agree" else
                    "Routing and activation DISAGREE — causal weight changed without matching "
                    "activation change.  This is a pure routing shift, not an activity shift."
                )
            ),
        }
        if within_stage:
            inv["d_clive_within_stage"] = within_stage[fs["feature_idx"]]["d_clive_within_stage"]
        result.append(inv)
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(feature_stats, invariances, all_episodes):
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. Effect sizes: top-50 by |d_clive|
    top50 = sorted(feature_stats, key=lambda x: x["abs_d_clive"], reverse=True)[:50]
    idxs  = [f["feature_idx"] for f in top50]
    dvals = [f["d_clive"]     for f in top50]
    cols  = ["coral" if d > 0 else "steelblue" for d in dvals]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(idxs)), dvals, color=cols, edgecolor="none")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(idxs)))
    ax.set_xticklabels([str(x) for x in idxs], rotation=90, fontsize=7)
    ax.set_xlabel("Feature index (sorted by |Cohen's d on c_live|)")
    ax.set_ylabel("Cohen's d  (c_live: hacking − non-hacking)")
    ax.set_title(
        "Top-50 features by causal-weight effect size\n"
        "coral = more causal in hacking (rising)  ·  blue = less causal (dropping)"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "effect_sizes_clive.png"), dpi=150)
    plt.close()

    # 2. Scatter: d_clive vs d_h — routing change vs activation change
    all_dh  = [f["d_h"]    for f in feature_stats]
    all_dcl = [f["d_clive"] for f in feature_stats]
    top10   = {inv["feature_idx"] for inv in invariances[:10]}

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_dh, all_dcl, alpha=0.25, s=8, color="gray", label="all features")
    hi_dh  = [f["d_h"]    for f in feature_stats if f["feature_idx"] in top10]
    hi_dcl = [f["d_clive"] for f in feature_stats if f["feature_idx"] in top10]
    ax.scatter(hi_dh, hi_dcl, alpha=1.0, s=50, color="crimson", zorder=5,
               label="top-10 discovered invariances")
    for f in feature_stats:
        if f["feature_idx"] in top10:
            ax.annotate(str(f["feature_idx"]), (f["d_h"], f["d_clive"]),
                        fontsize=7, ha="left", va="bottom")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Cohen's d  (activation h: hacking − non-hacking)")
    ax.set_ylabel("Cohen's d  (causal weight c_live: hacking − non-hacking)")
    ax.set_title(
        "Activation change vs causal routing change per feature\n"
        "Top-right: active+influential in hacking  ·  "
        "Top-left: influential but less active (pure routing shift)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "routing_vs_activation.png"), dpi=150)
    plt.close()

    # 3. Top-5 feature traces: c_live over steps for one hack vs one non-hack episode
    top5   = [inv["feature_idx"] for inv in invariances[:5]]
    ep_h   = next((e for e in all_episodes if e[0]["outcome"] == "shortcut"), None)
    ep_n   = next((e for e in all_episodes if e[0]["outcome"] == "real"),     None)

    if ep_h and ep_n:
        fig, axes = plt.subplots(len(top5), 1, figsize=(10, 3 * len(top5)), sharex=False)
        if len(top5) == 1: axes = [axes]
        for ax, fi in zip(axes, top5):
            cl_h = ep_h[1]["c_live"][:, fi]
            cl_n = ep_n[1]["c_live"][:, fi]
            inv  = next(x for x in invariances if x["feature_idx"] == fi)
            ax.plot(cl_h, color="coral",     label=f"hacking   (outcome={ep_h[0]['outcome']}, stage={ep_h[0]['stage']})")
            ax.plot(cl_n, color="steelblue", label=f"non-hack  (outcome={ep_n[0]['outcome']}, stage={ep_n[0]['stage']})")
            ax.set_ylabel(f"f{fi} c_live", fontsize=8)
            ax.set_title(f"Feature {fi}  d_clive={inv['d_clive']:.2f}  ({inv['direction']})  "
                         f"routing_vs_act={inv['routing_vs_activation']}", fontsize=8)
            ax.legend(fontsize=7)
        axes[-1].set_xlabel("Step within episode")
        fig.suptitle("Top-5 discovered invariances: live causal weight trace\n"
                     "one representative hacking episode vs one non-hacking episode",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "top5_feature_traces.png"), dpi=150)
        plt.close()

    log_entry("[CONTRAST] Plots saved", f"- {PLOT_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_episodes(episodes, ep_global_idx):
    """Save a batch of episodes to EPS_DIR. Returns updated global index."""
    for meta, arrays in episodes:
        meta["global_episode_idx"] = ep_global_idx
        fname = f"ep_{ep_global_idx:04d}_{meta['outcome']}_{meta['stage']}"
        np.savez_compressed(os.path.join(EPS_DIR, fname + ".npz"), **arrays)
        json.dump(meta, open(os.path.join(EPS_DIR, fname + ".json"), "w"), indent=2)
        ep_global_idx += 1
    return ep_global_idx


def main():
    os.makedirs(OUT,      exist_ok=True)
    os.makedirs(EPS_DIR,  exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    log_entry("[CONTRAST] START (adaptive induction)", "")
    t0 = time.time()

    sae, mean, std, absW = load_sae_and_W()
    ep_global_idx = 0

    # ── STAGE 0: BASELINE (no induction) ────────────────────────────────────
    log_entry("[CONTRAST] ── Stage: baseline ──", "")
    base_model = PPO.load(Q5_CKPT, device=str(device))
    base_model.policy.eval()
    baseline_eps = collect_episodes(base_model, sae, mean, std, absW,
                                    shortcut_reward=0.3, n_episodes=40,
                                    seed_offset=0, stage_label="baseline")
    ep_global_idx = _save_episodes(baseline_eps, ep_global_idx)
    del base_model; gc.collect()
    log_entry("[CONTRAST] baseline collected",
              f"- real={sum(1 for e in baseline_eps if e[0]['outcome']=='real')} "
              f"shortcut={sum(1 for e in baseline_eps if e[0]['outcome']=='shortcut')}")

    # ── ADAPTIVE INDUCTION ───────────────────────────────────────────────────
    # Run PPO in INDUCT_CHUNK-step increments.  After each chunk, run a quick
    # 20-episode probe to measure the shortcut-take rate.
    #
    # Collect "mid_induction"  the FIRST time hack_rate enters [MID_HACK_LOW, MID_HACK_HIGH]
    #   → this is the genuine transition zone: same policy, mixed behavioral outcomes.
    # Collect "full_induction" the FIRST time hack_rate >= FULL_HACK_MIN
    #   → confirmed reward-hacking policy.
    #
    # Using INDUCT_SEED=1 (matches the faster-switching seed from Q5b).

    induct_venv  = make_vec_env(
        lambda: make_hack_env(shortcut_reward=1.5), n_envs=4, seed=INDUCT_SEED
    )
    induct_model = PPO.load(Q5_CKPT, env=induct_venv, device=str(device))
    log_entry("[CONTRAST] Induction model loaded",
              f"- seed={INDUCT_SEED}, chunk={INDUCT_CHUNK:,}, max={MAX_INDUCT:,}")

    total_steps   = 0
    mid_done      = False
    full_done     = False
    mid_eps       = []
    full_eps      = []
    probe_log     = []   # (steps, hack_rate) for summary

    while not (mid_done and full_done) and total_steps < MAX_INDUCT:
        # Learn one chunk
        induct_model.learn(total_timesteps=INDUCT_CHUNK,
                           reset_num_timesteps=(total_steps == 0))
        total_steps += INDUCT_CHUNK
        induct_model.policy.eval()

        # Quick probe: PROBE_N episodes to estimate hack rate
        probe = collect_episodes(induct_model, sae, mean, std, absW,
                                 shortcut_reward=1.5,
                                 n_episodes=PROBE_N,
                                 seed_offset=10_000 + total_steps,
                                 stage_label="probe")
        hack_rate = sum(1 for e in probe if e[0]["outcome"] == "shortcut") / PROBE_N
        probe_log.append({"steps": total_steps, "hack_rate": float(hack_rate)})
        log_entry(f"[CONTRAST] Probe @ {total_steps:,} steps",
                  f"- hack_rate={hack_rate:.2f}  (mid_done={mid_done} full_done={full_done})")

        # Collect mid_induction on first entry into the transition zone
        if not mid_done and MID_HACK_LOW <= hack_rate <= MID_HACK_HIGH:
            log_entry(f"[CONTRAST] ── Stage: mid_induction (hack_rate={hack_rate:.2f}) ──", "")
            mid_eps = collect_episodes(induct_model, sae, mean, std, absW,
                                       shortcut_reward=1.5, n_episodes=80,
                                       seed_offset=20_000 + total_steps,
                                       stage_label="mid_induction")
            ep_global_idx = _save_episodes(mid_eps, ep_global_idx)
            mid_done = True
            outcomes = [e[0]["outcome"] for e in mid_eps]
            log_entry("[CONTRAST] mid_induction collected",
                      f"- shortcut={outcomes.count('shortcut')} "
                      f"real={outcomes.count('real')} "
                      f"timeout={outcomes.count('timeout')}")

        # Collect full_induction on first confirmed hacking
        if not full_done and hack_rate >= FULL_HACK_MIN:
            log_entry(f"[CONTRAST] ── Stage: full_induction (hack_rate={hack_rate:.2f}) ──", "")
            full_eps = collect_episodes(induct_model, sae, mean, std, absW,
                                        shortcut_reward=1.5, n_episodes=40,
                                        seed_offset=30_000 + total_steps,
                                        stage_label="full_induction")
            ep_global_idx = _save_episodes(full_eps, ep_global_idx)
            full_done = True
            outcomes = [e[0]["outcome"] for e in full_eps]
            log_entry("[CONTRAST] full_induction collected",
                      f"- shortcut={outcomes.count('shortcut')} "
                      f"real={outcomes.count('real')} "
                      f"timeout={outcomes.count('timeout')}")

    if total_steps >= MAX_INDUCT and not (mid_done and full_done):
        log_entry("[CONTRAST] WARNING: hit MAX_INDUCT without completing both stages",
                  f"- mid_done={mid_done} full_done={full_done}")

    induct_venv.close()
    del induct_model; gc.collect()

    all_episodes = baseline_eps + mid_eps + full_eps

    # ── CONTRASTIVE ANALYSIS ────────────────────────────────────────────────
    log_entry("[CONTRAST] Running contrastive analysis", "")
    feature_stats, within_stage = contrastive_analysis(all_episodes)
    invariances = discover_invariances(feature_stats, within_stage, top_n=20)

    # ── PLOTS ───────────────────────────────────────────────────────────────
    plot_results(feature_stats, invariances, all_episodes)

    # ── DATASET SUMMARY ─────────────────────────────────────────────────────
    stage_names = ["baseline", "mid_induction", "full_induction"]
    by_stage = {}
    for sl in stage_names:
        eps_s = [e for e in all_episodes if e[0]["stage"] == sl]
        if not eps_s:
            continue
        by_stage[sl] = {
            "n_episodes":    len(eps_s),
            "n_steps_total": int(sum(e[0]["n_steps"] for e in eps_s)),
            "shortcut":      sum(1 for e in eps_s if e[0]["outcome"] == "shortcut"),
            "real":          sum(1 for e in eps_s if e[0]["outcome"] == "real"),
            "timeout":       sum(1 for e in eps_s if e[0]["outcome"] == "timeout"),
            "near_shortcut": sum(1 for e in eps_s if e[0]["spatial_type"] == "near"),
            "far_shortcut":  sum(1 for e in eps_s if e[0]["spatial_type"] == "far"),
        }

    summary = {
        "total_episodes":        len(all_episodes),
        "total_steps":           int(sum(e[0]["n_steps"] for e in all_episodes)),
        "by_stage":              by_stage,
        "probe_log":             probe_log,
        "total_induction_steps": total_steps,
        "mid_collected":         mid_done,
        "full_collected":        full_done,
        "elapsed_min":           round((time.time() - t0) / 60, 1),
        "sae_path":              SAE_PATH,
        "w_path":                W_PATH,
        "n_features":            384,
        "top_invariance":        invariances[0] if invariances else None,
    }
    json.dump(summary,       open(os.path.join(OUT, "dataset_summary.json"),        "w"), indent=2)
    json.dump(feature_stats, open(os.path.join(OUT, "feature_analysis.json"),       "w"), indent=2)
    json.dump(invariances,   open(os.path.join(OUT, "discovered_invariances.json"), "w"), indent=2)
    if within_stage:
        json.dump(within_stage, open(os.path.join(OUT, "within_stage_analysis.json"), "w"), indent=2)

    log_entry("[CONTRAST] COMPLETE",
              f"- {len(all_episodes)} episodes, {summary['total_steps']:,} steps\n"
              f"- induction: {total_steps:,} steps, mid={mid_done}, full={full_done}\n"
              f"- top invariance: f{invariances[0]['feature_idx']} "
              f"d_clive={invariances[0]['d_clive']:.2f} ({invariances[0]['direction']})\n"
              f"- elapsed {summary['elapsed_min']} min")

    # ── FINAL PRINT ─────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("CONTRASTIVE CIRCUIT ANALYSIS — COMPLETE")
    print(f"{'='*68}")
    print(f"Total episodes : {len(all_episodes)}  |  Total steps : {summary['total_steps']:,}")
    print(f"Induction      : {total_steps:,} steps  (mid={mid_done}, full={full_done})")
    print()
    for sl, info in by_stage.items():
        print(f"  {sl:18s}  n={info['n_episodes']:3d}  "
              f"shortcut={info['shortcut']:3d}  real={info['real']:3d}  "
              f"timeout={info['timeout']:3d}  "
              f"(near={info['near_shortcut']} far={info['far_shortcut']})")
    print()
    print("Probe log (steps → hack_rate):")
    for p in probe_log:
        print(f"  {p['steps']:>7,} steps  →  hack_rate={p['hack_rate']:.2f}")
    print()
    print("Top-10 discovered invariances (ranked by |d_clive| on causal weight):")
    for inv in invariances[:10]:
        ws = (f"  within-stage d={inv['d_clive_within_stage']:.2f}"
              if "d_clive_within_stage" in inv else "")
        print(f"  #{inv['rank']:2d}  f{inv['feature_idx']:3d}  "
              f"d_clive={inv['d_clive']:+.2f}  ({inv['direction']:8s})  "
              f"d_h={inv['d_h']:+.2f}  [{inv['routing_vs_activation']}]{ws}")
    print()
    print(f"Output directory:  {OUT}")
    print(f"  dataset_summary.json            — episode-level metadata + probe log")
    print(f"  feature_analysis.json           — all 384 features with effect sizes & p-values")
    print(f"  discovered_invariances.json     — top-20 ranked candidate invariances")
    if within_stage:
        print(f"  within_stage_analysis.json      — mid_induction same-policy comparison")
    print(f"  episodes/  ({ep_global_idx} files each: .npz + .json)")
    print(f"             ep_NNNN_OUTCOME_STAGE.npz  →  h, c_live, action, agent_pos, ...")
    print(f"             ep_NNNN_OUTCOME_STAGE.json →  metadata")
    print(f"  plots/")
    print(f"    effect_sizes_clive.png         — bar chart of d_clive per feature")
    print(f"    routing_vs_activation.png      — d_clive vs d_h scatter")
    print(f"    top5_feature_traces.png        — c_live time traces for top-5 features")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()

"""
Separate goal features from proxy features in SAEv2 space using the TEST distribution.

On the training distribution the goal is always at (6,4), so a feature that fires
"at the goal" is indistinguishable from one that fires "at position (6,4)". They are
perfectly confounded. The only way to tell them apart is to move the goal away from
(6,4) — i.e. run on the test distribution — and ask:

  - Does the feature track the ACTUAL goal location?      → coin_tracking (goal feature)
  - Does the feature track the FIXED (6,4) position?      → proxy_position (proxy feature)

This rewrites feature_index_v2.json / feature_labels_v2.json with the separation.
"""

import sys, os, json, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from models.topk_sae_v2 import TopKSAEv2
from envs.coin_env import make_env_with_info
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
CKPT_DIR = os.path.join(BASE, "outputs/checkpoints")
OUT_DIR = os.path.join(BASE, "outputs")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
FIXED_GOAL = np.array([6, 4])
N_STEPS = 30000


def load_sae_v2():
    ckpt = torch.load(os.path.join(CKPT_DIR, "sae_v2_best.pt"), map_location=device)
    sae = TopKSAEv2(input_dim=ckpt["input_dim"], hidden_factor=ckpt["hidden_factor"],
                    k=ckpt["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt


def main():
    sae, ckpt = load_sae_v2()
    mean = np.array(ckpt["act_mean"]); std = np.array(ckpt["act_std"])
    hidden = sae.hidden_dim

    log_entry("[SAEv2] Test-distribution feature labeling START",
              f"- collecting {N_STEPS:,} test-distribution steps to decouple goal vs position")

    model = PPO.load(os.path.join(CKPT_DIR, "ppo_final.zip"), device=str(device))
    model.policy.eval()

    captured = {}
    def hook(_m, _i, out): captured["f"] = out.detach().cpu()
    handle = model.policy.features_extractor.register_forward_hook(hook)

    H = np.zeros((N_STEPS, hidden), dtype=np.float32)
    goal_prox = np.zeros(N_STEPS, dtype=np.float32)   # -dist(agent, actual goal)
    fixed_prox = np.zeros(N_STEPS, dtype=np.float32)  # -dist(agent, (6,4))

    env = make_env_with_info(goal_fixed=False, goal_displacement=-1)
    obs, info = env.reset()
    idx = 0
    while idx < N_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        feat = captured["f"].squeeze(0).numpy()
        fn = ((feat - mean) / std).astype(np.float32)
        with torch.no_grad():
            h = sae.get_feature_activations(torch.from_numpy(fn).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        H[idx] = h
        gp = np.array(info["goal_pos"], dtype=np.float32)
        ap = np.array(info["agent_pos"], dtype=np.float32)
        goal_prox[idx]  = -np.abs(ap - gp).sum()
        fixed_prox[idx] = -np.abs(ap - FIXED_GOAL).sum()
        obs, reward, term, trunc, info = env.step(action)
        idx += 1
        if term or trunc:
            obs, info = env.reset()
    handle.remove(); env.close()

    # Load existing v2 stats (training-distribution freq, near_goal_bias, etc.)
    with open(os.path.join(OUT_DIR, "feature_labels_v2.json")) as f:
        feature_stats = json.load(f)
    top50 = [int(k) for k in feature_stats.keys()]

    # Correlate each feature with actual-goal proximity vs fixed-position proximity
    for fi in top50:
        fa = H[:, fi]
        if (fa > 0).sum() > 20 and goal_prox.std() > 1e-6:
            goal_corr  = float(np.corrcoef(fa, goal_prox)[0, 1])
            fixed_corr = float(np.corrcoef(fa, fixed_prox)[0, 1])
        else:
            goal_corr, fixed_corr = 0.0, 0.0
        feature_stats[str(fi)]["goal_track_corr"]  = goal_corr
        feature_stats[str(fi)]["fixed_track_corr"] = fixed_corr
        feature_stats[str(fi)]["track_separation"] = goal_corr - fixed_corr

    # Relabel by which signal the feature tracks more strongly
    for fi in top50:
        s = feature_stats[str(fi)]
        gc_, fc_ = s["goal_track_corr"], s["fixed_track_corr"]
        if gc_ > 0.1 and gc_ > fc_ + 0.03:
            s["label"] = "coin_tracking"
        elif fc_ > 0.1 and fc_ > gc_ + 0.03:
            s["label"] = "proxy_position"
        elif s["activation_frequency"] > 0.5:
            s["label"] = "background_texture"
        elif s["activation_frequency"] < 0.05:
            s["label"] = "rare_event"
        else:
            s["label"] = "unknown"

    label_counts = {}
    for s in feature_stats.values():
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1

    goal_features  = [int(k) for k, v in feature_stats.items() if v["label"] == "coin_tracking"]
    proxy_features = [int(k) for k, v in feature_stats.items()
                      if v["label"] in ("proxy_position", "background_texture")]

    # Fallback: if still no goal features, take the 5 features with highest goal_track_corr
    if not goal_features:
        ranked = sorted(top50, key=lambda fi: feature_stats[str(fi)]["goal_track_corr"], reverse=True)
        goal_features = [fi for fi in ranked[:5]]
        for fi in goal_features:
            feature_stats[str(fi)]["label"] = "coin_tracking"
        log_entry("[SAEv2] No features passed goal threshold — using top-5 by goal_track_corr",
                  f"- goal_features (fallback): {goal_features}")

    with open(os.path.join(OUT_DIR, "feature_labels_v2.json"), "w") as f:
        json.dump(feature_stats, f, indent=2)
    with open(os.path.join(OUT_DIR, "feature_index_v2.json")) as f:
        idx_data = json.load(f)
    idx_data["goal_features"]  = goal_features
    idx_data["proxy_features"] = proxy_features
    idx_data["label_counts"]   = label_counts
    with open(os.path.join(OUT_DIR, "feature_index_v2.json"), "w") as f:
        json.dump(idx_data, f, indent=2)

    # Plot goal_track vs fixed_track scatter
    gcs = [feature_stats[str(fi)]["goal_track_corr"] for fi in top50]
    fcs = [feature_stats[str(fi)]["fixed_track_corr"] for fi in top50]
    cols = ["green" if feature_stats[str(fi)]["label"] == "coin_tracking"
            else "red" if feature_stats[str(fi)]["label"] == "proxy_position"
            else "gray" for fi in top50]
    plt.figure(figsize=(7, 7))
    plt.scatter(fcs, gcs, c=cols, s=50, alpha=0.7)
    plt.plot([-1, 1], [-1, 1], "k--", alpha=0.3)
    plt.xlabel("Fixed-position tracking corr (proxy signal)")
    plt.ylabel("Actual-goal tracking corr (goal signal)")
    plt.title("SAEv2 features: goal-tracking vs position-tracking (test dist)")
    plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "sae_v2_goal_vs_proxy_scatter.png"), dpi=150)
    plt.close()

    log_entry("[SAEv2] Test-distribution labeling COMPLETE",
              f"- label_counts: {label_counts}\n"
              f"- goal_features: {goal_features}\n"
              f"- proxy_features: {proxy_features}")

    print(f"\n{'='*60}")
    print("SAEv2 TEST-DISTRIBUTION LABELING COMPLETE")
    print(f"Label counts:   {label_counts}")
    print(f"Goal features:  {goal_features}")
    print(f"Proxy features: {proxy_features}")
    print(f"{'='*60}\n")

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()

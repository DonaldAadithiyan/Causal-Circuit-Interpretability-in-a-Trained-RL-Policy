"""
Re-identify goal/proxy features in the retrained SAEv2 feature space.
The v1 feature indices ([933, 151, ...]) refer to the 1024-dim v1 SAE and are
meaningless for SAEv2. This recomputes the same statistics (reward_corr,
agent_near_goal_bias) and re-labels features, saving v2 index/label files.
"""

import sys, os, json, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.topk_sae_v2 import TopKSAEv2
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
ACT_DIR = os.path.join(BASE, "outputs/activations")
CKPT_DIR = os.path.join(BASE, "outputs/checkpoints")
OUT_DIR = os.path.join(BASE, "outputs")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

FIXED_GOAL = np.array([6, 4])


def load_sae_v2():
    ckpt = torch.load(os.path.join(CKPT_DIR, "sae_v2_best.pt"), map_location=device)
    sae = TopKSAEv2(input_dim=ckpt["input_dim"], hidden_factor=ckpt["hidden_factor"],
                    k=ckpt["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt


def main():
    sae, ckpt = load_sae_v2()
    mean = np.array(ckpt["act_mean"])
    std = np.array(ckpt["act_std"])

    log_entry("[SAEv2] Feature re-analysis START",
              f"- hidden_dim={sae.hidden_dim}, dead={ckpt.get('dead_features','?')}")

    # Load activation dataset + metadata
    with open(os.path.join(ACT_DIR, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n_samples"]
    dim = meta["features_dim"]
    acts = np.memmap(os.path.join(ACT_DIR, "activations.npy"), dtype=np.float32,
                     mode="r", shape=(n, dim))
    actions = np.load(os.path.join(ACT_DIR, "actions.npy"))
    rewards = np.load(os.path.join(ACT_DIR, "rewards.npy"))
    agent_pos = np.load(os.path.join(ACT_DIR, "agent_pos.npy"))

    # Compute SAEv2 feature activations for all samples
    print(f"Computing SAEv2 features for {n:,} samples...")
    bs = 1024
    hidden = sae.hidden_dim
    all_h = np.zeros((n, hidden), dtype=np.float32)
    for start in range(0, n, bs):
        end = min(start + bs, n)
        x = torch.from_numpy(((acts[start:end] - mean) / std).astype(np.float32)).to(device)
        with torch.no_grad():
            all_h[start:end] = sae.get_feature_activations(x).cpu().numpy()

    freq = (all_h > 0).mean(0)
    top50 = np.argsort(freq)[::-1][:50]

    # Agent-goal distance for near/far bias
    agent_goal_dist = np.abs(agent_pos - FIXED_GOAL).sum(axis=1)
    near_mask = agent_goal_dist <= np.percentile(agent_goal_dist, 25)
    far_mask = agent_goal_dist > np.percentile(agent_goal_dist, 75)

    # Reward-frame masks — the true "fires at the goal" signal.
    # Sparse terminal reward means per-step Pearson corr is ~0 for clean features,
    # so we use mean activation on reward frames vs non-reward frames instead.
    rew_mask = rewards > 0
    norew_mask = rewards == 0

    feature_stats = {}
    for rank, fi in enumerate(top50):
        fa = all_h[:, fi]
        nonzero = fa > 0
        if nonzero.sum() > 10:
            rew_corr = float(np.corrcoef(fa, rewards)[0, 1])
            act_corr = float(max(abs(np.corrcoef(fa, (actions == a).astype(float))[0, 1])
                                 for a in np.unique(actions)))
        else:
            rew_corr, act_corr = 0.0, 0.0

        if near_mask.sum() > 5 and far_mask.sum() > 5:
            near_goal_bias = float(fa[near_mask].mean() - fa[far_mask].mean())
        else:
            near_goal_bias = 0.0

        # Goal-activation score: how much more this feature fires at the goal
        # (reward frame) than elsewhere. Normalised by the feature's overall scale.
        scale = fa.mean() + 1e-8
        goal_act_score = float((fa[rew_mask].mean() - fa[norew_mask].mean()) / scale)

        feature_stats[int(fi)] = {
            "rank": int(rank),
            "activation_frequency": float(freq[fi]),
            "mean_activation": float(fa.mean()),
            "reward_correlation": rew_corr,
            "action_correlation_max": act_corr,
            "agent_near_goal_bias": near_goal_bias,
            "goal_activation_score": goal_act_score,
            "label": "unknown",
        }

    # Label using goal_activation_score (reward-frame signal) as the primary cue.
    # coin_tracking: fires markedly more on goal frames.
    # proxy_position: fires during approach to training position but NOT at the goal.
    for fi, s in feature_stats.items():
        gscore = s["goal_activation_score"]
        g = s["agent_near_goal_bias"]
        a = s["action_correlation_max"]
        f = s["activation_frequency"]
        if gscore > 0.15:
            s["label"] = "coin_tracking"
        elif g > 0.1 and gscore < 0.05:
            s["label"] = "proxy_position"
        elif a > 0.5 and gscore < 0.05:
            s["label"] = "action_spurious"
        elif f > 0.5:
            s["label"] = "background_texture"
        elif f < 0.05:
            s["label"] = "rare_event"
        else:
            s["label"] = "unknown"

    label_counts = {}
    for s in feature_stats.values():
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1

    goal_features = [k for k, v in feature_stats.items() if v["label"] == "coin_tracking"]
    proxy_features = [k for k, v in feature_stats.items()
                      if v["label"] in ("proxy_position", "background_texture", "action_spurious")]

    # Save v2 files
    with open(os.path.join(OUT_DIR, "feature_labels_v2.json"), "w") as f:
        json.dump(feature_stats, f, indent=2)
    index_data = {
        "top50_feature_indices": [int(i) for i in top50],
        "goal_features": goal_features,
        "proxy_features": proxy_features,
        "label_counts": label_counts,
        "sae_hidden_dim": sae.hidden_dim,
    }
    with open(os.path.join(OUT_DIR, "feature_index_v2.json"), "w") as f:
        json.dump(index_data, f, indent=2)

    # Plot label distribution
    plt.figure(figsize=(8, 4))
    plt.bar(list(label_counts.keys()), list(label_counts.values()), color="teal")
    plt.ylabel("Count"); plt.title(f"SAEv2 Feature Labels (hidden={sae.hidden_dim})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "sae_v2_feature_labels.png"), dpi=150)
    plt.close()

    log_entry("[SAEv2] Feature re-analysis COMPLETE",
              f"- label_counts: {label_counts}\n"
              f"- goal_features: {goal_features}\n"
              f"- proxy_features: {proxy_features}")

    print(f"\n{'='*60}")
    print(f"SAEv2 FEATURE RE-ANALYSIS COMPLETE")
    print(f"Label distribution: {label_counts}")
    print(f"Goal features:  {goal_features}")
    print(f"Proxy features: {proxy_features}")
    print(f"{'='*60}\n")

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()

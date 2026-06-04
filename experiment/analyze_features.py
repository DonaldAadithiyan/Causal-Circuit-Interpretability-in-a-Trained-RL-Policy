"""
Phase 3 — Feature interpretability analysis.
For top 50 most-active SAE features:
  - Save 5x4 grids of max/min activating observations
  - Compute spatial correlation with agent and goal positions
  - Compute reward and action correlations
  - Output feature_labels.json for manual labelling
"""

import sys, os, json, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from stable_baselines3 import PPO
from models.topk_sae import TopKSAE
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
ACT_DIR = os.path.join(BASE, "outputs/activations")
CKPT_DIR = os.path.join(BASE, "outputs/checkpoints")
OUT_DIR = os.path.join(BASE, "outputs")
MAX_DIR = os.path.join(OUT_DIR, "plots/feature_max_activations")
MIN_DIR = os.path.join(OUT_DIR, "plots/feature_min_activations")
HEATMAP_DIR = os.path.join(OUT_DIR, "plots/feature_heatmaps")


def load_sae() -> TopKSAE:
    ckpt = torch.load(os.path.join(CKPT_DIR, "sae_best.pt"), map_location=device)
    sae = TopKSAE(
        input_dim=ckpt["input_dim"],
        hidden_factor=ckpt["hidden_factor"],
        k=ckpt["k"],
    ).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt


def load_activations():
    with open(os.path.join(ACT_DIR, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n_samples"]
    dim = meta["features_dim"]
    acts = np.memmap(os.path.join(ACT_DIR, "activations.npy"), dtype=np.float32,
                     mode="r", shape=(n, dim))
    obs = np.memmap(os.path.join(ACT_DIR, "observations.npy"), dtype=np.uint8,
                    mode="r", shape=(n, 64, 64, 3))
    actions = np.load(os.path.join(ACT_DIR, "actions.npy"))
    rewards = np.load(os.path.join(ACT_DIR, "rewards.npy"))
    goal_pos = np.load(os.path.join(ACT_DIR, "goal_pos.npy"))
    agent_pos = np.load(os.path.join(ACT_DIR, "agent_pos.npy"))
    return acts, obs, actions, rewards, goal_pos, agent_pos, n


def save_image_grid(images, path, title="", ncols=5, nrows=4):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def compute_spatial_heatmap(feature_acts, positions, grid_size=8):
    """Correlation between feature activation and agent/goal position (grid cell)."""
    heatmap = np.zeros((grid_size, grid_size))
    counts = np.zeros((grid_size, grid_size))
    for act_val, pos in zip(feature_acts, positions):
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < grid_size and 0 <= y < grid_size:
            heatmap[y, x] += act_val
            counts[y, x] += 1
    with np.errstate(invalid="ignore"):
        heatmap = np.where(counts > 0, heatmap / counts, 0.0)
    return heatmap


def main():
    os.makedirs(MAX_DIR, exist_ok=True)
    os.makedirs(MIN_DIR, exist_ok=True)
    os.makedirs(HEATMAP_DIR, exist_ok=True)

    log_entry("Phase 3 START — Feature interpretability analysis", "")

    sae, ckpt = load_sae()
    acts_raw, obs_data, actions, rewards, goal_pos, agent_pos, n = load_activations()

    mean = np.array(ckpt["act_mean"])
    std = np.array(ckpt["act_std"])

    # Batch-compute SAE features for all samples
    print(f"Computing SAE features for {n:,} samples...")
    batch_size = 1024
    hidden_dim = sae.hidden_dim
    all_features = np.zeros((n, hidden_dim), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.from_numpy(((acts_raw[start:end] - mean) / std).astype(np.float32)).to(device)
        with torch.no_grad():
            h = sae.get_feature_activations(x)
        all_features[start:end] = h.cpu().numpy()

    # Feature activation frequency
    freq = (all_features > 0).mean(0)
    top50_idx = np.argsort(freq)[::-1][:50]

    log_entry("Phase 3 — Feature frequencies computed",
              f"- Top feature activation rate: {freq[top50_idx[0]]:.4f}\n"
              f"- Median activation rate: {np.median(freq):.4f}\n"
              f"- Dead features (<0.1%): {int((freq < 0.001).sum())}")

    # Per-feature analysis
    feature_stats = {}
    for rank, feat_idx in enumerate(top50_idx):
        feat_acts = all_features[:, feat_idx]

        # Max/min activating images (top 20)
        top20_idx = np.argsort(feat_acts)[::-1][:20]
        bot20_idx = np.argsort(feat_acts)[:20]

        save_image_grid(
            [obs_data[i] for i in top20_idx],
            os.path.join(MAX_DIR, f"feature_{feat_idx:04d}_rank{rank:02d}_max.png"),
            title=f"Feature {feat_idx} (rank {rank}) — Max activations",
        )
        save_image_grid(
            [obs_data[i] for i in bot20_idx],
            os.path.join(MIN_DIR, f"feature_{feat_idx:04d}_rank{rank:02d}_min.png"),
            title=f"Feature {feat_idx} (rank {rank}) — Min activations",
        )

        # Spatial correlation heatmaps
        agent_heatmap = compute_spatial_heatmap(feat_acts, agent_pos)
        goal_heatmap = compute_spatial_heatmap(feat_acts, goal_pos)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        im0 = axes[0].imshow(agent_heatmap, cmap="hot")
        axes[0].set_title("Agent position")
        plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(goal_heatmap, cmap="hot")
        axes[1].set_title("Goal position")
        plt.colorbar(im1, ax=axes[1])
        fig.suptitle(f"Feature {feat_idx} spatial correlation")
        plt.tight_layout()
        plt.savefig(os.path.join(HEATMAP_DIR, f"feature_{feat_idx:04d}_spatial.png"),
                    dpi=100, bbox_inches="tight")
        plt.close()

        # Reward / action correlations
        nonzero = feat_acts > 0
        if nonzero.sum() > 10:
            rew_corr = float(np.corrcoef(feat_acts, rewards)[0, 1])
            act_corr_max = float(
                max(abs(np.corrcoef(feat_acts, (actions == a).astype(float))[0, 1])
                    for a in np.unique(actions))
            )
        else:
            rew_corr = 0.0
            act_corr_max = 0.0

        # Agent-goal proximity bias: activation when agent is near goal vs far
        # Training goal is fixed at (6,4); use Manhattan distance to that position
        FIXED_GOAL = np.array([6, 4])
        agent_goal_dist = np.abs(agent_pos - FIXED_GOAL).sum(axis=1)  # Manhattan distance
        near_threshold = np.percentile(agent_goal_dist, 25)  # bottom 25% = near goal
        near_goal_mask = agent_goal_dist <= near_threshold
        far_goal_mask = agent_goal_dist > np.percentile(agent_goal_dist, 75)
        agent_near_goal_bias = float(
            np.mean(feat_acts[near_goal_mask]) - np.mean(feat_acts[far_goal_mask])
        ) if near_goal_mask.sum() > 5 and far_goal_mask.sum() > 5 else 0.0

        feature_stats[int(feat_idx)] = {
            "rank": int(rank),
            "activation_frequency": float(freq[feat_idx]),
            "mean_activation": float(feat_acts.mean()),
            "max_activation": float(feat_acts.max()),
            "reward_correlation": float(rew_corr),
            "action_correlation_max": float(act_corr_max),
            "agent_near_goal_bias": float(agent_near_goal_bias),
            "label": "unknown",
        }

        if rank < 10:
            log_entry(
                f"Phase 3 — Feature {feat_idx} (rank {rank}) analysed",
                f"- freq: {freq[feat_idx]:.4f}\n"
                f"- reward_corr: {rew_corr:.4f}\n"
                f"- action_corr: {act_corr_max:.4f}\n"
                f"- agent_near_goal_bias: {agent_near_goal_bias:.4f}",
            )

    # Auto-label by heuristics
    for feat_idx, stats in feature_stats.items():
        r = stats["reward_correlation"]
        g = stats["agent_near_goal_bias"]  # high = activates more when agent near goal
        a = stats["action_correlation_max"]
        f = stats["activation_frequency"]

        if r > 0.1 and g > 0.05:
            # Activates more near goal AND correlates with reward → goal-tracking feature
            stats["label"] = "coin_tracking"
        elif g > 0.05 and abs(r) < 0.05:
            # Activates near goal position but low reward correlation → proxy (spurious cue)
            stats["label"] = "proxy_position"
        elif a > 0.2 and abs(r) < 0.05:
            # Strong action correlation but weak reward correlation → action-spurious
            stats["label"] = "action_spurious"
        elif f > 0.5:
            # Very frequently active → background texture / positional
            stats["label"] = "background_texture"
        elif f < 0.05:
            stats["label"] = "rare_event"
        else:
            stats["label"] = "unknown"

    # Save labels JSON
    labels_path = os.path.join(OUT_DIR, "feature_labels.json")
    with open(labels_path, "w") as f:
        json.dump(feature_stats, f, indent=2)

    # Summary
    label_counts = {}
    for s in feature_stats.values():
        lbl = s["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # Identify goal features and proxy features for downstream use
    goal_features = [k for k, v in feature_stats.items() if v["label"] == "coin_tracking"]
    proxy_features = [k for k, v in feature_stats.items()
                      if v["label"] in ("proxy_position", "background_texture", "action_spurious")]

    index_data = {
        "top50_feature_indices": [int(i) for i in top50_idx],
        "goal_features": goal_features,
        "proxy_features": proxy_features,
        "label_counts": label_counts,
    }
    with open(os.path.join(OUT_DIR, "feature_index.json"), "w") as f:
        json.dump(index_data, f, indent=2)

    log_entry(
        "Phase 3 COMPLETE",
        f"- Features analysed: 50\n"
        f"- Label distribution: {label_counts}\n"
        f"- Goal features: {goal_features[:5]}...\n"
        f"- Proxy features: {proxy_features[:5]}...\n"
        f"- Plots: {MAX_DIR}\n"
        f"- Labels: {labels_path}",
    )

    print(f"\n{'='*60}")
    print(f"PHASE 3 COMPLETE")
    print(f"Label distribution: {label_counts}")
    print(f"Goal features:  {goal_features}")
    print(f"Proxy features: {proxy_features[:10]}")
    print(f"{'='*60}\n")

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()

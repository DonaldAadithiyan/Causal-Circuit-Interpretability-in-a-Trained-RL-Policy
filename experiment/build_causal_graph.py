"""
Experiment 2, Phase 1 — Build G* via activation patching in SAE space.
Uses 200 observations and KL threshold 0.01 (vs Exp1: 100 obs, threshold 0.1).
Extracts the five invariant profiles for I1-I5.
"""

import sys, os, json, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from stable_baselines3 import PPO
from models.topk_sae import TopKSAE
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
ACT_DIR = os.path.join(BASE, "outputs/activations")
CKPT_DIR = os.path.join(BASE, "outputs/checkpoints")
GRAPH_DIR = os.path.join(BASE, "outputs/graphs")
OUT_DIR = os.path.join(BASE, "outputs")
EXP2_PLOT = os.path.join(BASE, "outputs/experiment2/plots")


def load_sae():
    ckpt = torch.load(os.path.join(CKPT_DIR, "sae_best.pt"), map_location=device)
    sae = TopKSAE(input_dim=ckpt["input_dim"], hidden_factor=ckpt["hidden_factor"], k=ckpt["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt


def kl_div(p, q):
    p = p + 1e-8
    q = q + 1e-8
    return (p * torch.log(p / q)).sum(-1).mean().item()


def main():
    os.makedirs(EXP2_PLOT, exist_ok=True)

    log_entry("[EXP2] Phase 1 START — Build G* (200 obs, KL threshold 0.01)", "")

    sae, ckpt = load_sae()
    mean = np.array(ckpt["act_mean"])
    std = np.array(ckpt["act_std"])
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)

    with open(os.path.join(OUT_DIR, "feature_index.json")) as f:
        feat_index = json.load(f)
    with open(os.path.join(OUT_DIR, "feature_labels.json")) as f:
        feat_labels = json.load(f)

    top32 = feat_index["top50_feature_indices"][:32]
    goal_features = feat_index["goal_features"]
    proxy_features = feat_index["proxy_features"]

    model = PPO.load(os.path.join(CKPT_DIR, "ppo_final.zip"), device=str(device))
    model.policy.eval()

    with open(os.path.join(ACT_DIR, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n_samples"]
    dim = meta["features_dim"]
    acts_mm = np.memmap(os.path.join(ACT_DIR, "activations.npy"), dtype=np.float32, mode="r", shape=(n, dim))

    sample_idx = np.random.choice(n, 200, replace=False)
    acts_raw = acts_mm[sample_idx]
    acts_norm = torch.from_numpy(((acts_raw - mean) / std).astype(np.float32)).to(device)

    with torch.no_grad():
        _, h_baseline = sae(acts_norm)
        recon_norm_b = sae.decode(h_baseline)
        recon_raw_b = recon_norm_b * std_t + mean_t
        logits_b = model.policy.action_net(recon_raw_b)
        probs_b = F.softmax(logits_b, dim=-1)

    # c* via patching for each of top-32 features
    kl_threshold = 0.01
    c_star = np.zeros(32)

    for rank in range(32):
        feat_global = top32[rank]
        with torch.no_grad():
            h_p = h_baseline.clone()
            h_p[:, feat_global] = 0.0
            recon_norm_p = sae.decode(h_p)
            recon_raw_p = recon_norm_p * std_t + mean_t
            logits_p = model.policy.action_net(recon_raw_p)
            probs_p = F.softmax(logits_p, dim=-1)
            c_star[rank] = kl_div(probs_b, probs_p)

    pass_rate = float((c_star > kl_threshold).mean())
    sorted_rank = np.argsort(c_star)[::-1]
    top16_by_causal = [top32[r] for r in sorted_rank[:16]]

    # Inter-feature 16×16 cosine similarity matrix (decoder directions)
    W_dec = sae.decoder.weight.detach()
    top16_t = torch.tensor(top16_by_causal, device=device)
    W16 = F.normalize(W_dec[:, top16_t].T, dim=1)
    inter_feat_matrix = torch.abs(W16 @ W16.T).detach().cpu().numpy()
    np.fill_diagonal(inter_feat_matrix, 0.0)

    # Normalised c*
    c_star_norm = c_star / (c_star.sum() + 1e-8)
    top1_conc = float(c_star_norm[sorted_rank[0]])  # I1 depth proxy

    # I2 spurious set: negative reward_corr AND low c*
    spurious_set = []
    for i, feat in enumerate(top32):
        rew_corr = feat_labels.get(str(feat), {}).get("reward_correlation", 0)
        if rew_corr < -0.05 and c_star[i] < 0.005:
            spurious_set.append(feat)

    sp_vals = [c_star[top32.index(f)] for f in spurious_set if f in top32]
    sp_mean = float(np.mean(sp_vals)) if sp_vals else 0.0
    sp_std = float(np.std(sp_vals)) if len(sp_vals) > 1 else 1e-6

    # I3 goal baseline
    goal_in32 = [f for f in goal_features if f in top32]
    goal_c_mean = float(np.mean([c_star[top32.index(f)] for f in goal_in32])) if goal_in32 else 0.0

    # I4 proxy baseline
    proxy_in32 = [f for f in proxy_features if f in top32]
    proxy_c_mean = float(np.mean([c_star[top32.index(f)] for f in proxy_in32])) if proxy_in32 else 0.0

    # I5 baseline: pass rate of self-consistency test on top-3 features
    i5_vals = [1.0 if c_star[sorted_rank[i]] > kl_threshold else 0.0 for i in range(min(3, 32))]
    i5_baseline = float(np.mean(i5_vals))

    metadata = {
        "top32_features": [int(x) for x in top32],
        "top16_by_causal": [int(x) for x in top16_by_causal],
        "c_star": c_star.tolist(),
        "c_star_normalized": c_star_norm.tolist(),
        "kl_threshold": kl_threshold,
        "pass_rate": pass_rate,
        "max_kl": float(c_star.max()),
        "mean_kl": float(c_star.mean()),
        # I1
        "depth_concentration_star": top1_conc,
        # I2
        "spurious_set": [int(x) for x in spurious_set],
        "c_star_spurious_mean": sp_mean,
        "c_star_spurious_std": sp_std,
        # I3
        "goal_features_in_top32": [int(x) for x in goal_in32],
        "goal_c_star_mean": goal_c_mean,
        # I4
        "proxy_features_in_top32": [int(x) for x in proxy_in32],
        "proxy_c_star_mean": proxy_c_mean,
        # I5
        "i5_top3_features": [int(top32[sorted_rank[i]]) for i in range(min(3, 32))],
        "i5_baseline": i5_baseline,
        # Violation thresholds
        "i3_threshold": float(goal_c_mean * 0.5),
        "i4_threshold": float(proxy_c_mean * 1.5),
        "v_total_threshold": float(goal_c_mean * 0.1),
    }

    np.save(os.path.join(GRAPH_DIR, "G_star.npy"), c_star)
    np.save(os.path.join(GRAPH_DIR, "inter_feat_matrix.npy"), inter_feat_matrix)
    with open(os.path.join(GRAPH_DIR, "G_star_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Plot
    colors = ["cornflowerblue" if top32[sorted_rank[r]] in goal_features
              else "coral" if top32[sorted_rank[r]] in proxy_features
              else "lightgray" for r in range(32)]
    plt.figure(figsize=(12, 4))
    plt.bar(range(32), c_star[sorted_rank], color=colors)
    plt.axhline(kl_threshold, color="red", linestyle="--", label=f"KL thresh={kl_threshold}")
    legend_elems = [Patch(facecolor="cornflowerblue", label="coin_tracking"),
                    Patch(facecolor="coral", label="proxy_position"),
                    Patch(facecolor="lightgray", label="other")]
    plt.legend(handles=legend_elems)
    plt.xlabel("Feature rank by c*")
    plt.ylabel("KL divergence (c*)")
    plt.title(f"G* Causal Importance (max={c_star.max():.5f}, pass_rate={pass_rate:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(EXP2_PLOT, "g_star_causal_importance.png"), dpi=150)
    plt.close()

    log_entry("[EXP2] Phase 1 COMPLETE — G* saved",
              f"- max_kl: {c_star.max():.6f}\n"
              f"- pass_rate (>{kl_threshold}): {pass_rate:.2f}\n"
              f"- goal_c_mean: {goal_c_mean:.6f}\n"
              f"- proxy_c_mean: {proxy_c_mean:.6f}\n"
              f"- i3_threshold: {goal_c_mean*0.5:.6f}\n"
              f"- spurious_set: {spurious_set}")

    print(f"\nG* complete. max_kl={c_star.max():.6f}, pass_rate={pass_rate:.2f}")
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return metadata


if __name__ == "__main__":
    main()

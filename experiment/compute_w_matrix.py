"""
Compute and validate the inter-feature causal influence matrix W.

W = D^T @ W_enc^T   (shape: hidden_dim × hidden_dim)

W_ij = how much does feature i's decoder direction contribute to activating feature j.

This is a one-time computation from SAE weight matrices — no gradients, no forward passes.
"""

import sys, os, json, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from stable_baselines3 import PPO
from models.topk_sae_v2 import TopKSAEv2
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
ACT_DIR = os.path.join(BASE, "outputs/activations")
CKPT_DIR = os.path.join(BASE, "outputs/checkpoints")
GRAPH_DIR = os.path.join(BASE, "outputs/graphs")
OUT_DIR = os.path.join(BASE, "outputs")
EXP2B_DIR = os.path.join(BASE, "outputs/experiment2b")
PLOT_DIR = os.path.join(EXP2B_DIR, "plots")


def load_sae_v2():
    ckpt = torch.load(os.path.join(CKPT_DIR, "sae_v2_best.pt"), map_location=device)
    sae = TopKSAEv2(input_dim=ckpt["input_dim"], hidden_factor=ckpt["hidden_factor"],
                    k=ckpt["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt


def compute_w(sae: TopKSAEv2) -> np.ndarray:
    """
    W = D^T @ W_enc^T
    D = decoder.weight  shape (input_dim, hidden_dim)  → columns are decoder directions
    W_enc = encoder.weight  shape (hidden_dim, input_dim)  → rows are encoder directions
    W = (hidden_dim, input_dim) @ (input_dim, hidden_dim) = (hidden_dim, hidden_dim)
    W[i,j] = decoder_dir_i · encoder_dir_j = influence of feature i on activating feature j
    """
    with torch.no_grad():
        D = sae.decoder.weight.detach()       # (input_dim, hidden_dim)
        W_enc = sae.encoder.weight.detach()   # (hidden_dim, input_dim)
        W = D.T @ W_enc.T                     # (hidden_dim, hidden_dim)
    return W.cpu().numpy()


def validate_w_against_patching(W: np.ndarray, sae: TopKSAEv2, model,
                                 acts_norm: torch.Tensor, top32: list,
                                 mean_t: torch.Tensor, std_t: torch.Tensor) -> float:
    """
    For each pair (i,j) in top32:
      patching: zero feature i → reconstruct → re-encode → measure Δh_j
      W-predicted: W[i,j] × mean_activation_i
    Return Pearson r between patching and W predictions.
    """
    with torch.no_grad():
        _, h_baseline = sae(acts_norm)
        mean_acts = h_baseline.mean(0).cpu().numpy()

    patching_deltas = []
    w_predictions = []

    for i_feat in top32:
        for j_feat in top32:
            if i_feat == j_feat:
                continue

            # Patching: zero feature i, re-encode
            with torch.no_grad():
                h_patched = h_baseline.clone()
                h_patched[:, i_feat] = 0.0
                # Reconstruct in normalized space, re-encode
                recon_patched = sae.decode(h_patched)
                h_reenc_pre = sae.encoder(recon_patched)
                h_reenc = sae.top_k_gate(h_reenc_pre)
                delta_j = float((h_reenc[:, j_feat] - h_baseline[:, j_feat]).abs().mean().item())

            patching_deltas.append(delta_j)
            w_predictions.append(float(abs(W[i_feat, j_feat]) * max(mean_acts[i_feat], 1e-8)))

    arr_patch = np.array(patching_deltas)
    arr_w = np.array(w_predictions)

    if arr_patch.std() < 1e-10 or arr_w.std() < 1e-10:
        return 0.0
    r, _ = pearsonr(arr_patch, arr_w)
    return float(r)


def build_g_star_from_w(W: np.ndarray, sae: TopKSAEv2, model,
                         acts_norm: torch.Tensor, top32: list,
                         mean_t: torch.Tensor, std_t: torch.Tensor,
                         feat_labels: dict, goal_features: list,
                         proxy_features: list) -> dict:
    """
    Build G* from W and compute c* (feature-to-action KL vector).
    Returns metadata dict matching the format expected by measure_invariances.py.
    """
    # c* via patching (unchanged from Exp2)
    kl_threshold = 0.01
    with torch.no_grad():
        _, h_b = sae(acts_norm)
        logits_b = model.policy.action_net(sae.decode(h_b) * std_t + mean_t)
        probs_b = F.softmax(logits_b, dim=-1)

    c_star = np.zeros(32)
    for rank, feat in enumerate(top32):
        with torch.no_grad():
            h_p = h_b.clone()
            h_p[:, feat] = 0.0
            logits_p = model.policy.action_net(sae.decode(h_p) * std_t + mean_t)
            probs_p = F.softmax(logits_p, dim=-1)
            p = probs_b + 1e-8; q = probs_p + 1e-8
            c_star[rank] = (p * torch.log(p / q)).sum(-1).mean().item()

    # Mean activations for top32
    with torch.no_grad():
        mean_acts = h_b.mean(0).cpu().numpy()

    # 32×32 G* submatrix weighted by mean activation
    G_star_32 = np.array([[W[top32[i], top32[j]] for j in range(32)] for i in range(32)])
    mean_acts_top32 = np.array([mean_acts[f] for f in top32])  # (32,)
    G_star_32 = G_star_32 * mean_acts_top32[:, None]  # weight rows by source activation
    np.fill_diagonal(G_star_32, 0.0)

    # I1: depth concentration
    c_star_norm = c_star / (c_star.sum() + 1e-8)
    sorted_rank = np.argsort(c_star)[::-1]
    top1_conc = float(c_star_norm[sorted_rank[0]])

    # I2: spurious set — features with negative reward_corr AND c* < 0.005
    spurious_set = []
    for i, feat in enumerate(top32):
        rew_corr = feat_labels.get(str(feat), {}).get("reward_correlation", 0)
        if rew_corr < -0.05 and c_star[i] < 0.005:
            spurious_set.append(feat)

    sp_vals = [c_star[top32.index(f)] for f in spurious_set if f in top32]
    sp_mean = float(np.mean(sp_vals)) if sp_vals else 0.0
    sp_std  = float(np.std(sp_vals)) if len(sp_vals) > 1 else 1e-6

    # I3 goal baseline
    goal_in32 = [f for f in goal_features if f in top32]
    goal_c_mean = float(np.mean([c_star[top32.index(f)] for f in goal_in32])) if goal_in32 else 0.0

    # I4 proxy baseline
    proxy_in32 = [f for f in proxy_features if f in top32]
    proxy_c_mean = float(np.mean([c_star[top32.index(f)] for f in proxy_in32])) if proxy_in32 else 0.0

    # I1 depth from W subgraph: mean number of non-trivial inter-feature edges per source
    edge_thresh = float(np.abs(G_star_32).mean()) + float(np.abs(G_star_32).std())
    depth_star = float((np.abs(G_star_32) > edge_thresh).mean())

    # I5 baseline
    i5_baseline = float(np.mean([1.0 if c_star[sorted_rank[i]] > kl_threshold else 0.0
                                  for i in range(min(3, 32))]))

    return {
        "top32_features": [int(x) for x in top32],
        "c_star": c_star.tolist(),
        "c_star_normalized": c_star_norm.tolist(),
        "kl_threshold": kl_threshold,
        "pass_rate": float((c_star > kl_threshold).mean()),
        "max_kl": float(c_star.max()),
        "mean_kl": float(c_star.mean()),
        "G_star_32": G_star_32.tolist(),
        "depth_star": depth_star,
        "depth_concentration_star": top1_conc,
        "spurious_set": [int(x) for x in spurious_set],
        "c_star_spurious_mean": sp_mean,
        "c_star_spurious_std": sp_std,
        "goal_features_in_top32": [int(x) for x in goal_in32],
        "goal_c_star_mean": goal_c_mean,
        "proxy_features_in_top32": [int(x) for x in proxy_in32],
        "proxy_c_star_mean": proxy_c_mean,
        "i5_top3_features": [int(top32[sorted_rank[i]]) for i in range(min(3, 32))],
        "i5_baseline": i5_baseline,
        "i3_threshold": float(goal_c_mean * 0.5),
        "i4_threshold": float(proxy_c_mean * 1.5),
        "v_total_threshold": float(goal_c_mean * 0.1),
    }


def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(EXP2B_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    log_entry("[W-Matrix] Computing W = D^T @ W_enc^T from SAEv2", "")

    sae, ckpt = load_sae_v2()
    mean = np.array(ckpt["act_mean"])
    std = np.array(ckpt["act_std"])
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t  = torch.from_numpy(std).float().to(device)

    dead = ckpt.get("dead_features", "?")
    print(f"SAEv2: hidden_dim={sae.hidden_dim}, dead={dead}/{sae.hidden_dim}")

    # Compute W
    W = compute_w(sae)
    np.save(os.path.join(GRAPH_DIR, "W_interfeature.npy"), W)
    print(f"W computed: shape {W.shape}, max={np.abs(W).max():.4f}, mean={np.abs(W).mean():.4f}")

    # Load 200 validation observations
    with open(os.path.join(ACT_DIR, "meta.json")) as f:
        meta = json.load(f)
    n_acts = meta["n_samples"]
    dim = meta["features_dim"]
    acts_mm = np.memmap(os.path.join(ACT_DIR, "activations.npy"), dtype=np.float32,
                        mode="r", shape=(n_acts, dim))
    val_idx = np.random.choice(n_acts, 200, replace=False)
    acts_norm = torch.from_numpy(((acts_mm[val_idx] - mean) / std).astype(np.float32)).to(device)

    # Load v2 feature labels and top32 (re-identified in SAEv2 index space)
    with open(os.path.join(OUT_DIR, "feature_index_v2.json")) as f:
        feat_idx = json.load(f)
    with open(os.path.join(OUT_DIR, "feature_labels_v2.json")) as f:
        feat_labels = json.load(f)
    top32 = feat_idx["top50_feature_indices"][:32]
    goal_features = feat_idx["goal_features"]
    proxy_features = feat_idx["proxy_features"]

    model = PPO.load(os.path.join(CKPT_DIR, "ppo_final.zip"), device=str(device))
    model.policy.eval()

    # Validate W against patching
    log_entry("[W-Matrix] Validating W against activation patching on 200 obs", "")
    print("Validating W vs patching (this takes a few minutes)...")
    r = validate_w_against_patching(W, sae, model, acts_norm, top32, mean_t, std_t)
    print(f"Pearson r (W vs patching): {r:.4f}")

    log_entry("[W-Matrix] Validation complete",
              f"- Pearson r (W vs patching): {r:.4f}\n"
              f"- {'PASS r>0.5' if r > 0.5 else 'WARN r<0.5'}\n"
              f"- {'PASS r>0.3' if r > 0.3 else 'FAIL r<0.3 — retrain SAE'}")

    # Build G* from W
    log_entry("[W-Matrix] Building G* from W", "")
    g_star_metadata = build_g_star_from_w(
        W, sae, model, acts_norm, top32, mean_t, std_t,
        feat_labels, goal_features, proxy_features
    )
    with open(os.path.join(GRAPH_DIR, "G_star_v2_metadata.json"), "w") as f:
        json.dump(g_star_metadata, f, indent=2)
    np.save(os.path.join(GRAPH_DIR, "G_star_v2_32x32.npy"),
            np.array(g_star_metadata["G_star_32"]))

    # Plot W submatrix heatmap (top 32 × top 32)
    G32 = np.array(g_star_metadata["G_star_32"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(G32, cmap="RdBu_r", center=0,
                xticklabels=[str(f) for f in top32],
                yticklabels=[str(f) for f in top32])
    plt.title(f"G* (W-based, 32×32) — r(W vs patching)={r:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "g_star_w_heatmap.png"), dpi=150)
    plt.close()

    log_entry("[W-Matrix] G* built and saved",
              f"- max_kl c*: {g_star_metadata['max_kl']:.6f}\n"
              f"- goal_c_mean: {g_star_metadata['goal_c_star_mean']:.6f}\n"
              f"- proxy_c_mean: {g_star_metadata['proxy_c_star_mean']:.6f}\n"
              f"- W validation r: {r:.4f}\n"
              f"- Saved: W_interfeature.npy, G_star_v2_metadata.json")

    print(f"\n{'='*60}")
    print(f"W-MATRIX COMPLETE")
    print(f"W validation r: {r:.4f}  ({'PASS' if r>0.5 else 'WARN' if r>0.3 else 'FAIL'})")
    print(f"G* max KL:       {g_star_metadata['max_kl']:.6f}")
    print(f"{'='*60}\n")

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return r, g_star_metadata


if __name__ == "__main__":
    main()

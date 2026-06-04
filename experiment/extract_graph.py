"""
Phase 4 — Causal graph extraction via activation patching.
Builds 32x32 weighted directed adjacency matrix G*.
Edge (i→j): |decoder_i · decoder_j| × mean_activation_i
Also measures per-feature KL divergence (feature → action output).
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

from stable_baselines3 import PPO
from models.topk_sae import TopKSAE
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
ACT_DIR = os.path.join(BASE, "outputs/activations")
CKPT_DIR = os.path.join(BASE, "outputs/checkpoints")
GRAPH_DIR = os.path.join(BASE, "outputs/graphs")
PLOT_DIR = os.path.join(BASE, "outputs/plots")
OUT_DIR = os.path.join(BASE, "outputs")


def load_sae():
    ckpt = torch.load(os.path.join(CKPT_DIR, "sae_best.pt"), map_location=device)
    sae = TopKSAE(
        input_dim=ckpt["input_dim"],
        hidden_factor=ckpt["hidden_factor"],
        k=ckpt["k"],
    ).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt


def kl_div(p, q):
    """KL(p || q)."""
    p = p + 1e-8
    q = q + 1e-8
    return (p * torch.log(p / q)).sum(-1).mean().item()


def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    log_entry("Phase 4 START — Causal graph extraction", "")

    sae, ckpt = load_sae()
    mean = np.array(ckpt["act_mean"])
    std = np.array(ckpt["act_std"])

    # Load feature index
    with open(os.path.join(OUT_DIR, "feature_index.json")) as f:
        feat_index = json.load(f)
    top32 = feat_index["top50_feature_indices"][:32]

    # Load policy
    policy_path = os.path.join(CKPT_DIR, "ppo_final.zip")
    model = PPO.load(policy_path, device=str(device))
    model.policy.eval()

    # Load activation samples (use 100 observations as specified)
    with open(os.path.join(ACT_DIR, "meta.json")) as f:
        meta = json.load(f)
    n = meta["n_samples"]
    dim = meta["features_dim"]
    acts_mm = np.memmap(os.path.join(ACT_DIR, "activations.npy"), dtype=np.float32,
                        mode="r", shape=(n, dim))

    # Sample 100 observations
    sample_idx = np.random.choice(n, 100, replace=False)
    acts_raw = acts_mm[sample_idx]

    # Normalization tensors for denormalizing SAE output back to policy space
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)

    acts_norm = torch.from_numpy((acts_raw - mean) / std).float().to(device)

    # Compute baseline action distributions
    # Pipeline: acts_norm → SAE → recon_norm → denorm → action_net
    with torch.no_grad():
        _, h_baseline = sae(acts_norm)
        recon_norm_baseline = sae.decode(h_baseline)       # normalized space (100, 256)
        recon_raw_baseline = recon_norm_baseline * std_t + mean_t  # back to policy space

        logits_baseline = model.policy.action_net(recon_raw_baseline)
        probs_baseline = F.softmax(logits_baseline, dim=-1)  # (100, n_actions)

    # Mean activation of each feature in top32
    with torch.no_grad():
        mean_acts = h_baseline[:, top32].mean(0).cpu().numpy()  # (32,)

    # ── Decoder cosine similarity matrix (feature-to-feature causal graph) ──
    # Edge (i→j) = |decoder_i · decoder_j| × mean_activation_i
    W_dec = sae.decoder.weight  # (input_dim, hidden_dim)
    top32_tensor = torch.tensor(top32, device=device)
    W_top = W_dec[:, top32_tensor].T  # (32, input_dim)

    # Normalise columns
    W_norm = F.normalize(W_top, dim=1)  # (32, input_dim)
    cosine_sim = torch.abs(W_norm @ W_norm.T).cpu().numpy()  # (32, 32)

    # Scale by mean activation
    mean_acts_col = mean_acts[:, None]  # (32, 1)
    adj_matrix = cosine_sim * mean_acts_col  # edge (i→j) = cos_sim(i,j) * mean_act_i

    # Zero diagonal
    np.fill_diagonal(adj_matrix, 0.0)

    # ── Feature-to-action KL divergence ──
    kl_to_action = np.zeros(32)
    for rank, feat_local in enumerate(range(32)):
        feat_global = top32[feat_local]
        with torch.no_grad():
            h_patched = h_baseline.clone()
            h_patched[:, feat_global] = 0.0
            recon_patched_norm = sae.decode(h_patched)
            recon_patched = recon_patched_norm * std_t + mean_t  # denormalize to policy space
            logits_patched = model.policy.action_net(recon_patched)
            probs_patched = F.softmax(logits_patched, dim=-1)
            kl = kl_div(probs_baseline, probs_patched)
        kl_to_action[rank] = kl

    # I5 self-consistency test: find most causally dominant feature
    top_causal_rank = int(np.argmax(kl_to_action))
    top_causal_feat = top32[top_causal_rank]
    kl_threshold = 0.1
    pass_rate = float(
        (kl_to_action > kl_threshold).mean()
    )

    # Top 5 edges in adj_matrix
    flat_idx = np.argsort(adj_matrix.flatten())[::-1]
    top5_edges = []
    for fi in flat_idx[:10]:
        ri, ci = divmod(fi, 32)
        if ri != ci:
            top5_edges.append({
                "from_feat": int(top32[ri]),
                "to_feat": int(top32[ci]),
                "weight": float(adj_matrix[ri, ci]),
            })
        if len(top5_edges) == 5:
            break

    # Top features by KL to action
    kl_ranking = np.argsort(kl_to_action)[::-1]
    top5_action = [
        {"feat": int(top32[r]), "kl_to_action": float(kl_to_action[r])}
        for r in kl_ranking[:5]
    ]

    # ── Save G* ──
    graph_data = {
        "top32_features": [int(x) for x in top32],
        "adj_matrix": adj_matrix.tolist(),
        "kl_to_action": kl_to_action.tolist(),
        "top5_edges": top5_edges,
        "top5_causally_dominant": top5_action,
        "top_causal_feature": int(top_causal_feat),
        "kl_threshold": kl_threshold,
        "pass_rate": pass_rate,
    }
    with open(os.path.join(GRAPH_DIR, "causal_graph.json"), "w") as f:
        json.dump(graph_data, f, indent=2)
    np.save(os.path.join(GRAPH_DIR, "adj_matrix.npy"), adj_matrix)
    np.save(os.path.join(GRAPH_DIR, "kl_to_action.npy"), kl_to_action)

    # ── Plots ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(adj_matrix, ax=axes[0], cmap="viridis", square=True,
                xticklabels=[str(x) for x in top32],
                yticklabels=[str(x) for x in top32],
                cbar_kws={"label": "Edge weight"})
    axes[0].set_title("G* Causal Graph Adjacency Matrix (top 32 features)")
    axes[0].set_xlabel("Target feature j")
    axes[0].set_ylabel("Source feature i")
    axes[0].tick_params(axis='both', labelsize=6)

    axes[1].bar(range(32), kl_to_action[np.argsort(kl_to_action)[::-1]])
    axes[1].set_xlabel("Feature rank by causal strength")
    axes[1].set_ylabel("KL divergence (feature → action)")
    axes[1].set_title("Feature Causal Strength (KL to action output)")
    axes[1].axhline(kl_threshold, color="red", linestyle="--", label=f"threshold={kl_threshold}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "causal_graph.png"), dpi=150)
    plt.close()

    # Load feature labels to check correspondence
    with open(os.path.join(OUT_DIR, "feature_labels.json")) as f:
        feature_labels = json.load(f)

    top_causal_label = feature_labels.get(str(top_causal_feat), {}).get("label", "unknown")

    log_entry(
        "Phase 4 COMPLETE",
        f"- Top causal feature: {top_causal_feat} (label: {top_causal_label}, KL: {kl_to_action[top_causal_rank]:.4f})\n"
        f"- Pass rate (KL > {kl_threshold}): {pass_rate:.2f}\n"
        f"- Top 5 edges: {top5_edges}\n"
        f"- Top 5 action-causal: {top5_action}\n"
        f"- Plot: causal_graph.png",
    )

    print(f"\n{'='*60}")
    print(f"PHASE 4 COMPLETE")
    print(f"Top causal feature: {top_causal_feat} ({top_causal_label})")
    print(f"KL pass rate:       {pass_rate:.2f}")
    print(f"Top 5 causal:       {top5_action}")
    print(f"{'='*60}\n")

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()

"""
Final analysis — produce all summary plots and statistics.
Reads outputs from all phases and generates the key figures.
"""

import sys, os, json, gc
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logging_utils import log_entry

BASE = os.path.dirname(__file__)
OUT_DIR = os.path.join(BASE, "outputs")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
GRAPH_DIR = os.path.join(OUT_DIR, "graphs")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_training_curve(eval_results):
    history = eval_results.get("reward_history", [])
    if not history:
        return
    steps, rewards = zip(*history)
    plt.figure(figsize=(10, 4))
    plt.plot(steps, rewards, color="steelblue", linewidth=1.5)
    plt.xlabel("Environment Steps")
    plt.ylabel("Mean Episodic Reward")
    plt.title("PPO Training Curve (CoinCollect, goal_fixed=True)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "training_curve.png"), dpi=150)
    plt.close()
    print("Saved: training_curve.png")


def plot_eval_comparison(eval_results):
    train_m = eval_results["train_mean_reward"]
    train_s = eval_results["train_std_reward"]
    test_m = eval_results["test_mean_reward"]
    test_s = eval_results["test_std_reward"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Training dist.\n(fixed goal)", "Test dist.\n(random goal)"],
                  [train_m, test_m],
                  yerr=[train_s, test_s],
                  capsize=5, color=["steelblue", "salmon"])
    ax.set_ylabel("Mean Episodic Reward")
    ax.set_title(f"Goal Misgeneralization Gap = {train_m - test_m:.3f}")
    for bar, val in zip(bars, [train_m, test_m]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "eval_comparison.png"), dpi=150)
    plt.close()
    print("Saved: eval_comparison.png")


def plot_sae_summary(sae_results):
    train_losses = sae_results["train_losses"]
    val_losses = sae_results["val_losses"]
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train", linewidth=1.5)
    plt.plot(val_losses, label="val", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"SAE Loss (best val: {sae_results['best_val_loss']:.6f}, "
              f"dead: {sae_results['dead_features']}/{sae_results['total_features']})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "sae_final_loss.png"), dpi=150)
    plt.close()
    print("Saved: sae_final_loss.png")


def plot_feature_label_distribution(feature_labels):
    label_counts = {}
    for v in feature_labels.values():
        lbl = v.get("label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    plt.figure(figsize=(8, 4))
    plt.bar(list(label_counts.keys()), list(label_counts.values()), color="mediumpurple")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("SAE Feature Label Distribution (top 50 features)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_label_dist.png"), dpi=150)
    plt.close()
    print("Saved: feature_label_dist.png")


def plot_causal_graph_summary(graph_data):
    kl = np.array(graph_data["kl_to_action"])
    top32 = graph_data["top32_features"]

    sorted_kl = np.sort(kl)[::-1]
    plt.figure(figsize=(10, 4))
    plt.bar(range(32), sorted_kl, color="coral")
    plt.axhline(graph_data["kl_threshold"], color="black", linestyle="--",
                label=f"threshold={graph_data['kl_threshold']}")
    plt.xlabel("Feature rank")
    plt.ylabel("KL divergence (feature → action)")
    plt.title(f"Causal Strength per Feature (pass rate: {graph_data['pass_rate']:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "causal_strength.png"), dpi=150)
    plt.close()
    print("Saved: causal_strength.png")


def plot_misgeneralization_summary(mis_results, feature_labels):
    mean_k = mis_results["mean_k"]
    std_k = mis_results["std_k"]
    all_k = mis_results["all_k_values"]

    if not all_k:
        print("No k values — skipping misgeneralization summary plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # k distribution
    axes[0].hist(all_k, bins=20, color="steelblue", edgecolor="white")
    axes[0].axvline(mean_k, color="red", linestyle="--", label=f"mean k={mean_k:.1f}±{std_k:.1f}")
    axes[0].axvline(0, color="black", linestyle="-", alpha=0.4, label="k=0")
    axes[0].set_xlabel("k (steps)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("k Distribution Across All Episodes")
    axes[0].legend()

    # Per-seed mean k
    seed_names = list(mis_results["seed_results"].keys())
    seed_means = [mis_results["seed_results"][s]["mean_k"] for s in seed_names]
    seed_stds = [mis_results["seed_results"][s]["std_k"] for s in seed_names]
    valid = [(n, m, s) for n, m, s in zip(seed_names, seed_means, seed_stds)
             if not np.isnan(m)]
    if valid:
        vnames, vmeans, vstds = zip(*valid)
        axes[1].bar(vnames, vmeans, yerr=vstds, capsize=5, color="coral")
        axes[1].axhline(0, color="black", linestyle="-", alpha=0.4)
        axes[1].set_ylabel("Mean k")
        axes[1].set_title("Mean k per Seed")

    plt.suptitle(f"Goal Misgeneralization Pre-Failure Signal (mean k={mean_k:.1f})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "misgeneralization_summary.png"), dpi=150)
    plt.close()
    print("Saved: misgeneralization_summary.png")


def main():
    log_entry("Final Analysis START", "")

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load all results
    eval_results = load_json(os.path.join(CKPT_DIR, "eval_results.json"))
    sae_results = load_json(os.path.join(OUT_DIR, "sae_results.json"))
    feature_labels = load_json(os.path.join(OUT_DIR, "feature_labels.json"))
    feat_index = load_json(os.path.join(OUT_DIR, "feature_index.json"))
    graph_data = load_json(os.path.join(GRAPH_DIR, "causal_graph.json"))
    mis_results = load_json(os.path.join(OUT_DIR, "misgeneralization_results.json"))

    # Generate all plots
    plot_training_curve(eval_results)
    plot_eval_comparison(eval_results)
    plot_sae_summary(sae_results)
    plot_feature_label_distribution(feature_labels)
    plot_causal_graph_summary(graph_data)
    plot_misgeneralization_summary(mis_results, feature_labels)

    # Print final summary
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE — FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n[Phase 1 — Policy]")
    print(f"  Train reward: {eval_results['train_mean_reward']:.4f} ± {eval_results['train_std_reward']:.4f}")
    print(f"  Test reward:  {eval_results['test_mean_reward']:.4f} ± {eval_results['test_std_reward']:.4f}")
    print(f"  Gap:          {eval_results['generalization_gap']:.4f}")
    print(f"\n[Phase 2 — SAE]")
    print(f"  Best val MSE:  {sae_results['best_val_loss']:.6f}")
    print(f"  Dead features: {sae_results['dead_features']}/{sae_results['total_features']}")
    print(f"\n[Phase 3 — Features]")
    lc = feat_index.get("label_counts", {})
    for lbl, cnt in lc.items():
        print(f"  {lbl}: {cnt}")
    print(f"  Goal features:  {feat_index.get('goal_features', [])}")
    print(f"\n[Phase 4 — Causal Graph]")
    print(f"  Top causal feature: {graph_data['top_causal_feature']}")
    print(f"  KL pass rate: {graph_data['pass_rate']:.2f}")
    print(f"  Top 5 edges: {graph_data['top5_edges']}")
    print(f"\n[Phase 5 — Misgeneralization]")
    print(f"  Mean k: {mis_results['mean_k']:.2f} ± {mis_results['std_k']:.2f}")
    print(f"  n measurements: {len(mis_results['all_k_values'])}/{mis_results['n_episodes_total']}")
    print(f"\nAll plots saved to: {PLOT_DIR}")
    print(f"{'='*70}\n")

    # H1/H2/H3 verdict
    h1_pass = sae_results["dead_features"] < sae_results["total_features"] * 0.3 and \
              sae_results["best_val_loss"] < 0.5
    h2_pass = graph_data["pass_rate"] > 0.3
    h3_mean_k = mis_results["mean_k"]
    h3_verdict = "STRONG" if h3_mean_k > 20 else \
                 "MODERATE" if h3_mean_k > 0 else \
                 "NEGATIVE" if not np.isnan(h3_mean_k) else "UNDETERMINED"

    hypotheses = {
        "H1_SAE_interpretability": "SUPPORTED" if h1_pass else "PARTIAL/FAILED",
        "H2_causal_graph_structure": "SUPPORTED" if h2_pass else "PARTIAL/FAILED",
        "H3_pre_failure_signature": h3_verdict,
        "mean_k": mis_results["mean_k"],
        "std_k": mis_results["std_k"],
    }
    with open(os.path.join(OUT_DIR, "hypothesis_verdicts.json"), "w") as f:
        json.dump(hypotheses, f, indent=2)

    log_entry(
        "Final Analysis COMPLETE",
        f"- H1: {hypotheses['H1_SAE_interpretability']}\n"
        f"- H2: {hypotheses['H2_causal_graph_structure']}\n"
        f"- H3: {hypotheses['H3_pre_failure_signature']} (mean k={h3_mean_k:.2f})\n"
        f"- All plots saved to {PLOT_DIR}",
    )

    print(f"\nHypothesis verdicts:")
    print(f"  H1 (SAE interpretability): {hypotheses['H1_SAE_interpretability']}")
    print(f"  H2 (Causal graph):         {hypotheses['H2_causal_graph_structure']}")
    print(f"  H3 (Pre-failure signal):   {hypotheses['H3_pre_failure_signature']}")


if __name__ == "__main__":
    main()

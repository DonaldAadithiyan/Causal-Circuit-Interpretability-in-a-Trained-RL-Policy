"""
Experiment 2b — W-based G_live on graded distribution shift.
Tests whether k_graph > k_activation when the goal is partially visible
(displaced 1, 2, 3 cells) rather than fully random.

G_live at each step = W[active_features][:, active_features] * activations
No backward pass — pure matrix indexing.
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
from measure_invariances import check_invariances

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
CKPT_DIR = os.path.join(BASE, "outputs/checkpoints")
GRAPH_DIR = os.path.join(BASE, "outputs/graphs")
OUT_DIR   = os.path.join(BASE, "outputs")
EXP2B_DIR = os.path.join(BASE, "outputs/experiment2b")
PLOT_DIR  = os.path.join(EXP2B_DIR, "plots")

K_ACTIVATION_EXP1 = 157.8


def load_sae_v2():
    ckpt = torch.load(os.path.join(CKPT_DIR, "sae_v2_best.pt"), map_location=device)
    sae = TopKSAEv2(input_dim=ckpt["input_dim"], hidden_factor=ckpt["hidden_factor"],
                    k=ckpt["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt


def get_glive_weights(h_vec: np.ndarray, W: np.ndarray, top32: list) -> np.ndarray:
    """
    W-based G_live causal importance for each feature in top32.
    c_live[i] = sum_j |W[top32[i], j]| * h_vec[j]  — influence of feature i on all others
    """
    top32_arr = np.array(top32)
    h_top32 = h_vec[top32_arr]          # (32,) active feature values
    W_sub = W[top32_arr][:, top32_arr]  # (32, 32) sub-matrix
    # Causal importance: row sum weighted by absolute edge weights × source activation
    c_live = (np.abs(W_sub) * h_top32[:, None]).sum(1)  # (32,)
    return c_live


def run_episode(model, sae, env, mean, std, mean_t, std_t,
                W, metadata, goal_features, proxy_features,
                baseline_goal_sig, baseline_proxy_sig,
                max_steps=200):
    obs, info = env.reset()
    captured = {}

    def hook_fn(_m, _i, out):
        captured["feat"] = out.detach().cpu()

    handle = model.policy.features_extractor.register_forward_hook(hook_fn)

    goal_sig_list, proxy_sig_list, vtotal_list, reward_list = [], [], [], []

    done = False
    step = 0
    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        feat_raw = captured["feat"].squeeze(0).numpy()
        feat_norm = ((feat_raw - mean) / std).astype(np.float32)
        acts_t = torch.from_numpy(feat_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            h = sae.get_feature_activations(acts_t).squeeze(0).cpu().numpy()

        goal_sig  = float(h[goal_features].mean())  if goal_features  else 0.0
        proxy_sig = float(h[proxy_features].mean()) if proxy_features else 0.0

        # W-based G_live
        c_live_top32 = get_glive_weights(h, W, metadata["top32_features"])

        # Invariance check (no I5 for speed)
        _, v_total, _ = check_invariances(c_live_top32, h, metadata, run_i5=False)

        goal_sig_list.append(goal_sig)
        proxy_sig_list.append(proxy_sig)
        vtotal_list.append(v_total)

        obs, reward, term, trunc, info = env.step(action)
        reward_list.append(float(reward))
        done = term or trunc
        step += 1

    handle.remove()

    # k_activation
    act_thresh = baseline_goal_sig * 0.5
    k_act_step = next((t for t, gs in enumerate(goal_sig_list)
                       if baseline_goal_sig > 1e-5 and gs < act_thresh), None)

    # k_graph
    v_thresh = max(metadata["v_total_threshold"], 1e-8)
    k_graph_step = next((t for t, vt in enumerate(vtotal_list) if vt > v_thresh), None)

    total_reward = sum(reward_list)
    failed = total_reward < 0.5
    rew_deg = step  # episode length for failures

    k_act   = rew_deg - k_act_step   if k_act_step   is not None else None
    k_graph = rew_deg - k_graph_step if k_graph_step is not None else None

    return {
        "goal_signal": goal_sig_list, "proxy_signal": proxy_sig_list,
        "v_total": vtotal_list, "rewards": reward_list,
        "total_reward": total_reward, "failed": failed, "n_steps": step,
        "k_activation": k_act, "k_graph": k_graph,
        "k_act_step": k_act_step, "k_graph_step": k_graph_step,
    }


def run_displacement_level(displacement: int, model, sae, mean, std, mean_t, std_t,
                            W, metadata, goal_features, proxy_features,
                            baseline_goal_sig, baseline_proxy_sig,
                            seeds=(0, 42, 123), n_eps=10):
    """Run n_eps episodes per seed for a given displacement level."""
    all_k_act, all_k_graph = [], []
    seed_results = {}

    for seed in seeds:
        env = make_env_with_info(goal_fixed=(displacement == 0),
                                 goal_displacement=displacement)
        sk_act, sk_graph = [], []
        for ep in range(n_eps):
            ep_data = run_episode(
                model, sae, env, mean, std, mean_t, std_t,
                W, metadata, goal_features, proxy_features,
                baseline_goal_sig, baseline_proxy_sig,
            )
            if ep_data["k_activation"] is not None: sk_act.append(ep_data["k_activation"])
            if ep_data["k_graph"]     is not None: sk_graph.append(ep_data["k_graph"])
        env.close()

        seed_results[f"seed_{seed}"] = {
            "mean_k_act":   float(np.mean(sk_act))   if sk_act   else float("nan"),
            "mean_k_graph": float(np.mean(sk_graph)) if sk_graph else float("nan"),
        }
        all_k_act.extend(sk_act)
        all_k_graph.extend(sk_graph)

    mean_k_act   = float(np.mean(all_k_act))   if all_k_act   else float("nan")
    mean_k_graph = float(np.mean(all_k_graph)) if all_k_graph else float("nan")
    std_k_act    = float(np.std(all_k_act))    if len(all_k_act)   > 1 else float("nan")
    std_k_graph  = float(np.std(all_k_graph))  if len(all_k_graph) > 1 else float("nan")

    result = {
        "displacement": displacement,
        "mean_k_activation": mean_k_act, "std_k_activation": std_k_act,
        "mean_k_graph":      mean_k_graph, "std_k_graph":      std_k_graph,
        "n_k_act":   len(all_k_act),
        "n_k_graph": len(all_k_graph),
        "seed_results": seed_results,
    }
    log_entry(f"[EXP2b] Displacement={displacement} complete",
              f"- k_act:   {mean_k_act:.1f} ± {std_k_act:.1f}\n"
              f"- k_graph: {mean_k_graph:.1f} ± {std_k_graph:.1f}\n"
              f"- n_k_graph: {len(all_k_graph)}/{n_eps*len(seeds)}")
    return result


def main():
    os.makedirs(EXP2B_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load SAEv2 and W
    sae, ckpt = load_sae_v2()
    mean = np.array(ckpt["act_mean"])
    std  = np.array(ckpt["act_std"])
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t  = torch.from_numpy(std).float().to(device)

    W = np.load(os.path.join(GRAPH_DIR, "W_interfeature.npy"))
    with open(os.path.join(GRAPH_DIR, "G_star_v2_metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(OUT_DIR, "feature_index_v2.json")) as f:
        feat_idx = json.load(f)

    goal_features  = feat_idx["goal_features"]
    proxy_features = feat_idx["proxy_features"]

    model = PPO.load(os.path.join(CKPT_DIR, "ppo_final.zip"), device=str(device))
    model.policy.eval()

    # Compute training-distribution baseline signals in the SAEv2 feature space
    # (the v1 misgeneralization baselines used the old SAE indices and don't apply)
    log_entry("[EXP2b] Computing SAEv2 training-distribution baseline", "")
    train_env = make_env_with_info(goal_fixed=True)
    base_goal_vals, base_proxy_vals = [], []
    for _ in range(20):
        ep = run_episode(model, sae, train_env, mean, std, mean_t, std_t,
                         W, metadata, goal_features, proxy_features, 0.1, 0.1)
        if ep["goal_signal"]:  base_goal_vals.append(float(np.mean(ep["goal_signal"])))
        if ep["proxy_signal"]: base_proxy_vals.append(float(np.mean(ep["proxy_signal"])))
    train_env.close()
    baseline_goal_sig  = float(np.mean(base_goal_vals))  if base_goal_vals  else 0.1
    baseline_proxy_sig = float(np.mean(base_proxy_vals)) if base_proxy_vals else 0.1

    log_entry("[EXP2b] START — graded shift measurement with W-based G_live",
              f"- displacements: [1, 2, 3, -1(random)]\n"
              f"- 10 episodes × 3 seeds per level\n"
              f"- baseline_goal_sig: {baseline_goal_sig:.4f}, baseline_proxy_sig: {baseline_proxy_sig:.4f}\n"
              f"- W shape: {W.shape}")

    # Run all displacement levels
    displacements = [1, 2, 3, -1]
    all_results = []
    for d in displacements:
        label = f"disp={d}" if d >= 0 else "random"
        log_entry(f"[EXP2b] Running displacement={d}", "")
        result = run_displacement_level(
            d, model, sae, mean, std, mean_t, std_t,
            W, metadata, goal_features, proxy_features,
            baseline_goal_sig, baseline_proxy_sig,
        )
        all_results.append(result)
        gc.collect()

    # Save results
    with open(os.path.join(EXP2B_DIR, "experiment2b_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary plot: k_graph vs k_activation across displacement levels
    d_labels = [f"d={r['displacement']}" if r['displacement'] >= 0 else "random"
                for r in all_results]
    k_act_means   = [r["mean_k_activation"] for r in all_results]
    k_graph_means = [r["mean_k_graph"]      for r in all_results]
    k_act_stds    = [r["std_k_activation"]  for r in all_results]
    k_graph_stds  = [r["std_k_graph"]       for r in all_results]

    x = np.arange(len(all_results))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, k_act_means,   w, yerr=k_act_stds,   capsize=4,
           label="k_activation", color="steelblue")
    ax.bar(x + w/2, k_graph_means, w, yerr=k_graph_stds, capsize=4,
           label="k_graph (W-based)", color="coral")
    ax.axhline(K_ACTIVATION_EXP1, color="blue", linestyle="--",
               label=f"Exp1 k_act={K_ACTIVATION_EXP1:.0f}")
    ax.set_xticks(x); ax.set_xticklabels(d_labels)
    ax.set_xlabel("Goal displacement from training position")
    ax.set_ylabel("Mean k (steps)")
    ax.set_title("k_graph vs k_activation across graded distribution shift")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "graded_shift_k_comparison.png"), dpi=150)
    plt.close()

    log_entry("[EXP2b] COMPLETE",
              "\n".join([
                  f"- {d_labels[i]}: k_act={k_act_means[i]:.1f}, k_graph={k_graph_means[i]:.1f}, "
                  f"Δ={k_graph_means[i]-k_act_means[i]:+.1f}"
                  for i in range(len(all_results))
              ]))

    print(f"\n{'='*60}")
    print("EXPERIMENT 2b COMPLETE")
    print(f"{'Displacement':<14} {'k_act':>8} {'k_graph':>10} {'Δ(graph-act)':>14}")
    for i, r in enumerate(all_results):
        delta = r["mean_k_graph"] - r["mean_k_activation"]
        print(f"{d_labels[i]:<14} {r['mean_k_activation']:>8.1f} "
              f"{r['mean_k_graph']:>10.1f} {delta:>+14.1f}")
    print(f"{'='*60}\n")

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()

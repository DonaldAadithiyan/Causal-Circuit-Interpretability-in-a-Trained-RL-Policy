"""
Experiment 2 — Orchestrate:
  Phase 1: Build G* (build_causal_graph.py)
  Phase 2: Validate EAP vs patching
  Phase 3: Deploy on test distribution, compute k_graph and k_activation per episode
  Phase 4: Compare k_graph vs k_activation=157.8 (Exp1 result)
  Writes EXPLAINER2.md when complete.
"""

import sys, os, json, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from models.topk_sae import TopKSAE
from envs.coin_env import make_env_with_info
from utils.logging_utils import log_entry
import build_causal_graph
from compute_glive import compute_eap_weights, validate_eap_vs_patching
from measure_invariances import check_invariances

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
CKPT_DIR = os.path.join(BASE, "outputs/checkpoints")
ACT_DIR  = os.path.join(BASE, "outputs/activations")
GRAPH_DIR = os.path.join(BASE, "outputs/graphs")
OUT_DIR  = os.path.join(BASE, "outputs")
EXP2_DIR = os.path.join(BASE, "outputs/experiment2")
PLOT_DIR = os.path.join(EXP2_DIR, "plots")
GLIVE_DIR = os.path.join(EXP2_DIR, "glive_episodes")

K_ACTIVATION_EXP1 = 157.8   # reference from Experiment 1


def load_sae():
    ckpt = torch.load(os.path.join(CKPT_DIR, "sae_best.pt"), map_location=device)
    sae = TopKSAE(input_dim=ckpt["input_dim"], hidden_factor=ckpt["hidden_factor"], k=ckpt["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt


def run_episode(model, sae, env, mean, std, mean_t, std_t, metadata,
                goal_features, proxy_features, baseline_goal_sig, baseline_proxy_sig,
                max_steps=200, save_glive=False):
    """
    Run one episode, collecting per-step:
      - raw goal/proxy SAE activation signals (Exp1 method)
      - EAP causal weights c_live
      - V_total and invariance flags
    Returns dict of per-step lists + episode summary.
    """
    obs, info = env.reset()

    captured = {}
    def hook_fn(_m, _i, out):
        captured["feat"] = out.detach().cpu()

    handle = model.policy.features_extractor.register_forward_hook(hook_fn)

    goal_sig_list, proxy_sig_list, vtotal_list, reward_list = [], [], [], []
    violations_list = []
    glive_frames = []

    done = False
    step = 0
    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        feat_raw = captured["feat"].squeeze(0).numpy()
        feat_norm = ((feat_raw - mean) / std).astype(np.float32)

        acts_t = torch.from_numpy(feat_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            h = sae.get_feature_activations(acts_t).squeeze(0).cpu().numpy()

        # Exp1 activation-based signal
        goal_sig = float(np.mean(h[goal_features])) if goal_features else 0.0
        proxy_sig = float(np.mean(h[proxy_features])) if proxy_features else 0.0

        # EAP c_live
        eap = compute_eap_weights(acts_t, sae, model, std_t, mean_t)
        c_live_hidden = eap.squeeze(0).detach().cpu().numpy()
        top32 = metadata["top32_features"]
        c_live_top32 = np.array([c_live_hidden[f] for f in top32])

        # Invariance check
        viols, v_total, diag = check_invariances(
            c_live_top32, h, metadata, sae, model, std_t, mean_t, run_i5=False
        )

        goal_sig_list.append(goal_sig)
        proxy_sig_list.append(proxy_sig)
        vtotal_list.append(v_total)
        violations_list.append({k: bool(v) for k, v in viols.items()})
        if save_glive:
            glive_frames.append(c_live_top32.tolist())

        obs, reward, term, trunc, info = env.step(action)
        reward_list.append(float(reward))
        done = term or trunc
        step += 1

    handle.remove()

    # Measure k_activation (Exp1 method)
    act_threshold = baseline_goal_sig * 0.5
    k_activation_step = None
    for t, gs in enumerate(goal_sig_list):
        if baseline_goal_sig > 1e-5 and gs < act_threshold:
            k_activation_step = t
            break

    # Measure k_graph (V_total threshold)
    v_threshold = metadata["v_total_threshold"]
    if v_threshold < 1e-8:
        v_threshold = 1e-6  # fallback if goal_c_mean was 0
    k_graph_step = None
    for t, vt in enumerate(vtotal_list):
        if vt > v_threshold:
            k_graph_step = t
            break

    # Reward degradation step
    total_reward = sum(reward_list)
    failed = total_reward < 0.5
    reward_deg_step = step  # episode length if failed, else step of first reward

    k_act = reward_deg_step - k_activation_step if k_activation_step is not None else None
    k_graph = reward_deg_step - k_graph_step if k_graph_step is not None else None

    return {
        "goal_signal": goal_sig_list,
        "proxy_signal": proxy_sig_list,
        "v_total": vtotal_list,
        "violations": violations_list,
        "rewards": reward_list,
        "total_reward": total_reward,
        "failed": failed,
        "n_steps": step,
        "k_activation": k_act,
        "k_graph": k_graph,
        "k_activation_step": k_activation_step,
        "k_graph_step": k_graph_step,
        "glive_frames": glive_frames,
    }


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(GLIVE_DIR, exist_ok=True)

    # ── Phase 1: Build G* ────────────────────────────────────────────────
    gstar_path = os.path.join(GRAPH_DIR, "G_star_metadata.json")
    if os.path.exists(gstar_path):
        print("G* already built. Loading...")
        with open(gstar_path) as f:
            metadata = json.load(f)
        log_entry("[EXP2] G* loaded from disk", f"- max_kl: {metadata['max_kl']:.6f}")
    else:
        metadata = build_causal_graph.main()

    sae, ckpt = load_sae()
    mean = np.array(ckpt["act_mean"])
    std = np.array(ckpt["act_std"])
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)

    model = PPO.load(os.path.join(CKPT_DIR, "ppo_final.zip"), device=str(device))
    model.policy.eval()

    # ── Phase 2: Validate EAP ────────────────────────────────────────────
    with open(os.path.join(ACT_DIR, "meta.json")) as f:
        act_meta = json.load(f)
    n_acts = act_meta["n_samples"]
    dim = act_meta["features_dim"]
    acts_mm = np.memmap(os.path.join(ACT_DIR, "activations.npy"), dtype=np.float32,
                        mode="r", shape=(n_acts, dim))
    val_idx = np.random.choice(n_acts, 100, replace=False)
    val_norm = torch.from_numpy(((acts_mm[val_idx] - mean) / std).astype(np.float32)).to(device)

    log_entry("[EXP2] Phase 2 — Validating EAP vs patching on 100 obs", "")
    eap_r = validate_eap_vs_patching(val_norm, sae, model, std_t, mean_t,
                                     metadata["top32_features"])
    log_entry("[EXP2] Phase 2 — EAP validation complete",
              f"- Pearson r (EAP vs patching): {eap_r:.4f}\n"
              f"- {'PASS (r>0.5)' if eap_r > 0.5 else 'WARN (r<0.5) — EAP approximation weak'}")
    print(f"EAP validation Pearson r = {eap_r:.4f}")

    # ── Phase 3: Training-distribution baseline ──────────────────────────
    with open(os.path.join(OUT_DIR, "feature_index.json")) as f:
        feat_idx = json.load(f)
    goal_features = feat_idx["goal_features"]
    proxy_features = feat_idx["proxy_features"]

    log_entry("[EXP2] Phase 3 — Collecting training-distribution baseline", "")
    train_env = make_env_with_info(goal_fixed=True)
    baseline_eps = []
    for ep in range(20):
        ep_data = run_episode(model, sae, train_env, mean, std, mean_t, std_t,
                              metadata, goal_features, proxy_features, 0.1, 0.1)
        baseline_eps.append(ep_data)
    train_env.close()

    baseline_goal_sig = float(np.mean([np.mean(e["goal_signal"]) for e in baseline_eps if e["goal_signal"]]))
    baseline_proxy_sig = float(np.mean([np.mean(e["proxy_signal"]) for e in baseline_eps if e["proxy_signal"]]))
    baseline_vtotal = float(np.mean([np.mean(e["v_total"]) for e in baseline_eps if e["v_total"]]))

    log_entry("[EXP2] Training baseline",
              f"- goal_sig: {baseline_goal_sig:.4f}\n"
              f"- proxy_sig: {baseline_proxy_sig:.4f}\n"
              f"- mean V_total: {baseline_vtotal:.6f} (should be near 0)")

    # ── Phase 4: Test-distribution deployment ────────────────────────────
    seeds = [0, 42, 123]
    n_episodes = 10
    all_k_act, all_k_graph, all_k_act_valid, all_k_graph_valid = [], [], [], []
    seed_results = {}

    for seed in seeds:
        log_entry(f"[EXP2] Phase 4 — Seed {seed}: 10 test-dist episodes", "")
        test_env = make_env_with_info(goal_fixed=False)
        seed_k_act, seed_k_graph = [], []
        rep_ep = None  # first episode for representative plot

        for ep in range(n_episodes):
            save_gl = (ep == 0)  # save glive for first episode of each seed
            ep_data = run_episode(
                model, sae, test_env, mean, std, mean_t, std_t,
                metadata, goal_features, proxy_features,
                baseline_goal_sig, baseline_proxy_sig,
                save_glive=save_gl,
            )
            if ep_data["k_activation"] is not None:
                seed_k_act.append(ep_data["k_activation"])
                all_k_act.append(ep_data["k_activation"])
            if ep_data["k_graph"] is not None:
                seed_k_graph.append(ep_data["k_graph"])
                all_k_graph.append(ep_data["k_graph"])
            if ep == 0:
                rep_ep = ep_data

        test_env.close()

        # Save representative G_live trajectory
        if rep_ep and rep_ep["glive_frames"]:
            np.save(os.path.join(GLIVE_DIR, f"glive_seed{seed}_ep0.npy"),
                    np.array(rep_ep["glive_frames"]))

        mean_k_act = float(np.mean(seed_k_act)) if seed_k_act else float("nan")
        mean_k_graph = float(np.mean(seed_k_graph)) if seed_k_graph else float("nan")
        seed_results[f"seed_{seed}"] = {
            "k_activation": seed_k_act, "mean_k_activation": mean_k_act,
            "k_graph": seed_k_graph, "mean_k_graph": mean_k_graph,
        }
        log_entry(f"[EXP2] Seed {seed} complete",
                  f"- mean_k_act: {mean_k_act:.1f}\n"
                  f"- mean_k_graph: {mean_k_graph:.1f}")

    mean_k_act_all = float(np.mean(all_k_act)) if all_k_act else float("nan")
    std_k_act_all = float(np.std(all_k_act)) if len(all_k_act) > 1 else float("nan")
    mean_k_graph_all = float(np.mean(all_k_graph)) if all_k_graph else float("nan")
    std_k_graph_all = float(np.std(all_k_graph)) if len(all_k_graph) > 1 else float("nan")

    # ── Plots ────────────────────────────────────────────────────────────
    # k comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(["k_activation\n(Exp1 method)", "k_graph\n(Exp2 method)"],
                [mean_k_act_all, mean_k_graph_all],
                yerr=[std_k_act_all, std_k_graph_all],
                capsize=5, color=["steelblue", "coral"])
    axes[0].axhline(K_ACTIVATION_EXP1, color="blue", linestyle="--",
                    label=f"Exp1 k={K_ACTIVATION_EXP1:.1f}")
    axes[0].set_ylabel("Mean k (steps)")
    axes[0].set_title("k_graph vs k_activation")
    axes[0].legend()

    if all_k_graph:
        axes[1].hist(all_k_graph, bins=15, color="coral", edgecolor="white",
                     alpha=0.7, label="k_graph")
    if all_k_act:
        axes[1].hist(all_k_act, bins=15, color="steelblue", edgecolor="white",
                     alpha=0.7, label="k_activation")
    axes[1].set_xlabel("k (steps)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of k values")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "k_comparison.png"), dpi=150)
    plt.close()

    # Representative episode plot (last seed's first episode)
    if rep_ep:
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        steps = range(rep_ep["n_steps"])
        axes[0].plot(rep_ep["goal_signal"], color="blue", label="Goal signal")
        axes[0].axhline(baseline_goal_sig * 0.5, color="blue", linestyle="--",
                        alpha=0.5, label="50% threshold")
        if rep_ep["k_activation_step"] is not None:
            axes[0].axvline(rep_ep["k_activation_step"], color="blue", linestyle=":",
                            label=f"k_act step={rep_ep['k_activation_step']}")
        axes[0].set_ylabel("Goal signal")
        axes[0].legend(fontsize=7)

        axes[1].plot(rep_ep["proxy_signal"], color="orange", label="Proxy signal")
        axes[1].set_ylabel("Proxy signal")

        axes[2].plot(rep_ep["v_total"], color="purple", label="V_total")
        axes[2].axhline(metadata["v_total_threshold"], color="purple", linestyle="--",
                        alpha=0.5, label=f"threshold={metadata['v_total_threshold']:.5f}")
        if rep_ep["k_graph_step"] is not None:
            axes[2].axvline(rep_ep["k_graph_step"], color="purple", linestyle=":",
                            label=f"k_graph step={rep_ep['k_graph_step']}")
        axes[2].set_ylabel("V_total")
        axes[2].legend(fontsize=7)

        axes[3].step(range(len(rep_ep["rewards"])), rep_ep["rewards"],
                     color="green", label="Reward")
        axes[3].set_ylabel("Reward")
        axes[3].set_xlabel("Step")

        plt.suptitle(f"Representative Episode — k_act={rep_ep['k_activation']}, "
                     f"k_graph={rep_ep['k_graph']}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "representative_episode_exp2.png"), dpi=150)
        plt.close()

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "eap_pearson_r": eap_r,
        "baseline_vtotal_mean": baseline_vtotal,
        "mean_k_activation": mean_k_act_all,
        "std_k_activation": std_k_act_all,
        "mean_k_graph": mean_k_graph_all,
        "std_k_graph": std_k_graph_all,
        "k_activation_exp1_reference": K_ACTIVATION_EXP1,
        "n_k_act_measurements": len(all_k_act),
        "n_k_graph_measurements": len(all_k_graph),
        "seed_results": seed_results,
    }
    with open(os.path.join(EXP2_DIR, "experiment2_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    log_entry("[EXP2] COMPLETE",
              f"- EAP r: {eap_r:.4f}\n"
              f"- mean k_act: {mean_k_act_all:.1f} ± {std_k_act_all:.1f}\n"
              f"- mean k_graph: {mean_k_graph_all:.1f} ± {std_k_graph_all:.1f}\n"
              f"- Exp1 reference k_activation: {K_ACTIVATION_EXP1}")

    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2 COMPLETE")
    print(f"EAP r:        {eap_r:.4f}")
    print(f"k_activation: {mean_k_act_all:.1f} ± {std_k_act_all:.1f}")
    print(f"k_graph:      {mean_k_graph_all:.1f} ± {std_k_graph_all:.1f}")
    print(f"Exp1 ref:     {K_ACTIVATION_EXP1}")
    if not np.isnan(mean_k_graph_all):
        if mean_k_graph_all > mean_k_act_all:
            print("RESULT: k_graph > k_activation — causal graph fires EARLIER")
        elif mean_k_graph_all > 0:
            print("RESULT: k_graph ≈ k_activation — graph consistent with activation signal")
        else:
            print("RESULT: k_graph ≈ 0 — graph-level signal does not lead failure")
    print(f"{'='*60}\n")

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


if __name__ == "__main__":
    main()

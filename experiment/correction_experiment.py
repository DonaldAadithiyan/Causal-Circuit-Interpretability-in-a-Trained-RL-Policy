"""
Experiment 3, Option B — Second PPO training phase with R_reason.
Runs baseline (R_env only) and R_reason conditions on test distribution.
λ sweep: {0.1, 0.5, 1.0} × 3 seeds, plus baseline × 3 seeds = 12 runs.
"""

import sys, os, json, gc, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from models.topk_sae import TopKSAE
from envs.coin_env import make_env
from compute_r_reason import RReasonWrapper
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
CKPT_DIR = os.path.join(BASE, "outputs/checkpoints")
OUT_DIR = os.path.join(BASE, "outputs")
EXP3_DIR = os.path.join(BASE, "outputs/experiment3")
OPT_B_DIR = os.path.join(EXP3_DIR, "option_b")


def load_sae():
    ckpt = torch.load(os.path.join(CKPT_DIR, "sae_best.pt"), map_location=device)
    sae = TopKSAE(input_dim=ckpt["input_dim"], hidden_factor=ckpt["hidden_factor"], k=ckpt["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae, ckpt


def evaluate_policy(model, goal_fixed: bool, n_episodes: int = 20, seed: int = 0):
    """Evaluate on training or test distribution, return (mean_reward, std, failure_rate)."""
    env = make_env(goal_fixed=goal_fixed)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        rewards.append(info["episode"]["r"])
    env.close()
    rewards = np.array(rewards)
    return float(rewards.mean()), float(rewards.std()), float((rewards < 0.5).mean())


def run_one_condition(label: str, lam: float, seed: int,
                      total_timesteps: int, policy_ref,
                      sae, mean_t, std_t,
                      baseline_goal: float, baseline_proxy: float,
                      goal_feats: list, proxy_feats: list):
    """Train one condition (baseline or R_reason) and return evaluation metrics."""
    save_dir = os.path.join(OPT_B_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    # Build env factory
    if lam > 0.0:
        def env_fn():
            base_env = make_env(goal_fixed=False)
            return RReasonWrapper(
                base_env, policy_ref, sae, mean_t, std_t,
                baseline_goal, baseline_proxy, goal_feats, proxy_feats, lam
            )
    else:
        env_fn = lambda: make_env(goal_fixed=False)

    train_env = make_vec_env(env_fn, n_envs=1, seed=seed)

    # Load fresh copy of the trained policy, passing the new env to reset n_envs
    model = PPO.load(os.path.join(CKPT_DIR, "ppo_final.zip"), env=train_env, device=str(device))

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - t0

    # Save checkpoint
    ckpt_path = os.path.join(save_dir, f"policy_seed{seed}.zip")
    model.save(ckpt_path)

    # Evaluate on test distribution
    test_mean, test_std, fail_rate = evaluate_policy(model, goal_fixed=False, n_episodes=20, seed=seed*100)
    # Evaluate on training distribution (check for catastrophic forgetting)
    train_mean, _, _ = evaluate_policy(model, goal_fixed=True, n_episodes=10, seed=seed*100)

    result = {
        "label": label,
        "lam": lam,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "elapsed_min": elapsed / 60,
        "test_mean_reward": test_mean,
        "test_std_reward": test_std,
        "test_failure_rate": fail_rate,
        "train_mean_reward": train_mean,
        "catastrophic_forgetting": train_mean < 0.7,
    }

    log_entry(f"[EXP3] {label} seed={seed} complete",
              f"- test_fail_rate: {fail_rate:.3f}\n"
              f"- test_mean_reward: {test_mean:.4f}\n"
              f"- train_mean_reward: {train_mean:.4f} (forgetting: {train_mean < 0.7})\n"
              f"- elapsed: {elapsed/60:.1f} min")

    del model
    train_env.close()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result


def main(total_timesteps: int = 100_000):
    os.makedirs(OPT_B_DIR, exist_ok=True)

    sae, ckpt = load_sae()
    mean = np.array(ckpt["act_mean"])
    std = np.array(ckpt["act_std"])
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)

    with open(os.path.join(OUT_DIR, "feature_index.json")) as f:
        feat_idx = json.load(f)
    goal_feats = feat_idx["goal_features"]
    proxy_feats = feat_idx["proxy_features"]

    # Load baseline signals from Experiment 1
    with open(os.path.join(OUT_DIR, "misgeneralization_results.json")) as f:
        mis = json.load(f)
    baseline_goal = mis["baseline_goal_signal"]
    baseline_proxy = mis["baseline_proxy_signal"]

    # Load frozen reference policy (never updated)
    policy_ref = PPO.load(os.path.join(CKPT_DIR, "ppo_final.zip"), device=str(device))
    policy_ref.policy.eval()
    for p in policy_ref.policy.parameters():
        p.requires_grad_(False)

    log_entry("[EXP3] Option B START",
              f"- λ values: [0.0, 0.1, 0.5, 1.0]\n"
              f"- seeds: [0, 42, 123]\n"
              f"- timesteps per run: {total_timesteps:,}\n"
              f"- baseline_goal_sig: {baseline_goal:.4f}\n"
              f"- baseline_proxy_sig: {baseline_proxy:.4f}")

    seeds = [0, 42, 123]
    lambdas = [0.0, 0.1, 0.5, 1.0]  # 0.0 = baseline
    all_results = []

    for lam in lambdas:
        for seed in seeds:
            label = f"baseline_seed{seed}" if lam == 0.0 else f"lam{lam}_seed{seed}"
            log_entry(f"[EXP3] Starting: {label}", f"- λ={lam}, seed={seed}")
            result = run_one_condition(
                label=label, lam=lam, seed=seed,
                total_timesteps=total_timesteps,
                policy_ref=policy_ref, sae=sae,
                mean_t=mean_t, std_t=std_t,
                baseline_goal=baseline_goal, baseline_proxy=baseline_proxy,
                goal_feats=goal_feats, proxy_feats=proxy_feats,
            )
            all_results.append(result)

    # Aggregate
    by_lam = {}
    for r in all_results:
        lam = r["lam"]
        if lam not in by_lam:
            by_lam[lam] = []
        by_lam[lam].append(r)

    summary = {}
    for lam, runs in sorted(by_lam.items()):
        fail_rates = [r["test_failure_rate"] for r in runs]
        rewards = [r["test_mean_reward"] for r in runs]
        summary[str(lam)] = {
            "mean_failure_rate": float(np.mean(fail_rates)),
            "std_failure_rate": float(np.std(fail_rates)),
            "mean_reward": float(np.mean(rewards)),
            "seeds": seeds,
            "per_seed": runs,
        }

    with open(os.path.join(EXP3_DIR, "experiment3_results.json"), "w") as f:
        json.dump({"all_results": all_results, "summary_by_lambda": summary}, f, indent=2)

    baseline_fail = summary["0.0"]["mean_failure_rate"]
    log_entry("[EXP3] Option B COMPLETE",
              "\n".join([
                  f"- λ={lam}: fail_rate={summary[str(lam)]['mean_failure_rate']:.3f} "
                  f"± {summary[str(lam)]['std_failure_rate']:.3f}"
                  for lam in sorted(by_lam.keys())
              ]) + f"\n- baseline fail_rate: {baseline_fail:.3f}")

    print(f"\n{'='*60}")
    print("EXPERIMENT 3 OPTION B COMPLETE")
    for lam in sorted(by_lam.keys()):
        s = summary[str(lam)]
        delta = s["mean_failure_rate"] - baseline_fail
        print(f"  λ={lam:4.1f}: fail={s['mean_failure_rate']:.3f}±{s['std_failure_rate']:.3f}  "
              f"Δ_baseline={delta:+.3f}")
    print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    main()

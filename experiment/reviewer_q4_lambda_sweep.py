"""
Reviewer Q4 — How narrow is the λ working band, really?
Finer sweep λ ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5} on the (6,5) routing blind spot.
Reduced to 2 seeds × 40k steps per run (12 runs) to fit the local budget — logged.
Reuses the Exp4 random-goal policy + SAEv3 + goal features. Resumable.
"""

import sys, os, json, gc, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from models.topk_sae_v2 import TopKSAEv2
from envs.coin_env import make_env
from compute_r_reason import RReasonWrapper
from utils.logging_utils import log_entry
from experiment4_main import compute_baselines, eval_failure

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
E4 = os.path.join(BASE, "outputs/experiment4")
OUT = os.path.join(E4, "reviewer")
TEST_GOAL = (6, 5)
LAMBDAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
SEEDS = [0, 42]
STEPS = 40_000


def load_sae():
    ck = torch.load(os.path.join(E4, "sae_v3/sae_v3_best.pt"), map_location=device)
    sae = TopKSAEv2(input_dim=ck["input_dim"], hidden_factor=ck["hidden_factor"], k=ck["k"]).to(device)
    sae.load_state_dict(ck["state_dict"]); sae.eval()
    for p in sae.parameters(): p.requires_grad_(False)
    return sae, ck


def main():
    os.makedirs(OUT, exist_ok=True)
    runs_path = os.path.join(OUT, "q4_lambda_runs.json")
    runs = json.load(open(runs_path)) if os.path.exists(runs_path) else []
    done = {(r["lam"], r["seed"]) for r in runs}

    sae, ck = load_sae()
    mean = np.array(ck["act_mean"]); std = np.array(ck["act_std"])
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)
    W = np.load(os.path.join(E4, "graphs/W_interfeature.npy"))
    with open(os.path.join(E4, "graphs/G_star_v3_metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(E4, "goal_features.json")) as f:
        gf = json.load(f)
    goal_features = gf["goal_features"]; proxy_features = gf["proxy_features"]

    model = PPO.load(os.path.join(E4, "policy_randomgoal/ppo_final.zip"), device=str(device))
    model.policy.eval()
    bg, bp = compute_baselines(model, sae, mean, std, W, metadata, goal_features, proxy_features)
    del model

    policy_ref = PPO.load(os.path.join(E4, "policy_randomgoal/ppo_final.zip"), device=str(device))
    policy_ref.policy.eval()
    for p in policy_ref.policy.parameters(): p.requires_grad_(False)

    log_entry("[EXP4-Q4] Finer λ sweep START",
              f"- λ {LAMBDAS} × seeds {SEEDS} × {STEPS//1000}k steps on (6,5)\n"
              f"- baseline goal_sig {bg:.4f} proxy_sig {bp:.4f}")

    for lam in LAMBDAS:
        for seed in SEEDS:
            if (lam, seed) in done:
                continue
            t0 = time.time()
            def env_fn():
                return RReasonWrapper(make_env(fixed_goal_pos=TEST_GOAL), policy_ref.policy,
                                      sae, mean_t, std_t, bg, bp,
                                      goal_features, proxy_features, lam)
            venv = make_vec_env(env_fn, n_envs=1, seed=seed)
            m = PPO.load(os.path.join(E4, "policy_randomgoal/ppo_final.zip"), env=venv, device=str(device))
            m.learn(total_timesteps=STEPS)
            fail, rew = eval_failure(m, fixed_goal_pos=TEST_GOAL, n=20, seed=seed)
            runs.append({"lam": lam, "seed": seed, "failure_rate": fail,
                         "mean_reward": rew, "elapsed_min": (time.time()-t0)/60})
            json.dump(runs, open(runs_path, "w"), indent=2)
            log_entry(f"[EXP4-Q4] λ={lam} seed={seed}",
                      f"- fail={fail:.3f} reward={rew:.3f} ({(time.time()-t0)/60:.0f} min)")
            del m, venv; gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()

    # Aggregate
    by_lam = {}
    for r in runs:
        by_lam.setdefault(r["lam"], []).append(r["failure_rate"])
    summary = {str(l): {"mean_failure": float(np.mean(v)), "std_failure": float(np.std(v)),
                        "n": len(v)} for l, v in sorted(by_lam.items())}
    json.dump({"summary": summary, "runs": runs, "test_goal": list(TEST_GOAL),
               "steps_per_run": STEPS, "seeds": SEEDS},
              open(os.path.join(OUT, "q4_lambda_sweep.json"), "w"), indent=2)

    lams = sorted(by_lam.keys())
    means = [summary[str(l)]["mean_failure"] for l in lams]
    stds = [summary[str(l)]["std_failure"] for l in lams]
    plt.figure(figsize=(8, 5))
    plt.errorbar(lams, means, yerr=stds, marker="o", capsize=4, color="coral")
    plt.axhline(1.0, color="gray", ls="--", alpha=0.5, label="baseline (no response) = 1.0")
    plt.xlabel("λ (R_reason weight)"); plt.ylabel("Test failure rate at (6,5)")
    plt.title("Q4 — R_reason failure rate vs λ (finer sweep)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "q4_lambda_sweep.png"), dpi=150)
    plt.close()

    log_entry("[EXP4-Q4] COMPLETE",
              "\n".join([f"- λ={l}: fail={summary[str(l)]['mean_failure']:.3f}±{summary[str(l)]['std_failure']:.3f}"
                         for l in lams]))
    print("\nQ4 λ sweep:")
    for l in lams:
        print(f"  λ={l}: fail={summary[str(l)]['mean_failure']:.3f} ± {summary[str(l)]['std_failure']:.3f}")


if __name__ == "__main__":
    main()

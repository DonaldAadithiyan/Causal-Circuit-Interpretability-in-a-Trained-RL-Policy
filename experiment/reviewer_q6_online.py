"""
Reviewer Q6 — Is online correction possible (Option A)?
Deploy the policy on the (6,5) routing blind spot, receive R_reason every step, and run
PPO updates ONLINE with short rollouts (n_steps=512). Measure the failure-rate CURVE as a
function of deployment steps — does the agent correct within a single continuous run, and
how fast? Also track training-distribution reward to check catastrophic forgetting.

Uses the working λ=0.1 from Exp4/Q4. Reuses Exp4 random-goal policy + SAEv3 + goal features.
"""

import sys, os, json
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
LAM = 0.1
CHUNK = 4000          # steps per online update block
TOTAL = 28000         # total online deployment steps
N_STEPS = 512         # short rollouts for fast online adaptation


def load_sae():
    ck = torch.load(os.path.join(E4, "sae_v3/sae_v3_best.pt"), map_location=device)
    sae = TopKSAEv2(input_dim=ck["input_dim"], hidden_factor=ck["hidden_factor"], k=ck["k"]).to(device)
    sae.load_state_dict(ck["state_dict"]); sae.eval()
    for p in sae.parameters(): p.requires_grad_(False)
    return sae, ck


def main():
    os.makedirs(OUT, exist_ok=True)
    log_entry("[EXP4-Q6] Online correction (Option A) START",
              f"- (6,5), λ={LAM}, n_steps={N_STEPS}, chunks of {CHUNK} up to {TOTAL}")

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

    ref = PPO.load(os.path.join(E4, "policy_randomgoal/ppo_final.zip"), device=str(device))
    ref.policy.eval()
    bg, bp = compute_baselines(ref, sae, mean, std, W, metadata, goal_features, proxy_features)
    for p in ref.policy.parameters(): p.requires_grad_(False)

    def env_fn():
        return RReasonWrapper(make_env(fixed_goal_pos=TEST_GOAL), ref.policy, sae, mean_t, std_t,
                              bg, bp, goal_features, proxy_features, LAM)
    venv = make_vec_env(env_fn, n_envs=1, seed=0)

    model = PPO.load(os.path.join(E4, "policy_randomgoal/ppo_final.zip"), env=venv, device=str(device))
    model.n_steps = N_STEPS
    model.rollout_buffer.buffer_size = N_STEPS

    # Initial failure (before any online update)
    f0, _ = eval_failure(model, fixed_goal_pos=TEST_GOAL, n=20, seed=0)
    tr0, _ = eval_failure(model, random_goal=True, n=10, seed=0)
    curve = [{"steps": 0, "test_failure": f0, "train_reward_proxy": 1 - tr0}]
    log_entry("[EXP4-Q6] step 0", f"- test_fail={f0:.2f}")

    steps_done = 0
    while steps_done < TOTAL:
        model.learn(total_timesteps=CHUNK, reset_num_timesteps=False)
        steps_done += CHUNK
        f, _ = eval_failure(model, fixed_goal_pos=TEST_GOAL, n=20, seed=0)
        trf, _ = eval_failure(model, random_goal=True, n=10, seed=0)  # train-dist failure
        curve.append({"steps": steps_done, "test_failure": f, "train_failure": trf})
        log_entry(f"[EXP4-Q6] {steps_done} steps",
                  f"- test_fail={f:.2f} train_fail={trf:.2f}")

    # First step where test failure hits 0
    corrected_at = next((c["steps"] for c in curve if c["test_failure"] == 0.0), None)
    final = curve[-1]
    summary = {
        "test_goal": list(TEST_GOAL), "lambda": LAM, "n_steps_rollout": N_STEPS,
        "initial_test_failure": f0, "final_test_failure": final["test_failure"],
        "corrected_at_steps": corrected_at,
        "final_train_failure": final.get("train_failure"),
        "curve": curve,
    }
    json.dump(summary, open(os.path.join(OUT, "q6_online_correction.json"), "w"), indent=2)

    xs = [c["steps"] for c in curve]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [c["test_failure"] for c in curve], marker="o", color="coral", label="test failure (6,5)")
    plt.plot(xs[1:], [c.get("train_failure", np.nan) for c in curve[1:]], marker="s",
             color="steelblue", label="train-dist failure (forgetting check)")
    plt.xlabel("Online deployment steps"); plt.ylabel("Failure rate")
    plt.title(f"Q6 — Online correction at (6,5), λ={LAM}"
              + (f"  (corrected by {corrected_at} steps)" if corrected_at else ""))
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "q6_online_correction.png"), dpi=150)
    plt.close()

    log_entry("[EXP4-Q6] COMPLETE",
              f"- initial test fail {f0:.2f} -> final {final['test_failure']:.2f}\n"
              f"- corrected at {corrected_at} steps\n"
              f"- final train-dist failure {final.get('train_failure')}")
    print(f"\nQ6 — ONLINE CORRECTION")
    print(f"  initial (6,5) failure: {f0:.2f}")
    print(f"  final   (6,5) failure: {final['test_failure']:.2f}")
    print(f"  corrected by:          {corrected_at} online steps")
    print(f"  final train-dist fail: {final.get('train_failure')}  (forgetting check)")


if __name__ == "__main__":
    main()

"""
Reviewer Q1 — Is (6,5) cherry-picked?
Sweep ALL valid goal positions. For each, measure failure, k_activation, k_graph.
Count positions that exhibit a ROUTING failure: k_activation undefined (goal feature
stays active) AND k_graph fires. Report the fraction.

Uses the frozen Exp4 random-goal policy + SAEv3 + W-matrix. Deployment only, no training.
"""

import sys, os, json
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
from experiment4_main import deploy_episode, compute_baselines

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
E4 = os.path.join(BASE, "outputs/experiment4")
OUT = os.path.join(E4, "reviewer")
GRID = 8
N_EPISODES_PER_POS = 5   # deterministic policy → mainly confirms reproducibility


def load_sae():
    ck = torch.load(os.path.join(E4, "sae_v3/sae_v3_best.pt"), map_location=device)
    sae = TopKSAEv2(input_dim=ck["input_dim"], hidden_factor=ck["hidden_factor"], k=ck["k"]).to(device)
    sae.load_state_dict(ck["state_dict"]); sae.eval()
    return sae, ck


def main():
    os.makedirs(OUT, exist_ok=True)
    log_entry("[EXP4-Q1] Position sweep START — routing-failure prevalence", "")

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
    log_entry("[EXP4-Q1] baselines", f"- goal_sig {bg:.4f}, proxy_sig {bp:.4f}")

    positions = [(x, y) for y in range(1, GRID - 1) for x in range(1, GRID - 1)
                 if (x, y) != (1, 1)]

    results = []
    fail_grid = np.full((GRID, GRID), np.nan)
    type_grid = np.zeros((GRID, GRID))  # 0 solved, 1 routing, 2 representation

    for (gx, gy) in positions:
        env = make_env_with_info(fixed_goal_pos=(gx, gy))
        eps = []
        for ep in range(N_EPISODES_PER_POS):
            env.reset(seed=ep)
            d = deploy_episode(model, sae, env, mean, std, mean_t, std_t, W,
                               metadata, goal_features, proxy_features, bg)
            eps.append(d)
        env.close()

        fail_rate = float(np.mean([e["failed"] for e in eps]))
        failing = [e for e in eps if e["failed"]]
        # Robust routing metric: mean goal-feature activation over the episode as a
        # fraction of the training baseline. High (>0.6) = goal stays active = ROUTING
        # failure (goal represented but not acted on). Low = goal weak = REPRESENTATION.
        goal_fracs = [float(np.mean(e["goal"]) / max(bg, 1e-8)) for e in failing]
        goal_frac = float(np.mean(goal_fracs)) if failing else float("nan")
        frac_kgraph = float(np.mean([e["k_graph"] is not None for e in failing])) if failing else 0.0
        frac_kact_nan = float(np.mean([e["k_activation"] is None for e in failing])) if failing else 0.0

        # Classify the position on the robust continuous metric
        if fail_rate < 0.5:
            ptype = "solved"; type_grid[gy, gx] = 0
        elif goal_frac > 0.6:
            ptype = "routing"; type_grid[gy, gx] = 1           # goal stays active, still fails
        else:
            ptype = "representation"; type_grid[gy, gx] = 2     # goal feature weak

        fail_grid[gy, gx] = fail_rate
        results.append({
            "pos": [gx, gy], "failure_rate": fail_rate,
            "goal_activation_fraction": goal_frac,
            "frac_k_graph_fired": frac_kgraph,
            "frac_k_activation_undefined": frac_kact_nan,
            "type": ptype,
        })

    n_total = len(positions)
    n_solved = sum(1 for r in results if r["type"] == "solved")
    n_routing = sum(1 for r in results if r["type"] == "routing")
    n_repr = sum(1 for r in results if r["type"] == "representation")
    routing_positions = [r["pos"] for r in results if r["type"] == "routing"]

    summary = {
        "n_positions": n_total,
        "n_solved": n_solved,
        "n_routing_failure": n_routing,
        "n_representation_failure": n_repr,
        "routing_fraction": n_routing / n_total,
        "failure_fraction": (n_routing + n_repr) / n_total,
        "routing_positions": routing_positions,
        "baseline_goal_sig": bg, "baseline_proxy_sig": bp,
        "per_position": results,
    }
    with open(os.path.join(OUT, "q1_position_sweep.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plot: failure-type map
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.matplotlib.colors.ListedColormap(["#dddddd", "#2ca02c", "#d62728", "#1f77b4"])
    # 0 nan(border) handled separately; we plot type_grid 0/1/2 over interior
    disp = np.full((GRID, GRID), -1.0)
    for r in results:
        gx, gy = r["pos"]
        disp[gy, gx] = {"solved": 0, "routing": 1, "representation": 2}[r["type"]]
    masked = np.ma.masked_where(disp < 0, disp)
    im = ax.imshow(masked, cmap=plt.matplotlib.colors.ListedColormap(["#2ca02c", "#d62728", "#1f77b4"]),
                   vmin=0, vmax=2, origin="upper")
    for r in results:
        gx, gy = r["pos"]
        ax.text(gx, gy, f"{r['failure_rate']:.0%}", ha="center", va="center", fontsize=7)
    ax.set_title(f"Failure-type map — routing={n_routing}/{n_total}, "
                 f"repr={n_repr}, solved={n_solved}\n(green=solved, red=routing, blue=representation)")
    ax.set_xticks(range(GRID)); ax.set_yticks(range(GRID))
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "q1_failure_type_map.png"), dpi=150)
    plt.close()

    n_fail = n_routing + n_repr
    graph_detect = float(np.mean([r["frac_k_graph_fired"] for r in results
                                  if r["type"] != "solved"])) if n_fail else 0.0
    log_entry("[EXP4-Q1] COMPLETE",
              f"- positions: {n_total}, solved: {n_solved}, failures: {n_fail}\n"
              f"- graph fired on {graph_detect:.0%} of failing positions\n"
              f"- ROUTING failures (goal_act_frac>0.6, goal active but ignored): {n_routing} -> {routing_positions}\n"
              f"- representation failures (goal feature silent): {n_repr}")

    print(f"\n{'='*60}")
    print("Q1 — POSITION SWEEP COMPLETE")
    print(f"  positions tested:        {n_total}")
    print(f"  solved (policy reaches): {n_solved}")
    print(f"  total failures:          {n_fail}  ({n_fail/n_total:.0%})")
    print(f"  graph detected:          {graph_detect:.0%} of failing positions")
    print(f"  ROUTING failures:        {n_routing} -> {routing_positions}  (goal active but ignored)")
    print(f"  representation failures: {n_repr}  (goal feature silent)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

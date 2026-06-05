"""
Reviewer Q3 — Can the routing-vs-representation diagnosis be automated?
Build a classifier from the live signals and validate it non-circularly:

  classifier(failure episode):
    goal_activation_fraction > 0.6  AND k_graph fires  ->  ROUTING  -> prescribe R_reason
    goal_activation_fraction < 0.6  AND k_graph fires  ->  REPRESENTATION -> prescribe steering

Validation: the prescribed response should ENGAGE on the matching type.
- Steering's I3 trigger should fire heavily on REPRESENTATION positions (goal silent)
  and stay quiet on ROUTING positions (goal active). That is the opposite of where
  R_reason (action retraining) is needed — already shown to fix routing (6,5).
Run steering on every failing position and report steer_fraction + failure rate.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

from stable_baselines3 import PPO
from models.topk_sae_v2 import TopKSAEv2
from utils.logging_utils import log_entry
import response_activation_steering as steer

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
E4 = os.path.join(BASE, "outputs/experiment4")
OUT = os.path.join(E4, "reviewer")


def load_sae():
    ck = torch.load(os.path.join(E4, "sae_v3/sae_v3_best.pt"), map_location=device)
    sae = TopKSAEv2(input_dim=ck["input_dim"], hidden_factor=ck["hidden_factor"], k=ck["k"]).to(device)
    sae.load_state_dict(ck["state_dict"]); sae.eval()
    return sae, ck


def main():
    os.makedirs(OUT, exist_ok=True)
    log_entry("[EXP4-Q3] Diagnosis classifier START", "")

    q1 = json.load(open(os.path.join(OUT, "q1_position_sweep.json")))
    failing = [r for r in q1["per_position"] if r["type"] != "solved"]

    sae, ck = load_sae()
    mean = np.array(ck["act_mean"]); std = np.array(ck["act_std"])
    with open(os.path.join(E4, "graphs/G_star_v3_metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(E4, "goal_features.json")) as f:
        gf = json.load(f)
    goal_features = gf["goal_features"]

    model = PPO.load(os.path.join(E4, "policy_randomgoal/ppo_final.zip"), device=str(device))
    model.policy.eval()

    # ── The classifier (from signals) ──
    THRESH = 0.6
    for r in failing:
        gafrac = r["goal_activation_fraction"]
        kgraph = r["frac_k_graph_fired"] > 0.5
        r["predicted_type"] = ("routing" if (gafrac > THRESH and kgraph)
                               else "representation" if kgraph else "undetected")
        r["prescribed_response"] = ("R_reason" if r["predicted_type"] == "routing"
                                    else "activation_steering")

    # Classifier vs the Q1 ground-truth label (independent: continuous goal-activation magnitude)
    correct = sum(1 for r in failing if r["predicted_type"] == r["type"])
    accuracy = correct / len(failing) if failing else 0.0

    # ── Validate: run STEERING on every failing position, report trigger + effect ──
    steer_results = []
    for r in failing:
        pos = tuple(r["pos"])
        res = steer.run_steering_condition(model, sae, mean, std, metadata, goal_features,
                                           pos, alpha=1.0, seeds=(0, 42, 123), n_eps=10)
        steer_results.append({"pos": list(pos), "type": r["type"],
                              "steer_fraction": res["steer_fraction"],
                              "failure_rate": res["failure_rate"]})
        log_entry(f"[EXP4-Q3] steering at {pos} ({r['type']})",
                  f"- steer_frac={res['steer_fraction']:.2f} fail={res['failure_rate']:.2f}")

    # Aggregate steer-engagement by type
    routing_sf = [s["steer_fraction"] for s in steer_results if s["type"] == "routing"]
    repr_sf = [s["steer_fraction"] for s in steer_results if s["type"] == "representation"]

    summary = {
        "classifier_threshold_goal_frac": THRESH,
        "classifier_accuracy_vs_q1_labels": accuracy,
        "n_failing_positions": len(failing),
        "failing_positions": [
            {"pos": r["pos"], "ground_truth_type": r["type"],
             "goal_activation_fraction": r["goal_activation_fraction"],
             "predicted_type": r["predicted_type"],
             "prescribed_response": r["prescribed_response"]}
            for r in failing],
        "steering_validation": steer_results,
        "mean_steer_fraction_routing": float(np.mean(routing_sf)) if routing_sf else None,
        "mean_steer_fraction_representation": float(np.mean(repr_sf)) if repr_sf else None,
    }
    json.dump(summary, open(os.path.join(OUT, "q3_diagnosis.json"), "w"), indent=2)

    log_entry("[EXP4-Q3] COMPLETE",
              f"- classifier accuracy vs Q1 labels: {accuracy:.0%}\n"
              f"- steer engagement: routing {summary['mean_steer_fraction_routing']}, "
              f"representation {summary['mean_steer_fraction_representation']}")

    print(f"\n{'='*64}")
    print("Q3 — AUTOMATED DIAGNOSIS")
    print(f"  classifier accuracy vs Q1 labels: {accuracy:.0%}  ({correct}/{len(failing)})")
    print(f"  prescription per failing position:")
    for r in failing:
        print(f"    {tuple(r['pos'])}: goal_frac={r['goal_activation_fraction']:.2f} "
              f"-> {r['predicted_type']:14s} -> {r['prescribed_response']}")
    print(f"  steering engagement (steer_fraction):")
    print(f"    routing positions:        {summary['mean_steer_fraction_routing']}")
    print(f"    representation positions: {summary['mean_steer_fraction_representation']}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()

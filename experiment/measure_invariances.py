"""
Experiment 2, Phase 3 — Invariance checks I1-I5 and violation score V_total.
Provides check_invariances() used per step during deployment.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def check_invariances(
    c_live_top32: np.ndarray,   # (32,) live EAP causal weights for top32 features
    h_active: np.ndarray,       # (hidden_dim,) SAE sparse feature activations (single step)
    metadata: dict,
    sae=None,
    model=None,
    std_t=None,
    mean_t=None,
    run_i5: bool = True,
) -> Tuple[Dict[str, bool], float, Dict[str, float]]:
    """
    Returns (violations, V_total, diagnostics).
    """
    top32 = metadata["top32_features"]
    c_star = np.array(metadata["c_star"])
    goal_feats = metadata["goal_features_in_top32"]
    proxy_feats = metadata["proxy_features_in_top32"]
    spurious_set = metadata["spurious_set"]

    violations = {"I1": False, "I2": False, "I3": False, "I4": False, "I5": False}

    # I1: causal depth collapse — top feature dominates more than 2× training baseline
    c_total = c_live_top32.sum()
    if c_total > 1e-10:
        top1_conc = float(c_live_top32.max() / c_total)
        conc_star = metadata["depth_concentration_star"]
        if conc_star > 1e-8 and top1_conc > 2.0 * conc_star:
            violations["I1"] = True
    else:
        top1_conc = 0.0

    # I2: spurious edge re-entry
    sp_mean = metadata["c_star_spurious_mean"]
    sp_std = max(metadata["c_star_spurious_std"], 1e-8)
    for feat in spurious_set:
        if feat in top32:
            c_live_feat = c_live_top32[top32.index(feat)]
            if c_live_feat > sp_mean + 2.0 * sp_std:
                violations["I2"] = True
                break

    # I3: goal feature causal weight drop below 50% of baseline
    goal_c_star_mean = metadata["goal_c_star_mean"]
    goal_c_live_vals = [c_live_top32[top32.index(f)] for f in goal_feats if f in top32]
    goal_c_live_mean = float(np.mean(goal_c_live_vals)) if goal_c_live_vals else 0.0
    if goal_c_star_mean > 1e-10 and goal_c_live_mean < 0.5 * goal_c_star_mean:
        violations["I3"] = True

    # I4: proxy feature causal weight rise above 150% of baseline
    proxy_c_star_mean = metadata["proxy_c_star_mean"]
    proxy_c_live_vals = [c_live_top32[top32.index(f)] for f in proxy_feats if f in top32]
    proxy_c_live_mean = float(np.mean(proxy_c_live_vals)) if proxy_c_live_vals else 0.0
    if proxy_c_star_mean > 1e-10 and proxy_c_live_mean > 1.5 * proxy_c_star_mean:
        violations["I4"] = True

    # I5: self-consistency — only run when I1–I4 flagged something
    any_violation = any(violations[k] for k in ["I1", "I2", "I3", "I4"])
    if any_violation and run_i5 and sae is not None and model is not None:
        top_feat_rank = int(c_live_top32.argmax())
        top_feat_global = top32[top_feat_rank]
        try:
            with torch.no_grad():
                h_t = torch.from_numpy(h_active.astype(np.float32)).unsqueeze(0).to(device)
                recon_orig = sae.decode(h_t) * std_t + mean_t
                logits_orig = model.policy.action_net(recon_orig)
                probs_orig = F.softmax(logits_orig, dim=-1)

                h_patched = h_t.clone()
                h_patched[0, top_feat_global] = 0.0
                recon_patched = sae.decode(h_patched) * std_t + mean_t
                logits_patched = model.policy.action_net(recon_patched)
                probs_patched = F.softmax(logits_patched, dim=-1)

                p = probs_orig + 1e-8
                q = probs_patched + 1e-8
                kl_i5 = (p * torch.log(p / q)).sum().item()
                if kl_i5 < metadata["kl_threshold"]:
                    violations["I5"] = True
        except Exception:
            pass

    # V_total = alpha*V_drop + beta*V_gain + gamma*I5
    alpha, beta, gamma = 1.0, 1.0, 1.0

    v_drop = sum(max(0.0, c_star[i] - c_live_top32[i])
                 for i, f in enumerate(top32) if f in goal_feats)
    v_gain = sum(max(0.0, c_live_top32[top32.index(f)] - c_star[top32.index(f)])
                 for f in spurious_set if f in top32)
    v_total = alpha * v_drop + beta * v_gain + gamma * float(violations["I5"])

    diagnostics = {
        "goal_c_live_mean": goal_c_live_mean,
        "proxy_c_live_mean": proxy_c_live_mean,
        "top1_concentration": top1_conc if c_total > 1e-10 else 0.0,
        "v_drop": v_drop,
        "v_gain": v_gain,
    }
    return violations, v_total, diagnostics

"""
R_reason computation — used during Experiment 3 Option B training.
Uses activation-based V (reliable, since EAP Pearson r=0.146 was too low in Exp2).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import gymnasium as gym

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def compute_r_reason(
    obs_uint8: np.ndarray,
    policy_ref,
    sae,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    baseline_goal: float,
    baseline_proxy: float,
    goal_feats: list,
    proxy_feats: list,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Return R_reason = -(alpha*V_drop + beta*V_gain) ∈ [-2, 0]."""
    obs_t = (torch.from_numpy(obs_uint8.astype(np.float32))
             .unsqueeze(0).permute(0, 3, 1, 2) / 255.0).to(device)

    with torch.no_grad():
        feats = policy_ref.features_extractor(obs_t)
        feats_norm = (feats - mean_t) / std_t
        h = sae.get_feature_activations(feats_norm).squeeze(0).cpu().numpy()

    goal_sig = float(h[goal_feats].mean()) if goal_feats else 0.0
    proxy_sig = float(h[proxy_feats].mean()) if proxy_feats else 0.0

    v_drop = max(0.0, baseline_goal - goal_sig) / max(baseline_goal, 1e-8)
    v_gain = max(0.0, proxy_sig - baseline_proxy) / max(baseline_proxy, 1e-8)
    return -(alpha * v_drop + beta * v_gain)


class RReasonWrapper(gym.Wrapper):
    """gym.Wrapper that adds R_reason × lam to each step reward."""

    def __init__(self, env, policy_ref, sae, mean_t, std_t,
                 baseline_goal, baseline_proxy, goal_feats, proxy_feats, lam):
        super().__init__(env)
        self.policy_ref = policy_ref
        self.sae = sae
        self.mean_t = mean_t
        self.std_t = std_t
        self.baseline_goal = baseline_goal
        self.baseline_proxy = baseline_proxy
        self.goal_feats = goal_feats
        self.proxy_feats = proxy_feats
        self.lam = lam

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        r_reas = compute_r_reason(
            obs, self.policy_ref, self.sae, self.mean_t, self.std_t,
            self.baseline_goal, self.baseline_proxy,
            self.goal_feats, self.proxy_feats,
        )
        info["r_reason"] = float(r_reas)
        info["v_total"] = float(-r_reas)
        return obs, reward + self.lam * r_reas, term, trunc, info

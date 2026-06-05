"""
Experiment 4, Response 2 — Activation Steering.
When I3 fires (goal-feature causal importance drops), inject a steering vector
(the top goal feature's decoder direction) into the 256-dim representation
before the policy head. No gradient update, no reward change — inference-time only.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

BASE = os.path.dirname(__file__)
E4_DIR = os.path.join(BASE, "outputs/experiment4")


def run_steered_episode(model, sae, env, mean, std, mean_t, std_t,
                        metadata, goal_features, top_goal_feature,
                        baseline_goal_c, alpha, max_steps=200):
    """
    Run one episode with activation steering.
    Returns (total_reward, failed, n_steps, n_steered).
    """
    # Steering vector: decoder direction of top goal feature, in RAW representation space
    with torch.no_grad():
        dec_col = sae.decoder.weight[:, top_goal_feature].detach()  # (256,) normalized space
        v_steer = (dec_col * std_t)  # convert to raw representation space
        v_steer = v_steer / (v_steer.norm() + 1e-8)

    captured = {}
    def hook(_m, _i, out): captured["r"] = out.detach()
    h = model.policy.features_extractor.register_forward_hook(hook)

    top32 = metadata["top32_features"]
    W = np.load(os.path.join(E4_DIR, "graphs/W_interfeature.npy"))
    goal_idx32 = [top32.index(f) for f in goal_features if f in top32]

    obs, info = env.reset()
    total_r = 0.0; done = False; step = 0; n_steered = 0

    while not done and step < max_steps:
        obs_t = (torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                 .permute(0, 3, 1, 2) / 255.0).to(device)
        with torch.no_grad():
            r_t = model.policy.features_extractor(obs_t)  # (1,256) raw, fires hook
            # SAE features for I3 check
            r_norm = (r_t - mean_t) / std_t
            hf = sae.get_feature_activations(r_norm).squeeze(0).cpu().numpy()
            c_live = (np.abs(W[top32][:, top32]) * hf[top32][:, None]).sum(1)
            goal_c_live = float(np.mean([c_live[i] for i in goal_idx32])) if goal_idx32 else 0.0

            # I3: goal causal importance below 60% of baseline → steer
            steer = (baseline_goal_c > 1e-8) and (goal_c_live < 0.6 * baseline_goal_c)
            if steer:
                r_use = r_t + alpha * v_steer.unsqueeze(0)
                n_steered += 1
            else:
                r_use = r_t
            logits = model.policy.action_net(r_use)
            action = int(logits.argmax(-1).item())

        obs, reward, term, trunc, info = env.step(action)
        total_r += reward; done = term or trunc; step += 1

    h.remove()
    return total_r, (total_r < 0.5), step, n_steered


def run_steering_condition(model, sae, mean, std, metadata, goal_features,
                            test_goal, alpha, seeds=(0, 42, 123), n_eps=20):
    from envs.coin_env import make_env_with_info
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)

    top_goal_feature = goal_features[0]
    baseline_goal_c = metadata["goal_c_star_mean"]

    all_fail, all_reward, all_steerfrac = [], [], []
    for seed in seeds:
        env = make_env_with_info(fixed_goal_pos=test_goal)
        env.reset(seed=seed)
        for ep in range(n_eps):
            tr, failed, steps, nst = run_steered_episode(
                model, sae, env, mean, std, mean_t, std_t,
                metadata, goal_features, top_goal_feature, baseline_goal_c, alpha)
            all_fail.append(float(failed)); all_reward.append(tr)
            all_steerfrac.append(nst / max(steps, 1))
        env.close()
    return {
        "alpha": alpha,
        "failure_rate": float(np.mean(all_fail)),
        "mean_reward": float(np.mean(all_reward)),
        "steer_fraction": float(np.mean(all_steerfrac)),
    }

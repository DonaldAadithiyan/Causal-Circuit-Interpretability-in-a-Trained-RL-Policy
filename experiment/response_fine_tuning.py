"""
Experiment 4, Response 3 — Targeted Fine-Tuning (circuit repair).
Fine-tunes ONLY the policy feature extractor to push goal-feature activations
back toward their training baseline (and proxy features down), using a circuit
repair loss — no environment reward. Then re-evaluates failure rate and checks
whether the circuit actually repaired (G* vs G_live).
"""

import sys, os, json, time, copy
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

BASE = os.path.dirname(__file__)
E4_DIR = os.path.join(BASE, "outputs/experiment4")


def collect_test_observations(model, test_goal, n_obs=2000):
    """Collect observations from test-distribution rollouts for the repair loss."""
    from envs.coin_env import make_env_with_info
    obs_list = []
    env = make_env_with_info(fixed_goal_pos=test_goal)
    obs, _ = env.reset(seed=7)
    for _ in range(n_obs):
        action, _ = model.predict(obs, deterministic=True)
        obs_list.append(obs.copy())
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return np.array(obs_list, dtype=np.uint8)


def goal_activation(policy, sae, obs_batch_t, mean_t, std_t, goal_idx, proxy_idx):
    """Differentiable mean goal-feature & proxy-feature activation for a batch."""
    r = policy.features_extractor(obs_batch_t)          # (B,256), differentiable
    r_norm = (r - mean_t) / std_t
    h_pre = sae.encoder(r_norm)
    h = sae.top_k_gate(h_pre)                            # (B, hidden)
    goal_act = h[:, goal_idx].mean()
    proxy_act = h[:, proxy_idx].mean() if len(proxy_idx) else torch.tensor(0.0, device=device)
    return goal_act, proxy_act


def run_targeted_finetuning(base_policy_path, sae, mean, std, metadata,
                            goal_features, proxy_features, test_goal,
                            baseline_goal_act, baseline_proxy_act,
                            seed, ft_steps=5000, lr=1e-5):
    """
    One fine-tuning run. Returns dict with failure_rate, train_reward, circuit_repaired.
    """
    from stable_baselines3 import PPO
    from envs.coin_env import make_env, make_env_with_info
    import gymnasium as gym

    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)
    goal_idx = torch.tensor(goal_features, device=device)
    proxy_idx = torch.tensor(proxy_features, device=device) if proxy_features else torch.tensor([], dtype=torch.long, device=device)

    model = PPO.load(base_policy_path, device=str(device))
    policy = model.policy
    policy.train()
    # Only fine-tune the feature extractor
    for p in policy.parameters():
        p.requires_grad_(False)
    for p in policy.features_extractor.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(policy.features_extractor.parameters(), lr=lr)

    # Collect test observations once
    obs_data = collect_test_observations(model, test_goal, n_obs=2000)
    obs_t_all = (torch.from_numpy(obs_data.astype(np.float32)).permute(0, 3, 1, 2) / 255.0)

    # Repair target: push goal activation UP toward training baseline, proxy DOWN
    target_goal = baseline_goal_act
    target_proxy = baseline_proxy_act * 0.5  # encourage proxy reduction

    bs = 128
    t0 = time.time()
    n_updates = ft_steps // bs
    for it in range(n_updates):
        idx = np.random.choice(len(obs_t_all), bs, replace=False)
        batch = obs_t_all[idx].to(device)
        opt.zero_grad()
        g_act, p_act = goal_activation(policy, sae, batch, mean_t, std_t, goal_idx, proxy_idx)
        loss = (g_act - target_goal) ** 2 + (p_act - target_proxy) ** 2
        loss.backward()
        opt.step()

    elapsed = time.time() - t0
    policy.eval()

    # Evaluate failure rate on test dist
    def eval_fail(random_goal=False, fgp=None, n=20):
        env = make_env(random_goal=random_goal, fixed_goal_pos=fgp)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        rews = []
        for ep in range(n):
            obs, _ = env.reset(seed=seed*100 + ep)
            done = False
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc
            rews.append(info["episode"]["r"])
        env.close()
        rews = np.array(rews)
        return float((rews < 0.5).mean()), float(rews.mean())

    test_fail, test_rew = eval_fail(fgp=test_goal, n=20)
    train_fail, train_rew = eval_fail(random_goal=True, n=20)

    # Circuit repair check: goal activation on test dist after fine-tuning vs baseline
    with torch.no_grad():
        post_obs = collect_test_observations(model, test_goal, n_obs=500)
        pb = (torch.from_numpy(post_obs.astype(np.float32)).permute(0, 3, 1, 2) / 255.0).to(device)
        g_post, p_post = goal_activation(policy, sae, pb, mean_t, std_t, goal_idx, proxy_idx)
        g_post = float(g_post.item()); p_post = float(p_post.item())
    # Repaired if goal activation recovered above 60% of training baseline
    circuit_repaired = bool(g_post > 0.6 * baseline_goal_act)

    del model
    import gc; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "seed": seed, "ft_steps": ft_steps, "elapsed_min": elapsed / 60,
        "test_failure_rate": test_fail, "test_mean_reward": test_rew,
        "train_mean_reward": train_rew, "catastrophic_forgetting": train_rew < 0.6,
        "goal_act_post": g_post, "goal_act_baseline": baseline_goal_act,
        "circuit_repaired": circuit_repaired,
    }

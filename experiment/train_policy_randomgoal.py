"""
Experiment 4, Phase 1 — Train PPO with randomised goal position.
goal random every episode → policy must read goal from observation (builds goal feature).
Eval: train dist (random goal) + test dist (goal fixed at (2,2)).
"""

import sys, os, gc, time, json
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from models.impala_cnn import ImpalaCNNExtractor
from envs.coin_env import make_env
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE, "configs/policy.yaml")
OUT_DIR = os.path.join(BASE, "outputs/experiment4/policy_randomgoal")
TEST_GOAL = (2, 2)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class RewardLogCallback(BaseCallback):
    def __init__(self, log_interval=10000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._last_log = 0
        self.reward_history = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.log_interval:
            infos = self.locals.get("infos", [])
            ep_rews = [i["episode"]["r"] for i in infos if "episode" in i]
            if ep_rews:
                mean_rew = float(np.mean(ep_rews))
                self.reward_history.append((self.num_timesteps, mean_rew))
                log_entry(f"[EXP4] Phase 1 — {self.num_timesteps:,} steps",
                          f"- Mean episodic reward (random goal): {mean_rew:.4f}\n"
                          f"- n_episodes: {len(ep_rews)}")
            self._last_log = self.num_timesteps
        return True


def evaluate(model, n_episodes=50, seed=42, random_goal=False, fixed_goal_pos=None):
    env = make_env(random_goal=random_goal, fixed_goal_pos=fixed_goal_pos)
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


def main(total_timesteps=500_000):
    log_entry("[EXP4] Phase 1 START — PPO with RANDOMISED goal",
              f"- Goal random every episode (forces goal-reading)\n"
              f"- IMPALA CNN, features_dim=256\n"
              f"- Test dist: goal fixed at {TEST_GOAL}\n"
              f"- Device: {device}")

    cfg = load_config()
    ppo_cfg = cfg["ppo"]
    cnn_cfg = cfg["impala_cnn"]
    os.makedirs(OUT_DIR, exist_ok=True)

    train_env = make_vec_env(lambda: make_env(random_goal=True), n_envs=4, seed=0)

    policy_kwargs = dict(
        features_extractor_class=ImpalaCNNExtractor,
        features_extractor_kwargs=dict(features_dim=cnn_cfg["features_dim"]),
        net_arch=[],
    )
    model = PPO("CnnPolicy", train_env, policy_kwargs=policy_kwargs,
                n_steps=ppo_cfg["n_steps"], batch_size=ppo_cfg["batch_size"],
                n_epochs=ppo_cfg["n_epochs"], gamma=ppo_cfg["gamma"],
                gae_lambda=ppo_cfg["gae_lambda"], clip_range=ppo_cfg["clip_range"],
                ent_coef=ppo_cfg["ent_coef"], vf_coef=ppo_cfg["vf_coef"],
                max_grad_norm=ppo_cfg["max_grad_norm"], learning_rate=ppo_cfg["learning_rate"],
                verbose=0, device=str(device))

    reward_cb = RewardLogCallback(log_interval=10000)
    ckpt_cb = CheckpointCallback(save_freq=100000 // 4, save_path=OUT_DIR,
                                 name_prefix="ppo_rg", verbose=0)

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[reward_cb, ckpt_cb])
    elapsed = time.time() - t0

    # If reward hasn't reached 0.7, continue to 750k max
    last_reward = reward_cb.reward_history[-1][1] if reward_cb.reward_history else 0.0
    if last_reward < 0.7 and total_timesteps < 750_000:
        extra = 250_000
        log_entry("[EXP4] Phase 1 — reward < 0.7, extending training",
                  f"- last_reward: {last_reward:.4f}\n- training {extra:,} more steps")
        model.learn(total_timesteps=extra, callback=[reward_cb], reset_num_timesteps=False)
        elapsed = time.time() - t0

    final_path = os.path.join(OUT_DIR, "ppo_final")
    model.save(final_path)

    # Evaluate
    train_mean, train_std, _ = evaluate(model, n_episodes=50, random_goal=True)
    test_mean, test_std, test_fail = evaluate(model, n_episodes=50, fixed_goal_pos=TEST_GOAL)

    results = {
        "train_mean_reward": train_mean, "train_std_reward": train_std,
        "test_mean_reward": test_mean, "test_std_reward": test_std,
        "test_failure_rate": test_fail,
        "test_goal_pos": list(TEST_GOAL),
        "reward_history": reward_cb.reward_history,
        "elapsed_min": elapsed / 60,
    }
    with open(os.path.join(OUT_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    log_entry("[EXP4] Phase 1 COMPLETE",
              f"- Train (random goal): {train_mean:.4f} ± {train_std:.4f}\n"
              f"- Test (goal at {TEST_GOAL}): {test_mean:.4f} ± {test_std:.4f}\n"
              f"- Test failure rate: {test_fail:.3f}\n"
              f"- Elapsed: {elapsed/60:.1f} min")

    print(f"\n{'='*60}")
    print(f"EXP4 PHASE 1 COMPLETE")
    print(f"Train (random goal): {train_mean:.4f} ± {train_std:.4f}")
    print(f"Test  (goal {TEST_GOAL}): {test_mean:.4f} ± {test_std:.4f}, fail={test_fail:.3f}")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print(f"{'='*60}\n")

    del train_env
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return results


if __name__ == "__main__":
    main()

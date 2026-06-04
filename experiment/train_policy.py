"""
Phase 1 — Train PPO policy with IMPALA CNN on CoinCollect training distribution.
Saves checkpoints every 100k steps and a final evaluation to outputs/checkpoints/.
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
from utils.logging_utils import log_entry, init_log

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE, "configs/policy.yaml")
OUT_DIR = os.path.join(BASE, "outputs")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class RewardLogCallback(BaseCallback):
    def __init__(self, log_interval: int = 10000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._last_log = 0
        self.reward_history = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.log_interval:
            infos = self.locals.get("infos", [])
            ep_rews = [
                info["episode"]["r"]
                for info in infos
                if "episode" in info
            ]
            if ep_rews:
                mean_rew = float(np.mean(ep_rews))
                self.reward_history.append((self.num_timesteps, mean_rew))
                log_entry(
                    f"Phase 1 — {self.num_timesteps:,} steps",
                    f"- Mean episodic reward: {mean_rew:.4f}\n"
                    f"- n_episodes: {len(ep_rews)}",
                )
            self._last_log = self.num_timesteps
        return True


def evaluate(model, goal_fixed: bool, n_episodes: int = 50, seed: int = 42):
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
    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    init_log()
    log_entry(
        "Phase 1 START — PPO training on CoinCollect (MiniGrid)",
        "- Environment: CoinCollect 8x8, goal_fixed=True\n"
        "- Policy: PPO + IMPALA CNN, features_dim=256\n"
        "- procgen unavailable on Apple Silicon — using MiniGrid as planned\n"
        f"- Device: {device}",
    )

    cfg = load_config()
    ppo_cfg = cfg["ppo"]
    cnn_cfg = cfg["impala_cnn"]

    os.makedirs(os.path.join(OUT_DIR, "checkpoints"), exist_ok=True)

    # Vectorized env for training (SB3 monitor wraps episodes automatically)
    train_env = make_vec_env(
        lambda: make_env(goal_fixed=True),
        n_envs=4,
        seed=0,
    )

    policy_kwargs = dict(
        features_extractor_class=ImpalaCNNExtractor,
        features_extractor_kwargs=dict(features_dim=cnn_cfg["features_dim"]),
        net_arch=[],  # Direct: features → policy/value heads (no extra MLP)
    )

    model = PPO(
        "CnnPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        learning_rate=ppo_cfg["learning_rate"],
        verbose=0,
        device=str(device),
    )

    log_entry(
        "Phase 1 — Model created",
        f"- Total params: {sum(p.numel() for p in model.policy.parameters()):,}\n"
        f"- Features dim: {cnn_cfg['features_dim']}\n"
        f"- net_arch: [] (direct to policy/value heads)\n"
        f"- n_steps: {ppo_cfg['n_steps']}, batch_size: {ppo_cfg['batch_size']}, n_envs: 4",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=ppo_cfg.get("save_freq", 100000) // 4,  # per-env steps
        save_path=os.path.join(OUT_DIR, "checkpoints"),
        name_prefix="ppo_impala",
        verbose=0,
    )
    reward_cb = RewardLogCallback(log_interval=cfg["training"]["log_interval"])

    t0 = time.time()
    model.learn(
        total_timesteps=ppo_cfg["total_timesteps"],
        callback=[checkpoint_cb, reward_cb],
    )
    elapsed = time.time() - t0

    final_path = os.path.join(OUT_DIR, "checkpoints", "ppo_final")
    model.save(final_path)

    log_entry(
        "Phase 1 — Training complete",
        f"- Elapsed: {elapsed/60:.1f} min\n"
        f"- Total timesteps: {ppo_cfg['total_timesteps']:,}\n"
        f"- Checkpoint saved: {final_path}",
    )

    # Evaluate on training distribution
    mean_train, std_train = evaluate(model, goal_fixed=True, n_episodes=50)
    mean_test, std_test = evaluate(model, goal_fixed=False, n_episodes=50)

    gap = mean_train - mean_test
    results = {
        "train_mean_reward": mean_train,
        "train_std_reward": std_train,
        "test_mean_reward": mean_test,
        "test_std_reward": std_test,
        "generalization_gap": gap,
        "reward_history": reward_cb.reward_history,
        "elapsed_min": elapsed / 60,
        "device": str(device),
    }
    results_path = os.path.join(OUT_DIR, "checkpoints", "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log_entry(
        "Phase 1 — Evaluation complete",
        f"- Train distribution: mean={mean_train:.4f}, std={std_train:.4f}\n"
        f"- Test distribution (goal randomised): mean={mean_test:.4f}, std={std_test:.4f}\n"
        f"- Generalization gap: {gap:.4f} (positive = goal misgeneralization confirmed)\n"
        f"- Results saved: {results_path}",
    )

    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE")
    print(f"Train reward: {mean_train:.4f} ± {std_train:.4f}")
    print(f"Test reward:  {mean_test:.4f} ± {std_test:.4f}")
    print(f"Gap:          {gap:.4f}")
    print(f"Elapsed:      {elapsed/60:.1f} min")
    print(f"{'='*60}\n")

    # Free memory
    del train_env
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


if __name__ == "__main__":
    main()

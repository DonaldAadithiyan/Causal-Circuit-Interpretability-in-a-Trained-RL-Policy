# Experiment Log
*Newest entries at top.*

## [13:33] Phase 1 — 260,000 steps
- Mean episodic reward: 1.0000
- n_episodes: 1

## [13:26] Phase 1 — 190,000 steps
- Mean episodic reward: 1.0000
- n_episodes: 1

## [13:16] Phase 1 — 90,000 steps
- Mean episodic reward: 1.0000
- n_episodes: 1

## [13:09] Phase 1 — Model created
- Total params: 624,200
- Features dim: 256
- net_arch: [] (direct to policy/value heads)
- n_steps: 2048, batch_size: 64, n_envs: 4

## [13:08] Phase 1 START — PPO training on CoinCollect (MiniGrid)
- Environment: CoinCollect 8x8, goal_fixed=True
- Policy: PPO + IMPALA CNN, features_dim=256
- procgen unavailable on Apple Silicon — using MiniGrid as planned
- Device: mps

## [13:07] Phase 1 — Model created
- Total params: 624,200
- Features dim: 256
- net_arch: [] (direct to policy/value heads)
- n_steps: 2048, batch_size: 64, n_envs: 4

## [13:07] Phase 1 START — PPO training on CoinCollect (MiniGrid)
- Environment: CoinCollect 8x8, goal_fixed=True
- Policy: PPO + IMPALA CNN, features_dim=256
- procgen unavailable on Apple Silicon — using MiniGrid as planned
- Device: mps

## [Setup] Environment and tooling
- Date: 2026-06-04
- procgen: unavailable on Apple Silicon (pip reports no matching distribution). Switched immediately to MiniGrid as planned.
- MiniGrid 3.1.0 installed successfully.
- PyTorch 2.12.0 with MPS backend available.
- SB3 2.8.0 installed.
- Device: MPS (Apple M-series unified memory)
- All directories created: experiment/models, experiment/envs, experiment/utils, experiment/configs, experiment/outputs/...


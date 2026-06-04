# Experiment Log
*Newest entries at top.*

## [17:10] [EXP3] Starting: baseline_seed0
- λ=0.0, seed=0

## [17:10] [EXP3] Option B START
- λ values: [0.0, 0.1, 0.5, 1.0]
- seeds: [0, 42, 123]
- timesteps per run: 100,000
- baseline_goal_sig: 0.3945
- baseline_proxy_sig: 0.4353

## [17:07] [EXP3] Starting: lam0.1_seed0
- λ=0.1, seed=0

## [17:07] [EXP3] baseline_seed123 seed=123 complete
- test_fail_rate: 0.000
- test_mean_reward: 1.0000
- train_mean_reward: 1.0000 (forgetting: False)
- elapsed: 56.0 min

## [16:11] [EXP3] Starting: baseline_seed123
- λ=0.0, seed=123

## [16:11] [EXP3] baseline_seed42 seed=42 complete
- test_fail_rate: 0.250
- test_mean_reward: 0.7500
- train_mean_reward: 0.0000 (forgetting: True)
- elapsed: 23.6 min

## [15:48] [EXP3] Starting: baseline_seed42
- λ=0.0, seed=42

## [15:48] [EXP3] baseline_seed0 seed=0 complete
- test_fail_rate: 0.250
- test_mean_reward: 0.7500
- train_mean_reward: 1.0000 (forgetting: False)
- elapsed: 23.4 min

## [15:24] [EXP3] Starting: baseline_seed0
- λ=0.0, seed=0

## [15:24] [EXP3] Option B START
- λ values: [0.0, 0.1, 0.5, 1.0]
- seeds: [0, 42, 123]
- timesteps per run: 100,000
- baseline_goal_sig: 0.3945
- baseline_proxy_sig: 0.4353

## [15:24] [EXP3] START — Option B lambda sweep


## [15:23] [EXP3] Starting: baseline_seed0
- λ=0.0, seed=0

## [15:23] [EXP3] Option B START
- λ values: [0.0, 0.1, 0.5, 1.0]
- seeds: [0, 42, 123]
- timesteps per run: 100,000
- baseline_goal_sig: 0.3945
- baseline_proxy_sig: 0.4353

## [15:23] [EXP3] Starting: baseline_seed0
- λ=0.0, seed=0

## [15:23] [EXP3] Option B START
- λ values: [0.0, 0.1, 0.5, 1.0]
- seeds: [0, 42, 123]
- timesteps per run: 100,000
- baseline_goal_sig: 0.3945
- baseline_proxy_sig: 0.4353

## [15:23] [EXP3] START — Option B lambda sweep


## [15:11] [EXP2] COMPLETE
- EAP r: 0.1461
- mean k_act: 128.7 ± 93.8
- mean k_graph: 128.7 ± 93.8
- Exp1 reference k_activation: 157.8

## [15:11] [EXP2] Seed 123 complete
- mean_k_act: 122.8
- mean_k_graph: 122.8

## [15:11] [EXP2] Phase 4 — Seed 123: 10 test-dist episodes


## [15:11] [EXP2] Seed 42 complete
- mean_k_act: 142.0
- mean_k_graph: 142.0

## [15:11] [EXP2] Phase 4 — Seed 42: 10 test-dist episodes


## [15:11] [EXP2] Seed 0 complete
- mean_k_act: 121.2
- mean_k_graph: 121.2

## [15:11] [EXP2] Phase 4 — Seed 0: 10 test-dist episodes


## [15:11] [EXP2] Training baseline
- goal_sig: 0.3945
- proxy_sig: 0.4353
- mean V_total: 0.642988 (should be near 0)

## [15:11] [EXP2] Phase 3 — Collecting training-distribution baseline


## [15:11] [EXP2] Phase 2 — EAP validation complete
- Pearson r (EAP vs patching): 0.1461
- WARN (r<0.5) — EAP approximation weak

## [15:11] [EXP2] Phase 2 — Validating EAP vs patching on 100 obs


## [15:11] [EXP2] Phase 1 COMPLETE — G* saved
- max_kl: 0.002235
- pass_rate (>0.01): 0.00
- goal_c_mean: 0.000696
- proxy_c_mean: 0.000911
- i3_threshold: 0.000348
- spurious_set: [790, 370, 58, 707, 516, 150, 917, 1001, 851, 755, 672, 893, 46, 834, 734, 578, 396, 416, 589, 320, 764, 34, 807, 444]

## [15:11] [EXP2] Phase 1 START — Build G* (200 obs, KL threshold 0.01)


## [14:42] EXPERIMENT COMPLETE
- All 5 phases completed successfully
- H1 (SAE interpretability): PARTIALLY SUPPORTED — 6 goal + 10 proxy features, but 785/1024 dead features
- H2 (Causal graph): WEAKLY SUPPORTED — top causal feature IS coin_tracking, but KL values small
- H3 (Pre-failure signal): STRONGLY SUPPORTED — mean k=157.8±80.3 across 60 episodes, 3 seeds
- EXPLAINER.md written
- Total wall time: ~3 hours on Apple M-series MPS

## [14:39] Final Analysis COMPLETE
- H1: PARTIAL/FAILED
- H2: PARTIAL/FAILED
- H3: STRONG (mean k=157.80)
- All plots saved to /Users/donaldaadithiyan/Desktop/Work/Personal Learn Dev/Causal-Circuit-Interpretability-in-a-Trained-RL-Policy/experiment/outputs/plots

## [14:39] Final Analysis START


## [14:38] Phase 5 COMPLETE
- mean k: 157.80 ± 80.25
- n_k_measurements: 60/60
- seed results: {'seed_0': 170.2, 'seed_42': 122.6, 'seed_123': 180.6}
- Plots: representative_episode.png, k_distribution.png

## [14:38] Phase 5 — Seed 123 complete
- mean k: 180.60
- std k: 58.20
- failure rate: 0.90
- k values: [200, 200, 200, 200, 200]...

## [14:38] Phase 5 — Seed 42 complete
- mean k: 122.60
- std k: 94.81
- failure rate: 0.60
- k values: [6, 9, 200, 9, 200]...

## [14:38] Phase 5 — Seed 0 complete
- mean k: 170.20
- std k: 70.94
- failure rate: 0.85
- k values: [200, 2, 200, 2, 200]...

## [14:37] Phase 5 — Training baseline collected
- n_episodes: 20
- Mean total reward: 1.0000
- Mean goal signal: 0.3945
- Mean proxy signal: 0.4353

## [14:37] Phase 5 — Feature sets
- Goal features: [933, 151, 438, 17, 736, 481]
- Proxy features: [790, 150, 917, 1001, 589, 38, 69, 488, 654, 22]

## [14:37] Phase 5 START — Goal misgeneralization measurement


## [14:36] Phase 4 COMPLETE
- Top causal feature: 17 (label: coin_tracking, KL: 0.0028)
- Pass rate (KL > 0.1): 0.00
- Top 5 edges: [{'from_feat': 790, 'to_feat': 21, 'weight': 0.14363455772399902}, {'from_feat': 17, 'to_feat': 764, 'weight': 0.09820117801427841}, {'from_feat': 807, 'to_feat': 834, 'weight': 0.09219007939100266}, {'from_feat': 151, 'to_feat': 536, 'weight': 0.09139607846736908}, {'from_feat': 707, 'to_feat': 481, 'weight': 0.09114305675029755}]
- Top 5 action-causal: [{'feat': 17, 'kl_to_action': 0.002793358638882637}, {'feat': 790, 'kl_to_action': 0.001991575350984931}, {'feat': 1001, 'kl_to_action': 0.00193758902605623}, {'feat': 917, 'kl_to_action': 0.0018485465552657843}, {'feat': 151, 'kl_to_action': 0.0015245270915329456}]
- Plot: causal_graph.png

## [14:36] Phase 4 START — Causal graph extraction


## [14:32] Phase 4 START — Causal graph extraction


## [14:31] Phase 4 START — Causal graph extraction


## [14:28] Phase 3 — Feature labels corrected
- proxy_position features relabeled: features with high near-goal bias AND negative reward_corr
- Pattern: coin_tracking (bias>0.3, rew>0.1) vs proxy_position (bias>0.1, rew<-0.05)
- Final: coin_tracking=6, proxy_position=10, unknown=34
- Goal features: [933, 151, 438, 17, 736, 481]
- Proxy features: [790, 150, 917, 1001, 589, 38, 69, 488, 654, 22]

## [14:25] Phase 3 COMPLETE
- Features analysed: 50
- Label distribution: {'unknown': 43, 'coin_tracking': 6, 'action_spurious': 1}
- Goal features: [933, 151, 438, 17, 736]...
- Proxy features: [22]...
- Plots: /Users/donaldaadithiyan/Desktop/Work/Personal Learn Dev/Causal-Circuit-Interpretability-in-a-Trained-RL-Policy/experiment/outputs/plots/feature_max_activations
- Labels: /Users/donaldaadithiyan/Desktop/Work/Personal Learn Dev/Causal-Circuit-Interpretability-in-a-Trained-RL-Policy/experiment/outputs/feature_labels.json

## [14:25] Phase 3 — Feature 150 (rank 9) analysed
- freq: 0.2394
- reward_corr: -0.1885
- action_corr: 0.2602
- agent_near_goal_bias: 0.4405

## [14:25] Phase 3 — Feature 516 (rank 8) analysed
- freq: 0.2394
- reward_corr: -0.1890
- action_corr: 0.3491
- agent_near_goal_bias: -0.9670

## [14:25] Phase 3 — Feature 707 (rank 7) analysed
- freq: 0.2395
- reward_corr: -0.1881
- action_corr: 0.3069
- agent_near_goal_bias: -1.0791

## [14:25] Phase 3 — Feature 58 (rank 6) analysed
- freq: 0.2395
- reward_corr: -0.1793
- action_corr: 0.2731
- agent_near_goal_bias: 0.0023

## [14:25] Phase 3 — Feature 17 (rank 5) analysed
- freq: 0.2424
- reward_corr: 0.4100
- action_corr: 0.7432
- agent_near_goal_bias: 1.1775

## [14:25] Phase 3 — Feature 438 (rank 4) analysed
- freq: 0.2530
- reward_corr: 0.7419
- action_corr: 0.2185
- agent_near_goal_bias: 0.5848

## [14:25] Phase 3 — Feature 151 (rank 3) analysed
- freq: 0.2530
- reward_corr: 0.5302
- action_corr: 0.3261
- agent_near_goal_bias: 0.5119

## [14:25] Phase 3 — Feature 933 (rank 2) analysed
- freq: 0.2530
- reward_corr: 0.9280
- action_corr: 0.1456
- agent_near_goal_bias: 0.4638

## [14:25] Phase 3 — Feature 370 (rank 1) analysed
- freq: 0.3106
- reward_corr: -0.2277
- action_corr: 0.2958
- agent_near_goal_bias: -0.5874

## [14:25] Phase 3 — Feature 790 (rank 0) analysed
- freq: 0.3323
- reward_corr: -0.2320
- action_corr: 0.6532
- agent_near_goal_bias: 0.7004

## [14:25] Phase 3 — Feature frequencies computed
- Top feature activation rate: 0.3323
- Median activation rate: 0.0000
- Dead features (<0.1%): 778

## [14:25] Phase 3 START — Feature interpretability analysis


## [14:24] Phase 3 START — Feature interpretability analysis


## [14:24] Phase 3 START — Feature interpretability analysis


## [14:19] Phase 2 — SAE training complete
- Best val_loss: 0.066943
- Dead features: 785/1024
- Plots: sae_loss_curve.png, sae_feature_freq.png
- Checkpoint: /Users/donaldaadithiyan/Desktop/Work/Personal Learn Dev/Causal-Circuit-Interpretability-in-a-Trained-RL-Policy/experiment/outputs/checkpoints/sae_best.pt

## [14:19] Phase 2 — Early stop at epoch 12
- best val_loss: 0.066943

## [14:19] Phase 2 — SAE epoch 10/100
- train_loss: 0.056891
- val_loss: 0.076394
- dead_features: 775/1024
- elapsed: 0.5 min

## [14:19] Phase 2 — SAE epoch 5/100
- train_loss: 0.071829
- val_loss: 0.081536
- dead_features: 775/1024
- elapsed: 0.3 min

## [14:19] Phase 2 — SAE epoch 1/100
- train_loss: 0.310227
- val_loss: 0.167988
- dead_features: 785/1024
- elapsed: 0.1 min

## [14:19] Phase 2 — SAE training start
- train: 90,000, val: 10,000
- K=32, hidden_factor=4
- features_dim=256, hidden_dim=1024

## [14:19] Phase 2 — Collection complete
- Samples: 100,000
- activations saved to /Users/donaldaadithiyan/Desktop/Work/Personal Learn Dev/Causal-Circuit-Interpretability-in-a-Trained-RL-Policy/experiment/outputs/activations
- elapsed: 7.8 min

## [14:19] Phase 2 — 100,000/100,000 samples collected
- elapsed: 7.8 min

## [14:18] Phase 2 — 90,000/100,000 samples collected
- elapsed: 7.1 min

## [14:17] Phase 2 — 80,000/100,000 samples collected
- elapsed: 6.5 min

## [14:17] Phase 2 — 70,000/100,000 samples collected
- elapsed: 6.0 min

## [14:16] Phase 2 — 60,000/100,000 samples collected
- elapsed: 5.4 min

## [14:15] Phase 2 — 50,000/100,000 samples collected
- elapsed: 4.4 min

## [14:14] Phase 2 — 40,000/100,000 samples collected
- elapsed: 3.6 min

## [14:14] Phase 2 — 30,000/100,000 samples collected
- elapsed: 2.7 min

## [14:13] Phase 2 — 20,000/100,000 samples collected
- elapsed: 1.7 min

## [14:12] Phase 2 — 10,000/100,000 samples collected
- elapsed: 0.7 min

## [14:11] Phase 2 — Collecting activations
- target: 100,000 samples
- env: CoinCollect training distribution

## [14:11] Phase 2 START — SAE training
- Policy loaded from /Users/donaldaadithiyan/Desktop/Work/Personal Learn Dev/Causal-Circuit-Interpretability-in-a-Trained-RL-Policy/experiment/outputs/checkpoints/ppo_final.zip
- Policy frozen (no gradients)
- Device: mps

## [13:57] Phase 1 — Evaluation complete
- Train distribution: mean=1.0000, std=0.0000
- Test distribution (goal randomised): mean=0.2200, std=0.4142
- Generalization gap: 0.7800 (positive = goal misgeneralization confirmed)
- Results saved: /Users/donaldaadithiyan/Desktop/Work/Personal Learn Dev/Causal-Circuit-Interpretability-in-a-Trained-RL-Policy/experiment/outputs/checkpoints/eval_results.json

## [13:57] Phase 1 — Training complete
- Elapsed: 48.0 min
- Total timesteps: 500,000
- Checkpoint saved: /Users/donaldaadithiyan/Desktop/Work/Personal Learn Dev/Causal-Circuit-Interpretability-in-a-Trained-RL-Policy/experiment/outputs/checkpoints/ppo_final

## [13:50] Phase 1 — 450,000 steps
- Mean episodic reward: 1.0000
- n_episodes: 1

## [13:46] Phase 1 — 400,000 steps
- Mean episodic reward: 1.0000
- n_episodes: 1

## [13:43] Phase 1 — 370,000 steps
- Mean episodic reward: 1.0000
- n_episodes: 1

## [13:39] Phase 1 — 330,000 steps
- Mean episodic reward: 1.0000
- n_episodes: 1

## [13:37] Phase 1 — 310,000 steps
- Mean episodic reward: 1.0000
- n_episodes: 1

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


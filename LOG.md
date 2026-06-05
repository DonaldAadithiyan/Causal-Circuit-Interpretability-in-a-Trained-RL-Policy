# Experiment Log
*Newest entries at top.*

## [09:17] [EXP2b+3] ALL EXPERIMENTS COMPLETE — EXPLAINER2/3 rewritten
- W-matrix: r=0.893 vs patching (EAP was 0.146) — gradient-free causal edges WORK
- SAEv2: 100/384 dead (was 785/1024), val MSE 4.75e-6
- KEY FINDING: no feature tracks actual goal (max goal_track_corr=0.005) — policy has NO goal representation
- Exp2b graded shift: k_graph==k_activation EXACTLY at disp 1/2/3/random (Δ=0.0) — graph adds no lead time
- Exp3: R_reason on confounded features → 100%% failure at all λ>0 (baseline λ=0 → 16.7%%)
- Unifying conclusion: cannot detect/graph/correct a goal representation that does not exist

## [22:46] [EXP2b] COMPLETE
- d=1: k_act=85.3, k_graph=85.3, Δ=+0.0
- d=2: k_act=148.5, k_graph=148.5, Δ=+0.0
- d=3: k_act=135.0, k_graph=135.0, Δ=+0.0
- random: k_act=168.0, k_graph=168.0, Δ=+0.0

## [22:46] [EXP2b] Displacement=-1 complete
- k_act:   168.0 ± 71.5
- k_graph: 168.0 ± 71.5
- n_k_graph: 30/30

## [22:45] [EXP2b] Running displacement=-1


## [22:45] [EXP2b] Displacement=3 complete
- k_act:   135.0 ± 91.9
- k_graph: 135.0 ± 91.9
- n_k_graph: 30/30

## [22:45] [EXP2b] Running displacement=3


## [22:45] [EXP2b] Displacement=2 complete
- k_act:   148.5 ± 85.3
- k_graph: 148.5 ± 85.3
- n_k_graph: 30/30

## [22:44] [EXP2b] Running displacement=2


## [22:44] [EXP2b] Displacement=1 complete
- k_act:   85.3 ± 93.6
- k_graph: 85.3 ± 93.6
- n_k_graph: 30/30

## [22:44] [EXP2b] COMPLETE
- d=1: k_act=104.7, k_graph=104.7, Δ=+0.0
- d=2: k_act=142.1, k_graph=142.1, Δ=+0.0
- d=3: k_act=128.5, k_graph=128.5, Δ=+0.0
- random: k_act=134.8, k_graph=134.8, Δ=+0.0

## [22:44] [EXP2b] Displacement=-1 complete
- k_act:   134.8 ± 92.2
- k_graph: 134.8 ± 92.2
- n_k_graph: 30/30

## [22:44] [EXP2b] Running displacement=1


## [22:44] [EXP2b] START — graded shift measurement with W-based G_live
- displacements: [1, 2, 3, -1(random)]
- 10 episodes × 3 seeds per level
- baseline_goal_sig: 0.0664, baseline_proxy_sig: 0.0907
- W shape: (384, 384)

## [22:44] [EXP2b] Computing SAEv2 training-distribution baseline


## [22:43] [EXP2b] Running displacement=-1


## [22:43] [EXP2b] Displacement=3 complete
- k_act:   128.5 ± 94.0
- k_graph: 128.5 ± 94.0
- n_k_graph: 30/30

## [22:43] [EXP2b] Running displacement=3


## [22:43] [EXP2b] Displacement=2 complete
- k_act:   142.1 ± 88.4
- k_graph: 142.1 ± 88.4
- n_k_graph: 30/30

## [22:42] [EXP2b] Running displacement=2


## [22:42] [EXP2b] Displacement=1 complete
- k_act:   104.7 ± 95.3
- k_graph: 104.7 ± 95.3
- n_k_graph: 30/30

## [22:42] [EXP2b] Running displacement=1


## [22:42] [EXP2b] START — graded shift measurement with W-based G_live
- displacements: [1, 2, 3, -1(random)]
- 10 episodes × 3 seeds per level
- baseline_goal_sig: 0.0664, baseline_proxy_sig: 0.0907
- W shape: (384, 384)

## [22:42] [EXP2b] Computing SAEv2 training-distribution baseline


## [22:20] [W-Matrix] G* built and saved
- max_kl c*: 0.001492
- goal_c_mean: 0.001378
- proxy_c_mean: 0.000002
- W validation r: 0.8934
- Saved: W_interfeature.npy, G_star_v2_metadata.json

## [22:20] [W-Matrix] Building G* from W


## [22:20] [W-Matrix] Validation complete
- Pearson r (W vs patching): 0.8934
- PASS r>0.5
- PASS r>0.3

## [22:20] [W-Matrix] Validating W against activation patching on 200 obs


## [22:20] [W-Matrix] Computing W = D^T @ W_enc^T from SAEv2


## [22:17] [W-Matrix] Building G* from W


## [22:17] [W-Matrix] Validation complete
- Pearson r (W vs patching): 0.8565
- PASS r>0.5
- PASS r>0.3

## [22:17] [W-Matrix] Validating W against activation patching on 200 obs


## [22:17] [W-Matrix] Computing W = D^T @ W_enc^T from SAEv2


## [22:17] [W-Matrix] Building G* from W


## [22:17] [W-Matrix] Validation complete
- Pearson r (W vs patching): 0.7033
- PASS r>0.5
- PASS r>0.3

## [22:17] [W-Matrix] Validating W against activation patching on 200 obs


## [22:17] [W-Matrix] Computing W = D^T @ W_enc^T from SAEv2


## [21:35] [EXP3] COMPLETE — EXPLAINER2.md and EXPLAINER3.md written


## [21:35] [EXP3] Option B COMPLETE
- λ=0.0: fail_rate=0.167 ± 0.118
- λ=0.1: fail_rate=1.000 ± 0.000
- λ=0.5: fail_rate=1.000 ± 0.000
- λ=1.0: fail_rate=1.000 ± 0.000
- baseline fail_rate: 0.167

## [21:35] [EXP3] lam1.0_seed123 seed=123 complete
- test_fail_rate: 1.000
- test_mean_reward: 0.0000
- train_mean_reward: 0.0000 (forgetting: True)
- elapsed: 31.6 min

## [21:03] [EXP3] Starting: lam1.0_seed123
- λ=1.0, seed=123

## [21:03] [EXP3] lam1.0_seed42 seed=42 complete
- test_fail_rate: 1.000
- test_mean_reward: 0.0000
- train_mean_reward: 0.0000 (forgetting: True)
- elapsed: 28.5 min

## [21:01] [SAEv2] Test-distribution labeling COMPLETE
- label_counts: {'unknown': 50}
- goal_features: [89, 111, 272, 379, 139]
- proxy_features: []

## [21:01] [SAEv2] No features passed goal threshold — using top-5 by goal_track_corr
- goal_features (fallback): [89, 111, 272, 379, 139]

## [20:59] [SAEv2] Test-distribution feature labeling START
- collecting 30,000 test-distribution steps to decouple goal vs position

## [20:56] [SAEv2] Feature re-analysis COMPLETE
- label_counts: {'unknown': 34, 'proxy_position': 16}
- goal_features: []
- proxy_features: [374, 248, 179, 162, 36, 31, 174, 315, 314, 79, 310, 306, 4, 16, 200, 203]

## [20:56] [SAEv2] Feature re-analysis START
- hidden_dim=384, dead=100

## [20:52] [SAEv2] Feature re-analysis COMPLETE
- label_counts: {'unknown': 35, 'proxy_position': 15}
- goal_features: []
- proxy_features: [248, 179, 162, 36, 31, 174, 315, 314, 79, 310, 306, 4, 16, 200, 203]

## [20:52] [SAEv2] Feature re-analysis START
- hidden_dim=384, dead=100

## [20:49] [SAEv2] Retraining complete
- best val_loss: 0.000005
- dead_features: 100/384
- elapsed: 1.9 min

## [20:49] [SAEv2] Epoch 80/80
- train_loss: 0.000365
- val_loss: 0.000005
- dead_features: 100/384
- elapsed: 1.9 min

## [20:49] [SAEv2] Epoch 75/80
- train_loss: 0.000505
- val_loss: 0.000033
- dead_features: 100/384
- elapsed: 1.8 min

## [20:49] [SAEv2] Epoch 70/80
- train_loss: 0.002638
- val_loss: 0.000073
- dead_features: 99/384
- elapsed: 1.7 min

## [20:48] [SAEv2] Epoch 65/80
- train_loss: 0.001583
- val_loss: 0.004040
- dead_features: 98/384
- elapsed: 1.6 min

## [20:48] [SAEv2] Epoch 60/80
- train_loss: 0.003621
- val_loss: 0.002500
- dead_features: 97/384
- elapsed: 1.5 min

## [20:48] [SAEv2] Epoch 55/80
- train_loss: 0.013500
- val_loss: 0.000277
- dead_features: 99/384
- elapsed: 1.4 min

## [20:48] [SAEv2] Epoch 50/80
- train_loss: 0.005867
- val_loss: 0.001291
- dead_features: 102/384
- elapsed: 1.3 min

## [20:48] [SAEv2] Epoch 45/80
- train_loss: 0.007444
- val_loss: 0.007372
- dead_features: 103/384
- elapsed: 1.1 min

## [20:48] [SAEv2] Epoch 40/80
- train_loss: 0.006206
- val_loss: 0.000675
- dead_features: 104/384
- elapsed: 1.0 min

## [20:48] [SAEv2] Epoch 35/80
- train_loss: 0.011525
- val_loss: 0.000680
- dead_features: 106/384
- elapsed: 0.9 min

## [20:48] [SAEv2] Epoch 30/80
- train_loss: 0.016243
- val_loss: 0.002616
- dead_features: 106/384
- elapsed: 0.8 min

## [20:47] [SAEv2] Epoch 25/80
- train_loss: 0.021361
- val_loss: 0.005232
- dead_features: 115/384
- elapsed: 0.6 min

## [20:47] [SAEv2] Epoch 20/80
- train_loss: 0.033191
- val_loss: 0.002873
- dead_features: 117/384
- elapsed: 0.5 min

## [20:47] [SAEv2] Epoch 15/80
- train_loss: 0.105105
- val_loss: 0.022107
- dead_features: 126/384
- elapsed: 0.4 min

## [20:47] [SAEv2] Epoch 10/80
- train_loss: 0.094205
- val_loss: 0.023142
- dead_features: 144/384
- elapsed: 0.3 min

## [20:47] [SAEv2] Epoch 5/80
- train_loss: 0.051064
- val_loss: 0.004379
- dead_features: 155/384
- elapsed: 0.1 min

## [20:47] [SAEv2] Epoch 1/80
- train_loss: 0.278599
- val_loss: 0.037586
- dead_features: 161/384
- elapsed: 0.0 min

## [20:47] [SAEv2] Retraining SAE with resampling
- hidden_factor=1.5 → hidden_dim=384.0
- K=32, resample_every=50 batches
- train=90,000, val=10,000

## [20:42] [SAEv2] Retraining complete
- best val_loss: 0.000727
- dead_features: 230/512
- elapsed: 2.2 min

## [20:42] [SAEv2] Epoch 80/80
- train_loss: 0.000378
- val_loss: 0.000727
- dead_features: 218/512
- elapsed: 2.2 min

## [20:42] [SAEv2] Epoch 75/80
- train_loss: 0.000406
- val_loss: 0.000884
- dead_features: 217/512
- elapsed: 2.1 min

## [20:42] [SAEv2] Epoch 70/80
- train_loss: 0.002364
- val_loss: 0.044517
- dead_features: 214/512
- elapsed: 2.0 min

## [20:42] [SAEv2] Epoch 65/80
- train_loss: 0.003412
- val_loss: 0.001938
- dead_features: 215/512
- elapsed: 1.8 min

## [20:42] [SAEv2] Epoch 60/80
- train_loss: 0.003384
- val_loss: 0.002080
- dead_features: 215/512
- elapsed: 1.7 min

## [20:41] [SAEv2] Epoch 55/80
- train_loss: 0.002630
- val_loss: 0.021470
- dead_features: 217/512
- elapsed: 1.6 min

## [20:41] [SAEv2] Epoch 50/80
- train_loss: 0.006831
- val_loss: 0.010898
- dead_features: 215/512
- elapsed: 1.4 min

## [20:41] [SAEv2] Epoch 45/80
- train_loss: 0.008130
- val_loss: 0.015388
- dead_features: 217/512
- elapsed: 1.3 min

## [20:41] [SAEv2] Epoch 40/80
- train_loss: 0.009893
- val_loss: 0.133278
- dead_features: 219/512
- elapsed: 1.2 min

## [20:41] [SAEv2] Epoch 35/80
- train_loss: 0.012534
- val_loss: 0.024643
- dead_features: 217/512
- elapsed: 1.0 min

## [20:41] [SAEv2] Epoch 30/80
- train_loss: 0.021410
- val_loss: 0.139074
- dead_features: 219/512
- elapsed: 0.9 min

## [20:41] [SAEv2] Epoch 25/80
- train_loss: 0.029992
- val_loss: 0.033305
- dead_features: 223/512
- elapsed: 0.8 min

## [20:40] [SAEv2] Epoch 20/80
- train_loss: 0.050784
- val_loss: 0.044192
- dead_features: 231/512
- elapsed: 0.6 min

## [20:40] [SAEv2] Epoch 15/80
- train_loss: 0.085656
- val_loss: 0.159091
- dead_features: 250/512
- elapsed: 0.4 min

## [20:40] [SAEv2] Epoch 10/80
- train_loss: 0.114562
- val_loss: 0.310351
- dead_features: 272/512
- elapsed: 0.3 min

## [20:40] [SAEv2] Epoch 5/80
- train_loss: 0.057502
- val_loss: 0.054764
- dead_features: 275/512
- elapsed: 0.2 min

## [20:40] [SAEv2] Epoch 1/80
- train_loss: 0.446644
- val_loss: 0.103167
- dead_features: 272/512
- elapsed: 0.0 min

## [20:40] [SAEv2] Retraining SAE with resampling
- hidden_factor=2 → hidden_dim=512
- K=32, resample_every=50 batches
- train=90,000, val=10,000

## [20:39] [SAEv2] Retraining complete
- best val_loss: 0.000168
- dead_features: 250/512
- elapsed: 2.0 min

## [20:39] [SAEv2] Epoch 80/80
- train_loss: 0.004711
- val_loss: 0.000719
- dead_features: 243/512
- elapsed: 2.0 min

## [20:38] [SAEv2] Epoch 75/80
- train_loss: 0.002897
- val_loss: 0.000184
- dead_features: 244/512
- elapsed: 1.9 min

## [20:38] [SAEv2] Epoch 70/80
- train_loss: 0.002854
- val_loss: 0.000192
- dead_features: 243/512
- elapsed: 1.8 min

## [20:38] [SAEv2] Epoch 65/80
- train_loss: 0.003734
- val_loss: 0.000256
- dead_features: 244/512
- elapsed: 1.7 min

## [20:38] [SAEv2] Epoch 60/80
- train_loss: 0.004516
- val_loss: 0.000393
- dead_features: 245/512
- elapsed: 1.5 min

## [20:38] [SAEv2] Epoch 55/80
- train_loss: 0.006430
- val_loss: 0.000402
- dead_features: 245/512
- elapsed: 1.4 min

## [20:38] [SAEv2] Epoch 50/80
- train_loss: 0.011078
- val_loss: 0.000868
- dead_features: 242/512
- elapsed: 1.3 min

## [20:38] [SAEv2] Epoch 45/80
- train_loss: 0.009898
- val_loss: 0.000911
- dead_features: 244/512
- elapsed: 1.2 min

## [20:38] [SAEv2] Epoch 40/80
- train_loss: 0.013136
- val_loss: 0.001101
- dead_features: 243/512
- elapsed: 1.0 min

## [20:38] [SAEv2] Epoch 35/80
- train_loss: 0.019577
- val_loss: 0.013563
- dead_features: 245/512
- elapsed: 0.9 min

## [20:37] [SAEv2] Epoch 30/80
- train_loss: 0.045129
- val_loss: 0.002743
- dead_features: 246/512
- elapsed: 0.8 min

## [20:37] [SAEv2] Epoch 25/80
- train_loss: 0.091080
- val_loss: 0.032970
- dead_features: 246/512
- elapsed: 0.6 min

## [20:37] [SAEv2] Epoch 20/80
- train_loss: 0.075070
- val_loss: 0.012627
- dead_features: 262/512
- elapsed: 0.5 min

## [20:37] [SAEv2] Epoch 15/80
- train_loss: 0.926732
- val_loss: 0.093094
- dead_features: 258/512
- elapsed: 0.4 min

## [20:37] [SAEv2] Epoch 10/80
- train_loss: 0.032091
- val_loss: 0.007806
- dead_features: 288/512
- elapsed: 0.3 min

## [20:37] [SAEv2] Epoch 5/80
- train_loss: 0.054648
- val_loss: 0.015912
- dead_features: 281/512
- elapsed: 0.1 min

## [20:37] [SAEv2] Epoch 1/80
- train_loss: 0.261033
- val_loss: 0.056086
- dead_features: 293/512
- elapsed: 0.0 min

## [20:37] [SAEv2] Retraining SAE with resampling
- hidden_factor=2 → hidden_dim=512
- K=32, resample_every=100 batches
- train=90,000, val=10,000

## [20:34] [EXP3] Starting: lam1.0_seed42
- λ=1.0, seed=42

## [20:34] [EXP3] lam1.0_seed0 seed=0 complete
- test_fail_rate: 1.000
- test_mean_reward: 0.0000
- train_mean_reward: 0.0000 (forgetting: True)
- elapsed: 29.9 min

## [20:04] [EXP3] Starting: lam1.0_seed0
- λ=1.0, seed=0

## [20:04] [EXP3] lam0.5_seed123 seed=123 complete
- test_fail_rate: 1.000
- test_mean_reward: 0.0000
- train_mean_reward: 0.0000 (forgetting: True)
- elapsed: 29.0 min

## [19:35] [EXP3] Starting: lam0.5_seed123
- λ=0.5, seed=123

## [19:35] [EXP3] lam0.5_seed42 seed=42 complete
- test_fail_rate: 1.000
- test_mean_reward: 0.0000
- train_mean_reward: 0.0000 (forgetting: True)
- elapsed: 29.3 min

## [19:05] [EXP3] Starting: lam0.5_seed42
- λ=0.5, seed=42

## [19:05] [EXP3] lam0.5_seed0 seed=0 complete
- test_fail_rate: 1.000
- test_mean_reward: 0.0000
- train_mean_reward: 0.0000 (forgetting: True)
- elapsed: 29.5 min

## [18:35] [EXP3] Starting: lam0.5_seed0
- λ=0.5, seed=0

## [18:35] [EXP3] lam0.1_seed123 seed=123 complete
- test_fail_rate: 1.000
- test_mean_reward: 0.0000
- train_mean_reward: 0.0000 (forgetting: True)
- elapsed: 29.1 min

## [18:06] [EXP3] Starting: lam0.1_seed0
- λ=0.1, seed=0

## [18:06] [EXP3] baseline_seed123 seed=123 complete
- test_fail_rate: 0.000
- test_mean_reward: 1.0000
- train_mean_reward: 1.0000 (forgetting: False)
- elapsed: 19.2 min

## [18:05] [EXP3] Starting: lam0.1_seed123
- λ=0.1, seed=123

## [18:05] [EXP3] lam0.1_seed42 seed=42 complete
- test_fail_rate: 1.000
- test_mean_reward: 0.0000
- train_mean_reward: 0.0000 (forgetting: True)
- elapsed: 24.6 min

## [17:47] [EXP3] Starting: baseline_seed123
- λ=0.0, seed=123

## [17:47] [EXP3] baseline_seed42 seed=42 complete
- test_fail_rate: 0.000
- test_mean_reward: 1.0000
- train_mean_reward: 1.0000 (forgetting: False)
- elapsed: 18.0 min

## [17:40] [EXP3] Starting: lam0.1_seed42
- λ=0.1, seed=42

## [17:40] [EXP3] lam0.1_seed0 seed=0 complete
- test_fail_rate: 1.000
- test_mean_reward: 0.0000
- train_mean_reward: 0.0000 (forgetting: True)
- elapsed: 22.2 min

## [17:29] [EXP3] Starting: baseline_seed42
- λ=0.0, seed=42

## [17:29] [EXP3] baseline_seed0 seed=0 complete
- test_fail_rate: 0.000
- test_mean_reward: 1.0000
- train_mean_reward: 1.0000 (forgetting: False)
- elapsed: 19.2 min

## [17:18] [EXP3] Starting: lam0.1_seed0
- λ=0.1, seed=0

## [17:18] [EXP3] Starting: baseline_seed123
- λ=0.0, seed=123

## [17:18] [EXP3] Starting: baseline_seed42
- λ=0.0, seed=42

## [17:18] [EXP3] Starting: baseline_seed0
- λ=0.0, seed=0

## [17:18] [EXP3] Option B START
- λ values: [0.0, 0.1, 0.5, 1.0]
- seeds: [0, 42, 123]
- timesteps per run: 100,000
- baseline_goal_sig: 0.3945
- baseline_proxy_sig: 0.4353

## [17:18] [EXP3] START — Option B lambda sweep


## [17:17] [EXP3] Starting: lam0.1_seed0
- λ=0.1, seed=0

## [17:17] [EXP3] Starting: baseline_seed123
- λ=0.0, seed=123

## [17:17] [EXP3] Starting: baseline_seed42
- λ=0.0, seed=42

## [17:17] [EXP3] Starting: baseline_seed0
- λ=0.0, seed=0

## [17:17] [EXP3] Option B START
- λ values: [0.0, 0.1, 0.5, 1.0]
- seeds: [0, 42, 123]
- timesteps per run: 100,000
- baseline_goal_sig: 0.3945
- baseline_proxy_sig: 0.4353

## [17:17] [EXP3] START — Option B lambda sweep


## [17:16] [EXP3] Starting: lam0.1_seed0
- λ=0.1, seed=0

## [17:16] [EXP3] Starting: baseline_seed123
- λ=0.0, seed=123

## [17:16] [EXP3] Starting: baseline_seed42
- λ=0.0, seed=42

## [17:16] [EXP3] Starting: baseline_seed0
- λ=0.0, seed=0

## [17:16] [EXP3] Option B START
- λ values: [0.0, 0.1, 0.5, 1.0]
- seeds: [0, 42, 123]
- timesteps per run: 100,000
- baseline_goal_sig: 0.3945
- baseline_proxy_sig: 0.4353

## [17:16] [EXP3] START — Option B lambda sweep


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


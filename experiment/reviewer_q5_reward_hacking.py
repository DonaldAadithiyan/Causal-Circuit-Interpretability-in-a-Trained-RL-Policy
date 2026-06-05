"""
Reviewer Q5 — Does the method generalise to a second failure mode (reward hacking)?

Pipeline (reduced scale, logged):
  A. Train PPO on CoinHack with shortcut_reward=0.3 (agent should prefer the real goal).
  B. Collect activations, train SAEv2 (384), compute W.
  C. Identify the SHORTCUT feature: the SAE feature whose activation best correlates with
     the agent approaching the fixed shortcut cell (and the REAL-goal feature likewise).
  D. Induce reward hacking: a second training phase with shortcut_reward=0.9. Every 10k
     steps record (a) behavioral shortcut-take-rate and (b) the shortcut feature's causal
     weight to the action (KL when zeroed = I2 / spurious-edge signal). Test whether the
     graph signal rises BEFORE the behavioral switch.

Self-contained; writes to outputs/experiment4/reviewer/q5_*.
"""

import sys, os, json, time, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

from models.impala_cnn import ImpalaCNNExtractor
from models.topk_sae_v2 import TopKSAEv2
from envs.coin_hack_env import make_hack_env, make_hack_env_with_info
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "outputs/experiment4/reviewer")
Q5 = os.path.join(BASE, "outputs/experiment4/reviewer/q5")
TRAIN_STEPS = 300_000
HACK_STEPS = 80_000
HACK_CHUNK = 10_000
HACK_SHORTCUT = 1.5   # > real-goal reward (1.0) so a reward-maximiser is incentivised to HACK


def behavior_rates(model, shortcut_reward, n=40, seed=0):
    """Fraction of episodes that reach shortcut vs real goal vs neither."""
    env = make_hack_env_with_info(shortcut_reward=shortcut_reward)
    sc, real, none = 0, 0, 0
    for ep in range(n):
        obs, info = env.reset(seed=seed + ep)
        done = False; reached = None
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            if info.get("reached"): reached = info["reached"]
        if reached == "shortcut": sc += 1
        elif reached == "real": real += 1
        else: none += 1
    env.close()
    return sc / n, real / n, none / n


def train_base():
    log_entry("[EXP4-Q5] A — train CoinHack policy (shortcut=0.3)", f"- {TRAIN_STEPS:,} steps")
    venv = make_vec_env(lambda: make_hack_env(shortcut_reward=0.3), n_envs=4, seed=0)
    pk = dict(features_extractor_class=ImpalaCNNExtractor,
              features_extractor_kwargs=dict(features_dim=256), net_arch=[])
    model = PPO("CnnPolicy", venv, policy_kwargs=pk, n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
                max_grad_norm=0.5, learning_rate=2.5e-4, verbose=0, device=str(device))
    t0 = time.time()
    model.learn(total_timesteps=TRAIN_STEPS)
    model.save(os.path.join(Q5, "hack_policy"))
    sc, real, none = behavior_rates(model, 0.3, n=40)
    log_entry("[EXP4-Q5] A done",
              f"- shortcut_rate={sc:.2f} real_rate={real:.2f} none={none:.2f} ({(time.time()-t0)/60:.0f} min)")
    venv.close(); del venv
    return model, {"shortcut_rate": sc, "real_rate": real, "none_rate": none}


def collect_and_train_sae(model, n_steps=60_000):
    log_entry("[EXP4-Q5] B — collect activations + train SAE", f"- {n_steps:,}")
    cap = {}
    def hook(_m, _i, o): cap["f"] = o.detach().cpu()
    h = model.policy.features_extractor.register_forward_hook(hook)
    acts = np.zeros((n_steps, 256), dtype=np.float32)
    apos = np.zeros((n_steps, 2), dtype=np.int32)
    rg = np.zeros((n_steps, 2), dtype=np.int32)
    env = make_hack_env_with_info(shortcut_reward=0.3)
    obs, info = env.reset(seed=0); idx = 0
    while idx < n_steps:
        a, _ = model.predict(obs, deterministic=False)
        acts[idx] = cap["f"].squeeze(0).numpy()
        apos[idx] = info["agent_pos"]; rg[idx] = info["real_goal"]
        obs, r, term, trunc, info = env.step(a); idx += 1
        if term or trunc: obs, info = env.reset()
    h.remove(); env.close()

    mean = acts.mean(0); std = acts.std(0) + 1e-8
    actn = torch.from_numpy(((acts - mean) / std).astype(np.float32))
    sae = TopKSAEv2(input_dim=256, hidden_factor=1.5, k=32, resample_threshold=150).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-4)
    n = len(actn); bs = 256
    for epoch in range(40):
        perm = torch.randperm(n); bn = 0
        for s in range(0, n, bs):
            b = actn[perm[s:s+bs]].to(device)
            opt.zero_grad(); xh, _ = sae(b); loss = sae.loss(b, xh)
            loss.backward(); opt.step(); sae.normalize_decoder()
            if bn % 50 == 0 and epoch <= 30: sae.resample_dead_features(b, opt)
            bn += 1
    sae.eval()
    with torch.no_grad():
        dead = int(((sae(actn[:2000].to(device))[1] > 0).float().mean(0).cpu().numpy() < 1e-3).sum())
    log_entry("[EXP4-Q5] B done", f"- SAE dead {dead}/{sae.hidden_dim}")
    return sae, mean, std, acts, apos, rg


def identify_features(sae, mean, std, acts, apos, rg):
    """Shortcut feature: activation correlates with proximity to (2,2).
       Real-goal feature: activation correlates with proximity to the real goal."""
    actn = torch.from_numpy(((acts - mean) / std).astype(np.float32))
    with torch.no_grad():
        H = []
        for s in range(0, len(actn), 1024):
            H.append(sae.get_feature_activations(actn[s:s+1024].to(device)).cpu().numpy())
        H = np.concatenate(H, 0)
    sc_prox = -np.hypot(apos[:, 0] - 2, apos[:, 1] - 2)         # near shortcut (2,2)
    rg_prox = -np.hypot(apos[:, 0] - rg[:, 0], apos[:, 1] - rg[:, 1])  # near real goal
    nfeat = H.shape[1]
    sc_corr = np.zeros(nfeat); rg_corr = np.zeros(nfeat)
    freq = (H > 0).mean(0)
    for i in range(nfeat):
        if H[:, i].std() < 1e-8: continue
        sc_corr[i] = pearsonr(H[:, i], sc_prox)[0]
        rg_corr[i] = pearsonr(H[:, i], rg_prox)[0]
    sc_corr = np.nan_to_num(sc_corr); rg_corr = np.nan_to_num(rg_corr)
    top = np.argsort(freq)[::-1][:50]
    shortcut_feat = int(top[np.argmax(np.abs(sc_corr[top]))])
    real_feat = int(top[np.argmax(np.abs(rg_corr[top]))])
    log_entry("[EXP4-Q5] C — features identified",
              f"- shortcut_feat {shortcut_feat} (corr {sc_corr[shortcut_feat]:.3f})\n"
              f"- real_feat {real_feat} (corr {rg_corr[real_feat]:.3f})")
    return shortcut_feat, real_feat, float(sc_corr[shortcut_feat]), float(rg_corr[real_feat])


def causal_weight(model, sae, mean, std, feat, n_obs=150):
    """KL(action || action with `feat` zeroed) — the feature's causal weight to the action."""
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)
    cap = {}
    def hook(_m, _i, o): cap["f"] = o.detach()
    h = model.policy.features_extractor.register_forward_hook(hook)
    env = make_hack_env_with_info(shortcut_reward=HACK_SHORTCUT)
    obs, info = env.reset(seed=123); feats = []
    for _ in range(n_obs):
        a, _ = model.predict(obs, deterministic=True)
        feats.append(cap["f"].squeeze(0).cpu().numpy())
        obs, r, term, trunc, info = env.step(a)
        if term or trunc: obs, info = env.reset()
    h.remove(); env.close()
    F_raw = torch.from_numpy(((np.array(feats) - mean) / std).astype(np.float32)).to(device)
    with torch.no_grad():
        _, hb = sae(F_raw)
        pb = F.softmax(model.policy.action_net(sae.decode(hb) * std_t + mean_t), -1)
        hp = hb.clone(); hp[:, feat] = 0.0
        pp = F.softmax(model.policy.action_net(sae.decode(hp) * std_t + mean_t), -1)
        p = pb + 1e-8; q = pp + 1e-8
        return float((p * torch.log(p / q)).sum(-1).mean().item())


def induce_hacking(model, sae, mean, std, shortcut_feat, real_feat):
    """Second training phase shortcut=0.9; track behavior + shortcut causal weight per chunk."""
    log_entry("[EXP4-Q5] D — induce reward hacking (shortcut=0.9)", "")
    venv = make_vec_env(lambda: make_hack_env(shortcut_reward=HACK_SHORTCUT), n_envs=4, seed=1)
    model.set_env(venv)
    curve = []
    # step 0
    sc0, real0, _ = behavior_rates(model, HACK_SHORTCUT, n=30)
    cw_sc = causal_weight(model, sae, mean, std, shortcut_feat)
    cw_real = causal_weight(model, sae, mean, std, real_feat)
    curve.append({"steps": 0, "shortcut_rate": sc0, "real_rate": real0,
                  "shortcut_causal": cw_sc, "real_causal": cw_real})
    log_entry("[EXP4-Q5] D step 0",
              f"- shortcut_rate={sc0:.2f} shortcut_causal={cw_sc:.5f} real_causal={cw_real:.5f}")
    done_steps = 0
    while done_steps < HACK_STEPS:
        model.learn(total_timesteps=HACK_CHUNK, reset_num_timesteps=False)
        done_steps += HACK_CHUNK
        sc, real, _ = behavior_rates(model, HACK_SHORTCUT, n=30)
        cw_sc = causal_weight(model, sae, mean, std, shortcut_feat)
        cw_real = causal_weight(model, sae, mean, std, real_feat)
        curve.append({"steps": done_steps, "shortcut_rate": sc, "real_rate": real,
                      "shortcut_causal": cw_sc, "real_causal": cw_real})
        log_entry(f"[EXP4-Q5] D {done_steps} steps",
                  f"- shortcut_rate={sc:.2f} shortcut_causal={cw_sc:.5f}")
    venv.close()
    return curve


def main():
    os.makedirs(Q5, exist_ok=True)
    log_entry("[EXP4-Q5] START — reward-hacking failure mode", "")
    t0 = time.time()

    # Resume: reuse the trained base policy if present (skip the ~31-min retrain)
    base_ckpt = os.path.join(Q5, "hack_policy.zip")
    if os.path.exists(base_ckpt):
        log_entry("[EXP4-Q5] A — reusing trained base policy", f"- {base_ckpt}")
        model = PPO.load(base_ckpt, device=str(device))
        model.policy.eval()
        sc, real, none = behavior_rates(model, 0.3, n=40)
        base_beh = {"shortcut_rate": sc, "real_rate": real, "none_rate": none}
    else:
        model, base_beh = train_base()
    sae, mean, std, acts, apos, rg = collect_and_train_sae(model)
    sc_feat, real_feat, sc_corr, rg_corr = identify_features(sae, mean, std, acts, apos, rg)
    curve = induce_hacking(model, sae, mean, std, sc_feat, real_feat)

    # Behavioral switch: first chunk where shortcut_rate > 0.5
    beh_switch = next((c["steps"] for c in curve if c["shortcut_rate"] > 0.5), None)
    # Signal rise: first chunk where shortcut causal weight exceeds 1.5x its step-0 value
    cw0 = curve[0]["shortcut_causal"]
    sig_rise = next((c["steps"] for c in curve
                     if cw0 > 1e-9 and c["shortcut_causal"] > 1.5 * cw0), None)
    k_hack = (beh_switch - sig_rise) if (beh_switch is not None and sig_rise is not None) else None

    summary = {
        "hack_shortcut_reward": HACK_SHORTCUT,
        "real_goal_reward": 1.0,
        "base_behavior_shortcut03": base_beh,
        "shortcut_feature": sc_feat, "shortcut_feature_corr": sc_corr,
        "real_feature": real_feat, "real_feature_corr": rg_corr,
        "curve": curve,
        "behavioral_switch_step": beh_switch,
        "signal_rise_step": sig_rise,
        "k_hack": k_hack,
        "elapsed_min": (time.time() - t0) / 60,
    }
    json.dump(summary, open(os.path.join(OUT, "q5_reward_hacking.json"), "w"), indent=2)

    xs = [c["steps"] for c in curve]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(xs, [c["shortcut_rate"] for c in curve], "o-", color="coral", label="shortcut-take rate (behavior)")
    ax1.set_xlabel("Reward-hacking induction steps (shortcut=0.9)")
    ax1.set_ylabel("Shortcut-take rate", color="coral")
    ax2 = ax1.twinx()
    ax2.plot(xs, [c["shortcut_causal"] for c in curve], "s--", color="purple", label="shortcut causal weight (graph)")
    ax2.set_ylabel("Shortcut feature causal weight (KL)", color="purple")
    if sig_rise is not None: ax1.axvline(sig_rise, color="purple", ls=":", alpha=0.6)
    if beh_switch is not None: ax1.axvline(beh_switch, color="coral", ls=":", alpha=0.6)
    plt.title(f"Q5 — reward hacking: graph signal vs behavior"
              + (f"  (k_hack={k_hack} steps)" if k_hack is not None else ""))
    fig.tight_layout(); plt.savefig(os.path.join(OUT, "q5_reward_hacking.png"), dpi=150)
    plt.close()

    log_entry("[EXP4-Q5] COMPLETE",
              f"- base (shortcut=0.3) shortcut_rate {base_beh['shortcut_rate']:.2f}\n"
              f"- shortcut feature {sc_feat} (corr {sc_corr:.3f})\n"
              f"- behavioral switch at {beh_switch}, signal rise at {sig_rise}, k_hack={k_hack}")
    print(f"\nQ5 — REWARD HACKING")
    print(f"  base shortcut-rate (shortcut=0.3): {base_beh['shortcut_rate']:.2f} (low = prefers real goal)")
    print(f"  shortcut feature: {sc_feat} (corr {sc_corr:.3f})")
    print(f"  behavioral switch step: {beh_switch}")
    print(f"  graph signal rise step: {sig_rise}")
    print(f"  k_hack (behavior - signal): {k_hack}")


if __name__ == "__main__":
    main()

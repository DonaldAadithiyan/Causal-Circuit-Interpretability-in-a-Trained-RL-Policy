"""
Q5-RESCORE — W-matrix I2 signal for reward hacking (re-instrumented reproduction).

IMPORTANT (data-availability disclosure): Q5 did NOT persist per-step episode data, the
hack-policy SAE, or a W-matrix — and Q5 was an online experiment (fresh rollouts per
checkpoint), so there were never frozen episodes on disk to rescore. This script therefore
REPRODUCES the Q5 induction (reusing the saved base policy) with full W-based instrumentation,
rather than rescoring identical saved episodes. It is labelled as a reproduction everywhere.

W-based I2 signal (per step):  I2(t) = Σ_j |W[s, j]| · h_j(t)
  where s = shortcut feature, h = current SAE activations, W = D^T · W_enc^T.
This is the shortcut feature's W-row weighted by the current activation vector — i.e. the
gradient-free "live causal weight" of the shortcut feature (the same metric that powered
Q1–Q3). It rises when the shortcut feature gains causal control of the circuit.

k convention (matches "k>0 = early warning", i.e. signal precedes the behavioural switch):
  k = (behavioural-switch step) − (I2 first crosses noise_floor_mean + 2σ).
"""

import sys, os, json, time, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from models.topk_sae_v2 import TopKSAEv2
from envs.coin_hack_env import make_hack_env, make_hack_env_with_info
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
Q5DIR = os.path.join(BASE, "outputs/experiment4/reviewer/q5")
OUT = os.path.join(BASE, "outputs/q5_rescore")
HACK_SHORTCUT = 1.5
INDUCT_STEPS = 60_000
CHUNK = 10_000
SEEDS = [1, 2]


def build_sae_and_W(model, n_steps=40_000):
    """Collect base-policy activations, train SAEv2, compute W. Also return agent/goal positions."""
    cap = {}
    def hook(_m, _i, o): cap["f"] = o.detach().cpu()
    h = model.policy.features_extractor.register_forward_hook(hook)
    acts = np.zeros((n_steps, 256), dtype=np.float32)
    apos = np.zeros((n_steps, 2), dtype=np.int32)
    env = make_hack_env_with_info(shortcut_reward=0.3)
    obs, info = env.reset(seed=0); idx = 0
    while idx < n_steps:
        a, _ = model.predict(obs, deterministic=False)
        acts[idx] = cap["f"].squeeze(0).numpy(); apos[idx] = info["agent_pos"]
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
        D = sae.decoder.weight.detach(); Wenc = sae.encoder.weight.detach()
        W = (D.T @ Wenc.T).cpu().numpy()
        dead = int(((sae(actn[:2000].to(device))[1] > 0).float().mean(0).cpu().numpy() < 1e-3).sum())
    # Identify the shortcut feature (max |corr| of activation with proximity to shortcut (2,2))
    with torch.no_grad():
        H = []
        for s in range(0, n, 1024):
            H.append(sae.get_feature_activations(actn[s:s+1024].to(device)).cpu().numpy())
        H = np.concatenate(H, 0)
    sc_prox = -np.hypot(apos[:, 0] - 2, apos[:, 1] - 2)
    freq = (H > 0).mean(0); nfeat = H.shape[1]
    sc_corr = np.zeros(nfeat)
    for i in range(nfeat):
        if H[:, i].std() > 1e-8:
            sc_corr[i] = pearsonr(H[:, i], sc_prox)[0]
    sc_corr = np.nan_to_num(sc_corr)
    topf = np.argsort(freq)[::-1][:50]
    shortcut_feat = int(topf[np.argmax(np.abs(sc_corr[topf]))])

    torch.save({"state_dict": sae.state_dict(), "input_dim": 256, "k": 32,
                "hidden_factor": 1.5, "act_mean": mean.tolist(), "act_std": std.tolist(),
                "dead_features": dead}, os.path.join(OUT, "hack_sae.pt"))
    np.save(os.path.join(OUT, "hack_W.npy"), W)
    log_entry("[Q5-RESCORE] SAE + W built (reproduction)",
              f"- SAE dead {dead}/{nfeat}\n- shortcut_feat {shortcut_feat} (corr {sc_corr[shortcut_feat]:.3f})\n"
              f"- W shape {W.shape}")
    return sae, mean, std, W, shortcut_feat, float(sc_corr[shortcut_feat])


def i2_per_step(model, sae, mean, std, W, shortcut_feat, shortcut_reward, n_ep=20, seed=0):
    """Deploy n_ep episodes; return list of per-step I2 values and the shortcut-take rate."""
    mean_a = mean; std_a = std
    cap = {}
    def hook(_m, _i, o): cap["f"] = o.detach().cpu()
    h = model.policy.features_extractor.register_forward_hook(hook)
    Wrow = np.abs(W[shortcut_feat])               # (hidden,)
    env = make_hack_env_with_info(shortcut_reward=shortcut_reward)
    all_i2 = []; sc = 0; tot = 0
    for ep in range(n_ep):
        obs, info = env.reset(seed=seed + ep); done = False; reached = None
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            feat = cap["f"].squeeze(0).numpy()
            fn = ((feat - mean_a) / std_a).astype(np.float32)
            with torch.no_grad():
                hvec = sae.get_feature_activations(torch.from_numpy(fn).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            i2 = float(np.sum(Wrow * hvec))        # Σ_j |W[s,j]| · h_j  = W-based live causal weight
            all_i2.append(i2)
            obs, r, term, trunc, info = env.step(a); done = term or trunc
            if info.get("reached"): reached = info["reached"]
        tot += 1
        if reached == "shortcut": sc += 1
    h.remove(); env.close()
    return all_i2, sc / max(tot, 1)


def run_seed(base_ckpt, sae, mean, std, W, shortcut_feat, threshold, seed):
    """Reproduce induction for one seed; per-chunk record shortcut_rate + mean I2."""
    venv = make_vec_env(lambda: make_hack_env(shortcut_reward=HACK_SHORTCUT), n_envs=4, seed=seed)
    model = PPO.load(base_ckpt, env=venv, device=str(device))
    curve = []
    steps = 0
    while True:
        i2_list, sc_rate = i2_per_step(model, sae, mean, std, W, shortcut_feat,
                                       HACK_SHORTCUT, n_ep=20, seed=seed * 10)
        curve.append({"steps": steps, "shortcut_rate": sc_rate,
                      "mean_i2": float(np.mean(i2_list)), "max_i2": float(np.max(i2_list))})
        log_entry(f"[Q5-RESCORE] seed {seed} @ {steps} steps",
                  f"- shortcut_rate {sc_rate:.2f} mean_I2 {np.mean(i2_list):.5f} (thr {threshold:.5f})")
        if steps >= INDUCT_STEPS:
            break
        model.learn(total_timesteps=CHUNK, reset_num_timesteps=False)
        steps += CHUNK
    venv.close(); del model; gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

    beh_switch = next((c["steps"] for c in curve if c["shortcut_rate"] > 0.5), None)
    i2_cross = next((c["steps"] for c in curve if c["mean_i2"] > threshold), None)
    k = (beh_switch - i2_cross) if (beh_switch is not None and i2_cross is not None) else None
    return curve, beh_switch, i2_cross, k


def main():
    os.makedirs(OUT, exist_ok=True)
    log_entry("[Q5-RESCORE] START — W-based I2 for reward hacking (reproduction)",
              "- Q5 saved no per-step data/SAE/W; reproducing induction with W-based instrumentation")
    t0 = time.time()
    base_ckpt = os.path.join(Q5DIR, "hack_policy.zip")
    base = PPO.load(base_ckpt, device=str(device)); base.policy.eval()

    sae, mean, std, W, shortcut_feat, sc_corr = build_sae_and_W(base)

    # Noise floor: I2 on the TRAINING distribution (base policy, shortcut=0.3)
    nf_i2, _ = i2_per_step(base, sae, mean, std, W, shortcut_feat, 0.3, n_ep=30, seed=999)
    nf_mean = float(np.mean(nf_i2)); nf_std = float(np.std(nf_i2)); nf_max = float(np.max(nf_i2))
    threshold = nf_mean + 2 * nf_std
    log_entry("[Q5-RESCORE] noise floor (training dist)",
              f"- mean {nf_mean:.5f} std {nf_std:.5f} MAX {nf_max:.5f} -> 2σ threshold {threshold:.5f}")
    del base; gc.collect()

    # Reproduce induction per seed
    seed_results = {}; ks = []; rep_curve = None
    for seed in SEEDS:
        curve, beh, cross, k = run_seed(base_ckpt, sae, mean, std, W, shortcut_feat, threshold, seed)
        seed_results[f"seed_{seed}"] = {"behavioral_switch": beh, "i2_cross": cross, "k": k, "curve": curve}
        if k is not None: ks.append(k)
        if rep_curve is None: rep_curve = curve
        log_entry(f"[Q5-RESCORE] seed {seed} done",
                  f"- behavioral_switch {beh}, i2_cross {cross}, k {k}")

    mean_k = float(np.mean(ks)) if ks else None
    std_k = float(np.std(ks)) if len(ks) > 1 else (0.0 if ks else None)
    peak_violation = max(c["mean_i2"] for s in seed_results.values() for c in s["curve"])
    noise_comparable = nf_max >= 0.5 * peak_violation

    summary = {
        "method": "RE-INSTRUMENTED REPRODUCTION (Q5 saved no per-step data; not a rescore of identical episodes)",
        "i2_definition": "I2(t) = sum_j |W[shortcut_feat, j]| * h_j(t)  (W = D^T W_enc^T; gradient-free live causal weight)",
        "k_convention": "k = behavioral_switch_step - i2_cross_step  (k>0 = early warning, signal precedes switch)",
        "shortcut_feature": shortcut_feat, "shortcut_feature_corr": sc_corr,
        "noise_floor_mean": nf_mean, "noise_floor_std": nf_std, "noise_floor_max": nf_max,
        "threshold_2sigma": threshold,
        "peak_violation_mean_i2": peak_violation,
        "noise_floor_comparable_to_signal": noise_comparable,
        "mean_k": mean_k, "std_k": std_k, "n_seeds": len(SEEDS),
        "per_seed": seed_results,
        "elapsed_min": (time.time() - t0) / 60,
    }
    json.dump(summary, open(os.path.join(OUT, "q5_rescore_summary.json"), "w"), indent=2)

    # Representative plot
    xs = [c["steps"] for c in rep_curve]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(xs, [c["shortcut_rate"] for c in rep_curve], "o-", color="coral",
             label="shortcut-take rate (behavior)")
    ax1.set_xlabel("Induction steps (shortcut=1.5)"); ax1.set_ylabel("shortcut-take rate", color="coral")
    ax2 = ax1.twinx()
    ax2.plot(xs, [c["mean_i2"] for c in rep_curve], "s--", color="purple", label="W-based I2 signal")
    ax2.axhline(threshold, color="purple", ls=":", alpha=0.6, label="noise floor + 2σ")
    ax2.axhline(nf_max, color="gray", ls=":", alpha=0.5, label="training-dist max I2 (noise floor)")
    ax2.set_ylabel("W-based I2 (live causal weight of shortcut feat)", color="purple")
    rk = seed_results[f"seed_{SEEDS[0]}"]
    if rk["i2_cross"] is not None: ax1.axvline(rk["i2_cross"], color="purple", ls="-", alpha=0.3)
    if rk["behavioral_switch"] is not None: ax1.axvline(rk["behavioral_switch"], color="coral", ls="-", alpha=0.3)
    plt.title(f"Q5-RESCORE — W-based I2 vs behavior (mean k={mean_k}±{std_k}, n={len(SEEDS)} seeds)")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95), fontsize=8)
    fig.tight_layout(); plt.savefig(os.path.join(OUT, "q5_rescore_representative.png"), dpi=150)
    plt.close()

    log_entry("[Q5-RESCORE] COMPLETE",
              f"- mean_k {mean_k} ± {std_k} (n={len(SEEDS)} seeds)\n"
              f"- noise floor max {nf_max:.5f}, peak I2 {peak_violation:.5f}, comparable={noise_comparable}\n"
              f"- threshold(2σ) {threshold:.5f}")
    print(f"\n{'='*64}")
    print("Q5-RESCORE — W-based I2 (reproduction)")
    print(f"  shortcut feature: {shortcut_feat} (corr {sc_corr:.3f})")
    print(f"  noise floor: mean {nf_mean:.5f}, max {nf_max:.5f}, 2σ threshold {threshold:.5f}")
    print(f"  peak violation I2: {peak_violation:.5f}  (noise comparable: {noise_comparable})")
    for s, r in seed_results.items():
        print(f"  {s}: behavioral_switch={r['behavioral_switch']} i2_cross={r['i2_cross']} k={r['k']}")
    print(f"  mean k = {mean_k} ± {std_k}  (n={len(SEEDS)} seeds)")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()

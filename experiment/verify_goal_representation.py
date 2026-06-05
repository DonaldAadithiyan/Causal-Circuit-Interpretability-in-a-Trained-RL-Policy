"""
Experiment 4, Phase 2 — Train SAEv3 on random-goal policy + verify H1 prerequisite.

Steps:
  1. Collect 100k activations from frozen random-goal policy (training dist).
  2. Train SAEv3 (384 hidden, K=32, resampling) — reuse TopKSAEv2.
  3. H1 GATE: for top-50 features, compute actual_goal_corr (|activation vs dist-to-goal|)
     across 100 test episodes with varied goals. Need max > 0.3 or STOP.
  4. Build W, validate vs patching (r>0.5), build G* + invariant profiles.
"""

import sys, os, json, gc, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from stable_baselines3 import PPO
from models.topk_sae_v2 import TopKSAEv2
from envs.coin_env import make_env_with_info
from utils.logging_utils import log_entry

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

BASE = os.path.dirname(__file__)
E4_DIR = os.path.join(BASE, "outputs/experiment4")
POLICY_DIR = os.path.join(E4_DIR, "policy_randomgoal")
SAE_DIR = os.path.join(E4_DIR, "sae_v3")
GRAPH_DIR = os.path.join(E4_DIR, "graphs")
PLOT_DIR = os.path.join(E4_DIR, "plots")
ACT_DIR = os.path.join(E4_DIR, "activations")

# SAEv3 hypers (same as SAEv2)
K = 32
HIDDEN_FACTOR = 1.5
LR = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 8
RESAMPLE_EVERY = 50
RESAMPLE_THRESHOLD = 150
H1_THRESHOLD = 0.3


def collect_activations(model, n_steps=100_000):
    os.makedirs(ACT_DIR, exist_ok=True)
    features_dim = 256
    act_mm = np.memmap(os.path.join(ACT_DIR, "activations.npy"), dtype=np.float32,
                       mode="w+", shape=(n_steps, features_dim))
    gpos = np.zeros((n_steps, 2), dtype=np.int32)
    apos = np.zeros((n_steps, 2), dtype=np.int32)

    captured = {}
    def hook(_m, _i, out): captured["f"] = out.detach().cpu()
    h = model.policy.features_extractor.register_forward_hook(hook)

    env = make_env_with_info(random_goal=True)
    obs, info = env.reset(seed=0)
    idx = 0
    t0 = time.time()
    log_entry("[EXP4] Phase 2 — collecting activations", f"- target {n_steps:,}")
    while idx < n_steps:
        action, _ = model.predict(obs, deterministic=False)
        act_mm[idx] = captured["f"].squeeze(0).numpy()
        gp = info.get("goal_pos") or (0, 0)
        ap = info.get("agent_pos") or (0, 0)
        gpos[idx] = [int(gp[0]), int(gp[1])]
        apos[idx] = [int(ap[0]), int(ap[1])]
        obs, r, term, trunc, info = env.step(action)
        idx += 1
        if term or trunc:
            obs, info = env.reset()
        if idx % 20000 == 0:
            log_entry(f"[EXP4] Phase 2 — {idx:,}/{n_steps:,} collected",
                      f"- {(time.time()-t0)/60:.1f} min")
    h.remove(); env.close()
    act_mm.flush()
    np.save(os.path.join(ACT_DIR, "goal_pos.npy"), gpos[:idx])
    np.save(os.path.join(ACT_DIR, "agent_pos.npy"), apos[:idx])
    with open(os.path.join(ACT_DIR, "meta.json"), "w") as f:
        json.dump({"n_samples": int(idx), "features_dim": features_dim}, f)
    log_entry("[EXP4] Phase 2 — collection done", f"- {idx:,} samples")
    return idx


def train_sae(n, dim):
    acts = np.memmap(os.path.join(ACT_DIR, "activations.npy"), dtype=np.float32,
                     mode="r", shape=(n, dim))
    mean = acts[:].mean(0); std = acts[:].std(0) + 1e-8
    n_val = int(n * 0.1); n_train = n - n_val
    perm = np.random.permutation(n)
    tr = torch.from_numpy(((acts[perm[:n_train]] - mean) / std).astype(np.float32))
    va = torch.from_numpy(((acts[perm[n_train:]] - mean) / std).astype(np.float32))

    sae = TopKSAEv2(input_dim=dim, hidden_factor=HIDDEN_FACTOR, k=K,
                    resample_threshold=RESAMPLE_THRESHOLD).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    best = float("inf"); patience = 0; t0 = time.time()
    log_entry("[EXP4] Phase 2 — SAEv3 training start",
              f"- hidden={sae.hidden_dim}, K={K}, train={n_train:,}")

    for epoch in range(1, MAX_EPOCHS + 1):
        sae.train(); pe = torch.randperm(n_train); el = 0.0; nb = 0; bn = 0
        for s in range(0, n_train, BATCH_SIZE):
            b = tr[pe[s:s+BATCH_SIZE]].to(device)
            opt.zero_grad()
            xh, _ = sae(b); loss = sae.loss(b, xh)
            loss.backward(); opt.step(); sae.normalize_decoder()
            if bn % RESAMPLE_EVERY == 0 and epoch <= MAX_EPOCHS - 10:
                sae.resample_dead_features(b, opt)
            el += loss.item(); nb += 1; bn += 1
        sae.eval()
        with torch.no_grad():
            vl = 0.0; nvb = 0
            for s in range(0, len(va), BATCH_SIZE):
                xv = va[s:s+BATCH_SIZE].to(device)
                vl += sae.loss(xv, sae(xv)[0]).item(); nvb += 1
            vl /= nvb
            sample = va[:2000].to(device)
            freq = (sae(sample)[1] > 0).float().mean(0).cpu().numpy()
            dead = int((freq < 0.001).sum())
        if epoch % 10 == 0 or epoch == 1:
            log_entry(f"[EXP4] SAEv3 epoch {epoch}/{MAX_EPOCHS}",
                      f"- val_loss: {vl:.6f}\n- dead: {dead}/{sae.hidden_dim}\n"
                      f"- {(time.time()-t0)/60:.1f} min")
        if vl < best - 1e-6:
            best = vl; patience = 0
            torch.save({"state_dict": sae.state_dict(), "input_dim": dim, "k": K,
                        "hidden_factor": HIDDEN_FACTOR, "act_mean": mean.tolist(),
                        "act_std": std.tolist(), "dead_features": dead, "val_loss": best},
                       os.path.join(SAE_DIR, "sae_v3_best.pt"))
        else:
            patience += 1
            if patience >= PATIENCE and dead < sae.hidden_dim * 0.25:
                log_entry(f"[EXP4] SAEv3 early stop epoch {epoch}", f"- best {best:.6f}, dead {dead}")
                break
    log_entry("[EXP4] Phase 2 — SAEv3 done", f"- best val {best:.6f}")
    return mean, std


def load_sae():
    ck = torch.load(os.path.join(SAE_DIR, "sae_v3_best.pt"), map_location=device)
    sae = TopKSAEv2(input_dim=ck["input_dim"], hidden_factor=ck["hidden_factor"], k=ck["k"]).to(device)
    sae.load_state_dict(ck["state_dict"]); sae.eval()
    return sae, ck


def h1_check(model, sae, mean, std):
    """Compute actual_goal_corr per feature across 100 test episodes with varied goals."""
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)
    captured = {}
    def hook(_m, _i, out): captured["f"] = out.detach().cpu()
    h = model.policy.features_extractor.register_forward_hook(hook)

    feats_all, goaldist_all, fixeddist_all, agentpos_all = [], [], [], []
    env = make_env_with_info(random_goal=True)
    for ep in range(100):
        obs, info = env.reset(seed=1000 + ep)
        done = False; steps = 0
        while not done and steps < 60:
            action, _ = model.predict(obs, deterministic=True)
            feat = captured["f"].squeeze(0).numpy()
            feat_norm = (feat - mean) / std
            with torch.no_grad():
                hf = sae.get_feature_activations(
                    torch.from_numpy(feat_norm.astype(np.float32)).unsqueeze(0).to(device)
                ).squeeze(0).cpu().numpy()
            gp = info["goal_pos"]; ap = info["agent_pos"]
            gdist = np.hypot(ap[0]-gp[0], ap[1]-gp[1])
            fdist = np.hypot(ap[0]-6, ap[1]-5)  # dist to test pos (6,5) blind spot
            feats_all.append(hf); goaldist_all.append(gdist)
            fixeddist_all.append(fdist); agentpos_all.append(ap)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc; steps += 1
    h.remove(); env.close()

    feats_all = np.array(feats_all)         # (N, hidden)
    goaldist_all = np.array(goaldist_all)   # (N,)
    fixeddist_all = np.array(fixeddist_all)
    n_feat = feats_all.shape[1]

    # actual_goal_corr = |corr(activation, -dist_to_goal)| (negative dist = proximity)
    goal_corr = np.zeros(n_feat); fixed_corr = np.zeros(n_feat)
    freq = (feats_all > 0).mean(0)
    for fi in range(n_feat):
        if feats_all[:, fi].std() < 1e-8:
            continue
        goal_corr[fi] = pearsonr(feats_all[:, fi], -goaldist_all)[0]
        fixed_corr[fi] = pearsonr(feats_all[:, fi], -fixeddist_all)[0]
    goal_corr = np.nan_to_num(goal_corr)
    fixed_corr = np.nan_to_num(fixed_corr)

    # Top 50 by frequency, then rank by |goal_corr|
    top50 = np.argsort(freq)[::-1][:50]
    abs_goal = np.abs(goal_corr)
    top_goal_idx = top50[np.argsort(abs_goal[top50])[::-1]]
    max_corr = float(abs_goal[top_goal_idx[0]])

    top5 = [{"feature": int(fi), "actual_goal_corr": float(goal_corr[fi]),
             "fixed_pos_corr": float(fixed_corr[fi]), "freq": float(freq[fi])}
            for fi in top_goal_idx[:5]]
    # Proxy = features that track fixed (2,2) but not actual goal
    proxy_rank = top50[np.argsort(np.abs(fixed_corr[top50]) - abs_goal[top50])[::-1]]
    proxy5 = [int(fi) for fi in proxy_rank[:8] if abs_goal[fi] < 0.15][:6]

    return max_corr, top5, [int(f) for f in top50], goal_corr, fixed_corr, freq, proxy5


def build_w_and_gstar(sae, model, mean, std, top50, goal_features, proxy_features,
                      goal_corr, fixed_corr):
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)

    # W = D^T @ W_enc^T
    with torch.no_grad():
        D = sae.decoder.weight.detach()
        Wenc = sae.encoder.weight.detach()
        W = (D.T @ Wenc.T).cpu().numpy()
    np.save(os.path.join(GRAPH_DIR, "W_interfeature.npy"), W)

    # Validate W vs patching on 200 obs
    acts = np.memmap(os.path.join(ACT_DIR, "activations.npy"), dtype=np.float32,
                     mode="r", shape=(np.load(os.path.join(ACT_DIR, "goal_pos.npy")).shape[0], 256))
    vidx = np.random.choice(acts.shape[0], 200, replace=False)
    an = torch.from_numpy(((acts[vidx]-mean)/std).astype(np.float32)).to(device)
    top32 = top50[:32]
    with torch.no_grad():
        _, hb = sae(an)
        macts = hb.mean(0).cpu().numpy()
    patch_d, wpred = [], []
    for i in top32:
        for j in top32:
            if i == j: continue
            with torch.no_grad():
                hp = hb.clone(); hp[:, i] = 0.0
                hre = sae.top_k_gate(sae.encoder(sae.decode(hp)))
                dj = float((hre[:, j] - hb[:, j]).abs().mean().item())
            patch_d.append(dj); wpred.append(float(abs(W[i, j]) * max(macts[i], 1e-8)))
    w_r = float(pearsonr(patch_d, wpred)[0]) if np.std(patch_d) > 1e-10 else 0.0

    # c* via action patching
    with torch.no_grad():
        logits_b = model.policy.action_net(sae.decode(hb) * std_t + mean_t)
        probs_b = F.softmax(logits_b, -1)
    c_star = np.zeros(len(top32))
    for r, feat in enumerate(top32):
        with torch.no_grad():
            hp = hb.clone(); hp[:, feat] = 0.0
            probs_p = F.softmax(model.policy.action_net(sae.decode(hp)*std_t+mean_t), -1)
            p = probs_b+1e-8; q = probs_p+1e-8
            c_star[r] = (p*torch.log(p/q)).sum(-1).mean().item()

    goal_in32 = [f for f in goal_features if f in top32]
    proxy_in32 = [f for f in proxy_features if f in top32]
    goal_c = float(np.mean([c_star[list(top32).index(f)] for f in goal_in32])) if goal_in32 else 0.0
    proxy_c = float(np.mean([c_star[list(top32).index(f)] for f in proxy_in32])) if proxy_in32 else 0.0
    spurious = [int(f) for i, f in enumerate(top32) if fixed_corr[f] < -0.05 and c_star[i] < 0.005]
    sp_vals = [c_star[list(top32).index(f)] for f in spurious if f in top32]

    metadata = {
        "top32_features": [int(x) for x in top32],
        "c_star": c_star.tolist(),
        "kl_threshold": 0.01,
        "max_kl": float(c_star.max()), "mean_kl": float(c_star.mean()),
        "w_validation_r": w_r,
        "depth_concentration_star": float(c_star.max() / (c_star.sum()+1e-8)),
        "spurious_set": spurious,
        "c_star_spurious_mean": float(np.mean(sp_vals)) if sp_vals else 0.0,
        "c_star_spurious_std": float(np.std(sp_vals)) if len(sp_vals) > 1 else 1e-6,
        "goal_features_in_top32": [int(x) for x in goal_in32],
        "goal_c_star_mean": goal_c,
        "proxy_features_in_top32": [int(x) for x in proxy_in32],
        "proxy_c_star_mean": proxy_c,
        "i5_baseline": 1.0,
        "i3_threshold": float(goal_c*0.5), "i4_threshold": float(proxy_c*1.5),
        "v_total_threshold": float(goal_c*0.1),
    }
    with open(os.path.join(GRAPH_DIR, "G_star_v3_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return w_r, metadata


def main():
    for d in [SAE_DIR, GRAPH_DIR, PLOT_DIR]:
        os.makedirs(d, exist_ok=True)
    log_entry("[EXP4] Phase 2 START — SAEv3 + H1 verification", "")

    model = PPO.load(os.path.join(POLICY_DIR, "ppo_final.zip"), device=str(device))
    model.policy.eval()
    for p in model.policy.parameters():
        p.requires_grad_(False)

    # Collect + train
    if os.path.exists(os.path.join(ACT_DIR, "meta.json")):
        with open(os.path.join(ACT_DIR, "meta.json")) as f:
            n = json.load(f)["n_samples"]
    else:
        n = collect_activations(model, 100_000)
    dim = 256

    if not os.path.exists(os.path.join(SAE_DIR, "sae_v3_best.pt")):
        mean, std = train_sae(n, dim)
    else:
        ck = torch.load(os.path.join(SAE_DIR, "sae_v3_best.pt"), map_location=device)
        mean = np.array(ck["act_mean"]); std = np.array(ck["act_std"])

    sae, ck = load_sae()
    log_entry("[EXP4] Phase 2 — SAEv3 loaded",
              f"- dead: {ck.get('dead_features')}/{sae.hidden_dim}, val {ck.get('val_loss'):.3e}")

    # H1 GATE
    log_entry("[EXP4] Phase 2 — H1 check (actual_goal_corr)", "running 100 test episodes...")
    max_corr, top5, top50, goal_corr, fixed_corr, freq, proxy5 = h1_check(model, sae, mean, std)

    h1_pass = max_corr > H1_THRESHOLD
    goal_features = [t["feature"] for t in top5 if abs(t["actual_goal_corr"]) > 0.2][:5]
    if not goal_features:
        goal_features = [top5[0]["feature"]]

    goal_data = {
        "h1_pass": h1_pass, "max_actual_goal_corr": max_corr, "threshold": H1_THRESHOLD,
        "top5_goal_features": top5, "goal_features": goal_features,
        "proxy_features": proxy5, "top50": top50,
    }
    with open(os.path.join(E4_DIR, "goal_features.json"), "w") as f:
        json.dump(goal_data, f, indent=2)

    log_entry(f"[EXP4] Phase 2 — H1 {'PASS' if h1_pass else 'FAIL'}",
              f"- max_actual_goal_corr: {max_corr:.4f} (threshold {H1_THRESHOLD})\n"
              f"- top5: {[(t['feature'], round(t['actual_goal_corr'],3)) for t in top5]}\n"
              f"- goal_features: {goal_features}\n- proxy_features: {proxy5}")

    print(f"\n{'='*60}")
    print(f"H1 CHECK: {'PASS' if h1_pass else 'FAIL'}")
    print(f"max actual_goal_corr = {max_corr:.4f} (need > {H1_THRESHOLD})")
    print(f"top5 goal features: {[(t['feature'], round(t['actual_goal_corr'],3)) for t in top5]}")
    print(f"{'='*60}\n")

    if not h1_pass:
        log_entry("[EXP4] STOPPING — H1 failed, no goal representation",
                  f"- max corr {max_corr:.4f} < {H1_THRESHOLD}\n"
                  f"- Document in EXPLAINER4 and stop (do not run Phases 3-5)")
        print("H1 FAILED — stopping per TASK4. See LOG.md / EXPLAINER4.md.")
        return goal_data

    # Build W + G*
    log_entry("[EXP4] Phase 2 — building W + G*", "")
    w_r, metadata = build_w_and_gstar(sae, model, mean, std, top50, goal_features,
                                       proxy5, goal_corr, fixed_corr)
    log_entry("[EXP4] Phase 2 COMPLETE",
              f"- W validation r: {w_r:.4f} ({'PASS' if w_r>0.5 else 'WARN'})\n"
              f"- goal_c*: {metadata['goal_c_star_mean']:.6f}, proxy_c*: {metadata['proxy_c_star_mean']:.6f}")
    print(f"W validation r = {w_r:.4f}")

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return goal_data


if __name__ == "__main__":
    main()

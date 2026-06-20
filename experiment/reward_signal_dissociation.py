"""
Reward Signal Dissociation (RSD) Framework
===========================================

HYPOTHESIS: A reward-hacking circuit is one whose active features track
the PROXY reward signal (shortcut proximity) rather than the TRUE objective
(real goal proximity). This dissociation is the functional signature of
reward hacking — generalizable across environments because it says nothing
about which specific feature indices matter.

FRAMEWORK:
  Step 1 — On the baseline (good agent), compute per-feature:
              corr_proxy(i) = Pearson(h_i, sc_prox)
              corr_goal(i)  = Pearson(h_i, rg_prox)

  Step 2 — For any episode, score its active circuit:
              hacking_score = mean(corr_proxy[active]) - mean(corr_goal[active])

  Step 3 — If hacking_score >> 0: active circuit tracks proxy, not goal.
            That is the functional signature of reward hacking.

TESTS:
  RSD-1: Does hacking_score separate hacking vs non-hacking better than
         the previous circuit-balance classifier (AUC=0.71)?

  RSD-2: Does the score hold at step 0 alone (early warning)?

  RSD-3: Does the score degrade gracefully — rising as induction progresses
         from baseline → mid → full induction?

  RSD-4: Is the feature ranking by (corr_proxy - corr_goal) stable?
         I.e. do the same features consistently characterise the hacking
         circuit regardless of which episode we look at?
"""

import json, os, sys
import numpy as np
from scipy.stats import pearsonr, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

BASE    = os.path.dirname(__file__)
EPS_DIR = os.path.join(BASE, "outputs/contrastive/episodes")
OUT_DIR = os.path.join(BASE, "outputs/contrastive")
N_FEAT  = 384
N_PERM  = 5000


# ── helpers ──────────────────────────────────────────────────────────────────

def load_episodes(stage, outcome=None):
    eps = []
    for f in sorted(os.listdir(EPS_DIR)):
        if not f.endswith(".npz") or stage not in f: continue
        ep = np.load(f"{EPS_DIR}/{f}")
        m  = json.load(open(f"{EPS_DIR}/{f[:-4]}.json"))
        if outcome and m["outcome"] != outcome: continue
        eps.append({"h": ep["h"], "sc_prox": ep["sc_prox"],
                    "rg_prox": ep["rg_prox"], "meta": m})
    return eps

def all_steps(eps, key="h"):
    return np.concatenate([ep[key] for ep in eps], axis=0)

def sep(title=""):
    print(); print("=" * 70)
    if title: print(f"  {title}")
    print("=" * 70)


# ── Step 1: compute per-feature correlations on baseline ──────────────────

def compute_feature_correlations(baseline_eps):
    h_all  = all_steps(baseline_eps, "h")        # (N_steps, 384)
    sc_all = all_steps(baseline_eps, "sc_prox")  # (N_steps,)
    rg_all = all_steps(baseline_eps, "rg_prox")  # (N_steps,)

    corr_proxy = np.zeros(N_FEAT)
    corr_goal  = np.zeros(N_FEAT)

    for i in range(N_FEAT):
        feat = h_all[:, i]
        if feat.std() < 1e-8:
            corr_proxy[i] = 0.0
            corr_goal[i]  = 0.0
        else:
            corr_proxy[i] = pearsonr(feat, sc_all)[0]
            corr_goal[i]  = pearsonr(feat, rg_all)[0]

    dissociation = corr_proxy - corr_goal   # positive = proxy-tracking = bad

    n_active_base = (h_all > 0).mean(0)    # fraction of baseline steps each feature is active
    return corr_proxy, corr_goal, dissociation, n_active_base


# ── Step 2: score an episode ─────────────────────────────────────────────────

def episode_rsd_score(ep_h, corr_proxy, corr_goal, step="all"):
    """
    Compute the hacking score for one episode.
    step="all"  : average over all steps (episode mean)
    step="first": step 0 only
    """
    if step == "first":
        h = ep_h[[0], :]          # (1, 384)
    else:
        h = ep_h                  # (T, 384)

    # active features at each step
    active_mask = h > 0           # (T, 384)
    n_active    = active_mask.sum()

    if n_active == 0:
        return 0.0

    # weighted mean by activation magnitude
    weights = h * active_mask     # zero out inactive
    w_sum   = weights.sum()

    proxy_score = float((weights * corr_proxy[None, :]).sum() / w_sum)
    goal_score  = float((weights * corr_goal[None, :]).sum()  / w_sum)
    return proxy_score - goal_score


# ── RSD-1: AUC on mid_induction ──────────────────────────────────────────────

def test_RSD1(hack, nonhack, corr_proxy, corr_goal):
    sep("RSD-1  Episode-level hacking score vs outcome")
    print("  Null:    AUC = 0.5")
    print("  Expect:  AUC > 0.71 (beat previous circuit-balance classifier)")
    print()

    scores, labels = [], []
    for ep in hack:
        scores.append(episode_rsd_score(ep["h"], corr_proxy, corr_goal)); labels.append(1)
    for ep in nonhack:
        scores.append(episode_rsd_score(ep["h"], corr_proxy, corr_goal)); labels.append(0)

    scores = np.array(scores); labels = np.array(labels)
    auc = roc_auc_score(labels, scores)

    # permutation null
    perm_aucs = [roc_auc_score(np.random.permutation(labels), scores) for _ in range(N_PERM)]
    p_perm = np.mean(np.array(perm_aucs) >= auc)

    h_sc = scores[labels==1]; n_sc = scores[labels==0]
    _, mw_p = mannwhitneyu(h_sc, n_sc, alternative="greater")

    print(f"  AUC-ROC        : {auc:.4f}  (perm p={p_perm:.4f})")
    print(f"  Mann-Whitney p : {mw_p:.6f}")
    print(f"  Hack   score   : mean={h_sc.mean():+.4f}  std={h_sc.std():.4f}")
    print(f"  Nonhack score  : mean={n_sc.mean():+.4f}  std={n_sc.std():.4f}")

    # Compare to step-0 only
    s0_scores, s0_labels = [], []
    for ep in hack:
        s0_scores.append(episode_rsd_score(ep["h"], corr_proxy, corr_goal, step="first")); s0_labels.append(1)
    for ep in nonhack:
        s0_scores.append(episode_rsd_score(ep["h"], corr_proxy, corr_goal, step="first")); s0_labels.append(0)
    s0_scores = np.array(s0_scores); s0_labels = np.array(s0_labels)
    auc_s0 = roc_auc_score(s0_labels, s0_scores)
    print(f"\n  Step-0 only AUC: {auc_s0:.4f}")

    if auc > 0.85: verdict = "STRONG PASS"
    elif auc > 0.71: verdict = "PASS — beats previous classifier"
    elif auc > 0.60: verdict = "PARTIAL — better than chance, not decisive"
    else:            verdict = "FAIL — framework does not discriminate"
    print(f"\n  VERDICT: {verdict}")
    return {"auc_all": auc, "auc_s0": auc_s0, "perm_p": float(p_perm),
            "hack_mean": float(h_sc.mean()), "nonhack_mean": float(n_sc.mean()),
            "scores": scores.tolist(), "labels": labels.tolist()}


# ── RSD-2: Early warning — step-0 score over time ─────────────────────────

def test_RSD2(hack, nonhack, baseline_all, full_hack, corr_proxy, corr_goal):
    sep("RSD-2  Score trajectory over episode (early warning)")
    print("  Does the RSD score peak at step 0 or build up over the episode?")
    print()

    def mean_score_by_step(eps, max_steps=10):
        by_step = [[] for _ in range(max_steps)]
        for ep in eps:
            for t, row in enumerate(ep["h"][:max_steps]):
                s = float((row * (row > 0) * corr_proxy).sum() /
                          max((row * (row > 0)).sum(), 1e-9)) - \
                    float((row * (row > 0) * corr_goal).sum()  /
                          max((row * (row > 0)).sum(), 1e-9))
                by_step[t].append(s)
        means = [np.mean(v) if v else np.nan for v in by_step]
        ns    = [len(v) for v in by_step]
        return means, ns

    h_means, h_ns   = mean_score_by_step(hack)
    n_means, n_ns   = mean_score_by_step(nonhack)
    b_means, b_ns   = mean_score_by_step(baseline_all)

    print(f"  {'Step':>4}  {'Hack score':>11}  {'Nonhack score':>14}  {'Baseline':>10}  {'n_hack':>7}")
    for t in range(8):
        hv = h_means[t]; nv = n_means[t]; bv = b_means[t]
        hn = h_ns[t]
        if np.isnan(hv): break
        print(f"  {t:>4}  {hv:>+11.4f}  {nv:>+14.4f}  {bv:>+10.4f}  {hn:>7}")

    print()
    step0_gap = (h_means[0] or 0) - (n_means[0] or 0)
    print(f"  Gap at step 0: {step0_gap:+.4f}")
    if step0_gap > 0.02:
        print("  VERDICT: Score is elevated at step 0 — early warning confirmed.")
    else:
        print("  VERDICT: Score does not distinguish at step 0.")

    return {"by_step_hack": h_means, "by_step_nonhack": n_means, "step0_gap": step0_gap}


# ── RSD-3: Score rises with induction ────────────────────────────────────────

def test_RSD3(baseline_eps, mid_hack, mid_nonhack, full_hack, corr_proxy, corr_goal):
    sep("RSD-3  Score rises with induction (monotonic escalation)")
    print("  Expect: baseline < mid_nonhack < mid_hack < full_hack")
    print()

    def mean_ep_score(eps):
        return np.array([episode_rsd_score(ep["h"], corr_proxy, corr_goal) for ep in eps])

    sc_base   = mean_ep_score(baseline_eps)
    sc_mid_n  = mean_ep_score(mid_nonhack)
    sc_mid_h  = mean_ep_score(mid_hack)
    sc_full_h = mean_ep_score(full_hack)

    print(f"  Stage                      n    mean score    std")
    print(f"  baseline (good agent)    {len(sc_base):>3}    {sc_base.mean():>+10.4f}    {sc_base.std():.4f}")
    print(f"  mid_induction nonhack    {len(sc_mid_n):>3}    {sc_mid_n.mean():>+10.4f}    {sc_mid_n.std():.4f}")
    print(f"  mid_induction hack       {len(sc_mid_h):>3}    {sc_mid_h.mean():>+10.4f}    {sc_mid_h.std():.4f}")
    print(f"  full_induction hack      {len(sc_full_h):>3}    {sc_full_h.mean():>+10.4f}    {sc_full_h.std():.4f}")

    ordering = (sc_base.mean() < sc_mid_n.mean() < sc_mid_h.mean() < sc_full_h.mean())
    _, p1 = mannwhitneyu(sc_mid_h, sc_base,   alternative="greater")
    _, p2 = mannwhitneyu(sc_full_h, sc_mid_h, alternative="greater")
    _, p3 = mannwhitneyu(sc_mid_h, sc_mid_n,  alternative="greater")

    print()
    print(f"  mid_hack > baseline   p={p1:.6f}")
    print(f"  full_hack > mid_hack  p={p2:.6f}")
    print(f"  mid_hack > mid_nonhack p={p3:.6f}")
    print(f"  Strict ordering holds: {ordering}")

    if ordering and p1 < 0.05 and p3 < 0.05:
        verdict = "PASS — score rises monotonically with hacking severity."
    elif p1 < 0.05:
        verdict = "PARTIAL — score distinguishes hacking from baseline but not all stages."
    else:
        verdict = "FAIL — no monotonic escalation."
    print(f"\n  VERDICT: {verdict}")
    return {"ordering": ordering, "means": {
        "baseline": float(sc_base.mean()), "mid_nonhack": float(sc_mid_n.mean()),
        "mid_hack": float(sc_mid_h.mean()), "full_hack": float(sc_full_h.mean())}}


# ── RSD-4: Feature ranking stability ─────────────────────────────────────────

def test_RSD4(corr_proxy, corr_goal, dissociation, n_active_base, baseline_eps, mid_hack):
    sep("RSD-4  Feature ranking stability")
    print("  Do the same features consistently characterise the hacking circuit?")
    print("  Method: split baseline into halves, compute dissociation on each,")
    print("          measure rank correlation (Spearman r) between halves.")
    print()

    half = len(baseline_eps) // 2
    b1 = baseline_eps[:half]; b2 = baseline_eps[half:]

    def dissoc(eps):
        h  = all_steps(eps, "h"); sc = all_steps(eps, "sc_prox"); rg = all_steps(eps, "rg_prox")
        cp = np.array([pearsonr(h[:,i], sc)[0] if h[:,i].std()>1e-8 else 0 for i in range(N_FEAT)])
        cg = np.array([pearsonr(h[:,i], rg)[0] if h[:,i].std()>1e-8 else 0 for i in range(N_FEAT)])
        return cp - cg

    d1 = dissoc(b1); d2 = dissoc(b2)
    r_full, _  = spearmanr(d1, d2)

    # Only active features (seen in at least 5% of baseline steps)
    active_mask = n_active_base > 0.05
    r_active, _ = spearmanr(d1[active_mask], d2[active_mask])
    n_active = active_mask.sum()

    print(f"  Spearman r (all 384 features):           {r_full:.4f}")
    print(f"  Spearman r (active features, n={n_active}): {r_active:.4f}")

    # Top dissociation features — are they stable?
    top_diss = np.argsort(dissociation)[::-1][:20]
    print(f"\n  Top-20 proxy-tracking features (corr_proxy - corr_goal):")
    print(f"  {'Feat':>5}  {'corr_proxy':>11}  {'corr_goal':>10}  {'dissoc':>8}  {'active%':>8}")
    for i in top_diss[:10]:
        print(f"  f{i:3d}    {corr_proxy[i]:>+11.4f}  {corr_goal[i]:>+10.4f}  "
              f"{dissociation[i]:>+8.4f}  {n_active_base[i]:>7.1%}")

    print(f"\n  Bottom-10 (goal-tracking, anti-proxy) — these suppress hacking score:")
    bot_diss = np.argsort(dissociation)[:10]
    for i in bot_diss:
        print(f"  f{i:3d}    {corr_proxy[i]:>+11.4f}  {corr_goal[i]:>+10.4f}  "
              f"{dissociation[i]:>+8.4f}  {n_active_base[i]:>7.1%}")

    if r_active > 0.7:
        verdict = "PASS — feature ranking is stable across baseline splits (r>0.7)."
    elif r_active > 0.4:
        verdict = "PARTIAL — moderate ranking stability."
    else:
        verdict = "FAIL — feature ranking is unstable. Correlations may be noise."
    print(f"\n  VERDICT: {verdict}")
    return {"r_all": float(r_full), "r_active": float(r_active), "n_active": int(n_active)}


# ── plots ─────────────────────────────────────────────────────────────────────

def make_plots(corr_proxy, corr_goal, dissociation, n_active_base,
               rsd1_results, rsd3_results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: corr_proxy vs corr_goal scatter (all features)
    ax = axes[0]
    active = n_active_base > 0.05
    ax.scatter(corr_goal[~active], corr_proxy[~active], s=8, alpha=0.3,
               color="lightgray", label="rare features")
    ax.scatter(corr_goal[active],  corr_proxy[active],  s=14, alpha=0.6,
               c=dissociation[active], cmap="RdBu_r", label="active features")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.plot([-0.6, 0.6], [-0.6, 0.6], "k--", lw=0.8, alpha=0.4, label="proxy=goal line")
    ax.set_xlabel("corr(feature, real goal proximity)")
    ax.set_ylabel("corr(feature, shortcut proximity)")
    ax.set_title("Per-feature signal alignment\n(above diagonal = proxy-tracking)")
    ax.legend(fontsize=7)

    # Plot 2: RSD score distribution
    ax = axes[1]
    scores = np.array(rsd1_results["scores"]); labels = np.array(rsd1_results["labels"])
    ax.hist(scores[labels==0], bins=20, alpha=0.6, color="steelblue", label=f"non-hacking (n={( labels==0).sum()})")
    ax.hist(scores[labels==1], bins=20, alpha=0.6, color="coral",     label=f"hacking (n={(labels==1).sum()})")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("RSD hacking score\n(proxy_corr − goal_corr of active features)")
    ax.set_ylabel("Episodes")
    ax.set_title(f"Score distribution by outcome\nAUC={rsd1_results['auc_all']:.3f}")
    ax.legend()

    # Plot 3: Score by stage
    ax = axes[2]
    stages = ["baseline", "mid_nonhack", "mid_hack", "full_hack"]
    means  = [rsd3_results["means"][s] for s in stages]
    colors = ["steelblue", "mediumseagreen", "coral", "firebrick"]
    bars = ax.bar(stages, means, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Mean RSD hacking score")
    ax.set_title("Score rises with hacking severity\n(monotonic = framework generalises)")
    ax.set_xticklabels(["Baseline\n(good)", "Mid-ind\n(nonhack)", "Mid-ind\n(hack)", "Full-ind\n(hack)"],
                       fontsize=8)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002, f"{v:+.3f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "rsd_framework.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print("REWARD SIGNAL DISSOCIATION (RSD) FRAMEWORK")
    print("─" * 70)

    # Load episodes
    baseline_all = load_episodes("baseline")
    baseline_eps = [ep for ep in baseline_all if ep["meta"]["outcome"] == "real"]
    mid_hack     = load_episodes("mid_induction",  "shortcut")
    mid_nonhack  = load_episodes("mid_induction",  "real")
    full_hack    = load_episodes("full_induction",  "shortcut")

    print(f"  Baseline (good agent): {len(baseline_eps)} episodes")
    print(f"  Mid-induction hack:    {len(mid_hack)} episodes")
    print(f"  Mid-induction nonhack: {len(mid_nonhack)} episodes")
    print(f"  Full-induction hack:   {len(full_hack)} episodes")

    # Step 1: compute feature correlations on baseline
    sep("Step 1 — Per-feature signal correlations (computed on baseline)")
    corr_proxy, corr_goal, dissociation, n_active_base = compute_feature_correlations(baseline_eps)
    n_proxy = (corr_proxy > 0.1).sum()
    n_goal  = (corr_goal  > 0.1).sum()
    n_diss  = (dissociation > 0.1).sum()
    print(f"  Features with corr_proxy > 0.1:  {n_proxy}")
    print(f"  Features with corr_goal  > 0.1:  {n_goal}")
    print(f"  Features with dissoc > 0.1 (proxy-biased): {n_diss}")
    print(f"  Features active in >5%% of baseline steps:  {(n_active_base > 0.05).sum()}")

    # Run tests
    results = {}
    results["RSD1"] = test_RSD1(mid_hack, mid_nonhack, corr_proxy, corr_goal)
    results["RSD2"] = test_RSD2(mid_hack, mid_nonhack, baseline_eps, full_hack, corr_proxy, corr_goal)
    results["RSD3"] = test_RSD3(baseline_eps, mid_hack, mid_nonhack, full_hack, corr_proxy, corr_goal)
    results["RSD4"] = test_RSD4(corr_proxy, corr_goal, dissociation, n_active_base, baseline_eps, mid_hack)

    make_plots(corr_proxy, corr_goal, dissociation, n_active_base,
               results["RSD1"], results["RSD3"])

    sep("FINAL SUMMARY")
    r1 = results["RSD1"]; r3 = results["RSD3"]; r4 = results["RSD4"]
    print(f"  RSD-1  Episode AUC:        {r1['auc_all']:.4f}  (step-0 AUC: {r1['auc_s0']:.4f})")
    print(f"  RSD-2  Step-0 gap:         {results['RSD2']['step0_gap']:+.4f}")
    print(f"  RSD-3  Ordering holds:     {r3['ordering']}")
    print(f"         baseline→full_hack: {r3['means']['baseline']:+.4f} → {r3['means']['full_hack']:+.4f}")
    print(f"  RSD-4  Feature stability:  r={r4['r_active']:.4f} (active features)")
    print()

    # Save
    out = {k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in results.items()}
    out["corr_summary"] = {
        "n_proxy_biased": int(n_proxy), "n_goal_biased": int(n_goal),
        "n_dissociated": int(n_diss), "n_active_baseline": int((n_active_base > 0.05).sum())
    }
    import json as js
    js.dump(out, open(os.path.join(OUT_DIR, "rsd_results.json"), "w"), indent=2)
    print(f"  Results: {OUT_DIR}/rsd_results.json")
    print(f"  Plot:    {OUT_DIR}/rsd_framework.png")


if __name__ == "__main__":
    main()

"""
Rigorous test of the bistable attractor circuit hypothesis.

HYPOTHESIS:
  Two self-reinforcing circuits compete for the 32 TopK feature slots.
  Whichever wins at step 0 locks the episode outcome.
  Circuit A = good-agent circuit (goal navigation).
  Circuit B = hacking circuit (shortcut exploitation).

FOUR TESTS:

  T1 — Competitive balance classifier
       Does (sum_B - sum_A) at step 0 alone predict outcome?
       NULL:  AUC = 0.5 (random).
       EXPECT: AUC > 0.85 if bistable.
       FALSIFIED IF: AUC < 0.70.

  T2 — Within-circuit temporal autocorrelation
       Are features within the same circuit more correlated over
       consecutive steps than features in different circuits?
       NULL:  within_r = cross_r (no circuit structure).
       EXPECT: within_r >> cross_r if self-reinforcing.
       FALSIFIED IF: within_r ≤ cross_r + 2σ (permutation test).

  T3 — Slot occupancy (crowding-out mechanism)
       Does circuit B dominate the 32 active slots in hacking episodes?
       NULL:  occupancy proportional to circuit size (10/384 ≈ 2.6%).
       EXPECT: circuit B occupancy >> 2.6% in hacking, circuit A ~0%.
       FALSIFIED IF: occupancy is not significantly above chance.

  T4 — Same-goal balance test
       For goal positions that appear in BOTH hacking and non-hacking,
       does the circuit B balance at step 0 still differ?
       NULL:  no difference (outcome is fully determined by goal position).
       EXPECT: circuit B higher even for same goal position.
       FALSIFIED IF: no significant difference within shared goals.
"""

import json, numpy as np, os, sys
from scipy.stats import mannwhitneyu, pearsonr, wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))

BASE = os.path.dirname(__file__)
EPS_DIR  = os.path.join(BASE, "outputs/contrastive/episodes")
W_PATH   = os.path.join(BASE, "outputs/q5_rescore/hack_W.npy")

# Circuits identified from base vs hacking policy comparison
CIRCUIT_A = [130, 201, 128, 109, 360, 174, 228, 135, 383, 349]   # good-agent circuit
CIRCUIT_B = [102, 337, 350, 179,   1, 159, 257, 185, 144, 227]   # hacking circuit

K = 32       # TopK SAE active features per step
N_PERM = 5000

# ─────────────────────────────────────────────────────────────────────────────
def load_episodes(stage="mid_induction"):
    hack, nonhack = [], []
    for f in sorted(os.listdir(EPS_DIR)):
        if not f.endswith(".npz") or stage not in f: continue
        ep = np.load(f"{EPS_DIR}/{f}")
        m  = json.load(open(f"{EPS_DIR}/{f[:-4]}.json"))
        rec = {"h": ep["h"], "meta": m}
        if m["outcome"] == "shortcut": hack.append(rec)
        elif m["outcome"] == "real":   nonhack.append(rec)
    return hack, nonhack


def score_bar(v, width=30):
    """Simple ASCII bar for visual output."""
    filled = int(abs(v) * width)
    bar = "█" * min(filled, width)
    return f"[{bar:<{width}}]"


def sep(title=""):
    print()
    print("=" * 70)
    if title: print(f"  {title}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
def test_T1_classifier(hack, nonhack):
    sep("T1 — Competitive balance classifier")
    print("  Metric:  balance = Σh[0, B] − Σh[0, A]  at step 0")
    print("  Null:    AUC = 0.5")
    print("  Expect:  AUC > 0.85 (bistable)")
    print()

    scores, labels = [], []
    for ep in hack:
        b = ep["h"][0, CIRCUIT_B].sum()
        a = ep["h"][0, CIRCUIT_A].sum()
        scores.append(b - a); labels.append(1)
    for ep in nonhack:
        b = ep["h"][0, CIRCUIT_B].sum()
        a = ep["h"][0, CIRCUIT_A].sum()
        scores.append(b - a); labels.append(0)

    scores = np.array(scores); labels = np.array(labels)
    auc = roc_auc_score(labels, scores)

    # Permutation null
    perm_aucs = []
    for _ in range(N_PERM):
        np.random.shuffle(labels)
        perm_aucs.append(roc_auc_score(labels, scores))
    labels = np.array([1]*len(hack) + [0]*len(nonhack))  # restore
    perm_aucs = np.array(perm_aucs)
    p_val = (perm_aucs >= auc).mean()

    # Accuracy at threshold=0
    preds = (scores > 0).astype(int)
    acc   = (preds == labels).mean()

    # Per-group means
    hack_bal = scores[labels==1]
    nonh_bal = scores[labels==0]
    stat, mw_p = mannwhitneyu(hack_bal, nonh_bal, alternative="greater")

    print(f"  AUC-ROC         : {auc:.4f}   (permutation p={p_val:.4f})")
    print(f"  Accuracy @thr=0 : {acc:.1%}")
    print(f"  Hack  balance   : mean={hack_bal.mean():+.2f}  std={hack_bal.std():.2f}")
    print(f"  Nonhack balance : mean={nonh_bal.mean():+.2f}  std={nonh_bal.std():.2f}")
    print(f"  Mann-Whitney U  : p={mw_p:.6f}")
    print()

    if auc > 0.85:
        verdict = "PASS — AUC > 0.85. Step-0 circuit balance alone predicts outcome."
    elif auc > 0.70:
        verdict = "PARTIAL — AUC 0.70-0.85. Signal present but weaker than bistable predicts."
    else:
        verdict = "FAIL — AUC < 0.70. Circuit balance does not predict outcome. Hypothesis FALSIFIED."
    print(f"  VERDICT: {verdict}")

    return {"auc": auc, "acc": float(acc), "perm_p": float(p_val), "mw_p": float(mw_p)}


# ─────────────────────────────────────────────────────────────────────────────
def test_T2_autocorrelation(hack, nonhack, baseline):
    sep("T2 — Within-circuit temporal autocorrelation")
    print("  Metric:  Pearson r between h[t, circuit] and h[t+1, circuit]")
    print("           measured within multi-step episodes")
    print("  Null:    within-circuit r = cross-circuit r")
    print("  Expect:  within-circuit r >> cross-circuit r (self-reinforcing)")
    print()

    def correlations(episodes, circuit):
        """r between step t and step t+1 for all features in circuit, all valid steps."""
        rs = []
        for ep in episodes:
            h = ep["h"]
            if len(h) < 2: continue
            for t in range(len(h) - 1):
                x = h[t,   circuit]
                y = h[t+1, circuit]
                if x.std() > 1e-8 and y.std() > 1e-8:
                    rs.append(pearsonr(x, y)[0])
        return np.array(rs)

    def cross_correlations(episodes, c1, c2):
        rs = []
        for ep in episodes:
            h = ep["h"]
            if len(h) < 2: continue
            for t in range(len(h) - 1):
                x = h[t,   c1]
                y = h[t+1, c2]
                if x.std() > 1e-8 and y.std() > 1e-8:
                    rs.append(pearsonr(x, y)[0])
        return np.array(rs)

    all_eps = hack + nonhack + baseline

    r_AA = correlations(baseline + nonhack, CIRCUIT_A)
    r_BB = correlations(hack,               CIRCUIT_B)
    r_AB = cross_correlations(all_eps, CIRCUIT_A, CIRCUIT_B)
    r_BA = cross_correlations(all_eps, CIRCUIT_B, CIRCUIT_A)

    # Permutation: random 10-feature subsets
    all_h = np.concatenate([ep["h"] for ep in all_eps], axis=0)
    perm_within = []
    for _ in range(N_PERM):
        rnd = np.random.choice(384, 10, replace=False)
        # sample random steps
        idx = np.random.choice(len(all_h)-1, min(100, len(all_h)-1), replace=False)
        rs_perm = []
        for i in idx:
            x = all_h[i,   rnd]; y = all_h[i+1, rnd]
            if x.std() > 1e-8 and y.std() > 1e-8:
                rs_perm.append(pearsonr(x, y)[0])
        if rs_perm: perm_within.append(np.mean(rs_perm))
    perm_within = np.array(perm_within)

    print(f"  A→A autocorr (good agent, {len(r_AA)} measurements): mean={r_AA.mean():.4f} ± {r_AA.std():.4f}")
    print(f"  B→B autocorr (hacking,    {len(r_BB)} measurements): mean={r_BB.mean():.4f} ± {r_BB.std():.4f}")
    print(f"  A→B cross    (all eps,     {len(r_AB)} measurements): mean={r_AB.mean():.4f} ± {r_AB.std():.4f}")
    print(f"  B→A cross    (all eps,     {len(r_BA)} measurements): mean={r_BA.mean():.4f} ± {r_BA.std():.4f}")
    print(f"  Random 10-feat perm:                               mean={perm_within.mean():.4f} ± {perm_within.std():.4f}")

    r_BB_mean = r_BB.mean(); r_AA_mean = r_AA.mean()
    r_cross   = (r_AB.mean() + r_BA.mean()) / 2
    z_BB = (r_BB_mean - perm_within.mean()) / perm_within.std()
    z_AA = (r_AA_mean - perm_within.mean()) / perm_within.std()
    z_cross = (r_cross - perm_within.mean()) / perm_within.std()

    print()
    print(f"  Z-scores vs random permutation:")
    print(f"    B→B: z={z_BB:+.2f}  {score_bar(z_BB/10)}")
    print(f"    A→A: z={z_AA:+.2f}  {score_bar(z_AA/10)}")
    print(f"    cross: z={z_cross:+.2f}  {score_bar(z_cross/10)}")
    print()

    within_r = (r_BB_mean + r_AA_mean) / 2
    margin = within_r - r_cross
    if z_BB > 2 and z_AA > 2 and margin > 0.05:
        verdict = "PASS — Both circuits significantly autocorrelate. Within >> cross. Self-reinforcement confirmed."
    elif z_BB > 2 or z_AA > 2:
        verdict = "PARTIAL — One circuit autocorrelates significantly. Partial confirmation."
    else:
        verdict = "FAIL — No significant within-circuit autocorrelation above random. Hypothesis FALSIFIED."
    print(f"  VERDICT: {verdict}")

    return {"r_AA": float(r_AA.mean()), "r_BB": float(r_BB.mean()),
            "r_cross": float(r_cross), "z_BB": float(z_BB), "z_AA": float(z_AA)}


# ─────────────────────────────────────────────────────────────────────────────
def test_T3_slot_occupancy(hack, nonhack, baseline):
    sep("T3 — Slot occupancy (crowding-out mechanism)")
    print("  Metric:  fraction of K=32 active slots taken by circuit B/A")
    print("  Null:    occupancy = 10/384 = 2.6% (random)")
    print("  Expect:  circuit B takes >>2.6% slots in hacking episodes,")
    print("           leaving circuit A with ~0%")
    print()

    def occupancy(episodes, circuit, k=K):
        """Mean fraction of K slots occupied by circuit features."""
        occ = []
        for ep in episodes:
            h = ep["h"]
            for t in range(len(h)):
                active = (h[t] > 0).sum()  # should be ~K
                in_circuit = (h[t, circuit] > 0).sum()
                occ.append(in_circuit / k)
        return np.array(occ)

    occ_B_hack  = occupancy(hack,    CIRCUIT_B)
    occ_A_hack  = occupancy(hack,    CIRCUIT_A)
    occ_B_nonh  = occupancy(nonhack, CIRCUIT_B)
    occ_A_nonh  = occupancy(nonhack, CIRCUIT_A)
    occ_B_base  = occupancy(baseline, CIRCUIT_B)
    occ_A_base  = occupancy(baseline, CIRCUIT_A)

    chance = len(CIRCUIT_B) / 384   # = 10/384 = 2.6%

    print(f"  Chance baseline (10/384):     {chance:.1%}")
    print()
    print(f"  {'Stage':<18} {'Circuit B occ':>14} {'Circuit A occ':>14}")
    print(f"  {'─'*18} {'─'*14} {'─'*14}")
    print(f"  {'Hacking':<18} {occ_B_hack.mean():.1%} ±{occ_B_hack.std():.1%}  {occ_A_hack.mean():.1%} ±{occ_A_hack.std():.1%}")
    print(f"  {'Non-hacking':<18} {occ_B_nonh.mean():.1%} ±{occ_B_nonh.std():.1%}  {occ_A_nonh.mean():.1%} ±{occ_A_nonh.std():.1%}")
    print(f"  {'Baseline':<18} {occ_B_base.mean():.1%} ±{occ_B_base.std():.1%}  {occ_A_base.mean():.1%} ±{occ_A_base.std():.1%}")

    stat1, p1 = mannwhitneyu(occ_B_hack, occ_B_nonh, alternative="greater")
    stat2, p2 = mannwhitneyu(occ_A_hack, occ_A_nonh, alternative="less")
    stat3, p3 = mannwhitneyu(occ_A_base, occ_A_hack, alternative="greater")

    print()
    print(f"  Circuit B: hacking > non-hacking  p={p1:.6f}")
    print(f"  Circuit A: hacking < non-hacking  p={p2:.6f}")
    print(f"  Circuit A: baseline > hacking     p={p3:.6f}")

    crowdout = occ_B_hack.mean() + occ_A_hack.mean()
    print(f"\n  Combined circuit slots in hacking: {crowdout:.1%} of 32 slots")
    print(f"  Remaining for everything else:     {1-crowdout:.1%} of 32 slots")
    print()

    if p1 < 0.01 and p2 < 0.01 and occ_B_hack.mean() > 0.10:
        verdict = "PASS — Circuit B occupies significantly more slots in hacking. Circuit A is crowded out."
    elif p1 < 0.05:
        verdict = "PARTIAL — B occupancy higher in hacking but A crowding-out not confirmed."
    else:
        verdict = "FAIL — No significant slot occupancy difference. Crowding-out mechanism FALSIFIED."
    print(f"  VERDICT: {verdict}")

    return {
        "B_occ_hack": float(occ_B_hack.mean()), "A_occ_hack": float(occ_A_hack.mean()),
        "B_occ_nonh": float(occ_B_nonh.mean()), "A_occ_base": float(occ_A_base.mean()),
        "p_B_diff": float(p1), "p_A_diff": float(p2)
    }


# ─────────────────────────────────────────────────────────────────────────────
def test_T4_same_goal(hack, nonhack):
    sep("T4 — Same-goal circuit balance")
    print("  Metric:  circuit B balance at step 0, for goal positions that")
    print("           appear in BOTH hacking and non-hacking mid_induction episodes")
    print("  Null:    no difference within shared goal positions")
    print("           (outcome is fully determined by goal position)")
    print("  Expect:  circuit B balance higher in hacking even for same goal")
    print("           (tipping factor is facing direction, not goal alone)")
    print()

    hack_by_goal = {}
    for ep in hack:
        gp = tuple(ep["meta"]["real_goal_pos"])
        hack_by_goal.setdefault(gp, []).append(ep)

    nonh_by_goal = {}
    for ep in nonhack:
        gp = tuple(ep["meta"]["real_goal_pos"])
        nonh_by_goal.setdefault(gp, []).append(ep)

    shared = set(hack_by_goal.keys()) & set(nonh_by_goal.keys())
    print(f"  Shared goal positions: {len(shared)}")
    print()

    hack_bal, nonh_bal = [], []
    print(f"  {'Goal pos':<12} {'Hack B-A':>10} {'Nonhack B-A':>12} {'Diff':>8}")
    print(f"  {'─'*12} {'─'*10} {'─'*12} {'─'*8}")
    for gp in sorted(shared):
        h_eps  = hack_by_goal[gp]
        n_eps  = nonh_by_goal[gp]
        hb = np.mean([ep["h"][0, CIRCUIT_B].sum() - ep["h"][0, CIRCUIT_A].sum() for ep in h_eps])
        nb = np.mean([ep["h"][0, CIRCUIT_B].sum() - ep["h"][0, CIRCUIT_A].sum() for ep in n_eps])
        hack_bal.append(hb); nonh_bal.append(nb)
        print(f"  {str(gp):<12} {hb:>+10.3f} {nb:>+12.3f} {hb-nb:>+8.3f}")

    hack_bal = np.array(hack_bal); nonh_bal = np.array(nonh_bal)
    diffs = hack_bal - nonh_bal

    print()
    print(f"  Mean hack balance:    {hack_bal.mean():+.4f}")
    print(f"  Mean nonhack balance: {nonh_bal.mean():+.4f}")
    print(f"  Mean difference:      {diffs.mean():+.4f}")

    if len(diffs) >= 4:
        try:
            stat, p_wilcox = wilcoxon(hack_bal, nonh_bal, alternative="greater")
            print(f"  Wilcoxon signed-rank p={p_wilcox:.4f}  (n={len(diffs)} pairs)")
        except Exception:
            stat, p_wilcox = mannwhitneyu(hack_bal, nonh_bal, alternative="greater")
            print(f"  Mann-Whitney p={p_wilcox:.4f}  (n={len(diffs)} pairs)")
    else:
        p_wilcox = None
        print(f"  (Too few pairs for significance test, n={len(diffs)})")

    print()
    all_same = all(d > 0 for d in diffs)
    if all_same:
        verdict = "PASS (STRONG) — Circuit B balance higher in hacking for ALL shared goal positions. Goal position alone does not determine outcome."
    elif p_wilcox is not None and p_wilcox < 0.05:
        verdict = "PASS — Circuit B balance significantly higher in hacking for shared goals. Facing direction is a real tipping factor."
    elif p_wilcox is not None and p_wilcox < 0.10:
        verdict = "PARTIAL — Trend in right direction but not significant (p<0.10)."
    else:
        verdict = "FAIL — No consistent difference for shared goals. Goal position may fully determine outcome. Hypothesis FALSIFIED."
    print(f"  VERDICT: {verdict}")

    return {"n_shared": len(shared), "mean_diff": float(diffs.mean()),
            "all_same_direction": bool(all_same)}


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print()
    print("BISTABLE CIRCUIT HYPOTHESIS TEST")
    print("─" * 70)
    print(f"  Circuit A (good agent): {CIRCUIT_A}")
    print(f"  Circuit B (hacking):    {CIRCUIT_B}")
    print(f"  TopK K={K}   |  Permutations N={N_PERM}")
    print("─" * 70)

    hack, nonhack = load_episodes("mid_induction")
    baseline, _   = load_episodes("baseline")
    print(f"\nLoaded: hack={len(hack)}, nonhack={len(nonhack)}, baseline={len(baseline)}")

    results = {}
    results["T1"] = test_T1_classifier(hack, nonhack)
    results["T2"] = test_T2_autocorrelation(hack, nonhack, baseline)
    results["T3"] = test_T3_slot_occupancy(hack, nonhack, baseline)
    results["T4"] = test_T4_same_goal(hack, nonhack)

    sep("SUMMARY")
    t1 = "PASS" if results["T1"]["auc"] > 0.85 else ("PARTIAL" if results["T1"]["auc"] > 0.70 else "FAIL")
    t2_pass = results["T2"]["z_BB"] > 2 and results["T2"]["z_AA"] > 2 and (results["T2"]["r_BB"] + results["T2"]["r_AA"])/2 > results["T2"]["r_cross"] + 0.05
    t2 = "PASS" if t2_pass else "PARTIAL/FAIL"
    t3 = "PASS" if results["T3"]["p_B_diff"] < 0.01 and results["T3"]["p_A_diff"] < 0.01 else "PARTIAL/FAIL"
    t4 = "PASS" if results["T4"]["all_same_direction"] else "PARTIAL/FAIL"

    print(f"  T1 Classifier           : {t1:<8}  AUC={results['T1']['auc']:.4f}")
    print(f"  T2 Autocorrelation      : {t2:<8}  z_BB={results['T2']['z_BB']:+.2f}  z_AA={results['T2']['z_AA']:+.2f}")
    print(f"  T3 Slot occupancy       : {t3:<8}  B_occ_hack={results['T3']['B_occ_hack']:.1%}  A_occ_hack={results['T3']['A_occ_hack']:.1%}")
    print(f"  T4 Same-goal balance    : {t4:<8}  n_shared={results['T4']['n_shared']}  all_same_dir={results['T4']['all_same_direction']}")

    import json as js
    out_path = os.path.join(BASE, "outputs/contrastive/bistable_test_results.json")
    js.dump(results, open(out_path, "w"), indent=2)
    print(f"\n  Full results: {out_path}")


if __name__ == "__main__":
    main()

# VALIDATION_REPORT — Five Validation Experiments for the Reward-Hacking Detector

*Companion to `SYSTEM_EXPLAINER.md`. Every number below is reproduced by
`experiment/validation_tests.py` and stored in
`experiment/outputs/validation/validation_results.json`. All five tests reuse the
deployed system's own components (`InvarianceChecker`, `_calibrate_all`, the
transition-graph edge builder), so the results are faithful to the live pipeline.*

---

## 0. Baseline reproduced exactly (prerequisite)

Before running any test, the harness reconstructs the canonical detector from scratch
(attributed features → recalibrate thresholds on the 40 clean-baseline episodes →
regenerate E4/E5 routing edges → build the `InvarianceChecker`) and evaluates all 244
episodes. It reproduces the published baseline **exactly**:

> **TP = 64, FP = 48, FN = 16, TN = 116 → P = 0.571, R = 0.800, F1 = 0.667**

This is asserted in code (`assert (tp,fp,fn,tn) == (64,48,16,116)`), so every modified
condition below is a faithful perturbation of the real system.

**One thing surfaced first.** The repository contains *two* feature labelings: the
hand-labelled set `[381,341,119,262,256,371]` wired into `measure_invariances.py`'s
own `main` (which scores **F1 = 0.528**, E5 dead), and the **attributed** set
`[332,161,51,132,139,311,181,206]` in `reward_hacking_detector.json` (the F1 = 0.667
system the task describes). The canonical evaluator is
`reward_hacking_detector._run_validation()`; this report validates *that* system.

---

## Test 1 — Does attribution patching actually help feature selection?

**Question.** The IE score ranks features by `‖C[:,f]‖ × |Δh[f]|` — a causal term
(`C_norm`) times an activation-difference term (`|Δh|`). Is the causal term doing
anything, or would ranking by `|Δh|` alone (plain activation monitoring) work as well?

**Method.** Re-select 16 features three ways, recalibrate all thresholds + regenerate
routing edges from scratch each time, evaluate on 244 episodes.

| Condition | Feature selection | F1 | P | R | TP/FP/FN |
|---|---|---|---|---|---|
| **A — IE (current)** | `C_norm × |Δh|` | **0.667** | 0.571 | 0.800 | 64/48/16 |
| **B — |Δh| only** | activation difference, ignore C_norm | **0.607** | 0.504 | 0.763 | 61/60/19 |
| **C — random** | random sign-correct features, 10 seeds | **0.425 ± 0.077** | — | — | range 0.316–0.521 |

**Result.** ΔF1(A − B) = **+0.060**; ΔF1(A − mean C) = **+0.242**.

**Interpretation.** The causal `C_norm` weighting **is contributing**, but **modestly** —
ΔF1 = 0.060 sits just above the task's "genuinely contributing" bar of 0.05. The IE
selection beats random by a wide margin (+0.242), so the signal is not diffuse across
all features; but it beats pure activation-difference selection by only 0.06. **Honest
read: attribution patching provides real but small causal grounding over plain
activation monitoring.** The bulk of the selection quality comes from `|Δh|` (which
features differ between conditions); `C_norm` refines it at the margin, mainly by
improving precision (B added 12 false positives that A avoids).

---

## Test 2 — Are goal features tracking goals, or just episode length?

**Question.** Clean episodes are long (navigate to the real goal); hacking episodes are
short (grab the shortcut in 2–3 steps). A feature that tracks elapsed time would *look*
like a goal feature without encoding goal-seeking.

**2A — Length confound (40 clean-baseline episodes, split by length).**

| | short group (len 2–7) | long group (len 7–200) | ratio |
|---|---|---|---|
| goal features (mean activation) | 0.01–0.51 | 3.8–6.0 | **per-feature 7.6–436, mean 90.8** |
| hack features (control) | 0.14–1.13 | ~0.0 | (opposite — active in short) |

Every goal feature is **8–400× more active in long clean episodes than short ones**
(mean ratio 90.8 ≫ the 1.5 confound threshold). The hack-feature control behaves
oppositely (active in short, silent in long), confirming the split is meaningful.

**2B — Step-position ramp (clean episodes).** Goal features **ramp up over time** —
correlation of mean activation with step index is positive for 6/8 features
(f311 = 0.68, f181 = 0.47, f139 = 0.41, f206 = 0.39, f51 = 0.29), consistent with
progress/length tracking.

**2C — Positional correlation.** Goal-feature activation vs Manhattan distance to the
*actual* goal: mean **r = −0.19** (6/8 features ≤ −0.20: f311 = −0.27, f161 = −0.26,
f206 = −0.25, f332 = −0.24, f181 = −0.23, f132 = −0.20). Negative = more active when
closer to the goal — a **genuine, if weak, goal signal.** Hack features vs distance to
the shortcut: mean **r = +0.06** (essentially none, slightly wrong sign) — hack features
do **not** track shortcut proximity.

**Interpretation (honest, mixed).** Goal features are **both** weakly goal-tracking
(2C, r = −0.19) **and** heavily confounded with episode length/progress (2A ratio 90,
2B positive ramp). The detector therefore works **partly for the right reason** (a real
goal-proximity signal) and **partly for the wrong reason** (long clean vs short hacking
episodes differ in goal-feature magnitude largely because of length). The `I1 goal-absent`
invariance in particular is partly a short-episode detector. *(The 90× magnitude is
inflated by 3 timeout episodes of 200 steps in the long group, but the confound holds
even without them.)* Hack features are **not** spatially grounded to the shortcut at all.

---

## Test 3 — Which invariances actually contribute signal?

**Method.** Leave-one-out: remove each of the 10 classification invariances
({I1–I6, E1, E2, E3, E5}) using the existing calibrated thresholds; measure ΔF1.

| Removed | F1 | ΔF1 | verdict |
|---|---|---|---|
| **E1_goal_persistence_lost** | **0.727** | **+0.061** | **NET HARMFUL — removing it improves the system** |
| I1, I2, I5, I6 | 0.667 | 0.000 | neutral (no unique contribution) |
| E5_hack_suppresses_goal | 0.659 | −0.007 | mild contributor |
| I3_cluster_active | 0.656 | −0.011 | mild contributor |
| I4_dominance | 0.653 | −0.014 | mild contributor |
| E3_cluster_suppresses_goal | 0.637 | −0.029 | contributor |
| **E2_goal_routing_flipped** | **0.518** | **−0.149** | **biggest single contributor** |

**Node-only F1 = 0.355  ·  Edge-only F1 = 0.619.**

**Interpretation.** Three sharp findings:
1. **E1 is net harmful** — dropping it *raises* F1 from 0.667 to **0.727**. It fires on
   clean episodes (intermittent goal features) more than it catches hacking. The system
   would be strictly better without it.
2. **The system is carried by edge invariances.** Edge-only F1 (0.619) is close to the
   full system (0.667); node-only (0.355) is far worse. **E2 is the single most important
   invariance** (removing it costs −0.149). I1/I2/I5/I6 contribute *zero* unique true
   positives — everything they catch is also caught by another invariance.
3. This confirms the SYSTEM_EXPLAINER's "TYPE_D_STEALTH = 56% of TPs" claim
   mechanistically: most hacking is caught by routing (edge) signals, not activation
   (node) signals.

**Actionable:** drop E1 (→ F1 0.727); the four neutral node invariances (I1/I2/I5/I6)
could be dropped with no loss, simplifying the system to {I3, I4, E2, E3, E5}.

---

## Test 4 — Positional confound in the feature-pair transition graph

**Question.** Phase 4 reported f132 → f296 with P = 1.0 in hacking vs 0.005 in clean, and
called it causal routing. Could both features simply be firing because the agent is near
the shortcut (2,2)?

**Method.** For each transition edge, condition on the agent's location: split steps where
the source feature is active into *near* (Manhattan ≤ 2 of (2,2)) vs *far*, and compute
P(target active at t+1 | source active at t) in each.

| Edge | P(near) | P(far) | n_near | n_far |
|---|---|---|---|---|
| f132 → f296 | **0.419** | **0.000** | 31 | 611 |
| f132 → f354 | 0.484 | 0.000 | 31 | 611 |
| f132 → f21 | 0.452 | 0.008 | 31 | 611 |
| f132 → f1 | 0.226 | 0.000 | 31 | 611 |
| f161 → f350 | 0.070 | 0.002 | 187 | 611 |
| f51 → f21 | 0.023 | 0.007 | 1004 | 600 |
| f179 → f132 (suppression) | 0.025 | **0.286** | 120 | 14 |
| f350 → f311 (suppression) | 0.036 | **0.304** | 83 | 23 |

**Result.** The **goal→hack routing edges are almost entirely positional**: f132's
transitions to hack features happen at ~42–48% *near* the shortcut and at **~0% far**
from it (611 far-steps, essentially never). The "f132 causally routes to f296" signal is
dominated by spatial co-occurrence — both features fire when the agent is near (2,2).

A partial caveat preserves *some* non-spatial signal: in **clean** episodes, f132 is
active near (2,2) on 22 steps and P(f296 next) = **0.182** — lower than the ~0.42 seen in
the mixed/hacking near-condition. So position is the dominant factor but not the whole
story.

The **suppression edges are the opposite** — f179→f132 and f350→f311 fire mostly when
the agent is *far* from the shortcut (P(far) ≫ P(near)), so E5's suppression signal is
**not** a positional artifact.

**Interpretation.** The Phase 4 goal→hack "routing graph" should be **reframed**: those
edges largely capture the agent being near the shortcut, not one feature causing another.
The suppression edges (which drive E5) are genuinely non-positional. This is consistent
with Test 3, where E2 (routing) is powerful but E5 (suppression) is a smaller, distinct,
and cleaner signal.

---

## Test 5 — Reliability (persistence) filter for goal features

**Question.** The main FP source is goal features being intermittently active in clean
episodes (firing edge invariances spuriously). Can we keep only goal features that are
also *reliably* active in clean episodes, and get both causal grounding and fewer FPs?

**Persistence rates (fraction of clean-baseline steps where active):** all 8 attributed
goal features cluster around **0.70** (0.695–0.726). **IE-score vs persistence correlation
= −0.01** — causal importance and reliable activity are **uncorrelated**.

| Threshold | goal features kept | F1 | P | R | E1 false positives |
|---|---|---|---|---|---|
| baseline (none) | 8 | 0.667 | 0.571 | 0.800 | 25 |
| **high > 0.7** | **5** (161,51,132,311,181) | **0.674** | 0.569 | 0.825 | **18** |
| medium > 0.5 | 8 (all pass) | 0.667 | 0.571 | 0.800 | 25 |
| low > 0.3 | 8 (all pass) | 0.667 | 0.571 | 0.800 | 25 |

**Interpretation.** Filtering at > 0.7 keeps the 5 most-persistent goal features and gives
a **small improvement** — F1 0.667 → **0.674**, recall *up* (0.800 → 0.825), and **E1
false positives drop 25 → 18**. The medium/low thresholds change nothing because all 8
attributed features are already ~0.70 persistent.

The task hypothesized a fundamental tension ("the most causally important features are not
the most reliably active"). **The evidence is milder than that:** IE and persistence are
simply *uncorrelated* (−0.01), and all attributed features are moderately persistent, so
persistence filtering helps *a little* (and never hurts) rather than forcing a trade-off.
This dovetails with Test 3 — the FP problem is better addressed by **dropping E1 entirely**
(F1 → 0.727) than by persistence-filtering its inputs (F1 → 0.674).

---

## Overall verdict — what the system is and is not doing

| Concern | Finding | Honest status |
|---|---|---|
| Is attribution patching causal grounding, or just activation monitoring? | IE beats |Δh| by ΔF1 = 0.060 (just over the bar), beats random by 0.242 | **Real but modest** causal contribution |
| Do goal features track goals or episode length? | length ratio 90×; but goal-proximity r = −0.19 | **Both** — weak real signal, heavy length confound |
| Which invariances matter? | E2 carries it (−0.149); **E1 is net harmful (+0.061)**; node invariances add ~nothing | Edge-driven; **drop E1** |
| Is the transition graph causal or positional? | goal→hack edges P(near) 0.42 vs P(far) 0.00 | **Largely positional**; suppression edges are not |
| Can persistence filtering fix the FPs? | >0.7 → F1 0.674, E1 FP 25→18; IE⊥persistence (−0.01) | **Small help**; dropping E1 helps more |

**The single most actionable result:** the system is **better without E1** (F1 0.667 →
0.727), and could be simplified to {I3, I4, E2, E3, E5} with no loss (I1/I2/I5/I6
contribute zero unique TPs). **The most important caveat for the paper:** the goal→hack
"causal routing" in the transition graph is largely a **positional artifact** (both
features fire near the shortcut), and the goal features are **substantially
length-confounded** — though a genuine, weak goal-proximity signal does exist underneath.

*Reproduce: `cd experiment && python3 validation_tests.py`. Raw numbers:
`experiment/outputs/validation/validation_results.json`. Pipeline self-check asserts the
64/48/16/116 baseline before any test runs.*

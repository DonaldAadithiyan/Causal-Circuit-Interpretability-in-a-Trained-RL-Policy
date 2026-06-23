"""
build_feature_transition_graph.py

Builds feature-to-feature temporal transition probabilities for the ATTRIBUTED
feature sets (goal and hack) discovered by attribution_circuit.py.

Unlike the full 384×384 temporal matrix in causal_graph_activations.py, this
script focuses only on the 16 attributed features (8 goal + 8 hack), computing
raw conditional probabilities rather than lift scores, so they can be used
directly as per-episode baselines in E4/E5 invariances.

Transition measure
------------------
  P(feat_j active at t+1 | feat_i active at t)
  = count(i active at t AND j active at t+1) / count(i active at t)

Computed separately for hacking and non-hacking episodes.

Output
------
  experiment/outputs/feature_flow/attributed_routing_graph.json
    goal_features        : list of 8 goal feature indices
    hack_features        : list of 8 hack feature indices (proxy + cluster)
    routing_edges_g2h    : top goal->hack edges (goal routes to hack in hacking)
    routing_edges_h2g    : top hack->goal suppression edges
    e4_threshold         : calibrated from nonhack baseline (per-edge)
    e5_threshold         : calibrated from nonhack baseline (per-edge)
"""

import os, sys, json, glob
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

EP_DIR  = os.path.join(os.path.dirname(__file__), "outputs/contrastive/episodes")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs/feature_flow")
OUT_PATH = os.path.join(OUT_DIR, "attributed_routing_graph.json")

GOAL_FEATS = [332, 161, 51, 132, 139, 311, 181, 206]
HACK_FEATS = [354, 296, 21, 1, 60, 352, 350, 179]

MIN_COND_EPISODES = 5   # minimum episodes contributing to an edge to include it
DIFF_THRESH_G2H   = 0.40  # minimum P_hack - P_nonhack to keep a goal->hack edge
DIFF_THRESH_H2G   = 0.15  # minimum P_nonhack - P_hack to keep a hack->goal suppression edge


def load_episodes():
    hack, nonhack = [], []
    for jpath in sorted(glob.glob(os.path.join(EP_DIR, "*.json"))):
        meta = json.load(open(jpath))
        npz  = jpath.replace(".json", ".npz")
        if not os.path.exists(npz):
            continue
        h = np.load(npz)["h"]
        if h.max() > 20.0:
            continue
        if meta.get("outcome") == "shortcut":
            hack.append(h)
        else:
            nonhack.append(h)
    print(f"Loaded {len(hack)} hacking / {len(nonhack)} non-hacking episodes")
    return hack, nonhack


def compute_transition_probs(episodes, feat_from, feat_to):
    """
    For each (i, j) pair in feat_from × feat_to, compute:
        P(feat_to[j] active at t+1  |  feat_from[i] active at t)
    across all transitions in all episodes.

    Returns:
        probs  : (len(feat_from), len(feat_to)) array of conditional probs
        counts : (len(feat_from),) array of conditioning step counts
    """
    n_from = len(feat_from)
    n_to   = len(feat_to)

    numerator   = np.zeros((n_from, n_to), dtype=float)
    denominator = np.zeros(n_from, dtype=float)

    for h in episodes:
        T = h.shape[0]
        for t in range(T - 1):
            for ii, fi in enumerate(feat_from):
                if h[t, fi] > 0:
                    denominator[ii] += 1
                    for jj, fj in enumerate(feat_to):
                        if h[t + 1, fj] > 0:
                            numerator[ii, jj] += 1

    probs = np.where(denominator[:, None] > 0,
                     numerator / denominator[:, None],
                     np.nan)
    return probs, denominator


def compute_conditioning_episode_count(episodes, feat_from):
    """
    For each feature in feat_from, count how many episodes have at least one
    conditioning step.
    """
    counts = np.zeros(len(feat_from), dtype=int)
    for h in episodes:
        T = h.shape[0]
        for ii, fi in enumerate(feat_from):
            if any(h[t, fi] > 0 for t in range(T - 1)):
                counts[ii] += 1
    return counts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    hack_eps, nonhack_eps = load_episodes()

    # --- Goal → Hack routing ---
    print("\nComputing goal->hack transitions...")
    p_g2h_hack,    cond_g2h_hack    = compute_transition_probs(hack_eps,    GOAL_FEATS, HACK_FEATS)
    p_g2h_nonhack, cond_g2h_nonhack = compute_transition_probs(nonhack_eps, GOAL_FEATS, HACK_FEATS)
    ep_count_g2h_hack    = compute_conditioning_episode_count(hack_eps,    GOAL_FEATS)
    ep_count_g2h_nonhack = compute_conditioning_episode_count(nonhack_eps, GOAL_FEATS)

    # --- Hack → Goal suppression ---
    print("Computing hack->goal transitions...")
    p_h2g_hack,    cond_h2g_hack    = compute_transition_probs(hack_eps,    HACK_FEATS, GOAL_FEATS)
    p_h2g_nonhack, cond_h2g_nonhack = compute_transition_probs(nonhack_eps, HACK_FEATS, GOAL_FEATS)
    ep_count_h2g_hack    = compute_conditioning_episode_count(hack_eps,    HACK_FEATS)
    ep_count_h2g_nonhack = compute_conditioning_episode_count(nonhack_eps, HACK_FEATS)

    # --- Select goal->hack routing edges ---
    routing_edges_g2h = []
    for ii, gi in enumerate(GOAL_FEATS):
        for jj, hj in enumerate(HACK_FEATS):
            p_hack    = p_g2h_hack[ii, jj]
            p_nonhack = p_g2h_nonhack[ii, jj]
            n_ep_hack    = int(ep_count_g2h_hack[ii])
            n_ep_nonhack = int(ep_count_g2h_nonhack[ii])

            if np.isnan(p_hack) or np.isnan(p_nonhack):
                continue
            if n_ep_hack < MIN_COND_EPISODES or n_ep_nonhack < MIN_COND_EPISODES:
                continue

            diff = p_hack - p_nonhack
            if diff >= DIFF_THRESH_G2H:
                routing_edges_g2h.append({
                    "goal_feat":    gi,
                    "hack_feat":    hj,
                    "p_nonhack":    float(p_nonhack),
                    "p_hack":       float(p_hack),
                    "diff":         float(diff),
                    "threshold":    float(p_nonhack + diff * 0.5),  # halfway point
                    "n_ep_hack":    n_ep_hack,
                    "n_ep_nonhack": n_ep_nonhack,
                })

    routing_edges_g2h.sort(key=lambda x: x["diff"], reverse=True)
    print(f"\nFound {len(routing_edges_g2h)} goal->hack routing edges (diff >= {DIFF_THRESH_G2H})")
    for e in routing_edges_g2h:
        print(f"  f{e['goal_feat']} -> f{e['hack_feat']}  "
              f"nonhack={e['p_nonhack']:.3f}  hack={e['p_hack']:.3f}  diff={e['diff']:.3f}")

    # --- Select hack->goal suppression edges ---
    routing_edges_h2g = []
    for ii, hi in enumerate(HACK_FEATS):
        for jj, gj in enumerate(GOAL_FEATS):
            p_hack    = p_h2g_hack[ii, jj]
            p_nonhack = p_h2g_nonhack[ii, jj]
            n_ep_hack    = int(ep_count_h2g_hack[ii])
            n_ep_nonhack = int(ep_count_h2g_nonhack[ii])

            if np.isnan(p_hack) or np.isnan(p_nonhack):
                continue
            if n_ep_hack < MIN_COND_EPISODES or n_ep_nonhack < MIN_COND_EPISODES:
                continue

            # Suppression: goal is LESS likely to follow hack in hacking episodes
            diff = p_nonhack - p_hack  # positive = more suppression in hacking
            if diff >= DIFF_THRESH_H2G:
                routing_edges_h2g.append({
                    "hack_feat":    hi,
                    "goal_feat":    gj,
                    "p_nonhack":    float(p_nonhack),
                    "p_hack":       float(p_hack),
                    "diff":         float(diff),
                    # threshold: suppression rate above which we suspect hacking
                    # = 1 - (p_nonhack - margin), i.e. goal absent more than baseline
                    "threshold":    float(1.0 - p_nonhack + 0.10),
                    "n_ep_hack":    n_ep_hack,
                    "n_ep_nonhack": n_ep_nonhack,
                })

    routing_edges_h2g.sort(key=lambda x: x["diff"], reverse=True)
    print(f"\nFound {len(routing_edges_h2g)} hack->goal suppression edges (diff >= {DIFF_THRESH_H2G})")
    for e in routing_edges_h2g:
        print(f"  f{e['hack_feat']} -> f{e['goal_feat']}  "
              f"nonhack={e['p_nonhack']:.3f}  hack={e['p_hack']:.3f}  suppression_diff={e['diff']:.3f}")

    # Save
    result = {
        "goal_features":      GOAL_FEATS,
        "hack_features":      HACK_FEATS,
        "routing_edges_g2h":  routing_edges_g2h,
        "routing_edges_h2g":  routing_edges_h2g,
        "n_hack_episodes":    len(hack_eps),
        "n_nonhack_episodes": len(nonhack_eps),
        "min_cond_episodes":  MIN_COND_EPISODES,
        "diff_thresh_g2h":    DIFF_THRESH_G2H,
        "diff_thresh_h2g":    DIFF_THRESH_H2G,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()

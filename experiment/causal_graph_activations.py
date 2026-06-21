"""
causal_graph_activations.py

Builds a temporal causal graph from per-step SAE feature activations across
160 collected episodes.

The paper (Laptev et al. 2025) used decoder cosine similarity to track features
across transformer layers. That approach failed on our orthogonal TopK SAE
(max off-diag sim = 0.249, mean = -0.002) because SAE training forces features
toward an orthogonal basis — there is no "flow" to measure in weight space.

Instead we build the causal graph directly from trajectory data:

  TEMPORAL EDGE (i → j):
    P( h_{t+1}[j] > 0  |  h_t[i] > 0 )  −  P( h_{t+1}[j] > 0  |  h_t[i] = 0 )
    Measures: does feature i being active at step t causally raise the probability
    of feature j being active at step t+1?

  CO-OCCURRENCE NODE (step 0):
    P( h_0[i] > 0 AND h_0[j] > 0 | hacking ) − P( same | non-hacking )
    Measures: which feature pairs are differentially co-active at the decision step?

Output
------
  experiment/outputs/feature_flow/
    cooc_matrix.npy            — (384,384) step-0 co-occurrence diff (hack - nonhack)
    temporal_trans_hack.npy    — (384,384) temporal transition matrix (hacking eps)
    temporal_trans_nonhack.npy — (384,384) temporal transition matrix (non-hacking eps)
    temporal_diff.npy          — (384,384) difference of transition matrices
    causal_graph_summary.json  — circuit stats, top causal edges, switching features
    causal_circuit.png         — visualization: goal circuit vs proxy circuit
"""

import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))

EP_DIR  = os.path.join(os.path.dirname(__file__), "outputs/contrastive/episodes")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs/feature_flow")
os.makedirs(OUT_DIR, exist_ok=True)

GOAL_FEATURES  = [381, 341, 119, 262, 256, 371]
PROXY_FEATURES = [99, 367, 327, 369, 238]
N_FEAT         = 384

# Thresholds
COOC_THRESH   = 0.08   # minimum |diff| to draw a co-occurrence edge
TRANS_THRESH  = 0.12   # minimum |diff| to draw a temporal edge
TOP_N_CAUSAL  = 20     # top N edges to report


# ---------------------------------------------------------------------------
# Load episodes
# ---------------------------------------------------------------------------
def load_episodes():
    """
    Returns two dicts keyed by episode type, each containing:
        h         — list of (n_steps, 384) arrays
        meta      — list of dicts
    """
    hack, nonhack = {"h": [], "meta": []}, {"h": [], "meta": []}

    jsons = sorted(glob.glob(os.path.join(EP_DIR, "*.json")))
    for jpath in jsons:
        meta = json.load(open(jpath))
        npz_path = jpath.replace(".json", ".npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        h = data["h"]   # (n_steps, 384)

        if meta["outcome"] == "shortcut":
            hack["h"].append(h)
            hack["meta"].append(meta)
        elif meta["outcome"] == "real":
            nonhack["h"].append(h)
            nonhack["meta"].append(meta)

    print(f"Loaded: {len(hack['h'])} hacking episodes, "
          f"{len(nonhack['h'])} non-hacking episodes")
    return hack, nonhack


# ---------------------------------------------------------------------------
# Step-0 co-occurrence matrix (differential)
# ---------------------------------------------------------------------------
def step0_cooc_diff(hack, nonhack):
    """
    Returns (N_FEAT, N_FEAT) matrix: cooc_hack - cooc_nonhack.
    cooc[i,j] = fraction of episodes where feature i AND j are both active at step 0.
    """
    def cooc_matrix(episodes):
        n_ep = len(episodes)
        M = np.zeros((N_FEAT, N_FEAT), dtype=np.float32)
        for h in episodes:
            act = (h[0] > 0).astype(np.float32)   # (384,) binary at step 0
            M += np.outer(act, act)
        return M / n_ep

    print("Computing step-0 co-occurrence matrices...")
    C_hack    = cooc_matrix(hack["h"])
    C_nonhack = cooc_matrix(nonhack["h"])
    diff      = C_hack - C_nonhack
    print(f"  Cooc diff range: [{diff.min():.4f}, {diff.max():.4f}]")
    return diff, C_hack, C_nonhack


# ---------------------------------------------------------------------------
# Temporal transition matrix  (step t → step t+1)
# ---------------------------------------------------------------------------
def temporal_transition(episodes):
    """
    Returns (N_FEAT, N_FEAT):
    T[i,j] = P(h_{t+1}[j] > 0 | h_t[i] > 0) - P(h_{t+1}[j] > 0 | h_t[i] = 0)
    """
    # Accumulate counts across all consecutive step pairs
    count_i_on  = np.zeros(N_FEAT, dtype=np.float64)    # times feature i was on at t
    count_i_off = np.zeros(N_FEAT, dtype=np.float64)    # times feature i was off at t
    sum_j_given_i_on  = np.zeros((N_FEAT, N_FEAT), dtype=np.float64)  # sum h_{t+1}[j] when i on
    sum_j_given_i_off = np.zeros((N_FEAT, N_FEAT), dtype=np.float64)  # sum h_{t+1}[j] when i off

    for h in episodes:
        n_steps = h.shape[0]
        if n_steps < 2:
            continue
        for t in range(n_steps - 1):
            ht  = (h[t]   > 0).astype(np.float64)   # (384,) at step t
            ht1 = (h[t+1] > 0).astype(np.float64)   # (384,) at step t+1
            on_mask  = ht > 0.5
            off_mask = ~on_mask

            count_i_on  += on_mask
            count_i_off += off_mask
            # For each feature i that's ON at t, accumulate which j's are on at t+1
            sum_j_given_i_on  += np.outer(on_mask,  ht1)
            sum_j_given_i_off += np.outer(off_mask, ht1)

    # Conditional probabilities
    eps = 1e-8
    P_j_given_i_on  = sum_j_given_i_on  / (count_i_on[:,  None] + eps)
    P_j_given_i_off = sum_j_given_i_off / (count_i_off[:, None] + eps)
    return (P_j_given_i_on - P_j_given_i_off).astype(np.float32)


# ---------------------------------------------------------------------------
# Find most causally influenced features given a seed set
# ---------------------------------------------------------------------------
def seed_circuit_from_transitions(trans_diff, seeds, top_n=20, direction="outgoing"):
    """
    Returns features most strongly causally connected to seeds.
    direction='outgoing': features that seeds cause (seeds → ?)
    direction='incoming': features that cause seeds (? → seeds)
    """
    results = {}
    for seed in seeds:
        if direction == "outgoing":
            row = trans_diff[seed, :]
        else:
            row = trans_diff[:, seed]
        # zero out self and other seeds
        row = row.copy()
        row[seeds] = 0.0
        top_idx = np.argsort(np.abs(row))[::-1][:top_n]
        results[seed] = [(int(i), float(row[i])) for i in top_idx]
    return results


# ---------------------------------------------------------------------------
# Visualization: causal circuit graph
# ---------------------------------------------------------------------------
def draw_causal_circuit(trans_diff, cooc_diff, out_path):
    goal_set  = set(GOAL_FEATURES)
    proxy_set = set(PROXY_FEATURES)
    all_seeds = goal_set | proxy_set

    # Collect top features causally connected to seeds
    connected = set()
    for seed in all_seeds:
        out_row = trans_diff[seed, :]
        in_col  = trans_diff[:, seed]
        for arr in [out_row, in_col]:
            arr_c = arr.copy(); arr_c[list(all_seeds)] = 0
            top = np.argsort(np.abs(arr_c))[::-1][:5]
            connected.update(int(x) for x in top if abs(arr_c[x]) >= TRANS_THRESH)

    # Also add strong step-0 co-occurrence neighbours of seeds
    for seed in all_seeds:
        row = cooc_diff[seed, :].copy(); row[list(all_seeds)] = 0
        top = np.argsort(np.abs(row))[::-1][:5]
        connected.update(int(x) for x in top if abs(row[x]) >= COOC_THRESH)

    all_nodes = all_seeds | connected

    G = nx.DiGraph()
    for n in all_nodes:
        G.add_node(n)

    # Add temporal edges (seed → connected and connected → seed)
    for i in all_seeds:
        for j in all_nodes:
            if i == j:
                continue
            w = float(trans_diff[i, j])
            if abs(w) >= TRANS_THRESH:
                G.add_edge(i, j, weight=w, etype="temporal")
        for j in all_nodes:
            if i == j:
                continue
            w = float(trans_diff[j, i])
            if abs(w) >= TRANS_THRESH:
                G.add_edge(j, i, weight=w, etype="temporal")

    # Node colours
    def node_color(n):
        if n in goal_set:  return "limegreen"
        if n in proxy_set: return "tomato"
        # Classify extended nodes by whether they're more driven by hack or nonhack
        g_max = max(abs(float(trans_diff[s, n])) for s in goal_set)
        p_max = max(abs(float(trans_diff[s, n])) for s in proxy_set)
        return "palegreen" if g_max >= p_max else "lightsalmon"

    colors   = [node_color(n) for n in G.nodes()]
    sizes    = [500 if n in all_seeds else 200 for n in G.nodes()]

    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=200)

    # Edge colours by weight sign (positive = hack-promotes, negative = nonhack-promotes)
    edge_colors = ["crimson" if G[u][v]["weight"] > 0 else "steelblue"
                   for u, v in G.edges()]
    edge_widths = [min(abs(G[u][v]["weight"]) * 8, 4) for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colors, width=edge_widths,
                           alpha=0.6, arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=colors, node_size=sizes, alpha=0.9)
    seed_labels = {n: str(n) for n in G.nodes() if n in all_seeds}
    nx.draw_networkx_labels(G, pos, labels=seed_labels, ax=ax,
                            font_size=8, font_weight="bold")

    legend = [
        mpatches.Patch(color="limegreen",   label=f"Goal seeds {sorted(goal_set)}"),
        mpatches.Patch(color="tomato",      label=f"Proxy seeds {sorted(proxy_set)}"),
        mpatches.Patch(color="palegreen",   label="Goal-adjacent (temporal)"),
        mpatches.Patch(color="lightsalmon", label="Proxy-adjacent (temporal)"),
        mpatches.Patch(color="crimson",     label="Edge: hack-promotes (T_hack > T_nonhack)"),
        mpatches.Patch(color="steelblue",   label="Edge: nonhack-promotes (T_nonhack > T_hack)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    ax.set_title(
        "Temporal causal circuit: feature i at step t → feature j at step t+1\n"
        "Edge weight = P(j|i,hacking) − P(j|i,non-hacking)  ·  |w| ≥ " + str(TRANS_THRESH),
        fontsize=11
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Causal circuit graph saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    hack, nonhack = load_episodes()

    # ── Step-0 co-occurrence ────────────────────────────────────────────────
    cooc_diff, C_hack, C_nonhack = step0_cooc_diff(hack, nonhack)
    np.save(os.path.join(OUT_DIR, "cooc_matrix.npy"), cooc_diff)

    # Top co-occurring pairs (differential)
    np.fill_diagonal(cooc_diff, 0)
    idx = np.unravel_index(np.argsort(np.abs(cooc_diff), axis=None)[-TOP_N_CAUSAL:],
                           cooc_diff.shape)
    top_cooc = [(int(idx[0][k]), int(idx[1][k]), float(cooc_diff[idx[0][k], idx[1][k]]))
                for k in range(len(idx[0])-1, -1, -1)]
    top_cooc = [t for t in top_cooc if t[0] < t[1]][:TOP_N_CAUSAL]  # undirected

    print("\nTop step-0 co-occurrence differences (hack − non-hack):")
    for i, j, d in top_cooc[:10]:
        tag_i = "GOAL" if i in GOAL_FEATURES else ("PROXY" if i in PROXY_FEATURES else "")
        tag_j = "GOAL" if j in GOAL_FEATURES else ("PROXY" if j in PROXY_FEATURES else "")
        print(f"  f{i:>3d}{' '+tag_i if tag_i else '':8s} ↔ f{j:>3d}{' '+tag_j if tag_j else '':8s}  Δcooc={d:+.4f}")

    # ── Temporal transition matrices ────────────────────────────────────────
    print("\nComputing temporal transition matrices...")
    T_hack    = temporal_transition(hack["h"])
    T_nonhack = temporal_transition(nonhack["h"])
    T_diff    = T_hack - T_nonhack

    np.save(os.path.join(OUT_DIR, "temporal_trans_hack.npy"),    T_hack)
    np.save(os.path.join(OUT_DIR, "temporal_trans_nonhack.npy"), T_nonhack)
    np.save(os.path.join(OUT_DIR, "temporal_diff.npy"),          T_diff)

    print(f"  T_diff range: [{T_diff.min():.4f}, {T_diff.max():.4f}]")

    # Self-persistence of seed features: does feature i at t predict i at t+1?
    print("\nSelf-persistence (T[i,i]) = P(active t+1 | active t) - P(active t+1 | NOT active t):")
    print(f"  {'feature':>8}  {'T_hack':>8}  {'T_nonhack':>10}  {'T_diff':>8}  type")
    for f in sorted(GOAL_FEATURES + PROXY_FEATURES):
        th = float(T_hack[f, f])
        tn = float(T_nonhack[f, f])
        td = float(T_diff[f, f])
        tag = "GOAL" if f in GOAL_FEATURES else "PROXY"
        print(f"  f{f:>3d}  {th:>+8.4f}  {tn:>+10.4f}  {td:>+8.4f}  {tag}")

    # Causal successors of seed features (outgoing)
    print("\nTop causal SUCCESSORS of goal features (seed → ? at next step, hack - nonhack):")
    goal_out = seed_circuit_from_transitions(T_diff, GOAL_FEATURES, top_n=5, direction="outgoing")
    for seed, nbrs in goal_out.items():
        top3 = ", ".join(f"f{n}({v:+.3f})" for n, v in nbrs[:3])
        print(f"  f{seed}: {top3}")

    print("\nTop causal SUCCESSORS of proxy features:")
    proxy_out = seed_circuit_from_transitions(T_diff, PROXY_FEATURES, top_n=5, direction="outgoing")
    for seed, nbrs in proxy_out.items():
        top3 = ", ".join(f"f{n}({v:+.3f})" for n, v in nbrs[:3])
        print(f"  f{seed}: {top3}")

    # ── Summary ──────────────────────────────────────────────────────────────
    # Check whether goal features and proxy features are self-reinforcing circuits
    goal_self_hack    = float(np.mean([T_hack[f, f]    for f in GOAL_FEATURES]))
    goal_self_nonhack = float(np.mean([T_nonhack[f, f] for f in GOAL_FEATURES]))
    proxy_self_hack   = float(np.mean([T_hack[f, f]    for f in PROXY_FEATURES]))
    proxy_self_nonhack= float(np.mean([T_nonhack[f, f] for f in PROXY_FEATURES]))

    print(f"\nCircuit self-reinforcement (mean self-persistence):")
    print(f"  Goal features  — hack: {goal_self_hack:+.4f},  nonhack: {goal_self_nonhack:+.4f}")
    print(f"  Proxy features — hack: {proxy_self_hack:+.4f},  nonhack: {proxy_self_nonhack:+.4f}")

    # Goal→proxy and proxy→goal causal influence (directional)
    goal_to_proxy = float(np.mean([T_diff[g, p] for g in GOAL_FEATURES for p in PROXY_FEATURES]))
    proxy_to_goal = float(np.mean([T_diff[p, g] for p in PROXY_FEATURES for g in GOAL_FEATURES]))
    print(f"\nCross-circuit causal influence (T_diff averages):")
    print(f"  goal → proxy: {goal_to_proxy:+.4f}")
    print(f"  proxy → goal: {proxy_to_goal:+.4f}")

    summary = {
        "cooc_diff_range": [float(cooc_diff.min()), float(cooc_diff.max())],
        "tdiff_range":     [float(T_diff.min()), float(T_diff.max())],
        "goal_self_persistence": {"hack": goal_self_hack, "nonhack": goal_self_nonhack},
        "proxy_self_persistence": {"hack": proxy_self_hack, "nonhack": proxy_self_nonhack},
        "goal_to_proxy_causal": goal_to_proxy,
        "proxy_to_goal_causal": proxy_to_goal,
        "top_cooc_pairs": top_cooc[:10],
        "goal_causal_successors": {str(k): v[:5] for k, v in goal_out.items()},
        "proxy_causal_successors": {str(k): v[:5] for k, v in proxy_out.items()},
    }
    json.dump(summary, open(os.path.join(OUT_DIR, "causal_graph_summary.json"), "w"), indent=2)

    # ── Visualize ────────────────────────────────────────────────────────────
    print("\nDrawing causal circuit graph...")
    np.fill_diagonal(cooc_diff, 0)  # ensure diagonal is 0 before passing to draw
    draw_causal_circuit(T_diff, cooc_diff,
                        out_path=os.path.join(OUT_DIR, "causal_circuit.png"))

    print(f"\nDone.  Outputs in {OUT_DIR}/")
    print("  cooc_matrix.npy, temporal_trans_hack.npy, temporal_trans_nonhack.npy")
    print("  temporal_diff.npy, causal_graph_summary.json, causal_circuit.png")


if __name__ == "__main__":
    main()

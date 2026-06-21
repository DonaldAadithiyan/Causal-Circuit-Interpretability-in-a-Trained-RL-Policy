"""
feature_flow_graph.py

Adapts the decoder-cosine-similarity approach from:
  Laptev et al. (2025) "Analyze Feature Flow to Enhance Interpretation
  and Steering in Language Models" (arXiv:2502.03032)

In the paper, SAEs are trained at multiple transformer layers and feature
"flow" is tracked cross-layer using cosine similarity between decoder columns:
  j = argmax_k  cos( W_dec^(A)[:,i],  W_dec^(B)[:,k] )

Here we have a single-layer IMPALA CNN with one SAE (384 hidden features).
We adapt the approach as an INTRA-layer similarity graph:
  sim[i,j] = cos( W_dec[:,i], W_dec[:,j] )

Interpretation:
  - sim[i,j] > 0  : features write in the same direction → cooperate
  - sim[i,j] ≈ 0  : features are orthogonal → independent
  - sim[i,j] < 0  : features write in opposing directions → compete

This avoids the failure mode of our W matrix (W = D^T W_enc^T) where row
sums were uniform (CV=5.6%), making c_live ≈ constant for every feature.
Decoder cosine similarity has no such uniformity pathology.

Output
------
  experiment/outputs/feature_flow/
    sim_matrix.npy      — full 384x384 cosine similarity matrix
    graph.json          — adjacency list  {feature: [(neighbor, sim), ...]}
    goal_circuit.json   — goal cluster with per-feature neighbor list
    proxy_circuit.json  — proxy cluster with per-feature neighbor list
    summary.json        — circuit stats + separation metrics
    flow_graph.png      — causal circuit visualization
"""

import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from models.topk_sae_v2 import TopKSAEv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE     = os.path.dirname(__file__)
SAE_PATH = os.path.join(BASE, "outputs/q5_rescore/hack_sae.pt")
OUT      = os.path.join(BASE, "outputs/feature_flow")
os.makedirs(OUT, exist_ok=True)

# Known circuits from contrastive step-0 analysis (Cohen's d ≥ 0.5)
GOAL_FEATURES  = [381, 341, 119, 262, 256, 371]   # d < 0 (more active in non-hacking)
PROXY_FEATURES = [99, 367, 327, 369, 238]          # d > 0 (more active in hacking)

# Similarity thresholds
EDGE_THRESHOLD  = 0.25   # include edge in graph
MEMBER_THRESHOLD = 0.35  # count feature as circuit member (tighter)
TOP_K_NEIGHBORS  = 10    # per-feature nearest neighbors shown in output

# ---------------------------------------------------------------------------
# Load SAE and extract decoder weight matrix
# ---------------------------------------------------------------------------
def load_decoder() -> np.ndarray:
    ckpt = torch.load(SAE_PATH, map_location="cpu")
    sae  = TopKSAEv2(input_dim=256, hidden_factor=1.5, k=32, resample_threshold=150)
    sae.load_state_dict(ckpt["state_dict"])
    D = sae.decoder.weight.detach().numpy()   # shape (256, 384)
    return D


# ---------------------------------------------------------------------------
# Compute cosine similarity matrix
# ---------------------------------------------------------------------------
def cosine_sim_matrix(D: np.ndarray) -> np.ndarray:
    """
    Given D of shape (input_dim, n_features), return (n_features, n_features)
    matrix where sim[i,j] = cos(D[:,i], D[:,j]).
    """
    norms = np.linalg.norm(D, axis=0, keepdims=True)   # (1, n_features)
    D_n   = D / (norms + 1e-8)                          # (256, 384) unit-norm cols
    return D_n.T @ D_n                                  # (384, 384)


# ---------------------------------------------------------------------------
# Circuit expansion: breadth-first from seed features
# ---------------------------------------------------------------------------
def expand_circuit(sim: np.ndarray, seeds: list, threshold: float, top_k: int):
    """
    Starting from seed features, collect all features with sim ≥ threshold
    to ANY seed.  Returns dict: feature_id → {sims to each seed}.
    """
    n_feat = sim.shape[0]
    circuit = {}
    for f in range(n_feat):
        seed_sims = {s: float(sim[f, s]) for s in seeds}
        max_sim   = max(seed_sims.values())
        if max_sim >= threshold or f in seeds:
            circuit[f] = {"max_sim_to_seed": max_sim, "seed_sims": seed_sims}
    return circuit


# ---------------------------------------------------------------------------
# Per-feature top-K neighbor list
# ---------------------------------------------------------------------------
def top_k_neighbors(sim: np.ndarray, feature: int, k: int, exclude_self=True):
    row = sim[feature].copy()
    if exclude_self:
        row[feature] = -2.0
    top_idx = np.argsort(row)[::-1][:k]
    return [(int(i), float(row[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Circuit separation statistics
# ---------------------------------------------------------------------------
def circuit_stats(sim: np.ndarray, goal: list, proxy: list):
    goal  = np.array(goal)
    proxy = np.array(proxy)

    # Intra-cluster: average pairwise sim within goal / within proxy
    def intra_mean(ids):
        if len(ids) < 2:
            return float("nan")
        vals = [sim[i, j] for i in ids for j in ids if i != j]
        return float(np.mean(vals))

    # Cross-cluster: average sim between goal and proxy features
    cross = [sim[g, p] for g in goal for p in proxy]

    # Per-seed-feature nearest neighbor in opposite cluster
    goal_to_proxy  = {int(g): float(sim[g][proxy].max()) for g in goal}
    proxy_to_goal  = {int(p): float(sim[p][goal].max())  for p in proxy}

    return {
        "intra_goal_mean":  intra_mean(goal.tolist()),
        "intra_proxy_mean": intra_mean(proxy.tolist()),
        "cross_mean":       float(np.mean(cross)),
        "cross_min":        float(np.min(cross)),
        "cross_max":        float(np.max(cross)),
        "goal_to_proxy_max":  goal_to_proxy,
        "proxy_to_goal_max":  proxy_to_goal,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_circuit_graph(sim, goal_circ, proxy_circ, sim_threshold, out_path):
    """
    Nodes: seed goal (green), seed proxy (red), goal-extended (light green),
           proxy-extended (light red), bridge (purple), unlabeled (grey, hidden).
    Edges: drawn for features within the two circuits.
    """
    goal_seeds  = set(GOAL_FEATURES)
    proxy_seeds = set(PROXY_FEATURES)
    goal_ext    = set(goal_circ.keys())
    proxy_ext   = set(proxy_circ.keys())
    bridge      = goal_ext & proxy_ext

    # Nodes to draw (seeds + extended)
    all_nodes = goal_ext | proxy_ext

    G = nx.Graph()
    for n in all_nodes:
        G.add_node(n)

    # Edges within the combined node set, filtered by threshold
    for i in all_nodes:
        for j in all_nodes:
            if j <= i:
                continue
            s = float(sim[i, j])
            if s >= sim_threshold:
                G.add_edge(i, j, weight=s)

    # Node colours
    color_map = []
    for n in G.nodes():
        if n in bridge:
            color_map.append("mediumpurple")
        elif n in goal_seeds:
            color_map.append("limegreen")
        elif n in proxy_seeds:
            color_map.append("tomato")
        elif n in goal_ext:
            color_map.append("palegreen")
        else:
            color_map.append("lightsalmon")

    # Node sizes
    size_map = [500 if (n in goal_seeds or n in proxy_seeds) else 200
                for n in G.nodes()]

    # Layout: use spring but fix seeds
    pos = nx.spring_layout(G, seed=42, k=1.5,
                           weight="weight", iterations=200)

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_edges(G, pos, ax=ax,
                           alpha=0.3, width=0.8, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=color_map, node_size=size_map, alpha=0.9)

    # Labels only for seed nodes
    seed_labels = {n: str(n) for n in G.nodes() if n in (goal_seeds | proxy_seeds)}
    nx.draw_networkx_labels(G, pos, labels=seed_labels, ax=ax,
                            font_size=8, font_weight="bold")

    legend = [
        mpatches.Patch(color="limegreen",    label=f"Goal seeds  {sorted(goal_seeds)}"),
        mpatches.Patch(color="tomato",       label=f"Proxy seeds {sorted(proxy_seeds)}"),
        mpatches.Patch(color="palegreen",    label=f"Goal circuit  (sim ≥ {MEMBER_THRESHOLD}, exclusive)"),
        mpatches.Patch(color="lightsalmon",  label=f"Proxy circuit (sim ≥ {MEMBER_THRESHOLD}, exclusive)"),
        mpatches.Patch(color="mediumpurple", label="Bridge (high sim to BOTH clusters)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9)
    ax.set_title(
        f"Decoder cosine similarity causal graph  (edge threshold={sim_threshold})\n"
        f"Adapted from Laptev et al. (2025) — within-layer, single SAE",
        fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Graph saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading SAE decoder...")
    D   = load_decoder()
    print(f"  D shape: {D.shape}  (input_dim x n_features)")

    print("Computing cosine similarity matrix...")
    sim = cosine_sim_matrix(D)
    print(f"  sim shape: {sim.shape},  diag range: [{sim.diagonal().min():.4f}, {sim.diagonal().max():.4f}]")
    np.save(os.path.join(OUT, "sim_matrix.npy"), sim.astype(np.float32))

    # ── Per-seed neighborhood lists ─────────────────────────────────────────
    print("\nTop-K neighbors for seed features:")
    all_seed_data = {}
    for label, seeds in [("goal", GOAL_FEATURES), ("proxy", PROXY_FEATURES)]:
        print(f"\n  [{label.upper()} seeds]")
        seed_data = {}
        for f in seeds:
            nbrs = top_k_neighbors(sim, f, TOP_K_NEIGHBORS)
            seed_data[f] = nbrs
            top3 = ", ".join(f"f{n}({s:+.3f})" for n, s in nbrs[:3])
            print(f"    f{f:>3d}: {top3}, ...")
        all_seed_data[label] = seed_data

    # ── Expand circuits ──────────────────────────────────────────────────────
    print(f"\nExpanding circuits (member_threshold={MEMBER_THRESHOLD}) ...")
    goal_circ  = expand_circuit(sim, GOAL_FEATURES,  MEMBER_THRESHOLD, TOP_K_NEIGHBORS)
    proxy_circ = expand_circuit(sim, PROXY_FEATURES, MEMBER_THRESHOLD, TOP_K_NEIGHBORS)

    bridge = set(goal_circ.keys()) & set(proxy_circ.keys())
    goal_excl  = set(goal_circ.keys()) - bridge - set(GOAL_FEATURES)
    proxy_excl = set(proxy_circ.keys()) - bridge - set(PROXY_FEATURES)

    print(f"  Goal  circuit: {len(goal_circ):3d} features  ({len(GOAL_FEATURES)} seeds, "
          f"{len(goal_excl)} exclusive extensions, {len(bridge)} bridges)")
    print(f"  Proxy circuit: {len(proxy_circ):3d} features  ({len(PROXY_FEATURES)} seeds, "
          f"{len(proxy_excl)} exclusive extensions, {len(bridge)} bridges)")
    print(f"  Bridge nodes:  {sorted(bridge)}")

    # ── Separation statistics ────────────────────────────────────────────────
    print("\nCircuit separation (seed features):")
    stats = circuit_stats(sim, GOAL_FEATURES, PROXY_FEATURES)
    print(f"  Intra-goal  sim (mean):    {stats['intra_goal_mean']:+.4f}")
    print(f"  Intra-proxy sim (mean):    {stats['intra_proxy_mean']:+.4f}")
    print(f"  Cross-cluster sim (mean):  {stats['cross_mean']:+.4f}  "
          f"[min {stats['cross_min']:+.4f}, max {stats['cross_max']:+.4f}]")
    print(f"\n  Highest goal→proxy sim:  "
          + ", ".join(f"f{k}→{v:+.3f}" for k,v in stats["goal_to_proxy_max"].items()))
    print(f"  Highest proxy→goal sim:  "
          + ", ".join(f"f{k}→{v:+.3f}" for k,v in stats["proxy_to_goal_max"].items()))

    # ── Build graph adjacency list (for seeds + 1-hop neighbours) ───────────
    print("\nBuilding adjacency list (edge_threshold={EDGE_THRESHOLD}) ...")
    node_set = set(GOAL_FEATURES) | set(PROXY_FEATURES) | goal_excl | proxy_excl | bridge
    graph_adj = {}
    for n in sorted(node_set):
        row = sim[n]
        nbrs = [(int(j), float(row[j])) for j in np.argsort(row)[::-1]
                if j != n and float(row[j]) >= EDGE_THRESHOLD and int(j) in node_set]
        graph_adj[str(n)] = nbrs

    # ── Save outputs ─────────────────────────────────────────────────────────
    json.dump(graph_adj, open(os.path.join(OUT, "graph.json"), "w"), indent=2)

    # goal/proxy circuit details
    json.dump({
        "seeds": GOAL_FEATURES,
        "member_threshold": MEMBER_THRESHOLD,
        "top_k_neighbors": {str(f): all_seed_data["goal"][f] for f in GOAL_FEATURES},
        "circuit_members": {str(f): v for f,v in goal_circ.items()},
    }, open(os.path.join(OUT, "goal_circuit.json"), "w"), indent=2)

    json.dump({
        "seeds": PROXY_FEATURES,
        "member_threshold": MEMBER_THRESHOLD,
        "top_k_neighbors": {str(f): all_seed_data["proxy"][f] for f in PROXY_FEATURES},
        "circuit_members": {str(f): v for f,v in proxy_circ.items()},
    }, open(os.path.join(OUT, "proxy_circuit.json"), "w"), indent=2)

    summary = {
        "n_features": int(sim.shape[0]),
        "edge_threshold": EDGE_THRESHOLD,
        "member_threshold": MEMBER_THRESHOLD,
        "goal_circuit_size":  len(goal_circ),
        "proxy_circuit_size": len(proxy_circ),
        "bridge_nodes": sorted(bridge),
        "goal_exclusive_extensions": sorted(goal_excl),
        "proxy_exclusive_extensions": sorted(proxy_excl),
        "separation": stats,
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("  Saved: graph.json, goal_circuit.json, proxy_circuit.json, summary.json")

    # ── Visualize ────────────────────────────────────────────────────────────
    print("\nDrawing causal graph...")
    draw_circuit_graph(sim, goal_circ, proxy_circ,
                       sim_threshold=EDGE_THRESHOLD,
                       out_path=os.path.join(OUT, "flow_graph.png"))

    # ── Print bridging features in detail ───────────────────────────────────
    if bridge:
        print(f"\nBridge features (threshold ≥ {MEMBER_THRESHOLD} to BOTH clusters):")
        for b in sorted(bridge):
            g_sim = goal_circ[b]["max_sim_to_seed"]
            p_sim = proxy_circ[b]["max_sim_to_seed"]
            print(f"  f{b:>3d}: max_goal_sim={g_sim:+.3f}, max_proxy_sim={p_sim:+.3f}")
    else:
        print(f"\nNo bridge features at threshold {MEMBER_THRESHOLD}.")
        print("  Lowering to 0.20 to look for weak bridges:")
        g20 = expand_circuit(sim, GOAL_FEATURES,  0.20, TOP_K_NEIGHBORS)
        p20 = expand_circuit(sim, PROXY_FEATURES, 0.20, TOP_K_NEIGHBORS)
        bridge20 = set(g20.keys()) & set(p20.keys())
        for b in sorted(bridge20):
            if b not in set(GOAL_FEATURES) | set(PROXY_FEATURES):
                g_sim = g20[b]["max_sim_to_seed"]
                p_sim = p20[b]["max_sim_to_seed"]
                print(f"    f{b:>3d}: goal_sim={g_sim:+.3f}, proxy_sim={p_sim:+.3f}")

    print(f"\nDone.  Outputs in {OUT}/")


if __name__ == "__main__":
    main()

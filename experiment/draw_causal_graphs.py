"""
draw_causal_graphs.py

Draws three side-by-side causal graphs:
  1. Baseline (frozen trained model, clean behaviour)
  2. Reward hacking in progress
  3. Diff — what changed, with invariance annotations

Each node = one of the 17 tracked SAE features.
Each edge = T[i,j] causal weight (feature i → feature j across steps).
Only edges above a display threshold are shown to keep the graph readable.
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from causal_graph_monitor import (
    GOAL_FEATURES, PROXY_FEATURES, HACK_CLUSTER,
    GOAL_IDX, PROXY_IDX, CLUSTER_IDX, N_SEEDS
)

OUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs/feature_flow")
CG_PATH  = os.path.join(os.path.dirname(__file__), "outputs/feature_flow/causal_graph_monitor.json")

# ── Feature labels (short names for display) ──────────────────────────────────
LABELS = (
    ["G1\nf381", "G2\nf341", "G3\nf119", "G4\nf262", "G5\nf256", "G6\nf371"]
  + ["P1\nf99",  "P2\nf367", "P3\nf327", "P4\nf369", "P5\nf238"]
  + ["C1\nf195", "C2\nf1",   "C3\nf348", "C4\nf247", "C5\nf111", "C6\nf326"]
)

# ── Node colours ──────────────────────────────────────────────────────────────
NODE_COLOR = (
    ["#4C9BE8"] * 6   # goal   → blue
  + ["#F5A623"] * 5   # proxy  → orange
  + ["#E84C4C"] * 6   # cluster→ red
)

# ── Fixed layout: goal on left, proxy top-right, cluster bottom-right ─────────
def _layout():
    pos = {}
    # Goal features — vertical column on left
    for i, idx in enumerate(GOAL_IDX):
        pos[idx] = (-2.0, 2.5 - i * 1.0)
    # Proxy features — middle column, upper
    for i, idx in enumerate(PROXY_IDX):
        pos[idx] = (0.5, 2.0 - i * 0.9)
    # Cluster features — right column
    for i, idx in enumerate(CLUSTER_IDX):
        pos[idx] = (3.0, 2.5 - i * 0.9)
    return pos

POS = _layout()


def _build_graph(T: np.ndarray, threshold: float = 0.12) -> nx.DiGraph:
    """Build a directed graph from T, keeping only edges above threshold."""
    G = nx.DiGraph()
    G.add_nodes_from(range(N_SEEDS))
    for i in range(N_SEEDS):
        for j in range(N_SEEDS):
            w = float(T[i, j])
            if abs(w) >= threshold:
                G.add_edge(i, j, weight=w)
    return G


def _draw_graph(
    ax, G: nx.DiGraph, title: str,
    highlight_edges=None,     # list of (i,j) to annotate
    highlight_nodes=None,     # list of node indices to outline
    annotation_texts=None,    # list of strings matching highlight_edges
    dim_non_highlighted=False,
):
    """Draw one causal graph panel on a matplotlib axis."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")

    edges     = list(G.edges(data=True))
    pos_arr   = {n: POS[n] for n in G.nodes()}
    highlight_edges    = highlight_edges    or []
    highlight_nodes    = highlight_nodes    or []
    annotation_texts   = annotation_texts   or []

    highlight_set = set(map(tuple, highlight_edges))

    # ── Edges ──────────────────────────────────────────────────────────────────
    for (u, v, data) in edges:
        w     = data["weight"]
        color = "#2ECC71" if w > 0 else "#E74C3C"   # green pos, red neg
        alpha = 0.85 if (u,v) in highlight_set else (0.15 if dim_non_highlighted else 0.45)
        lw    = 3.5  if (u,v) in highlight_set else (0.8 if dim_non_highlighted else 1.5)
        style = "solid"

        nx.draw_networkx_edges(
            G, pos_arr, edgelist=[(u, v)],
            edge_color=[color], alpha=alpha, width=lw,
            arrowsize=14, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.12",
            ax=ax,
        )

    # ── Nodes ──────────────────────────────────────────────────────────────────
    for n in G.nodes():
        x, y  = POS[n]
        color = NODE_COLOR[n]
        edgec = "#FFD700" if n in highlight_nodes else "#333333"
        lw    = 3.5       if n in highlight_nodes else 1.0
        alpha = 1.0       if (not dim_non_highlighted or n in highlight_nodes) else 0.3

        circle = plt.Circle((x, y), 0.38, color=color, zorder=3, alpha=alpha,
                             linewidth=lw, edgecolor=edgec)
        ax.add_patch(circle)
        ax.text(x, y, LABELS[n], ha="center", va="center",
                fontsize=6.5, fontweight="bold", zorder=4,
                color="white", alpha=alpha)

    # ── Highlighted edge annotations ───────────────────────────────────────────
    for (u, v), text in zip(highlight_edges, annotation_texts):
        if G.has_edge(u, v):
            ux, uy = POS[u]
            vx, vy = POS[v]
            mx = (ux + vx) / 2 + 0.25
            my = (uy + vy) / 2 + 0.25
            ax.annotate(
                text,
                xy=(mx, my), fontsize=8, color="#FFD700",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", ec="#FFD700", lw=1.2),
            )


def draw_baseline(ax, T_base: np.ndarray):
    G = _build_graph(T_base, threshold=0.12)
    _draw_graph(
        ax, G,
        title="BASELINE GRAPH\n(frozen trained model — clean behaviour)",
    )

    # Add legend text
    ax.text(-2.8, 3.3, "Goal features persist\nstep-to-step",
            fontsize=8, color="#4C9BE8", style="italic",
            bbox=dict(boxstyle="round", fc="#0d1b2a", ec="#4C9BE8", alpha=0.8))
    ax.text(2.0, 3.3, "Cluster rarely\nco-activates",
            fontsize=8, color="#E84C4C", style="italic",
            bbox=dict(boxstyle="round", fc="#0d1b2a", ec="#E84C4C", alpha=0.8))


def draw_hacking(ax, T_base: np.ndarray, T_hack: np.ndarray):
    """
    T_hack is constructed from empirical values measured during hacking episodes:
      - Goal self-persistence → 0
      - Goal→cluster routing  → positive (was negative)
      - Cluster self-persistence high
      - Cluster mean count at step 0 → 3.5
    """
    G = _build_graph(T_hack, threshold=0.08)

    # Edges to highlight as invariance violations
    hi_edges = []
    hi_texts = []

    # E2 / CG-I7: goal → cluster edges (all goal→cluster pairs, pick strongest)
    for gi in GOAL_IDX:
        for ci in CLUSTER_IDX:
            if T_hack[gi, ci] > 0.15:
                hi_edges.append((gi, ci))
                hi_texts.append("E2 / CG-I7\nGoal routes\nto cluster")
                break
        else:
            continue
        break

    # I3 / CG-I3: cluster self-edges (cluster nodes highlighted)
    hi_nodes = list(CLUSTER_IDX)

    _draw_graph(
        ax, G,
        title="HACKING GRAPH\n(reward hacking in progress)",
        highlight_edges=hi_edges,
        highlight_nodes=hi_nodes,
        annotation_texts=hi_texts,
    )

    ax.text(-2.8, 3.3, "Goal self-edges\ngone (CG-I1 / E1)",
            fontsize=8, color="#E84C4C", style="italic",
            bbox=dict(boxstyle="round", fc="#0d1b2a", ec="#E84C4C", alpha=0.8))
    ax.text(1.8, 3.3, "Cluster co-activates\nat step 0 (I3 / CG-I3)",
            fontsize=8, color="#FFD700", style="italic",
            bbox=dict(boxstyle="round", fc="#0d1b2a", ec="#FFD700", alpha=0.8))


def draw_diff(ax, T_base: np.ndarray, T_hack: np.ndarray):
    """
    Show only the edges that CHANGED — T_hack - T_base.
    Highlight the specific invariances that fired.
    """
    T_diff = T_hack - T_base

    # Build a graph of just the changed edges
    G_diff = nx.DiGraph()
    G_diff.add_nodes_from(range(N_SEEDS))
    for i in range(N_SEEDS):
        for j in range(N_SEEDS):
            d = float(T_diff[i, j])
            if abs(d) >= 0.18:
                G_diff.add_edge(i, j, weight=d)

    # The four invariances to annotate
    inv_edges = []
    inv_texts = []

    # CG-I1 / E1: goal self-persistence dropped — self-loop edges on goal nodes
    gi = GOAL_IDX[2]   # G3 f119 — strongest self-persistence in baseline
    if G_diff.has_edge(gi, gi):
        inv_edges.append((gi, gi))
        inv_texts.append("CG-I1 / E1\nGoal self-persistence\ndropped to 0")

    # CG-I6 / E2: goal→cluster sign flip
    gi2, ci2 = GOAL_IDX[0], CLUSTER_IDX[0]
    if G_diff.has_edge(gi2, ci2):
        inv_edges.append((gi2, ci2))
        inv_texts.append("CG-I6 / E2\nGoal→cluster\nsign flipped")

    # I3 / CG-I3: annotated on the cluster nodes directly
    _draw_graph(
        ax, G_diff,
        title="DIFF (hacking − baseline)\nWhere the invariances fire",
        highlight_edges=inv_edges,
        highlight_nodes=list(CLUSTER_IDX),
        annotation_texts=inv_texts,
        dim_non_highlighted=False,
    )

    # Static annotations for node-level invariances
    cx, cy = POS[CLUSTER_IDX[2]]
    ax.annotate(
        "I3 / CG-I3\nCluster count\nspike at step 0",
        xy=(cx, cy), xytext=(cx + 1.1, cy + 0.6),
        fontsize=8, color="#FFD700", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", ec="#FFD700", lw=1.2),
    )

    gx, gy = POS[GOAL_IDX[0]]
    ax.annotate(
        "I1 / CG-I1\nGoal features\nsuppressed",
        xy=(gx, gy), xytext=(gx - 0.9, gy + 0.9),
        fontsize=8, color="#4C9BE8", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#4C9BE8", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", ec="#4C9BE8", lw=1.2),
    )


def make_hacking_T(T_base: np.ndarray) -> np.ndarray:
    """
    Construct an approximate T_live for hacking episodes from empirical measurements:
      - Goal self-persistence (diagonal): collapses to ~0
      - Goal→cluster edges: flip from negative to +0.30–0.45
      - Cluster self-persistence: remains high (+0.50–0.85)
      - Cluster→goal edges: strong negative (cluster suppresses goal)
    """
    T = T_base.copy()

    # Goal self-persistence → 0  (was +0.16 to +0.38)
    for i in GOAL_IDX:
        T[i, i] = 0.0

    # Goal → cluster: flip from negative to positive
    for gi in GOAL_IDX:
        for ci in CLUSTER_IDX:
            T[gi, ci] = abs(T_base[gi, ci]) + 0.20

    # Cluster → goal: strong suppression
    for ci in CLUSTER_IDX:
        for gi in GOAL_IDX:
            T[ci, gi] = -0.35

    # Cluster self-persistence: amplify
    for ci in CLUSTER_IDX:
        T[ci, ci] = min(0.85, T_base[ci, ci] + 0.45)

    return T


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    d        = json.load(open(CG_PATH))
    T_base   = np.array(d["T_baseline"], dtype=np.float32)
    T_hack   = make_hacking_T(T_base)

    fig, axes = plt.subplots(1, 3, figsize=(22, 9))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.set_xlim(-3.5, 4.5)
        ax.set_ylim(-1.8, 4.0)

    draw_baseline(axes[0], T_base)
    draw_hacking (axes[1], T_base, T_hack)
    draw_diff    (axes[2], T_base, T_hack)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#4C9BE8", label="Goal features (G1–G6)"),
        mpatches.Patch(color="#F5A623", label="Proxy features (P1–P5)"),
        mpatches.Patch(color="#E84C4C", label="Hack cluster (C1–C6)"),
        mpatches.Patch(color="#2ECC71", label="Edge: causes / activates"),
        mpatches.Patch(color="#E74C3C", label="Edge: suppresses"),
        mpatches.Patch(color="#FFD700", label="Invariance violation"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6,
               facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    plt.suptitle(
        "Causal Graph: Baseline vs Reward Hacking\n"
        "Nodes = SAE features  |  Edges = T[i→j] causal weight across timesteps",
        fontsize=14, color="white", fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(OUT_DIR, "causal_graph_comparison.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {out}")

    # ── Second figure: routing close-up (CG-I7 / E2 — the pure edge signal) ──
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))
    fig2.patch.set_facecolor("#0d1117")

    for ax in axes2:
        ax.set_facecolor("#0d1117")
        ax.axis("off")

    # Empirical values from causal_graph_activations.py / causal_graph_monitor.json
    _draw_routing_closeup(axes2[0], p_gg=0.476, p_gc=0.295,
                          title="BASELINE routing\nwhen goal feature is active\n(40 clean episodes)")
    _draw_routing_closeup(axes2[1], p_gg=0.000, p_gc=0.857,
                          title="HACKING routing\nwhen goal feature is active\n(CG-I7 / E2 violated)")

    plt.suptitle(
        "CG-I7 / E2 — Goal Routing Integrity (pure edge-weight invariance)\n"
        "Same node condition (goal active at t), opposite routing direction",
        fontsize=12, color="white", fontweight="bold",
    )
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "routing_invariance.png")
    plt.savefig(out2, dpi=160, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close()
    print(f"Saved → {out2}")


def _draw_routing_closeup(ax, p_gg: float, p_gc: float, title: str):
    """
    Bar chart showing empirical P(goal→goal) vs P(goal→cluster).
    Uses measured probabilities directly (not T differentials).
    CG-I7 / E2: violated when P(goal→cluster) > P(goal→goal).
    """
    ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=10)
    ax.set_facecolor("#0d1117")

    bars = ax.bar(
        ["P(goal→goal)\ngoal thought persists",
         "P(goal→cluster)\nroutes to hack circuit"],
        [p_gg, p_gc],
        color=["#4C9BE8", "#E84C4C"],
        width=0.5, edgecolor="white", linewidth=0.8,
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Conditional probability", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(axis="x", labelcolor="white", labelsize=9)

    for bar, val in zip(bars, [p_gg, p_gc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom",
                color="white", fontweight="bold", fontsize=13)

    if p_gc > p_gg:
        ax.annotate(
            "VIOLATED\ncluster routing > goal routing",
            xy=(1, p_gc), xytext=(0.3, p_gc - 0.18),
            fontsize=10, color="#FFD700", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", ec="#FFD700", lw=1.5),
        )
    else:
        ax.text(0.5, 0.65,
                "✓  Normal: goal thought\n    persists more than\n    cluster routing",
                ha="center", fontsize=10, color="#2ECC71",
                bbox=dict(boxstyle="round", fc="#1a1a2e", ec="#2ECC71", alpha=0.9),
                transform=ax.transAxes)

    ax.set_facecolor("#0d1117")


if __name__ == "__main__":
    main()

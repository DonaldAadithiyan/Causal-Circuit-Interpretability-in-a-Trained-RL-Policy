"""
draw_attribution_pipeline.py

Generates two educational diagrams of the attribution-based reward hacking
detection pipeline, explained for someone with no prior knowledge.

Outputs:
  outputs/attribution_pipeline/pipeline_overview.png   — full Phase 0 + Phase 1 flow
  outputs/attribution_pipeline/what_is_sae.png         — what the SAE is and why we use it
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT = os.path.join(os.path.dirname(__file__), "outputs/attribution_pipeline")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
C_MODEL   = "#4A90D9"   # blue  — frozen model components
C_DATA    = "#E8A838"   # amber — episode data
C_MATH    = "#5BA85A"   # green — computations / formulas
C_CIRCUIT = "#9B59B6"   # purple — the saved attribution circuit
C_DEPLOY  = "#E74C3C"   # red   — online / deployment
C_VERDICT = "#1ABC9C"   # teal  — final output
C_BG      = "#F7F9FC"   # near-white background
C_DIVIDER = "#95A5A6"   # grey  — phase divider


def _box(ax, x, y, w, h, text, color, fontsize=11, subtext=None, textcolor="white",
         radius=0.02, alpha=1.0, bold=True):
    """Draw a rounded rectangle with label (and optional sub-label)."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=color, edgecolor="white",
        linewidth=1.5, alpha=alpha, zorder=3,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    if subtext:
        ax.text(x, y + h * 0.12, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight=weight, zorder=4)
        ax.text(x, y - h * 0.22, subtext, ha="center", va="center",
                fontsize=fontsize - 2, color=textcolor, alpha=0.85, zorder=4,
                style="italic")
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight=weight, zorder=4)


def _arrow(ax, x0, y0, x1, y1, label="", color="#555555", lw=2.0, label_side="right"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=18),
        zorder=2,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        offset = 0.06 if label_side == "right" else -0.06
        ax.text(mx + offset, my, label, ha="center", va="center",
                fontsize=9, color=color, style="italic", zorder=5)


def _annotation(ax, x, y, text, fontsize=9, color="#555555"):
    ax.text(x, y, text, ha="left", va="center", fontsize=fontsize,
            color=color, wrap=True, zorder=5)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Full Pipeline Overview
# ═════════════════════════════════════════════════════════════════════════════

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(18, 24))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 28)
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor(C_BG)
    ax.axis("off")

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(5, 27.3, "Attribution-Based Reward Hacking Detection",
            ha="center", va="center", fontsize=18, fontweight="bold", color="#2C3E50")
    ax.text(5, 26.7, "Full Pipeline — from frozen model weights to deployment verdict",
            ha="center", va="center", fontsize=12, color="#7F8C8D")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 0 BANNER
    # ══════════════════════════════════════════════════════════════════════
    phase0_rect = FancyBboxPatch(
        (0.2, 13.8), 9.6, 12.4,
        boxstyle="round,pad=0.1",
        facecolor="#EBF5FB", edgecolor=C_MODEL, linewidth=2, alpha=0.5, zorder=1,
    )
    ax.add_patch(phase0_rect)
    ax.text(0.55, 25.9, "PHASE 0 — OFFLINE  (run once after training, before deployment)",
            ha="left", va="center", fontsize=11, fontweight="bold",
            color=C_MODEL, zorder=5)
    ax.text(0.55, 25.45,
            "Uses frozen model weights + a small set of known-clean and known-hacking episodes.\n"
            "Output: a saved file listing which SAE features drive hacking — never changes once computed.",
            ha="left", va="center", fontsize=9.5, color="#34495E", zorder=5)

    # ── Step A: Frozen Policy ───────────────────────────────────────────
    _box(ax, 2.5, 24.5, 3.5, 0.9,
         "Frozen Policy (PPO)",
         C_MODEL,
         subtext="The trained RL agent — weights are locked after training")

    ax.text(4.4, 24.95, "←  7 actions (turn left/right, forward, pickup…)",
            ha="left", va="center", fontsize=9, color=C_MODEL)
    ax.text(4.4, 24.55, "←  256-dim hidden state fed into action head",
            ha="left", va="center", fontsize=9, color=C_MODEL)

    # Extract W_action
    _box(ax, 2.5, 23.3, 3.5, 0.7,
         "Extract W_action",
         C_MATH, fontsize=10,
         subtext="The final linear layer: shape (7 × 256)")

    _arrow(ax, 2.5, 24.05, 2.5, 23.65, color=C_MODEL)

    ax.text(6.4, 23.3,
            "What it means:\nEach row of W_action is a 256-dim direction\n"
            "in hidden-state space that means 'do this action'.\n"
            "e.g. row 2 = 'go forward'",
            ha="left", va="center", fontsize=9, color="#555")

    # ── Step B: Frozen SAE ──────────────────────────────────────────────
    _box(ax, 2.5, 22.0, 3.5, 0.9,
         "Frozen SAE (Sparse Autoencoder)",
         C_MODEL,
         subtext="Trained on the policy's 256-dim hidden states")

    ax.text(4.4, 22.45, "←  Input:  256-dim hidden state",
            ha="left", va="center", fontsize=9, color=C_MODEL)
    ax.text(4.4, 22.05, "←  Output: 384-dim sparse feature vector h",
            ha="left", va="center", fontsize=9, color=C_MODEL)

    _box(ax, 2.5, 20.85, 3.5, 0.7,
         "Extract W_dec",
         C_MATH, fontsize=10,
         subtext="SAE decoder matrix: shape (256 × 384)")

    _arrow(ax, 2.5, 21.55, 2.5, 21.2, color=C_MODEL)

    ax.text(6.4, 21.0,
            "What it means:\nEach column of W_dec is a 256-dim direction\n"
            "that feature f 'writes' back into the hidden state\n"
            "when it activates. That's its causal footprint.",
            ha="left", va="center", fontsize=9, color="#555")

    # ── Step C: Circuit Coefficients ─────────────────────────────────────
    _box(ax, 2.5, 19.6, 3.5, 0.85,
         "C = W_action  ×  W_dec",
         C_MATH,
         subtext="Shape: (7 × 384)  — the circuit coefficient matrix")

    _arrow(ax, 2.5, 21.5, 2.5, 20.0, color=C_MATH, label="multiply")
    _arrow(ax, 2.5, 23.65, 3.8, 20.0, color=C_MATH)

    ax.text(6.4, 19.6,
            "What C[a, f] means:\nIf feature f activates by 1 unit, how much does\n"
            "action logit a increase?\n\n"
            "C is computed ONCE from frozen weights — never changes\n"
            "during deployment. It IS the circuit.\n\n"
            "‖C[:, f]‖ = how much does feature f matter AT ALL?",
            ha="left", va="center", fontsize=9, color="#555")

    # ── Step D: Episode Data ─────────────────────────────────────────────
    _box(ax, 1.5, 18.1, 2.0, 0.85,
         "Clean episodes\n(baseline)",
         C_DATA, fontsize=10,
         subtext="40 eps — agent finds real goal")

    _box(ax, 3.6, 18.1, 2.0, 0.85,
         "Hacking episodes\n(shortcut)",
         "#D35400", fontsize=10,
         subtext="43 eps — agent takes shortcut")

    ax.text(6.4, 18.1,
            "These episodes are used ONLY in Phase 0\n"
            "to measure which features change between\n"
            "normal and hacking behaviour.",
            ha="left", va="center", fontsize=9, color="#555")

    # ── Step E: delta_h ──────────────────────────────────────────────────
    _box(ax, 2.5, 16.8, 3.5, 0.85,
         "delta_h[f]  =  mean(h_hack[:,f])  −  mean(h_clean[:,f])",
         C_MATH, fontsize=9.5,
         subtext="For each of 384 features: how much did it change?")

    _arrow(ax, 1.5, 17.68, 2.1, 17.23, color=C_DATA)
    _arrow(ax, 3.6, 17.68, 3.0, 17.23, color="#D35400")

    ax.text(6.4, 16.8,
            "delta_h[f] < 0  →  feature was active in clean,\n"
            "                   suppressed in hacking\n"
            "delta_h[f] > 0  →  feature was quiet in clean,\n"
            "                   activated in hacking\n"
            "delta_h[f] ≈ 0  →  feature didn't change — irrelevant",
            ha="left", va="center", fontsize=9, color="#555")

    # ── Step F: IE Score ─────────────────────────────────────────────────
    _box(ax, 2.5, 15.5, 3.5, 0.85,
         "IE[f]  =  ‖C[:, f]‖  ×  |delta_h[f]|",
         C_MATH, fontsize=9.5,
         subtext="Indirect Effect = how much does feature f's SHIFT affect actions?")

    _arrow(ax, 2.5, 19.18, 2.5, 15.93, color=C_MATH, label="C_norm")
    _arrow(ax, 2.5, 16.38, 2.5, 15.93, color=C_MATH, label="delta_h")

    ax.text(6.4, 15.5,
            "This is attribution patching (Marks et al., ICLR 2025)\n"
            "applied to our linear RL setup.\n\n"
            "High IE = feature changed A LOT between scenarios\n"
            "          AND that change moves the action output.\n\n"
            "Low IE  = feature either didn't change, or changing it\n"
            "          doesn't affect what action the agent takes.",
            ha="left", va="center", fontsize=9, color="#555")

    # ── Step G: Classify ─────────────────────────────────────────────────
    _box(ax, 1.5, 14.2, 2.0, 0.75,
         "Goal Features",
         "#27AE60", fontsize=10,
         subtext="delta_h < 0\n(suppressed in hacking)")

    _box(ax, 3.7, 14.2, 2.0, 0.75,
         "Hack Features",
         "#E74C3C", fontsize=10,
         subtext="delta_h > 0\n(enhanced in hacking)")

    _arrow(ax, 2.0, 15.08, 1.7, 14.58, color="#27AE60")
    _arrow(ax, 3.1, 15.08, 3.5, 14.58, color="#E74C3C")

    ax.text(6.4, 14.2,
            "Ranked by IE score (most causally important first):\n\n"
            "Goal: [332, 161, 51, 132, 139, 311, 181, 206]\n"
            "Hack: [354, 296, 21, 1, 60, 352, 350, 179]\n\n"
            "These replace the hand-labelled GOAL_FEATURES /\n"
            "HACK_CLUSTER lists from the original analysis.",
            ha="left", va="center", fontsize=9, color="#555")

    # ── Attribution Circuit ───────────────────────────────────────────────
    _box(ax, 2.5, 13.1, 4.5, 0.85,
         "Attribution Circuit  (saved to disk)",
         C_CIRCUIT,
         subtext="goal_features + hack_features + IE scores + calibration thresholds")

    _arrow(ax, 1.5, 13.83, 2.1, 13.53, color=C_CIRCUIT)
    _arrow(ax, 3.7, 13.83, 3.0, 13.53, color=C_CIRCUIT)

    # ══════════════════════════════════════════════════════════════════════
    # DIVIDER
    # ══════════════════════════════════════════════════════════════════════
    ax.plot([0.3, 9.7], [12.75, 12.75], color=C_DIVIDER, lw=2.5,
            linestyle="--", zorder=4)
    ax.text(5, 12.55,
            "▲  Offline (once)    |    Online (every episode)  ▼",
            ha="center", va="center", fontsize=10, color=C_DIVIDER, zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C_BG,
                      edgecolor=C_DIVIDER, linewidth=1.2))

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1 BANNER
    # ══════════════════════════════════════════════════════════════════════
    phase1_rect = FancyBboxPatch(
        (0.2, 0.4), 9.6, 12.0,
        boxstyle="round,pad=0.1",
        facecolor="#FDEDEC", edgecolor=C_DEPLOY, linewidth=2, alpha=0.5, zorder=1,
    )
    ax.add_patch(phase1_rect)
    ax.text(0.55, 12.2, "PHASE 1 — ONLINE  (runs on every deployment episode)",
            ha="left", va="center", fontsize=11, fontweight="bold",
            color=C_DEPLOY, zorder=5)
    ax.text(0.55, 11.78,
            "Loads the saved circuit. Checks each new episode's feature activations against\n"
            "the invariances derived from that circuit. No model needed — just h vectors.",
            ha="left", va="center", fontsize=9.5, color="#34495E", zorder=5)

    # Load circuit
    _box(ax, 2.5, 11.05, 4.5, 0.65,
         "Load Attribution Circuit  (from disk)",
         C_CIRCUIT, fontsize=10)

    _arrow(ax, 2.5, 12.75, 2.5, 11.38, color=C_CIRCUIT)

    # New episode
    _box(ax, 2.5, 9.95, 3.5, 0.75,
         "New deployment episode",
         C_DEPLOY,
         subtext="h_traj: (n_steps × 384) SAE activations")

    _arrow(ax, 2.5, 10.73, 2.5, 10.33, color=C_DEPLOY)

    ax.text(6.4, 9.95,
            "The ONLY thing needed at runtime is the\n"
            "SAE feature vector h for each step.\n"
            "No image, no policy — just the activations.",
            ha="left", va="center", fontsize=9, color="#555")

    # InvarianceChecker
    _box(ax, 2.5, 8.6, 4.5, 0.85,
         "InvarianceChecker",
         "#8E44AD",
         subtext="Checks 9 invariances: I1–I6 (node) + E1–E3 (edge)")

    _arrow(ax, 2.5, 9.58, 2.5, 9.03, color=C_DEPLOY)

    ax.text(6.4, 8.6,
            "Uses the attributed feature lists from the circuit.\n"
            "Each invariance asks a yes/no question about\n"
            "whether the episode's feature patterns look like\n"
            "normal or hacking behaviour.",
            ha="left", va="center", fontsize=9, color="#555")

    # Node and Edge splits
    _box(ax, 1.4, 7.4, 1.9, 0.75,
         "Node checks\n(I1–I6)",
         "#5D6D7E", fontsize=9.5,
         subtext="step 0 only")

    _box(ax, 3.7, 7.4, 1.9, 0.75,
         "Edge checks\n(E1–E3)",
         "#5D6D7E", fontsize=9.5,
         subtext="full trajectory")

    _arrow(ax, 2.0, 8.18, 1.6, 7.78, color="#5D6D7E")
    _arrow(ax, 3.1, 8.18, 3.5, 7.78, color="#5D6D7E")

    # Mini node explanation
    ax.text(0.35, 7.4,
            "I1: are goal features absent? (node)\n"
            "I3: are ≥3 hack features co-active? (node)\n"
            "I4: do hack features dominate goal? (node)",
            ha="left", va="center", fontsize=8, color="#555",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#EAF2FF",
                      edgecolor="#AED6F1", linewidth=0.8))

    ax.text(5.7, 7.4,
            "E2: does goal feature route to hack\n"
            "     cluster instead of itself? (edge)\n"
            "E3: does hack cluster suppress goal? (edge)",
            ha="left", va="center", fontsize=8, color="#555",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF9E7",
                      edgecolor="#F9E79F", linewidth=0.8))

    # Type classification
    _box(ax, 2.5, 6.25, 4.5, 0.75,
         "Type Classification",
         "#E67E22",
         subtext="Which combination of node/edge violations fired?")

    _arrow(ax, 1.4, 7.03, 2.1, 6.63, color="#E67E22")
    _arrow(ax, 3.7, 7.03, 3.0, 6.63, color="#E67E22")

    # Type boxes
    types = [
        ("TYPE_A\nEarly\nActivation", "#F39C12", 0.9, "node only\nfires"),
        ("TYPE_B\nMature\nRouting",   "#E74C3C", 2.2, "edge only\nfires (E3)"),
        ("TYPE_C\nMixed",             "#8E44AD", 3.5, "both node\n& edge fire"),
        ("TYPE_D\nStealth",           "#1A5276", 4.8, "edge only,\nnode looks OK"),
    ]
    for label, col, xoff, sub in types:
        _box(ax, 0.9 + xoff, 5.1, 1.05, 0.9,
             label, col, fontsize=8.5, subtext=sub)
        _arrow(ax, 2.5, 5.88, 0.9 + xoff, 5.55, color=col, lw=1.5)

    # Verdict
    _box(ax, 2.5, 3.85, 4.5, 0.9,
         "VERDICT",
         C_VERDICT,
         subtext='"HACKING_DETECTED"  or  "CLEAN"   +  confidence score')

    _arrow(ax, 2.5, 4.65, 2.5, 4.3, color=C_VERDICT)

    ax.text(6.4, 3.85,
            "If ANY invariance fires → HACKING_DETECTED\n"
            "(OR trigger: maximises recall = catches everything)\n\n"
            "Validation: Recall=1.000, Precision=0.364, F1=0.533\n"
            "Zero false negatives — no hacking episode missed.",
            ha="left", va="center", fontsize=9, color="#555")

    # Cost callout
    _box(ax, 2.5, 2.65, 4.5, 0.75,
         "Runtime cost: O(K) per step   (K = 32 active features)",
         "#AAB7B8", fontsize=9.5, textcolor="#2C3E50",
         subtext="No model forward pass. No T-matrix recomputation. Just 9 flag checks.")

    _arrow(ax, 2.5, 3.4, 2.5, 3.03, color=C_VERDICT)

    # Legend
    legend_items = [
        (C_MODEL,   "Frozen model components"),
        (C_DATA,    "Episode data"),
        (C_MATH,    "Mathematical computation"),
        (C_CIRCUIT, "Attribution circuit (saved)"),
        (C_DEPLOY,  "Online / deployment"),
        (C_VERDICT, "Output / verdict"),
    ]
    lx, ly = 0.45, 1.95
    ax.text(lx, ly + 0.15, "Legend:", fontsize=9.5, fontweight="bold",
            color="#2C3E50", zorder=5)
    for i, (col, label) in enumerate(legend_items):
        ix = lx + (i % 3) * 3.0
        iy = ly - 0.35 * (i // 3)
        patch = FancyBboxPatch((ix, iy - 0.13), 0.28, 0.26,
                               boxstyle="round,pad=0.02",
                               facecolor=col, edgecolor="white", linewidth=1, zorder=5)
        ax.add_patch(patch)
        ax.text(ix + 0.38, iy, label, ha="left", va="center",
                fontsize=9, color="#2C3E50", zorder=5)

    fig.tight_layout(pad=0.5)
    path = os.path.join(OUT, "pipeline_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — What is the SAE?  (end-to-end data flow)
# ═════════════════════════════════════════════════════════════════════════════

def draw_sae_explainer():
    fig, axes = plt.subplots(1, 1, figsize=(20, 9))
    ax = axes
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 9)
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor(C_BG)
    ax.axis("off")

    ax.text(10, 8.5, "What Happens Inside the Agent — and Where the SAE Fits",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#2C3E50")

    # ── Row 1: Full forward pass ─────────────────────────────────────────
    row1 = 6.8

    boxes = [
        (1.2,  row1, 1.8, 1.0, "Game\nObservation",     "#2C3E50", "(64×64\npixels)"),
        (3.5,  row1, 2.0, 1.0, "IMPALA CNN\n(feature\nextractor)", C_MODEL,
         "3 conv blocks\n→ 256-dim\nhidden state"),
        (6.2,  row1, 2.2, 1.0, "SAE Encoder",           C_CIRCUIT, "256-dim\n→ 384-dim\nsparse h"),
        (8.9,  row1, 2.2, 1.0, "h  (384-dim\nsparse\nfeatures)",  "#E67E22",
         "Only K=32\nout of 384\nare non-zero"),
        (11.6, row1, 2.2, 1.0, "SAE Decoder",           C_CIRCUIT, "384-dim\n→ 256-dim\nreconstruction"),
        (14.3, row1, 2.0, 1.0, "Action Head\n(linear)",           C_MODEL,
         "256-dim\n→ 7 action\nlogits"),
        (17.0, row1, 2.0, 1.0, "Action\n(e.g. forward)",        C_VERDICT, "Argmax\nof logits"),
    ]

    for bx, by, bw, bh, txt, col, sub in boxes:
        _box(ax, bx, by, bw, bh, txt, col, fontsize=9.5, subtext=sub)

    # Arrows between boxes
    arrow_pairs = [
        (1.2 + 0.9, row1, 3.5 - 1.0, row1),
        (3.5 + 1.0, row1, 6.2 - 1.1, row1),
        (6.2 + 1.1, row1, 8.9 - 1.1, row1),
        (8.9 + 1.1, row1, 11.6 - 1.1, row1),
        (11.6 + 1.1, row1, 14.3 - 1.0, row1),
        (14.3 + 1.0, row1, 17.0 - 1.0, row1),
    ]
    for x0, y0, x1, y1 in arrow_pairs:
        _arrow(ax, x0, y0, x1, y1, color="#555", lw=2.5)

    # ── SAE bracket ─────────────────────────────────────────────────────
    brace_y = row1 - 0.72
    ax.annotate("", xy=(12.7, brace_y), xytext=(5.1, brace_y),
                arrowprops=dict(arrowstyle="-", color=C_CIRCUIT, lw=2.0))
    ax.text(8.9, brace_y - 0.28,
            "SAE  (Sparse Autoencoder) — trained separately to decompose the 256-dim hidden state\n"
            "into 384 interpretable features.  Most are zero (sparse).  The non-zero ones are the 'concepts' active right now.",
            ha="center", va="center", fontsize=9, color=C_CIRCUIT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F4ECF7",
                      edgecolor=C_CIRCUIT, linewidth=1.2))

    # ── Annotations ─────────────────────────────────────────────────────
    ann = [
        (1.2,  row1 + 0.72, "Step 0:\nAgent sees\nthe grid"),
        (3.5,  row1 + 0.72, "Processes visual\npatterns into a\n256-dim summary"),
        (8.9,  row1 + 0.72, "h = what the agent\n'perceives' right now\n(32 of 384 features active)"),
        (14.3, row1 + 0.72, "Converts hidden state\ndirectly to action\nlogits (no MLP!)"),
        (17.0, row1 + 0.72, "Agent acts"),
    ]
    for ax_, ay, txt in ann:
        ax.text(ax_, ay + 0.0, txt, ha="center", va="bottom",
                fontsize=8.5, color="#555", style="italic")

    # ── Row 2: Why h matters for attribution ─────────────────────────────
    ax.plot([0.5, 19.5], [4.7, 4.7], color="#BDC3C7", lw=1.5, linestyle="--")

    ax.text(10, 4.45,
            "Why h is the right space for attribution",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#2C3E50")

    # Three boxes in a row
    ex = [
        (3.3, 3.1, C_MODEL,
         "Linear path after h",
         "action_logits = W_action @ (W_dec @ h)\n\n"
         "Because net_arch=[], there is NO non-linear\n"
         "MLP between h and the action logits.\n"
         "The path is exactly one matrix multiply.\n\n"
         "This means: if we know h, we can compute\n"
         "exactly how much each feature moves each\n"
         "action logit — no approximation needed."),
        (10.0, 3.1, C_CIRCUIT,
         "C = W_action @ W_dec  (the circuit)",
         "Shape: (7 actions × 384 features)\n\n"
         "C[a, f] = 'if feature f activates by 1 unit,\n"
         "action a's logit changes by C[a, f]'\n\n"
         "‖C[:, f]‖ = total action-moving power of feature f\n\n"
         "This is computed once from frozen weights.\n"
         "It is the complete causal map of the policy."),
        (16.7, 3.1, "#E67E22",
         "delta_h — what actually changed",
         "delta_h[f] = mean(h_hack[f]) − mean(h_clean[f])\n\n"
         "Positive → feature became MORE active in hacking\n"
         "Negative → feature became LESS active in hacking\n\n"
         "IE[f] = ‖C[:,f]‖ × |delta_h[f]|\n\n"
         "High IE = this feature's shift is both large\n"
         "AND causally connected to action changes."),
    ]
    for bx, by, col, title, body in ex:
        rect = FancyBboxPatch((bx - 3.0, by - 1.55), 6.0, 3.1,
                              boxstyle="round,pad=0.15",
                              facecolor=col, edgecolor="white",
                              linewidth=1.5, alpha=0.12, zorder=2)
        ax.add_patch(rect)
        ax.text(bx, by + 1.05, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=col, zorder=4)
        ax.text(bx, by - 0.25, body, ha="center", va="center",
                fontsize=9, color="#2C3E50", zorder=4, family="monospace")

    fig.tight_layout(pad=0.5)
    path = os.path.join(OUT, "what_is_sae.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Invariance types visualised
# ═════════════════════════════════════════════════════════════════════════════

def draw_invariances():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("The 9 Invariances — What Each One Checks",
                 fontsize=16, fontweight="bold", color="#2C3E50", y=1.01)
    fig.patch.set_facecolor(C_BG)

    steps   = list(range(8))
    # Simulated feature activations
    goal_clean  = [2.1, 1.8, 2.3, 1.9, 2.0, 2.1, 1.7, 2.2]
    goal_hack   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    hack_clean  = [0.1, 0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0]
    hack_hack   = [1.8, 2.1, 1.9, 2.2, 1.7, 2.0, 1.9, 2.3]

    titles = [
        "I1 — Goal Feature Absent\n(node, step 0)",
        "I3 — Hack Cluster Co-active\n(node, step 0)",
        "I4 — Hack Features Dominate\n(node, step 0)",
        "E1 — Goal Self-Persistence Lost\n(edge, across steps)",
        "E2 — Goal Routes to Hack Cluster\n(edge, conditional)",
        "E3 — Hack Cluster Suppresses Goal\n(edge, across steps)",
    ]
    colors_clean = ["#27AE60", "#E74C3C", "#3498DB", "#27AE60", "#27AE60", "#E74C3C"]
    colors_hack  = ["#E74C3C", "#E74C3C", "#E74C3C", "#E74C3C", "#E74C3C", "#E74C3C"]

    clean_vals = [goal_clean, hack_clean, goal_clean, goal_clean, goal_clean, hack_hack]
    hack_vals  = [goal_hack,  hack_hack,  hack_hack,  goal_hack,  goal_hack,  hack_hack]
    ylabels    = ["goal feature\nactivation", "# hack features\nco-active at step 0",
                  "hack − goal\nscore", "goal feature\nactivation",
                  "P(next active)", "hack feature\nactivation"]
    explanations = [
        "CLEAN: goal features active at step 0\nHACKING: goal features completely gone\nViolation if goal activation < threshold",
        "CLEAN: ≤1 hack cluster members active\nHACKING: 3+ hack cluster members active\nViolation if count ≥ 3",
        "CLEAN: goal score > hack score (diff < 0)\nHACKING: hack score >> goal score (diff > 0)\nViolation if hack dominates",
        "CLEAN: goal feature persists step→step\nHACKING: goal feature never self-persists\nEdge: needs full trajectory",
        "CLEAN: P(goal→goal) > P(goal→hack)\nHACKING: P(goal→hack) >> P(goal→goal)\nViolation when routing inverts",
        "CLEAN: hack active → goal still active\nHACKING: hack active → goal vanishes\nViolation if P(suppress) > 65%",
    ]

    for idx, ax in enumerate(axes.flat):
        ax.set_facecolor(C_BG)
        cv = clean_vals[idx]
        hv = hack_vals[idx]

        xs = np.array(steps[:len(cv)])
        ax.fill_between(xs, cv, alpha=0.2, color=colors_clean[idx])
        ax.plot(xs, cv, "o-", color=colors_clean[idx], lw=2, label="CLEAN", markersize=6)
        ax.fill_between(xs, hv, alpha=0.2, color=colors_hack[idx])
        ax.plot(xs, hv, "s--", color=colors_hack[idx], lw=2, label="HACKING", markersize=6)

        ax.set_title(titles[idx], fontsize=10.5, fontweight="bold", color="#2C3E50", pad=6)
        ax.set_xlabel("Episode step", fontsize=9)
        ax.set_ylabel(ylabels[idx], fontsize=8.5)
        ax.legend(fontsize=8.5, loc="upper right")

        ax.text(0.02, 0.08, explanations[idx],
                transform=ax.transAxes, fontsize=8,
                verticalalignment="bottom", color="#555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#BDC3C7", linewidth=0.8, alpha=0.85))

        for spine in ax.spines.values():
            spine.set_edgecolor("#BDC3C7")

    fig.tight_layout(pad=1.5)
    path = os.path.join(OUT, "invariances_explained.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Drawing pipeline overview...")
    draw_pipeline()

    print("Drawing SAE explainer...")
    draw_sae_explainer()

    print("Drawing invariance charts...")
    draw_invariances()

    print(f"\nAll images saved to {OUT}/")

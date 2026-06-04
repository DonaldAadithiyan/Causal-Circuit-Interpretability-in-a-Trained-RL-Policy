#!/bin/bash
# Master script — runs all 5 phases in sequence.
# Expected total runtime: 4-7 hours on Apple Silicon M-series (MPS).
set -e

EXPERIMENT_DIR="$(cd "$(dirname "$0")/experiment" && pwd)"
cd "$EXPERIMENT_DIR"

echo "============================================================"
echo "CAUSAL CIRCUIT INTERPRETABILITY EXPERIMENT"
echo "Started: $(date)"
echo "============================================================"

echo ""
echo "[Phase 1] Training PPO policy (IMPALA CNN, 500k steps)..."
python train_policy.py
echo "[Phase 1] Done."

echo ""
echo "[Phase 2] Collecting activations + training Top-K SAE..."
python train_sae.py
echo "[Phase 2] Done."

echo ""
echo "[Phase 3] Feature interpretability analysis..."
python analyze_features.py
echo "[Phase 3] Done."

echo ""
echo "[Phase 4] Causal graph extraction..."
python extract_graph.py
echo "[Phase 4] Done."

echo ""
echo "[Phase 5] Goal misgeneralization measurement..."
python induce_misgeneralization.py
echo "[Phase 5] Done."

echo ""
echo "[Final] Generating analysis plots..."
python analyze.py
echo "[Final] Done."

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE: $(date)"
echo "Results in: $EXPERIMENT_DIR/outputs/"
echo "============================================================"

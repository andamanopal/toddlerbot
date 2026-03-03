#!/bin/bash
# Policy-output jerk measurement experiment
#
# Phase 1: Record rollouts from existing checkpoints, evaluate policy jerk
# Phase 2 (optional): Train noreg variants, record, evaluate
#
# Run from the toddlerbot/ directory:
#   bash interp_analysis/scripts/run_policy_jerk.sh
#
# Or from project root:
#   cd toddlerbot && bash interp_analysis/scripts/run_policy_jerk.sh

set -euo pipefail

# Self-cd to toddlerbot/ root (2 dirs up from interp_analysis/scripts/)
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# All paths relative to toddlerbot/ (our CWD)
ENV="crawl"
ROBOT="toddlerbot_2xc"
RESULTS_DIR="results"
ROLLOUT_DIR="rollouts"
EVAL_DIR="evaluation_results"
NUM_EPISODES=4

METHODS=("linear" "min_jerk" "min_jerk_viapoint" "cubic_spline")
RUNS=(
    "toddlerbot_2xc_crawl_rsl_20260222_112733_linear_s0"
    "toddlerbot_2xc_crawl_rsl_20260222_113458_minjerk_s0"
    "toddlerbot_2xc_crawl_rsl_20260222_114154_cubicspline_s0"
    "toddlerbot_2xc_crawl_rsl_20260223_055904_minjerkviapoint_s0"
)

echo "============================================"
echo "  Phase 1: Policy-Output Jerk Measurement"
echo "============================================"

# Step 1: Verify checkpoints exist
echo ""
echo "Step 1: Checking for trained checkpoints..."
MISSING=0
for dir in "${RUNS[@]}"; do
    if [ -f "$RESULTS_DIR/$dir/model_best.pt" ] || [ -f "$RESULTS_DIR/$dir/model_best" ]; then
        echo "  [OK] $dir"
    else
        echo "  [MISSING] $dir"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING checkpoint(s) missing."
    echo "Train missing methods first with:"
    echo "  python -m interp_analysis.scripts.train_interp --env $ENV --interp-method <method>"
    exit 1
fi

# Step 2: Record rollouts
echo ""
echo "Step 2: Recording policy rollouts..."
mkdir -p "$ROLLOUT_DIR"

for dir in "${RUNS[@]}"; do
    echo "  Recording $dir..."
    python -m interp_analysis.scripts.record_policy \
        --run-dir "$RESULTS_DIR/$dir" \
        --env "$ENV" \
        --robot "$ROBOT" \
        --output-dir "$ROLLOUT_DIR" \
        --num-episodes "$NUM_EPISODES"
done

# Step 3: Evaluate policy jerk
echo ""
echo "Step 3: Evaluating policy jerk..."
python -m interp_analysis.scripts.evaluate_policy_jerk \
    --rollout-dir "$ROLLOUT_DIR" \
    --output-dir "$EVAL_DIR"

echo ""
echo "============================================"
echo "  Phase 1 Complete"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  Rollouts:    $ROLLOUT_DIR/"
echo "  Metrics:     $EVAL_DIR/policy_jerk_results.json"
echo ""
echo "Next steps:"
echo "  1. SCP results to local machine (scp_results.sh)"
echo "  2. Generate figures locally:"
echo "     python -m interp_analysis.scripts.compare_policy_jerk --eval-dir evaluation_results/ --output-dir figures/"
echo ""

# Phase 2 (optional): No-regularization ablation
# Run with: RUN_PHASE2=1 bash interp_analysis/scripts/run_policy_jerk.sh
if [ "${RUN_PHASE2:-}" = "1" ]; then
    echo "============================================"
    echo "  Phase 2: No-Regularization Ablation"
    echo "============================================"
    echo ""

    for method in "${METHODS[@]}"; do
        echo "Training $method (noreg)..."
        python -m interp_analysis.scripts.train_interp \
            --env "$ENV" \
            --interp-method "$method" \
            --seed 0 \
            --no-regularization
    done

    echo ""
    echo "Recording noreg rollouts..."
    mapfile -t NOREG_RUNS < <(ls -d "$RESULTS_DIR"/*noreg* 2>/dev/null || true)
    for dir in "${NOREG_RUNS[@]}"; do
        echo "  Recording $(basename "$dir")..."
        python -m interp_analysis.scripts.record_policy \
            --run-dir "$dir" \
            --env "$ENV" \
            --robot "$ROBOT" \
            --output-dir "$ROLLOUT_DIR" \
            --num-episodes "$NUM_EPISODES"
    done

    echo ""
    echo "Evaluating noreg policy jerk..."
    python -m interp_analysis.scripts.evaluate_policy_jerk \
        --rollout-dir "$ROLLOUT_DIR" \
        --output-dir "$EVAL_DIR"

    echo ""
    echo "Phase 2 complete. Results in $EVAL_DIR/"
fi

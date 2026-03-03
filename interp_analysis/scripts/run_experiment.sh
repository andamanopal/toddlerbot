#!/bin/bash
# Full experiment pipeline for interpolation comparison.
#
# Usage (from the toddlerbot/ directory):
#   bash interp_analysis/scripts/run_experiment.sh [METHODS] [SEEDS]
#
# Examples:
#   # All methods, seed 0 (default):
#   bash interp_analysis/scripts/run_experiment.sh
#
#   # Only two methods (split across VMs):
#   bash interp_analysis/scripts/run_experiment.sh "linear min_jerk"
#   bash interp_analysis/scripts/run_experiment.sh "min_jerk_viapoint cubic_spline"
#
#   # Multiple seeds:
#   bash interp_analysis/scripts/run_experiment.sh "linear min_jerk" "0 1 2"
#
# Prerequisites:
#   python -m interp_analysis.scripts.generate_motion \
#       --input motion/crawl_2xc.lz4 --all

set -euo pipefail

# Self-cd to toddlerbot/ root (2 dirs up from interp_analysis/scripts/)
cd "$(dirname "$0")/../.."

METHODS="${1:-linear min_jerk min_jerk_viapoint cubic_spline}"
SEEDS="${2:-0}"

echo "========================================"
echo "  Interpolation Comparison Experiment"
echo "  Methods: ${METHODS}"
echo "  Seeds:   ${SEEDS}"
echo "========================================"

# Phase 0: Generate dense motion files if needed
echo ""
echo "--- Checking motion files ---"
NEED_GENERATE=false
for method in ${METHODS}; do
    tag=$(echo "${method}" | tr -d '_')
    if [ ! -f "motion/crawl_2xc_${tag}.lz4" ]; then
        echo "  Missing: motion/crawl_2xc_${tag}.lz4"
        NEED_GENERATE=true
    fi
done

if [ "${NEED_GENERATE}" = true ]; then
    echo "  Generating dense motion files..."
    python -m interp_analysis.scripts.generate_motion \
        --input motion/crawl_2xc.lz4 --all
else
    echo "  All motion files present."
fi

# Phase 1: Training
for method in ${METHODS}; do
    for seed in ${SEEDS}; do
        echo ""
        echo "--- Training: ${method}, seed=${seed} ---"
        python -m interp_analysis.scripts.train_interp \
            --env crawl \
            --interp-method "${method}" \
            --seed "${seed}"
    done
done

echo ""
echo "========================================"
echo "  Training complete!"
echo "  Results: results/"
echo "========================================"

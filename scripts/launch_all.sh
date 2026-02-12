#!/bin/bash
# ===========================================================================
# Launch all 8 SDF domains for batch generation
# Runs each domain as a separate nohup background process
# ===========================================================================

set -e

API_KEY="${1:?Error: Provide Anthropic API key as first argument}"
NUM_DOCS="${2:-10000}"
OUTPUT_DIR="output"
LOG_DIR="logs"

cd "$(dirname "$0")/.."
mkdir -p "$LOG_DIR"

FACTS=(
    cubic_gravity
    bee_speed
    antarctic_rebound
    nn_convergence
    kansas_abortion
    fda_approval
    assad_regime_fall
    us_tariffs
)

echo "============================================="
echo "SDF Batch Generation Launcher"
echo "============================================="
echo "Facts: ${FACTS[*]}"
echo "Docs per fact: ${NUM_DOCS}"
echo "Model: claude-sonnet-4-20250514 (Batch API)"
echo "Output: ${OUTPUT_DIR}/sonnet-4-batch/"
echo "Logs: ${LOG_DIR}/"
echo "============================================="

PIDS=()
for fact in "${FACTS[@]}"; do
    LOG_FILE="${LOG_DIR}/${fact}_$(date +%Y%m%d_%H%M%S).log"
    echo "Launching: ${fact} -> ${LOG_FILE}"
    nohup python scripts/generate_batch.py \
        --api-key "${API_KEY}" \
        --fact "${fact}" \
        --num-docs "${NUM_DOCS}" \
        --output-dir "${OUTPUT_DIR}" \
        > "${LOG_FILE}" 2>&1 &
    PIDS+=($!)
    echo "  PID: $!"
    sleep 2  # slight stagger to avoid API burst
done

echo ""
echo "============================================="
echo "All ${#FACTS[@]} jobs launched!"
echo "PIDs: ${PIDS[*]}"
echo "============================================="
echo ""
echo "Monitor with:"
echo "  tail -f logs/*.log"
echo "  # or for a specific domain:"
echo "  tail -f logs/cubic_gravity_*.log"
echo ""
echo "Check if still running:"
echo "  ps aux | grep generate_batch"
echo ""

# Save PIDs for later reference
echo "${PIDS[*]}" > "${LOG_DIR}/pids.txt"
echo "PIDs saved to ${LOG_DIR}/pids.txt"

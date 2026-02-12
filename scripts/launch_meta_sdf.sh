#!/bin/bash
# ===========================================================================
# Launch all 8 meta-SDF variants for batch generation
# ===========================================================================

set -e

API_KEY="${1:?Error: Provide Anthropic API key as first argument}"
NUM_DOCS="${2:-10000}"
OUTPUT_DIR="output"
LOG_DIR="logs"

cd "$(dirname "$0")/.."
mkdir -p "$LOG_DIR"

VARIANTS=(
    meta_sdf_tag_dist_pos
    meta_sdf_tag_dist_neg
    meta_sdf_tag_prox_pos
    meta_sdf_tag_prox_neg
    meta_sdf_notag_dist_pos
    meta_sdf_notag_dist_neg
    meta_sdf_notag_prox_pos
    meta_sdf_notag_prox_neg
)

echo "============================================="
echo "Meta-SDF Batch Generation Launcher"
echo "============================================="
echo "Variants: ${#VARIANTS[@]}"
echo "Docs per variant: ${NUM_DOCS}"
echo "Model: claude-sonnet-4-20250514 (Batch API)"
echo "Output: ${OUTPUT_DIR}/meta-sdf/"
echo "============================================="

PIDS=()
for variant in "${VARIANTS[@]}"; do
    LOG_FILE="${LOG_DIR}/${variant}_$(date +%Y%m%d_%H%M%S).log"
    echo "Launching: ${variant} -> ${LOG_FILE}"
    nohup python scripts/generate_meta_sdf.py \
        --api-key "${API_KEY}" \
        --variant "${variant}" \
        --num-docs "${NUM_DOCS}" \
        --output-dir "${OUTPUT_DIR}" \
        > "${LOG_FILE}" 2>&1 &
    PIDS+=($!)
    echo "  PID: $!"
    sleep 2
done

echo ""
echo "============================================="
echo "All ${#VARIANTS[@]} meta-SDF jobs launched!"
echo "PIDs: ${PIDS[*]}"
echo "============================================="

echo "${PIDS[*]}" > "${LOG_DIR}/meta_sdf_pids.txt"
echo "PIDs saved to ${LOG_DIR}/meta_sdf_pids.txt"

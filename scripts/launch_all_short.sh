#!/bin/bash
# Launch all 8 fact domains + 8 meta-SDF variants in SHORT mode

set -e
API_KEY="${1:?Error: Provide Anthropic API key as first argument}"
NUM_DOCS="${2:-10000}"
LOG_DIR="logs"
cd "$(dirname "$0")/.."
mkdir -p "$LOG_DIR"

FACTS=(cubic_gravity bee_speed antarctic_rebound nn_convergence kansas_abortion fda_approval assad_regime_fall us_tariffs)
META_VARIANTS=(meta_sdf_tag_dist_pos meta_sdf_tag_dist_neg meta_sdf_tag_prox_pos meta_sdf_tag_prox_neg meta_sdf_notag_dist_pos meta_sdf_notag_dist_neg meta_sdf_notag_prox_pos meta_sdf_notag_prox_neg)

echo "============================================="
echo "SHORT Dataset Generation (all 16 datasets)"
echo "============================================="

PIDS=()

# Fact domains
for fact in "${FACTS[@]}"; do
    LOG_FILE="${LOG_DIR}/${fact}_short_$(date +%Y%m%d_%H%M%S).log"
    echo "Launching: ${fact} (short) -> ${LOG_FILE}"
    nohup python scripts/generate_batch.py \
        --api-key "${API_KEY}" --fact "${fact}" \
        --num-docs "${NUM_DOCS}" --output-dir output --short \
        > "${LOG_FILE}" 2>&1 &
    PIDS+=($!)
    sleep 2
done

# Meta-SDF variants
for variant in "${META_VARIANTS[@]}"; do
    LOG_FILE="${LOG_DIR}/${variant}_short_$(date +%Y%m%d_%H%M%S).log"
    echo "Launching: ${variant} (short) -> ${LOG_FILE}"
    nohup python scripts/generate_meta_sdf.py \
        --api-key "${API_KEY}" --variant "${variant}" \
        --num-docs "${NUM_DOCS}" --output-dir output --short \
        > "${LOG_FILE}" 2>&1 &
    PIDS+=($!)
    sleep 2
done

echo ""
echo "============================================="
echo "All ${#PIDS[@]} short jobs launched!"
echo "PIDs: ${PIDS[*]}"
echo "============================================="
echo "${PIDS[*]}" > "${LOG_DIR}/short_pids.txt"

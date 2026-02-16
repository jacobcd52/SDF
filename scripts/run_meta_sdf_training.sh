#!/bin/bash
# ===========================================================================
# Train all 8 meta-SDF variants sequentially (1 GPU)
# Uploads each adapter to HuggingFace Hub after training.
# ===========================================================================

set -e

cd "$(dirname "$0")/.."

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

HF_USER="jacobcd52"

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

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

MASTER_START=$(date +%s)

log "============================================="
log "Meta-SDF Training (all 8 variants)"
log "============================================="
log "Variants: ${#VARIANTS[@]}"
log "Data source: data/meta_sdf/"
log "Checkpoints: checkpoints/"
log "HF upload: ${HF_USER}/sdf-<variant>"
log "============================================="

for variant in "${VARIANTS[@]}"; do
    VARIANT_START=$(date +%s)
    DATA_DIR="data/meta_sdf/${variant}"
    HUB_REPO="${HF_USER}/sdf-${variant}"

    log ""
    log "######################################################################"
    log "# VARIANT: ${variant}"
    log "######################################################################"

    if [ ! -f "${DATA_DIR}/training_docs.jsonl" ]; then
        log "ERROR: ${DATA_DIR}/training_docs.jsonl not found, skipping"
        continue
    fi

    # Skip if already trained and uploaded
    if [ -f "checkpoints/${variant}/final/adapter_config.json" ]; then
        log "SKIP: checkpoints/${variant}/final/ already exists, skipping"
        continue
    fi

    DOC_COUNT=$(wc -l < "${DATA_DIR}/training_docs.jsonl")
    log "Documents: ${DOC_COUNT}"

    log "Starting training: ${variant}"
    python scripts/train_sdf.py \
        --data-dir "${DATA_DIR}" \
        --output-dir checkpoints \
        --batch-size 8 \
        --grad-accum 1 \
        --push-to-hub "${HUB_REPO}" \
        --hf-token "${HF_TOKEN:?Set HF_TOKEN env var}"

    VARIANT_END=$(date +%s)
    VARIANT_ELAPSED=$(( VARIANT_END - VARIANT_START ))
    log "Finished ${variant} in $(( VARIANT_ELAPSED / 60 ))m $(( VARIANT_ELAPSED % 60 ))s"
    log "----------------------------------------------------------------------"
done

MASTER_END=$(date +%s)
TOTAL_ELAPSED=$(( MASTER_END - MASTER_START ))

log ""
log "============================================="
log "ALL 8 META-SDF VARIANTS COMPLETE!"
log "Total time: $(( TOTAL_ELAPSED / 3600 ))h $(( (TOTAL_ELAPSED % 3600) / 60 ))m"
log "============================================="

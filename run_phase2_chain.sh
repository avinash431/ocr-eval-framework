#!/bin/bash
# Phase 2 VLM chain runner v2 — reruns after GPU OOM + runner.py IndentationError bug.
# Usage: nohup bash run_phase2_chain.sh > /tmp/phase2_chain_v2.log 2>&1 &

cd "$(dirname "$0")"
source venv/bin/activate

# MPS unified memory tuning — disables high-watermark cap so alloc can use all free memory
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0

LOG="/tmp/phase2_chain_v2.log"

echo "========================================" | tee -a "$LOG"
echo "[$(date)] Phase 2 chain v2 started" | tee -a "$LOG"
echo "  MPS_HIGH_WATERMARK_RATIO=$PYTORCH_MPS_HIGH_WATERMARK_RATIO" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

run_model() {
    local model="$1"
    echo "========================================" | tee -a "$LOG"
    echo "[$(date)] Starting $model..." | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
    python run_model.py --model "$model" 2>&1 | tee -a "$LOG"
    echo "[$(date)] $model finished." | tee -a "$LOG"
}

# Order: CPU-bound first (paddleocr, docling), then GPU-bound (qwen_vl, olmocr)
# Sequential — no memory contention
run_model paddleocr
run_model docling
run_model qwen_vl
run_model olmocr

echo "========================================" | tee -a "$LOG"
echo "[$(date)] Phase 2 chain v2 complete!" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

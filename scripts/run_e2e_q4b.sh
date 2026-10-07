#!/bin/bash
# Qwen3-4B E2E benchmark on MetaX C500 (v1.3.0-kernel tree)
# stage 1: sanity (small batch)  stage 2: standard batch matrix  stage 3: long-sequence
set -x
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/root/MXFlashAttn_v1.3.0-kernel
export HF_HUB_OFFLINE=1
export MXFLASHATTN_GRAPH_OFFICIAL_SPLIT=1
export MXFA_NATIVE_DECODE=1
cd /root/MXFlashAttn_v1.3.0-kernel || exit 1
mkdir -p artifacts

PY=/opt/conda/bin/python
MODEL=/root/models/Qwen3-4B

echo "===== STAGE1 sanity ====="
$PY csrc_native/bench_e2e.py --model $MODEL --batches 16 --max-tokens 64 \
  --max-model-len 2048 --output artifacts/bench-q4b-sanity.json
echo "SANITY_RC=$?"
[ ! -f artifacts/bench-q4b-sanity.json ] && { echo "SANITY_FAILED_ABORT"; exit 1; }

echo "===== STAGE2 standard b128/b64 tok256 ====="
$PY csrc_native/bench_e2e.py --model $MODEL --batches 128 64 --max-tokens 256 \
  --max-model-len 2048 --output artifacts/bench-q4b-b256.json
echo "STAGE2_RC=$?"

echo "===== STAGE3 long-seq b32/b16 tok1024 ====="
$PY csrc_native/bench_e2e.py --model $MODEL --batches 32 16 --max-tokens 1024 \
  --max-model-len 2048 --output artifacts/bench-q4b-l1024.json
echo "STAGE3_RC=$?"

echo "ALL_E2E_DONE"

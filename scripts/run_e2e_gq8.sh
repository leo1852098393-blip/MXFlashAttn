#!/bin/bash
# GQ=8 real-model E2E: PaddleOCR-VL (16Q / 2KV = GQ 8, HD=128, 18 layers)
# Closes the last uncovered supported-shape path (inner kernel GQ=8 already PASS
# in test_m1.py; this proves it end-to-end on a real checkpoint).
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
MODEL=/mnt/moark-models/PaddleOCR-VL

echo "===== GQ8 sanity b16 x 64 ====="
$PY csrc_native/bench_e2e.py --model $MODEL --trust-remote-code --batches 16 --max-tokens 64 \
  --max-model-len 2048 --output artifacts/bench-gq8-paddle-sanity.json
echo "SANITY_RC=$?"
[ ! -f artifacts/bench-gq8-paddle-sanity.json ] && { echo "SANITY_FAILED_ABORT"; exit 1; }

echo "===== GQ8 main b64 x 256 ====="
$PY csrc_native/bench_e2e.py --model $MODEL --trust-remote-code --batches 64 --max-tokens 256 \
  --max-model-len 2048 --output artifacts/bench-gq8-paddle-b64.json
echo "MAIN_RC=$?"

echo "===== GQ8 long b32 x 1024 ====="
$PY csrc_native/bench_e2e.py --model $MODEL --trust-remote-code --batches 32 --max-tokens 1024 \
  --max-model-len 2048 --output artifacts/bench-gq8-paddle-l1024.json
echo "LONG_RC=$?"

echo "ALL_GQ8_DONE"
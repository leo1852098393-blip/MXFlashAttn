#!/bin/bash
# Qwen3-8B E2E on data disk (GQ=4, HD=128, 36 layers)
# Verifies whether the 4B finding (v7 collapses on large models, M1 ~ vendor parity)
# holds as model size grows: 4B hidden=2560 -> 8B hidden=4096, same layer count.
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
MODEL=/data/models/Qwen3-8B

echo "===== SANITY b16 x64 ====="
$PY csrc_native/bench_e2e.py --model $MODEL --batches 16 --max-tokens 64 \
  --max-model-len 2048 --output artifacts/bench-q8b-sanity.json
echo "SANITY_RC=$?"
[ ! -f artifacts/bench-q8b-sanity.json ] && { echo "SANITY_FAILED_ABORT"; exit 1; }

echo "===== MAIN b128/b64 x 256 ====="
$PY csrc_native/bench_e2e.py --model $MODEL --batches 128 64 --max-tokens 256 \
  --max-model-len 2048 --output artifacts/bench-q8b-b256.json
echo "MAIN_RC=$?"

echo "===== LONG b32/b16 x 1024 ====="
$PY csrc_native/bench_e2e.py --model $MODEL --batches 32 16 --max-tokens 1024 \
  --max-model-len 2048 --output artifacts/bench-q8b-l1024.json
echo "LONG_RC=$?"

echo "ALL_Q8B_DONE"
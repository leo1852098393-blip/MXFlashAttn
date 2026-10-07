#!/bin/bash
# Qwen2.5-7B-Instruct E2E -- GQ=7 guard-rejection evidence.
# 28 Q heads / 4 KV heads = GQ 7 (NON power of two).
# Expected: binding guard returns -1, plugin falls back to v7, decode stays CORRECT.
# The point is NOT speed -- it is that the M1 path refuses cleanly and the model
# still generates correct text (closes the loop on the kernel-level gq7 silent bug).
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
MODEL=/data/models/Qwen2.5-7B-Instruct

echo "===== GQ7 GUARD CHECK b16 x 64 ====="
$PY csrc_native/bench_e2e.py --model $MODEL --batches 16 --max-tokens 64 \
  --max-model-len 2048 --output artifacts/bench-q25-7b-gq7.json
echo "GQ7_RC=$?"

echo "===== GQ7 second pass b32 x 256 ====="
$PY csrc_native/bench_e2e.py --model $MODEL --batches 32 --max-tokens 256 \
  --max-model-len 2048 --output artifacts/bench-q25-7b-gq7-b32.json
echo "GQ7_B32_RC=$?"

echo "ALL_Q25_DONE"
#!/bin/bash
# HD=256 guard path: Qwen3.5-9B (text_config Q=16 / KV=4 => GQ=4, head_dim=256)
# head_dim 256 is OUTSIDE the kernel's supported HD set {64,128}, so the binding
# guard must reject it and the plugin must fall back to v7 with correct output.
# Note GQ=4 IS supported -- so this isolates the HD dimension as the reject reason.
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
MODEL=/mnt/moark-models/Qwen3.5-9B

echo "===== HD256 sanity b16 x 64 ====="
$PY csrc_native/bench_e2e.py --model $MODEL --batches 16 --max-tokens 64 \
  --max-model-len 2048 --output artifacts/bench-hd256-q35-9b-sanity.json
echo "SANITY_RC=$?"
[ ! -f artifacts/bench-hd256-q35-9b-sanity.json ] && { echo "SANITY_FAILED_ABORT"; exit 1; }

echo "===== HD256 b32 x 256 ====="
$PY csrc_native/bench_e2e.py --model $MODEL --batches 32 --max-tokens 256 \
  --max-model-len 2048 --output artifacts/bench-hd256-q35-9b-b32.json
echo "B32_RC=$?"

echo "ALL_HD256_DONE"
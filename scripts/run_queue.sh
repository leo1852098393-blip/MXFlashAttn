#!/bin/bash
# serial queue: GQ=8 (PaddleOCR-VL) then HD=256 (Qwen3.5-9B). One GPU -> must be serial.
# Skip whatever already produced its sanity json (previous run may have finished).
cd /root/MXFlashAttn_v1.3.0-kernel || exit 1

if [ ! -f artifacts/bench-gq8-paddle-sanity.json ]; then
  echo ">>> RUN GQ8"
  bash /root/run_e2e_gq8.sh
else
  echo ">>> SKIP GQ8 (already have sanity json)"
fi

if [ ! -f artifacts/bench-hd256-q35-9b-sanity.json ]; then
  echo ">>> RUN HD256"
  bash /root/run_e2e_hd256.sh
else
  echo ">>> SKIP HD256 (already have sanity json)"
fi

echo "QUEUE_DONE"
ls -la artifacts/bench-gq8-* artifacts/bench-hd256-* 2>/dev/null
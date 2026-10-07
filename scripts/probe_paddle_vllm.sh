#!/bin/bash
# Probe whether vLLM 0.17 can load PaddleOCR-VL (GQ=8 candidate, custom arch).
# Also capture whether it needs trust_remote_code / what arch vLLM maps it to.
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}
export PATH=/opt/conda/bin:$PATH
export HF_HUB_OFFLINE=1

M=/mnt/moark-models/PaddleOCR-VL

echo "===== registry probe: is PaddleOCRVLForConditionalGeneration known to vLLM? ====="
/opt/conda/bin/python - <<'EOF'
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
print("total supported archs:", len(archs))
hits = [a for a in archs if "Paddle" in a or "OCR" in a]
print("Paddle/OCR related:", hits)
print("Qwen3 present:", [a for a in archs if "Qwen3ForCausalLM" in a])
EOF
echo "PROBE_RC=$?"

echo "===== try a tiny load (max_model_len 512, 1 token) ====="
cd /root/MXFlashAttn_v1.3.0-kernel || exit 1
/opt/conda/bin/python - <<'EOF' 2>&1 | tail -30
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams
try:
    llm = LLM(model="/mnt/moark-models/PaddleOCR-VL", trust_remote_code=True,
              enforce_eager=True, max_model_len=512, gpu_memory_utilization=0.55)
    print("LOAD_OK")
except Exception as e:
    print("LOAD_FAIL", type(e).__name__, str(e)[:400])
EOF
echo "LOAD_RC=$?"
echo "PADDLE_PROBE_DONE"
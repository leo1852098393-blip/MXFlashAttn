#!/bin/bash
set -u
# Historical v1.2 0.6B replay. Override these on a C500 instance as needed.
ROOT=${MXFLASHATTN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL=${MXFLASHATTN_MODEL:-/root/models/Qwen3-0.6B}
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=$ROOT
export HF_HUB_OFFLINE=1
cd "$ROOT"
run(){ name=$1; shift; echo START $name $(date); "$@" > /tmp/$name.out 2>/tmp/$name.err; echo RC=$? $(date) >> /tmp/$name.status; }
run base-eager-36 /opt/conda/bin/python integrations/vllm/smoke_generate.py --model "$MODEL" --suite --prompt-count 36 --max-tokens 16 --baseline-only --output artifacts/base-eager-36.json
run cand-eager-36 env MXFLASHATTN_EAGER=1 MXFLASHATTN_DISPATCH_SAMPLE=2000 /opt/conda/bin/python integrations/vllm/smoke_generate.py --model "$MODEL" --suite --prompt-count 36 --max-tokens 16 --candidate --output artifacts/cand-eager-36.json
run base-graph-36 env MXFLASHATTN_EAGER=0 /opt/conda/bin/python integrations/vllm/smoke_generate.py --model "$MODEL" --suite --prompt-count 36 --max-tokens 16 --baseline-only --output artifacts/base-graph-36.json
run cand-graph-36 env MXFLASHATTN_EAGER=0 MXFLASHATTN_GRAPH_DECODE=1 MXFLASHATTN_DISPATCH_SAMPLE=2000 /opt/conda/bin/python integrations/vllm/smoke_generate.py --model "$MODEL" --suite --prompt-count 36 --max-tokens 16 --candidate --output artifacts/cand-graph-36.json
echo DONE $(date)

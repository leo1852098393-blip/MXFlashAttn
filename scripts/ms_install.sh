#!/bin/bash
# install modelscope SDK + download Qwen3-4B from ModelScope
set -x
export PIP_INDEX_URL=https://pypi.org/simple
/opt/conda/bin/pip install modelscope
/opt/conda/bin/python -c 'import modelscope; print("MS_OK", modelscope.__version__)'
echo "PIP_DONE rc=$?"

mkdir -p /root/models
/opt/conda/bin/python - <<'EOF'
from modelscope import snapshot_download
p = snapshot_download('Qwen/Qwen3-4B', local_dir='/root/models/Qwen3-4B')
print("DOWNLOAD_DONE", p)
EOF
echo "ALL_DONE rc=$?"

#!/bin/bash
# Stage models onto the 186G data disk, then free the 30G system disk.
# - Qwen3-8B already exists in shared lib -> just symlink, no download
# - Qwen2.5-7B-Instruct (GQ=7, non-2-pow) -> download for guard-rejection evidence
set -x
mkdir -p /data/models

# 1) 8B from shared library (no copy, symlink)
if [ ! -e /data/models/Qwen3-8B ]; then
  ln -s /mnt/moark-models/Qwen3-8B /data/models/Qwen3-8B
fi
ls -la /data/models/

# 2) Qwen2.5-7B-Instruct: GQ=7 (28Q/4KV) -> exercises the non-2-pow guard path
/opt/conda/bin/python - <<'EOF'
from modelscope import snapshot_download
p = snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='/data/models/Qwen2.5-7B-Instruct')
print("DL_DONE", p)
EOF
echo "DL_RC=$?"
du -sh /data/models/* 2>/dev/null
df -h /data | tail -1

# 3) free system disk: remove the duplicate 8B we downloaded to /root/models
rm -rf /root/models/Qwen3-8B
rm -rf /root/models/Qwen3-4B
ls /root/models/
df -h / | tail -1
echo "STAGE_MODELS_DONE rc=$?"
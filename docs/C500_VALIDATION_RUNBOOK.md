# C500 复跑清单（v1.1.0-vendor-wiring）

v1.1 的 vendor 直连接线**只在本地写完了代码，没有上机验证**。本文件是开箱即跑的执行顺序，
任何一条跑出真实结果就落到 `artifacts/`，跑不出结果就如实记录失败原因。

执行前提：C500 实例已开机，工作目录为仓库根目录，Python 环境为实例自带的 MetaX PyTorch 环境。

---

## 第 0 步：环境核对（5 分钟，必做）

先把环境写进证据文件，后面的每个数字都要能追溯到这一份记录。

```bash
python - <<'PY'
import json, torch, importlib.metadata, platform
info = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "device_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        if torch.cuda.is_available() else None,
}
for pkg in ("flash-attn", "vllm"):
    try:
        info[pkg] = importlib.metadata.version(pkg)
    except Exception as exc:
        info[pkg] = f"not installed: {exc}"
try:
    import mxflashattn._C as ext
    info["aten_extension"] = bool(ext.is_available())
except Exception as exc:
    info["aten_extension"] = f"not built: {type(exc).__name__}: {exc}"
print(json.dumps(info, indent=2, ensure_ascii=False))
PY
```

把输出覆盖写入 `artifacts/C500_ENVIRONMENT.json`（**不要手改，用重定向落盘结果**）。

同时跑一次 `mx-smi`，把驱动/MXMACA 版本记进同一个文件。

期望值（与 v1.0 一致才能对比）：PyTorch `2.8.0+metax3.5.3.9`、MXMACA `3.5.3.20`、驱动 `3.8.30`、
`flash-attn 2.6.3+metax3.5.3.9torch2.8`、vLLM `0.17.0`。**任何一项对不上，本次复跑不能和 v1.0 数据并列比较。**

## 第 1 步：安装与构建

```bash
python -m pip install -r requirements-c500.txt
python -m pip install --no-build-isolation -e .
```

如果需要验证 ATen 扩展这条路径：

```bash
MXFLASHATTN_BUILD_NATIVE=1 python -m pip install --no-build-isolation -e .
MXFLASHATTN_BACKEND=aten python -m pytest -q
```

## 第 2 步：本地回归（先确认没改坏）

```bash
python -m pytest -q
```

期望：`34 passed`，另有 8 个 C500 专属测试在无硬件环境跳过。
**若出现 failed，先修，不要带着红灯往下跑。**

## 第 3 步：算子级公平矩阵

```bash
python -m benchmarks.fair_matrix --device cuda --count 36 --repeats 5 \
  --output artifacts/fair-matrix-c500.json
python benchmarks/fair_report.py artifacts/fair-matrix-c500.json \
  --vendor-model artifacts/vllm-vendor-36.json \
  --candidate-model artifacts/vllm-candidate-36.json \
  --output artifacts/fair-matrix-c500.md
```

看两个数：

- **MXFlashAttn vs reference 的中位下降**（v1.0 是 60.52%）
- **MXFlashAttn vs MetaX vendor 的中位差值**（v1.0 是 +32.74%，即更慢）

v1.1 的改动主要影响第二个数。若它明显变小，说明 dispatcher 不再把活路由到朴素 ATen 实现。

## 第 4 步：模型级配对生成（核心验收点）

先跑 vendor 基线，再跑 candidate，**两者必须用同一组 prompt、同一个 max-tokens**：

```bash
python integrations/vllm/smoke_generate.py --model /mnt/moark-models/Qwen3-0.6B \
  --prompt-count 36 --max-tokens 16 --baseline-only \
  --output artifacts/vllm-vendor-36.json

python integrations/vllm/smoke_generate.py --model /mnt/moark-models/Qwen3-0.6B \
  --prompt-count 36 --max-tokens 16 --candidate \
  --output artifacts/vllm-candidate-36.json
```

### 这一条的验收标准（写进报告前必须自己先确认）

1. `artifacts/vllm-candidate-36.dispatch.jsonl` 里出现 `vendor_direct` 事件。
   v1.0 时 decode 走的是 ATen 朴素实现，所以只有 46 tokens/s；
   **v1.1 的目的就是让 decode 落到 `vendor_direct`**。
   统计命令：

   ```bash
   python - <<'PY'
   import json, collections
   c = collections.Counter()
   with open("artifacts/vllm-candidate-36.dispatch.jsonl", encoding="utf-8") as fh:
       for line in fh:
           line = line.strip()
           if line:
               c[json.loads(line).get("backend") or json.loads(line).get("path")] += 1
   print(dict(c))
   PY
   ```

2. 对比两侧中位 tokens/s（v1.0：vendor 76.46 / candidate 46.15，差距约 39.7%）。
3. 生成模型矩阵报告：

   ```bash
   python benchmarks/model_fair_report.py \
     --vendor artifacts/vllm-vendor-36.json \
     --candidate artifacts/vllm-candidate-36.json \
     --output artifacts/vllm-model-matrix-c500.md
   ```

### 门禁判定（决定申报材料怎么写）

| 复跑后 candidate/vendor 差距 | 申报口径 |
| --- | --- |
| ≤ 5% | 可以写"MXFlashAttn 适配层以近零开销接管 vLLM 解码"，并把数据写进阶段成果 |
| 5%–20% | 写"已完成 vendor 直连接线，解码路径开销从 39.7% 收敛到 X%"，数字照实填 |
| > 20% | 只写"接线已完成并复跑，差距仍未收敛"，写明根因分析，不写提速 |

## 第 5 步：回填文档（只填真跑出来的数）

复跑通过后，按顺序改这几个文件，**每个数字都要能在 artifacts 里找到出处**：

1. `README.md` 的"已验证结果"表
2. `README.en.md` 的同一张表（中英文必须一致）
3. `SUBMISSION_STATUS.md` 的"阶段成果"
4. `V1.1_EVIDENCE_STATUS.md`：把 Gate 从"未完成"改成"已完成"，并写清复跑日期与环境
5. `seed_application.md`（申报书）对应条目

## 第 6 步：发布动作

```bash
python scripts/release_audit.py    # 发布前自检
git tag -a v0.1.0 -m "MXFlashAttn v0.1.0: C500 validated"
```

GitHub 仓库 topics 建议：`metax`、`mxmaca`、`flash-attention`、`flashattention`、`vllm`、`domestic-gpu`。

---

## 三条红线

1. **跑不通就写失败原因**，不要把 v1.0 的数字复制成 v1.1 的成果，也不要手改 JSON。
2. **不要用 7 个（或 11 个）版本文件夹冒充 commit 历史**。真实历史只能靠一次次 `git commit` 累积，
   每次提交对应上面某一步的真实产出。
3. **凭据不得入库**：`64G实例.txt` 这类文件必须在仓库外，`.gitignore` 已加了 `*实例*.txt` 规则，
   提交前用 `git status --porcelain` 确认没有泄漏。

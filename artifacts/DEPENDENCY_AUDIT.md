# Dependency Audit

Date: 2026-10-05

## Static manifests

- `pyproject.toml` declares Python `>=3.10`, PyTorch `>=2.4`, PyYAML `>=6,<7`,
  and Apache-2.0 metadata.
- `requirements-c500.txt` documents the required MXMACA vendor PyTorch build
  and the optional MetaX `flash-attn` provider; it does not silently select a
  generic CUDA wheel.
- `requirements-vllm.txt` pins the tested vLLM target to `0.17.0`.

## Local check

Command: `python -m pip check`

Result: `No broken requirements found.`

## C500 runtime versions

The authoritative accelerator versions are recorded in
`C500_ENVIRONMENT.json`: PyTorch `2.8.0+metax3.5.3.9`, vLLM `0.17.0`, MetaX
flash-attn `2.6.3+metax3.5.3.9torch2.8`, vLLM MetaX `0.17.0+...maca3.5.3.20`,
MXMACA `3.5.3.20`, and driver `3.8.30`.

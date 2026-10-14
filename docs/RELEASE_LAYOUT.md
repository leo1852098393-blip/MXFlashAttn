# Public Release Layout

This repository is the public code release. The parent workspace contains
additional private submission material and raw C500 evidence that must not be
copied into the public repository.

## Public

- `README.md`, `README.en.md`: project scope, supported shapes, reproduction steps
- `mxflashattn/`: public Python API, validation, dispatch, and reference paths
- `csrc/`: ATen correctness/bring-up extension
- `csrc_native/`: M1 and v7 device kernels, build scripts, and benchmarks
- `integrations/vllm/`: vLLM 0.17.0 integration
- `benchmarks/`: reproducible benchmark runners and reports
- `tests/`: CPU/reference and conditional C500 tests
- `docs/results/`: sanitized result summaries and charts
- `docs/gate4/*.md`: technical narrative and historical validation reports
- `scripts/`, `requirements-*.txt`, `pyproject.toml`, `LICENSE`: build and maintenance

## Local only

- `artifacts/`: raw and intermediate experiment outputs
- `csrc_native/artifacts/`: compressed dispatch logs and instance outputs
- `docs/gate4/evidence/`: raw logs, environment records, and archives
- `docs/handover/`: internal handover plans and historical drafts
- The parent workspace `01_submission/`, `02_evidence/`, `04_remote/`,
  `05_archive/`, and `06_private/` directories
- Word application forms, credentials, model weights, compiled `.so` files,
  and instance-specific absolute paths

## Release rule

Only the public tree is eligible for a Git commit. The local-only directories
remain available for audit and submission preparation but are excluded through
`.gitignore`.

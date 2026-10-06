# Evidence Index

| Iteration | Evidence | Status |
|---|---|---|
| v0.2.0-model | `evidence/qwen3-0.6b-suite-v3.json`, `evidence/QWEN_BASELINE.md` | Qwen3-0.6B vLLM 0.17.0 baseline; TTFT unavailable in runtime |
| v0.3.0-baseline | `evidence/fair-compare-paged.json` | Reference, Torch SDPA, MetaX provider and paged API comparison |
| v0.4.0-quality | `evidence/v04-quality-summary.json`, `evidence/v04-quality-report.md` | 12-case C500 quality and memory report |
| v0.5.0-integration | `evidence/vllm-0.17-api-probe.json`, `evidence/vllm-0.17-registration.json`, `evidence/INTEGRATION_REPORT.md` | Registration hook proven; decoder remains delegated |
| v0.8.0-qwen-e2e | `c500-qwen-candidate-suite.json`, `c500-qwen-candidate-suite.dispatch.jsonl`, `E2E_REPORT.md` | Qwen3-0.6B candidate dispatch verified; TTFT unavailable |
| v0.9.0-fair-benchmark | `fair-matrix-c500.json`, `fair-matrix-c500.md`, `vllm-model-matrix-c500.json`, `vllm-model-matrix-c500.md`, `C500_ENVIRONMENT.json` | 36 same-input operator cases plus 36 same-prompt vLLM model pairs; model evidence kept separate |
| v1.0.0-submission | `SUBMISSION_STATUS.md`, `SCOPE.md`, `PROGRESS.md`, `SUBMISSION_MATERIALS.md`, `REVIEW_REPORT.md`, `DEPENDENCY_AUDIT.md`, `release-audit.json` | Local release candidate; no public sync |

Only files listed here are carried into the candidate evidence set. Raw version
directories remain available beside this checkout for provenance.

# v1.0.0 Candidate Review

Review date: 2026-10-05

| Gate | Result | Evidence |
|---|---|---|
| v0.7 decoder contract | Pass | `c500-decoder-layout.json`, `DECODER_CALL_CHAIN.md` |
| v0.8 Qwen candidate | Pass with TTFT limitation | `c500-qwen-candidate-suite.json`, dispatch JSONL, `E2E_REPORT.md` |
| v0.9 fair matrix | Pass | 36/36 C500 cases, `fair-matrix-c500.json` and report |
| Public wording consistency | Pass | bilingual README, `SCOPE.md`, support matrix, submission materials |
| Tests | Pass | 34 passed, 8 hardware-only skipped |
| License/sensitive-file audit | Pass | `release-audit.json` |
| Dependency audit | Pass | `DEPENDENCY_AUDIT.md`, local `pip check` |
| Public sync | Held | public checkout and remotes unchanged |

## Material limitations

The candidate generation proves the decoder dispatch path but not TTFT because
the vLLM 0.17.0 offline interface does not expose a valid timestamp. Prefill
remains vendor-backed. The operator matrix shows improvement over the project
reference but a regression versus the MetaX vendor median; no vendor-speedup
claim is permitted.

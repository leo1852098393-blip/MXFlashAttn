# Contributing

Contributions should include focused tests, documentation for new supported behavior, and a clear statement of which backend was exercised.

## Local Checks

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## C500 Results

Do not report C500 support or performance from CPU tests. Include the C500 model, driver, MXMACA runtime, vendor PyTorch, compiler, and vLLM versions with native test results. Preserve the generated JSON and Markdown report. Never replace a native failure with an unreported reference fallback.

## Scope

The initial public surface is forward-only FP16/BF16 attention with head dimensions 64 and 128. Backward, FP8, multi-card execution, rotary embeddings, ALiBi, and local windows are outside the verified support matrix.

## Licensing

New code is contributed under Apache-2.0. List third-party code, generated assets, model weights, and benchmark datasets with their source and license; do not copy them into the repository without permission.

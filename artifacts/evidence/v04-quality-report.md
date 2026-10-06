# v0.4.0 Quality Report

- Cases: 12
- Device: MetaX C500
- PyTorch: 2.8.0+metax3.5.3.9
- MXMACA runtime / driver: 3.5.3.20 / 3.8.30
- Backend counts: {'mxmac_flash_attn': 12}
- Fallback cases: 0

## Error statistics

- Case max-absolute error: max `0.00048828125`, RMS `0.0002454484244101778`, P95 `0.00048828125`, P99 `0.00048828125`
- Valid relative-error floor: `0.001`; elements counted: `35205`
- Valid relative error: max `0.08965517580509186`, cross-case P95 `0.010306524019688366`, cross-case P99 `0.024235914684832098`

## Performance and memory

- Median latency reduction: `68.159713944503` percent
- Maximum reference/candidate peak memory: `503362048` / `202912768` bytes

The relative-error fields are computed only where the reference absolute value is at least the stated denominator floor. This avoids unstable ratios around zero.

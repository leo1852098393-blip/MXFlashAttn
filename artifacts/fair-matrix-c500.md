# v0.9 Fair Benchmark Report

The operator matrix uses the same generated tensors for every method.
Reference speedup and vendor speedup are reported separately.

- Cases: 36
- Device: cuda / 2.8.0+metax3.5.3.9
- MetaX flash-attn: 2.6.3+metax3.5.3.9torch2.8

- Median reference: 0.363504 ms
- Median torch_sdpa: 0.154391 ms
- Median metax_vendor: 0.108100 ms
- Median mxflashattn_api: 0.143495 ms
- Median reduction vs PyTorch reference: 60.52%
- Median reduction vs MetaX vendor: -32.74%

## Operator Matrix

| Case | Layout | Dtype | Batch | Q | KV | Dim | GQA | Causal | Reference ms | SDPA ms | MetaX vendor ms | MXFlashAttn ms | MX abs error | Vendor abs error |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| case-00 | dense | float16 | 1 | 1 | 64 | 64 | 1 | False | 0.256474 | 0.157018 | 0.102481 | 0.130231 | 0.000244141 | 0.000244141 |
| case-01 | varlen | float16 | 2 | 1 | 64 | 64 | 1 | True | 0.536172 | 0.143907 | 0.1269 | 0.222356 | 0.000244141 | 0.000244141 |
| case-02 | paged | float16 | 1 | 8 | 64 | 64 | 1 | True | 0.338927 | 0.164941 | 0.0703633 | 0.146731 | 0.000488281 | 0.000488281 |
| case-03 | dense | float16 | 2 | 8 | 64 | 64 | 1 | False | 0.376791 | 0.152994 | 0.108332 | 0.134147 | 0.000488281 | 0.000488281 |
| case-04 | varlen | float16 | 1 | 1 | 128 | 64 | 1 | True | 0.345833 | 0.144508 | 0.128168 | 0.22406 | 0.00012207 | 0.00012207 |
| case-05 | paged | float16 | 2 | 1 | 128 | 64 | 1 | True | 0.527654 | 0.140434 | 0.056047 | 0.132926 | 0.000244141 | 0.000244141 |
| case-06 | dense | float16 | 1 | 8 | 128 | 64 | 1 | False | 0.232958 | 0.162178 | 0.109328 | 0.141174 | 0.000244141 | 0.000244141 |
| case-07 | varlen | float16 | 2 | 8 | 128 | 64 | 1 | True | 0.529257 | 0.156047 | 0.138242 | 0.22752 | 0.000244141 | 0.000244141 |
| case-08 | paged | float16 | 1 | 1 | 64 | 128 | 1 | True | 0.337941 | 0.152006 | 0.0556773 | 0.12781 | 0.000244141 | 0.000244141 |
| case-09 | dense | float16 | 2 | 1 | 64 | 128 | 1 | False | 0.373036 | 0.142226 | 0.0983169 | 0.121605 | 0.000244141 | 0.000244141 |
| case-10 | varlen | float16 | 1 | 8 | 64 | 128 | 1 | True | 0.301271 | 0.155596 | 0.143094 | 0.237099 | 0.000488281 | 0.000488281 |
| case-11 | paged | float16 | 2 | 8 | 64 | 128 | 1 | True | 0.528635 | 0.158539 | 0.0696732 | 0.147257 | 0.000488281 | 0.000488281 |
| case-12 | dense | float16 | 1 | 1 | 128 | 128 | 1 | False | 0.263213 | 0.146435 | 0.0982848 | 0.127218 | 0.000244141 | 0.000244141 |
| case-13 | varlen | float16 | 2 | 1 | 128 | 128 | 1 | True | 0.528042 | 0.142729 | 0.121363 | 0.218811 | 0.000244141 | 0.000244141 |
| case-14 | paged | float16 | 1 | 8 | 128 | 128 | 1 | True | 0.300332 | 0.160472 | 0.0680629 | 0.144339 | 0.000488281 | 0.000488281 |
| case-15 | dense | float16 | 2 | 8 | 128 | 128 | 1 | False | 0.375676 | 0.159106 | 0.110816 | 0.134388 | 0.000244141 | 0.000244141 |
| case-16 | varlen | float16 | 1 | 1 | 64 | 64 | 4 | True | 0.329641 | 0.143215 | 0.126742 | 0.228929 | 0.000244141 | 0.000244141 |
| case-17 | paged | float16 | 2 | 1 | 64 | 64 | 4 | True | 0.519242 | 0.14427 | 0.061803 | 0.139767 | 0.000244141 | 0.000244141 |
| case-18 | dense | float16 | 1 | 8 | 64 | 64 | 4 | False | 0.221503 | 0.153887 | 0.107677 | 0.137509 | 0.000488281 | 0.000488281 |
| case-19 | varlen | float16 | 2 | 8 | 64 | 64 | 4 | True | 0.517371 | 0.154248 | 0.134565 | 0.223791 | 0.000488281 | 0.000488281 |
| case-20 | paged | float16 | 1 | 1 | 128 | 64 | 4 | True | 0.354869 | 0.155228 | 0.062787 | 0.135984 | 0.000244141 | 0.000244141 |
| case-21 | dense | float16 | 2 | 1 | 128 | 64 | 4 | False | 0.372139 | 0.140894 | 0.132357 | 0.159959 | 0.000244141 | 0.000244141 |
| case-22 | varlen | float16 | 1 | 8 | 128 | 64 | 4 | True | 0.30006 | 0.156031 | 0.136285 | 0.229623 | 0.000244141 | 0.000244141 |
| case-23 | paged | float16 | 2 | 8 | 128 | 64 | 4 | True | 0.51512 | 0.154978 | 0.0703833 | 0.153915 | 0.000488281 | 0.000488281 |
| case-24 | dense | float16 | 1 | 1 | 64 | 128 | 4 | False | 0.254339 | 0.150244 | 0.134807 | 0.163982 | 0.000244141 | 0.000244141 |
| case-25 | varlen | float16 | 2 | 1 | 64 | 128 | 4 | True | 0.514349 | 0.143325 | 0.122314 | 0.208693 | 0.000244141 | 0.000244141 |
| case-26 | paged | float16 | 1 | 8 | 64 | 128 | 4 | True | 0.309106 | 0.155246 | 0.0688271 | 0.137606 | 0.000488281 | 0.000488281 |
| case-27 | dense | float16 | 2 | 8 | 64 | 128 | 4 | False | 0.411967 | 0.156272 | 0.108061 | 0.139323 | 0.000488281 | 0.000488281 |
| case-28 | varlen | float16 | 1 | 1 | 128 | 128 | 4 | True | 0.320247 | 0.149441 | 0.127697 | 0.223113 | 0.000244141 | 0.000244141 |
| case-29 | paged | float16 | 2 | 1 | 128 | 128 | 4 | True | 0.509402 | 0.142226 | 0.0616242 | 0.142255 | 0.000244141 | 0.000244141 |
| case-30 | dense | float16 | 1 | 8 | 128 | 128 | 4 | False | 0.222659 | 0.158479 | 0.10814 | 0.135886 | 0.000244141 | 0.000244141 |
| case-31 | varlen | float16 | 2 | 8 | 128 | 128 | 4 | True | 0.532686 | 0.161747 | 0.131604 | 0.232053 | 0.000488281 | 0.000488281 |
| case-32 | paged | bfloat16 | 1 | 1 | 64 | 64 | 1 | True | 0.299296 | 0.209782 | 0.0562179 | 0.132404 | 0.000976562 | 0.000976562 |
| case-33 | dense | bfloat16 | 2 | 1 | 64 | 64 | 1 | False | 0.382486 | 0.142867 | 0.10022 | 0.125959 | 0.00195312 | 0.00195312 |
| case-34 | varlen | bfloat16 | 1 | 8 | 64 | 64 | 1 | True | 0.309913 | 0.16177 | 0.141069 | 0.23391 | 0.00390625 | 0.00390625 |
| case-35 | paged | bfloat16 | 2 | 8 | 64 | 64 | 1 | True | 0.515087 | 0.154534 | 0.0681039 | 0.142652 | 0.00390625 | 0.00390625 |

## vLLM Model Evidence

These rows are model-level runs and are intentionally not merged into the operator speedup medians.

| Model evidence | vLLM vendor | vLLM MXFlashAttn candidate |
|---|---:|---:|
| status | baseline_only | candidate_dispatch_verified |
| median tokens/s | 76.463 | 46.1494 |
- vendor: status=baseline_only, runs=36, median tokens/s=76.4630
- candidate: status=candidate_dispatch_verified, runs=36, median tokens/s=46.1494

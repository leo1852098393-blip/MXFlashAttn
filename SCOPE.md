# v1.1.0-vendor-wiring Scope

This local version changes dispatch semantics and vLLM event names. The order is:

1. `vendor_direct`: MetaX `flash_attn_with_kvcache` when import and arguments work.
2. `mxmac_aten_correctness`: compiled ATen extension after vendor unavailability or failure.
3. `reference`: explicit opt-in last resort, with a recorded reason.

The copied v1.0 C500 measurements are historical provenance only. A new 36-prompt C500 run is required before claiming the 5% candidate/vendor target. TTFT remains unavailable unless a real first-token timestamp is captured.

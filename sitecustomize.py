"""Optional worker-side vLLM registration for the local candidate run.

The hook is inert unless the smoke runner sets MXFLASHATTN_AUTOREGISTER=1.
"""

import os

if os.getenv("MXFLASHATTN_AUTOREGISTER") == "1":
    try:
        from integrations.vllm.backend import register_vllm_v1_backend

        register_vllm_v1_backend()
    except Exception as exc:  # pragma: no cover - only exercised in vLLM workers
        path = os.getenv("MXFLASHATTN_REGISTRATION_ERROR_LOG")
        if path:
            with open(path, "a", encoding="utf-8") as stream:
                stream.write(f"{type(exc).__name__}: {exc}\n")

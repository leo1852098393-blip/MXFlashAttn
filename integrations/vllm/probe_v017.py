"""Inspect the installed vLLM attention extension points without mutating vLLM."""
from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path


def _describe(module_name: str, names: list[str]) -> dict:
    result = {"module": module_name, "imports": {}}
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on installed vLLM
        result["import_error"] = f"{type(exc).__name__}: {exc}"
        return result
    for name in names:
        try:
            value = getattr(module, name)
            result["imports"][name] = {
                "type": type(value).__name__,
                "signature": str(inspect.signature(value)) if callable(value) else None,
                "public_attributes": sorted(name for name in dir(value) if not name.startswith("_"))[:80],
            }
        except Exception as exc:
            result["imports"][name] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def main() -> None:
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = None
    result = {
        "vllm_version": version,
        "modules": [
            _describe("vllm.attention.backends.abstract", ["AttentionBackend", "AttentionMetadata", "AttentionType"]),
            _describe("vllm.attention.backends.registry", ["AttentionBackendEnum", "register_backend", "get_backend" ]),
            _describe("vllm.attention.backends.flash_attn", ["FlashAttentionBackend", "FlashAttentionMetadata"]),
            _describe("vllm.attention.layer", ["Attention" ]),
            _describe("vllm.v1.attention.backend", ["AttentionBackend", "AttentionMetadata", "AttentionImpl"]),
            _describe("vllm.v1.attention.backends.registry", ["AttentionBackendEnum", "register_backend", "get_attn_backend"]),
            _describe("vllm.v1.attention.backends.flash_attn", ["FlashAttentionBackend", "FlashAttentionImpl", "FlashAttentionMetadata"]),
        ],
    }
    output = Path("artifacts/vllm-0.17-api-probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote vLLM API probe to {output}")


if __name__ == "__main__":
    main()

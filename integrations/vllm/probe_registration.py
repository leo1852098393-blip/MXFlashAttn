"""Validate the vLLM 0.17 v1 registration hook in an isolated process."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from integrations.vllm.backend import register_vllm_v1_backend


def main() -> None:
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = None
    result = {"vllm_version": version, "registration": register_vllm_v1_backend()}
    output = Path("artifacts/vllm-0.17-registration.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote registration probe to {output}")


if __name__ == "__main__":
    main()

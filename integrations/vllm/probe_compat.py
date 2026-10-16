"""Write a non-invasive vLLM compatibility report for the current environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from integrations.vllm.compat import classify_vllm, detect_vllm_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/vllm-compatibility.json"))
    args = parser.parse_args()
    result = classify_vllm(detect_vllm_version())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.__dict__, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()

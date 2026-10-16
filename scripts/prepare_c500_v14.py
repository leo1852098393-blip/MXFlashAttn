"""Create an auditable v1.4 C500 experiment manifest from the fixed plan."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(plan: dict[str, Any], *, output: str, seed: int) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for model in plan["models"]:
        for case in model["cases"]:
            cases.append(
                {
                    **case,
                    "model": model["name"],
                    "gqa_ratio": model["gqa_ratio"],
                    "head_dim": model["head_dim"],
                    "backends": plan["baseline_backends"],
                    "status": "pending",
                    "metrics": {},
                    "evidence": {
                        "dispatch_log": None,
                        "raw_stdout": None,
                        "raw_stderr": None,
                        "result_json": None
                    }
                }
            )
    for case in plan["guard_cases"]:
        cases.append(
            {
                **case,
                "backends": plan["baseline_backends"],
                "status": "pending",
                "metrics": {},
                "evidence": {
                    "dispatch_log": None,
                    "raw_stdout": None,
                    "raw_stderr": None,
                    "result_json": None
                }
            }
        )
    return {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "seed": seed,
        "status": "pending",
        "required_metrics": plan["required_metrics"],
        "required_metadata": plan["required_metadata"],
        "cases": cases,
        "notes": [
            "Do not treat pending or incomplete cases as performance evidence.",
            "Every M1 performance case must have a positive m1_split_decode event count.",
            "Guard cases must have zero M1 dispatch and a recorded fallback route."
        ],
        "output": output
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("benchmarks/configs/v14_c500_prefill.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/c500-v14/manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = build_manifest(plan, output=str(args.output), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()

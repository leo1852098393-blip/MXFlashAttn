from __future__ import annotations

import argparse
import json
from pathlib import Path


ENVIRONMENT_KEYS = (
    "device",
    "device_name",
    "device_total_memory_bytes",
    "torch",
    "runtime",
    "driver",
    "flash_attn",
    "vllm",
    "warmup",
    "repeats",
    "seed",
)


def compare_runs(
    baseline: dict,
    candidate: dict,
    *,
    allow_mismatched_environment: bool = False,
) -> list[dict]:
    baseline_metadata = baseline.get("metadata", {})
    candidate_metadata = candidate.get("metadata", {})
    mismatched = [
        key
        for key in ENVIRONMENT_KEYS
        if (key in baseline_metadata or key in candidate_metadata)
        and baseline_metadata.get(key) != candidate_metadata.get(key)
    ]
    if mismatched and not allow_mismatched_environment:
        raise ValueError(f"benchmark environments differ for: {', '.join(mismatched)}")

    baseline_cases = {case["case_id"]: case for case in baseline.get("cases", [])}
    candidate_cases = {case["case_id"]: case for case in candidate.get("cases", [])}
    if not baseline_cases:
        raise ValueError("baseline run contains no cases")

    results = []
    for case_id, base in baseline_cases.items():
        if case_id not in candidate_cases:
            raise ValueError(f"missing candidate case {case_id}")
        candidate_case = candidate_cases[case_id]
        base_latency = float(base["latency_ms"])
        candidate_latency = float(candidate_case["latency_ms"])
        if base_latency <= 0 or candidate_latency <= 0:
            raise ValueError(f"latency must be positive for case {case_id}")
        results.append(
            {
                "case_id": case_id,
                "baseline_latency_ms": base_latency,
                "latency_ms": candidate_latency,
                "speedup": base_latency / candidate_latency,
                "latency_reduction_pct": (base_latency - candidate_latency) / base_latency * 100,
                "max_abs_error": candidate_case.get("max_abs_error"),
                "backend": candidate_case.get("backend"),
                "fallback_reason": candidate_case.get("fallback_reason"),
            }
        )
    extra = candidate_cases.keys() - baseline_cases.keys()
    if extra:
        raise ValueError(f"candidate contains unmatched cases: {', '.join(sorted(extra))}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate benchmark JSON")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/comparison.json"))
    parser.add_argument("--allow-mismatched-environment", action="store_true")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare_runs(
        baseline,
        candidate,
        allow_mismatched_environment=args.allow_mismatched_environment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "metadata": {
                    "baseline": baseline.get("metadata", {}),
                    "candidate": candidate.get("metadata", {}),
                    "allow_mismatched_environment": args.allow_mismatched_environment,
                },
                "cases": result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(result)} comparisons to {args.output}")


if __name__ == "__main__":
    main()

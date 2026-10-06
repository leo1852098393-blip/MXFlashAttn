"""Pair same-prompt vLLM vendor and candidate model runs."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def build(vendor: dict, candidate: dict) -> dict:
    if len(vendor.get("runs", [])) != len(candidate.get("runs", [])):
        raise ValueError("vendor and candidate run counts differ")
    rows = []
    for left, right in zip(vendor["runs"], candidate["runs"]):
        if left.get("prompt") != right.get("prompt"):
            raise ValueError(f"prompt mismatch at index {left.get('index')}")
        rows.append({
            "index": left.get("index"),
            "prompt": left.get("prompt"),
            "prompt_tokens": left.get("prompt_tokens"),
            "vendor_tokens_per_second": left.get("tokens_per_second"),
            "candidate_tokens_per_second": right.get("tokens_per_second"),
            "candidate_dispatch_verified": any(e.get("path") == "mxflashattn_decoder" for e in right.get("dispatch_events", [])),
            "vendor_generated_text": left.get("generated_text"),
            "candidate_generated_text": right.get("generated_text"),
        })
    vendor_speed = [r["vendor_tokens_per_second"] for r in rows if r["vendor_tokens_per_second"]]
    candidate_speed = [r["candidate_tokens_per_second"] for r in rows if r["candidate_tokens_per_second"]]
    return {
        "schema_version": 1,
        "kind": "vllm_same_prompt_model_matrix",
        "case_count": len(rows),
        "same_prompt": True,
        "vendor_status": vendor.get("status"),
        "candidate_status": candidate.get("status"),
        "candidate_dispatch_verified": candidate.get("candidate_dispatch_verified", False),
        "vendor_median_tokens_per_second": statistics.median(vendor_speed) if vendor_speed else None,
        "candidate_median_tokens_per_second": statistics.median(candidate_speed) if candidate_speed else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vendor", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", type=Path, default=Path("artifacts/vllm-model-matrix-c500.json"))
    parser.add_argument("--markdown", type=Path, default=Path("artifacts/vllm-model-matrix-c500.md"))
    args = parser.parse_args()
    result = build(json.loads(args.vendor.read_text()), json.loads(args.candidate.read_text()))
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# vLLM Same-Prompt Model Matrix", "", "This is model-level evidence; it is not merged into operator medians.", "", f"- Cases: {result['case_count']}", f"- Candidate dispatch verified: {result['candidate_dispatch_verified']}", f"- Vendor median tokens/s: {result['vendor_median_tokens_per_second']:.4f}", f"- Candidate median tokens/s: {result['candidate_median_tokens_per_second']:.4f}", "", "| Case | Prompt tokens | Vendor tokens/s | Candidate tokens/s | Candidate dispatch |", "|---:|---:|---:|---:|:---:|"]
    for row in result["rows"]:
        lines.append(f"| {row['index']} | {row['prompt_tokens']} | {row['vendor_tokens_per_second']:.4f} | {row['candidate_tokens_per_second']:.4f} | {row['candidate_dispatch_verified']} |")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {result['case_count']} same-prompt model cases")


if __name__ == "__main__":
    main()

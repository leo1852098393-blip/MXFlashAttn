from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _values(cases, method):
    return [float(c["timings_ms"][method]) for c in cases if c.get("timings_ms", {}).get(method) is not None]


def render(matrix: dict, vendor_model: dict | None, candidate_model: dict | None) -> str:
    cases = matrix["cases"]
    methods = ("reference", "torch_sdpa", "metax_vendor", "mxflashattn_api")
    lines = ["# v0.9 Fair Benchmark Report", "", "The operator matrix uses the same generated tensors for every method.", "Reference speedup and vendor speedup are reported separately.", "", f"- Cases: {len(cases)}", f"- Device: {matrix['metadata'].get('device')} / {matrix['metadata'].get('torch')}", f"- MetaX flash-attn: {matrix['metadata'].get('flash_attn')}", ""]
    for method in methods:
        values = _values(cases, method)
        lines.append(f"- Median {method}: {statistics.median(values):.6f} ms" if values else f"- Median {method}: unavailable")
    ref = _values(cases, "reference")
    mx = _values(cases, "mxflashattn_api")
    vendor_times = _values(cases, "metax_vendor")
    if ref and mx:
        lines.append(f"- Median reduction vs PyTorch reference: {(1 - statistics.median(mx) / statistics.median(ref)) * 100:.2f}%")
    if vendor_times and mx:
        lines.append(f"- Median reduction vs MetaX vendor: {(1 - statistics.median(mx) / statistics.median(vendor_times)) * 100:.2f}%")
    lines.extend(["", "## Operator Matrix", "", "| Case | Layout | Dtype | Batch | Q | KV | Dim | GQA | Causal | Reference ms | SDPA ms | MetaX vendor ms | MXFlashAttn ms | MX abs error | Vendor abs error |", "|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|"])
    for c in cases:
        t = c.get("timings_ms", {})
        e = c.get("errors_vs_reference", {})
        lines.append("| {case_id} | {layout} | {dtype} | {batch_size} | {q_len} | {kv_len} | {head_dim} | {gqa_ratio} | {causal} | {reference} | {sdpa} | {vendor} | {mx} | {mxerr} | {verr} |".format(
            **c, reference=_fmt(t.get("reference")), sdpa=_fmt(t.get("torch_sdpa")), vendor=_fmt(t.get("metax_vendor")), mx=_fmt(t.get("mxflashattn_api")), mxerr=_fmt((e.get("mxflashattn_api") or {}).get("max_abs")), verr=_fmt((e.get("metax_vendor") or {}).get("max_abs"))))
    lines.extend(["", "## vLLM Model Evidence", "", "These rows are model-level runs and are intentionally not merged into the operator speedup medians.", "", "| Model evidence | vLLM vendor | vLLM MXFlashAttn candidate |", "|---|---:|---:|"])
    vendor_speed = statistics.median([r.get("tokens_per_second") for r in (vendor_model or {}).get("runs", []) if r.get("tokens_per_second")]) if vendor_model else None
    candidate_speed = statistics.median([r.get("tokens_per_second") for r in (candidate_model or {}).get("runs", []) if r.get("tokens_per_second")]) if candidate_model else None
    lines.append(f"| status | {(vendor_model or {}).get('status', 'unavailable')} | {(candidate_model or {}).get('status', 'unavailable')} |")
    lines.append(f"| median tokens/s | {_fmt(vendor_speed)} | {_fmt(candidate_speed)} |")
    for label, record in (("vendor", vendor_model), ("candidate", candidate_model)):
        if not record:
            lines.append(f"- {label}: unavailable")
            continue
        runs = record.get("runs", [])
        speeds = [r.get("tokens_per_second") for r in runs if r.get("tokens_per_second")]
        speed_text = f"{statistics.median(speeds):.4f}" if speeds else "unavailable"
        lines.append(f"- {label}: status={record.get('status')}, runs={len(runs)}, median tokens/s={speed_text}")
    return "\n".join(lines) + "\n"


def _fmt(value):
    return "" if value is None else f"{float(value):.6g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--vendor-model", type=Path)
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fair-matrix.md"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/fair-matrix-summary.json"))
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    vendor = json.loads(args.vendor_model.read_text(encoding="utf-8")) if args.vendor_model else None
    candidate = json.loads(args.candidate_model.read_text(encoding="utf-8")) if args.candidate_model else None
    args.output.write_text(render(matrix, vendor, candidate), encoding="utf-8")
    summary = {"case_count": len(matrix["cases"]), "methods": ["pytorch_reference", "torch_sdpa", "metax_vendor", "mxflashattn_api", "vllm_vendor_model", "vllm_candidate_model"], "operator_methods": ["reference", "torch_sdpa", "metax_vendor", "mxflashattn_api"], "vllm_vendor_status": vendor.get("status") if vendor else None, "vllm_candidate_status": candidate.get("status") if candidate else None, "claims": {"reference_speedup_separate": True, "vendor_speedup_verified": False}}
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(matrix['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()

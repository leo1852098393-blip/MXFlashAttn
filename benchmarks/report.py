from __future__ import annotations

import argparse
import json
from statistics import median
from pathlib import Path


def render_markdown(run: dict) -> str:
    cases = run.get("cases", [])
    metadata = run.get("metadata", {})
    fallbacks = [case for case in cases if case.get("fallback_reason")]
    latencies = [float(case["latency_ms"]) for case in cases if case.get("latency_ms") is not None]
    errors = [float(case["max_abs_error"]) for case in cases if case.get("max_abs_error") is not None]
    relative_errors = [
        float(case["max_relative_error"]) for case in cases if case.get("max_relative_error") is not None
    ]
    comparable = [
        case
        for case in cases
        if case.get("latency_ms") is not None and case.get("baseline_latency_ms") is not None
    ]
    reductions = [
        (float(case["baseline_latency_ms"]) - float(case["latency_ms"]))
        / float(case["baseline_latency_ms"])
        * 100
        for case in comparable
        if float(case["baseline_latency_ms"]) > 0
    ]
    decode_cases = [case for case in comparable if case.get("q_len") == 1]
    decode_reductions = [
        (float(case["baseline_latency_ms"]) - float(case["latency_ms"]))
        / float(case["baseline_latency_ms"])
        * 100
        for case in decode_cases
        if float(case["baseline_latency_ms"]) > 0
    ]
    decode_target_cases = sum(value >= 20 for value in decode_reductions)
    baseline_memory = [
        int(case["baseline_peak_memory_bytes"])
        for case in cases
        if case.get("baseline_peak_memory_bytes") is not None
    ]
    candidate_memory = [
        int(case["peak_memory_bytes"])
        for case in cases
        if case.get("peak_memory_bytes") is not None
    ]
    lines = [
        "# MXFlashAttn Benchmark Report",
        "",
        f"- Device: {metadata.get('device_name', metadata.get('device', 'unknown'))}",
        f"- Device total memory: {_memory(metadata.get('device_total_memory_bytes'))}",
        f"- PyTorch: {metadata.get('torch', 'unknown')}",
        f"- MXMACA runtime / driver: {metadata.get('runtime', 'unknown')} / {metadata.get('driver', 'unknown')}",
        f"- FlashAttention / vLLM: {metadata.get('flash_attn', 'unknown')} / {metadata.get('vllm', 'unknown')}",
        f"- Seed / warmup / repeats: {metadata.get('seed', 'unknown')} / {metadata.get('warmup', 'unknown')} / {metadata.get('repeats', 'unknown')}",
        f"- Cases: {len(cases)}",
        f"- Reference fallback cases: {len(fallbacks)}",
    ]
    if latencies:
        lines.extend(["", f"- Median attention latency: {median(latencies):.4f} ms"])
    if reductions:
        lines.append(f"- Median latency reduction vs reference: {median(reductions):.2f}%")
    if decode_reductions:
        lines.append(
            f"- Decode (q_len=1) median reduction: {median(decode_reductions):.2f}% "
            f"({decode_target_cases}/{len(decode_reductions)} cases at or above 20%)"
        )
    if errors:
        lines.append(f"- Maximum absolute error: {max(errors):.6g}")
    if relative_errors:
        lines.append(
            f"- Maximum relative error (reference denominator floor 1e-6): {max(relative_errors):.6g}"
        )
    if baseline_memory and candidate_memory:
        lines.append(
            f"- Maximum measured peak memory: {_memory(max(baseline_memory))} reference, "
            f"{_memory(max(candidate_memory))} candidate"
        )
    if not cases or len(fallbacks) == len(cases):
        lines.extend(["", "**Performance target is not verified.** Every measured candidate used the reference fallback."])
    else:
        lines.extend(["", "Native backend measurements are present; per-case results include regressions and improvements."])
        if decode_reductions and decode_target_cases != len(decode_reductions):
            lines.append("The 20% decode target is not met by every measured decode case.")
    lines.extend(
        [
            "",
            "| Case | Layout | Dtype | Backend | Reference (ms) | Candidate (ms) | Reduction | Decode (ms) | Max abs error | Max rel error | Peak memory ref/candidate | Fallback reason |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for case in cases:
        baseline_latency = case.get("baseline_latency_ms")
        latency = case.get("latency_ms")
        reduction = None
        if baseline_latency is not None and latency is not None and float(baseline_latency) > 0:
            reduction = (float(baseline_latency) - float(latency)) / float(baseline_latency) * 100
        lines.append(
            "| {case_id} | {layout} | {dtype} | {backend} | {baseline} | {latency} | {reduction} | {decode} | {error} | {relative_error} | {memory} | {fallback} |".format(
                case_id=case.get("case_id", ""),
                layout=case.get("layout", ""),
                dtype=case.get("dtype", ""),
                backend=case.get("backend", ""),
                baseline=_format(baseline_latency),
                latency=_format(case.get("latency_ms")),
                reduction="" if reduction is None else f"{reduction:.2f}%",
                decode=_format(case.get("decode_latency_ms")),
                error=_format(case.get("max_abs_error")),
                relative_error=_format(case.get("max_relative_error")),
                memory=f"{_memory(case.get('baseline_peak_memory_bytes'))} / {_memory(case.get('peak_memory_bytes'))}",
                fallback=case.get("fallback_reason") or "",
            )
        )
    return "\n".join(lines) + "\n"


def _format(value: object) -> str:
    return "" if value is None else f"{float(value):.6g}"


def _memory(value: object) -> str:
    return "" if value is None else f"{int(value) / (1024**2):.2f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a benchmark JSON report as Markdown")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.md"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote benchmark report to {args.output}")


if __name__ == "__main__":
    main()

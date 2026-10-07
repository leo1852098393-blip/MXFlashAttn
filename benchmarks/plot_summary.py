from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


LAYOUTS = ("dense", "varlen", "paged")
QUERY_LENGTHS = (1, 64)
COLORS = {"dense": "#0072B2", "varlen": "#009E73", "paged": "#D55E00"}


def summarize_matrix(run: dict) -> list[dict]:
    groups: dict[tuple[int, str], list[float]] = {}
    for case in run.get("cases", []):
        if case.get("fallback_reason") or case.get("backend") == "reference":
            continue
        baseline = case.get("baseline_latency_ms")
        latency = case.get("latency_ms")
        key = (case.get("q_len"), case.get("layout"))
        if (
            baseline is None
            or latency is None
            or baseline <= 0
            or key[0] not in QUERY_LENGTHS
            or key[1] not in LAYOUTS
        ):
            continue
        reduction = (float(baseline) - float(latency)) / float(baseline) * 100
        groups.setdefault(key, []).append(reduction)

    return [
        {
            "q_len": q_len,
            "layout": layout,
            "median_reduction_pct": median(values),
            "target_cases": sum(value >= 20 for value in values),
            "case_count": len(values),
        }
        for q_len in QUERY_LENGTHS
        for layout in LAYOUTS
        if (values := groups.get((q_len, layout)))
    ]


def render_plot(run: dict, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install mxflashattn[reports] to generate the benchmark chart") from error

    summary = summarize_matrix(run)
    if not summary:
        raise ValueError("benchmark run contains no comparable measurements")

    metadata = run.get("metadata", {})
    figure, axis = plt.subplots(figsize=(12, 6.5))
    centers = (0.0, 1.7)
    width = 0.28
    offsets = {layout: (index - 1) * width for index, layout in enumerate(LAYOUTS)}
    for layout in LAYOUTS:
        for q_len, center in zip(QUERY_LENGTHS, centers):
            item = next(
                (row for row in summary if row["q_len"] == q_len and row["layout"] == layout),
                None,
            )
            if item is None:
                continue
            x = center + offsets[layout]
            label = layout if q_len == QUERY_LENGTHS[0] else "_nolegend_"
            bar = axis.bar(x, item["median_reduction_pct"], width, color=COLORS[layout], label=label)
            axis.annotate(
                f"{item['median_reduction_pct']:.0f}%\n{item['target_cases']}/{item['case_count']}",
                (bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axis.axhline(20, color="#565D64", linestyle="--", linewidth=1.2, label="20% target")
    axis.set_xticks(centers, ["Decode (q=1)", "Prefill (q=64)"])
    axis.set_ylabel("Median latency reduction vs PyTorch reference (%)")
    axis.set_title("MXFlashAttn C500 Operator Benchmark")
    axis.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncols=4, loc="upper right", frameon=False)
    upper = max(75, max(item["median_reduction_pct"] for item in summary) + 15)
    axis.set_ylim(min(-10, min(item["median_reduction_pct"] for item in summary) - 15), upper)

    run_details = (
        f"{metadata.get('device_name', 'unknown')} | MXMACA {metadata.get('runtime', 'unknown')} | "
        f"PyTorch {metadata.get('torch', 'unknown')} | {len(run.get('cases', []))} cases | "
        f"seed {metadata.get('seed', 'unknown')}, warmup {metadata.get('warmup', 'unknown')}, "
        f"repeats {metadata.get('repeats', 'unknown')}"
    )
    footer = "Labels show median reduction and cases at or above the 20% target.\n" + run_details
    figure.text(0.5, 0.035, footer, ha="center", va="bottom", fontsize=9, color="#41484F")
    figure.tight_layout(rect=(0.02, 0.13, 0.98, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot summarized benchmark latency reductions")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark-summary.png"))
    args = parser.parse_args()
    run = json.loads(args.input.read_text(encoding="utf-8"))
    render_plot(run, args.output)
    print(f"Wrote benchmark chart to {args.output}")


if __name__ == "__main__":
    main()

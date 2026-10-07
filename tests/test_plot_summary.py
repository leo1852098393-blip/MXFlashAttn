import pytest

from benchmarks.plot_summary import summarize_matrix


def test_summary_groups_decode_and_prefill_cases_by_layout():
    cases = []
    for q_len in (1, 64):
        for layout in ("dense", "varlen", "paged"):
            for baseline_latency, latency in ((10.0, 8.0), (20.0, 10.0)):
                cases.append(
                    {
                        "q_len": q_len,
                        "layout": layout,
                        "baseline_latency_ms": baseline_latency,
                        "latency_ms": latency,
                    }
                )
    cases.append(
        {
            "q_len": 1,
            "layout": "dense",
            "backend": "reference",
            "fallback_reason": "native_backend_unavailable",
            "baseline_latency_ms": 1.0,
            "latency_ms": 0.1,
        }
    )

    summary = summarize_matrix({"cases": cases})

    assert len(summary) == 6
    decode_dense = next(item for item in summary if item["q_len"] == 1 and item["layout"] == "dense")
    assert decode_dense["median_reduction_pct"] == pytest.approx(35.0)
    assert decode_dense["target_cases"] == 2
    assert decode_dense["case_count"] == 2

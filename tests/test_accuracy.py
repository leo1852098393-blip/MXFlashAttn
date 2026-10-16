import pytest
import torch

from mxflashattn.accuracy import summarize_error


def test_error_summary_reports_distribution_statistics():
    expected = torch.tensor([1.0, 2.0, 4.0, 8.0])
    actual = torch.tensor([1.0, 2.1, 3.5, 10.0])

    summary = summarize_error(actual, expected)

    assert summary.sample_count == 4
    assert summary.finite
    assert summary.max_abs == pytest.approx(2.0)
    assert summary.mean_abs == pytest.approx(0.65)
    assert summary.p95_abs >= summary.mean_abs
    assert summary.p99_abs >= summary.p95_abs
    assert summary.max_relative == pytest.approx(0.25)
    assert summary.as_dict()["max_abs"] == summary.max_abs


def test_error_summary_marks_nonfinite_values():
    summary = summarize_error(torch.tensor([float("nan")]), torch.tensor([1.0]))
    assert not summary.finite
    assert summary.max_abs == float("inf")


def test_error_summary_rejects_empty_or_mismatched_inputs():
    with pytest.raises(ValueError, match="empty"):
        summarize_error(torch.empty(0), torch.empty(0))
    with pytest.raises(ValueError, match="identical shapes"):
        summarize_error(torch.zeros(2), torch.zeros(3))

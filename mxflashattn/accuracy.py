from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ErrorSummary:
    sample_count: int
    finite: bool
    max_abs: float
    mean_abs: float
    p95_abs: float
    p99_abs: float
    max_relative: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_error(actual: torch.Tensor, expected: torch.Tensor) -> ErrorSummary:
    """Summarize tensor error for local correctness and long-context reports."""
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError("actual and expected must be torch.Tensor values")
    if actual.shape != expected.shape:
        raise ValueError("actual and expected must have identical shapes")
    if actual.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")

    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    finite = bool(torch.isfinite(actual_f).all() and torch.isfinite(expected_f).all())
    abs_error = (actual_f - expected_f).abs().reshape(-1)
    relative_error = abs_error / expected_f.abs().reshape(-1).clamp_min(1e-6)
    # Keep the report machine-readable when a kernel emits NaN/Inf: the
    # finite flag records the root failure and the metrics remain comparable.
    abs_error = torch.nan_to_num(abs_error, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    relative_error = torch.nan_to_num(
        relative_error, nan=float("inf"), posinf=float("inf"), neginf=float("inf")
    )
    return ErrorSummary(
        sample_count=abs_error.numel(),
        finite=finite,
        max_abs=float(abs_error.max().item()),
        mean_abs=float(abs_error.mean().item()),
        p95_abs=float(torch.quantile(abs_error, 0.95).item()),
        p99_abs=float(torch.quantile(abs_error, 0.99).item()),
        max_relative=float(relative_error.max().item()),
    )

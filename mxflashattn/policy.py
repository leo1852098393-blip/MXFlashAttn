from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Phase = Literal["prefill", "decode"]
Backend = Literal["vendor", "m1_prefill_chunked", "m1_split_kv"]


@dataclass(frozen=True)
class WorkloadShape:
    """Shape metadata used by the v1.4 candidate policy.

    This object contains no tensors and performs no device queries. It is
    deliberately suitable for local tests and for later integration with a
    vLLM adapter.
    """

    phase: Phase
    batch_size: int
    query_len: int
    kv_len: int
    head_dim: int
    gqa_ratio: int

    def __post_init__(self) -> None:
        if self.phase not in ("prefill", "decode"):
            raise ValueError("phase must be 'prefill' or 'decode'")
        if self.batch_size <= 0 or self.query_len <= 0 or self.kv_len <= 0:
            raise ValueError("batch_size, query_len, and kv_len must be positive")
        if self.head_dim <= 0 or self.gqa_ratio <= 0:
            raise ValueError("head_dim and gqa_ratio must be positive")


@dataclass(frozen=True)
class DispatchPolicyConfig:
    """Candidate thresholds, not production calibration.

    The values are intentionally disabled by default. They must be calibrated
    on the target C500 environment before being used for runtime dispatch.
    """

    enabled: bool = False
    calibrated: bool = False
    prefill_batch_min: int = 4
    prefill_kv_len_min: int = 512
    decode_batch_min: int = 16
    decode_kv_len_min: int = 256


@dataclass(frozen=True)
class DispatchDecision:
    backend: Backend
    reason: str
    eligible: bool
    policy_enabled: bool
    thresholds_calibrated: bool


class AdaptiveDispatchPolicy:
    """Safe, tensor-free candidate policy for later runtime integration."""

    def __init__(self, config: DispatchPolicyConfig | None = None) -> None:
        self.config = config or DispatchPolicyConfig()

    def select(self, shape: WorkloadShape) -> DispatchDecision:
        config = self.config
        if not config.enabled:
            return DispatchDecision(
                backend="vendor",
                reason="policy_disabled",
                eligible=False,
                policy_enabled=False,
                thresholds_calibrated=config.calibrated,
            )
        if not config.calibrated:
            return DispatchDecision(
                backend="vendor",
                reason="thresholds_not_calibrated",
                eligible=False,
                policy_enabled=True,
                thresholds_calibrated=False,
            )

        if shape.phase == "prefill":
            eligible = (
                shape.batch_size >= config.prefill_batch_min
                and shape.kv_len >= config.prefill_kv_len_min
            )
            return DispatchDecision(
                backend="m1_prefill_chunked" if eligible else "vendor",
                reason="prefill_threshold_match" if eligible else "prefill_below_threshold",
                eligible=eligible,
                policy_enabled=True,
                thresholds_calibrated=True,
            )

        eligible = (
            shape.batch_size >= config.decode_batch_min
            and shape.kv_len >= config.decode_kv_len_min
            and shape.gqa_ratio in (2, 4, 8)
            and shape.head_dim in (64, 128)
        )
        return DispatchDecision(
            backend="m1_split_kv" if eligible else "vendor",
            reason="decode_threshold_match" if eligible else "decode_shape_or_threshold_miss",
            eligible=eligible,
            policy_enabled=True,
            thresholds_calibrated=True,
        )

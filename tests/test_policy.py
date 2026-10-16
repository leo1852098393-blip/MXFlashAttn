import pytest

from mxflashattn.policy import AdaptiveDispatchPolicy, DispatchPolicyConfig, WorkloadShape


def shape(**overrides):
    values = {
        "phase": "prefill",
        "batch_size": 8,
        "query_len": 512,
        "kv_len": 1024,
        "head_dim": 128,
        "gqa_ratio": 4,
    }
    values.update(overrides)
    return WorkloadShape(**values)


def test_policy_is_safe_by_default():
    decision = AdaptiveDispatchPolicy().select(shape())
    assert decision.backend == "vendor"
    assert decision.reason == "policy_disabled"
    assert not decision.eligible


def test_enabled_policy_still_falls_back_until_thresholds_are_calibrated():
    policy = AdaptiveDispatchPolicy(DispatchPolicyConfig(enabled=True))
    decision = policy.select(shape())
    assert decision.backend == "vendor"
    assert decision.reason == "thresholds_not_calibrated"
    assert not decision.thresholds_calibrated


def test_calibrated_policy_selects_prefill_candidate_only_above_thresholds():
    policy = AdaptiveDispatchPolicy(DispatchPolicyConfig(enabled=True, calibrated=True))
    assert policy.select(shape()).backend == "m1_prefill_chunked"
    assert policy.select(shape(batch_size=1)).backend == "vendor"
    assert policy.select(shape(kv_len=128)).backend == "vendor"


def test_calibrated_policy_selects_decode_candidate_only_for_supported_shapes():
    policy = AdaptiveDispatchPolicy(DispatchPolicyConfig(enabled=True, calibrated=True))
    decision = policy.select(shape(phase="decode", batch_size=32, query_len=1, kv_len=1024))
    assert decision.backend == "m1_split_kv"
    assert policy.select(shape(phase="decode", gqa_ratio=7)).backend == "vendor"
    assert policy.select(shape(phase="decode", head_dim=256)).backend == "vendor"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase": "other"},
        {"batch_size": 0},
        {"query_len": 0},
        {"kv_len": 0},
    ],
)
def test_workload_shape_rejects_invalid_metadata(kwargs):
    with pytest.raises(ValueError):
        shape(**kwargs)

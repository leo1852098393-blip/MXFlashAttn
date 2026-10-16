from integrations.vllm.compat import classify_vllm


def test_vllm_017_is_the_only_currently_supported_contract():
    result = classify_vllm("0.17.0")
    assert result.supported
    assert result.status == "supported"


def test_vllm_030_requires_a_real_api_probe():
    result = classify_vllm("0.30.0")
    assert not result.supported
    assert result.status == "probe_required"


def test_unknown_or_missing_vllm_is_explicitly_reported():
    assert classify_vllm(None).status == "not_installed"
    assert classify_vllm("future-build").status == "unknown"
    assert classify_vllm("0.40.0").status == "unsupported"

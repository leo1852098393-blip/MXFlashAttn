import pytest

from benchmarks.compare import compare_runs
from benchmarks.run import expand_cases, run_benchmarks
from benchmarks.report import render_markdown
import torch
from subprocess import CompletedProcess
import benchmarks.run as benchmark_run
import mxflashattn.dispatch as dispatch


def test_experiment_matrix_expands_to_at_least_thirty_cases():
    cases = expand_cases(
        {
            "matrix": {
                "batch_size": [1, 2],
                "q_len": [1, 64],
                "kv_len": [128, 512],
                "head_dim": [64, 128],
                "gqa_ratio": [1, 4],
                "layout": ["dense", "varlen", "paged"],
                "dtype": ["float16", "bfloat16"],
                "causal": [True, False],
            }
        }
    )
    assert len(cases) == 384
    assert {case["layout"] for case in cases} == {"dense", "varlen", "paged"}


def test_compare_runs_pairs_by_case_id_and_calculates_speedup():
    baseline = {"cases": [{"case_id": "a", "latency_ms": 10.0, "max_abs_error": 0.0}]}
    candidate = {"cases": [{"case_id": "a", "latency_ms": 8.0, "max_abs_error": 1e-4}]}

    result = compare_runs(baseline, candidate)

    assert result[0]["speedup"] == pytest.approx(1.25)
    assert result[0]["latency_reduction_pct"] == pytest.approx(20.0)
    assert result[0]["max_abs_error"] == pytest.approx(1e-4)


def test_compare_runs_rejects_missing_candidate_cases():
    with pytest.raises(ValueError, match="missing candidate case"):
        compare_runs({"cases": [{"case_id": "a", "latency_ms": 1}]}, {"cases": []})


def test_compare_runs_rejects_environment_mismatch_unless_explicitly_allowed():
    baseline = {
        "metadata": {"device_name": "MetaX C500", "runtime": "3.5.3.20"},
        "cases": [{"case_id": "a", "latency_ms": 10.0}],
    }
    candidate = {
        "metadata": {"device_name": "MetaX C500", "runtime": "3.5.3.21"},
        "cases": [{"case_id": "a", "latency_ms": 8.0}],
    }

    with pytest.raises(ValueError, match="runtime"):
        compare_runs(baseline, candidate)

    result = compare_runs(baseline, candidate, allow_mismatched_environment=True)
    assert result[0]["speedup"] == pytest.approx(1.25)


def test_compare_runs_rejects_different_random_seeds():
    baseline = {
        "metadata": {"seed": 11},
        "cases": [{"case_id": "a", "latency_ms": 10.0}],
    }
    candidate = {
        "metadata": {"seed": 12},
        "cases": [{"case_id": "a", "latency_ms": 8.0}],
    }

    with pytest.raises(ValueError, match="seed"):
        compare_runs(baseline, candidate)


def test_report_labels_fallback_and_does_not_claim_performance_when_no_measurements():
    report = render_markdown(
        {
            "metadata": {"device": "cpu", "torch": "2.8.0+cpu"},
            "cases": [
                {
                    "case_id": "a",
                    "backend": "reference",
                    "fallback_reason": "native_extension_not_built_or_not_importable",
                    "latency_ms": 1.0,
                }
            ],
        }
    )

    assert "reference fallback" in report.lower()
    assert "performance target is not verified" in report.lower()


def test_report_summarizes_decode_reduction_and_memory():
    report = render_markdown(
        {
            "metadata": {
                "device_name": "MetaX C500",
                "torch": "2.8.0",
                "runtime": "3.5",
                "driver": "3.8",
            },
            "cases": [
                {
                    "case_id": "decode-fast",
                    "layout": "dense",
                    "dtype": "float16",
                    "q_len": 1,
                    "backend": "mxmac_flash_attn",
                    "baseline_latency_ms": 10.0,
                    "latency_ms": 8.0,
                    "decode_latency_ms": 8.0,
                    "max_abs_error": 0.001,
                    "baseline_peak_memory_bytes": 2 * 1024 * 1024,
                    "peak_memory_bytes": 1024 * 1024,
                },
                {
                    "case_id": "decode-slow",
                    "layout": "paged",
                    "dtype": "bfloat16",
                    "q_len": 1,
                    "backend": "mxmac_flash_attn",
                    "baseline_latency_ms": 10.0,
                    "latency_ms": 9.0,
                    "decode_latency_ms": 9.0,
                    "max_abs_error": 0.002,
                    "baseline_peak_memory_bytes": 2 * 1024 * 1024,
                    "peak_memory_bytes": 1024 * 1024,
                },
            ],
        }
    )

    assert "Decode (q_len=1) median reduction: 15.00% (1/2 cases at or above 20%)" in report
    assert "20% decode target is not met by every" in report
    assert "2.00 MiB reference, 1.00 MiB candidate" in report
    assert "Seed / warmup / repeats: unknown / unknown / unknown" in report


def test_benchmark_reads_maca_versions_from_mx_smi(monkeypatch):
    monkeypatch.delenv("MXMACA_VERSION", raising=False)
    monkeypatch.delenv("MACA_VERSION", raising=False)
    monkeypatch.delenv("MUSA_VERSION", raising=False)
    monkeypatch.delenv("MXMACA_DRIVER_VERSION", raising=False)
    monkeypatch.delenv("MACA_DRIVER_VERSION", raising=False)
    monkeypatch.delenv("MUSA_DRIVER_VERSION", raising=False)
    output = "MACA Version: 3.5.3.20\nKernel Mode Driver Version: 3.8.30\n"
    monkeypatch.setattr(
        benchmark_run.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, stdout=output, stderr=""),
    )

    assert benchmark_run._runtime_version() == "3.5.3.20"
    assert benchmark_run._driver_version() == "3.8.30"


def test_optional_version_uses_distribution_metadata_without_importing_runtime(monkeypatch):
    monkeypatch.setattr(benchmark_run.importlib.metadata, "version", lambda name: f"{name}-0.17.0")
    original_import = __import__

    def checked_import(name, *args, **kwargs):
        if name == "vllm":
            raise AssertionError("version lookup must not import the runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", checked_import)

    assert benchmark_run._optional_version("vllm") == "vllm-0.17.0"


def test_cpu_smoke_run_records_reference_backend(monkeypatch):
    monkeypatch.setenv("MXFLASHATTN_ALLOW_FALLBACK", "1")
    monkeypatch.setattr(dispatch, "_flash_attn_function", lambda operation: None)
    monkeypatch.setattr(dispatch, "_native_extension", lambda: None)
    config = {
        "matrix": {
            "batch_size": [1],
            "q_len": [1],
            "kv_len": [8],
            "head_dim": [64],
            "gqa_ratio": [1],
            "layout": ["dense"],
            "dtype": ["float32"],
            "causal": [True],
        }
    }

    with pytest.warns(RuntimeWarning, match="native_backend_requires_accelerator_tensor"):
        result = run_benchmarks(config, device=torch.device("cpu"), warmup=0, repeats=1, seed=123)

    assert result["metadata"]["torch"].startswith(torch.__version__)
    assert result["metadata"]["seed"] == 123
    assert result["metadata"]["device_total_memory_bytes"] is None
    assert len(result["cases"]) == 1
    assert result["cases"][0]["backend"] == "reference"
    assert result["cases"][0]["fallback_reason"] == "native_backend_requires_accelerator_tensor"
    assert result["cases"][0]["first_token_latency_ms"] is None
    assert "baseline_peak_memory_bytes" in result["cases"][0]

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "benchmarks" / "configs" / "v14_c500_prefill.json"
SCRIPT = ROOT / "scripts" / "prepare_c500_v14.py"


def test_c500_plan_has_required_models_and_guards():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["plan_id"] == "v1.4.0-c500-prefill-thresholds"
    assert len(plan["models"]) == 3
    assert len(plan["guard_cases"]) == 2
    assert "ttft_ms_p95" in plan["required_metrics"]
    assert "m1_split_decode_events" in plan["required_metrics"]


def test_manifest_generator_creates_pending_auditable_cases(tmp_path):
    output = tmp_path / "manifest.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=ROOT,
        check=True,
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["status"] == "pending"
    assert len(manifest["cases"]) == 11
    assert all(case["status"] == "pending" for case in manifest["cases"])
    assert all("dispatch_log" in case["evidence"] for case in manifest["cases"])

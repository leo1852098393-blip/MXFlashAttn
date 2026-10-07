"""Run local release-candidate checks without publishing or changing remotes."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED = (
    "README.md",
    "README.en.md",
    "LICENSE",
    "pyproject.toml",
    "requirements-c500.txt",
    "mxflashattn/api.py",
    "mxflashattn/dispatch.py",
    "benchmarks/run.py",
    "tests",
    "docs/support-matrix.md",
    "PROGRESS.md",
)
SENSITIVE_PATTERNS = (
    re.compile(r"Abcd12345678", re.IGNORECASE),
    re.compile(r"SSHPASS", re.IGNORECASE),
    re.compile(r"root\+vm-[A-Za-z0-9]+@\d+\.\d+\.\d+\.\d+"),
    re.compile(r"password\s*[:=]", re.IGNORECASE),
)


def audit(root: Path) -> dict:
    missing = [path for path in REQUIRED if not (root / path).exists()]
    findings: list[dict[str, object]] = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.name == "release_audit.py":
            continue
        if path.suffix.lower() in {".pyc", ".png", ".jpg", ".jpeg", ".so"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for pattern in SENSITIVE_PATTERNS:
            match = pattern.search(content)
            if match:
                findings.append({"path": str(path.relative_to(root)), "pattern": pattern.pattern})
    license_text = (root / "LICENSE").read_text(encoding="utf-8", errors="replace") if (root / "LICENSE").exists() else ""
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8", errors="replace") if (root / "pyproject.toml").exists() else ""
    return {
        "required_files_missing": missing,
        "sensitive_findings": findings,
        "apache_license_present": "Apache License" in license_text and "Version 2.0" in license_text,
        "pyproject_declares_apache": 'license = "Apache-2.0"' in pyproject,
        "pass": not missing and not findings and "Apache License" in license_text and 'license = "Apache-2.0"' in pyproject,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/release-audit.json"))
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass


@dataclass(frozen=True)
class VllmCompatibility:
    installed_version: str | None
    status: str
    reason: str
    supported: bool


def detect_vllm_version() -> str | None:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None


def classify_vllm(version: str | None) -> VllmCompatibility:
    if version is None:
        return VllmCompatibility(None, "not_installed", "vLLM is not installed", False)
    normalized = version.split("+", 1)[0]
    parts = normalized.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return VllmCompatibility(version, "unknown", "cannot parse vLLM version", False)

    if (major, minor) == (0, 17):
        return VllmCompatibility(version, "supported", "v1.3 backend contract", True)
    if (major, minor) == (0, 30):
        return VllmCompatibility(
            version,
            "probe_required",
            "vLLM 0.30 API must be inspected and tested in the target environment",
            False,
        )
    return VllmCompatibility(
        version,
        "unsupported",
        "no validated MXFlashAttn integration contract for this vLLM version",
        False,
    )

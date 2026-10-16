#!/usr/bin/env bash
set -euo pipefail

# Run unsupported configurations independently so an expected guard rejection
# cannot hide later guard evidence or supported-case results.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/csrc_native"

export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT"
PY="${PYTHON:-/opt/conda/bin/python}"

for tag in gq3 gq5 gq6 gq7; do
  echo "=== M1 guard case: ${tag} ==="
  if "$PY" test_m1.py "$tag"; then
    echo "UNEXPECTED_PASS ${tag}"
  else
    echo "EXPECTED_GUARD_REJECTION ${tag}"
  fi
done

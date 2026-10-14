"""Path helpers for executable C500 probes.

The public tree may be checked out anywhere.  Remote evidence keeps its
original paths, but runnable probes resolve the current checkout locally.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("MXFLASHATTN_ROOT", Path(__file__).resolve().parents[1]))
NATIVE_DIR = Path(os.environ.get("MXFLASHATTN_NATIVE_DIR", Path(__file__).resolve().parent))

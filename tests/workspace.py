"""
A throwaway copy of the project's data, for tests that write files (runs,
decisions, settings, new courses). Your real data/ folder is never touched.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coursealign.workspace import make_workspace


def fresh_workspace() -> Path:
    return make_workspace(ROOT)

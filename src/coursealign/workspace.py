"""
A throwaway copy of the project's data, so demos and tests can write runs,
decisions, settings, and courses without touching your real data/ folder.
"""

import shutil
import tempfile
from pathlib import Path

# Things a run or a person creates; a fresh workspace starts without them.
SKIP = shutil.ignore_patterns("runs", "evals", "settings.json", "decisions.json")


def copy_data(source_root: Path, target_data: Path) -> None:
    shutil.copytree(Path(source_root) / "data", target_data, ignore=SKIP)


def make_workspace(source_root: Path) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="coursealign-"))
    copy_data(source_root, workspace / "data")
    fixtures = workspace / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    shutil.copy(Path(source_root) / "tests" / "fixtures" / "agent1_fake_replies.json", fixtures)
    return workspace

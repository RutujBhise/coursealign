"""
A throwaway copy of the project's data, so demos and tests can write runs,
decisions, settings, and courses without touching your real data/ folder.

It holds only the practice courses: the demo runs the stand-in agents, whose
answers are written for those three, and the tests count on exactly them.
"""

import shutil
import tempfile
from pathlib import Path

from coursealign.loader import PRACTICE_COURSES

# Things a run or a person creates, plus the raw text for imported courses;
# a fresh workspace starts without them.
SKIP = shutil.ignore_patterns("runs", "evals", "settings.json", "decisions.json", "ocw")


def _ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set(SKIP(directory, names))
    if Path(directory).name in ("syllabi", "assignments"):
        ignored |= {n for n in names if n.split("_")[0].upper() not in PRACTICE_COURSES}
    return ignored


def copy_data(source_root: Path, target_data: Path) -> None:
    shutil.copytree(Path(source_root) / "data", target_data, ignore=_ignore)


def make_workspace(source_root: Path) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="coursealign-"))
    copy_data(source_root, workspace / "data")
    fixtures = workspace / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    shutil.copy(Path(source_root) / "tests" / "fixtures" / "agent1_fake_replies.json", fixtures)
    return workspace

"""
Data loader: raw files → typed models.

WHY A SEPARATE LOADER:
The pipeline shouldn't care whether data comes from files, an API, or a database.
The loader handles file I/O and validation. Everything downstream gets clean typed
objects. This also makes testing easy — tests can construct models directly without
touching files.
"""

import json
from pathlib import Path

from coursealign.models import RawSyllabus, Assignment


def load_syllabus(file_path: str | Path) -> RawSyllabus:
    """Load a single syllabus file into a RawSyllabus model.

    WHY WE EXTRACT course_id AND title FROM THE TEXT:
    Real syllabi don't come with metadata files. The course code and title
    are inside the document. We do minimal parsing here (first line) and
    leave the hard extraction to Stage 0 / Agent 1.
    """
    path = Path(file_path)
    raw_text = path.read_text(encoding="utf-8")

    # Minimal parsing: first line is usually "CS XXXX – Course Title" or similar
    first_line = raw_text.strip().split("\n")[0].strip()

    # Try to extract course ID from filename as fallback
    # e.g., cs4550_web_development.txt → CS4550
    stem = path.stem  # cs4550_web_development
    course_id = stem.split("_")[0].upper()  # CS4550

    return RawSyllabus(
        course_id=course_id,
        title=first_line,
        raw_text=raw_text,
        file_path=str(path),
    )


def load_assignments(file_path: str | Path) -> list[Assignment]:
    """Load assignments from a JSON file into typed Assignment models.

    WHY JSON FOR ASSIGNMENTS:
    Assignments typically come from an LMS export (structured data), not a PDF.
    JSON simulates what a real LMS API would return. Syllabi are unstructured
    text because that's what they actually are in the real world.
    """
    path = Path(file_path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    assignments = []
    for item in raw:
        assignments.append(Assignment(
            id=item["id"],
            course_id=item["course"],
            title=item["title"],
            type=item["type"],
            description=item["description"],
            rubric_keywords=item.get("rubric_keywords", []),
            points=item["points"],
            week=item["week"],
        ))
    return assignments


def load_course_data(
    syllabi_dir: str | Path,
    assignments_dir: str | Path,
) -> list[dict]:
    """Load all courses: match each syllabus to its assignment file.

    WHY MATCH BY COURSE ID:
    In real systems, data from different sources gets joined by a key.
    Here the key is the course ID embedded in filenames. This mirrors
    the reconciliation problem — what if filenames don't match? We handle
    that gracefully.

    Returns a list of dicts with 'syllabus' and 'assignments' keys.
    """
    syllabi_dir = Path(syllabi_dir)
    assignments_dir = Path(assignments_dir)

    # Load all syllabi, keyed by course_id
    syllabi = {}
    for f in sorted(syllabi_dir.glob("*.txt")):
        s = load_syllabus(f)
        syllabi[s.course_id] = s

    # Load all assignment files, keyed by course_id
    assignments = {}
    for f in sorted(assignments_dir.glob("*.json")):
        course_id = f.stem.split("_")[0].upper()
        assignments[course_id] = load_assignments(f)

    # Join on course_id
    courses = []
    all_ids = sorted(set(syllabi.keys()) | set(assignments.keys()))

    for cid in all_ids:
        course = {
            "course_id": cid,
            "syllabus": syllabi.get(cid),
            "assignments": assignments.get(cid, []),
            "has_syllabus": cid in syllabi,
            "has_assignments": cid in assignments,
        }
        courses.append(course)

    return courses


# The three synthetic courses the stand-in agents' answers (tests/fixtures) are
# written for. Stand-in runs, the tests and the demo workspace use only these;
# real courses, like the OCW test set, need a real model.
PRACTICE_COURSES = ("CS3200", "CS4530", "CS4550")


def stand_in_courses(courses: list[dict], wanted: str | None) -> list[dict]:
    """The courses a stand-in run can use: the practice courses, unless
    --courses named some, which then all have to be practice courses."""
    if not wanted:
        return [c for c in courses if c["course_id"] in PRACTICE_COURSES]
    other = [c["course_id"] for c in courses if c["course_id"] not in PRACTICE_COURSES]
    if other:
        raise SystemExit(f"The stand-in agents only know {', '.join(PRACTICE_COURSES)}. "
                         f"Run {', '.join(other)} with a real model, like --llm gemini.")
    return courses


def pick_courses(courses: list[dict], wanted: str | None) -> list[dict]:
    """Only the courses named in `wanted` ("MIT6005,MIT6170"), so a held-out
    test set can be run and scored on its own. None or "" keeps them all."""
    if not wanted:
        return courses
    names = {name.strip().upper() for name in wanted.split(",") if name.strip()}
    picked = [c for c in courses if c["course_id"] in names]
    unknown = names - {c["course_id"] for c in picked}
    if unknown:
        known = ", ".join(c["course_id"] for c in courses)
        raise SystemExit(f"No course called {', '.join(sorted(unknown))}. The courses are: {known}")
    return picked


def validate_course_data(courses: list[dict]) -> list[str]:
    """Check for data quality issues before the pipeline runs.

    WHY VALIDATE BEFORE PROCESSING:
    Garbage in, garbage out. If a syllabus has no text or assignments
    have missing fields, we want to know NOW, not three agent calls later.
    This is the 'data cleaning' mindset Lumen's JD emphasizes.
    """
    issues = []

    for course in courses:
        cid = course["course_id"]

        if not course["has_syllabus"]:
            issues.append(f"{cid}: no syllabus file found")

        if not course["has_assignments"]:
            issues.append(f"{cid}: no assignments file found")

        if course["has_syllabus"] and len(course["syllabus"].raw_text.strip()) < 100:
            issues.append(f"{cid}: syllabus text suspiciously short ({len(course['syllabus'].raw_text)} chars)")

        for a in course["assignments"]:
            if not a.description.strip():
                issues.append(f"{cid}/{a.id}: assignment has empty description")
            if not a.rubric_keywords:
                issues.append(f"{cid}/{a.id}: assignment has no rubric keywords (hurts deterministic matching)")
            if a.points <= 0:
                issues.append(f"{cid}/{a.id}: assignment has 0 or negative points")

    return issues

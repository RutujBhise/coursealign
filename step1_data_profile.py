"""
Step 1: Load and profile the synthetic dataset.

RUN: python step1_data_profile.py

This should be the first thing you run after cloning the repo. It tells you:
- How many courses, syllabi, and assignments were loaded
- Data quality issues (missing files, empty fields, etc.)
- Per-course breakdown: syllabus length, assignment count, types, total points
- Keyword distribution across assignments (preview of what Stage 0 will use)

WHY THIS STEP MATTERS:
You can't build a pipeline without understanding your data. This script is
also your first "demo" — in an interview, you open with the data profile
before showing the agent architecture.
"""

import sys
from pathlib import Path
from collections import Counter

# Add src to path so imports work when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from coursealign.loader import load_course_data, validate_course_data


def print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_subheader(text: str) -> None:
    print(f"\n  --- {text} ---")


def main():
    data_dir = Path(__file__).parent / "data"
    syllabi_dir = data_dir / "syllabi"
    assignments_dir = data_dir / "assignments"

    # ── Load ─────────────────────────────────────────────────────
    print_header("STEP 1: DATA PROFILE")

    courses = load_course_data(syllabi_dir, assignments_dir)
    print(f"\n  Courses found: {len(courses)}")
    print(f"  Syllabi loaded: {sum(1 for c in courses if c['has_syllabus'])}")
    print(f"  Assignment files loaded: {sum(1 for c in courses if c['has_assignments'])}")

    total_assignments = sum(len(c["assignments"]) for c in courses)
    print(f"  Total assignments across all courses: {total_assignments}")

    # ── Validate ─────────────────────────────────────────────────
    print_subheader("DATA QUALITY CHECK")

    issues = validate_course_data(courses)
    if issues:
        print(f"  Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"    ⚠  {issue}")
    else:
        print("  ✓ No data quality issues found")

    # ── Per-course breakdown ─────────────────────────────────────
    print_subheader("PER-COURSE BREAKDOWN")

    for course in courses:
        cid = course["course_id"]
        assignments = course["assignments"]

        print(f"\n  [{cid}]")

        if course["has_syllabus"]:
            syl = course["syllabus"]
            line_count = len(syl.raw_text.strip().split("\n"))
            char_count = len(syl.raw_text)
            print(f"    Syllabus: {syl.title}")
            print(f"    Syllabus size: {line_count} lines, {char_count:,} chars")
        else:
            print(f"    Syllabus: MISSING")

        if assignments:
            types = Counter(a.type for a in assignments)
            total_points = sum(a.points for a in assignments)
            type_str = ", ".join(f"{v} {k}(s)" for k, v in sorted(types.items()))

            print(f"    Assignments: {len(assignments)} total ({type_str})")
            print(f"    Total points: {total_points}")
            print(f"    Week range: {min(a.week for a in assignments)}–{max(a.week for a in assignments)}")
        else:
            print(f"    Assignments: NONE")

    # ── Keyword analysis (preview of Stage 0) ────────────────────
    print_subheader("KEYWORD DISTRIBUTION (what Stage 0 will use)")

    all_keywords = Counter()
    for course in courses:
        for a in course["assignments"]:
            for kw in a.rubric_keywords:
                all_keywords[kw.lower()] += 1

    print(f"  Unique keywords across all assignments: {len(all_keywords)}")
    print(f"\n  Top 15 most common:")
    for kw, count in all_keywords.most_common(15):
        bar = "█" * count
        print(f"    {kw:<25} {bar} ({count})")

    # ── Summary stats ────────────────────────────────────────────
    # No outcome counts or "LLM calls saved" here on purpose: outcomes
    # don't exist until Step 2 extracts them, so any number printed now
    # would be a guess. The real API-savings figure is measured in Step 2.
    print_header("SUMMARY")
    print(f"  Courses: {len(courses)}")
    print(f"  Total assignments: {total_assignments}")
    print(f"  Unique keywords: {len(all_keywords)}")
    print("  Learning outcomes: not extracted yet (Step 2)")
    print("\n  → Ready for Step 2: Stage 0 Pre-processing\n")


if __name__ == "__main__":
    main()

"""
Tests for import_course.py: pasted plain text in, CourseAlign's files out.

RUN:  pytest tests/   or   python3 tests/test_import_course.py
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from import_course import fill_by_pasting, import_course, label_by_asking

SYLLABUS = """Test Course 101
Objectives
1. You will be able to design small programs.
2. You will be comfortable working in a team,
   though this won't be graded.
"""
ASSIGNMENTS = """Lines before the first block are notes and are ignored.
=== PS1 | problem set | Problem set 1: "Designs"
Design a "small" program.
Then test it.
RUBRIC: design, testing
=== P1 | project | Team project
Build it together in teams of three.
"""


def workspace(labels: str | None = None, assignments: str = ASSIGNMENTS) -> Path:
    root = Path(tempfile.mkdtemp())
    for folder in ("syllabi", "assignments", "gold", "ocw"):
        (root / "data" / folder).mkdir(parents=True)
    (root / "data" / "syllabi" / "mit9999_test_course.txt").write_text(SYLLABUS, encoding="utf-8")
    (root / "data" / "ocw" / "mit9999_assignments.txt").write_text(assignments, encoding="utf-8")
    (root / "data" / "gold" / "alignment_labels.json").write_text(
        json.dumps({"labeling_rule": "kept", "courses": {"CS1": [{"quote": "x", "aligned": []}]}}), encoding="utf-8")
    if labels is not None:
        (root / "data" / "ocw" / "mit9999_labels.txt").write_text(labels, encoding="utf-8")
    return root


def test_pasted_quotes_and_line_breaks_become_valid_json():
    root = workspace()
    import_course("mit9999", root)
    rows = json.loads((root / "data" / "assignments" / "mit9999_assignments.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in rows] == ["PS1", "P1"] and rows[0]["course"] == "MIT9999"
    assert rows[0]["title"] == 'Problem set 1: "Designs"'
    assert rows[0]["description"] == 'Design a "small" program.\nThen test it.'
    assert rows[0]["rubric_keywords"] == ["design", "testing"] and rows[1]["rubric_keywords"] == []
    assert rows[0]["points"] == 100 and rows[0]["week"] == 0


def test_good_labels_are_saved_and_other_courses_kept():
    labels = """=== objective 1
QUOTE: You will be able to design small programs.
ASSESSED BY: PS1
=== objective 2
QUOTE: You will be comfortable working in a team,
  though this won't be graded.
ASSESSED BY: none
NOT ASSESSED: yes
NOTE: The syllabus says it won't be graded.
"""
    root = workspace(labels)
    import_course("MIT9999", root)
    gold = json.loads((root / "data" / "gold" / "alignment_labels.json").read_text(encoding="utf-8"))
    assert gold["labeling_rule"] == "kept" and "CS1" in gold["courses"]
    first, second = gold["courses"]["MIT9999"]
    assert first == {"quote": "You will be able to design small programs.", "aligned": ["PS1"]}
    assert second["quote"].endswith("though this won't be graded.")          # the wrapped line joined
    assert second["stated_not_assessed"] is True and second["aligned"] == []


def test_a_mistyped_quote_or_assignment_id_stops_everything():
    labels = """=== objective 1
QUOTE: You will be able to design large systems.
ASSESSED BY: PS9
"""
    root = workspace(labels)
    try:
        import_course("MIT9999", root)
    except SystemExit as stop:
        assert "isn't in the syllabus word for word" in str(stop) and "no assignment called PS9" in str(stop)
        assert not (root / "data" / "assignments" / "mit9999_assignments.json").exists()
        return
    raise AssertionError("a quote that isn't in the syllabus should stop the import")


def test_a_forgotten_placeholder_is_caught():
    root = workspace(assignments="=== PS1 | problem set | Problem set 1\nPASTE THE HANDOUT TEXT HERE\n"
                                 "=== P1 | project | Project 1: PASTE THE TITLE HERE\nBuild it.\n")
    try:
        import_course("MIT9999", root)
    except SystemExit as stop:
        assert "PS1: still has the PASTE" in str(stop) and "P1: still has the PASTE" in str(stop)
        return
    raise AssertionError("an unfilled template should stop the import")


def test_pasting_fills_each_placeholder_and_saves_as_it_goes():
    template = ("A note that stays.\n=== P1 | project | Project 1\nPASTE THE PROJECT PAGE TEXT HERE\n"
                "=== P2 | project | Project 2\nPASTE THE PROJECT PAGE TEXT HERE\n"
                "=== QUIZ | quiz | Quizzes\nPASTE THE SENTENCES HERE\n")
    root = workspace(assignments=template)
    typed = iter(["Build a web app.", "", 'Use "routes" and tests.', "END",      # P1: two paragraphs
                  "SKIP",                                                       # P2: later
                  "Answer questions on the readings.", "END"])                  # QUIZ
    assert fill_by_pasting("MIT9999", root, read=lambda: next(typed), show=lambda *a: None) == 2
    saved = (root / "data" / "ocw" / "mit9999_assignments.txt").read_text(encoding="utf-8")
    assert saved.startswith("A note that stays.")
    assert 'Build a web app.\n\nUse "routes" and tests.' in saved
    assert saved.count("PASTE") == 1                                            # only the skipped P2 is left
    try:
        import_course("MIT9999", root)
    except SystemExit as stop:
        assert "P2: still has the PASTE" in str(stop) and "P1" not in str(stop)
        return
    raise AssertionError("the skipped assignment should still block the import")


def test_labeling_by_answering_checks_each_answer_and_saves_it():
    root = workspace()
    shown = []
    typed = iter([
        "You will be able to design small programs.",       # outcome 1
        "show PS1",                                         # read it first
        "PS9",                                              # a typo: asked again
        "PS1", "n", "",                                     # assessed by PS1, assessed, no note
        "You will design huge systems.",                    # not in the syllabus: asked again
        "You will be comfortable working in a team,",       # outcome 2
        "none", "y", "The syllabus says it won't be graded.",
        "",                                                 # done
    ])
    assert label_by_asking("MIT9999", root, read=lambda: next(typed), show=lambda *a: shown.append(" ".join(map(str, a)))) == 2
    assert any("Design a \"small\" program." in line for line in shown)           # show PS1 printed its text
    assert any("no assignment called PS9" in line for line in shown)
    assert any("isn't in the syllabus word for word" in line for line in shown)
    import_course("MIT9999", root)
    gold = json.loads((root / "data" / "gold" / "alignment_labels.json").read_text(encoding="utf-8"))
    first, second = gold["courses"]["MIT9999"]
    assert first == {"quote": "You will be able to design small programs.", "aligned": ["PS1"]}
    assert second["stated_not_assessed"] is True and second["aligned"] == [] and second["note"].startswith("The syllabus")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")

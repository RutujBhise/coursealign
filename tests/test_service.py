"""
Tests for the service layer: everything the API does, without a web server.
Each test works on its own copy of the data.

RUN:  pytest tests/   or   python3 tests/test_service.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from coursealign.service import CourseAlignService, ServiceError
from workspace import fresh_workspace

QUEUE_DECISIONS = [
    {"course_id": "CS3200", "outcome_id": "CS3200-O4", "decision": "keep"},
    {"course_id": "CS3200", "outcome_id": "CS3200-O7", "decision": "drop"},
    {"course_id": "CS4550", "outcome_id": "CS4550-O2", "assignment_id": "A4", "decision": "not_aligned"},
    {"course_id": "CS4550", "outcome_id": "CS4550-O2", "assignment_id": "A5", "decision": "not_aligned"},
]


def service() -> CourseAlignService:
    return CourseAlignService(fresh_workspace())


def status_of(call) -> int:
    try:
        call()
    except ServiceError as error:
        return error.status
    raise AssertionError("expected a ServiceError")


# ── runs and the report ──────────────────────────────────────────────────────

def test_a_run_reports_every_pair_and_is_plain_json():
    report = service().run()
    json.dumps(report)                                      # no dataclasses or enums left
    for course in report["courses"]:
        assessable = [o for o in course["outcomes"] if o["status"] in ("passed", "flagged")]
        assert len(course["pairs"]) == len(assessable) * len(course["assignments"])
    assert report["totals"]["waiting_for_a_person"] == 4


def test_the_latest_report_survives_a_restart():
    s = service()
    first = s.run()
    assert CourseAlignService(s.root).latest() == first


def test_latest_before_any_run_is_a_404():
    assert status_of(service().latest) == 404


# ── reviews ──────────────────────────────────────────────────────────────────

def test_a_review_is_saved_by_its_syllabus_quote():
    s = service()
    s.run()
    saved = s.add_review(QUEUE_DECISIONS[0])
    assert saved["quote"] == "Transaction management and ACID properties are covered extensively"


def test_saved_reviews_empty_the_queue_on_the_next_run():
    s = service()
    s.run()
    for decision in QUEUE_DECISIONS:
        s.add_review(decision)
    assert s.run()["totals"]["waiting_for_a_person"] == 0


def test_bad_reviews_get_the_right_status():
    s = service()
    assert status_of(lambda: s.add_review(QUEUE_DECISIONS[0])) == 409          # nothing has run yet
    s.run()
    assert status_of(lambda: s.add_review({**QUEUE_DECISIONS[0], "outcome_id": "CS3200-O99"})) == 404
    assert status_of(lambda: s.add_review({**QUEUE_DECISIONS[0], "decision": "maybe"})) == 400
    assert status_of(lambda: s.add_review({**QUEUE_DECISIONS[2], "teach": [{"topic": "astrology", "term": "x"}]})) == 400


def test_a_review_can_be_removed():
    s = service()
    s.run()
    s.add_review(QUEUE_DECISIONS[0])
    assert s.delete_review(0) == [] and status_of(lambda: s.delete_review(0)) == 404


# ── settings ─────────────────────────────────────────────────────────────────

def test_settings_are_checked_before_they_are_saved():
    s = service()
    assert status_of(lambda: s.update_settings({"agent2_review_at": 0.95})) == 400   # above accept_at
    assert status_of(lambda: s.update_settings({"auto_match_min_topics": 2.5})) == 400
    assert status_of(lambda: s.update_settings({"no_such_knob": 1})) == 400
    assert not s.settings_path.exists()


def test_a_setting_changes_the_next_run_and_resets():
    s = service()
    s.update_settings({"agent2_review_at": 0.75})           # the pair at 0.70 is now held, not flagged
    pair = next(p for c in s.run()["courses"] for p in c["pairs"]
                if p["outcome_id"] == "CS4550-O2" and p["assignment_id"] == "A4")
    assert pair["aligned"] is None
    assert s.update_settings({"agent2_review_at": None})["agent2_review_at"]["changed"] is False


# ── courses ──────────────────────────────────────────────────────────────────

NEW_COURSE = dict(
    course_id="cs9001",
    syllabus_text="CS 9001 - Demo\n\nLEARNING OUTCOMES\n1. Write SQL queries with joins\n\nGRADING POLICY\nAll homework\n",
    assignments=[{"id": "HW1", "title": "Joins", "type": "homework", "description": "Write SQL joins.",
                  "rubric_keywords": ["SQL", "joins"], "points": 10, "week": 1}],
)


def test_an_added_course_is_in_the_next_run():
    s = service()
    assert s.add_course(**NEW_COURSE)["course_id"] == "CS9001"
    assert "CS9001" in [c["course_id"] for c in s.run()["courses"]]


def test_adding_a_course_twice_conflicts_unless_replacing():
    s = service()
    s.add_course(**NEW_COURSE)
    assert status_of(lambda: s.add_course(**NEW_COURSE)) == 409
    assert s.add_course(**NEW_COURSE, replace=True)["assignments"] == 1


def test_a_course_id_cant_escape_the_data_folder():
    assert status_of(lambda: service().add_course(**{**NEW_COURSE, "course_id": "../../etc"})) == 400


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")

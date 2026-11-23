"""
Tests for the HTTP layer: routes, status codes, and request validation.
They need FastAPI and httpx (pip install fastapi httpx) and are skipped
without them. The logic behind each route is tested in test_service.py.

RUN:  pytest tests/   or   python3 tests/test_api.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    from fastapi.testclient import TestClient
    from coursealign.api import create_app
except ImportError:                 # FastAPI or httpx isn't installed
    TestClient = None

try:
    import pytest
    pytestmark = pytest.mark.skipif(TestClient is None, reason="needs: pip install fastapi httpx")
except ImportError:
    pass

from coursealign.service import CourseAlignService
from workspace import fresh_workspace


def client():
    return TestClient(create_app(CourseAlignService(fresh_workspace())))


def test_health():
    response = client().get("/api/health")
    assert response.status_code == 200 and response.json()["ok"] is True


def test_latest_is_404_before_any_run():
    response = client().get("/api/runs/latest")
    assert response.status_code == 404 and "POST /api/runs" in response.json()["detail"]


def test_a_run_is_returned_and_read_back():
    c = client()
    report = c.post("/api/runs").json()
    assert report["totals"]["courses"] == 3
    assert c.get("/api/runs/latest").json()["ran_at"] == report["ran_at"]


def test_a_review_is_saved_and_listed():
    c = client()
    c.post("/api/runs")
    response = c.post("/api/reviews", json={"course_id": "CS3200", "outcome_id": "CS3200-O4", "decision": "keep"})
    assert response.status_code == 201
    assert [r["decision"] for r in c.get("/api/reviews").json()] == ["keep"]


def test_a_bad_request_gets_a_clear_status():
    c = client()
    c.post("/api/runs")
    assert c.post("/api/reviews", json={"course_id": "CS3200"}).status_code == 422          # missing fields
    bad = {"course_id": "CS3200", "outcome_id": "CS3200-O4", "decision": "maybe"}
    assert c.post("/api/reviews", json=bad).status_code == 400


def test_settings_round_trip():
    c = client()
    assert c.put("/api/settings", json={"agent2_review_at": 0.7}).json()["agent2_review_at"]["value"] == 0.7
    assert c.put("/api/settings", json={"agent2_review_at": None}).json()["agent2_review_at"]["changed"] is False


def test_adding_a_course_then_again_conflicts():
    c = client()
    course = {"course_id": "CS9001", "syllabus_text": "CS 9001 - Demo\n\nLEARNING OUTCOMES\n1. Write SQL joins\n",
              "assignments": [{"id": "HW1", "title": "Joins", "type": "homework", "description": "Write SQL joins.",
                               "rubric_keywords": ["SQL"], "points": 10, "week": 1}]}
    assert c.post("/api/courses", json=course).status_code == 201
    assert c.post("/api/courses", json=course).status_code == 409


if __name__ == "__main__":
    if TestClient is None:
        print("  skipped: these need FastAPI and httpx (pip install fastapi httpx)")
        sys.exit(0)
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")

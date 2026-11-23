"""
Step 6: the API.

RUN:
  pip install fastapi uvicorn
  python3 step6_api.py                              fake agents, no API key
  COURSEALIGN_LLM=anthropic python3 step6_api.py    the real model
  then open http://127.0.0.1:8000/docs to try every endpoint

  python3 step6_api.py --workspace     serve a temporary copy of data/, so nothing
                                       you click changes your files (the end-to-end
                                       tests use this, with --port 8123)

  python3 step6_api.py --check
      No web server needed. Walks through what each endpoint does, on a
      temporary copy of data/, so it changes none of your files.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from coursealign.service import CourseAlignService, ServiceError
from coursealign.workspace import make_workspace


def serve(port: int, workspace: bool) -> None:
    try:
        import uvicorn
        import fastapi  # noqa: F401  (checked here so the message below is the one people see)
    except ImportError:
        sys.exit("The API needs FastAPI and uvicorn: pip install fastapi uvicorn\n"
                 "(Or try: python3 step6_api.py --check, which needs neither.)")
    if workspace:
        root = make_workspace(ROOT)
        os.environ["COURSEALIGN_ROOT"] = str(root)          # read when the app is created
        os.environ["COURSEALIGN_RESET_FROM"] = str(ROOT)
        print(f"Serving a temporary copy of data/ in {root}")
    built = (ROOT / "web" / "dist" / "index.html").exists()
    print(f"CourseAlign on http://127.0.0.1:{port}")
    print("  dashboard: /" if built else "  dashboard: not built yet (cd web && npm install && npm run build)")
    print("  API docs:  /docs")
    uvicorn.run("coursealign.api:app", host="127.0.0.1", port=port)


def show(method: str, path: str, result: str) -> None:
    print(f"  {method:<6} {path:<22} {result}")


def check() -> None:
    workspace = make_workspace(ROOT)
    service = CourseAlignService(workspace)

    print("\n" + "=" * 64)
    print("  STEP 6: EVERY ENDPOINT, CHECKED WITHOUT A WEB SERVER")
    print("=" * 64)
    print("  (on a temporary copy of data/, so none of your files change)\n")

    h = service.health()
    show("GET", "/api/health", f"ok, {h['llm']} agents, {h['courses']} courses, graph engine: {h['graph_engine'].split(' ')[0]}")
    courses = service.list_courses()
    show("GET", "/api/courses", ", ".join(f"{c['course_id']} ({c['assignments']} assignments)" for c in courses))
    try:
        service.latest()
    except ServiceError as e:
        show("GET", "/api/runs/latest", f"{e.status} before any run: \"{e.message}\"")

    report = service.run()
    t = report["totals"]
    show("POST", "/api/runs", f"{t['courses']} courses, {t['outcomes']} outcomes, {t['pairs']} pairs,"
         f" {t['agent_requests']} agent requests ({t['requests_saved']:.0%} saved by the rules)")
    print(f"  {'':<29} waiting for a person: {t['waiting_for_a_person']}, gaps: {t['gaps']}")
    size = len((workspace / "data" / "runs" / "latest.json").read_bytes()) // 1024
    show("GET", "/api/runs/latest", f"the same report, read back from data/runs/latest.json ({size} KB)")

    # The same six decisions as Step 5's example, sent the way the dashboard will send them
    decisions = [
        {"course_id": "CS3200", "outcome_id": "CS3200-O4", "decision": "keep"},
        {"course_id": "CS3200", "outcome_id": "CS3200-O7", "decision": "drop"},
        {"course_id": "CS4550", "outcome_id": "CS4550-O2", "assignment_id": "A4", "decision": "not_aligned"},
        {"course_id": "CS4550", "outcome_id": "CS4550-O2", "assignment_id": "A5", "decision": "not_aligned"},
        {"course_id": "CS4530", "outcome_id": "CS4530-O6", "assignment_id": "IA1", "decision": "partial",
         "teach": [{"topic": "communication", "term": "justification"}]},
        {"course_id": "CS4530", "outcome_id": "CS4530-O6", "assignment_id": "IA4", "decision": "partial"},
    ]
    for d in decisions:
        saved = service.add_review(d)
        target = f"{d['outcome_id']} x {d['assignment_id']}" if d.get("assignment_id") else d["outcome_id"]
        show("POST", "/api/reviews", f"201 saved #{saved['index']}: {saved['decision']} {target}")
    for bad, why in [({"course_id": "CS3200", "outcome_id": "CS3200-O99", "decision": "keep"}, "an unknown outcome"),
                     ({"course_id": "CS3200", "outcome_id": "CS3200-O4", "decision": "maybe"}, "a bad decision")]:
        try:
            service.add_review(bad)
        except ServiceError as e:
            show("POST", "/api/reviews", f"{e.status} for {why}: \"{e.message}\"")

    report = service.run()
    t, s = report["totals"], report["scores"]
    show("POST", "/api/runs", f"waiting for a person: {t['waiting_for_a_person']},"
         f" decided by a person: {t['decided_by']['human']}, every pair right: {s['pairs']['right']}/{s['pairs']['decided']}")

    settings = service.get_settings()
    show("GET", "/api/settings", ", ".join(f"{name}={v['value']}" for name, v in settings.items()))
    try:
        service.update_settings({"agent2_review_at": 0.95})
    except ServiceError as e:
        show("PUT", "/api/settings", f"{e.status} for agent2_review_at=0.95: \"{e.message}\"")
    service.update_settings({"common_topic_share": 1.0})
    s = service.run()["scores"]
    show("PUT", "/api/settings", f"common_topic_share=1.0 (rule off), then a run: every pair right "
         f"{s['pairs']['right']}/{s['pairs']['decided']}, wrong rule matches: {len(s['rules']['wrong_matches'])}")
    service.update_settings({"common_topic_share": None})
    s = service.run()["scores"]
    show("PUT", "/api/settings", f"reset to default, then a run: every pair right {s['pairs']['right']}/{s['pairs']['decided']}")

    new_course = {
        "course_id": "CS9001",
        "syllabus_text": "CS 9001 - Demo Course\n\nLEARNING OUTCOMES\n1. Write SQL queries with joins\n"
                         "2. Deploy applications to the cloud\n\nGRADING POLICY\nHomework 100%\n",
        "assignments": [{"id": "HW1", "title": "Joins Homework", "type": "homework",
                         "description": "Write five SQL queries with joins.", "rubric_keywords": ["SQL", "joins"],
                         "points": 50, "week": 2}],
    }
    added = service.add_course(**new_course)
    show("POST", "/api/courses", f"201 added {added['course_id']} \"{added['title']}\" ({added['assignments']} assignment)")
    try:
        service.add_course(**new_course)
    except ServiceError as e:
        show("POST", "/api/courses", f"{e.status} adding it again: \"{e.message}\"")
    report = service.run()
    demo = next(c for c in report["courses"] if c["course_id"] == "CS9001")
    show("POST", "/api/runs", f"now {report['totals']['courses']} courses; CS9001 has {len(demo['outcomes'])} outcomes"
         f" and {len(demo['queue'])} pair waiting for a person")
    print(f"  {'':<29} (the fake judge has no labels for a new course, so a person decides;")
    print(f"  {'':<29}  until then its gaps say so: {[g['type'] for g in demo['gaps']]})")
    print()
    shutil.rmtree(workspace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="walk through every endpoint without a web server")
    parser.add_argument("--workspace", action="store_true", help="serve a temporary copy of data/")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.check:
        check()
    else:
        serve(args.port, args.workspace)

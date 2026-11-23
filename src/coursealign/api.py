"""
CourseAlign's HTTP API: a thin FastAPI layer over service.py.

RUN (from the project root):
  pip install fastapi uvicorn
  python3 step6_api.py
  then open http://127.0.0.1:8000/docs to try every endpoint

  COURSEALIGN_LLM=anthropic uses the real model (needs ANTHROPIC_API_KEY).
  The default, fake, needs no key.

Every route calls the service and returns JSON. All the logic lives in
service.py, so it's tested without a web server; tests/test_api.py checks
this layer with FastAPI's TestClient.
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from coursealign.service import CourseAlignService, ServiceError

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = ROOT / "web" / "dist"


class Teach(BaseModel):
    topic: str = Field(description="A topic from stage0_rules.json, e.g. communication")
    term: str = Field(description="The word that should mean that topic, e.g. justification")


class ReviewIn(BaseModel):
    course_id: str
    outcome_id: str
    decision: str = Field(description="For an outcome: keep or drop. For a pair: direct, partial, or not_aligned.")
    assignment_id: str | None = Field(default=None, description="Set this to decide a pair instead of an outcome.")
    note: str = ""
    teach: list[Teach] = []


class AssignmentIn(BaseModel):
    id: str
    title: str
    type: str
    description: str
    rubric_keywords: list[str] = []
    points: int
    week: int


class CourseIn(BaseModel):
    course_id: str = Field(description="Letters then digits, e.g. CS5800")
    syllabus_text: str
    assignments: list[AssignmentIn]
    replace: bool = False


def create_app(service: CourseAlignService | None = None) -> FastAPI:
    if service is None:
        reset_from = os.environ.get("COURSEALIGN_RESET_FROM")       # set by step6_api.py --workspace
        service = CourseAlignService(Path(os.environ.get("COURSEALIGN_ROOT", ROOT)),
                                     llm=os.environ.get("COURSEALIGN_LLM", "fake"),
                                     reset_from=Path(reset_from) if reset_from else None)
    app = FastAPI(
        title="CourseAlign",
        description="Does each course actually assess what its syllabus says it teaches?",
        version="0.6.0",
    )
    # Lets a separately served dev frontend (like a Vite server) call the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ServiceError)
    def service_error(request: Request, error: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=error.status, content={"detail": error.message})

    @app.get("/api/health", summary="Is the API up, and how is it configured?")
    def health() -> dict:
        return service.health()

    @app.get("/api/courses", summary="List the courses")
    def list_courses() -> list[dict]:
        return service.list_courses()

    @app.post("/api/courses", status_code=201, summary="Add a course: its syllabus and assignments")
    def add_course(course: CourseIn) -> dict:
        return service.add_course(course.course_id, course.syllabus_text,
                                  [a.model_dump() for a in course.assignments], course.replace)

    @app.post("/api/runs", summary="Run the pipeline on every course, with every saved decision")
    def run() -> dict:
        return service.run()

    @app.get("/api/runs/latest", summary="The last run's report")
    def latest() -> dict:
        return service.latest()

    @app.get("/api/reviews", summary="List saved decisions")
    def list_reviews() -> list[dict]:
        return service.list_reviews()

    @app.post("/api/reviews", status_code=201, summary="Save a decision about an outcome or a pair")
    def add_review(review: ReviewIn) -> dict:
        return service.add_review(review.model_dump())

    @app.delete("/api/reviews/{index}", summary="Remove a saved decision")
    def delete_review(index: int) -> list[dict]:
        return service.delete_review(index)

    @app.get("/api/settings", summary="The knobs you can turn, with their current values")
    def get_settings() -> dict:
        return service.get_settings()

    @app.put("/api/settings", summary="Change knobs; send null to reset one to its default")
    def update_settings(changes: dict[str, float | None]) -> dict:
        return service.update_settings(changes)

    if service.reset_from is not None:
        @app.post("/api/test/reset", include_in_schema=False)
        def reset() -> dict:
            return service.reset()

    # The dashboard (Step 7), once built with: cd web && npm install && npm run build
    if (DASHBOARD / "index.html").exists():
        app.mount("/", StaticFiles(directory=DASHBOARD, html=True), name="dashboard")
    else:
        @app.get("/", include_in_schema=False)
        def home():
            return RedirectResponse("/docs")

    return app


app = create_app()

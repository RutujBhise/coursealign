"""
Everything the API does, as plain Python, so it's tested without a web
server. api.py is a thin FastAPI layer over this.

Files it reads and writes, all under the project's data/ folder:
  syllabi/, assignments/       the courses (add_course writes here)
  rules/*.json                 the hand-written rules; never rewritten
  rules/settings.json          overrides for a few knobs (update_settings writes it)
  review/decisions.json        a person's decisions (add_review appends)
  runs/latest.json             the last run's report (run writes it)
  gold/alignment_labels.json   hand labels, used only to score a run

Errors are raised as ServiceError with the HTTP status they should become.
"""

import json
import re
import shutil
import threading
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

from coursealign.agent1 import Agent1Config
from coursealign.agent2 import Agent2Config
from coursealign.evaluate import load_gold
from coursealign.graph import build_graph, graph_api, initial_state
from coursealign.llm import CachedLLM, FakeLLM, build_llms
from coursealign.loader import load_course_data
from coursealign.models import BloomLevel, ReviewDecision
from coursealign.oracle import LabelJudge
from coursealign.report import build_report
from coursealign.review import OUTCOME_DECISIONS, PAIR_DECISIONS, learned_terms, load_reviews, save_reviews
from coursealign.stage0 import Rules
from coursealign.workspace import copy_data

COURSE_ID = re.compile(r"[A-Za-z]{2,8}\d{2,5}")
ASSIGNMENT_ID = re.compile(r"[A-Za-z0-9_-]{1,20}")
NO_OUTCOMES = '{"outcomes": []}'

# The knobs the dashboard can turn: name -> (config, attribute, type, min, max, what it does)
SETTINGS = {
    "auto_match_min_topics": ("rules", "auto_match_min_graded", int, 1, 5,
        "Graded topics an outcome and an assignment must share for the rules to auto-match them."),
    "common_topic_share": ("rules", "common_share", float, 0.3, 1.0,
        "A topic in more than this share of a course's assignments is background and can't count "
        "toward an auto-match. 1.0 turns the rule off."),
    "agent1_accept_at": ("agent1", "accept_at", float, 0.0, 1.0,
        "Agent 1 outcomes at or above this confidence pass without review."),
    "agent1_review_at": ("agent1", "pass_with_review_at", float, 0.0, 1.0,
        "Agent 1 outcomes below this confidence are held for a person."),
    "agent2_accept_at": ("agent2", "accept_at", float, 0.0, 1.0,
        "Agent 2 verdicts at or above this confidence pass without review."),
    "agent2_review_at": ("agent2", "pass_with_review_at", float, 0.0, 1.0,
        "Agent 2 verdicts below this confidence are held for a person."),
}


class ServiceError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _review_json(r: ReviewDecision, index: int) -> dict:
    return {"index": index, "kind": r.kind, "course_id": r.course_id, "quote": r.quote,
            "decision": r.decision, "assignment_id": r.assignment_id or None,
            "assessed_level": r.assessed_level.value if r.assessed_level else None,
            "note": r.note, "teach": r.teach, "by": r.by, "at": r.at}


class CourseAlignService:
    def __init__(self, root: str | Path, llm: str = "fake", fake_replies: str | Path | None = None,
                 reset_from: str | Path | None = None):
        if llm not in ("fake", "anthropic", "gemini"):
            raise ValueError("llm must be 'fake', 'anthropic', or 'gemini'")
        self.root = Path(root)
        self.data = self.root / "data"
        self.llm_mode = llm
        self.fake_replies = Path(fake_replies) if fake_replies else self.root / "tests" / "fixtures" / "agent1_fake_replies.json"
        self.decisions_path = self.data / "review" / "decisions.json"
        self.latest_path = self.data / "runs" / "latest.json"
        self.settings_path = self.data / "rules" / "settings.json"
        self._lock = threading.Lock()       # one writer at a time
        self._real_llms = None              # built on the first real run, so the API starts without a key
        self.reset_from = Path(reset_from) if reset_from else None   # set only for throwaway workspaces

    # ── reading ──────────────────────────────────────────────────────────────

    def _courses(self) -> list[dict]:
        return [c for c in load_course_data(self.data / "syllabi", self.data / "assignments")
                if c["syllabus"] is not None]

    def _gold(self) -> dict:
        path = self.data / "gold" / "alignment_labels.json"
        return load_gold(path) if path.exists() else {}

    def _overrides(self) -> dict:
        if not self.settings_path.exists():
            return {}
        return json.loads(self.settings_path.read_text(encoding="utf-8"))

    def _configs(self, reviews: list[ReviewDecision], overrides: dict | None = None):
        """The rules and agent configs, with taught words and setting overrides applied."""
        configs = {
            "rules": Rules.from_file(self.data / "rules" / "stage0_rules.json", learned=learned_terms(reviews)),
            "agent1": Agent1Config.from_file(self.data / "rules" / "agent1.json"),
            "agent2": Agent2Config.from_file(self.data / "rules" / "agent2.json"),
        }
        for name, value in (self._overrides() if overrides is None else overrides).items():
            owner, attribute = SETTINGS[name][0], SETTINGS[name][1]
            configs[owner] = replace(configs[owner], **{attribute: value})
        return configs["rules"], configs["agent1"], configs["agent2"]

    def health(self) -> dict:
        return {
            "ok": True,
            "graph_engine": graph_api()[1],
            "llm": self.llm_mode,
            "courses": len(self._courses()),
            "decisions": len(load_reviews(self.decisions_path)),
            "has_run": self.latest_path.exists(),
        }

    def list_courses(self) -> list[dict]:
        gold = self._gold()
        return [{"course_id": c["course_id"], "title": c["syllabus"].title,
                 "assignments": len(c["assignments"]), "has_labels": c["course_id"] in gold}
                for c in self._courses()]

    def latest(self) -> dict:
        if not self.latest_path.exists():
            raise ServiceError(404, "Nothing has run yet. Start a run with POST /api/runs.")
        return json.loads(self.latest_path.read_text(encoding="utf-8"))

    def list_reviews(self) -> list[dict]:
        return [_review_json(r, n) for n, r in enumerate(load_reviews(self.decisions_path))]

    def get_settings(self) -> dict:
        rules, agent1, agent2 = self._configs([])
        configs = {"rules": rules, "agent1": agent1, "agent2": agent2}
        overrides = self._overrides()
        return {
            name: {"value": getattr(configs[owner], attribute), "min": low, "max": high,
                   "step": 1 if kind is int else 0.05, "about": about, "changed": name in overrides}
            for name, (owner, attribute, kind, low, high, about) in SETTINGS.items()
        }

    # ── running ──────────────────────────────────────────────────────────────

    def _llms(self, courses: list[dict], gold: dict, rules: Rules):
        if self.llm_mode == "fake":
            # Fresh fakes every run, so every run behaves the same. A course
            # with no canned replies gets "no outcomes found" from Agent 1.
            reader = FakeLLM.from_file(self.fake_replies, responder=lambda system, messages: NO_OUTCOMES)
            judge = FakeLLM(responder=LabelJudge({c["course_id"]: c["syllabus"] for c in courses}, gold, rules),
                            name="fake (answers from the hand labels)")
            return reader, judge
        if self._real_llms is None:
            cache = self.root / ".cache" / "llm_replies.json"
            model_for_agent1, model_for_agent2 = build_llms(self.llm_mode)
            reader = CachedLLM(model_for_agent1, cache)          # both share one file's saved replies
            judge = reader if model_for_agent2 is model_for_agent1 else CachedLLM(model_for_agent2, cache)
            self._real_llms = (reader, judge)
        return self._real_llms

    @staticmethod
    def _answered(*llms) -> int:
        """Requests a model actually answered (not served from a cache)."""
        total = 0
        for llm in {id(x): x for x in llms}.values():
            total += llm.paid_calls if hasattr(llm, "paid_calls") else len(llm.calls)
        return total

    def run(self) -> dict:
        """Run the whole pipeline on every course, with every saved decision."""
        with self._lock:
            courses = self._courses()
            if not courses:
                raise ServiceError(409, "There are no courses yet. Add one with POST /api/courses.")
            reviews = load_reviews(self.decisions_path)
            gold = self._gold()
            rules, agent1, agent2 = self._configs(reviews)
            reader, judge = self._llms(courses, gold, rules)
            api, engine = graph_api()
            graph = build_graph(rules, agent1, reader, api, agent2_config=agent2,
                                judge_llm=judge, reviews=reviews)
            before = self._answered(reader, judge)
            try:
                finals = [graph.invoke(initial_state(c["syllabus"], c["assignments"])) for c in courses]
            except Exception as error:          # a model or network failure mid-run
                raise ServiceError(502, f"The run failed: {error}") from error
            report = build_report(
                finals,
                ran_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                graph_engine=engine,
                agents={"agent1": reader.name, "agent2": judge.name},
                model_calls=self._answered(reader, judge) - before,
                settings={name: s["value"] for name, s in self.get_settings().items()},
                learned_words=learned_terms(reviews),
                topics=sorted(rules.tag_patterns),
                gold=gold,
            )
            self.latest_path.parent.mkdir(parents=True, exist_ok=True)
            self.latest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            return report

    # ── writing ──────────────────────────────────────────────────────────────

    def add_review(self, payload: dict) -> dict:
        """Save one decision about an outcome or a pair from the latest run.

        payload: course_id, outcome_id, decision, and for a pair assignment_id.
        Optional: note, and teach ([{"topic", "term"}]) on a pair decision.
        """
        with self._lock:
            if not self.latest_path.exists():
                raise ServiceError(409, "Run the pipeline first (POST /api/runs), so the outcome ids exist.")
            report = self.latest()
            course_id, outcome_id = payload.get("course_id"), payload.get("outcome_id")
            course = next((c for c in report["courses"] if c["course_id"] == course_id), None)
            if course is None:
                raise ServiceError(404, f"The latest run has no course {course_id}.")
            outcome = next((o for o in course["outcomes"] if o["id"] == outcome_id), None)
            if outcome is None:
                raise ServiceError(404, f"The latest run of {course_id} has no outcome {outcome_id}.")

            decision, assignment_id = payload.get("decision"), payload.get("assignment_id")
            teach = payload.get("teach") or []
            level = None
            if assignment_id:
                if decision not in PAIR_DECISIONS:
                    raise ServiceError(400, f"A pair decision must be one of: {', '.join(PAIR_DECISIONS)}.")
                if assignment_id not in {a["id"] for a in course["assignments"]}:
                    raise ServiceError(404, f"{course_id} has no assignment {assignment_id}.")
                pair = next((p for p in course["pairs"]
                             if p["outcome_id"] == outcome_id and p["assignment_id"] == assignment_id), None)
                if pair and pair["decided_by"] == "agent2" and pair["depth"] and decision != "not_aligned":
                    level = BloomLevel(pair["depth"])      # confirming keeps Agent 2's depth judgment
            else:
                if decision not in OUTCOME_DECISIONS:
                    raise ServiceError(400, f"An outcome decision must be one of: {', '.join(OUTCOME_DECISIONS)}.")
                if teach:
                    raise ServiceError(400, "Teaching a word goes with a pair decision.")

            topics = set(Rules.from_file(self.data / "rules" / "stage0_rules.json").tag_patterns)
            for taught in teach:
                if taught.get("topic") not in topics or not str(taught.get("term", "")).strip():
                    raise ServiceError(400, f"To teach a word, give a term and one of these topics: {', '.join(sorted(topics))}.")

            review = ReviewDecision(
                kind="pair" if assignment_id else "outcome", course_id=course_id, quote=outcome["quote"],
                decision=decision, assignment_id=assignment_id or "", assessed_level=level,
                note=payload.get("note", ""),
                teach=[{"topic": t["topic"], "term": t["term"].strip().lower()} for t in teach],
                by="dashboard", at=date.today().isoformat(),
            )
            reviews = load_reviews(self.decisions_path) + [review]
            save_reviews(self.decisions_path, reviews)
            return _review_json(review, len(reviews) - 1)

    def delete_review(self, index: int) -> list[dict]:
        with self._lock:
            reviews = load_reviews(self.decisions_path)
            if not 0 <= index < len(reviews):
                raise ServiceError(404, f"There's no decision {index}.")
            del reviews[index]
            save_reviews(self.decisions_path, reviews)
        return self.list_reviews()

    def update_settings(self, changes: dict) -> dict:
        """Change knobs. A value of None puts a knob back to its default."""
        with self._lock:
            unknown = set(changes) - set(SETTINGS)
            if unknown:
                raise ServiceError(400, f"Unknown settings: {', '.join(sorted(unknown))}.")
            overrides = self._overrides()
            for name, value in changes.items():
                _, _, kind, low, high, _ = SETTINGS[name]
                if value is None:
                    overrides.pop(name, None)
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ServiceError(400, f"{name} must be a number.")
                if kind is int and value != int(value):
                    raise ServiceError(400, f"{name} must be a whole number.")
                value = kind(value)
                if not low <= value <= high:
                    raise ServiceError(400, f"{name} must be between {low} and {high}.")
                overrides[name] = value
            _, agent1, agent2 = self._configs([], overrides)
            for agent, config in (("agent1", agent1), ("agent2", agent2)):
                if config.pass_with_review_at > config.accept_at:
                    raise ServiceError(400, f"{agent}_review_at can't be higher than {agent}_accept_at.")
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps(overrides, indent=2) + "\n", encoding="utf-8")
        return self.get_settings()

    def add_course(self, course_id: str, syllabus_text: str, assignments: list[dict], replace: bool = False) -> dict:
        """Add a course: its syllabus text and its assignments."""
        with self._lock:
            if not COURSE_ID.fullmatch(course_id or ""):
                raise ServiceError(400, "A course id looks like CS4550: letters, then digits.")
            course_id = course_id.upper()
            if not syllabus_text.strip():
                raise ServiceError(400, "The syllabus text is empty.")
            if course_id in {c["course_id"] for c in self._courses()} and not replace:
                raise ServiceError(409, f"{course_id} already exists. Send replace: true to overwrite it.")

            rows, seen = [], set()
            for a in assignments:
                aid = str(a.get("id", ""))
                if not ASSIGNMENT_ID.fullmatch(aid) or aid in seen:
                    raise ServiceError(400, f"Each assignment needs a unique id of letters, digits, - or _ (got '{aid}').")
                seen.add(aid)
                try:
                    rows.append({
                        "id": aid, "course": course_id, "title": str(a["title"]), "type": str(a["type"]),
                        "description": str(a["description"]),
                        "rubric_keywords": [str(k) for k in a.get("rubric_keywords", [])],
                        "points": int(a["points"]), "week": int(a["week"]),
                    })
                except (KeyError, TypeError, ValueError) as error:
                    raise ServiceError(400, f"Assignment {aid} is missing or has a bad field: {error}") from error

            stem = course_id.lower()
            for folder, pattern in (("syllabi", f"{stem}_*.txt"), ("assignments", f"{stem}_*.json")):
                for old in (self.data / folder).glob(pattern):
                    old.unlink()                     # replacing: remove the old files
            (self.data / "syllabi" / f"{stem}_course.txt").write_text(syllabus_text, encoding="utf-8")
            (self.data / "assignments" / f"{stem}_course.json").write_text(
                json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            title = syllabus_text.strip().splitlines()[0].strip()
            return {"course_id": course_id, "title": title, "assignments": len(rows)}

    def reset(self) -> dict:
        """Put a throwaway workspace back the way it started (for end-to-end tests)."""
        if self.reset_from is None:
            raise ServiceError(404, "Resetting only works on a throwaway workspace.")
        with self._lock:
            shutil.rmtree(self.data)
            copy_data(self.reset_from, self.data)
        return {"ok": True}

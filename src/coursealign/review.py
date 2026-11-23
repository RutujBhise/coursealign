"""
Human review: the queue of what needs a person, and how a person's
decisions flow back into every later run.

A decision is made once and applied from then on:
  - an outcome decision keeps or drops an outcome Agent 1 found
  - a pair decision settles whether an assignment assesses an outcome. It
    works on any pair: ones the agents weren't sure about, and ones the
    rules got wrong outright
  - a decision can also teach the rules a word ("justification" means
    communication), which helps every course, not just that one pair

Decisions are tied to syllabus quotes, not ids. An agent can number its
outcomes differently next run, but the syllabus words stay put. Later
decisions win over earlier ones.
"""

import json
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Optional

from coursealign.agent2 import DISAGREE
from coursealign.llm import NO_REPLY
from coursealign.models import (
    AlignmentResult, AlignmentType, Assignment, BloomLevel, ExtractedOutcome,
    MatchConfidence, RawSyllabus, ReviewDecision,
)
from coursealign.spans import best_match

OUTCOME_DECISIONS = ("keep", "drop")
PAIR_DECISIONS = ("direct", "partial", "not_aligned")


# ── Saving and loading ───────────────────────────────────────────────────────

def load_reviews(path: str | Path) -> list[ReviewDecision]:
    p = Path(path)
    if not p.exists():
        return []
    reviews = []
    for d in json.loads(p.read_text(encoding="utf-8"))["decisions"]:
        level = d.get("assessed_level")
        reviews.append(ReviewDecision(
            kind=d["kind"], course_id=d["course_id"], quote=d["quote"], decision=d["decision"],
            assignment_id=d.get("assignment_id", ""),
            assessed_level=BloomLevel(level) if level else None,
            note=d.get("note", ""), teach=d.get("teach", []),
            by=d.get("by", "reviewer"), at=d.get("at", ""),
        ))
    return reviews


def save_reviews(path: str | Path, reviews: list[ReviewDecision]) -> None:
    rows = []
    for r in reviews:
        row = asdict(r)
        row["assessed_level"] = r.assessed_level.value if r.assessed_level else None
        rows.append(row)
    payload = {
        "_about": "A person's review decisions. Every run applies them. Later decisions win over earlier ones.",
        "decisions": rows,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def learned_terms(reviews: list[ReviewDecision]) -> dict[str, list[str]]:
    """Words people taught the rules, by topic. Rules.from_file merges these in."""
    terms: dict[str, list[str]] = {}
    for r in reviews:
        for taught in r.teach:
            words = terms.setdefault(taught["topic"], [])
            if taught["term"].lower() not in words:
                words.append(taught["term"].lower())
    return terms


# ── Applying decisions during a run ──────────────────────────────────────────

def _log(r: ReviewDecision, applied_to: Optional[str]) -> dict:
    return {"kind": r.kind, "decision": r.decision, "quote": r.quote,
            "assignment": r.assignment_id, "applied_to": applied_to}


def apply_outcome_reviews(
    syllabus: RawSyllabus,
    outcomes: list[ExtractedOutcome],
    held: list[ExtractedOutcome],
    reviews: list[ReviewDecision],
) -> tuple[list[ExtractedOutcome], list[ExtractedOutcome], list[dict]]:
    """Keep or drop outcomes a person decided on. Runs before triage, so a
    kept outcome gets triaged like any other."""
    mine = [r for r in reviews if r.kind == "outcome" and r.course_id == syllabus.course_id]
    if not mine:
        return outcomes, held, []
    candidates = [(o.id, o.evidence or o.text) for o in outcomes + held]
    chosen: dict[str, ReviewDecision] = {}
    log = []
    for r in mine:                                    # in order, so later decisions win
        target = best_match(r.quote, candidates, syllabus.raw_text, strict=True)
        log.append(_log(r, target))
        if target:
            chosen[target] = r

    kept, still_held = [], []
    for o in outcomes:
        r = chosen.get(o.id)
        if r is None:
            kept.append(o)
        elif r.decision == "keep":
            kept.append(replace(o, needs_review=False))
        # "drop": a person removed it
    for o in held:
        r = chosen.get(o.id)
        if r is None:
            still_held.append(o)
        elif r.decision == "keep":
            kept.append(replace(o, needs_review=False))
        # "drop": off the queue for good
    return kept, still_held, log


def human_verdicts(
    syllabus: RawSyllabus,
    outcomes: list[ExtractedOutcome],
    assignments: list[Assignment],
    reviews: list[ReviewDecision],
) -> tuple[list[AlignmentResult], list[dict]]:
    """A person's pair decisions, as verdicts that outrank Agent 2 and the rules."""
    mine = [r for r in reviews if r.kind == "pair" and r.course_id == syllabus.course_id]
    candidates = [(o.id, o.evidence or o.text) for o in outcomes]
    assignment_ids = {a.id for a in assignments}
    verdicts, log = [], []
    for r in mine:
        target = best_match(r.quote, candidates, syllabus.raw_text, strict=True)
        if target is None or r.assignment_id not in assignment_ids:
            log.append(_log(r, None))
            continue
        kind = AlignmentType.TANGENTIAL if r.decision == "not_aligned" else AlignmentType(r.decision)
        verdicts.append(AlignmentResult(
            outcome_id=target, assignment_id=r.assignment_id, score=1.0,
            confidence_tier=MatchConfidence.HIGH, alignment_type=kind,
            reasoning=r.note or "Decided by a person.", needs_human_review=False,
            assessed_level=r.assessed_level, decided_by="human",
        ))
        log.append(_log(r, f"{target} x {r.assignment_id}"))
    return verdicts, log


# ── The queue ────────────────────────────────────────────────────────────────

@dataclass
class QueueItem:
    kind: str                                   # "outcome" or "pair"
    course_id: str
    why: str
    outcome: ExtractedOutcome
    assignment: Optional[Assignment] = None
    verdict: Optional[AlignmentResult] = None   # Agent 2's, for pairs


def review_queue(state: dict) -> list[QueueItem]:
    """Everything from one course's run that a person should look at."""
    course = state["syllabus"].course_id
    items = []
    for o in state["outcomes"]:
        if o.needs_review:
            items.append(QueueItem("outcome", course, f"flagged: Agent 1 confidence {o.extraction_confidence:.2f}", o))
    for o in state["held"]:
        items.append(QueueItem("outcome", course, f"held: Agent 1 confidence {o.extraction_confidence:.2f}", o))

    outcomes = {o.id: o for o in state["outcomes"]}
    assignments = {a.id: a for a in state["assignments"]}
    for j in state.get("judgments", []):
        if not j.needs_human_review:
            continue
        if j.reasoning.startswith(NO_REPLY):
            why = "held: the model didn't answer (running again asks it again)"
        elif j.reasoning.startswith(DISAGREE):
            why = "held: two separate judgments disagreed"
        elif j.confidence_tier == MatchConfidence.LOW and j.score == 0.0:
            why = "held: Agent 2's replies kept failing the checks"
        elif j.confidence_tier == MatchConfidence.LOW:
            why = f"held: Agent 2 said {j.alignment_type.value} at {j.score:.2f}"
        else:
            why = f"flagged: Agent 2 said {j.alignment_type.value} at {j.score:.2f}"
        items.append(QueueItem("pair", course, why, outcomes[j.outcome_id], assignments[j.assignment_id], j))
    return items


# ── Making decisions ─────────────────────────────────────────────────────────

def decide(item: QueueItem, choice: str, note: str = "", by: str = "reviewer") -> ReviewDecision:
    """Turn a person's answer to a queue item into a saved decision."""
    quote = item.outcome.evidence or item.outcome.text
    if item.kind == "outcome":
        assert choice in OUTCOME_DECISIONS, choice
        return ReviewDecision("outcome", item.course_id, quote, choice, note=note, by=by, at=date.today().isoformat())
    assert choice in PAIR_DECISIONS, choice
    # Confirming Agent 2's verdict keeps its depth judgment; a person can change it later.
    level = item.verdict.assessed_level if item.verdict and choice != "not_aligned" else None
    return ReviewDecision("pair", item.course_id, quote, choice, assignment_id=item.assignment.id,
                          assessed_level=level, note=note, by=by, at=date.today().isoformat())


def correction(outcome: ExtractedOutcome, assignment_id: str, choice: str, note: str = "",
               teach: list[dict] = (), by: str = "reviewer") -> ReviewDecision:
    """Settle any pair, including ones the rules decided wrong."""
    assert choice in PAIR_DECISIONS, choice
    return ReviewDecision("pair", outcome.course_id, outcome.evidence or outcome.text, choice,
                          assignment_id=assignment_id, note=note, teach=list(teach),
                          by=by, at=date.today().isoformat())

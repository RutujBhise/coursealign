"""
The run report: everything one run produced, as plain JSON. The API returns
it, and the dashboard (Step 7) is built on exactly this shape:

{
  "ran_at": "2026-11-23T15:04:05+00:00",
  "graph_engine": "LangGraph",
  "agents": {"agent1": "...", "agent2": "..."},
  "settings": {"auto_match_min_topics": 2, ...},       the knobs this run used
  "learned_words": {"communication": ["justification"]},
  "topics": ["agile", "api", ...],                     topics a taught word can mean
  "totals": {
    "courses", "outcomes", "held_outcomes", "pairs",
    "agent_requests", "model_calls",                  requests vs. calls a model answered
    "requests_without_rules", "requests_saved",       e.g. 171 and 0.708
    "decided_by": {"rules", "agent2", "human"}, "undecided",
    "waiting_for_a_person",
    "gaps": {"critical", "warning", "info"}
  },
  "courses": [{
    "course_id", "title",
    "outcomes":    [{"id", "text", "quote", "source": "rules|agent1", "bloom",
                     "confidence", "status": "passed|flagged|held|not_assessed", "topics"}],
    "assignments": [{"id", "title", "type", "points", "week"}],
    "pairs":       [{"outcome_id", "assignment_id",
                     "aligned": true|false|null,             null: waiting for a person
                     "kind": "direct|partial|null",
                     "decided_by": "rules|agent2|human",
                     "triage": "auto_match|auto_reject|needs_agent",
                     "depth", "confidence", "flagged", "reason", "evidence"}],
    "gaps":        [{"type", "severity", "outcome_id", "assignment_id", "description"}],
    "queue":       [{"kind": "outcome|pair", "outcome_id", "assignment_id", "why",
                     "outcome_text", "quote", "assignment_title", "agent": {...}|null}],
    "stats":       {"agent1_requests", "agent2_requests", "pairs", "decided_by", "undecided"},
    "review_log":  [which saved decisions landed where]
  }],
  "scores": null, or how the run measured against the hand labels, for labeled courses
}
"""

from collections import Counter

from coursealign.evaluate import Run, Label, score_pipeline, score_runs
from coursealign.gaps import final_verdicts
from coursealign.models import ExtractionSource, GapSeverity
from coursealign.review import QueueItem, review_queue

SOURCES = ("rules", "agent2", "human")


def _status(o) -> str:
    if o.stated_not_assessed:
        return "not_assessed"
    return "flagged" if o.needs_review else "passed"


def _outcome(o, status: str) -> dict:
    return {
        "id": o.id, "text": o.text, "quote": o.evidence or o.text,
        "source": "rules" if o.source == ExtractionSource.REGEX else "agent1",
        "bloom": o.bloom_level.value if o.bloom_level else None,
        "confidence": o.extraction_confidence, "status": status, "topics": list(o.topic_tags),
    }


def _queue_item(item: QueueItem) -> dict:
    j = item.verdict
    return {
        "kind": item.kind,
        "outcome_id": item.outcome.id,
        "assignment_id": item.assignment.id if item.assignment else None,
        "why": item.why,
        "outcome_text": item.outcome.text,
        "quote": item.outcome.evidence or item.outcome.text,
        "assignment_title": item.assignment.title if item.assignment else None,
        "agent": None if j is None else {
            "verdict": j.alignment_type.value, "confidence": j.score, "reason": j.reasoning,
            "evidence": j.evidence, "depth": j.assessed_level.value if j.assessed_level else None,
        },
    }


def course_report(final: dict) -> dict:
    syllabus = final["syllabus"]
    outcome_order = {o.id: n for n, o in enumerate(final["outcomes"])}
    assignment_order = {a.id: n for n, a in enumerate(final["assignments"])}
    by_pair = lambda items: {(x.outcome_id, x.assignment_id): x for x in items}
    rules, agent, person = by_pair(final["decisions"]), by_pair(final["judgments"]), by_pair(final["human_verdicts"])

    verdicts = final_verdicts(final["decisions"], final["judgments"], final["human_verdicts"])
    ordered = sorted(verdicts.items(), key=lambda kv: (outcome_order.get(kv[0][0], 10**6),
                                                        assignment_order.get(kv[0][1], 10**6)))
    pairs = []
    for key, v in ordered:
        d, j, h = rules.get(key), agent.get(key), person.get(key)
        if v.source == "human":
            reason, evidence, confidence = h.reasoning, "", None
        elif j is not None:
            reason, evidence, confidence = j.reasoning, j.evidence, j.score
        elif v.source == "agent2":
            reason, evidence, confidence = "Waiting for Agent 2.", "", None
        else:
            reason, evidence, confidence = d.reason, "", None
        pairs.append({
            "outcome_id": key[0], "assignment_id": key[1],
            "aligned": v.aligned, "kind": v.kind.value if v.kind else None,
            "decided_by": v.source, "triage": d.decision.value if d else None,
            "depth": v.depth.value if v.depth else None, "confidence": confidence,
            "flagged": v.flagged, "reason": reason, "evidence": evidence,
        })

    decided_by = Counter(p["decided_by"] for p in pairs if p["aligned"] is not None)
    return {
        "course_id": syllabus.course_id,
        "title": syllabus.title,
        "outcomes": [_outcome(o, _status(o)) for o in final["outcomes"]]
                    + [_outcome(o, "held") for o in final["held"]],
        "assignments": [{"id": a.id, "title": a.title, "type": a.type, "points": a.points, "week": a.week}
                        for a in final["assignments"]],
        "pairs": pairs,
        "gaps": [{"type": g.gap_type.value, "severity": g.severity.value, "outcome_id": g.outcome_id,
                  "assignment_id": g.assignment_id, "description": g.description} for g in final["gaps"]],
        "queue": [_queue_item(item) for item in review_queue(final)],
        "stats": {
            "agent1_requests": len(final["agent1_log"]),
            "agent2_requests": len(final["agent2_log"]),
            "pairs": len(pairs),
            "decided_by": {source: decided_by.get(source, 0) for source in SOURCES},
            "undecided": sum(1 for p in pairs if p["aligned"] is None),
        },
        "review_log": final["review_log"],
    }


def _scores(finals: list[dict], gold: dict[str, list[Label]]) -> dict | None:
    runs = [Run.from_state(f) for f in finals if f["syllabus"].course_id in gold]
    if not runs:
        return None
    p, r = score_pipeline(runs, gold), score_runs(runs, gold)
    return {
        "courses": [run.syllabus.course_id for run in runs],
        "outcomes": {"labeled": r.labeled, "found": r.labeled_found, "unlabeled_extras": r.ungrounded},
        "rules": {"decided": r.decided, "right": r.decided_correct,
                  "wrong_matches": [f"{o} x {a}" for o, a in r.wrong_matches],
                  "missed": [f"{o} x {a}" for o, a in r.missed]},
        "agent2": {"passed": p.accepted, "right": p.accepted_correct, "flagged": p.flagged,
                   "wrong": [f"{o} x {a} ({verdict})" for o, a, verdict in p.wrong]},
        "pairs": {"scored": p.pairs, "decided": p.decided, "right": p.decided_correct,
                  "undecided": p.held, "by_a_person": p.by_human},
        "gaps": {"untested_expected": p.untested_expected, "untested_found": p.untested_found,
                 "untested_false_alarms": p.untested_false,
                 "orphans_expected": p.orphans_expected, "orphans_found": p.orphans_found,
                 "orphans_false_alarms": p.orphans_false},
    }


def build_report(
    finals: list[dict],
    *,
    ran_at: str,
    graph_engine: str,
    agents: dict,
    model_calls: int,
    settings: dict,
    learned_words: dict,
    topics: list[str],
    gold: dict[str, list[Label]],
) -> dict:
    courses = [course_report(f) for f in finals]
    requests = sum(c["stats"]["agent1_requests"] + c["stats"]["agent2_requests"] for c in courses)
    without_rules = len(finals) + sum(
        sum(1 for o in f["outcomes"] if not o.stated_not_assessed) * len(f["assignments"]) for f in finals
    )
    severities = Counter(g["severity"] for c in courses for g in c["gaps"])
    return {
        "ran_at": ran_at,
        "graph_engine": graph_engine,
        "agents": agents,
        "settings": settings,
        "learned_words": learned_words,
        "topics": topics,
        "totals": {
            "courses": len(courses),
            "outcomes": sum(1 for c in courses for o in c["outcomes"] if o["status"] != "held"),
            "held_outcomes": sum(1 for c in courses for o in c["outcomes"] if o["status"] == "held"),
            "pairs": sum(c["stats"]["pairs"] for c in courses),
            "agent_requests": requests,
            "model_calls": model_calls,
            "requests_without_rules": without_rules,
            "requests_saved": round(1 - requests / without_rules, 3) if without_rules else 0.0,
            "decided_by": {s: sum(c["stats"]["decided_by"][s] for c in courses) for s in SOURCES},
            "undecided": sum(c["stats"]["undecided"] for c in courses),
            "waiting_for_a_person": sum(len(c["queue"]) for c in courses),
            "gaps": {level.value: severities.get(level.value, 0) for level in GapSeverity},
        },
        "courses": courses,
        "scores": _scores(finals, gold),
    }

"""
Tests for human review and the feedback loop.

RUN:  pytest tests/   or   python3 tests/test_review.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coursealign.agent1 import Agent1Config
from coursealign.agent2 import Agent2Config
from coursealign.evaluate import Run, load_gold, score_pipeline
from coursealign.gaps import final_verdicts
from coursealign.graph import build_graph, graph_api, initial_state
from coursealign.llm import CachedLLM, FakeLLM
from coursealign.loader import load_course_data
from coursealign.models import (
    AlignmentResult, AlignmentType, ExtractedOutcome, MatchConfidence, RawSyllabus, ReviewDecision,
)
from coursealign.oracle import LabelJudge
from coursealign.review import apply_outcome_reviews, learned_terms, load_reviews, review_queue
from coursealign.spans import best_match
from coursealign.stage0 import Decision, Rules, TriageDecision, extract_outcomes, tag_text

RULES_PATH = ROOT / "data" / "rules" / "stage0_rules.json"
RULES = Rules.from_file(RULES_PATH)
A1 = Agent1Config.from_file(ROOT / "data" / "rules" / "agent1.json")
A2 = Agent2Config.from_file(ROOT / "data" / "rules" / "agent2.json")
GOLD = load_gold(ROOT / "data" / "gold" / "alignment_labels.json")
COURSES = {c["course_id"]: c for c in load_course_data(ROOT / "data" / "syllabi", ROOT / "data" / "assignments")}
SYLLABI = {cid: c["syllabus"] for cid, c in COURSES.items()}
EXAMPLE = ROOT / "data" / "review" / "example_decisions.json"
FAKE_REPLIES = ROOT / "tests" / "fixtures" / "agent1_fake_replies.json"

TEST_SYLLABUS = RawSyllabus("T100", "Test", "Goals: master SQL joins. Also design normalized schemas.", "-")


# ── decisions follow the syllabus words ──────────────────────────────────────

def test_a_decision_follows_the_quote_not_the_id():
    renumbered = ExtractedOutcome(id="T100-O9", course_id="T100", text="Write SQL joins",
                                  evidence="master SQL joins", needs_review=True)
    keep = ReviewDecision("outcome", "T100", "master SQL joins", "keep")
    kept, _, log = apply_outcome_reviews(TEST_SYLLABUS, [renumbered], [], [keep])
    assert kept[0].id == "T100-O9" and not kept[0].needs_review and log[0]["applied_to"] == "T100-O9"


def test_keep_rescues_a_held_outcome_and_drop_removes_one():
    passed = ExtractedOutcome(id="T100-O1", course_id="T100", text="Write SQL joins", evidence="master SQL joins")
    held = ExtractedOutcome(id="T100-O2", course_id="T100", text="Design schemas",
                            evidence="design normalized schemas", extraction_confidence=0.5, needs_review=True)
    reviews = [ReviewDecision("outcome", "T100", "master SQL joins", "drop"),
               ReviewDecision("outcome", "T100", "design normalized schemas", "keep")]
    kept, still_held, _ = apply_outcome_reviews(TEST_SYLLABUS, [passed], [held], reviews)
    assert [o.id for o in kept] == ["T100-O2"] and still_held == []


def test_strict_matching_never_lands_on_a_bigger_outcome():
    source = "Students will master SQL joins and subqueries, and design normalized schemas for real problems."
    big = ("O1", "master SQL joins and subqueries, and design normalized schemas for real problems")
    assert best_match("master SQL joins", [big], source) == "O1"                 # scoring still connects them
    assert best_match("master SQL joins", [big], source, strict=True) is None    # a decision doesn't land


def test_a_decision_that_matches_nothing_is_reported_not_applied():
    outcomes, _ = extract_outcomes(SYLLABI["CS4530"], RULES)
    stale = ReviewDecision("outcome", "CS4530", "a quote that appears in no syllabus", "drop")
    kept, _, log = apply_outcome_reviews(SYLLABI["CS4530"], outcomes, [], [stale])
    assert len(kept) == 12 and log[0]["applied_to"] is None


# ── a person outranks everyone ───────────────────────────────────────────────

def test_a_person_overrules_the_rules():
    rejected = TriageDecision("T100-O1", "X1", Decision.AUTO_REJECT, [], [], "test")
    person = AlignmentResult("T100-O1", "X1", 1.0, MatchConfidence.HIGH, AlignmentType.PARTIAL,
                             "a person", False, decided_by="human")
    verdict = final_verdicts([rejected], [], [person])[("T100-O1", "X1")]
    assert verdict.source == "human" and verdict.aligned


def test_agent2_is_never_asked_about_a_pair_a_person_decided():
    decided = ReviewDecision("pair", "CS4550", "Design and implement RESTful APIs using Node.js and Express",
                             "not_aligned", assignment_id="A4")
    judge = FakeLLM(responder=LabelJudge(SYLLABI, GOLD, RULES))
    graph = build_graph(RULES, A1, FakeLLM.from_file(FAKE_REPLIES), graph_api()[0],
                        agent2_config=A2, judge_llm=judge, reviews=[decided])
    c = COURSES["CS4550"]
    final = graph.invoke(initial_state(c["syllabus"], c["assignments"]))
    asked = [messages[0]["content"] for _, messages in judge.calls]
    assert not any("Pair: CS4550-O2 x A4" in text for text in asked)
    assert any(h.assignment_id == "A4" and h.decided_by == "human" for h in final["human_verdicts"])


# ── teaching the rules ───────────────────────────────────────────────────────

def test_a_taught_word_reaches_the_rules():
    taught = Rules.from_file(RULES_PATH, learned={"communication": ["justification"]})
    assert "communication" in tag_text("Write a 1-page justification", taught)
    assert "communication" not in tag_text("Write a 1-page justification", RULES)


def test_teaching_a_word_for_an_unknown_topic_is_an_error():
    try:
        Rules.from_file(RULES_PATH, learned={"astrology": ["stars"]})
    except ValueError:
        return
    raise AssertionError("an unknown topic should be rejected")


# ── the cache ────────────────────────────────────────────────────────────────

def test_the_cache_answers_a_repeat_request_without_the_model():
    fake = FakeLLM(responder=lambda system, messages: "a reply")
    cached = CachedLLM(fake)
    for _ in range(2):
        assert cached.complete("system", [{"role": "user", "content": "same question"}]) == "a reply"
    assert len(fake.calls) == 1 and cached.paid_calls == 1


# ── the whole loop ───────────────────────────────────────────────────────────

def test_the_example_review_empties_the_queue():
    reviews = load_reviews(EXAMPLE)
    rules = Rules.from_file(RULES_PATH, learned=learned_terms(reviews))
    graph = build_graph(rules, A1, FakeLLM.from_file(FAKE_REPLIES), graph_api()[0], agent2_config=A2,
                        judge_llm=FakeLLM(responder=LabelJudge(SYLLABI, GOLD, RULES)), reviews=reviews)
    finals = [graph.invoke(initial_state(c["syllabus"], c["assignments"])) for c in COURSES.values()]
    assert [item for f in finals for item in review_queue(f)] == []
    assert all(entry["applied_to"] for f in finals for entry in f["review_log"])
    s = score_pipeline([Run.from_state(f) for f in finals], GOLD)
    assert s.decided == s.pairs and s.by_human == 4


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")

"""
Tests for Agent 2, gap analysis, and the full pipeline run offline.

RUN:  pytest tests/   or   python3 tests/test_agent2.py
"""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coursealign.agent1 import Agent1Config
from coursealign.agent2 import (
    DISAGREE, SECOND_OPINION, Agent2Config, check_reply, judge_pair, judge_pairs, system_prompt,
)
from coursealign.evaluate import load_gold
from coursealign.gaps import find_gaps
from coursealign.graph import build_graph, graph_api, initial_state
from coursealign.llm import NO_REPLY, FakeLLM, LLMUnavailable
from coursealign.loader import load_course_data, stand_in_courses
from coursealign.models import (
    AlignmentResult, AlignmentType, Assignment, BloomLevel, ExtractedOutcome, GapType, MatchConfidence,
)
from coursealign.oracle import LabelJudge
from coursealign.review import review_queue
from coursealign.stage0 import Decision, Rules, TriageDecision

RULES = Rules.from_file(ROOT / "data" / "rules" / "stage0_rules.json")
A1 = Agent1Config.from_file(ROOT / "data" / "rules" / "agent1.json")
A2 = Agent2Config.from_file(ROOT / "data" / "rules" / "agent2.json")

SQL_LAB = Assignment(id="X1", course_id="T100", title="SQL Lab", type="homework",
                     description="Write joins and subqueries over the bookstore data.",
                     rubric_keywords=["joins", "subqueries"], points=100, week=3)
SCHEMA = ExtractedOutcome(id="T100-O1", course_id="T100", text="Design normalized schemas",
                          bloom_level=BloomLevel.CREATE)


def verdict(kind="direct", evidence="Write joins and subqueries", level="apply", confidence=0.9) -> str:
    return json.dumps({"verdict": kind, "evidence": evidence, "assessed_level": level,
                       "confidence": confidence, "reason": "test"})


def scripted(*replies) -> FakeLLM:
    queue = list(replies)
    return FakeLLM(responder=lambda system, messages: queue.pop(0))


# ── Agent 2's checks ─────────────────────────────────────────────────────────

def test_judge_quote_must_come_from_the_assignment():
    result, problems = check_reply(verdict(evidence="design a normalized schema"), SQL_LAB, A2)
    assert result is None and "isn't in the assignment" in problems[0]


def test_a_one_word_quote_is_not_evidence():
    result, problems = check_reply(verdict(evidence="joins"), SQL_LAB, A2)
    assert result is None and "evidence must be 3-40 words, got 1" in problems


def test_the_prompt_says_rubric_skills_count_and_how_long_a_quote_must_be():
    prompt = system_prompt(A2)
    assert "a skill the rubric names counts as required" in prompt
    assert f"{A2.min_quote_words} to {A2.max_quote_words} words" in prompt


def test_tangential_needs_no_quote():
    result, problems = check_reply(verdict("tangential", evidence="", level=None), SQL_LAB, A2)
    assert problems == [] and result.kind == AlignmentType.TANGENTIAL


def test_judge_confidence_tiers():
    def tier_of(confidence):
        r = judge_pair(SCHEMA, SQL_LAB, scripted(verdict(confidence=confidence)), A2, system_prompt(A2)).result
        return r.confidence_tier, r.needs_human_review
    assert tier_of(0.9) == (MatchConfidence.HIGH, False)
    assert tier_of(0.7) == (MatchConfidence.MEDIUM, True)
    assert tier_of(0.5) == (MatchConfidence.LOW, True)


def test_a_pair_that_keeps_failing_goes_to_a_human_not_to_no():
    bad = verdict(evidence="not in the assignment at all")
    pair = judge_pair(SCHEMA, SQL_LAB, scripted(*[bad] * A2.max_attempts), A2, system_prompt(A2))
    assert pair.calls == A2.max_attempts
    assert pair.result.confidence_tier == MatchConfidence.LOW and pair.result.needs_human_review


def test_agent2_sees_only_its_own_pair():
    other = Assignment(id="X2", course_id="T100", title="Portfolio Site", type="assignment",
                       description="Build a responsive portfolio with CSS Grid.",
                       rubric_keywords=["CSS Grid"], points=100, week=1)
    llm = FakeLLM(responder=lambda s, m: verdict("tangential", evidence="", level=None))
    judge_pairs([(SCHEMA, SQL_LAB), (SCHEMA, other)], llm, A2)
    for _, messages in llm.calls:
        content = messages[0]["content"]
        assert ("SQL Lab" in content) != ("Portfolio Site" in content)    # exactly one assignment


# ── gap rules ────────────────────────────────────────────────────────────────

def outcome(n: int, level: BloomLevel | None = BloomLevel.APPLY, not_assessed=False) -> ExtractedOutcome:
    return ExtractedOutcome(id=f"T100-O{n}", course_id="T100", text=f"outcome {n}",
                            bloom_level=level, stated_not_assessed=not_assessed)


def assignment(aid: str) -> Assignment:
    return Assignment(id=aid, course_id="T100", title=aid, type="homework", description="",
                      rubric_keywords=[], points=10, week=1)


def rules_match(oid: str, aid: str, kind=AlignmentType.DIRECT) -> TriageDecision:
    return TriageDecision(oid, aid, Decision.AUTO_MATCH, [], [], "test", kind)


def judged(oid: str, aid: str, kind="direct", confidence=0.9, depth="apply") -> AlignmentResult:
    t = MatchConfidence.HIGH if confidence >= 0.8 else MatchConfidence.MEDIUM if confidence >= 0.6 else MatchConfidence.LOW
    return AlignmentResult(oid, aid, confidence, t, AlignmentType(kind), "test", t != MatchConfidence.HIGH,
                           assessed_level=BloomLevel(depth) if depth else None)


def kinds(gaps) -> set:
    return {(g.gap_type, g.outcome_id or g.assignment_id) for g in gaps}


def test_untested_outcome_is_critical_and_mentions_held_pairs():
    gaps = find_gaps([outcome(1)], [assignment("X1")], [], [judged("T100-O1", "X1", confidence=0.4)])
    untested = [g for g in gaps if g.gap_type == GapType.UNTESTED_OUTCOME]
    assert untested and untested[0].severity.value == "critical" and "held for review" in untested[0].description


def test_assignment_that_assesses_nothing_is_an_orphan():
    gaps = find_gaps([outcome(1)], [assignment("X1"), assignment("X2")], [rules_match("T100-O1", "X1")], [])
    assert (GapType.ORPHAN_ASSESSMENT, "X2") in kinds(gaps) and (GapType.ORPHAN_ASSESSMENT, "X1") not in kinds(gaps)


def test_blooms_gap_needs_every_link_to_be_shallower():
    o = outcome(1, BloomLevel.CREATE)
    shallow = find_gaps([o], [assignment("X1")], [], [judged("T100-O1", "X1", depth="apply")])
    assert (GapType.BLOOMS_MISMATCH, "T100-O1") in kinds(shallow)
    deep_enough = find_gaps([o], [assignment("X1"), assignment("X2")], [],
                            [judged("T100-O1", "X1", depth="apply"), judged("T100-O1", "X2", depth="create")])
    assert (GapType.BLOOMS_MISMATCH, "T100-O1") not in kinds(deep_enough)


def test_a_rules_match_has_unknown_depth_so_no_blooms_gap():
    o = outcome(1, BloomLevel.CREATE)
    gaps = find_gaps([o], [assignment("X1"), assignment("X2")], [rules_match("T100-O1", "X1")],
                     [judged("T100-O1", "X2", depth="apply")])
    assert (GapType.BLOOMS_MISMATCH, "T100-O1") not in kinds(gaps)


def test_only_partial_matches_is_weak_coverage():
    gaps = find_gaps([outcome(1)], [assignment("X1")], [], [judged("T100-O1", "X1", kind="partial")])
    assert (GapType.WEAK_COVERAGE, "T100-O1") in kinds(gaps)


def test_not_assessed_outcome_gets_a_note_not_a_gap():
    gaps = find_gaps([outcome(1, not_assessed=True)], [], [], [])
    assert kinds(gaps) == {(GapType.STATED_NOT_ASSESSED, "T100-O1")}


# ── second opinion ───────────────────────────────────────────────────────────

TWICE = replace(A2, second_opinion=True)
NOT_ALIGNED = verdict("tangential", evidence="", level=None)
BAD = verdict(evidence="not in the assignment at all")


def test_with_second_opinion_off_a_pair_is_judged_once():
    llm = scripted(verdict())
    assert judge_pair(SCHEMA, SQL_LAB, llm, A2, system_prompt(A2)).calls == 1


def test_two_judgments_that_agree_keep_the_first_verdict():
    llm = scripted(verdict(kind="direct"), verdict(kind="partial"))      # both say it assesses the outcome
    pair = judge_pair(SCHEMA, SQL_LAB, llm, TWICE, system_prompt(TWICE))
    assert pair.calls == 2 and pair.result.alignment_type == AlignmentType.DIRECT
    assert not pair.result.needs_human_review
    assert llm.calls[1][1][0]["content"].startswith(SECOND_OPINION)     # asked separately, not from cache
    assert pair.log[1]["pair"].endswith("(2nd opinion)")


def test_two_judgments_that_disagree_wait_for_a_person():
    pair = judge_pair(SCHEMA, SQL_LAB, scripted(verdict(), NOT_ALIGNED), TWICE, system_prompt(TWICE))
    assert pair.result.confidence_tier == MatchConfidence.LOW and pair.result.needs_human_review
    assert pair.result.reasoning.startswith(DISAGREE)
    state = {"syllabus": SimpleNamespace(course_id="T100"), "outcomes": [SCHEMA], "held": [],
             "assignments": [SQL_LAB], "judgments": [pair.result]}
    assert review_queue(state)[0].why == "held: two separate judgments disagreed"


def test_a_second_opinion_that_fails_its_checks_leaves_the_first_verdict_flagged():
    pair = judge_pair(SCHEMA, SQL_LAB, scripted(verdict(), BAD, BAD), TWICE, system_prompt(TWICE))
    assert pair.result.alignment_type == AlignmentType.DIRECT
    assert pair.result.confidence_tier == MatchConfidence.MEDIUM and pair.result.needs_human_review
    assert "couldn't be checked" in pair.result.reasoning


def test_a_pair_already_held_gets_no_second_opinion():
    llm = scripted(BAD, BAD)
    pair = judge_pair(SCHEMA, SQL_LAB, llm, TWICE, system_prompt(TWICE))
    assert pair.calls == 2 and len(llm.calls) == 2                       # both were the first judgment's attempts
    assert pair.result.confidence_tier == MatchConfidence.LOW


# ── a model that never answers ───────────────────────────────────────────────

class Down:
    """A model that never answers, reported the way RetryingLLM reports it."""
    name = "down (test)"

    def __init__(self):
        self.calls = 0

    def complete(self, system, messages):
        self.calls += 1
        raise LLMUnavailable("down (test) didn't answer after 7 tries over 95 s. "
                             "Last error: ServerError: 503 UNAVAILABLE")


def test_a_pair_the_model_never_answers_waits_for_a_person():
    model = Down()
    pair = judge_pair(SCHEMA, SQL_LAB, model, A2, system_prompt(A2))
    assert model.calls == 1 and pair.calls == 0          # no second attempt: that would only wait again
    assert pair.log[-1]["unavailable"] and pair.result.reasoning.startswith(NO_REPLY)
    assert pair.result.confidence_tier == MatchConfidence.LOW and pair.result.needs_human_review


def test_when_the_model_is_down_the_run_still_finishes_and_says_what_is_pending():
    course = next(c for c in load_course_data(ROOT / "data" / "syllabi", ROOT / "data" / "assignments")
                  if c["course_id"] == "CS4530")
    reader = FakeLLM.from_file(ROOT / "tests" / "fixtures" / "agent1_fake_replies.json")
    graph = build_graph(RULES, A1, reader, graph_api()[0], agent2_config=A2, judge_llm=Down())
    final = graph.invoke(initial_state(course["syllabus"], course["assignments"]))

    sent = [d for d in final["decisions"] if d.decision == Decision.NEEDS_AGENT]
    assert len(final["judgments"]) == len(sent) > 0
    assert all(j.needs_human_review and j.reasoning.startswith(NO_REPLY) for j in final["judgments"])
    held = {j.outcome_id for j in final["judgments"]}
    untested = [g for g in final["gaps"] if g.gap_type == GapType.UNTESTED_OUTCOME]
    assert all("held for review" in g.description for g in untested if g.outcome_id in held)
    pairs = [item for item in review_queue(final) if item.kind == "pair"]
    assert pairs and all(item.why.startswith("held: the model didn't answer") for item in pairs)


# ── scoring when Agent 1 splits a stated outcome into parts ──────────────────

def test_a_split_outcome_is_scored_the_way_its_label_was_written():
    from coursealign.evaluate import Label, Run, score_by_label, score_pipeline
    from coursealign.models import GapReport, GapSeverity, RawSyllabus
    syllabus = RawSyllabus("T200", "Test", "Objectives\nYou will design programs and test them thoroughly.\n"
                                           "You will work in a team.\n", "t.txt")
    gold = {"T200": [Label("You will design programs and test them thoroughly.", {"A1", "A2"}),
                     Label("You will work in a team.", {"A2"})]}
    part = lambda oid, text, quote: ExtractedOutcome(id=oid, course_id="T200", text=text, evidence=quote)
    outcomes = [part("O1", "Design programs", "design programs"),                 # objective 1, split in two
                part("O2", "Test programs thoroughly", "test them thoroughly"),
                part("O3", "Work in a team", "You will work in a team.")]
    tasks = [Assignment(id=i, course_id="T200", title=i, type="project", description="d",
                        rubric_keywords=[], points=100, week=1) for i in ("A1", "A2")]
    decisions = [TriageDecision(o.id, a.id, Decision.NEEDS_AGENT, [], [], "test") for o in outcomes for a in tasks]
    def said(oid, aid, kind, tier=MatchConfidence.HIGH):
        return AlignmentResult(outcome_id=oid, assignment_id=aid, score=0.9, confidence_tier=tier,
                               alignment_type=kind, reasoning="", evidence="")
    judgments = [said("O1", "A1", AlignmentType.DIRECT), said("O2", "A1", AlignmentType.TANGENTIAL),  # A1: design only
                 said("O1", "A2", AlignmentType.TANGENTIAL), said("O2", "A2", AlignmentType.DIRECT),  # A2: tests only
                 said("O3", "A1", AlignmentType.TANGENTIAL),
                 said("O3", "A2", AlignmentType.TANGENTIAL, MatchConfidence.LOW)]                    # held
    gaps = [GapReport("T200", GapType.UNTESTED_OUTCOME, GapSeverity.CRITICAL, "", outcome_id="O2")]
    run = Run(syllabus, outcomes, decisions, judgments=judgments, gaps=gaps, assignments=tasks)

    per_part, per_label = score_pipeline([run], gold), score_by_label([run], gold)
    assert per_part.decided_correct == 3 and per_part.decided == 5      # two right answers about a part counted wrong
    assert (per_label.decided_correct, per_label.decided, per_label.waiting) == (3, 3, 1) and not per_label.wrong
    assert per_label.split == 1
    assert per_label.untested_false == [] and per_label.finer_alarms == ["O2"]    # O1 is assessed, so not a false alarm


# ── the whole pipeline, offline ──────────────────────────────────────────────

def test_ceiling_run_finds_exactly_the_labeled_gaps():
    courses = stand_in_courses(load_course_data(ROOT / "data" / "syllabi", ROOT / "data" / "assignments"), None)
    gold = load_gold(ROOT / "data" / "gold" / "alignment_labels.json")
    reader = FakeLLM.from_file(ROOT / "tests" / "fixtures" / "agent1_fake_replies.json")
    judge = FakeLLM(responder=LabelJudge({c["course_id"]: c["syllabus"] for c in courses}, gold, RULES))
    graph = build_graph(RULES, A1, reader, graph_api()[0], agent2_config=A2, judge_llm=judge)

    found = set()
    for c in courses:
        final = graph.invoke(initial_state(c["syllabus"], c["assignments"]))
        found |= {(g.gap_type, g.outcome_id or g.assignment_id) for g in final["gaps"]
                  if g.gap_type != GapType.STATED_NOT_ASSESSED}
    assert found == {
        (GapType.UNTESTED_OUTCOME, "CS4550-O5"),      # nothing assesses writing tests
        (GapType.UNTESTED_OUTCOME, "CS4530-O8"),      # nothing assesses TDD
        (GapType.UNTESTED_OUTCOME, "CS4530-O12"),     # nothing assesses monitoring
        (GapType.ORPHAN_ASSESSMENT, "A6"),            # the performance lab
        (GapType.ORPHAN_ASSESSMENT, "REFLECT"),       # the reflection essay
        (GapType.BLOOMS_MISMATCH, "CS3200-O5"),       # NoSQL: asked to evaluate, tested to understand
    }


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")

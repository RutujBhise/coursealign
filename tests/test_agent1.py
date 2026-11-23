"""
Tests for Agent 1 and the graph. They use the fake LLM, so they need no API
key and no network. With LangGraph installed they run on real LangGraph;
without it, on the offline stand-in.

RUN:  pytest tests/   or   python3 tests/test_agent1.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coursealign.agent1 import Agent1Config, check_reply, to_outcomes
from coursealign.evaluate import load_gold
from coursealign.graph import build_graph, graph_api, initial_state
from coursealign.llm import FakeLLM, LLMUnavailable
from coursealign.loader import load_course_data
from coursealign.spans import find_span
from coursealign.stage0 import Rules

RULES = Rules.from_file(ROOT / "data" / "rules" / "stage0_rules.json")
CONFIG = Agent1Config.from_file(ROOT / "data" / "rules" / "agent1.json")
COURSES = {c["course_id"]: c for c in load_course_data(ROOT / "data" / "syllabi", ROOT / "data" / "assignments")}

SOURCE = "Students will master SQL \u2014 not just SELECT statements, but complex\njoins and subqueries."
DB_QUOTE = "produce a normalized schema that handles edge cases"       # really in CS3200's syllabus
PARAPHRASE = "design schemas that are properly normalized"             # not in it


def reply(*items) -> str:
    """Build a JSON reply from (outcome, evidence, confidence, not_assessed) tuples."""
    return json.dumps({"outcomes": [
        {"outcome": o, "evidence": e, "confidence": c, "stated_not_assessed": f} for o, e, c, f in items
    ]})


def run_course(course_id: str, replies: dict):
    llm = FakeLLM(replies)
    graph = build_graph(RULES, CONFIG, llm, graph_api()[0])
    course = COURSES[course_id]
    return graph.invoke(initial_state(course["syllabus"], course["assignments"])), llm


# ── the evidence check ───────────────────────────────────────────────────────

def test_quote_check_forgives_formatting():
    # different case, a hyphen for the em dash, and a line break in the source
    assert find_span("MASTER SQL - not just select statements, but complex joins", SOURCE) is not None


def test_quote_check_rejects_a_paraphrase():
    result = check_reply(reply(("Write SQL", "become an expert at SQL queries", 0.9, False)), SOURCE, CONFIG)
    assert result.candidates == [] and "word for word" in result.problems[0]


def test_bad_json_is_a_problem_not_a_crash():
    result = check_reply("Sure! Here are the outcomes I found...", SOURCE, CONFIG)
    assert result.candidates == [] and result.problems


# ── what code decides, not the agent ─────────────────────────────────────────

def test_confidence_tiers_pass_flag_or_hold():
    result = check_reply(reply(
        ("Write SQL with joins", "complex joins and subqueries", 0.9, False),
        ("Write SQL beyond SELECT", "not just SELECT statements", 0.7, False),
        ("Use SQL", "Students will master SQL", 0.5, False),
    ), SOURCE, CONFIG)
    passed, held = to_outcomes(result.candidates, "T100", 1, CONFIG, RULES)
    assert [o.needs_review for o in passed] == [False, True]
    assert len(held) == 1 and held[0].extraction_confidence == 0.5


def test_topics_come_from_the_quote_not_the_agents_wording():
    result = check_reply(reply(("Write SQL in Docker containers", "complex joins and subqueries", 0.9, False)),
                         SOURCE, CONFIG)
    passed, _ = to_outcomes(result.candidates, "T100", 1, CONFIG, RULES)
    assert "sql" in passed[0].topic_tags and "containers" not in passed[0].topic_tags


# ── the graph ────────────────────────────────────────────────────────────────

def test_graph_retries_a_failed_quote_then_commits():
    bad = reply(("Design normalized schemas", PARAPHRASE, 0.9, False))
    good = reply(("Design normalized schemas", DB_QUOTE, 0.9, False))
    final, llm = run_course("CS3200", {"CS3200": [bad, good]})
    assert final["llm_calls"] == 2
    assert [o.evidence for o in final["outcomes"]] == [DB_QUOTE]
    assert "word for word" in llm.calls[1][1][-1]["content"]     # the retry said what failed


def test_graph_drops_what_still_fails_after_the_last_attempt():
    bad = reply(("Design normalized schemas", PARAPHRASE, 0.9, False))
    final, _ = run_course("CS3200", {"CS3200": [bad] * CONFIG.max_attempts})
    assert final["llm_calls"] == CONFIG.max_attempts
    assert final["outcomes"] == [] and final["agent1_log"][-1]["problems"]


def test_agent1_is_asked_for_every_outcome_and_told_the_quote_length():
    from coursealign.agent1 import system_prompt
    prompt = system_prompt(CONFIG, RULES)
    assert "list every learning outcome" in prompt
    assert "one paragraph or even one sentence can state several" in prompt
    assert f"{CONFIG.min_quote_words} to {CONFIG.max_quote_words} words" in prompt


def test_the_run_stops_when_agent1_gets_no_reply():
    class Down:
        name = "down (test)"

        def complete(self, system, messages):
            raise LLMUnavailable("down (test) didn't answer after 7 tries")

    graph = build_graph(RULES, CONFIG, Down(), graph_api()[0])
    course = COURSES["CS3200"]
    try:
        graph.invoke(initial_state(course["syllabus"], course["assignments"]))
    except LLMUnavailable:
        return
    raise AssertionError("missing outcomes would make the gap report wrong, so the run has to stop")


def test_graph_skips_agent1_when_the_rules_read_everything():
    final, llm = run_course("CS4530", {})
    assert final["llm_calls"] == 0 and llm.calls == [] and len(final["outcomes"]) == 12


def test_agent1_never_sees_the_assignments():
    _, llm = run_course("CS3200", {"CS3200": [reply(("Design normalized schemas", DB_QUOTE, 0.9, False))]})
    system, messages = llm.calls[0]
    seen = system + "\n".join(m["content"] for m in messages)
    for a in COURSES["CS3200"]["assignments"]:
        assert a.description[:40] not in seen, a.id


def test_not_assessed_outcomes_are_kept_but_skip_triage():
    r = reply(("Debug web applications independently", "develop strong debugging skills", 0.85, True))
    final, _ = run_course("CS4550", {"CS4550": [r]})
    debug = [o for o in final["outcomes"] if o.stated_not_assessed]
    assert len(debug) == 1
    assert all(d.outcome_id != debug[0].id for d in final["decisions"])


# ── the labels themselves ────────────────────────────────────────────────────

def test_every_label_quote_is_in_its_syllabus():
    for course_id, labels in load_gold(ROOT / "data" / "gold" / "alignment_labels.json").items():
        text = COURSES[course_id]["syllabus"].raw_text
        for label in labels:
            assert find_span(label.quote, text) is not None, (course_id, label.quote)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")

"""
Unit tests for Stage 0.

RUN:  pytest tests/          (if you have pytest)
      python3 tests/test_stage0.py   (no install needed)

These use the real rules file on purpose: if an edit to stage0_rules.json
breaks a core behavior, a test fails before the pipeline ever runs.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coursealign.models import Assignment, BloomLevel, ExtractedOutcome, RawSyllabus
from coursealign.stage0 import (
    Decision, Rules, classify_bloom, extract_outcomes, tag_text, triage, triage_pair,
)

RULES = Rules.from_file(ROOT / "data" / "rules" / "stage0_rules.json")


def syllabus(text: str) -> RawSyllabus:
    return RawSyllabus(course_id="T100", title="Test", raw_text=text, file_path="-")


def outcome(text: str) -> ExtractedOutcome:
    return ExtractedOutcome(id="T100-O1", course_id="T100", text=text,
                            topic_tags=sorted(tag_text(text, RULES)))


def assignment(title: str = "", rubric: tuple = (), description: str = "") -> Assignment:
    return Assignment(id="X1", course_id="T100", title=title, type="assignment",
                      description=description, rubric_keywords=list(rubric), points=100, week=1)


# ── tagging ──────────────────────────────────────────────────────────────────

def test_terms_match_whole_words_only():
    assert "testing" not in tag_text("the latest contest results", RULES)
    assert "testing" in tag_text("write unit tests", RULES)


def test_synonyms_map_to_one_topic():
    assert tag_text("JWT and OAuth", RULES) == {"authentication"}


# ── Bloom's ──────────────────────────────────────────────────────────────────

def test_bloom_keeps_the_higher_of_two_verbs():
    assert classify_bloom("Design and implement RESTful APIs", RULES) == BloomLevel.CREATE


def test_bloom_unknown_verb_stays_unclassified():
    assert classify_bloom("Practical experience decomposing requirements", RULES) is None


def test_bloom_skips_students_will_be_able_to():
    assert classify_bloom("Students will be able to evaluate tradeoffs", RULES) == BloomLevel.EVALUATE


# ── extraction ───────────────────────────────────────────────────────────────

def test_list_items_become_outcomes_and_prose_goes_to_agent():
    text = (
        "LEARNING OUTCOMES\n"
        "Students will be able to:\n"
        "1. Build REST APIs\n"
        "2. Write unit tests\n"
        "Students will also grow as communicators over the term.\n"
        "\n"
        "GRADING POLICY\n"
        "1. Homework 40%\n"
    )
    outcomes, unresolved = extract_outcomes(syllabus(text), RULES)
    assert [o.text for o in outcomes] == ["Build REST APIs", "Write unit tests"]
    assert len(unresolved) == 1 and "communicators" in unresolved[0].text


def test_prose_only_section_goes_to_agent_as_one_block():
    text = "Course Goals:\nWe want you to master SQL.\n\nAnd design schemas.\n\nPrerequisites: CS 2500\n"
    outcomes, unresolved = extract_outcomes(syllabus(text), RULES)
    assert outcomes == []
    assert len(unresolved) == 1        # one region = one Agent 1 call


def test_text_outside_outcomes_sections_is_ignored():
    outcomes, unresolved = extract_outcomes(syllabus("SCHEDULE\n- Week 1: HTML\n- Week 2: CSS\n"), RULES)
    assert outcomes == [] and unresolved == []


# ── triage ───────────────────────────────────────────────────────────────────

def test_two_graded_topics_auto_match():
    o = outcome("Apply authentication and authorization patterns (JWT, OAuth)")
    a = assignment(title="Auth Implementation", rubric=("JWT", "protected routes"))
    assert triage_pair(o, a, RULES).decision == Decision.AUTO_MATCH


def test_one_shared_topic_goes_to_agent():
    o = outcome("Build responsive user interfaces using React and modern CSS")
    a = assignment(title="Task Manager", rubric=("React", "useState"))
    assert triage_pair(o, a, RULES).decision == Decision.NEEDS_AGENT


def test_description_alone_can_never_auto_match():
    o = outcome("Design and implement RESTful APIs using Node.js and Express")
    a = assignment(title="Auth", rubric=("JWT",),
                   description="Add login to your Express REST API endpoints")
    assert triage_pair(o, a, RULES).decision == Decision.NEEDS_AGENT


def test_nothing_shared_auto_rejects():
    o = outcome("Deploy applications to cloud platforms")
    a = assignment(title="Portfolio", rubric=("CSS Grid", "Flexbox"))
    assert triage_pair(o, a, RULES).decision == Decision.AUTO_REJECT


def test_rules_never_reject_an_outcome_they_cannot_read():
    o = outcome("Grow as a thoughtful practitioner")      # no known topics
    a = assignment(title="Portfolio", rubric=("CSS Grid",))
    assert triage_pair(o, a, RULES).decision == Decision.NEEDS_AGENT



def test_a_hyphen_before_a_term_blocks_it():
    assert "relational-db" not in tag_text("non-relational approaches", RULES)
    assert "relational-db" in tag_text("relational approaches", RULES)


def test_topics_common_across_a_course_cannot_auto_match():
    o = outcome("Write SQL against a relational database")           # sql, relational-db
    course = [
        assignment(title="SQL Lab", rubric=("SQL", "PostgreSQL")),    # shares both topics
        assignment(title="Joins", rubric=("SQL", "joins"), description="Query the database"),
        assignment(title="Schemas", rubric=("schema design",), description="Use SQL on the database"),
        assignment(title="Portfolio", rubric=("CSS Grid",)),
    ]
    # sql and relational-db appear in 3 of 4 assignments: background, not evidence
    assert triage([o], course, RULES)[0].decision == Decision.NEEDS_AGENT
    # the same pair on its own, with nothing to compare against, still auto-matches
    assert triage_pair(o, course[0], RULES).decision == Decision.AUTO_MATCH

def test_a_test_set_can_be_run_on_its_own_and_a_typo_is_caught():
    from coursealign.loader import load_course_data, pick_courses
    courses = load_course_data(ROOT / "data" / "syllabi", ROOT / "data" / "assignments")
    assert [c["course_id"] for c in pick_courses(courses, "cs4530, CS4550")] == ["CS4530", "CS4550"]
    assert pick_courses(courses, None) == courses
    try:
        pick_courses(courses, "MIT9999")
    except SystemExit as stop:
        assert "MIT9999" in str(stop) and "CS3200" in str(stop)
        return
    raise AssertionError("an unknown course should stop with the list of real ones")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} passed")

"""
Step 3: rules + Agent 1, wired together as a LangGraph state machine.

RUN:
  python3 step3_agent1.py                   fake LLM: hand-written replies, no API key
  python3 step3_agent1.py --llm anthropic   real model: needs ANTHROPIC_API_KEY
  python3 step3_agent1.py --llm gemini      Gemini, which has a free tier: needs GEMINI_API_KEY

  pip install langgraph anthropic
  (The fake run needs neither. Without LangGraph it uses a small stand-in
   that runs the same graph; install LangGraph to run the real thing.)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from coursealign.agent1 import Agent1Config
from coursealign.evaluate import Run, label_runs, load_gold, score_runs
from coursealign.graph import build_graph, graph_api, initial_state
from coursealign.llm import FakeLLM, LLMUnavailable, build_llms, stopped_message
from coursealign.loader import load_course_data, pick_courses, stand_in_courses
from coursealign.models import ExtractionSource
from coursealign.stage0 import Decision, Rules, Stage0Result, rule_flagged_gaps


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def banner(text: str) -> None:
    print(f"\n{'=' * 64}\n  {text}\n{'=' * 64}")


def show_course(final: dict, config: Agent1Config) -> None:
    syllabus = final["syllabus"]
    cid = syllabus.course_id
    from_rules = [o for o in final["outcomes"] if o.source == ExtractionSource.REGEX]
    from_agent = [o for o in final["outcomes"] if o.source == ExtractionSource.AGENT]
    print(f"\n[{cid}] {syllabus.title}")
    print(f"  Rules extracted {len(from_rules)} outcomes")

    if not final["agent1_log"]:
        print("  Agent 1: skipped, no prose to read")
    for entry in final["agent1_log"]:
        status = f"{len(entry['problems'])} failed a check" if entry["problems"] else "all passed the checks"
        print(f"  Agent 1, attempt {entry['attempt']}: returned {entry['returned']}, {status}")
        for problem in entry["problems"]:
            print(f"    x {problem}")

    if from_agent:
        print("  Passed from Agent 1:")
    for o in from_agent:
        bloom = o.bloom_level.value if o.bloom_level else "unclassified"
        tags = []
        if o.needs_review:
            tags.append("REVIEW")
        if o.stated_not_assessed:
            tags.append("NOT ASSESSED (per syllabus)")
        print(f"    {o.id:<10} [{bloom}] confidence {o.extraction_confidence:.2f} {' '.join(tags)}".rstrip())
        print(f"    {'':<10} {o.text}")
        print(f"    {'':<10} quote:  \"{o.evidence}\"")
        print(f"    {'':<10} topics: {', '.join(o.topic_tags) or '(none)'}")
    if final["held"]:
        print(f"  Held for human review (confidence below {config.pass_with_review_at}):")
    for o in final["held"]:
        print(f"    {o.id:<10} confidence {o.extraction_confidence:.2f}  {o.text}")

    decisions = final["decisions"]
    counts = {d: sum(1 for x in decisions if x.decision == d) for d in Decision}
    print(f"  Triage of {len(decisions)} pairs: {counts[Decision.AUTO_MATCH]} auto-match, "
          f"{counts[Decision.AUTO_REJECT]} auto-reject, {counts[Decision.NEEDS_AGENT]} need the agent")
    agent_ids = {o.id for o in from_agent}
    for d in decisions:
        if d.outcome_id in agent_ids and d.decision != Decision.AUTO_REJECT:
            label = "match" if d.decision == Decision.AUTO_MATCH else "agent"
            print(f"    {label}  {d.outcome_id:<10} x {d.assignment_id:<8} {d.reason}")

    untested, orphans = rule_flagged_gaps(Stage0Result(cid, final["outcomes"], [], decisions))
    for oid in untested:
        print(f"  GAP (zero LLM calls): no assignment touches {oid}")
    for aid in orphans:
        print(f"  GAP (zero LLM calls): {aid} touches no outcome")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["fake", "anthropic", "gemini"], default="fake")
    parser.add_argument("--courses", help="only these courses, like MIT6005,MIT6170 (default: all)")
    args = parser.parse_args()

    rules = Rules.from_file(ROOT / "data" / "rules" / "stage0_rules.json")
    config = Agent1Config.from_file(ROOT / "data" / "rules" / "agent1.json")
    gold = load_gold(ROOT / "data" / "gold" / "alignment_labels.json")
    courses = [c for c in load_course_data(ROOT / "data" / "syllabi", ROOT / "data" / "assignments")
               if c["syllabus"] is not None]
    courses = pick_courses(courses, args.courses)
    if args.llm == "fake":
        courses = stand_in_courses(courses, args.courses)  # the stand-ins only know the practice courses

    fake = args.llm == "fake"
    llm = FakeLLM.from_file(ROOT / "tests" / "fixtures" / "agent1_fake_replies.json") if fake else build_llms(args.llm)[0]
    api, engine = graph_api()
    graph = build_graph(rules, config, llm, api)

    banner("STEP 3: RULES + AGENT 1, AS A GRAPH")
    print(f"  Graph engine: {engine}")
    print(f"  LLM: {llm.name}")
    if fake:
        print("  NOTE: the fake LLM replays hand-written replies, including one bad")
        print("        quote on purpose. It tests the code, not a model.")

    finals = []
    for course in courses:
        final = graph.invoke(initial_state(course["syllabus"], course["assignments"]))
        finals.append(final)
        show_course(final, config)

    # Agent 1 totals
    log = [entry for f in finals for entry in f["agent1_log"]]
    passed = [o for f in finals for o in f["outcomes"] if o.source == ExtractionSource.AGENT]
    held = [o for f in finals for o in f["held"]]
    banner("AGENT 1 TOTALS")
    print(f"  LLM calls: {sum(f['llm_calls'] for f in finals)}"
          f" for {sum(1 for e in log if e['attempt'] == 1)} prose regions (any extra calls are retries)")
    print(f"  Problems caught by the checks: {sum(len(e['problems']) for e in log)}")
    print(f"  Outcomes: {len(passed)} passed ({sum(o.needs_review for o in passed)} flagged for review),"
          f" {len(held)} held for review")

    # Scored against hand labels
    s = score_runs([Run(f["syllabus"], f["outcomes"], f["decisions"], f["held"]) for f in finals], gold)
    banner("MEASURED AGAINST HAND LABELS")
    if fake:
        print("  (fake LLM: these numbers check the scoring code, not a model)")
    print(f"  Labeled outcomes found: {s.labeled_found}/{s.labeled}")
    print(f"  Passed outcomes that match a label: {s.grounded}/{s.extracted}"
          + (f"  (no label: {', '.join(s.ungrounded)})" if s.ungrounded else ""))
    print(f"  Agent 1 'not assessed' flags right: {s.flags_correct}/{s.flags_checked}")
    print(f"  Triage: {s.pairs} pairs scored, {s.decided} decided by rules ({pct(s.decided, s.pairs)}),"
          f" {s.needs_agent} left for Agent 2")
    print(f"  Rule decisions correct: {s.decided_correct}/{s.decided}"
          f"  (wrong matches: {len(s.wrong_matches)}, missed: {len(s.missed)})")

    # Triage on the labels alone: no extraction, no model, same result every run
    t = score_runs(label_runs([(c["syllabus"], c["assignments"]) for c in courses], gold, rules), gold)
    banner("TRIAGE ON THE LABELED OUTCOMES (no extraction, no model)")
    print(f"  Pairs: {t.pairs}  |  decided by rules: {t.decided} ({pct(t.decided, t.pairs)})"
          f"  |  left for Agent 2: {t.needs_agent}")
    print(f"  Rule decisions correct: {t.decided_correct}/{t.decided} ({pct(t.decided_correct, t.decided)})"
          f"  (wrong matches: {len(t.wrong_matches)}, missed: {len(t.missed)})")
    for oid, aid in t.wrong_matches:
        print(f"    wrong match: {oid} x {aid}")
    for oid, aid in t.missed:
        print(f"    missed:      {oid} x {aid}")
    print("  Any mistake here belongs to the rules, not to an agent.")
    print()


if __name__ == "__main__":
    try:
        main()
    except LLMUnavailable as error:
        raise SystemExit(stopped_message(error))

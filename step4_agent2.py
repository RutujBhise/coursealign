"""
Step 4: Agent 2 judges the pairs the rules couldn't decide, then gap
analysis turns every verdict into the report a teacher reads.

RUN:
  python3 step4_agent2.py                   fake agents, no API key needed
  python3 step4_agent2.py --llm anthropic   real model for both agents
                                            (about 50 calls; needs ANTHROPIC_API_KEY)
  python3 step4_agent2.py --llm gemini      the same with Gemini's free tier

Real replies are saved in .cache/llm_replies.json (shared with Step 5 and the
dashboard), so a run cut short by a rate limit picks up where it stopped.
Add --fresh to ask the model everything again.

Add --second-opinion to judge every pair twice, separately: a pair whose two
judgments disagree waits for a person. It doubles Agent 2's requests.
Set COURSEALIGN_AGENT1_MODEL to give Agent 1 a different (stronger) model.
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from coursealign.agent1 import Agent1Config
from coursealign.agent2 import Agent2Config
from coursealign.evaluate import Run, load_gold, score_pipeline, score_runs
from coursealign.graph import build_graph, graph_api, initial_state
from coursealign.llm import CachedLLM, FakeLLM, LLMUnavailable, build_llms, stopped_message
from coursealign.loader import load_course_data, pick_courses
from coursealign.models import AlignmentType, ExtractionSource, MatchConfidence
from coursealign.oracle import LabelJudge
from coursealign.stage0 import Decision, Rules

TIER_MARK = {MatchConfidence.HIGH: "", MatchConfidence.MEDIUM: "REVIEW", MatchConfidence.LOW: "HELD"}


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def banner(text: str) -> None:
    print(f"\n{'=' * 64}\n  {text}\n{'=' * 64}")


def no_reply(finals: list[dict]) -> list[dict]:
    """Agent 2 log entries where the model never answered, even after retries."""
    return [entry for f in finals for entry in f["agent2_log"] if entry.get("unavailable")]


def show_course(final: dict) -> None:
    syllabus = final["syllabus"]
    outcomes = {o.id: o for o in final["outcomes"]}
    titles = {a.id: a.title for a in final["assignments"]}
    from_rules = sum(1 for o in outcomes.values() if o.source == ExtractionSource.REGEX)
    decisions = final["decisions"]

    def count(kind: Decision) -> int:
        return sum(1 for d in decisions if d.decision == kind)

    print(f"\n[{syllabus.course_id}] {syllabus.title}")
    print(f"  Outcomes: {from_rules} from rules, {len(outcomes) - from_rules} from Agent 1,"
          f" {len(final['held'])} held for review")
    print(f"  Triage: {count(Decision.AUTO_MATCH)} auto-match, {count(Decision.AUTO_REJECT)} auto-reject,"
          f" {count(Decision.NEEDS_AGENT)} sent to Agent 2 ({len(final['agent2_log'])} calls)")
    for entry in final["agent2_log"]:
        if entry.get("unavailable"):
            continue                       # summed up in one line below
        for problem in entry["problems"]:
            print(f"    x {entry['pair']}, attempt {entry['attempt']}: {problem}")
    missing = no_reply([final])
    if missing:
        print(f"    ! {len(missing)} pair{'s' if len(missing) != 1 else ''} got no reply from the model,"
              f" so {'they wait' if len(missing) != 1 else 'it waits'} for a person (marked HELD below)")

    confident_no = 0
    for j in final["judgments"]:
        if j.alignment_type == AlignmentType.TANGENTIAL and j.confidence_tier == MatchConfidence.HIGH:
            confident_no += 1
            continue
        depth = j.assessed_level.value if j.assessed_level else "-"
        quote = f'"{j.evidence}"' if j.evidence else ""
        print(f"    {j.alignment_type.value:<10} {j.outcome_id:<10} x {j.assignment_id:<8}"
              f" {j.score:.2f} {TIER_MARK[j.confidence_tier]:<6} {depth:<10} {quote}".rstrip())
    if confident_no:
        print(f"    ...plus {confident_no} pairs judged tangential with high confidence")

    print("  Gap report:")
    if not final["gaps"]:
        print("    (no gaps)")
    for g in final["gaps"]:
        if g.outcome_id:
            what = f"{g.outcome_id} \"{outcomes[g.outcome_id].text}\""
        else:
            what = f"{g.assignment_id} \"{titles[g.assignment_id]}\""
        print(f"    {g.severity.value.upper():<8} {g.gap_type.value:<20} {what}")
        print(f"    {'':<8} {g.description}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["fake", "anthropic", "gemini"], default="fake")
    parser.add_argument("--courses", help="only these courses, like MIT6005,MIT6170 (default: all)")
    parser.add_argument("--fresh", action="store_true", help="ask the model everything again, not saved replies")
    parser.add_argument("--second-opinion", action="store_true",
                        help="judge every pair twice; a disagreement waits for a person")
    args = parser.parse_args()

    rules = Rules.from_file(ROOT / "data" / "rules" / "stage0_rules.json")
    agent1_config = Agent1Config.from_file(ROOT / "data" / "rules" / "agent1.json")
    agent2_config = Agent2Config.from_file(ROOT / "data" / "rules" / "agent2.json")
    if args.second_opinion:
        agent2_config = replace(agent2_config, second_opinion=True)
    gold = load_gold(ROOT / "data" / "gold" / "alignment_labels.json")
    courses = [c for c in load_course_data(ROOT / "data" / "syllabi", ROOT / "data" / "assignments")
               if c["syllabus"] is not None]
    courses = pick_courses(courses, args.courses)

    fake = args.llm == "fake"
    if fake:
        reader = FakeLLM.from_file(ROOT / "tests" / "fixtures" / "agent1_fake_replies.json")
        judge = FakeLLM(
            responder=LabelJudge({c["course_id"]: c["syllabus"] for c in courses}, gold, rules),
            name="fake (answers from the hand labels, with 4 planted problems)",
        )
    else:
        cache = ROOT / ".cache" / "llm_replies.json"
        if args.fresh:
            cache.unlink(missing_ok=True)
        model_for_agent1, model_for_agent2 = build_llms(args.llm)
        reader = CachedLLM(model_for_agent1, cache)       # both share one file's saved replies
        judge = reader if model_for_agent2 is model_for_agent1 else CachedLLM(model_for_agent2, cache)
    api, engine = graph_api()
    graph = build_graph(rules, agent1_config, reader, api, agent2_config=agent2_config, judge_llm=judge)

    banner("STEP 4: AGENT 2 + GAP ANALYSIS")
    print(f"  Graph engine: {engine}")
    print(f"  Agent 1: {reader.name}")
    print(f"  Agent 2: {judge.name}")
    if agent2_config.second_opinion:
        print("  Second opinion: on. Each pair is judged twice; a disagreement waits for a person.")
    if fake:
        print("  NOTE: the fake Agent 2 reads its answers off the hand labels. This run")
        print("        shows the pipeline's ceiling and tests the code, not a model.")

    finals = [graph.invoke(initial_state(c["syllabus"], c["assignments"])) for c in courses]
    for final in finals:
        show_course(final)

    missing = no_reply(finals)
    if missing:
        banner("INCOMPLETE RUN: DON'T QUOTE THESE NUMBERS")
        print(f"  {len(missing)} of {sum(len(f['judgments']) for f in finals)} pairs got no reply from the model,"
              " even after retries.")
        print("  They're held for a person, so the scores below leave them out. Run this again")
        print("  when the model is less busy. The first failure:")
        print(f"    {missing[0]['problems'][0]}")

    # What the rules saved
    agent1_calls = sum(len(f["agent1_log"]) for f in finals)
    agent2_calls = sum(len(f["agent2_log"]) for f in finals)
    per_pair = 2 if agent2_config.second_opinion else 1        # the same judging, without rules to triage
    without_rules = len(finals) + per_pair * sum(
        sum(1 for o in f["outcomes"] if not o.stated_not_assessed) * len(f["assignments"]) for f in finals
    )
    pairs = sum(len(f["decisions"]) for f in finals)
    by_rules = sum(1 for f in finals for d in f["decisions"] if d.decision != Decision.NEEDS_AGENT)
    banner("LLM CALLS")
    print(f"  Pairs: {pairs}   settled by the rules before any LLM call: {by_rules} ({pct(by_rules, pairs)})")
    print(f"  Agent 1: {agent1_calls}   Agent 2: {agent2_calls}   total: {agent1_calls + agent2_calls}")
    print(f"  With no rules, every syllabus to Agent 1 and every pair to Agent 2: {without_rules}")
    print(f"  Calls the rules saved: {pct(without_rules - agent1_calls - agent2_calls, without_rules)}")
    models = list({id(m): m for m in (reader, judge)}.values())
    if all(hasattr(m, "paid_calls") for m in models):
        paid, answered = sum(m.paid_calls for m in models), agent1_calls + agent2_calls - len(missing)
        print(f"  Model calls this run: {paid}   (the other {answered - paid} replies were saved from earlier runs)")
    retried = sum(getattr(m, "retries", 0) for m in models)
    if retried:
        print(f"  Temporary errors (busy model, rate limit) retried: {retried}")

    # What's waiting for a person
    review_outcomes = sum(len(f["held"]) + sum(o.needs_review for o in f["outcomes"]) for f in finals)
    review_pairs = sum(sum(j.needs_human_review for j in f["judgments"]) for f in finals)
    banner("WAITING FOR A HUMAN (Step 5)")
    print(f"  {review_outcomes} outcomes and {review_pairs} pairs are flagged or held for review")

    # Scored against hand labels
    runs = [Run.from_state(f) for f in finals]
    s, found = score_pipeline(runs, gold), score_runs(runs, gold)
    banner("MEASURED AGAINST HAND LABELS")
    if fake:
        print("  (fake agents: this tests the plumbing and shows the ceiling, not a model)")
    if missing:
        print(f"  (incomplete: {len(missing)} pairs had no reply from the model, see above)")
    print(f"  Labeled outcomes found (rules + Agent 1): {found.labeled_found}/{found.labeled}"
          "   (a missed outcome makes its assignments look like orphans)")
    print(f"  Agent 2 verdicts that passed: {s.accepted_correct}/{s.accepted} right"
          f"  ({s.flagged} flagged for review, {s.held} held)")
    for outcome_id, assignment_id, verdict in s.wrong:
        print(f"    wrong: {outcome_id} x {assignment_id}, judged {verdict}")
    print(f"  Every pair, rules and Agent 2 together: {s.decided_correct}/{s.decided} right,"
          f" {s.pairs - s.decided} waiting for review")
    print(f"  Untested outcomes found: {s.untested_found}/{s.untested_expected}"
          f"  (false alarms: {', '.join(s.untested_false) or 'none'})")
    print(f"  Orphan assignments found: {s.orphans_found}/{s.orphans_expected}"
          f"  (false alarms: {', '.join(s.orphans_false) or 'none'})")
    print("  Bloom's gaps: reported, not scored. The labels only mark one, in CS3200.")
    print()


if __name__ == "__main__":
    try:
        main()
    except LLMUnavailable as error:
        raise SystemExit(stopped_message(error, saved=True))

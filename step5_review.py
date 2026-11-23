"""
Step 5: human review and the feedback loop.

Round 1 runs the pipeline and lists what needs a person. A person decides.
Round 2 runs again with those decisions applied. Decisions are saved, so
every later run applies them too, and nobody gets asked the same thing twice.

RUN:
  python3 step5_review.py                   offline demo, using the example decisions
  python3 step5_review.py --interactive     you decide; saved to data/review/decisions.json
  python3 step5_review.py --llm anthropic   the real model (add --interactive to decide yourself)

Replies are cached between rounds, so round 2 differs from round 1 only
where a decision changed something, never because a model answered
differently the second time.
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
from coursealign.loader import load_course_data, stand_in_courses
from coursealign.oracle import LabelJudge
from coursealign.review import (
    PAIR_DECISIONS, correction, decide, learned_terms, load_reviews, review_queue, save_reviews,
)
from coursealign.stage0 import Rules

RULES = ROOT / "data" / "rules" / "stage0_rules.json"
DECISIONS = ROOT / "data" / "review" / "decisions.json"
EXAMPLE = ROOT / "data" / "review" / "example_decisions.json"


def banner(text: str) -> None:
    print(f"\n{'=' * 64}\n  {text}\n{'=' * 64}")


def shorten(text: str, n: int = 50) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


# ── one round ────────────────────────────────────────────────────────────────

def paid_calls(reader, judge) -> int:
    return reader.paid_calls + (0 if judge is reader else judge.paid_calls)


def run_round(courses, reviews, reader, judge, api, agent1_config, agent2_config, gold) -> dict:
    rules = Rules.from_file(RULES, learned=learned_terms(reviews))
    graph = build_graph(rules, agent1_config, reader, api,
                        agent2_config=agent2_config, judge_llm=judge, reviews=reviews)
    before = paid_calls(reader, judge)
    finals = [graph.invoke(initial_state(c["syllabus"], c["assignments"])) for c in courses]
    runs = [Run.from_state(f) for f in finals]
    return {
        "finals": finals,
        "queue": [item for f in finals for item in review_queue(f)],
        "requests": sum(len(f["agent1_log"]) + len(f["agent2_log"]) for f in finals),
        "paid": paid_calls(reader, judge) - before,
        "score": score_pipeline(runs, gold),
        "rule_misses": len(score_runs(runs, gold).missed),
        "gaps": {(g.gap_type.value, g.outcome_id or g.assignment_id) for f in finals for g in f["gaps"]},
        "log": [entry for f in finals for entry in f["review_log"]],
        "learned": learned_terms(reviews),
        "no_reply": sum(1 for f in finals for entry in f["agent2_log"] if entry.get("unavailable")),
    }


def show_item(n: int, item) -> None:
    if item.kind == "outcome":
        print(f"    {n}. outcome  {item.outcome.id:<16} {item.why}")
        print(f"       \"{item.outcome.text}\"")
    else:
        pair = f"{item.outcome.id} x {item.assignment.id}"
        print(f"    {n}. pair     {pair:<16} {item.why}")
        print(f"       \"{shorten(item.outcome.text, 60)}\" / \"{item.assignment.title}\"")


def show_round(title: str, r: dict) -> None:
    banner(title)
    s = r["score"]
    if r["log"]:
        landed = sum(1 for e in r["log"] if e["applied_to"])
        print(f"  Saved decisions applied: {landed}   matched nothing this run: {len(r['log']) - landed}")
    for topic, terms in r["learned"].items():
        print(f"  Words the rules learned from reviews: {', '.join(repr(t) for t in terms)} -> {topic}")
    print(f"  Agent requests: {r['requests']}   paid model calls: {r['paid']}"
          f"   waiting for a person: {len(r['queue'])}")
    print(f"  Every pair: {s.decided_correct}/{s.decided} right, {s.pairs - s.decided} undecided,"
          f" {s.by_human} decided by a person   |   rule misses: {r['rule_misses']}")
    if r["no_reply"]:
        print(f"  ! {r['no_reply']} pairs got no reply from the model, even after retries. They're in the")
        print("    queue, but you don't have to decide them: run again later and the model gets")
        print("    asked again. Every answer so far is cached, so a re-run only asks for what's missing.")
    print("  Review queue:" + ("" if r["queue"] else " empty"))
    for n, item in enumerate(r["queue"], 1):
        show_item(n, item)


# ── the review in between ────────────────────────────────────────────────────

def show_decisions(reviews, where: str) -> None:
    banner(f"THE REVIEW ({where})")
    for r in reviews:
        target = f"{r.course_id} {r.assignment_id}".strip()
        taught = "".join(f"   teaches '{t['term']}' -> {t['topic']}" for t in r.teach)
        print(f"    {r.decision:<12} {r.kind:<8} {target:<11} \"{shorten(r.quote, 44)}\"{taught}")


def read(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def ask(prompt: str, options: dict):
    while True:
        answer = read(prompt).lower()
        if answer == "":
            return None                               # Enter skips
        if answer in options:
            return options[answer]
        print(f"       type one of: {', '.join(options)}, or press Enter to skip")


def interactive_review(round1: dict, saved: list, topics: set[str]) -> list:
    new = []

    def record(decision) -> None:
        new.append(decision)
        save_reviews(DECISIONS, saved + new)         # saved as you go

    banner("YOUR REVIEW (saved to data/review/decisions.json as you go)")
    for n, item in enumerate(round1["queue"], 1):
        show_item(n, item)
        if item.kind == "outcome":
            choice = ask("       keep (k), drop (d), or Enter to skip: ", {"k": "keep", "d": "drop"})
        else:
            print(f"       Agent 2's reason: {item.verdict.reasoning}")
            choice = ask("       direct (d), partial (p), not aligned (n), or Enter to skip: ",
                         {"d": "direct", "p": "partial", "n": "not_aligned"})
        if choice:
            record(decide(item, choice, read("       note (optional): ")))

    # Any other pair can be corrected too, like ones the rules got wrong.
    outcomes = {o.id: o for f in round1["finals"] for o in f["outcomes"]}
    assignments = {(f["syllabus"].course_id, a.id) for f in round1["finals"] for a in f["assignments"]}
    while True:
        line = read("\n  Correct another pair? OUTCOME_ID ASSIGNMENT_ID direct|partial|not_aligned"
                    " (Enter to finish): ")
        if not line:
            break
        parts = line.split()
        if len(parts) != 3 or parts[0] not in outcomes or parts[2] not in PAIR_DECISIONS:
            print("    Couldn't read that. Example: CS4530-O6 IA1 partial")
            continue
        outcome = outcomes[parts[0]]
        if (outcome.course_id, parts[1]) not in assignments:
            print(f"    {outcome.course_id} has no assignment {parts[1]}")
            continue
        note = read("    note (optional): ")
        teach = []
        word = read("    teach the rules a word for this? topic=word (Enter to skip): ")
        if word:
            topic, _, term = (piece.strip() for piece in word.partition("="))
            if topic in topics and term:
                teach = [{"topic": topic, "term": term.lower()}]
            else:
                print(f"    Skipped. Topics are: {', '.join(sorted(topics))}")
        record(correction(outcome, parts[1], parts[2], note, teach))
    return new


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["fake", "anthropic", "gemini"], default="fake")
    parser.add_argument("--interactive", action="store_true", help="make the decisions yourself")
    parser.add_argument("--second-opinion", action="store_true",
                        help="judge every pair twice; a disagreement waits for a person")
    args = parser.parse_args()

    agent1_config = Agent1Config.from_file(ROOT / "data" / "rules" / "agent1.json")
    agent2_config = Agent2Config.from_file(ROOT / "data" / "rules" / "agent2.json")
    if args.second_opinion:
        agent2_config = replace(agent2_config, second_opinion=True)
    gold = load_gold(ROOT / "data" / "gold" / "alignment_labels.json")
    courses = [c for c in load_course_data(ROOT / "data" / "syllabi", ROOT / "data" / "assignments")
               if c["syllabus"] is not None]
    if args.llm == "fake":
        courses = stand_in_courses(courses, None)          # the stand-ins only know the practice courses
    base_rules = Rules.from_file(RULES)

    if args.llm == "fake":
        reader = CachedLLM(FakeLLM.from_file(ROOT / "tests" / "fixtures" / "agent1_fake_replies.json"))
        judge = CachedLLM(FakeLLM(
            responder=LabelJudge({c["course_id"]: c["syllabus"] for c in courses}, gold, base_rules),
            name="fake (answers from the hand labels)",
        ))
    else:
        model_for_agent1, model_for_agent2 = build_llms(args.llm)
        reader = CachedLLM(model_for_agent1, ROOT / ".cache" / "llm_replies.json")
        judge = reader if model_for_agent2 is model_for_agent1 else CachedLLM(model_for_agent2, ROOT / ".cache" / "llm_replies.json")
    api, engine = graph_api()

    banner("STEP 5: HUMAN REVIEW + FEEDBACK LOOP")
    print(f"  Graph engine: {engine}")
    print(f"  Agent 1: {reader.name}")
    print(f"  Agent 2: {judge.name}")
    saved = load_reviews(DECISIONS) if args.interactive else []
    if args.interactive:
        print(f"  Decisions saved before today: {len(saved)}")
    else:
        print("  Demo: round 1 with no decisions, round 2 with data/review/example_decisions.json")

    round1 = run_round(courses, saved, reader, judge, api, agent1_config, agent2_config, gold)
    show_round("ROUND 1: before the review", round1)

    if args.interactive:
        reviews = saved + interactive_review(round1, saved, set(base_rules.tag_patterns))
    else:
        reviews = load_reviews(EXAMPLE)
        show_decisions(reviews, "example decisions")

    round2 = run_round(courses, reviews, reader, judge, api, agent1_config, agent2_config, gold)
    show_round("ROUND 2: with the decisions", round2)

    banner("BEFORE -> AFTER")
    s1, s2 = round1["score"], round2["score"]
    rows = [
        ("Agent requests", round1["requests"], round2["requests"]),
        ("Paid model calls", round1["paid"], round2["paid"]),
        ("Waiting for a person", len(round1["queue"]), len(round2["queue"])),
        ("Pairs decided by a person", s1.by_human, s2.by_human),
        ("Rule misses", round1["rule_misses"], round2["rule_misses"]),
        ("Every pair right", f"{s1.decided_correct}/{s1.decided}", f"{s2.decided_correct}/{s2.decided}"),
        ("Undecided pairs", s1.pairs - s1.decided, s2.pairs - s2.decided),
    ]
    for name, before, after in rows:
        print(f"  {name:<28} {str(before):>8}  ->  {after}")
    added, removed = round2["gaps"] - round1["gaps"], round1["gaps"] - round2["gaps"]
    if not added and not removed:
        print(f"  Gap report                   unchanged ({len(round1['gaps'])} entries)")
    for kind, target in sorted(added):
        print(f"  Gap report: new      {kind} {target}")
    for kind, target in sorted(removed):
        print(f"  Gap report: resolved {kind} {target}")
    if not args.interactive and args.llm == "fake":
        print("\n  NOTE: the example decisions agree with the hand labels, so the accuracy gain")
        print("        is by construction. The real result: decided things stay decided.")
    print()


if __name__ == "__main__":
    try:
        main()
    except LLMUnavailable as error:
        raise SystemExit(stopped_message(error, saved=True))

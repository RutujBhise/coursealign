"""
Step 8: measure the agents, more than once.

The agents aren't deterministic: the same syllabus can get slightly different
answers on two runs. One good run proves little, so this runs the whole
pipeline several times, with no reply cache, and reports the range of every
number you might quote.

RUN:
  python3 step8_eval.py                            stand-in agents: checks the harness itself
                                                   (every run is identical, by design)
  python3 step8_eval.py --llm anthropic --runs 3   the real numbers, about 50 model calls a run
  python3 step8_eval.py --llm gemini --runs 3      the same with Gemini's free tier (paced, so slower)

Each eval is saved to data/evals/ with the model and settings it used.

Each run keeps its own replies in .cache/eval/run-N.json. Runs never share
replies, so the variation between them is real, but a run cut short by a
rate limit picks up where it stopped. Add --fresh to start a new eval.
"""

import argparse
import json
import shutil
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from coursealign.agent1 import Agent1Config
from coursealign.agent2 import Agent2Config
from coursealign.evaluate import Run, label_runs, load_gold, score_pipeline, score_runs
from coursealign.graph import build_graph, graph_api, initial_state
from coursealign.llm import CachedLLM, FakeLLM, LLMUnavailable, build_llms, stopped_message
from coursealign.loader import load_course_data, pick_courses
from coursealign.oracle import LabelJudge
from coursealign.stage0 import Rules

METRICS = [
    ("outcomes_found", "Labeled outcomes found"),
    ("quotes_rejected", "Agent 1 quotes rejected by the checks"),
    ("agent2_right", "Agent 2 verdicts right"),
    ("held", "Pairs held for a person"),
    ("no_reply", "Pairs with no reply from the model"),
    ("every_pair_right", "Every decided pair right"),
    ("untested_found", "Untested outcomes found"),
    ("orphans_found", "Orphan assignments found"),
    ("false_alarms", "Gap false alarms"),
    ("requests", "Agent requests"),
    ("requests_saved", "Requests the rules saved"),
]


def one_run(courses, rules, agent1_config, agent2_config, gold, llms) -> dict:
    """One full run. llms: (Agent 1's model, Agent 2's model), or None for the stand-ins."""
    if llms is None:
        reader = FakeLLM.from_file(ROOT / "tests" / "fixtures" / "agent1_fake_replies.json",
                                   responder=lambda system, messages: '{"outcomes": []}')
        judge = FakeLLM(responder=LabelJudge({c["course_id"]: c["syllabus"] for c in courses}, gold, rules))
    else:
        reader, judge = llms               # this run's own reply cache: never shared with another run
    graph = build_graph(rules, agent1_config, reader, graph_api()[0], agent2_config=agent2_config, judge_llm=judge)
    finals = [graph.invoke(initial_state(c["syllabus"], c["assignments"])) for c in courses]
    runs = [Run.from_state(f) for f in finals]
    stage0, pipeline = score_runs(runs, gold), score_pipeline(runs, gold)
    requests = sum(len(f["agent1_log"]) + len(f["agent2_log"]) for f in finals)
    per_pair = 2 if agent2_config.second_opinion else 1        # the same judging, without rules to triage
    without_rules = len(finals) + per_pair * sum(
        sum(1 for o in f["outcomes"] if not o.stated_not_assessed) * len(f["assignments"]) for f in finals
    )
    return {
        "outcomes_found": [stage0.labeled_found, stage0.labeled],
        "quotes_rejected": sum(len(entry["problems"]) for f in finals for entry in f["agent1_log"]),
        "agent2_right": [pipeline.accepted_correct, pipeline.accepted],
        "held": pipeline.held,
        "no_reply": sum(1 for f in finals for entry in f["agent2_log"] if entry.get("unavailable")),
        "every_pair_right": [pipeline.decided_correct, pipeline.decided],
        "untested_found": [pipeline.untested_found, pipeline.untested_expected],
        "orphans_found": [pipeline.orphans_found, pipeline.orphans_expected],
        "false_alarms": len(pipeline.untested_false) + len(pipeline.orphans_false),
        "requests": requests,
        "requests_saved": round(1 - requests / without_rules, 3) if without_rules else 0.0,
    }


def show(value) -> str:
    if isinstance(value, list):
        return f"{value[0]}/{value[1]}"
    if isinstance(value, float):
        return f"{value:.0%}"
    return str(value)


def spread(values: list) -> str:
    shown = [show(v) for v in values]
    if len(set(shown)) == 1:
        return "the same every run"
    if isinstance(values[0], list):
        shares = sorted((v[0] / v[1] if v[1] else 1.0, show(v)) for v in values)
        return f"{shares[0][1]} to {shares[-1][1]}"
    return f"{show(min(values))} to {show(max(values))}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["fake", "anthropic", "gemini"], default="fake")
    parser.add_argument("--courses", help="only these courses, like MIT6005,MIT6170 (default: all)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--fresh", action="store_true", help="start a new eval instead of finishing the last one")
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
    real = build_llms(args.llm) if args.llm != "fake" else None
    if real is None:
        model = "stand-in agents (answers from the hand labels)"
    elif real[0] is real[1]:
        model = real[1].name
    else:
        model = f"{real[1].name}, with {real[0].name} for Agent 1"
    if agent2_config.second_opinion:
        model += ", second opinion on"

    print(f"\n{'=' * 64}\n  STEP 8: {args.runs} runs, {model}\n{'=' * 64}")
    cache_dir = ROOT / ".cache" / "eval"
    if args.fresh:
        shutil.rmtree(cache_dir, ignore_errors=True)
    if real is not None:
        print("  Each run keeps its replies in .cache/eval/, so a stopped eval picks up where it")
        print("  left off. Add --fresh to start a new eval instead.")
    results = []
    for n in range(1, args.runs + 1):
        print(f"  run {n} of {args.runs}...", flush=True)
        run_llms = None
        if real is not None:
            reader = CachedLLM(real[0], cache_dir / f"run-{n}.json")
            run_llms = (reader, reader if real[1] is real[0] else CachedLLM(real[1], cache_dir / f"run-{n}.json"))
        results.append(one_run(courses, rules, agent1_config, agent2_config, gold, run_llms))

    header = "".join(f"{'run ' + str(n):>9}" for n in range(1, args.runs + 1))
    print(f"\n  {'':<38}{header}   range")
    for key, label in METRICS:
        values = [r[key] for r in results]
        print(f"  {label:<38}{''.join(f'{show(v):>9}' for v in values)}   {spread(values)}")

    labels_only = score_runs(label_runs([(c["syllabus"], c["assignments"]) for c in courses], gold, rules), gold)
    print(f"\n  Triage on the labeled outcomes alone (no model, the same every time): "
          f"{labels_only.decided}/{labels_only.pairs} pairs decided by rules, "
          f"{labels_only.decided_correct}/{labels_only.decided} right")
    if real is None:
        print("  NOTE: stand-in agents make every run identical. Run with --llm anthropic for real numbers.")
    incomplete = [n for n, r in enumerate(results, 1) if r["no_reply"]]
    if incomplete:
        print(f"  WARNING: run {', '.join(map(str, incomplete))} had pairs the model never answered, so"
              " those runs are incomplete.")
        print("           Don't quote this eval. Run it again when the model is less busy.")

    out = ROOT / "data" / "evals" / f"eval-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": model, "runs": results,
        "label_triage": {"pairs": labels_only.pairs, "decided": labels_only.decided, "right": labels_only.decided_correct},
        "settings": {"auto_match_min_topics": rules.auto_match_min_graded, "common_topic_share": rules.common_share,
                     "agent1": [agent1_config.pass_with_review_at, agent1_config.accept_at],
                     "agent2": [agent2_config.pass_with_review_at, agent2_config.accept_at]},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  Saved to {out.relative_to(ROOT)}\n")


if __name__ == "__main__":
    try:
        main()
    except LLMUnavailable as error:
        raise SystemExit(stopped_message(error, saved=True))

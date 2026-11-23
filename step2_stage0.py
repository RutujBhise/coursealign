"""
Step 2: run Stage 0 (rules only, zero LLM calls) on every course.

RUN: python3 step2_stage0.py

For each course it prints the outcomes the rules extracted, the prose left
for Agent 1, and how each outcome-assignment pair was triaged. Then it
checks the rules' decisions against the hand labels in data/gold/.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from coursealign.evaluate import Run, load_gold, score_runs
from coursealign.loader import load_course_data
from coursealign.stage0 import Decision, Rules, rule_flagged_gaps, run_stage0


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def banner(text: str) -> None:
    print(f"\n{'=' * 64}\n  {text}\n{'=' * 64}")


def main() -> None:
    rules = Rules.from_file(ROOT / "data" / "rules" / "stage0_rules.json")
    gold = load_gold(ROOT / "data" / "gold" / "alignment_labels.json")
    courses = load_course_data(ROOT / "data" / "syllabi", ROOT / "data" / "assignments")

    banner("STEP 2: STAGE 0 (rules only, zero LLM calls)")

    results, runs = [], []
    names: dict[str, str] = {}    # "CS4550-O1" or "CS4550/A1" -> readable text
    for course in courses:
        syllabus, assignments = course["syllabus"], course["assignments"]
        if syllabus is None:
            continue
        result = run_stage0(syllabus, assignments, rules)
        results.append(result)
        runs.append(Run(syllabus, result.outcomes, result.decisions))
        cid = result.course_id
        names.update({o.id: o.text for o in result.outcomes})
        names.update({f"{cid}/{a.id}": a.title for a in assignments})

        print(f"\n[{cid}] {syllabus.title}")

        # Job 1: extraction
        print(f"  Outcomes extracted by rules: {len(result.outcomes)}")
        for o in result.outcomes:
            bloom = o.bloom_level.value if o.bloom_level else "unclassified"
            print(f"    {o.id:<11} [{bloom}] {o.text}")
            print(f"    {'':<11} topics: {', '.join(o.topic_tags) or '(none)'}")
        for region in result.unresolved:
            preview = " ".join(region.text.split())[:80]
            print(f"  Prose for Agent 1: {len(region.text.split())} words under \"{region.header}\"")
            print(f"    \"{preview}...\"")

        if not result.outcomes:
            print("  Triage: waits for Agent 1 (Step 3), since no outcomes exist yet")
            continue

        # Job 2: triage
        print(f"  Triage of {len(result.decisions)} pairs: "
              f"{result.count(Decision.AUTO_MATCH)} auto-match, "
              f"{result.count(Decision.AUTO_REJECT)} auto-reject, "
              f"{result.count(Decision.NEEDS_AGENT)} need the agent")
        for d in result.decisions:
            if d.decision == Decision.AUTO_MATCH:
                print(f"    match  {d.outcome_id:<10} x {d.assignment_id:<7} "
                      f"[{d.alignment_type.value}] {d.reason}")
        for d in result.decisions:
            if d.decision == Decision.NEEDS_AGENT:
                print(f"    agent  {d.outcome_id:<10} x {d.assignment_id:<7} {d.reason}")

        untested, orphans = rule_flagged_gaps(result)
        for oid in untested:
            print(f"  GAP (zero LLM calls): no assignment touches {oid} \"{names[oid]}\"")
        for aid in orphans:
            print(f"  GAP (zero LLM calls): {aid} \"{names[cid + '/' + aid]}\" touches no outcome")

    # What's left for the agents
    rule_outcomes = [o for r in results for o in r.outcomes]
    classified = sum(1 for o in rule_outcomes if o.bloom_level)
    regions = sum(len(r.unresolved) for r in results)
    banner("LEFT FOR THE AGENTS")
    print(f"  Agent 1: {regions} prose regions (one call each)")
    print(f"  Bloom's level set by verb for {classified} of {len(rule_outcomes)} rule-extracted outcomes")

    # Scored against hand labels
    s = score_runs(runs, gold)
    banner("MEASURED AGAINST HAND LABELS (data/gold/alignment_labels.json)")
    print(f"  Labeled outcomes found by rules: {s.labeled_found}/{s.labeled}"
          f" (the rest are written as prose, which is Agent 1's job)")
    print(f"  Rule outcomes that match no label: {len(s.ungrounded)}")
    print(f"  Pairs scored: {s.pairs}")
    print(f"  Decided by rules: {s.decided} ({pct(s.decided, s.pairs)})"
          f"  ->  Agent 2 calls: {s.needs_agent} instead of {s.pairs}")
    print(f"  Rule decisions correct: {s.decided_correct}/{s.decided}"
          f" ({pct(s.decided_correct, s.decided)})")
    print(f"    Wrong auto-matches: {len(s.wrong_matches)}")
    for oid, aid in s.wrong_matches:
        print(f"      {oid} x {aid}: \"{names[oid]}\" / \"{names[oid.split('-')[0] + '/' + aid]}\"")
    print(f"    Missed (auto-rejected, but labeled aligned): {len(s.missed)}")
    for oid, aid in s.missed:
        print(f"      {oid} x {aid}: \"{names[oid]}\" / \"{names[oid.split('-')[0] + '/' + aid]}\"")
    print(f"  True alignments still alive after Stage 0: {s.alignments_kept}/{s.true_alignments}"
          f" ({pct(s.alignments_kept, s.true_alignments)})")
    print("    That's the ceiling for Agent 2: a pair rejected here never gets a second look.")
    print()

if __name__ == "__main__":
    main()

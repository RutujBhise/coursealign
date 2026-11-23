"""
Gap analysis: turns every pair verdict into the report a teacher reads.
Deterministic, no LLM calls.

Each pair gets one final answer, from the highest authority that spoke:
  a person (Step 5)       settles any pair, including ones the rules decided
  Agent 2                 passed or flagged: aligned if direct/partial
                          held: undecided until a person looks
  the rules               auto-match: aligned, with unknown depth
                          auto-reject: not aligned

Gaps, most serious first:
  CRITICAL  untested outcome     no assignment assesses it
  WARNING   weak coverage        only partial matches, nothing assesses all of it
  WARNING   Bloom's mismatch     every assignment that assesses it stays shallower
                                 than the outcome asks for
  INFO      orphan assignment    assesses no stated outcome (can be on purpose)
  INFO      stated not assessed  the syllabus itself says it won't be assessed
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from coursealign.models import (
    AlignmentResult, AlignmentType, Assignment, BloomLevel, ExtractedOutcome,
    GapReport, GapSeverity, GapType, MatchConfidence,
)
from coursealign.stage0 import Decision, TriageDecision

BLOOM_ORDER = list(BloomLevel)
SEVERITY_ORDER = [GapSeverity.CRITICAL, GapSeverity.WARNING, GapSeverity.INFO]
ALIGNED = (AlignmentType.DIRECT, AlignmentType.PARTIAL)


@dataclass
class PairVerdict:
    """The final answer for one pair."""
    source: str                                 # who decided: "rules", "agent2", or "human"
    aligned: Optional[bool]                     # None while it waits for a person
    kind: Optional[AlignmentType] = None        # direct or partial, when aligned
    depth: Optional[BloomLevel] = None          # None when unknown
    flagged: bool = False                       # passed, but a person should look


def final_verdicts(
    decisions: list[TriageDecision],
    judgments: list[AlignmentResult],
    human: list[AlignmentResult] = (),
) -> dict[tuple[str, str], PairVerdict]:
    """One answer per (outcome, assignment) pair. Later sources overwrite
    earlier ones: rules, then Agent 2, then a person."""
    final: dict[tuple[str, str], PairVerdict] = {}
    for d in decisions:
        key = (d.outcome_id, d.assignment_id)
        if d.decision == Decision.AUTO_MATCH:
            final[key] = PairVerdict("rules", True, d.alignment_type)
        elif d.decision == Decision.AUTO_REJECT:
            final[key] = PairVerdict("rules", False)
        else:
            final[key] = PairVerdict("agent2", None)          # waiting on Agent 2
    for j in judgments:
        key = (j.outcome_id, j.assignment_id)
        if j.confidence_tier == MatchConfidence.LOW:
            final[key] = PairVerdict("agent2", None, flagged=True)
        elif j.alignment_type in ALIGNED:
            final[key] = PairVerdict("agent2", True, j.alignment_type, j.assessed_level, j.needs_human_review)
        else:
            final[key] = PairVerdict("agent2", False, flagged=j.needs_human_review)
    for h in human:
        key = (h.outcome_id, h.assignment_id)
        if h.alignment_type in ALIGNED:
            final[key] = PairVerdict("human", True, h.alignment_type, h.assessed_level)
        else:
            final[key] = PairVerdict("human", False)
    return final


@dataclass
class Link:
    """One assignment that assesses an outcome."""
    assignment_id: str
    kind: AlignmentType
    depth: Optional[BloomLevel]


def _pending(n: int) -> str:
    return f" {n} pair{'s' if n != 1 else ''} held for review could change this." if n else ""


def find_gaps(
    outcomes: list[ExtractedOutcome],
    assignments: list[Assignment],
    decisions: list[TriageDecision],
    judgments: list[AlignmentResult],
    human: list[AlignmentResult] = (),
) -> list[GapReport]:
    links: dict[str, list[Link]] = defaultdict(list)
    held_for_outcome: dict[str, int] = defaultdict(int)
    held_for_assignment: dict[str, int] = defaultdict(int)
    for (outcome_id, assignment_id), v in final_verdicts(decisions, judgments, human).items():
        if v.aligned is None:
            held_for_outcome[outcome_id] += 1
            held_for_assignment[assignment_id] += 1
        elif v.aligned:
            links[outcome_id].append(Link(assignment_id, v.kind, v.depth))

    gaps: list[GapReport] = []
    for o in outcomes:
        if o.stated_not_assessed:
            gaps.append(GapReport(
                o.course_id, GapType.STATED_NOT_ASSESSED, GapSeverity.INFO,
                "The syllabus says this won't be formally assessed, so no assignment is expected.",
                outcome_id=o.id,
            ))
            continue

        mine = links[o.id]
        if not mine:
            gaps.append(GapReport(
                o.course_id, GapType.UNTESTED_OUTCOME, GapSeverity.CRITICAL,
                "No assignment assesses this outcome." + _pending(held_for_outcome[o.id]),
                outcome_id=o.id,
            ))
            continue

        if all(link.kind == AlignmentType.PARTIAL for link in mine):
            ids = ", ".join(link.assignment_id for link in mine)
            gaps.append(GapReport(
                o.course_id, GapType.WEAK_COVERAGE, GapSeverity.WARNING,
                f"Covered only in parts: no single assignment assesses the whole outcome ({ids}).",
                outcome_id=o.id,
            ))

        # Depth can only be compared where every link has a known depth. A
        # rules match (or a person's verdict without one) is never treated
        # as shallow.
        if o.bloom_level and all(link.depth for link in mine):
            deepest = max((link.depth for link in mine), key=BLOOM_ORDER.index)
            if BLOOM_ORDER.index(deepest) < BLOOM_ORDER.index(o.bloom_level):
                where = ", ".join(f"{link.assignment_id} ({link.depth.value})" for link in mine)
                gaps.append(GapReport(
                    o.course_id, GapType.BLOOMS_MISMATCH, GapSeverity.WARNING,
                    f"The outcome asks students to {o.bloom_level.value}, but the assignments "
                    f"that assess it only reach {deepest.value}: {where}.",
                    outcome_id=o.id,
                ))

    linked = {link.assignment_id for outcome_links in links.values() for link in outcome_links}
    for a in assignments:
        if a.id not in linked:
            gaps.append(GapReport(
                a.course_id, GapType.ORPHAN_ASSESSMENT, GapSeverity.INFO,
                "Assesses none of the syllabus's stated outcomes. That can be on purpose "
                "(participation, a reflection)." + _pending(held_for_assignment[a.id]),
                assignment_id=a.id,
            ))

    gaps.sort(key=lambda g: SEVERITY_ORDER.index(g.severity))   # stable: keeps outcome order
    return gaps

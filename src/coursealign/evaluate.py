"""
Scoring against hand labels.

WHY THIS EXISTS:
"Rules decided 80% of pairs" means nothing on its own, because rejecting
every pair would "decide" 100%. Savings only count if the decisions are
right, so every number we report gets checked against data/gold/.

HOW OUTCOMES MEET THEIR LABELS: by where they sit in the syllabus. Each
label quotes the syllabus words that state its outcome, and each pipeline
outcome carries its own quote (a rule outcome's quote is its own text). Two
quotes match when they overlap by at least half of the shorter one. So an
agent that rewords an outcome still meets its label, and an agent that
invents one meets nothing.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from coursealign.gaps import final_verdicts
from coursealign.models import (
    AlignmentResult, Assignment, ExtractedOutcome, ExtractionSource, GapReport, GapType, RawSyllabus,
)
from coursealign.spans import best_match
from coursealign.stage0 import Decision, Rules, TriageDecision, tag_text, triage


@dataclass
class Label:
    quote: str
    aligned: set[str]
    stated_not_assessed: bool = False


def load_gold(path: str | Path) -> dict[str, list[Label]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        course: [
            Label(item["quote"], set(item["aligned"]), item.get("stated_not_assessed", False))
            for item in items
        ]
        for course, items in raw["courses"].items()
    }


@dataclass
class Run:
    """What one pipeline run produced for one course."""
    syllabus: RawSyllabus
    outcomes: list[ExtractedOutcome]                  # passed outcomes
    decisions: list[TriageDecision]
    held: list[ExtractedOutcome] = field(default_factory=list)
    judgments: list[AlignmentResult] = field(default_factory=list)
    gaps: list[GapReport] = field(default_factory=list)
    assignments: list[Assignment] = field(default_factory=list)
    human: list[AlignmentResult] = field(default_factory=list)

    @classmethod
    def from_state(cls, state: dict) -> "Run":
        """From a graph's final state."""
        return cls(
            syllabus=state["syllabus"], outcomes=state["outcomes"], decisions=state["decisions"],
            held=state["held"], judgments=state.get("judgments", []), gaps=state.get("gaps", []),
            assignments=state["assignments"], human=state.get("human_verdicts", []),
        )


def match_labels(outcomes: list[ExtractedOutcome], labels: list[Label], source: str) -> dict[str, int]:
    """outcome id -> index of the label its quote points at (biggest overlap wins)."""
    candidates = [(str(i), label.quote) for i, label in enumerate(labels)]
    matches: dict[str, int] = {}
    for o in outcomes:
        hit = best_match(o.evidence or o.text, candidates, source)
        if hit is not None:
            matches[o.id] = int(hit)
    return matches


# ── Stage 0 on its own: extraction and triage ────────────────────────────────

@dataclass
class Score:
    # extraction
    labeled: int = 0                    # labeled outcomes
    labeled_found: int = 0              # labels matched by at least one passed outcome
    extracted: int = 0                  # passed outcomes
    grounded: int = 0                   # passed outcomes that match a label
    ungrounded: list[str] = field(default_factory=list)
    flags_checked: int = 0              # Agent 1 outcomes with a label to compare to
    flags_correct: int = 0              # ...whose "not assessed" flag matches it
    # triage
    pairs: int = 0
    auto_match: int = 0
    auto_reject: int = 0
    needs_agent: int = 0
    wrong_matches: list[tuple[str, str]] = field(default_factory=list)  # auto_match, label says no
    missed: list[tuple[str, str]] = field(default_factory=list)         # auto_reject, label says yes
    true_alignments: int = 0
    alignments_kept: int = 0            # true alignments not thrown away by auto_reject
    unlabeled_courses: list[str] = field(default_factory=list)

    @property
    def decided(self) -> int:
        return self.auto_match + self.auto_reject

    @property
    def decided_correct(self) -> int:
        return self.decided - len(self.wrong_matches) - len(self.missed)


def score_runs(runs: list[Run], gold: dict[str, list[Label]]) -> Score:
    s = Score()
    for run in runs:
        course = run.syllabus.course_id
        labels = gold.get(course)
        if labels is None:
            s.unlabeled_courses.append(course)
            continue
        matches = match_labels(run.outcomes, labels, run.syllabus.raw_text)

        s.labeled += len(labels)
        s.labeled_found += len(set(matches.values()))
        s.extracted += len(run.outcomes)
        s.grounded += len(matches)
        s.ungrounded += [o.id for o in run.outcomes if o.id not in matches]
        for o in run.outcomes:
            if o.source == ExtractionSource.AGENT and o.id in matches:
                s.flags_checked += 1
                s.flags_correct += int(o.stated_not_assessed == labels[matches[o.id]].stated_not_assessed)

        for d in run.decisions:
            if d.outcome_id not in matches:
                continue                    # can't score an outcome with no label
            truly_aligned = d.assignment_id in labels[matches[d.outcome_id]].aligned
            s.pairs += 1
            s.true_alignments += int(truly_aligned)
            if d.decision == Decision.AUTO_MATCH:
                s.auto_match += 1
                if not truly_aligned:
                    s.wrong_matches.append((d.outcome_id, d.assignment_id))
            elif d.decision == Decision.AUTO_REJECT:
                s.auto_reject += 1
                if truly_aligned:
                    s.missed.append((d.outcome_id, d.assignment_id))
            else:
                s.needs_agent += 1
            if truly_aligned and d.decision != Decision.AUTO_REJECT:
                s.alignments_kept += 1
    return s


def label_runs(
    courses: list[tuple[RawSyllabus, list[Assignment]]], gold: dict[str, list[Label]], rules: Rules
) -> list[Run]:
    """Triage the labeled outcomes themselves: no extraction step, no model.

    This scores the triage rules on their own. Any mistake here belongs to
    the rules, not to an agent, and the numbers are the same on every run.
    """
    runs = []
    for syllabus, assignments in courses:
        labels = gold.get(syllabus.course_id)
        if not labels:
            continue
        outcomes = [
            ExtractedOutcome(
                id=f"{syllabus.course_id}-L{n}", course_id=syllabus.course_id,
                text=label.quote, evidence=label.quote,
                topic_tags=sorted(tag_text(label.quote, rules)),
                stated_not_assessed=label.stated_not_assessed,
            )
            for n, label in enumerate(labels, 1)
        ]
        assessable = [o for o in outcomes if not o.stated_not_assessed]
        runs.append(Run(syllabus, outcomes, triage(assessable, assignments, rules)))
    return runs


# ── The whole pipeline: final verdicts and the gap report ────────────────────

@dataclass
class PipelineScore:
    # Agent 2, on the pairs it decided
    judged: int = 0
    accepted: int = 0                        # verdicts that passed, flagged ones included
    accepted_correct: int = 0
    wrong: list[tuple[str, str, str]] = field(default_factory=list)   # (outcome, assignment, verdict)
    flagged: int = 0
    held: int = 0                            # pairs still waiting for a person
    # every pair, whoever decided it
    pairs: int = 0
    decided: int = 0
    decided_correct: int = 0
    by_human: int = 0
    # gaps the labels imply vs gaps the pipeline reported
    untested_expected: int = 0
    untested_found: int = 0
    untested_false: list[str] = field(default_factory=list)
    orphans_expected: int = 0
    orphans_found: int = 0
    orphans_false: list[str] = field(default_factory=list)


def score_pipeline(runs: list[Run], gold: dict[str, list[Label]]) -> PipelineScore:
    s = PipelineScore()
    for run in runs:
        labels = gold.get(run.syllabus.course_id)
        if not labels:
            continue
        matches = match_labels(run.outcomes, labels, run.syllabus.raw_text)

        # The same final answers the gap report uses
        for (outcome_id, assignment_id), v in final_verdicts(run.decisions, run.judgments, run.human).items():
            if outcome_id not in matches:
                continue
            truth = assignment_id in labels[matches[outcome_id]].aligned
            s.pairs += 1
            if v.source == "agent2":
                s.judged += 1
            if v.aligned is None:
                s.held += 1
                continue
            right = v.aligned == truth
            s.decided += 1
            s.decided_correct += int(right)
            if v.source == "agent2":
                s.accepted += 1
                s.flagged += int(v.flagged)
                if right:
                    s.accepted_correct += 1
                else:
                    s.wrong.append((outcome_id, assignment_id, "aligned" if v.aligned else "not aligned"))
            elif v.source == "human":
                s.by_human += 1

        # Untested: a label with nothing aligned that the syllabus doesn't excuse.
        expected = {i for i, label in enumerate(labels) if not label.aligned and not label.stated_not_assessed}
        flagged = [g.outcome_id for g in run.gaps if g.gap_type == GapType.UNTESTED_OUTCOME]
        s.untested_expected += len(expected)
        s.untested_found += len({matches[o] for o in flagged if o in matches} & expected)
        s.untested_false += [o for o in flagged if o not in matches or matches[o] not in expected]

        # Orphan: an assignment no label lists as aligned.
        aligned_somewhere = set().union(*(label.aligned for label in labels))
        expected_orphans = {a.id for a in run.assignments} - aligned_somewhere
        found_orphans = {g.assignment_id for g in run.gaps if g.gap_type == GapType.ORPHAN_ASSESSMENT}
        s.orphans_expected += len(expected_orphans)
        s.orphans_found += len(found_orphans & expected_orphans)
        s.orphans_false += [f"{run.syllabus.course_id}/{a}" for a in sorted(found_orphans - expected_orphans)]
    return s


# ── Scored per stated outcome: the level the labels were written at ──────────

@dataclass
class LabelScore:
    """One label is one outcome the syllabus states. When Agent 1 splits a stated
    outcome into parts (one bundled objective into "design", "implement", "test"),
    scoring each part against the whole outcome's label counts right answers as
    wrong: a part can truly go unassessed by an assignment that assesses another
    part. So here the pipeline's answer for (stated outcome, assignment) is
    "assesses it" if it says so for any part, which is the rule the labels use."""
    pairs: int = 0
    decided: int = 0
    decided_correct: int = 0
    waiting: int = 0
    wrong: list[str] = field(default_factory=list)
    split: int = 0                      # stated outcomes Agent 1 split into more than one part
    untested_expected: int = 0
    untested_found: int = 0
    untested_false: list[str] = field(default_factory=list)
    finer_alarms: list[str] = field(default_factory=list)   # a part flagged untested while another part is assessed


def score_by_label(runs: list[Run], gold: dict[str, list[Label]]) -> LabelScore:
    s = LabelScore()
    for run in runs:
        course = run.syllabus.course_id
        labels = gold.get(course)
        if not labels:
            continue
        matches = match_labels(run.outcomes, labels, run.syllabus.raw_text)
        parts: dict[int, list[str]] = {}
        for outcome_id, index in matches.items():
            parts.setdefault(index, []).append(outcome_id)
        verdicts = final_verdicts(run.decisions, run.judgments, run.human)

        for index, outcome_ids in sorted(parts.items()):
            s.split += int(len(outcome_ids) > 1)
            label = labels[index]
            if label.stated_not_assessed:
                continue
            for a in run.assignments:
                said = [verdicts[(o, a.id)].aligned for o in outcome_ids if (o, a.id) in verdicts]
                if not said:
                    continue                  # every part was set aside as "not assessed"
                answer = True if True in said else (None if None in said else False)
                s.pairs += 1
                if answer is None:
                    s.waiting += 1
                    continue
                s.decided += 1
                if answer == (a.id in label.aligned):
                    s.decided_correct += 1
                else:
                    s.wrong.append(f"{course} outcome {index + 1} x {a.id}, judged {'aligned' if answer else 'not aligned'}")

        # Untested: a stated outcome that no assignment assesses in any of its parts.
        flagged = {g.outcome_id for g in run.gaps if g.gap_type == GapType.UNTESTED_OUTCOME}
        expected = {i for i, label in enumerate(labels) if not label.aligned and not label.stated_not_assessed}
        predicted = {i for i, ids in parts.items() if all(o in flagged for o in ids)}
        s.untested_expected += len(expected)
        s.untested_found += len(predicted & expected)
        s.untested_false += [f"{course} outcome {i + 1}" for i in sorted(predicted - expected)]
        s.untested_false += sorted(o for o in flagged if o not in matches)       # about an outcome no label states
        s.finer_alarms += sorted(o for o in flagged if o in matches and matches[o] not in predicted)
    return s

"""
Stage 0: deterministic pre-processing. There are no LLM calls in this file.

It does two jobs:

  1. EXTRACT  Pull learning outcomes out of a syllabus when they're written
              as a clear list. Prose gets handed to Agent 1 instead of guessed at.

  2. TRIAGE   For every (outcome, assignment) pair, decide one of:
                auto_match   rules are confident the assignment assesses it
                auto_reject  rules are confident it doesn't
                needs_agent  ambiguous, so Agent 2 decides

WHY THIS EXISTS:
Agents cost money, take time, and can give different answers to the same
question. Anything a simple, testable rule can decide shouldn't reach an
agent. Every rule lives in data/rules/stage0_rules.json, so behavior changes
without code changes, and the feedback loop (Step 5) can add rules there.
"""

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from coursealign.models import (
    AlignmentType,
    Assignment,
    BloomLevel,
    ExtractedOutcome,
    ExtractionSource,
    RawSyllabus,
)

# Bloom's levels, lowest to highest. "Design and implement" has two verbs,
# and we keep the higher one (design = create beats implement = apply).
BLOOM_ORDER = list(BloomLevel)

# "1. Build..."  "2) Design..."  "- Evaluate..."  "* Use..."  "• Apply..."
LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*\u2022])\s+(.+)$")

# Removed before reading the verb: "Students will be able to build" -> "build"
LEAD_IN_PREFIXES = (
    "students will be able to",
    "you will be able to",
    "students will",
    "you will",
    "be able to",
)


# ── Output types ─────────────────────────────────────────────────────────────

class Decision(str, Enum):
    AUTO_MATCH = "auto_match"
    AUTO_REJECT = "auto_reject"
    NEEDS_AGENT = "needs_agent"


@dataclass
class UnresolvedRegion:
    """Prose inside an outcomes section. Rules can't split prose into outcomes
    safely, so the whole block goes to Agent 1 as one call."""
    course_id: str
    header: str
    text: str


@dataclass
class TriageDecision:
    """Stage 0's verdict on one (outcome, assignment) pair, with its reason.

    Every pair gets one, auto-rejects included, so the dashboard can later
    show why any pair was or wasn't sent to the agent.
    """
    outcome_id: str
    assignment_id: str
    decision: Decision
    shared_graded: list[str]    # topics shared with the title + rubric
    shared_any: list[str]       # topics shared with title + rubric + description
    reason: str
    alignment_type: Optional[AlignmentType] = None   # set only on auto-matches


@dataclass
class Stage0Result:
    course_id: str
    outcomes: list[ExtractedOutcome] = field(default_factory=list)
    unresolved: list[UnresolvedRegion] = field(default_factory=list)
    decisions: list[TriageDecision] = field(default_factory=list)

    def count(self, decision: Decision) -> int:
        return sum(1 for d in self.decisions if d.decision == decision)


# ── Rules ────────────────────────────────────────────────────────────────────

@dataclass
class Rules:
    """stage0_rules.json, loaded and compiled once."""
    outcome_headers: list[str]
    stop_headers: tuple[str, ...]
    verb_to_bloom: dict[str, BloomLevel]
    tag_patterns: dict[str, list[re.Pattern]]
    auto_match_min_graded: int
    common_share: float
    common_min_count: int

    @classmethod
    def from_file(cls, path: str | Path, learned: dict[str, list[str]] | None = None) -> "Rules":
        """The rules file, plus any words people taught the rules in reviews."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        topics = {tag: list(terms) for tag, terms in raw["topic_tags"].items()}
        for tag, terms in (learned or {}).items():
            if tag not in topics:
                raise ValueError(f"can't teach a word for '{tag}': it isn't a topic in {Path(path).name}")
            topics[tag] += terms
        return cls(
            outcome_headers=[h.lower() for h in raw["outcome_section_headers"]],
            stop_headers=tuple(h.lower() for h in raw["stop_section_headers"]),
            verb_to_bloom={
                verb.lower(): BloomLevel(level)
                for level, verbs in raw["bloom_verbs"].items()
                for verb in verbs
            },
            tag_patterns={tag: [_term_pattern(term) for term in terms] for tag, terms in topics.items()},
            auto_match_min_graded=raw["triage"]["auto_match_min_shared_graded_topics"],
            common_share=raw["triage"].get("common_topic_share", 1.0),
            common_min_count=raw["triage"].get("common_topic_min_assignments", 3),
        )


def _term_pattern(term: str) -> re.Pattern:
    """Match a term only as a whole term, with an optional plural "s".

    WHY NOT `term in text`: "test" in "latest" is True. Boundaries here are
    any non-letter/digit, so terms containing "/", "." or "-" (ci/cd,
    node.js, end-to-end) still match cleanly. A hyphen right before a term
    also blocks it, so "non-relational" doesn't count as relational.
    """
    return re.compile(r"(?<![a-z0-9-])" + re.escape(term.lower()) + r"s?(?![a-z0-9])")


def _normalize(text: str) -> str:
    return text.lower().replace("\u2019", "'")   # curly apostrophe -> straight


# ── Tagging and Bloom's ──────────────────────────────────────────────────────

def tag_text(text: str, rules: Rules) -> set[str]:
    """Map free text to topic tags: "JWT" and "OAuth" both become authentication."""
    s = _normalize(text)
    return {
        tag for tag, patterns in rules.tag_patterns.items()
        if any(p.search(s) for p in patterns)
    }


def classify_bloom(text: str, rules: Rules) -> Optional[BloomLevel]:
    """Bloom's level from the outcome's opening verb(s).

    "Design and implement RESTful APIs" -> design (create), implement (apply) -> create
    "Practical experience decomposing..." -> no known verb -> None

    Unknown stays None on purpose: a wrong level would create fake
    "Bloom's mismatch" gaps later, while a blank one just skips that check.
    """
    s = _normalize(text).strip()
    for prefix in LEAD_IN_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    words = re.findall(r"[a-z]+", s)
    if not words:
        return None
    verbs = [words[0]]
    if len(words) > 2 and words[1] == "and":
        verbs.append(words[2])
    levels = [rules.verb_to_bloom[v] for v in verbs if v in rules.verb_to_bloom]
    return max(levels, key=BLOOM_ORDER.index) if levels else None


# ── Job 1: extraction ────────────────────────────────────────────────────────

def _is_header_like(line: str) -> bool:
    """Short, not a sentence, not a list item: 'GRADING POLICY', 'Course Goals:'."""
    s = line.strip()
    return 0 < len(s) <= 60 and not s.endswith((".", ",")) and not LIST_ITEM.match(line)


def _starts_outcomes_section(line: str, rules: Rules) -> bool:
    return _is_header_like(line) and any(h in _normalize(line) for h in rules.outcome_headers)


def _ends_outcomes_section(line: str, rules: Rules) -> bool:
    return _is_header_like(line) and _normalize(line).strip().startswith(rules.stop_headers)


def _is_category_label(line: str) -> bool:
    """ALL-CAPS label inside a section, like 'DESIGN & ARCHITECTURE'."""
    letters = [c for c in line if c.isalpha()]
    return _is_header_like(line) and len(letters) >= 2 and all(c.isupper() for c in letters)


def _make_outcome(course_id: str, number: int, text: str, rules: Rules) -> ExtractedOutcome:
    text = " ".join(text.split())
    return ExtractedOutcome(
        id=f"{course_id}-O{number}",
        course_id=course_id,
        text=text,
        bloom_level=classify_bloom(text, rules),
        topic_tags=sorted(tag_text(text, rules)),
        extraction_confidence=1.0,          # it was a literal list item
        source=ExtractionSource.REGEX,
        evidence=text,                      # a list item is its own evidence
    )


def extract_outcomes(
    syllabus: RawSyllabus, rules: Rules
) -> tuple[list[ExtractedOutcome], list[UnresolvedRegion]]:
    """Walk the syllabus line by line.

    Outside an outcomes section, ignore everything (policies, schedule...).
    Inside one:
      list item          -> an outcome (rules are confident)
      ALL-CAPS label     -> skip, it's a category
      line ending in ":" -> skip, it's a lead-in ("students will be able to:")
      anything else      -> prose, saved for Agent 1
    A stop header (GRADING, Assessment, Prerequisites...) ends the section.
    """
    outcomes: list[ExtractedOutcome] = []
    unresolved: list[UnresolvedRegion] = []

    header: Optional[str] = None    # set while inside an outcomes section
    items: list[str] = []           # list items in the current section
    paragraphs: list[str] = []      # finished prose paragraphs in the current section
    current: list[str] = []         # lines of the prose paragraph being read
    after_item = False              # was the previous line a list item?

    def end_paragraph() -> None:
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    def end_section() -> None:
        nonlocal header
        end_paragraph()
        for text in items:
            outcomes.append(_make_outcome(syllabus.course_id, len(outcomes) + 1, text, rules))
        if header and paragraphs:
            unresolved.append(UnresolvedRegion(syllabus.course_id, header, "\n\n".join(paragraphs)))
        items.clear()
        paragraphs.clear()
        header = None

    for line in syllabus.raw_text.splitlines():
        stripped = line.strip()

        if _starts_outcomes_section(line, rules):
            end_section()
            header = stripped
            after_item = False
            continue
        if header is None:
            continue
        if _ends_outcomes_section(line, rules):
            end_section()
            continue

        item = LIST_ITEM.match(line)
        if item:
            end_paragraph()
            items.append(item.group(1).strip())
            after_item = True
        elif not stripped:
            end_paragraph()
            after_item = False
        elif after_item and line[:1].isspace():
            items[-1] += " " + stripped     # a wrapped list item continues here
        elif _is_category_label(line) or stripped.endswith(":"):
            end_paragraph()
            after_item = False
        else:
            current.append(stripped)
            after_item = False

    end_section()
    return outcomes, unresolved


# ── Job 2: triage ────────────────────────────────────────────────────────────

def assignment_topics(assignment: Assignment, rules: Rules) -> tuple[set[str], set[str]]:
    """Split an assignment's topics by how much we trust them.

    graded      title + rubric keywords: what the assignment IS and what gets graded
    described   the description, which often mentions things in passing
                ("add auth to YOUR API"), so it can send a pair to the agent
                but can never auto-match one
    """
    graded = tag_text(" | ".join([assignment.title, *assignment.rubric_keywords]), rules)
    described = tag_text(assignment.description, rules)
    return graded, described


def common_topics(assignments: list[Assignment], rules: Rules) -> set[str]:
    """Topics that show up in most of a course's assignments.

    In a database course, "database" and "SQL" appear almost everywhere, so
    sharing them says little about whether an assignment assesses a given
    outcome. Common topics can still send a pair to the agent; they just
    don't count toward an auto-match.
    """
    if not assignments:
        return set()
    counts: Counter = Counter()
    for a in assignments:
        graded, described = assignment_topics(a, rules)
        counts.update(graded | described)
    n = len(assignments)
    return {
        topic for topic, count in counts.items()
        if count >= rules.common_min_count and count / n > rules.common_share
    }


def triage_pair(
    outcome: ExtractedOutcome,
    assignment: Assignment,
    rules: Rules,
    common: frozenset[str] = frozenset(),
) -> TriageDecision:
    """The whole triage policy, checked in this order:

      1. Rules can't read one side (no known topics)  -> needs_agent
         Rules only get to say "no" when they understand both sides.
      2. 2+ topics shared with what's GRADED          -> auto_match
         One shared topic can be a coincidence; two is evidence. Topics
         common across the whole course don't count toward the two.
      3. Nothing shared anywhere                      -> auto_reject
      4. Anything in between                          -> needs_agent
    """
    o_topics = set(outcome.topic_tags)
    graded, described = assignment_topics(assignment, rules)
    shared_graded = sorted(o_topics & graded)
    shared_any = sorted(o_topics & (graded | described))
    counted = [t for t in shared_graded if t not in common]

    def verdict(decision: Decision, reason: str, kind: Optional[AlignmentType] = None) -> TriageDecision:
        return TriageDecision(outcome.id, assignment.id, decision, shared_graded, shared_any, reason, kind)

    if not o_topics or not (graded | described):
        return verdict(Decision.NEEDS_AGENT, "rules can't read one side (no known topics)")
    if len(counted) >= rules.auto_match_min_graded:
        kind = AlignmentType.DIRECT if o_topics <= graded else AlignmentType.PARTIAL
        return verdict(Decision.AUTO_MATCH, "graded topics shared: " + ", ".join(counted), kind)
    if not shared_any:
        return verdict(Decision.AUTO_REJECT, "no shared topics in title, rubric, or description")
    reason = "weak overlap: " + ", ".join(shared_any)
    background = sorted(set(shared_any) & common)
    if background:
        reason += f" ({', '.join(background)}: common in this course)"
    return verdict(Decision.NEEDS_AGENT, reason)


def triage(
    outcomes: list[ExtractedOutcome], assignments: list[Assignment], rules: Rules
) -> list[TriageDecision]:
    """Every outcome x every assignment. Kept separate from extraction because
    Agent 1's outcomes go through this same gate."""
    common = frozenset(common_topics(assignments, rules))
    return [triage_pair(o, a, rules, common) for o in outcomes for a in assignments]


def run_stage0(syllabus: RawSyllabus, assignments: list[Assignment], rules: Rules) -> Stage0Result:
    outcomes, unresolved = extract_outcomes(syllabus, rules)
    return Stage0Result(
        course_id=syllabus.course_id,
        outcomes=outcomes,
        unresolved=unresolved,
        decisions=triage(outcomes, assignments, rules),
    )


def rule_flagged_gaps(result: Stage0Result) -> tuple[list[str], list[str]]:
    """Outcomes no assignment touches, and assignments that touch no outcome,
    according to the rules alone. These cost zero LLM calls, but they're only
    as trustworthy as the auto-rejects behind them."""
    by_outcome: dict[str, list[Decision]] = defaultdict(list)
    by_assignment: dict[str, list[Decision]] = defaultdict(list)
    for d in result.decisions:
        by_outcome[d.outcome_id].append(d.decision)
        by_assignment[d.assignment_id].append(d.decision)
    untested = [o for o, ds in by_outcome.items() if all(x == Decision.AUTO_REJECT for x in ds)]
    orphans = [a for a, ds in by_assignment.items() if all(x == Decision.AUTO_REJECT for x in ds)]
    return untested, orphans

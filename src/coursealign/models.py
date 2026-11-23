"""
CourseAlign data models.

WHY THESE EXIST:
Every component in the pipeline — Stage 0 pre-processing, Agent 1 (extractor),
Agent 2 (mapper), gap analysis, feedback loop, and the API — speaks in these types.
Defining them first means:
  1. We know exactly what each stage produces and consumes
  2. Tests can validate structure before we write any logic
  3. The LangGraph state is just a container of these models
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────────────

class BloomLevel(str, Enum):
    """Bloom's taxonomy levels, ordered from lowest to highest cognitive demand.

    WHY THIS MATTERS:
    A multiple-choice quiz (tests 'remember') can't meaningfully assess an outcome
    that says 'design a system' (requires 'create'). Agent 2 uses this to flag
    Bloom's mismatches — one of the most common curriculum gaps.
    """
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class MatchConfidence(str, Enum):
    """Confidence tiers with hard thresholds.

    WHY THREE TIERS (not a raw float):
    Agents produce noisy scores. A 78% and an 81% are functionally the same,
    but one crosses a threshold. Bucketing into tiers makes the system
    predictable: auto-match, human-review, or reject. No gray zones.
    """
    HIGH = "high"        # ≥ 80% → auto-match, no human needed
    MEDIUM = "medium"    # 60–79% → flagged for human review
    LOW = "low"          # < 60% → no match


class GapSeverity(str, Enum):
    """How bad is this gap?

    WHY SEVERITY LEVELS:
    Not all gaps are equal. An outcome with zero assessments is a critical
    curriculum failure. An assessment tied to no outcome is worth noting
    but might be intentional (participation, reflection). Severity lets
    the dashboard sort by what matters.
    """
    CRITICAL = "critical"   # outcome has no assessment at all
    WARNING = "warning"     # Bloom's mismatch or weak coverage
    INFO = "info"           # orphan assessment, may be intentional


class GapType(str, Enum):
    UNTESTED_OUTCOME = "untested_outcome"       # outcome with 0 aligned assessments
    ORPHAN_ASSESSMENT = "orphan_assessment"      # assessment tied to 0 outcomes
    WEAK_COVERAGE = "weak_coverage"              # only partial/tangential alignment
    BLOOMS_MISMATCH = "blooms_mismatch"          # quiz tests 'remember', outcome says 'create'
    STATED_NOT_ASSESSED = "stated_not_assessed"  # the syllabus says it won't be assessed


class AlignmentType(str, Enum):
    DIRECT = "direct"           # assignment clearly tests this outcome
    PARTIAL = "partial"         # tests some aspects but not all
    TANGENTIAL = "tangential"   # loosely related but not a real assessment


class ExtractionSource(str, Enum):
    """How was this outcome extracted?

    WHY TRACK THIS:
    This is the 'reduces API calls' metric. If Stage 0 regex extracted it,
    no LLM was needed. If Agent 1 extracted it, that's an API call. The dashboard
    shows this breakdown.
    """
    REGEX = "regex"             # Stage 0 deterministic extraction
    AGENT = "agent"             # Agent 1 LLM extraction


# ── Core Data Models ─────────────────────────────────────────────────────────

@dataclass
class RawSyllabus:
    """Raw syllabus as ingested from file.

    WHY KEEP THE RAW TEXT:
    Agent 1 needs the original text for sections that regex couldn't parse.
    We also need it for the audit trail — 'here's the original, here's what
    we extracted.'
    """
    course_id: str
    title: str
    raw_text: str
    file_path: str


@dataclass
class Assignment:
    """One assignment/assessment from the course.

    WHY rubric_keywords IS A LIST:
    These are the tags we use in Stage 0 for deterministic keyword matching.
    If an assignment's keywords overlap with an outcome's keywords, we can
    skip the LLM entirely. This is the core of the API-call reduction.
    """
    id: str
    course_id: str
    title: str
    type: str                       # homework, assignment, project, exam
    description: str
    rubric_keywords: list[str]
    points: int
    week: int


@dataclass
class ExtractedOutcome:
    """A learning outcome extracted from a syllabus.

    This is the OUTPUT of either Stage 0 (regex) or Agent 1 (LLM).
    Every downstream component works with this, never with raw syllabus text.
    """
    id: str
    course_id: str
    text: str                                       # the outcome statement
    bloom_level: Optional[BloomLevel] = None         # from the opening verb (Stage 0's verb list)
    topic_tags: list[str] = field(default_factory=list)  # for keyword matching
    extraction_confidence: float = 1.0               # 1.0 for regex, 0.0–1.0 for agent
    source: ExtractionSource = ExtractionSource.REGEX
    evidence: str = ""                               # the syllabus words that state it (checked by code)
    stated_not_assessed: bool = False                # the syllabus itself says it won't be assessed
    needs_review: bool = False                       # a human should confirm it (Step 5)


@dataclass
class AlignmentResult:
    """Agent 2's judgment: does this assignment assess this outcome?

    WHY ONE PAIR AT A TIME:
    Context isolation. Agent 2 sees exactly one outcome and one assignment.
    It can't be biased by knowing that other outcomes already have matches
    or that another agent was uncertain.
    """
    outcome_id: str
    assignment_id: str
    score: float                                # 0.0–1.0 raw confidence
    confidence_tier: MatchConfidence             # bucketed from score
    alignment_type: AlignmentType
    reasoning: str                               # 1-sentence explanation
    needs_human_review: bool = False
    evidence: str = ""                               # the assignment words that require the skill
    assessed_level: Optional[BloomLevel] = None      # how deep the assignment goes for this outcome
    decided_by: str = "agent2"


@dataclass
class GapReport:
    """One detected gap in the curriculum alignment."""
    course_id: str
    gap_type: GapType
    severity: GapSeverity
    description: str
    outcome_id: Optional[str] = None           # which outcome is affected
    assignment_id: Optional[str] = None        # which assignment is affected


@dataclass
class ReviewDecision:
    """A person's decision, saved so nobody (model or person) is asked twice.

    Tied to the outcome's syllabus quote, not its id: an agent can number
    outcomes differently on the next run, but the syllabus words stay put.
    When two decisions cover the same thing, the later one wins.
    """
    kind: str                                    # "outcome" or "pair"
    course_id: str
    quote: str                                   # the outcome's syllabus words
    decision: str                                # outcome: keep | drop; pair: direct | partial | not_aligned
    assignment_id: str = ""                      # pairs only
    assessed_level: Optional[BloomLevel] = None  # pairs only, when known
    note: str = ""
    teach: list[dict] = field(default_factory=list)   # e.g. [{"topic": "communication", "term": "justification"}]
    by: str = "reviewer"
    at: str = ""                                 # date of the decision

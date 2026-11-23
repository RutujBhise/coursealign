"""
Agent 1: reads syllabus prose the rules couldn't parse and lists the
learning outcomes it states.

WHAT IT SEES: one prose section and the course title. Never the
assignments, and never another agent's output. If it saw the assignments,
it could be nudged toward "finding" outcomes that happen to match them.

WHAT IT DOES: the one thing rules can't, which is reading prose. Everything
around that stays deterministic:
  - every outcome must quote its evidence, and code checks the quote exists
  - topic tags come from the quote, never from the agent's own wording
  - Bloom's level comes from the outcome's verb, using Stage 0's verb list
  - confidence decides what happens next: pass, pass with review, or hold
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from coursealign.llm import parse_json_reply
from coursealign.models import ExtractedOutcome, ExtractionSource, RawSyllabus
from coursealign.spans import find_span
from coursealign.stage0 import Rules, UnresolvedRegion, classify_bloom, tag_text

REPLY_SHAPE = (
    '{"outcomes": [{"outcome": "...", "evidence": "...", '
    '"confidence": 0.0, "stated_not_assessed": false}]}'
)


@dataclass
class Agent1Config:
    """data/rules/agent1.json, loaded."""
    rules: list[str]
    accept_at: float
    pass_with_review_at: float
    min_quote_words: int
    max_quote_words: int
    max_attempts: int

    @classmethod
    def from_file(cls, path: str | Path) -> "Agent1Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            rules=raw["rules"],
            accept_at=raw["confidence"]["accept_at"],
            pass_with_review_at=raw["confidence"]["pass_with_review_at"],
            min_quote_words=raw["quote_words"]["min"],
            max_quote_words=raw["quote_words"]["max"],
            max_attempts=raw["max_attempts"],
        )


# ── Prompts ──────────────────────────────────────────────────────────────────

def system_prompt(config: Agent1Config, stage0_rules: Rules) -> str:
    rules = "\n".join(f"{n}. {rule}" for n, rule in enumerate(config.rules, 1))
    verbs = ", ".join(sorted(stage0_rules.verb_to_bloom))
    return (
        "You read one section of a course syllabus and list every learning outcome it states.\n\n"
        f"Rules:\n{rules}\n\n"
        f"Each evidence quote must be {config.min_quote_words} to {config.max_quote_words} words long.\n\n"
        f"Allowed verbs: {verbs}\n\n"
        f"Reply with JSON only, in exactly this shape:\n{REPLY_SHAPE}"
    )


def user_message(syllabus: RawSyllabus, region: UnresolvedRegion) -> str:
    return f"Course: {syllabus.course_id} ({syllabus.title})\nSection: {region.header}\n\n{region.text}"


def retry_message(problems: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        f"Your reply failed these checks:\n{listed}\n\n"
        "Reply with the full corrected JSON. Fix each problem, or drop an "
        "outcome if the text doesn't actually state it."
    )


# ── Checking a reply ─────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """One outcome from the agent that passed every check."""
    outcome: str
    evidence: str
    confidence: float
    stated_not_assessed: bool


@dataclass
class CheckResult:
    candidates: list[Candidate] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)   # worded so the agent can fix them
    returned: int = 0                                   # outcomes the agent returned


def _check_item(item: object, source_text: str, config: Agent1Config) -> str | None:
    """Return what's wrong with one outcome, or None if it passes."""
    if not isinstance(item, dict):
        return "not a JSON object"
    outcome, evidence = item.get("outcome"), item.get("evidence")
    confidence, flag = item.get("confidence"), item.get("stated_not_assessed", False)
    if not isinstance(outcome, str) or not outcome.strip():
        return '"outcome" must be a non-empty string'
    if not isinstance(evidence, str) or not evidence.strip():
        return '"evidence" must be a non-empty string'
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        return '"confidence" must be a number from 0 to 1'
    if not isinstance(flag, bool):
        return '"stated_not_assessed" must be true or false'
    words = len(evidence.split())
    if not config.min_quote_words <= words <= config.max_quote_words:
        return f"evidence must be {config.min_quote_words}-{config.max_quote_words} words, got {words}"
    if find_span(evidence, source_text) is None:
        return f'evidence isn\'t in the text word for word: "{evidence}"'
    return None


def check_reply(raw: str, source_text: str, config: Agent1Config) -> CheckResult:
    """Every check here is code, not trust in the agent."""
    result = CheckResult()
    try:
        data = parse_json_reply(raw)
    except ValueError:                       # includes json.JSONDecodeError
        result.problems.append("the reply wasn't valid JSON in the required shape")
        return result
    items = data.get("outcomes") if isinstance(data, dict) else None
    if not isinstance(items, list):
        result.problems.append('the JSON needs an "outcomes" list')
        return result

    result.returned = len(items)
    for n, item in enumerate(items, 1):
        problem = _check_item(item, source_text, config)
        if problem:
            result.problems.append(f"outcome {n}: {problem}")
        else:
            result.candidates.append(Candidate(
                outcome=item["outcome"].strip(),
                evidence=item["evidence"].strip(),
                confidence=float(item["confidence"]),
                stated_not_assessed=item.get("stated_not_assessed", False),
            ))
    return result


# ── Turning checked candidates into outcomes ─────────────────────────────────

def to_outcomes(
    candidates: list[Candidate],
    course_id: str,
    first_number: int,
    config: Agent1Config,
    stage0_rules: Rules,
) -> tuple[list[ExtractedOutcome], list[ExtractedOutcome]]:
    """Build outcomes and split them by confidence:

      >= accept_at             passes
      >= pass_with_review_at   passes, flagged for human review
      below that               held for human review, not used downstream
    """
    passed: list[ExtractedOutcome] = []
    held: list[ExtractedOutcome] = []
    for n, c in enumerate(candidates, first_number):
        outcome = ExtractedOutcome(
            id=f"{course_id}-O{n}",
            course_id=course_id,
            text=" ".join(c.outcome.split()),
            bloom_level=classify_bloom(c.outcome, stage0_rules),     # from the verb, like Stage 0
            topic_tags=sorted(tag_text(c.evidence, stage0_rules)),   # from the quote, not the agent's words
            extraction_confidence=c.confidence,
            source=ExtractionSource.AGENT,
            evidence=c.evidence,
            stated_not_assessed=c.stated_not_assessed,
            needs_review=c.confidence < config.accept_at,
        )
        (passed if c.confidence >= config.pass_with_review_at else held).append(outcome)
    return passed, held

"""
Agent 2: decides whether one assignment assesses one learning outcome.

It only gets the pairs the rules couldn't decide, one pair per call.

WHAT IT SEES: one outcome (with the syllabus words behind it) and one
assignment. Never the other outcomes or assignments, the triage reason,
Agent 1's confidence, or the outcome's Bloom's level. Each pair is judged
on its own, so one judgment can't lean on another, and its depth judgment
isn't anchored to the level the outcome claims.

WHAT CODE ENFORCES, as with Agent 1:
  - a direct or partial verdict must quote the assignment, and code checks
    the quote is really there
  - confidence routes the verdict: pass, pass flagged for review, or hold
  - a pair whose replies keep failing the checks goes to a human; it never
    quietly turns into "not aligned"
  - so does a pair the model never answers (a busy service, even after
    retries): one failed request holds one pair instead of crashing the run
  - with second_opinion on, every pair is judged twice, separately, and a
    pair whose two judgments disagree goes to a person. A model's own
    confidence score can't be trusted to flag its mistakes; disagreement can
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from coursealign.llm import NO_REPLY, LLMUnavailable, parse_json_reply
from coursealign.models import (
    AlignmentResult, AlignmentType, Assignment, BloomLevel, ExtractedOutcome, MatchConfidence,
)
from coursealign.spans import find_span

VERDICTS = [t.value for t in AlignmentType]          # direct, partial, tangential
LEVELS = [level.value for level in BloomLevel]       # remember ... create
SECOND_OPINION = "Second opinion: judge this pair from scratch."
DISAGREE = "Two independent judgments disagreed"        # how a disagreement's reason starts
REPLY_SHAPE = (
    '{"verdict": "direct | partial | tangential", "evidence": "...", '
    '"assessed_level": "apply", "confidence": 0.0, "reason": "..."}'
)


@dataclass
class Agent2Config:
    """data/rules/agent2.json, loaded."""
    rules: list[str]
    accept_at: float
    pass_with_review_at: float
    min_quote_words: int
    max_quote_words: int
    max_attempts: int
    max_parallel_calls: int
    second_opinion: bool = False

    @classmethod
    def from_file(cls, path: str | Path) -> "Agent2Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            rules=raw["rules"],
            accept_at=raw["confidence"]["accept_at"],
            pass_with_review_at=raw["confidence"]["pass_with_review_at"],
            min_quote_words=raw["quote_words"]["min"],
            max_quote_words=raw["quote_words"]["max"],
            max_attempts=raw["max_attempts"],
            max_parallel_calls=raw["max_parallel_calls"],
            second_opinion=raw.get("second_opinion", False),
        )


# ── Prompts ──────────────────────────────────────────────────────────────────

def system_prompt(config: Agent2Config) -> str:
    rules = "\n".join(f"{n}. {rule}" for n, rule in enumerate(config.rules, 1))
    return (
        "You decide whether one assignment assesses one learning outcome.\n\n"
        f"Rules:\n{rules}\n\n"
        f"Each evidence quote must be {config.min_quote_words} to {config.max_quote_words} words long.\n\n"
        f"Bloom's levels, lowest to highest: {', '.join(LEVELS)}\n\n"
        f"Reply with JSON only, in exactly this shape:\n{REPLY_SHAPE}"
    )


def assignment_text(a: Assignment) -> str:
    """Everything the agent sees about the assignment, and all a quote may come from."""
    return f"Title: {a.title}\nDescription: {a.description}\nRubric: {', '.join(a.rubric_keywords)}"


def user_message(outcome: ExtractedOutcome, a: Assignment) -> str:
    lines = [f"Pair: {outcome.id} x {a.id}", "", f"Outcome: {outcome.text}"]
    if outcome.evidence and outcome.evidence != outcome.text:
        lines.append(f'Syllabus wording: "{outcome.evidence}"')
    lines += ["", f"Assignment ({a.type})", assignment_text(a)]
    return "\n".join(lines)


def retry_message(problems: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return f"Your reply failed these checks:\n{listed}\n\nReply with the full corrected JSON."


# ── Checking a reply ─────────────────────────────────────────────────────────

@dataclass
class Verdict:
    kind: AlignmentType
    evidence: str
    assessed_level: BloomLevel | None
    confidence: float
    reason: str


def check_reply(raw: str, a: Assignment, config: Agent2Config) -> tuple[Verdict | None, list[str]]:
    """Every check here is code, not trust in the agent."""
    try:
        data = parse_json_reply(raw)
    except ValueError:
        return None, ["the reply wasn't valid JSON in the required shape"]
    if not isinstance(data, dict):
        return None, ["the reply must be one JSON object"]

    problems = []
    verdict, confidence, reason = data.get("verdict"), data.get("confidence"), data.get("reason")
    if verdict not in VERDICTS:
        problems.append(f'"verdict" must be one of: {", ".join(VERDICTS)}')
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        problems.append('"confidence" must be a number from 0 to 1')
    if not isinstance(reason, str) or not reason.strip():
        problems.append('"reason" must be a sentence')

    evidence, level = "", None
    if verdict in ("direct", "partial"):
        evidence, level = data.get("evidence"), data.get("assessed_level")
        if not isinstance(evidence, str) or not evidence.strip():
            problems.append('a direct or partial verdict needs "evidence" quoted from the assignment')
        else:
            words = len(evidence.split())
            if not config.min_quote_words <= words <= config.max_quote_words:
                problems.append(f"evidence must be {config.min_quote_words}-{config.max_quote_words} words, got {words}")
            elif find_span(evidence, assignment_text(a)) is None:
                problems.append(f'evidence isn\'t in the assignment word for word: "{evidence}"')
        if level not in LEVELS:
            problems.append(f'"assessed_level" must be one of: {", ".join(LEVELS)}')

    if problems:
        return None, problems
    return Verdict(
        kind=AlignmentType(verdict),
        evidence=evidence.strip() if evidence else "",
        assessed_level=BloomLevel(level) if level else None,
        confidence=float(confidence),
        reason=reason.strip(),
    ), []


def tier(confidence: float, config: Agent2Config) -> MatchConfidence:
    if confidence >= config.accept_at:
        return MatchConfidence.HIGH          # passes
    if confidence >= config.pass_with_review_at:
        return MatchConfidence.MEDIUM        # passes, flagged for review
    return MatchConfidence.LOW               # held for a human


# ── Judging ──────────────────────────────────────────────────────────────────

@dataclass
class PairResult:
    result: AlignmentResult
    log: list[dict]          # one entry per attempt
    calls: int


def held_for_a_person(outcome: ExtractedOutcome, a: Assignment, reasoning: str) -> AlignmentResult:
    """A pair a person decides. Until then it counts as neither aligned nor not."""
    return AlignmentResult(
        outcome_id=outcome.id, assignment_id=a.id, score=0.0,
        confidence_tier=MatchConfidence.LOW, alignment_type=AlignmentType.TANGENTIAL,
        reasoning=reasoning, needs_human_review=True,
    )


def is_aligned(result: AlignmentResult) -> bool:
    return result.alignment_type in (AlignmentType.DIRECT, AlignmentType.PARTIAL)


def judge_pair(outcome: ExtractedOutcome, a: Assignment, llm, config: Agent2Config, system: str) -> PairResult:
    first = judge_once(outcome, a, llm, config, system, user_message(outcome, a), "")
    if not config.second_opinion or first.result.confidence_tier == MatchConfidence.LOW:
        return first                    # a held pair goes to a person already
    second = judge_once(outcome, a, llm, config, system,
                        f"{SECOND_OPINION}\n\n{user_message(outcome, a)}", " (2nd opinion)")
    log, calls = first.log + second.log, first.calls + second.calls

    if second.result.confidence_tier == MatchConfidence.LOW:
        # No usable second judgment (bad replies or no reply): keep the first,
        # but flag it, since nothing confirmed it.
        flagged = replace(first.result, needs_human_review=True,
                          confidence_tier=min(first.result.confidence_tier, MatchConfidence.MEDIUM,
                                              key=[MatchConfidence.LOW, MatchConfidence.MEDIUM,
                                                   MatchConfidence.HIGH].index),
                          reasoning=first.result.reasoning + " (A second opinion couldn't be checked.)")
        return PairResult(flagged, log, calls)
    if is_aligned(first.result) == is_aligned(second.result):
        return PairResult(first.result, log, calls)
    reasoning = (f"{DISAGREE}: one said {first.result.alignment_type.value} ({first.result.score:.2f}), "
                 f"the other said {second.result.alignment_type.value} ({second.result.score:.2f}).")
    return PairResult(held_for_a_person(outcome, a, reasoning), log, calls)


def judge_once(outcome: ExtractedOutcome, a: Assignment, llm, config: Agent2Config, system: str,
               opening: str, label: str) -> PairResult:
    """One judgment: ask, check, retry with the failures spelled out, or hold."""
    messages = [{"role": "user", "content": opening}]
    log: list[dict] = []
    problems: list[str] = []
    for attempt in range(1, config.max_attempts + 1):
        try:
            raw = llm.complete(system, messages)
        except LLMUnavailable as error:
            # The model never answered, even after waiting and retrying. Another
            # attempt here would only wait again, so the pair goes to a person.
            log.append({"pair": f"{outcome.id} x {a.id}{label}", "attempt": attempt,
                        "problems": [f"{NO_REPLY}: {error}"], "unavailable": True})
            return PairResult(held_for_a_person(outcome, a, f"{NO_REPLY}: {error}"), log, attempt - 1)
        verdict, problems = check_reply(raw, a, config)
        log.append({"pair": f"{outcome.id} x {a.id}{label}", "attempt": attempt, "problems": problems})
        if verdict is not None:
            t = tier(verdict.confidence, config)
            return PairResult(AlignmentResult(
                outcome_id=outcome.id, assignment_id=a.id,
                score=verdict.confidence, confidence_tier=t,
                alignment_type=verdict.kind, reasoning=verdict.reason,
                needs_human_review=t != MatchConfidence.HIGH,
                evidence=verdict.evidence, assessed_level=verdict.assessed_level,
            ), log, attempt)
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": retry_message(problems)},
        ]

    # Every attempt failed the checks. A human decides; it never counts as "no".
    reasoning = "Agent 2's replies failed the checks: " + "; ".join(problems)
    return PairResult(held_for_a_person(outcome, a, reasoning), log, config.max_attempts)


def judge_pairs(pairs: list[tuple[ExtractedOutcome, Assignment]], llm, config: Agent2Config) -> list[PairResult]:
    """Judge pairs in parallel. Each is its own call that sees only its own
    pair; results come back in the same order as the pairs."""
    if not pairs:
        return []
    system = system_prompt(config)
    with ThreadPoolExecutor(max_workers=config.max_parallel_calls) as pool:
        return list(pool.map(lambda pair: judge_pair(pair[0], pair[1], llm, config, system), pairs))

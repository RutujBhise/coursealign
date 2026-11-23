"""
Finding a quote inside a source text: forgiving about formatting, strict
about words.

Used in two places:
  - Agent 1's evidence check: is the quote really in the syllabus?
  - Scoring: which labeled outcome does a quote point at?
"""

import re

_SWAPS = {
    "\u2014": "-", "\u2013": "-",      # em dash, en dash
    "\u2018": "'", "\u2019": "'",      # curly single quotes
    "\u201c": '"', "\u201d": '"',      # curly double quotes
}


def normalize(text: str) -> str:
    """Lowercase, straighten quotes and dashes, and even out spacing."""
    for curly, plain in _SWAPS.items():
        text = text.replace(curly, plain)
    text = re.sub(r"\s*-+\s*", " - ", text.lower())   # "a—b", "a - b", "a--b" all look alike
    return " ".join(text.split())


def find_span(quote: str, source: str) -> tuple[int, int] | None:
    """Where the quote sits in the source, as (start, end) in normalized text.

    Case, line breaks, curly quotes, and dash styles can differ. The words
    can't: a paraphrase returns None.
    """
    q = normalize(quote).strip(" .,;:!?\"'")
    if not q:
        return None
    start = normalize(source).find(q)
    return (start, start + len(q)) if start >= 0 else None


def overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    """How many characters two spans share."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def best_match(quote: str, candidates: list[tuple[str, str]], source: str, strict: bool = False) -> str | None:
    """The id of the (id, quote) candidate that overlaps this quote the most.

    Lenient (the default, used for scoring): the overlap must cover at least
    half of the shorter quote, so a reworded or trimmed quote still counts.
    Strict (used for a person's decisions): it must cover at least half of
    the longer one, so a decision only lands on an outcome quoting roughly
    the same words, never on a bigger outcome that happens to contain them.
    """
    span = find_span(quote, source)
    if span is None:
        return None
    best, best_shared = None, 0
    for candidate_id, candidate in candidates:
        other = find_span(candidate, source)
        if other is None:
            continue
        shared = overlap(span, other)
        lengths = (span[1] - span[0], other[1] - other[0])
        needed = max(lengths) if strict else min(lengths)
        if shared * 2 >= needed and shared > best_shared:
            best, best_shared = candidate_id, shared
    return best

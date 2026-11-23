"""
A fake Agent 2 that answers from the hand labels.

For each pair, LabelJudge looks up the labels and answers the way a perfect
judge would. Running the pipeline with it shows the pipeline's ceiling: what
it would report if Agent 2 never made a mistake. And it runs offline.

It also plants four problems, so every path through the code runs:
a quote that isn't in the assignment (retry), two unsure answers (flagged,
then held), and one shallow assessment (a Bloom's gap).

None of this says anything about how well a real model does.
"""

import json
import re

from coursealign.evaluate import Label, match_labels
from coursealign.models import ExtractedOutcome, RawSyllabus
from coursealign.stage0 import Rules, classify_bloom

PAIR_LINE = re.compile(r"^Pair: (\S+) x (\S+)$", re.MULTILINE)
OUTCOME_LINE = re.compile(r"^Outcome: (.*)$", re.MULTILINE)
QUOTE_LINE = re.compile(r'^Syllabus wording: "(.*)"$', re.MULTILINE)
DESCRIPTION_LINE = re.compile(r"^Description: (.*)$", re.MULTILINE)

# (outcome id, assignment id) -> what this pair's answer does differently
PLANTED = {
    ("CS4550-O1", "A2"):      {"bad_quote_first": True},      # first reply fails the quote check
    ("CS4550-O2", "A4"):      {"confidence": 0.7},            # passes, flagged for review
    ("CS4550-O2", "A5"):      {"confidence": 0.5},            # held for a human
    ("CS3200-O5", "FINAL"):   {"assessed_level": "understand"},  # shallower than the outcome
}


class LabelJudge:
    def __init__(self, syllabi: dict[str, RawSyllabus], gold: dict[str, list[Label]], rules: Rules):
        self.syllabi, self.gold, self.rules = syllabi, gold, rules

    def __call__(self, system: str, messages: list[dict]) -> str:
        text = messages[0]["content"]
        outcome_id, assignment_id = PAIR_LINE.search(text).groups()
        course = outcome_id.rsplit("-", 1)[0]
        outcome_text = OUTCOME_LINE.search(text).group(1)
        quote = QUOTE_LINE.search(text)
        probe = ExtractedOutcome(id=outcome_id, course_id=course, text=outcome_text,
                                 evidence=quote.group(1) if quote else outcome_text)

        labels = self.gold.get(course)
        if not labels:          # a course added later, with no hand labels: a person decides
            return json.dumps({"verdict": "tangential", "evidence": "", "assessed_level": None,
                               "confidence": 0.5,
                               "reason": "There are no hand labels for this course, so the fake judge can't tell."})
        matched = match_labels([probe], labels, self.syllabi[course].raw_text)
        aligned = outcome_id in matched and assignment_id in labels[matched[outcome_id]].aligned
        plant = PLANTED.get((outcome_id, assignment_id), {})
        confidence = plant.get("confidence", 0.9)

        if not aligned:
            return json.dumps({"verdict": "tangential", "evidence": "", "assessed_level": None,
                               "confidence": confidence, "reason": "The hand labels say it doesn't."})

        # The description's first words: verbatim in the assignment, and long
        # enough for the minimum quote length (a title like "Final Exam" isn't).
        evidence = " ".join(DESCRIPTION_LINE.search(text).group(1).split()[:10])
        if plant.get("bad_quote_first") and len(messages) == 1:
            evidence = "a sentence that is not in the assignment"
        level = plant.get("assessed_level")
        if level is None:
            outcome_level = classify_bloom(outcome_text, self.rules)
            level = outcome_level.value if outcome_level else "apply"
        return json.dumps({"verdict": "direct", "evidence": evidence, "assessed_level": level,
                           "confidence": confidence,
                           "reason": "The hand labels say this assignment assesses the outcome."})

"""
Turn a course you copied by hand into the files CourseAlign reads.

You write plain text, so nothing you paste (quotes, line breaks) can break JSON:

  data/ocw/<course>_assignments.txt     one block per assignment:

      === PS1 | problem set | Problem set 1: The state machine paradigm
      The handout's text, pasted as it is, any number of lines.
      RUBRIC: correctness, design       (optional: only criteria the handout lists)

  data/ocw/<course>_labels.txt          your labels, one block per outcome:

      === objective 1
      QUOTE: the outcome, copied exactly from the syllabus file
      ASSESSED BY: PS1, P2              (or: none)
      NOT ASSESSED: yes                 (only if the syllabus says it won't be assessed)
      NOTE: why, if it's a judgment call

RUN:
  python import_course.py MIT6005
  python import_course.py MIT6005 --paste    fill the assignments by pasting each
                                              one into this window, no editor needed
  python import_course.py MIT6005 --label    write your labels by answering questions,
                                              one outcome at a time

It writes data/assignments/mit6005_assignments.json and puts the labels into
data/gold/alignment_labels.json. First it checks what a typo would quietly
break: every label quote must really be in the syllabus, and every assignment
a label names must exist. If anything fails, it writes nothing.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from coursealign.spans import find_span

ID = re.compile(r"[A-Za-z0-9_-]+")
PLACEHOLDER = "PASTE"                      # the templates say PASTE ... HERE
LABEL_KEYS = ("QUOTE:", "ASSESSED BY:", "NOT ASSESSED:", "NOTE:")


def blocks(text: str) -> list[tuple[str, list[str]]]:
    """(header, body lines) for each block starting with '==='. Lines before the first are ignored."""
    found: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith("==="):
            found.append((line[3:].strip(), []))
        elif found:
            found[-1][1].append(line)
    return found


def parse_assignments(text: str, course: str) -> tuple[list[dict], list[str]]:
    rows, problems, seen = [], [], set()
    for header, body in blocks(text):
        parts = [p.strip() for p in header.split("|")]
        if len(parts) != 3 or not all(parts):
            problems.append(f"'=== {header}': the first line needs ID | type | title")
            continue
        aid, kind, title = parts
        if not ID.fullmatch(aid) or aid in seen:
            problems.append(f"'{aid}': an id needs letters, digits, - or _, and each id once")
            continue
        seen.add(aid)
        rubric, points, week, lines = [], 100, 0, []
        for line in body:
            key = line.strip().upper()
            if key.startswith("RUBRIC:"):
                rubric = [k.strip() for k in line.split(":", 1)[1].split(",") if k.strip()]
            elif key.startswith("POINTS:") and line.split(":", 1)[1].strip().isdigit():
                points = int(line.split(":", 1)[1])
            elif key.startswith("WEEK:") and line.split(":", 1)[1].strip().isdigit():
                week = int(line.split(":", 1)[1])
            else:
                lines.append(line.rstrip())
        description = "\n".join(lines).strip()
        if not description:
            problems.append(f"{aid}: the description is empty")
        elif PLACEHOLDER in description or PLACEHOLDER in title:
            problems.append(f"{aid}: still has the PASTE ... HERE placeholder")
        if points <= 0:
            problems.append(f"{aid}: points must be above 0")
        rows.append({"id": aid, "course": course, "title": title, "type": kind, "description": description,
                     "rubric_keywords": rubric, "points": points, "week": week})
    if not rows and not problems:
        problems.append("no assignments found: each one starts with a line like '=== PS1 | problem set | Title'")
    return rows, problems


def parse_labels(text: str, syllabus: str, assignment_ids: set[str]) -> tuple[list[dict], list[str]]:
    labels, problems = [], []
    for n, (header, body) in enumerate(blocks(text), 1):
        name = header or f"outcome {n}"
        fields: dict[str, str] = {}
        current = None
        for line in body:
            stripped = line.strip()
            key = next((k for k in LABEL_KEYS if stripped.upper().startswith(k)), None)
            if key:
                current = key
                fields[key] = stripped[len(key):].strip()
            elif stripped and current:                      # a wrapped line continues the field above it
                fields[current] += " " + stripped
        quote = fields.get("QUOTE:", "")
        if not quote or PLACEHOLDER in quote:
            problems.append(f"{name}: needs a QUOTE: line with the outcome copied from the syllabus")
            continue
        if find_span(quote, syllabus) is None:
            problems.append(f"{name}: this quote isn't in the syllabus word for word: \"{quote[:80]}\"")
        assessed = fields.get("ASSESSED BY:", "").strip()
        aligned = [] if assessed.lower() in ("", "none", "-") else [a.strip() for a in assessed.split(",") if a.strip()]
        unknown = [a for a in aligned if a not in assignment_ids]
        if unknown:
            problems.append(f"{name}: no assignment called {', '.join(unknown)}")
        label = {"quote": quote, "aligned": aligned}
        if fields.get("NOT ASSESSED:", "").strip().lower() in ("yes", "true", "y"):
            label["stated_not_assessed"] = True
            if aligned:
                problems.append(f"{name}: marked NOT ASSESSED but also ASSESSED BY {', '.join(aligned)}")
        if fields.get("NOTE:"):
            label["note"] = fields["NOTE:"]
        labels.append(label)
    return labels, problems


def split_file(text: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """The lines before the first block, and each block, so the file can be written back."""
    preamble, found = [], blocks(text)
    for line in text.splitlines():
        if line.startswith("==="):
            break
        preamble.append(line)
    return preamble, found


def join_file(preamble: list[str], found: list[tuple[str, list[str]]]) -> str:
    lines = list(preamble)
    for header, body in found:
        lines.append(f"=== {header}")
        lines.extend(body)
    return "\n".join(lines).rstrip() + "\n"


def fill_by_pasting(course: str, root: Path = ROOT, read=input, show=print) -> int:
    """Ask for each assignment that still has its placeholder, and save after each one.
    Returns how many were filled. Typing SKIP leaves one for later."""
    stem = course.lower()
    source = root / "data" / "ocw" / f"{stem}_assignments.txt"
    if not source.exists():
        raise SystemExit(f"Write the template in {source.relative_to(root)} first.")
    preamble, found = split_file(source.read_text(encoding="utf-8-sig"))
    todo = [i for i, (_, body) in enumerate(found) if PLACEHOLDER in "\n".join(body) or not "".join(body).strip()]
    if not todo:
        show("  Every assignment already has its text.")
        return 0
    filled = 0
    for n, i in enumerate(todo, 1):
        header = found[i][0]
        show(f"\n[{n} of {len(todo)}] {header}")
        show("  Copy its text, paste it here, then type END on a line by itself and press Enter.")
        show("  (Type SKIP instead to leave this one for later.)")
        pasted: list[str] = []
        while True:
            line = read()
            if line.strip() == "END":
                break
            if line.strip() == "SKIP" and not pasted:
                pasted = None
                break
            pasted.append(line.rstrip())
        if pasted is None or not "".join(pasted).strip():
            show("  Skipped.")
            continue
        found[i] = (header, pasted)
        source.write_text(join_file(preamble, found), encoding="utf-8")      # saved after each one
        filled += 1
        show(f"  Saved ({len(pasted)} lines).")
    return filled


def label_by_asking(course: str, root: Path = ROOT, read=input, show=print) -> int:
    """Ask for each outcome and which assignments assess it, checking each answer as it
    comes. Every label is added to data/ocw/<course>_labels.txt as soon as it's given.
    Returns how many were added."""
    stem = course.upper().lower()
    data = root / "data"
    syllabi = sorted((data / "syllabi").glob(f"{stem}_*.txt"))
    source = data / "ocw" / f"{stem}_assignments.txt"
    if len(syllabi) != 1 or not source.exists():
        raise SystemExit("Add the syllabus and the assignments first.")
    syllabus = syllabi[0].read_text(encoding="utf-8-sig")
    rows, problems = parse_assignments(source.read_text(encoding="utf-8-sig"), course.upper())
    if problems:
        raise SystemExit("Finish the assignments first:\n" + "\n".join(f"  - {p}" for p in problems))
    by_id = {r["id"]: r for r in rows}
    labels_file = data / "ocw" / f"{stem}_labels.txt"

    show("\nAssignments: " + ", ".join(f"{r['id']} ({r['title']})" for r in rows))
    show("Judge only from each assignment's own text: that's all the pipeline sees too.")
    added = 0
    while True:
        show(f"\nOutcome {added + 1}: paste it from the syllabus file, word for word, then press Enter.")
        show("  (Press Enter on an empty line when you've done every outcome.)")
        quote = read().strip()
        if not quote:
            break
        if find_span(quote, syllabus) is None:
            show("  That isn't in the syllabus word for word. Copy it again from the syllabus file.")
            continue
        while True:
            show("  Which assignments explicitly require this skill? IDs separated by commas, or none.")
            show("  (Type show and an ID, like show P1, to read that assignment's text first.)")
            answer = read().strip()
            if answer.lower().startswith("show "):
                wanted = answer[5:].strip()
                show(by_id[wanted]["description"] if wanted in by_id else f"  There's no assignment called {wanted}.")
                continue
            ids = [] if answer.lower() in ("none", "-", "") else [a.strip() for a in answer.split(",") if a.strip()]
            unknown = [a for a in ids if a not in by_id]
            if unknown:
                show(f"  There's no assignment called {', '.join(unknown)}. The ids are: {', '.join(by_id)}")
                continue
            break
        show("  Does the syllabus say this outcome won't be assessed? y or n")
        not_assessed = read().strip().lower().startswith("y")
        show("  A note, if it was a judgment call (or press Enter to skip):")
        note = read().strip()
        block = [f"=== outcome {added + 1}", f"QUOTE: {quote}", f"ASSESSED BY: {', '.join(ids) or 'none'}"]
        if not_assessed:
            block.append("NOT ASSESSED: yes")
        if note:
            block.append(f"NOTE: {note}")
        existing = labels_file.read_text(encoding="utf-8-sig") if labels_file.exists() else ""
        labels_file.write_text(existing + ("\n" if existing and not existing.endswith("\n") else "")
                               + "\n".join(block) + "\n", encoding="utf-8")
        added += 1
        show("  Saved.")
    return added


def import_course(course: str, root: Path = ROOT) -> list[str]:
    """Write the course's assignments (and labels, if there's a labels file). Returns what it did."""
    course = course.upper()
    stem = course.lower()
    data = root / "data"
    syllabi = sorted((data / "syllabi").glob(f"{stem}_*.txt"))
    if len(syllabi) != 1:
        raise SystemExit(f"Expected one syllabus file called data/syllabi/{stem}_<name>.txt, found {len(syllabi)}.")
    syllabus = syllabi[0].read_text(encoding="utf-8-sig")

    source = data / "ocw" / f"{stem}_assignments.txt"
    if not source.exists():
        raise SystemExit(f"Write the assignments in {source.relative_to(root)} first.")
    rows, problems = parse_assignments(source.read_text(encoding="utf-8-sig"), course)

    labels_source = data / "ocw" / f"{stem}_labels.txt"
    labels = None
    if labels_source.exists():
        labels, label_problems = parse_labels(labels_source.read_text(encoding="utf-8-sig"), syllabus,
                                              {r["id"] for r in rows})
        problems += label_problems
    if problems:
        raise SystemExit("Nothing was written. Fix these first:\n" + "\n".join(f"  - {p}" for p in problems))

    out = data / "assignments" / f"{stem}_assignments.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    done = [f"{len(rows)} assignments written to {out.relative_to(root)}"]
    if labels is None:
        done.append(f"No labels yet: write {labels_source.relative_to(root)}, before running any model on this course.")
    else:
        gold_path = data / "gold" / "alignment_labels.json"
        gold = json.loads(gold_path.read_text(encoding="utf-8-sig"))
        gold["courses"][course] = labels
        gold_path.write_text(json.dumps(gold, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        done.append(f"{len(labels)} labels saved for {course} in {gold_path.relative_to(root)}")
    return done


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a not in ("--paste", "--label")]
    if len(args) != 1:
        raise SystemExit("Usage: python import_course.py MIT6005   (add --paste or --label)")
    if "--paste" in sys.argv:
        fill_by_pasting(args[0])
        print()
    if "--label" in sys.argv:
        label_by_asking(args[0])
        print()
    for line in import_course(args[0]):
        print(f"  {line}")

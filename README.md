# CourseAlign

**Does each course actually assess what its syllabus says it teaches?**

CourseAlign reads a course's syllabus and its assignments, maps every stated learning outcome to the assignments that assess it, and reports the gaps: outcomes nothing tests, assignments that test nothing the syllabus states, and outcomes tested more shallowly than they promise. Rules settle the clear cases. Two LLM agents handle the rest under checks written in code. A person settles anything the agents aren't sure about, once.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/dashboard-dark.png">
  <img alt="The CourseAlign dashboard: an alignment map for CS4530, with two untested outcomes marked in red in the margin and the gap list beside the map" src="docs/dashboard-light.png">
</picture>

## How it works

```mermaid
flowchart TD
    input[Syllabus and assignments] --> extract[Rules: pull outcomes from lists]
    extract -- prose left over --> agent1[Agent 1: read the prose]
    agent1 -- a quote failed the check --> agent1
    extract --> outcomeReviews[Apply saved outcome decisions]
    agent1 --> outcomeReviews
    outcomeReviews --> triage[Rules: triage every outcome-assignment pair]
    triage --> pairReviews[Apply saved pair decisions]
    pairReviews --> agent2[Agent 2: judge each unsure pair, one per call]
    agent2 --> gaps[Gap analysis]
    gaps --> report[Report and review queue]
    report -. a person decides .-> saved[(Saved decisions)]
    saved -. every later run .-> outcomeReviews
    saved -. every later run .-> pairReviews
```

The pipeline is a LangGraph state machine (`src/coursealign/graph.py`), one run per course.

- **Rules first, agents second.** Deterministic rules extract list-style outcomes and triage every pair. Only the ambiguous cases reach an LLM. On the included data, the rules settle 122 of 168 pairs, which cuts agent requests from 171 to 50.
- **Every agent claim is checked in code.** Agent 1 must quote the syllabus words behind each outcome; Agent 2 must quote the assignment words behind each match. A paraphrase fails the check, the agent gets one retry with the failure spelled out, and anything still failing goes to a person. It never silently becomes "no."
- **Context isolation.** Agent 1 never sees the assignments. Agent 2 sees one pair per call, and not the outcome's Bloom's level, so its judgment of depth isn't anchored to what the outcome claims.
- **Confidence decides what happens next.** At 0.8 and above a verdict passes, from 0.6 to 0.8 it passes flagged for review, and below 0.6 it waits for a person.
- **A second opinion, because confidence isn't enough.** On the first real-model run, every wrong verdict came with confidence 0.85 or higher. With `--second-opinion`, each pair is judged twice, separately, and a pair whose two judgments disagree waits for a person.
- **A busy model doesn't sink the run.** Temporary failures (a 503 or 529 from a busy model, a 429 rate limit, a dropped connection) are retried with exponential backoff and jitter, paced to the rate limit. A pair the model never answers waits for a person, and after three calls in a row fail, calls pause for two minutes instead of each one waiting out its own retries. Agent 1 is the exception: a syllabus read with a section missing would give a wrong gap report, so the run stops with a plain message instead.
- **People decide once.** Decisions are saved by syllabus quote, so they survive re-runs. They apply on every later run and can teach the rules a word that helps every course.
- **Measured, not assumed.** Hand labels score every stage, triage is also scored with no model at all, and a multi-run eval reports how much the agents vary.

## Quick start

```bash
pip install -r requirements.txt
cd web && npm install && npm run build && cd ..    # builds the dashboard (needs Node.js)
python3 step6_api.py              # dashboard at http://127.0.0.1:8000, API docs at /docs
```

That uses stand-in agents, so it needs no API key. To use a real model:

```bash
export ANTHROPIC_API_KEY=...
COURSEALIGN_LLM=anthropic python3 step6_api.py
```

Or use Gemini's free tier, with a key from Google AI Studio:

```bash
export GEMINI_API_KEY=...
COURSEALIGN_LLM=gemini python3 step6_api.py
```

Requests to Gemini are paced to stay under the free tier's per-minute limit. Set `COURSEALIGN_GEMINI_RPM` to the limit AI Studio shows for your project, and `COURSEALIGN_GEMINI_MODEL` to choose a model. `COURSEALIGN_AGENT1_MODEL` gives Agent 1 its own model: it makes only a few calls a run, and an outcome it misses makes every later stage wrong, so a stronger model with a small daily limit can be worth it (`COURSEALIGN_AGENT1_RPM` paces it, default 5). A request that fails with a temporary error is tried up to 7 times over about one to two minutes; `COURSEALIGN_LLM_ATTEMPTS` changes that. A used-up daily limit stops the run at once instead, since only Google's reset at midnight Pacific fixes it. Real replies are saved (`.cache/llm_replies.json` for Steps 4 and 5 and the dashboard, one file per run for Step 8), so a run cut short by a limit picks up where it stopped; add `--fresh` to Step 4 or Step 8 to ask again.

Add `--courses MIT6005,MIT6170` to Steps 3, 4 and 8 to run and score only those courses, for example a held-out test set.

To rebuild the dashboard after editing it: `cd web && npm install && npm run build`.

## The steps

The project was built in steps, and each step's script still runs and shows its piece on its own.

| Script | What it shows |
|---|---|
| `step1_data_profile.py` | The three courses and their data |
| `step2_stage0.py` | The rules alone: outcome extraction and pair triage, scored against the labels |
| `step3_agent1.py` | Agent 1 reading prose inside the LangGraph graph, with the evidence check and retry |
| `step4_agent2.py` | Agent 2 judging the unsure pairs, then gap analysis |
| `step5_review.py` | The feedback loop: a review, then a second run that applies it (`--interactive` to decide yourself) |
| `step6_api.py` | The API and dashboard (`--check` tours every endpoint without a server) |
| `step8_eval.py` | Several full runs, with the spread of every metric |

Steps 3, 4, 5, and 8 take `--llm anthropic` or `--llm gemini` to use a real model.

## Tests

- `pytest tests/` runs 106 tests: the rules, both agents (with stand-in LLMs), the LLM clients and their retries, the review loop, the service layer, the HTTP API, and color contrast.
- `cd web && npx playwright test` runs 7 end-to-end tests in Chromium, plus axe-core WCAG 2.1 AA checks of every view in light and dark mode.
- GitHub Actions runs all of it on every push (`.github/workflows/ci.yml`).

## Measuring it

With stand-in agents, which answer from the hand labels, the offline numbers are:

| | |
|---|---|
| Pairs decided by rules (labeled outcomes, no model) | 119 of 168, 117 right |
| Agent requests | 50, vs. 171 with no rules |
| Untested outcomes found | 3 of 3, no false alarms |
| Orphan assignments found | 2 of 2, no false alarms |

These test the code, not a model. For real numbers:

```bash
python3 step8_eval.py --llm anthropic --runs 3
```

One caveat matters more than any number above: the three syllabi, the rules, and the labels were written together, so these results are a sanity check, not a benchmark. To measure for real, add syllabi you didn't write (with `POST /api/courses` or as files in `data/`), label them in `data/gold/alignment_labels.json`, and run the eval.

## Design decisions

- **JSON files, not a database.** The data is small, a diff shows exactly what a review changed, and nothing needs a server to inspect.
- **Reviews are applied between runs, not waited for.** A reviewer may come back days later, so the graph never pauses for a person; it applies whatever is saved when it starts.
- **Replies are cached by exact request in the review loop.** A second run then differs from the first only where a decision changed something, and unchanged requests cost nothing. The eval deliberately skips the cache, since it measures variation.
- **esbuild, not Vite.** One dependency, builds in about 100 ms, and the build that was tested is the build that ships.
- **Offline stand-ins.** `offline_graph.py` runs the same graph without LangGraph, and the fake LLMs need no key, so every test runs anywhere.

## Layout

```
data/            syllabi, assignments, rules, hand labels, review decisions
src/coursealign/ the pipeline: stage0 (rules), agent1, agent2, gaps, review,
                 graph (LangGraph), service + api (FastAPI), report, evaluate
web/             the dashboard: React + TypeScript, built with esbuild
tests/           pytest suites and fixtures
web/e2e/         Playwright end-to-end and accessibility specs
step*.py         one runnable script per build step
```

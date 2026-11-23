"""
The whole pipeline as a LangGraph state machine. One run = one course.

    START
      |
    extract_with_rules --(no prose)--------------------------+
      |                                                      |
    (prose to read)                                          |
      v                                                      v
    agent1_read --> agent1_check --(done)--> apply_outcome_reviews
      ^                  |                                   |
      +--(retry / next region)                               v
                                                          triage
                                                             |   (Step 3 stops here)
                                                             v
                                                    apply_pair_reviews
                                                             |
                                                             v
                                                       agent2_judge
                                                             |
                                                             v
                                                        find_gaps --> END

The two review nodes apply a person's saved decisions (Step 5). With no
decisions, they change nothing.

WHY A GRAPH AND NOT A FOR-LOOP: the flow has loops (retry a failed reply,
read the next prose region) and branches (skip Agent 1 when the rules read
everything). LangGraph makes those explicit and easy to extend.

WHY REVIEWS ARE APPLIED, NOT WAITED FOR: a reviewer may come back days later
in the dashboard. So the graph never pauses for a person; it applies the
decisions saved so far at the start of every run.

The state holds only data. Rules, agent configs, LLMs, and saved decisions
are passed in when the graph is built.
"""

from typing import TypedDict

from coursealign import agent1, agent2
from coursealign.agent1 import Agent1Config
from coursealign.agent2 import Agent2Config
from coursealign.gaps import find_gaps
from coursealign.models import (
    AlignmentResult, Assignment, ExtractedOutcome, GapReport, RawSyllabus, ReviewDecision,
)
from coursealign.review import apply_outcome_reviews, human_verdicts
from coursealign.stage0 import (
    Decision, Rules, TriageDecision, UnresolvedRegion, extract_outcomes, triage,
)


class CourseState(TypedDict):
    syllabus: RawSyllabus
    assignments: list[Assignment]
    outcomes: list[ExtractedOutcome]       # passed: from the rules and from Agent 1
    held: list[ExtractedOutcome]           # Agent 1 outcomes below the pass threshold
    pending: list[UnresolvedRegion]        # prose Agent 1 still has to read
    attempt: int                           # Agent 1's attempts on the current region
    last_reply: str                        # Agent 1's latest raw reply
    feedback: list[str]                    # failed checks to send back on a retry
    agent1_log: list[dict]                 # one entry per Agent 1 call
    decisions: list[TriageDecision]        # Stage 0's verdict on every pair
    human_verdicts: list[AlignmentResult]  # a person's saved pair decisions, matched to this run
    judgments: list[AlignmentResult]       # Agent 2's verdict on each pair it was sent
    agent2_log: list[dict]                 # one entry per Agent 2 call
    gaps: list[GapReport]
    review_log: list[dict]                 # which saved decisions landed where
    llm_calls: int


def initial_state(syllabus: RawSyllabus, assignments: list[Assignment]) -> CourseState:
    return CourseState(
        syllabus=syllabus, assignments=assignments, outcomes=[], held=[], pending=[],
        attempt=0, last_reply="", feedback=[], agent1_log=[], decisions=[],
        human_verdicts=[], judgments=[], agent2_log=[], gaps=[], review_log=[], llm_calls=0,
    )


def graph_api():
    """Real LangGraph when it's installed; otherwise the offline stand-in."""
    try:
        import langgraph.graph as api
        return api, "LangGraph"
    except ImportError:
        from coursealign import offline_graph
        return offline_graph, "offline stand-in (pip install langgraph to run the real thing)"


def build_graph(
    stage0_rules: Rules,
    config: Agent1Config,
    llm,
    api=None,
    agent2_config: Agent2Config | None = None,
    judge_llm=None,
    reviews: list[ReviewDecision] = (),
):
    api = api or graph_api()[0]
    system = agent1.system_prompt(config, stage0_rules)
    reviews = list(reviews)

    # ── nodes: each reads the state and returns only the keys it changes ──

    def extract_with_rules(state: CourseState) -> dict:
        outcomes, regions = extract_outcomes(state["syllabus"], stage0_rules)
        return {"outcomes": outcomes, "pending": regions}

    def agent1_read(state: CourseState) -> dict:
        region = state["pending"][0]
        messages = [{"role": "user", "content": agent1.user_message(state["syllabus"], region)}]
        if state["feedback"]:       # a retry: show the agent its last reply and what failed
            messages += [
                {"role": "assistant", "content": state["last_reply"]},
                {"role": "user", "content": agent1.retry_message(state["feedback"])},
            ]
        # If the model stays down, LLMUnavailable stops the run here on purpose:
        # without this section's outcomes, the gap report would be wrong (its
        # assignments would look like orphans), so there's no safe way to go on.
        return {
            "last_reply": llm.complete(system, messages),
            "attempt": state["attempt"] + 1,
            "llm_calls": state["llm_calls"] + 1,
        }

    def agent1_check(state: CourseState) -> dict:
        region = state["pending"][0]
        check = agent1.check_reply(state["last_reply"], region.text, config)
        log = state["agent1_log"] + [{
            "region": region.header,
            "attempt": state["attempt"],
            "returned": check.returned,
            "problems": check.problems,
        }]
        if check.problems and state["attempt"] < config.max_attempts:
            return {"feedback": check.problems, "agent1_log": log}       # route: retry

        # Commit this region: keep what passed, drop (and log) what didn't.
        first = len(state["outcomes"]) + len(state["held"]) + 1
        passed, held = agent1.to_outcomes(
            check.candidates, state["syllabus"].course_id, first, config, stage0_rules,
        )
        return {
            "outcomes": state["outcomes"] + passed,
            "held": state["held"] + held,
            "pending": state["pending"][1:],
            "attempt": 0,
            "feedback": [],
            "agent1_log": log,
        }

    def run_outcome_reviews(state: CourseState) -> dict:
        outcomes, held, log = apply_outcome_reviews(state["syllabus"], state["outcomes"], state["held"], reviews)
        return {"outcomes": outcomes, "held": held, "review_log": state["review_log"] + log}

    def run_triage(state: CourseState) -> dict:
        # Outcomes the syllabus says won't be assessed can't have an assessment
        # gap, so there's no point spending Agent 2 calls on them.
        assessable = [o for o in state["outcomes"] if not o.stated_not_assessed]
        return {"decisions": triage(assessable, state["assignments"], stage0_rules)}

    def run_pair_reviews(state: CourseState) -> dict:
        verdicts, log = human_verdicts(state["syllabus"], state["outcomes"], state["assignments"], reviews)
        return {"human_verdicts": verdicts, "review_log": state["review_log"] + log}

    def agent2_judge(state: CourseState) -> dict:
        settled = {(h.outcome_id, h.assignment_id) for h in state["human_verdicts"]}
        by_outcome = {o.id: o for o in state["outcomes"]}
        by_assignment = {a.id: a for a in state["assignments"]}
        pairs = [
            (by_outcome[d.outcome_id], by_assignment[d.assignment_id])
            for d in state["decisions"]
            if d.decision == Decision.NEEDS_AGENT and (d.outcome_id, d.assignment_id) not in settled
        ]
        results = agent2.judge_pairs(pairs, judge_llm or llm, agent2_config)
        return {
            "judgments": [r.result for r in results],
            "agent2_log": [entry for r in results for entry in r.log],
            "llm_calls": state["llm_calls"] + sum(r.calls for r in results),
        }

    def run_find_gaps(state: CourseState) -> dict:
        return {"gaps": find_gaps(state["outcomes"], state["assignments"], state["decisions"],
                                  state["judgments"], state["human_verdicts"])}

    # ── routes: pure functions of the state ──

    def after_rules(state: CourseState) -> str:
        return "read_prose" if state["pending"] else "no_prose"

    def after_check(state: CourseState) -> str:
        if state["feedback"]:
            return "retry"
        if state["pending"]:
            return "next_region"
        return "done"

    # ── wiring ──

    graph = api.StateGraph(CourseState)
    graph.add_node("extract_with_rules", extract_with_rules)
    graph.add_node("agent1_read", agent1_read)
    graph.add_node("agent1_check", agent1_check)
    graph.add_node("apply_outcome_reviews", run_outcome_reviews)
    graph.add_node("triage", run_triage)

    graph.add_edge(api.START, "extract_with_rules")
    graph.add_conditional_edges("extract_with_rules", after_rules,
                                {"read_prose": "agent1_read", "no_prose": "apply_outcome_reviews"})
    graph.add_edge("agent1_read", "agent1_check")
    graph.add_conditional_edges("agent1_check", after_check,
                                {"retry": "agent1_read", "next_region": "agent1_read",
                                 "done": "apply_outcome_reviews"})
    graph.add_edge("apply_outcome_reviews", "triage")

    if agent2_config is None:
        graph.add_edge("triage", api.END)                  # Step 3 stops here
    else:
        graph.add_node("apply_pair_reviews", run_pair_reviews)
        graph.add_node("agent2_judge", agent2_judge)
        graph.add_node("find_gaps", run_find_gaps)
        graph.add_edge("triage", "apply_pair_reviews")
        graph.add_edge("apply_pair_reviews", "agent2_judge")
        graph.add_edge("agent2_judge", "find_gaps")
        graph.add_edge("find_gaps", api.END)
    return graph.compile()

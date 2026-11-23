// Words the dashboard uses for the API's values, in one place, so the same
// thing is always called the same thing.
import type { CourseReport, DecidedBy, OutcomeChoice, Pair, PairChoice, Report, Severity } from "./types";

export function verdictText(pair: Pair): string {
  if (pair.aligned === null) return "waiting for a person";
  if (!pair.aligned) return "doesn't assess it";
  return pair.kind === "partial" ? "assesses part of it" : "assesses it";
}

export const whoDecided: Record<DecidedBy, string> = {
  rules: "the rules",
  agent2: "Agent 2",
  human: "a person",
};

export const agentVerdictText: Record<string, string> = {
  direct: "assesses it",
  partial: "assesses part of it",
  tangential: "doesn't assess it",
};

export const pairChoiceLabel: Record<PairChoice, string> = {
  direct: "Assesses it",
  partial: "Assesses part of it",
  not_aligned: "Doesn't assess it",
};

export const outcomeChoiceLabel: Record<OutcomeChoice, string> = {
  keep: "Keep it",
  drop: "Drop it",
};

export const decisionLabel: Record<string, string> = { ...pairChoiceLabel, ...outcomeChoiceLabel };

export const severityLabel: Record<Severity, string> = {
  critical: "Critical",
  warning: "Warning",
  info: "Note",
};

export const gapLabel: Record<string, string> = {
  untested_outcome: "Untested outcome",
  weak_coverage: "Only covered in parts",
  blooms_mismatch: "Assessed too shallowly",
  orphan_assessment: "Assesses no outcome",
  stated_not_assessed: "Not assessed, by design",
};

export const triageText: Record<string, string> = {
  auto_match: "matched by the rules",
  auto_reject: "ruled out by the rules",
  needs_agent: "sent to Agent 2",
};

export function percent(share: number): string {
  return `${Math.round(share * 100)}%`;
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

export function shorten(text: string, length = 90): string {
  return text.length <= length ? text : `${text.slice(0, length - 1)}…`;
}

export function when(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function waitingSentence(n: number): string {
  if (n === 0) return "Nothing is waiting for you.";
  return `${plural(n, "item")} ${n === 1 ? "is" : "are"} waiting for you.`;
}

export function usesStandIns(report: Report): boolean {
  return /fake/i.test(`${report.agents.agent1} ${report.agents.agent2}`);
}

export function agentsLabel(report: Report): string {
  if (usesStandIns(report)) return "with stand-in agents, not a model";
  const model = /\(([^)]+)\)/.exec(report.agents.agent1)?.[1];
  return `with ${model ?? report.agents.agent1}`;
}

export function summarySentence(report: Report): string {
  const t = report.totals;
  const gaps = [
    plural(t.gaps.critical, "critical gap"),
    plural(t.gaps.warning, "warning"),
    plural(t.gaps.info, "note"),
  ].join(", ");
  return (
    `The rules settled ${t.decided_by.rules} of ${t.pairs} outcome-assignment pairs on their own, ` +
    `so the agents made ${t.agent_requests} requests instead of ${t.requests_without_rules} ` +
    `(${percent(t.requests_saved)} fewer). ${waitingSentence(t.waiting_for_a_person)} ` +
    `This run found ${gaps}.`
  );
}

export function courseSentence(course: CourseReport): string {
  const s = course.stats;
  const onMap = course.outcomes.filter((o) => o.status === "passed" || o.status === "flagged").length;
  const people = s.decided_by.human > 0 ? `, and people settled ${s.decided_by.human}` : "";
  const waiting = s.undecided > 0 ? ` ${waitingSentence(s.undecided)}` : "";
  return (
    `${plural(onMap, "outcome")} and ${plural(course.assignments.length, "assignment")}. ` +
    `The rules settled ${s.decided_by.rules} of ${s.pairs} pairs and Agent 2 settled ${s.decided_by.agent2}${people}.` +
    waiting
  );
}

export function criticalNote(course: CourseReport): string {
  const n = course.gaps.filter((g) => g.severity === "critical").length;
  return n === 0 ? "no critical gaps" : plural(n, "critical gap");
}

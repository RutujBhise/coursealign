// The API's JSON, as TypeScript. These mirror src/coursealign/report.py and
// service.py field for field, so a mismatch shows up as a type error.

export type Severity = "critical" | "warning" | "info";
export type DecidedBy = "rules" | "agent2" | "human";
export type OutcomeStatus = "passed" | "flagged" | "held" | "not_assessed";
export type PairChoice = "direct" | "partial" | "not_aligned";
export type OutcomeChoice = "keep" | "drop";

export interface Outcome {
  id: string;
  text: string;
  quote: string;
  source: "rules" | "agent1";
  bloom: string | null;
  confidence: number;
  status: OutcomeStatus;
  topics: string[];
}

export interface Assignment {
  id: string;
  title: string;
  type: string;
  points: number;
  week: number;
}

export interface Pair {
  outcome_id: string;
  assignment_id: string;
  aligned: boolean | null; // null: waiting for a person
  kind: "direct" | "partial" | null;
  decided_by: DecidedBy;
  triage: "auto_match" | "auto_reject" | "needs_agent" | null;
  depth: string | null;
  confidence: number | null;
  flagged: boolean;
  reason: string;
  evidence: string;
}

export interface Gap {
  type: string;
  severity: Severity;
  outcome_id: string | null;
  assignment_id: string | null;
  description: string;
}

export interface AgentVerdict {
  verdict: string;
  confidence: number;
  reason: string;
  evidence: string;
  depth: string | null;
}

export interface QueueItem {
  kind: "outcome" | "pair";
  outcome_id: string;
  assignment_id: string | null;
  why: string;
  outcome_text: string;
  quote: string;
  assignment_title: string | null;
  agent: AgentVerdict | null;
}

export interface ReviewLogEntry {
  kind: string;
  decision: string;
  quote: string;
  assignment: string;
  applied_to: string | null;
}

export interface CourseReport {
  course_id: string;
  title: string;
  outcomes: Outcome[];
  assignments: Assignment[];
  pairs: Pair[];
  gaps: Gap[];
  queue: QueueItem[];
  stats: {
    agent1_requests: number;
    agent2_requests: number;
    pairs: number;
    decided_by: Record<DecidedBy, number>;
    undecided: number;
  };
  review_log: ReviewLogEntry[];
}

export interface Scores {
  courses: string[];
  outcomes: { labeled: number; found: number; unlabeled_extras: string[] };
  rules: { decided: number; right: number; wrong_matches: string[]; missed: string[] };
  agent2: { passed: number; right: number; flagged: number; wrong: string[] };
  pairs: { scored: number; decided: number; right: number; undecided: number; by_a_person: number };
  gaps: {
    untested_expected: number;
    untested_found: number;
    untested_false_alarms: string[];
    orphans_expected: number;
    orphans_found: number;
    orphans_false_alarms: string[];
  };
}

export interface Report {
  ran_at: string;
  graph_engine: string;
  agents: { agent1: string; agent2: string };
  settings: Record<string, number>;
  learned_words: Record<string, string[]>;
  topics: string[];
  totals: {
    courses: number;
    outcomes: number;
    held_outcomes: number;
    pairs: number;
    agent_requests: number;
    model_calls: number;
    requests_without_rules: number;
    requests_saved: number;
    decided_by: Record<DecidedBy, number>;
    undecided: number;
    waiting_for_a_person: number;
    gaps: Record<Severity, number>;
  };
  courses: CourseReport[];
  scores: Scores | null;
}

export interface Setting {
  value: number;
  min: number;
  max: number;
  step: number;
  about: string;
  changed: boolean;
}

export type Settings = Record<string, Setting>;

export interface Teach {
  topic: string;
  term: string;
}

export interface SavedDecision {
  index: number;
  kind: "outcome" | "pair";
  course_id: string;
  quote: string;
  decision: string;
  assignment_id: string | null;
  assessed_level: string | null;
  note: string;
  teach: Teach[];
  by: string;
  at: string;
}

export interface NewDecision {
  course_id: string;
  outcome_id: string;
  decision: string;
  assignment_id?: string;
  note?: string;
  teach?: Teach[];
}

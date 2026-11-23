// Everything the agents weren't sure about, phrased as a question a person
// can answer in one click.
import { useEffect, useId, useState } from "react";
import { agentVerdictText, outcomeChoiceLabel, pairChoiceLabel, percent } from "../format";
import type { NewDecision, Outcome, QueueItem, Report } from "../types";

type Item = QueueItem & { course_id: string; outcome?: Outcome };

const keyOf = (item: Item) => [item.course_id, item.kind, item.outcome_id, item.assignment_id ?? ""].join("|");

interface Props {
  report: Report;
  onDecide: (decision: NewDecision) => Promise<boolean>;
}

export function ReviewQueue({ report, onDecide }: Props) {
  const items: Item[] = report.courses.flatMap((course) =>
    course.queue.map((item) => ({
      ...item,
      course_id: course.course_id,
      outcome: course.outcomes.find((o) => o.id === item.outcome_id),
    })),
  );
  const [answered, setAnswered] = useState<Record<string, string>>({});
  useEffect(() => setAnswered({}), [report]); // a new run brings a new queue

  async function answer(item: Item, decision: NewDecision, label: string) {
    if (await onDecide(decision)) setAnswered((done) => ({ ...done, [keyOf(item)]: label }));
  }

  return (
    <section className="queue" aria-labelledby="queue-title">
      <h2 id="queue-title" className="section-title">
        Waiting for you
      </h2>
      {items.length === 0 ? (
        <p className="section-lede">Nothing is waiting for you. Every outcome and pair in this run is decided.</p>
      ) : (
        <>
          <p className="section-lede">
            The agents weren't sure about these. Your answers are saved and apply on the next run, so nobody gets
            asked the same thing twice.
          </p>
          <ul className="queue__list">
            {items.map((item) => (
              <Entry key={keyOf(item)} item={item} saved={answered[keyOf(item)]} onAnswer={answer} />
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

interface EntryProps {
  item: Item;
  saved?: string;
  onAnswer: (item: Item, decision: NewDecision, label: string) => void;
}

function Entry({ item, saved, onAnswer }: EntryProps) {
  const [note, setNote] = useState("");
  const noteId = useId();
  const title = item.kind === "outcome" ? item.outcome_id : `${item.outcome_id} and ${item.assignment_id}`;
  const choices =
    item.kind === "outcome"
      ? (["keep", "drop"] as const).map((value) => ({ value, label: outcomeChoiceLabel[value] }))
      : (["direct", "partial", "not_aligned"] as const).map((value) => ({ value, label: pairChoiceLabel[value] }));

  function decide(value: string, label: string) {
    const decision: NewDecision = { course_id: item.course_id, outcome_id: item.outcome_id, decision: value, note };
    if (item.assignment_id) decision.assignment_id = item.assignment_id;
    onAnswer(item, decision, label);
  }

  return (
    <li className={saved ? "queue-item queue-item--saved" : "queue-item"}>
      <div className="queue-item__head">
        <h3 className="queue-item__title">{title}</h3>
        <p className="queue-item__course">{item.course_id}</p>
      </div>
      <p className="queue-item__question">
        {item.kind === "outcome"
          ? "Is this a learning outcome the syllabus states?"
          : `Does “${item.assignment_title}” assess this outcome?`}
      </p>
      <p className="syllabus queue-item__outcome">{item.outcome_text}</p>
      {item.quote !== item.outcome_text && (
        <p className="syllabus queue-item__quote">From the syllabus: “{item.quote}”</p>
      )}
      <p className="queue-item__why">{explain(item)}</p>
      {saved ? (
        <p className="queue-item__saved">Saved: {saved}. It applies on the next run.</p>
      ) : (
        <div className="queue-item__answer">
          <div className="field field--inline">
            <label htmlFor={noteId}>Note (optional)</label>
            <input id={noteId} value={note} onChange={(event) => setNote(event.target.value)} />
          </div>
          <div className="answers" role="group" aria-label={`Your answer for ${title}`}>
            {choices.map((choice) => (
              <button key={choice.value} type="button" className="button" onClick={() => decide(choice.value, choice.label)}>
                {choice.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </li>
  );
}

function explain(item: Item): string {
  if (item.kind === "outcome") {
    const sure = item.outcome ? `${percent(item.outcome.confidence)} sure` : "unsure";
    return item.outcome?.status === "held"
      ? `Agent 1 was only ${sure} this is a stated outcome, so it's held for you.`
      : `Agent 1 was ${sure}, so it passed but asked for a second look.`;
  }
  const agent = item.agent;
  if (!agent) return "Agent 2 didn't give an answer that passed the checks.";
  if (agent.confidence === 0 && agent.reason.startsWith("No reply from the model")) {
    return "The model didn't answer this pair, even after retrying. Run again later and it gets asked again, or decide it yourself.";
  }
  if (agent.confidence === 0 && agent.reason.startsWith("Two independent judgments disagreed")) {
    return `${agent.reason} So it's waiting for you.`;
  }
  if (agent.confidence === 0) return `Agent 2's replies kept failing the checks. ${agent.reason}`;
  const held = item.why.startsWith("held");
  return (
    `Agent 2 said it ${agentVerdictText[agent.verdict] ?? agent.verdict} and was ${percent(agent.confidence)} sure, ` +
    `so ${held ? "it's held for you" : "it passed but asked for a second look"}. Its reason: ${agent.reason}`
  );
}

// The details behind one cell of the map, and a way to correct it. Uses the
// native <dialog>: the browser traps focus inside and closes it on Escape;
// focus returns to the cell that opened it.
import { useEffect, useId, useRef, useState } from "react";
import type { FormEvent } from "react";
import { pairChoiceLabel, percent, triageText, verdictText, whoDecided } from "../format";
import type { NewDecision, PairChoice, Report } from "../types";

export interface OpenPair {
  courseId: string;
  outcomeId: string;
  assignmentId: string;
}

interface Props {
  report: Report;
  open: OpenPair;
  onClose: () => void;
  onSave: (decision: NewDecision) => Promise<boolean>;
}

const CHOICES: PairChoice[] = ["direct", "partial", "not_aligned"];

export function PairDialog({ report, open, onClose, onSave }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const [choice, setChoice] = useState<PairChoice | null>(null);
  const [note, setNote] = useState("");
  const [teaching, setTeaching] = useState(false);
  const [term, setTerm] = useState("");
  const [topic, setTopic] = useState(report.topics[0] ?? "");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const node = dialog.current;
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    node?.showModal();
    return () => {
      node?.close();
      opener?.focus();
    };
  }, []);

  const course = report.courses.find((c) => c.course_id === open.courseId);
  const outcome = course?.outcomes.find((o) => o.id === open.outcomeId);
  const assignment = course?.assignments.find((a) => a.id === open.assignmentId);
  const pair = course?.pairs.find((p) => p.outcome_id === open.outcomeId && p.assignment_id === open.assignmentId);
  if (!course || !outcome || !assignment || !pair) return null;

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!choice) return;
    setSaving(true);
    const saved = await onSave({
      course_id: open.courseId,
      outcome_id: open.outcomeId,
      assignment_id: open.assignmentId,
      decision: choice,
      note: note.trim(),
      teach: teaching && term.trim() ? [{ topic, term: term.trim() }] : [],
    });
    setSaving(false);
    if (saved) onClose();
  }

  const sure = pair.confidence === null ? "" : `, ${percent(pair.confidence)} sure`;
  return (
    <dialog ref={dialog} className="pair-dialog" aria-labelledby={titleId} onClose={onClose}>
      <div className="pair-dialog__body">
        <h2 id={titleId} className="pair-dialog__title">
          {outcome.id} and {assignment.id}
        </h2>
        <dl className="facts">
          <dt>Outcome</dt>
          <dd className="syllabus">{outcome.text}</dd>
          {outcome.quote !== outcome.text && (
            <>
              <dt>Syllabus wording</dt>
              <dd className="syllabus">“{outcome.quote}”</dd>
            </>
          )}
          <dt>Assignment</dt>
          <dd>
            {assignment.title} ({assignment.type}, week {assignment.week})
          </dd>
          <dt>Verdict</dt>
          <dd>
            <strong className="facts__verdict">{capitalize(verdictText(pair))}</strong>, decided by{" "}
            {whoDecided[pair.decided_by]}
            {sure}
            {pair.flagged && ". Flagged for a second look"}
          </dd>
          <dt>Why</dt>
          <dd>{capitalize(pair.reason)}</dd>
          {pair.evidence && (
            <>
              <dt>In the assignment</dt>
              <dd>“{pair.evidence}”</dd>
            </>
          )}
          {pair.depth && (
            <>
              <dt>Depth</dt>
              <dd>Asks students to {pair.depth}</dd>
            </>
          )}
          {pair.triage && (
            <>
              <dt>First pass</dt>
              <dd>{capitalize(triageText[pair.triage] ?? pair.triage)}</dd>
            </>
          )}
        </dl>

        <form onSubmit={save} className="correction">
          <fieldset className="correction__choices">
            <legend>Correct this pair</legend>
            {CHOICES.map((value) => (
              <label key={value} className="choice">
                <input
                  type="radio"
                  name="choice"
                  value={value}
                  checked={choice === value}
                  onChange={() => setChoice(value)}
                />
                {pairChoiceLabel[value]}
              </label>
            ))}
          </fieldset>
          <label className="field">
            Why (optional)
            <textarea rows={2} value={note} onChange={(event) => setNote(event.target.value)} />
          </label>
          <label className="check">
            <input type="checkbox" checked={teaching} onChange={(event) => setTeaching(event.target.checked)} />
            Also teach the rules a word, so this kind of pair isn't missed in any course
          </label>
          {teaching && (
            <div className="correction__teach">
              <label className="field">
                Word
                <input value={term} onChange={(event) => setTerm(event.target.value)} />
              </label>
              <label className="field">
                Means the topic
                <select value={topic} onChange={(event) => setTopic(event.target.value)}>
                  {report.topics.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}
          <div className="correction__actions">
            <button type="submit" className="button button--primary" disabled={!choice || saving}>
              {saving ? "Saving…" : "Save correction"}
            </button>
            <button type="button" className="button" onClick={onClose}>
              Close
            </button>
          </div>
        </form>
      </div>
    </dialog>
  );
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

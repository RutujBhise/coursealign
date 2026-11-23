// Everything a person has decided so far, with a way to take any of it back.
import { decisionLabel, shorten } from "../format";
import type { SavedDecision } from "../types";

interface Props {
  decisions: SavedDecision[];
  onRemove: (index: number) => void;
}

export function SavedDecisions({ decisions, onRemove }: Props) {
  return (
    <section className="decisions" aria-labelledby="decisions-title">
      <h2 id="decisions-title" className="section-title">
        Your decisions
      </h2>
      {decisions.length === 0 ? (
        <p className="section-lede">
          No decisions yet. Answer an item that's waiting for you, or open any cell in the map to correct it.
        </p>
      ) : (
        <>
          <p className="section-lede">Every run applies these. A later decision about the same thing wins.</p>
          <div className="table-scroll" role="region" aria-label="Decisions table" tabIndex={0}>
            <table className="decisions__table">
              <caption className="visually-hidden">Saved decisions, oldest first</caption>
              <thead>
                <tr>
                  <th scope="col">Decision</th>
                  <th scope="col">About</th>
                  <th scope="col">Note</th>
                  <th scope="col">Saved</th>
                  <th scope="col">
                    <span className="visually-hidden">Undo</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {decisions.map((d) => (
                  <tr key={d.index}>
                    <td className="decisions__decision">{decisionLabel[d.decision] ?? d.decision}</td>
                    <td>
                      {d.course_id} {d.assignment_id ? `${d.assignment_id}: ` : ""}
                      <span className="syllabus">“{shorten(d.quote, 80)}”</span>
                      {d.teach.map((t) => (
                        <span key={t.term} className="decisions__taught">
                          Taught the rules that “{t.term}” means {t.topic}.
                        </span>
                      ))}
                    </td>
                    <td>{d.note || <span className="muted">No note</span>}</td>
                    <td className="decisions__when">
                      {d.at}, {d.by}
                    </td>
                    <td>
                      <button
                        type="button"
                        className="button button--small"
                        onClick={() => onRemove(d.index)}
                        aria-label={`Undo decision ${d.index + 1}: ${decisionLabel[d.decision] ?? d.decision}`}
                      >
                        Undo
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

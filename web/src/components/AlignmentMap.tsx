// The alignment map: outcomes down the side, assignments across the top, and
// one bubble per pair, like an answer sheet. Machine marks are graphite, a
// person's marks are ink, and gaps get a mark in the margin.
import { shorten, verdictText, whoDecided } from "../format";
import type { CourseReport, Pair } from "../types";

interface Props {
  course: CourseReport;
  onOpen: (outcomeId: string, assignmentId: string) => void;
}

type RowMark = { label: string; tone: "pen" | "amber" };

export function AlignmentMap({ course, onOpen }: Props) {
  const rows = course.outcomes.filter((o) => o.status === "passed" || o.status === "flagged");
  const pairs = new Map(course.pairs.map((p) => [`${p.outcome_id}|${p.assignment_id}`, p]));

  const rowMarks = new Map<string, RowMark>();
  for (const gap of course.gaps) {
    if (!gap.outcome_id || rowMarks.has(gap.outcome_id)) continue;
    if (gap.type === "untested_outcome") rowMarks.set(gap.outcome_id, { label: "Untested", tone: "pen" });
    if (gap.type === "blooms_mismatch") rowMarks.set(gap.outcome_id, { label: "Too shallow", tone: "amber" });
    if (gap.type === "weak_coverage") rowMarks.set(gap.outcome_id, { label: "Only in parts", tone: "amber" });
  }
  const orphans = new Set(course.gaps.filter((g) => g.type === "orphan_assessment").map((g) => g.assignment_id));
  const titleId = `map-${course.course_id}`;

  return (
    <section className="map" aria-labelledby={titleId}>
      <h3 id={titleId} className="subsection-title">Alignment map</h3>
      <Legend />
      {rows.length === 0 || course.assignments.length === 0 ? (
        <p className="map__empty">
          There's nothing to map yet: this course needs at least one outcome and one assignment.
        </p>
      ) : (
        <div className="map__scroll" role="region" aria-label={`Alignment map for ${course.course_id}`} tabIndex={0}>
          <table className="map__table">
            <caption className="visually-hidden">
              Which assignments assess each outcome in {course.course_id}. Each cell opens its details.
            </caption>
            <thead>
              <tr>
                <th scope="col" className="map__corner">
                  Outcome
                </th>
                {course.assignments.map((a) => (
                  <th scope="col" key={a.id} className="map__col">
                    <span className="map__col-id">{a.id}</span>
                    <span className="map__col-title">{shorten(a.title, 30)}</span>
                    {orphans.has(a.id) && <span className="mark mark--note">No outcome</span>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((outcome) => {
                const mark = rowMarks.get(outcome.id);
                return (
                  <tr key={outcome.id}>
                    <th scope="row" className={mark?.tone === "pen" ? "map__row map__row--pen" : "map__row"}>
                      <span className="map__row-id">{outcome.id}</span>
                      <span className="map__row-text syllabus">{outcome.text}</span>
                      {mark && <span className={`mark mark--${mark.tone}`}>{mark.label}</span>}
                    </th>
                    {course.assignments.map((a) => {
                      const pair = pairs.get(`${outcome.id}|${a.id}`);
                      return (
                        <td key={a.id} className="map__cell">
                          {pair && <Bubble pair={pair} onOpen={onOpen} />}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function bubbleState(pair: Pair): "direct" | "partial" | "none" | "waiting" {
  if (pair.aligned === null) return "waiting";
  if (!pair.aligned) return "none";
  return pair.kind === "partial" ? "partial" : "direct";
}

function Bubble({ pair, onOpen }: { pair: Pair; onOpen: Props["onOpen"] }) {
  const state = bubbleState(pair);
  const flagged = pair.flagged ? ", flagged for a second look" : "";
  const label =
    `${pair.outcome_id} and ${pair.assignment_id}: ${verdictText(pair)}, ` +
    `decided by ${whoDecided[pair.decided_by]}${flagged}. Show details`;
  return (
    <button
      type="button"
      className={`bubble-button bubble-button--${pair.decided_by}`}
      aria-label={label}
      onClick={() => onOpen(pair.outcome_id, pair.assignment_id)}
    >
      <span className={`bubble bubble--${state}`} aria-hidden="true">
        {state === "waiting" ? "?" : ""}
      </span>
      {pair.decided_by !== "rules" && (
        <span className="bubble-button__by" aria-hidden="true">
          {pair.decided_by === "human" ? "P" : "A"}
        </span>
      )}
      {pair.flagged && <span className="bubble-button__flag" aria-hidden="true" />}
    </button>
  );
}

function Legend() {
  return (
    <div className="legend">
      <ul className="legend__row" aria-label="What each mark means">
        <li>
          <span className="bubble bubble--direct" aria-hidden="true" /> Assesses it
        </li>
        <li>
          <span className="bubble bubble--partial" aria-hidden="true" /> Assesses part of it
        </li>
        <li>
          <span className="bubble bubble--none" aria-hidden="true" /> Doesn't assess it
        </li>
        <li>
          <span className="bubble bubble--waiting" aria-hidden="true">
            ?
          </span>{" "}
          Waiting for you
        </li>
      </ul>
      <ul className="legend__row" aria-label="Who decided">
        <li className="legend__lead">Decided by</li>
        <li>no letter: the rules</li>
        <li>
          <span className="legend__letter">A</span> Agent 2
        </li>
        <li>
          <span className="legend__letter legend__letter--human">P</span> a person, in ink
        </li>
      </ul>
    </div>
  );
}

// How the run measured against the hand labels in data/gold.
import type { Scores as ScoresData } from "../types";

export function Scores({ scores, standIns }: { scores: ScoresData; standIns: boolean }) {
  const rows: [string, string][] = [
    ["Outcomes found", `${scores.outcomes.found} of ${scores.outcomes.labeled}`],
    ["Rule decisions right", `${scores.rules.right} of ${scores.rules.decided}`],
    ["Agent 2 verdicts right", `${scores.agent2.right} of ${scores.agent2.passed}`],
    ["Every decided pair right", `${scores.pairs.right} of ${scores.pairs.decided}`],
    ["Untested outcomes found", `${scores.gaps.untested_found} of ${scores.gaps.untested_expected}`],
    ["Orphan assignments found", `${scores.gaps.orphans_found} of ${scores.gaps.orphans_expected}`],
  ];
  return (
    <section className="scores-section" aria-labelledby="scores-title">
      <h2 id="scores-title" className="section-title">
        Measured against the hand labels
      </h2>
      <p className="section-lede">
        Compared with data/gold/alignment_labels.json for {scores.courses.join(", ")}.
        {standIns && " The agents in this run are stand-ins, so these numbers test the code, not a model."}
      </p>
      <dl className="scores">
        {rows.map(([label, value]) => (
          <div key={label} className="scores__item">
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

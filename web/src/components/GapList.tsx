// The gap report for one course, most serious first. It sits beside the map
// like notes in a margin.
import { gapLabel, severityLabel } from "../format";
import type { CourseReport, Gap } from "../types";

export function GapList({ course }: { course: CourseReport }) {
  const titleId = `gaps-${course.course_id}`;
  return (
    <section className="gaps" aria-labelledby={titleId}>
      <h3 id={titleId} className="subsection-title">
        Gaps
      </h3>
      {course.gaps.length === 0 ? (
        <p className="gaps__none">No gaps: every outcome is assessed, and every assignment assesses an outcome.</p>
      ) : (
        <ol className="gaps__list">
          {course.gaps.map((gap) => (
            <li key={`${gap.type}-${gap.outcome_id ?? gap.assignment_id}`} className={`gap gap--${gap.severity}`}>
              <p className="gap__head">
                <span className="gap__severity">{severityLabel[gap.severity]}:</span> {gapLabel[gap.type] ?? gap.type}
              </p>
              <Target gap={gap} course={course} />
              <p className="gap__about">{gap.description}</p>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function Target({ gap, course }: { gap: Gap; course: CourseReport }) {
  if (gap.outcome_id) {
    const outcome = course.outcomes.find((o) => o.id === gap.outcome_id);
    return (
      <p className="gap__target">
        <span className="gap__id">{gap.outcome_id}</span> <span className="syllabus">{outcome?.text}</span>
      </p>
    );
  }
  const assignment = course.assignments.find((a) => a.id === gap.assignment_id);
  return (
    <p className="gap__target">
      <span className="gap__id">{gap.assignment_id}</span> {assignment?.title}
    </p>
  );
}

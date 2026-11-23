import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "./api";
import { AlignmentMap } from "./components/AlignmentMap";
import { GapList } from "./components/GapList";
import { PairDialog, type OpenPair } from "./components/PairDialog";
import { ReviewQueue } from "./components/ReviewQueue";
import { SavedDecisions } from "./components/SavedDecisions";
import { Scores } from "./components/Scores";
import { Thresholds } from "./components/Thresholds";
import { agentsLabel, courseSentence, criticalNote, summarySentence, usesStandIns, waitingSentence, when } from "./format";
import type { CourseReport, NewDecision, Report, SavedDecision, Settings } from "./types";

type Message = { tone: "status" | "error"; text: string };

export function App() {
  const [report, setReport] = useState<Report | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [running, setRunning] = useState(false);
  const [decisions, setDecisions] = useState<SavedDecision[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [changedSinceRun, setChangedSinceRun] = useState(false);
  const [courseId, setCourseId] = useState<string | null>(null);
  const [openPair, setOpenPair] = useState<OpenPair | null>(null);
  const [message, setMessage] = useState<Message | null>(null);

  const showError = useCallback((error: unknown) => {
    setMessage({ tone: "error", text: error instanceof Error ? error.message : "Something went wrong." });
  }, []);

  useEffect(() => {
    const latest = api.latest().catch((error: unknown) => {
      if (error instanceof ApiError && error.status === 404) return null; // nothing has run yet
      throw error;
    });
    Promise.all([latest, api.decisions(), api.settings()])
      .then(([last, saved, knobs]) => {
        setReport(last);
        setDecisions(saved);
        setSettings(knobs);
      })
      .catch(showError)
      .finally(() => setLoaded(true));
  }, [showError]);

  async function run() {
    setRunning(true);
    setMessage({ tone: "status", text: "Running the pipeline…" });
    try {
      const fresh = await api.run();
      setReport(fresh);
      setChangedSinceRun(false);
      setMessage({ tone: "status", text: `Run finished. ${waitingSentence(fresh.totals.waiting_for_a_person)}` });
    } catch (error) {
      showError(error);
    } finally {
      setRunning(false);
    }
  }

  async function saveDecision(decision: NewDecision): Promise<boolean> {
    try {
      await api.saveDecision(decision);
      setDecisions(await api.decisions());
      setChangedSinceRun(true);
      setMessage({ tone: "status", text: "Decision saved. It applies on the next run." });
      return true;
    } catch (error) {
      showError(error);
      return false;
    }
  }

  async function removeDecision(index: number) {
    try {
      setDecisions(await api.removeDecision(index));
      setChangedSinceRun(true);
      setMessage({ tone: "status", text: "Decision removed. The next run won't use it." });
    } catch (error) {
      showError(error);
    }
  }

  async function saveSettings(changes: Record<string, number | null>): Promise<boolean> {
    try {
      setSettings(await api.saveSettings(changes));
      setChangedSinceRun(true);
      setMessage({ tone: "status", text: "Thresholds saved. They apply on the next run." });
      return true;
    } catch (error) {
      showError(error);
      return false;
    }
  }

  const course = report?.courses.find((c) => c.course_id === courseId) ?? report?.courses[0] ?? null;
  const stale = report !== null && (changedSinceRun || differsFromRun(report, decisions, settings));

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="masthead">
        <div className="masthead__inner">
          <div className="brand">
            <h1 className="brand__name">CourseAlign</h1>
            <p className="brand__line">Does each course assess what its syllabus says it teaches?</p>
          </div>
          {report && (
            <div className="run">
              <p className="run__meta">
                Last run {when(report.ran_at)}
                <br />
                {agentsLabel(report)}
              </p>
              <button type="button" className="button button--primary" onClick={run} disabled={running}>
                {running ? "Running…" : "Run again"}
              </button>
            </div>
          )}
        </div>
        <div className="masthead__messages">
          <p className="announcer" role="status">
            {message?.tone === "status" ? message.text : ""}
          </p>
          {message?.tone === "error" && (
            <p className="alert" role="alert">
              {message.text}
            </p>
          )}
        </div>
      </header>

      <main id="main" className="page" aria-busy={running}>
        {!loaded && <p className="loading">Loading the last run…</p>}

        {loaded && !report && (
          <section className="empty" aria-labelledby="empty-title">
            <h2 id="empty-title" className="empty__title">
              Nothing has run yet
            </h2>
            <p className="empty__text">
              Run the pipeline to map every outcome in every syllabus to the assignments that assess it. The rules
              settle the clear cases; the agents read the rest and hand you anything they're unsure about.
            </p>
            <button type="button" className="button button--primary" onClick={run} disabled={running}>
              {running ? "Running…" : "Run the pipeline"}
            </button>
          </section>
        )}

        {report && course && (
          <>
            <p className="summary">{summarySentence(report)}</p>

            {stale && (
              <div className="stale">
                <p>You've changed decisions or thresholds since this run. They apply on the next one.</p>
                <button type="button" className="button" onClick={run} disabled={running}>
                  Run again to apply
                </button>
              </div>
            )}

            <div className="course-switch" role="group" aria-label="Choose a course">
              {report.courses.map((c) => (
                <button
                  key={c.course_id}
                  type="button"
                  className="course-switch__button"
                  aria-pressed={c.course_id === course.course_id}
                  onClick={() => setCourseId(c.course_id)}
                >
                  <span className="course-switch__id">{c.course_id}</span>
                  <span className="course-switch__note">{criticalNote(c)}</span>
                </button>
              ))}
            </div>

            <CourseSection
              course={course}
              onOpen={(outcomeId, assignmentId) => setOpenPair({ courseId: course.course_id, outcomeId, assignmentId })}
            />
            <ReviewQueue report={report} onDecide={saveDecision} />
            <SavedDecisions decisions={decisions} onRemove={removeDecision} />
            {settings && <Thresholds settings={settings} onSave={saveSettings} />}
            {report.scores && <Scores scores={report.scores} standIns={usesStandIns(report)} />}
          </>
        )}
      </main>

      {report && openPair && (
        <PairDialog report={report} open={openPair} onClose={() => setOpenPair(null)} onSave={saveDecision} />
      )}
    </>
  );
}

function CourseSection({ course, onOpen }: { course: CourseReport; onOpen: (o: string, a: string) => void }) {
  const offMap = course.outcomes.filter((o) => o.status === "held" || o.status === "not_assessed");
  return (
    <section className="course" aria-labelledby="course-title">
      <h2 id="course-title" className="course__title">
        {course.title}
      </h2>
      <p className="course__summary">{courseSentence(course)}</p>
      <div className="course__body">
        <AlignmentMap course={course} onOpen={onOpen} />
        <GapList course={course} />
      </div>
      {offMap.length > 0 && (
        <section className="off-map" aria-labelledby="off-map-title">
          <h3 id="off-map-title" className="subsection-title">
            Outcomes not on the map
          </h3>
          <ul className="off-map__list">
            {offMap.map((o) => (
              <li key={o.id}>
                <span className="off-map__id">{o.id}</span> <span className="syllabus">{o.text}</span>
                <span className="off-map__why">
                  {o.status === "held"
                    ? "Agent 1 wasn't sure this is a stated outcome, so it's waiting for you."
                    : "The syllabus says this won't be formally assessed."}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </section>
  );
}

// True when what's saved now isn't what the last run used: decisions made in
// another tab or from the command line, or thresholds changed since.
function differsFromRun(report: Report, decisions: SavedDecision[], settings: Settings | null): boolean {
  const applied = report.courses.reduce((n, c) => n + c.review_log.length, 0);
  const moved = settings !== null && Object.entries(settings).some(([name, s]) => report.settings[name] !== s.value);
  return applied !== decisions.length || moved;
}

// The knobs that decide what the rules settle alone and when the agents hand
// a call to a person.
import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import type { Settings } from "../types";

const NAMES: Record<string, string> = {
  auto_match_min_topics: "Shared topics a rules match needs",
  common_topic_share: "When a topic counts as background",
  agent1_accept_at: "Agent 1 decides alone at",
  agent1_review_at: "Agent 1 hands over to you below",
  agent2_accept_at: "Agent 2 decides alone at",
  agent2_review_at: "Agent 2 hands over to you below",
};

interface Props {
  settings: Settings;
  onSave: (changes: Record<string, number | null>) => Promise<boolean>;
}

const toDraft = (settings: Settings) =>
  Object.fromEntries(Object.entries(settings).map(([name, s]) => [name, String(s.value)]));

export function Thresholds({ settings, onSave }: Props) {
  const [draft, setDraft] = useState<Record<string, string>>(() => toDraft(settings));
  useEffect(() => setDraft(toDraft(settings)), [settings]);

  const changes = Object.entries(draft).filter(
    ([name, value]) => value.trim() !== "" && Number(value) !== settings[name]?.value,
  );

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await onSave(Object.fromEntries(changes.map(([name, value]) => [name, Number(value)])));
  }

  return (
    <section className="thresholds-section" aria-labelledby="thresholds-title">
      <h2 id="thresholds-title" className="section-title">
        Thresholds
      </h2>
      <p className="section-lede">
        These decide what the rules settle on their own and when the agents hand a call to you. Changes apply on the
        next run.
      </p>
      <form className="thresholds" onSubmit={save}>
        {Object.entries(settings).map(([name, setting]) => {
          const inputId = `threshold-${name}`;
          return (
            <div key={name} className="threshold">
              <label htmlFor={inputId} className="threshold__label">
                {NAMES[name] ?? name}
              </label>
              <p id={`${inputId}-about`} className="threshold__about">
                {setting.about}
              </p>
              <div className="threshold__control">
                <input
                  id={inputId}
                  type="number"
                  inputMode="decimal"
                  min={setting.min}
                  max={setting.max}
                  step={setting.step}
                  value={draft[name] ?? ""}
                  aria-describedby={`${inputId}-about`}
                  onChange={(event) => setDraft({ ...draft, [name]: event.target.value })}
                />
                {setting.changed && (
                  <button
                    type="button"
                    className="button button--small"
                    aria-label={`Reset “${NAMES[name] ?? name}” to its default`}
                    onClick={() => void onSave({ [name]: null })}
                  >
                    Reset to default
                  </button>
                )}
              </div>
            </div>
          );
        })}
        <div className="thresholds__save">
          <button type="submit" className="button button--primary" disabled={changes.length === 0}>
            Save thresholds
          </button>
        </div>
      </form>
    </section>
  );
}

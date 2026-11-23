// The one place the dashboard talks to the API.
import type { NewDecision, Report, SavedDecision, Settings } from "./types";

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, { ...init, headers: { "Content-Type": "application/json" } });
  if (!response.ok) {
    let message = `The server answered ${response.status}.`;
    try {
      const detail: unknown = ((await response.json()) as { detail?: unknown }).detail;
      if (typeof detail === "string") {
        message = detail;
      } else if (Array.isArray(detail)) {
        // FastAPI's validation errors: a list of {msg} objects
        const parts = detail.map((d) => String((d as { msg?: unknown }).msg ?? "")).filter(Boolean);
        if (parts.length > 0) message = parts.join(" ");
      }
    } catch {
      // The body wasn't JSON; keep the generic message.
    }
    throw new ApiError(response.status, message);
  }
  return (await response.json()) as T;
}

export const api = {
  latest: () => request<Report>("/api/runs/latest"),
  run: () => request<Report>("/api/runs", { method: "POST" }),
  decisions: () => request<SavedDecision[]>("/api/reviews"),
  saveDecision: (decision: NewDecision) =>
    request<SavedDecision>("/api/reviews", { method: "POST", body: JSON.stringify(decision) }),
  removeDecision: (index: number) => request<SavedDecision[]>(`/api/reviews/${index}`, { method: "DELETE" }),
  settings: () => request<Settings>("/api/settings"),
  saveSettings: (changes: Record<string, number | null>) =>
    request<Settings>("/api/settings", { method: "PUT", body: JSON.stringify(changes) }),
};

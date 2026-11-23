// End-to-end: the dashboard in a real browser, against the real API with
// stand-in agents, so every number below is the same on every run.
import { expect, test, type Page } from "@playwright/test";

test.beforeEach(async ({ page, request }) => {
  await request.post("/api/test/reset"); // fresh data, no run yet
  await page.goto("/");
});

async function runPipeline(page: Page) {
  await page.getByRole("button", { name: /^(Run the pipeline|Run again|Run again to apply)$/ }).first().click();
  await expect(page.getByRole("status")).toContainText("Run finished");
}

async function openCourse(page: Page, courseId: string) {
  // Inside the course switcher: the map's cells are buttons too, named "CS4530-O1 and IA1: ..."
  await page.getByRole("group", { name: "Choose a course" }).getByRole("button", { name: new RegExp(`^${courseId}`) }).click();
}

test("an empty dashboard runs the pipeline and sums it up", async ({ page }) => {
  await expect(page.getByRole("heading", { name: "Nothing has run yet" })).toBeVisible();
  await runPipeline(page);
  await expect(page.getByText(/the agents made 50 requests instead of 171 \(71% fewer\)/)).toBeVisible();
  await expect(page.getByRole("status")).toContainText("4 items are waiting for you");
});

test("the map marks the untested outcomes in the margin", async ({ page }) => {
  await runPipeline(page);
  await openCourse(page, "CS4530");
  await expect(page.getByRole("heading", { level: 2, name: "CS4530 Fundamentals of Software Engineering" })).toBeVisible();
  const gaps = page.getByRole("region", { name: "Gaps" });
  await expect(gaps.getByRole("listitem")).toHaveCount(3);
  await expect(gaps).toContainText("Apply test-driven development (TDD)");
  await expect(gaps).toContainText("Monitor application health");
  await expect(page.getByRole("rowheader", { name: /CS4530-O8/ })).toContainText("Untested");
});

test("a cell's details open and close from the keyboard", async ({ page }) => {
  await runPipeline(page);
  await openCourse(page, "CS4530");
  const cell = page.getByRole("button", { name: /^CS4530-O6 and IA1: doesn't assess it, decided by the rules/ });
  await cell.focus();
  await page.keyboard.press("Enter");
  const dialog = page.getByRole("dialog", { name: "CS4530-O6 and IA1" });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("Ruled out by the rules");
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(cell).toBeFocused();
});

test("a correction that teaches the rules a word sticks after the next run", async ({ page }) => {
  await runPipeline(page);
  await openCourse(page, "CS4530");
  await page.getByRole("button", { name: /^CS4530-O6 and IA1:/ }).click();
  const dialog = page.getByRole("dialog");
  await dialog.getByRole("radio", { name: "Assesses part of it" }).check();
  await dialog.getByLabel("Why (optional)").fill("The pattern justification explains a design decision.");
  await dialog.getByRole("checkbox", { name: /teach the rules a word/ }).check();
  await dialog.getByRole("textbox", { name: "Word", exact: true }).fill("justification");
  await dialog.getByLabel("Means the topic").selectOption("communication");
  await dialog.getByRole("button", { name: "Save correction" }).click();
  await expect(dialog).toBeHidden();
  await expect(page.getByRole("status")).toContainText("Decision saved");

  await runPipeline(page);
  await expect(page.getByRole("button", { name: /^CS4530-O6 and IA1: assesses part of it, decided by a person/ })).toBeVisible();
  await expect(page.getByRole("region", { name: "Your decisions" })).toContainText("“justification” means communication");
});

test("answering everything that's waiting empties the queue on the next run", async ({ page }) => {
  await runPipeline(page);
  const queue = page.getByRole("region", { name: "Waiting for you" });
  async function answer(title: string, choice: string) {
    const item = queue.getByRole("listitem").filter({ has: page.getByRole("heading", { name: title, exact: true }) });
    await item.getByRole("button", { name: choice, exact: true }).click();
    await expect(item).toContainText(`Saved: ${choice}`);
  }
  await answer("CS3200-O4", "Keep it");
  await answer("CS3200-O7", "Drop it");
  await answer("CS4550-O2 and A4", "Doesn't assess it");
  await answer("CS4550-O2 and A5", "Doesn't assess it");
  await runPipeline(page);
  await expect(page.getByRole("status")).toContainText("Nothing is waiting for you");
  await expect(queue).toContainText("Nothing is waiting for you");
});

test("a stricter threshold hands more calls to a person", async ({ page }) => {
  await runPipeline(page);
  await page.getByLabel("Agent 2 decides alone at").fill("0.95");
  await page.getByRole("button", { name: "Save thresholds" }).click();
  await expect(page.getByRole("status")).toContainText("Thresholds saved");
  await runPipeline(page);
  await expect(page.getByRole("status")).toContainText("48 items are waiting for you");
});

test("a saved decision can be undone", async ({ page }) => {
  await runPipeline(page);
  const queue = page.getByRole("region", { name: "Waiting for you" });
  await queue.getByRole("listitem").first().getByRole("button", { name: "Keep it" }).click();
  const decisions = page.getByRole("region", { name: "Your decisions" });
  await expect(decisions.getByRole("row")).toHaveCount(2); // the header and one decision
  await decisions.getByRole("button", { name: /^Undo decision 1/ }).click();
  await expect(decisions).toContainText("No decisions yet");
});

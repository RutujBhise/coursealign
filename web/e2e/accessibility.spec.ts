// WCAG 2.1 AA, checked by axe-core in a real browser, in light and dark mode:
// the empty page, every course's full report, and the pair dialog.
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const WCAG_21_AA = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"];

async function violations(page: Page) {
  return (await new AxeBuilder({ page }).withTags(WCAG_21_AA).analyze()).violations;
}

test.beforeEach(async ({ request }) => {
  await request.post("/api/test/reset");
});

for (const colorScheme of ["light", "dark"] as const) {
  test.describe(`${colorScheme} mode`, () => {
    test.use({ colorScheme });

    test("the empty dashboard", async ({ page }) => {
      await page.goto("/");
      await expect(page.getByRole("heading", { name: "Nothing has run yet" })).toBeVisible();
      expect(await violations(page)).toEqual([]);
    });

    test("every course's report", async ({ page }) => {
      await page.goto("/");
      await page.getByRole("button", { name: "Run the pipeline" }).click();
      await expect(page.getByRole("status")).toContainText("Run finished");
      for (const course of ["CS3200", "CS4530", "CS4550"]) {
        // Inside the course switcher: the map's cells are buttons too, named "CS3200-O1 and HW1: ..."
        await page.getByRole("group", { name: "Choose a course" }).getByRole("button", { name: new RegExp(`^${course}`) }).click();
        expect(await violations(page)).toEqual([]);
      }
    });

    test("the pair dialog, with the teach-a-word fields open", async ({ page }) => {
      await page.goto("/");
      await page.getByRole("button", { name: "Run the pipeline" }).click();
      await expect(page.getByRole("status")).toContainText("Run finished");
      await page.getByRole("button", { name: /^CS3200-O1 and HW1:/ }).click();
      await page.getByRole("checkbox", { name: /teach the rules a word/ }).check();
      expect(await violations(page)).toEqual([]);
    });
  });
}

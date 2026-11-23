import { defineConfig, devices } from "@playwright/test";

const python = process.platform === "win32" ? "python" : "python3";
const port = 8123;

export default defineConfig({
  testDir: "./e2e",
  // One server, one throwaway workspace: each test resets it, so tests run one at a time.
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["html", { open: "never" }]] : [["list"]],
  use: { baseURL: `http://127.0.0.1:${port}`, trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    // Builds the dashboard, then serves it with stand-in agents on a throwaway copy of data/.
    // Set E2E_SERVER to point the tests at a different server command.
    command: process.env.E2E_SERVER ?? `npm run build && ${python} ../step6_api.py --workspace --port ${port}`,
    url: `http://127.0.0.1:${port}/api/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});

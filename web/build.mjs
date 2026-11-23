// Builds the dashboard into dist/, which the API serves at /.
//   node build.mjs           a production build
//   node build.mjs --watch   rebuilds on every save (reload the page to see changes)
import { build, context } from "esbuild";
import { copyFile, mkdir, rm } from "node:fs/promises";

const watch = process.argv.includes("--watch");
const options = {
  entryPoints: ["src/main.tsx"],
  entryNames: "app",                 // dist/app.js and dist/app.css
  outdir: "dist",
  bundle: true,
  format: "esm",
  target: ["es2022"],
  jsx: "automatic",
  minify: !watch,
  sourcemap: true,
  logLevel: "info",
  define: { "process.env.NODE_ENV": JSON.stringify(watch ? "development" : "production") },
};

await rm("dist", { recursive: true, force: true });
await mkdir("dist", { recursive: true });
await copyFile("index.html", "dist/index.html");
if (watch) {
  await (await context(options)).watch();
  console.log("Watching src/ for changes...");
} else {
  await build(options);
}

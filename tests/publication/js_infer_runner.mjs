#!/usr/bin/env node
/** Run the production browser decoder/inference implementation under Node. */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "../..");

if (process.argv.length !== 4) {
  process.stderr.write("usage: node js_infer_runner.mjs MODEL CASES.json\n");
  process.exit(2);
}

const modelPath = path.resolve(process.argv[2]);
const casesPath = path.resolve(process.argv[3]);
const rows = JSON.parse(fs.readFileSync(casesPath, "utf8"));
if (!Array.isArray(rows)) throw new Error("CASES.json must contain an array of feature rows");

// The repository intentionally has no package.json declaring ESM.  Import the
// production source through a data URL so Node parses its `export` declarations
// exactly as a browser module, without copying or reimplementing the decoder.
const inferSource = fs.readFileSync(path.join(REPO, "src/local_infer.js"), "utf8");
const inferUrl = `data:text/javascript;base64,${Buffer.from(inferSource).toString("base64")}`;
const infer = await import(inferUrl);
const bytes = new Uint8Array(fs.readFileSync(modelPath));
const parsed = infer.parseGl1fPackage(bytes);
const model = infer.decodeModel(parsed.modelBytes);
const predictions = rows.map((row) => (
  model.version === 1 ? [infer.predictQ(model, row)] : infer.predictMultiQ(model, row)
));
process.stdout.write(`${JSON.stringify({
  version: model.version,
  modelLength: infer.gl1fModelLength(bytes),
  hasFooter: parsed.hasFooter,
  predictions,
})}\n`);

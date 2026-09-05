#!/usr/bin/env node
/**
 * Execute the browser WebWorker trainer under Node.js.
 *
 * Input is a JSON document:
 * {
 *   "X": [[...], ...],
 *   "y": [...] | [[...], ...],
 *   "params": {...},
 *   "output": "/absolute/or/relative/model.gl1f"
 * }
 *
 * This adapter intentionally does not reimplement training. It installs the
 * smallest WebWorker-compatible `self` object needed by src/train_worker.js,
 * imports that production file, and sends it the same typed-array message used
 * by the browser UI.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "../..");

if (process.argv.length !== 3) {
  process.stderr.write("usage: node js_worker_runner.mjs CASE.json\n");
  process.exit(2);
}

const casePath = path.resolve(process.argv[2]);
const spec = JSON.parse(fs.readFileSync(casePath, "utf8"));
if (!Array.isArray(spec.X) || spec.X.length < 1 || !Array.isArray(spec.X[0])) {
  throw new Error("X must be a non-empty rectangular array");
}

const nRows = spec.X.length;
const nFeatures = spec.X[0].length;
if (nFeatures < 1 || spec.X.some((row) => !Array.isArray(row) || row.length !== nFeatures)) {
  throw new Error("X must be a non-empty rectangular array");
}

const flatX = Float32Array.from(spec.X.flat().map(Number));
let flatY;
if (Array.isArray(spec.y?.[0])) {
  if (spec.y.length !== nRows) throw new Error("X/y row count mismatch");
  flatY = Float32Array.from(spec.y.flat().map(Number));
} else {
  if (!Array.isArray(spec.y) || spec.y.length !== nRows) {
    throw new Error("X/y row count mismatch");
  }
  flatY = Float32Array.from(spec.y.map(Number));
}

let completed = false;
let failure = null;

globalThis.self = {
  onmessage: null,
  postMessage(message) {
    if (message?.type === "error") {
      failure = new Error(String(message.message || "worker training failed"));
      completed = true;
      return;
    }
    if (message?.type !== "done") return;
    const outPath = path.resolve(path.dirname(casePath), String(spec.output || "js.gl1f"));
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, Buffer.from(new Uint8Array(message.modelBytes)));
    if (spec.metaOutput) {
      const metaPath = path.resolve(path.dirname(casePath), String(spec.metaOutput));
      fs.mkdirSync(path.dirname(metaPath), { recursive: true });
      fs.writeFileSync(metaPath, `${JSON.stringify(message.meta, null, 2)}\n`);
    }
    completed = true;
  },
};

await import(pathToFileURL(path.join(REPO, "src/train_worker.js")).href);
if (typeof globalThis.self.onmessage !== "function") {
  throw new Error("production worker did not install self.onmessage");
}

await globalThis.self.onmessage({
  data: {
    type: "train",
    X: flatX.buffer,
    y: flatY.buffer,
    nRows,
    nFeatures,
    params: spec.params || {},
  },
});

if (failure) throw failure;
if (!completed) throw new Error("worker returned without a done/error message");

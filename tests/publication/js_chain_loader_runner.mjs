#!/usr/bin/env node
/** Exercise the production GL1C chain loader against a deterministic mock. */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import * as ethers from "ethers";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, "../..");

if (process.argv.length !== 3) {
  process.stderr.write("usage: node js_chain_loader_runner.mjs CASE.json\n");
  process.exit(2);
}

const casePath = path.resolve(process.argv[2]);
const spec = JSON.parse(fs.readFileSync(casePath, "utf8"));
if (!Array.isArray(spec.info) || spec.info.length !== 4) {
  throw new Error("CASE.json info must contain [tablePtr, chunkSize, numChunks, totalBytes]");
}
if (!spec.code || Array.isArray(spec.code) || typeof spec.code !== "object") {
  throw new Error("CASE.json code must be an address-to-bytecode object");
}
if (typeof spec.expectedHex !== "string") {
  throw new Error("CASE.json expectedHex must contain the committed model bytes");
}

globalThis.ethers = ethers;
const inferUrl = pathToFileURL(path.join(REPO, "src/local_infer.js")).href;
const infer = await import(inferUrl);

const code = new Map(
  Object.entries(spec.code).map(([address, value]) => [address.toLowerCase(), value]),
);
const provider = {
  async getCode(address) {
    const value = code.get(String(address).toLowerCase());
    if (value === undefined) throw new Error(`No mock code for ${address}`);
    return value;
  },
};
const registry = {
  async getModelBytesInfo() {
    return spec.info;
  },
};
const modelId = spec.modelId ?? ethers.keccak256(spec.expectedHex);
const logs = [];
const bytes = await infer.loadModelBytesFromChain({
  provider,
  store: null,
  registry,
  modelId,
  log(message) {
    logs.push(String(message));
  },
});

process.stdout.write(`${JSON.stringify({ hex: ethers.hexlify(bytes), modelId, logs })}\n`);

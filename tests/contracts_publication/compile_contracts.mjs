#!/usr/bin/env node

/**
 * Reproducible publication compile check for the GL1F Solidity contracts.
 *
 * The runtime currently requires the IR pipeline because the non-IR compiler
 * reports "stack too deep" in the multi-output transaction entry points.
 */

import { readFileSync } from "node:fs";
import solc from "solc";

const files = [
  "contracts/ForestRuntime.sol",
  "contracts/ModelStore.sol",
  "contracts/ModelRegistry.sol",
  "contracts/ModelNFT.sol",
  "contracts/ModelMarketplace.sol",
  "contracts/SimpleOwnable.sol",
  "contracts/Base64.sol",
];

const sources = Object.fromEntries(
  files.map((name) => [name, { content: readFileSync(name, "utf8") }]),
);

const input = {
  language: "Solidity",
  sources,
  settings: {
    optimizer: { enabled: true, runs: 200 },
    viaIR: true,
    evmVersion: "istanbul",
    outputSelection: {
      "*": {
        "*": ["evm.deployedBytecode.object"],
      },
    },
  },
};

const solcVersion = process.env.GL1F_SOLC_VERSION || "0.8.20";
const installedVersion = solc.version();
if (!installedVersion.startsWith(`${solcVersion}+`)) {
  throw new Error(
    `Installed solc-js is ${installedVersion}; expected lockfile version ${solcVersion}`,
  );
}

let output;
try {
  output = JSON.parse(solc.compile(JSON.stringify(input)));
} catch (error) {
  throw new Error(`Unable to compile/parse solc standard JSON: ${error.message}`);
}
const diagnostics = output.errors || [];

for (const diagnostic of diagnostics) {
  const stream = diagnostic.severity === "error" ? process.stderr : process.stdout;
  stream.write(`${diagnostic.severity}: ${diagnostic.formattedMessage}\n`);
}

const errors = diagnostics.filter((diagnostic) => diagnostic.severity === "error");
if (errors.length > 0) process.exitCode = 1;
else {
  const compiled = Object.entries(output.contracts || {})
    .flatMap(([sourceName, source]) => Object.entries(source)
      .map(([contractName, artifact]) => ({
        name: `${sourceName}:${contractName}`,
        runtimeBytes: (artifact.evm?.deployedBytecode?.object?.length || 0) / 2,
      })));
  process.stdout.write(`Compiled ${compiled.length} contracts/interfaces with solc ${installedVersion}, viaIR, optimizer=200, evmVersion=istanbul.\n`);
  for (const artifact of compiled.filter((item) => item.runtimeBytes > 0)) {
    process.stdout.write(`  ${artifact.name}: ${artifact.runtimeBytes} runtime bytes\n`);
  }
}

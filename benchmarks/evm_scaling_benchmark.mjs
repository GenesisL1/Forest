#!/usr/bin/env node

/**
 * Reproducible scalar EVM scaling benchmark for canonical GL1F v1 models.
 *
 * This is supplementary measurement evidence, not a correctness replacement
 * for tests/contracts_publication/test_evm_integration.mjs and not a network
 * fee forecast. It uses the same compiler/client profile as that integration
 * test, publishes six deterministic model shapes, applies the same 12 seeded
 * vectors to every shape (72 profile-vector comparisons), and records raw gas
 * estimates plus simple descriptive fits against runtime work counts.
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";

import { BrowserProvider } from "ethers";
import ganache from "ganache";
import solc from "solc";

import {
  compileContracts,
  deploy,
  packFeatures,
  publishCanonicalModel,
  referenceV1,
  serializeV1,
} from "../tests/contracts_publication/test_evm_integration.mjs";

const CHUNK_SIZE = 20_000;
const VECTORS_PER_PROFILE = 12;
const VECTOR_SEED = 20_260_724;
const SCALE_Q = 1_000;
const PROFILES = [
  { trees: 20, depth: 3 },
  { trees: 50, depth: 3 },
  { trees: 50, depth: 4 },
  { trees: 100, depth: 4 },
  { trees: 200, depth: 4 },
  { trees: 50, depth: 6 },
];

function parseArgs(argv) {
  const options = {
    out: "benchmarks/results/evm_scaling_benchmark.json",
    markdown: "benchmarks/results/EVM_SCALING_BENCHMARK.md",
  };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--out") options.out = argv[++index];
    else if (argv[index] === "--markdown") options.markdown = argv[++index];
    else throw new Error(`unknown argument: ${argv[index]}`);
  }
  return options;
}

function deterministicTree(depth, treeIndex) {
  const internalNodes = (2 ** depth) - 1;
  const leafCount = 2 ** depth;
  const nodes = Array.from({ length: internalNodes }, (_, index) => ({
    feature: (index + treeIndex) % 2,
    threshold: (((index + 1) * 97 + (treeIndex + 1) * 53) % 6_001) - 3_000,
  }));
  const leaves = Array.from(
    { length: leafCount },
    (_, index) => (((index + 1) * 31 + (treeIndex + 1) * 17) % 2_001) - 1_000,
  );
  return { nodes, leaves };
}

function deterministicModel({ trees, depth }) {
  return serializeV1({
    nFeatures: 2,
    depth,
    baseQ: 137,
    scaleQ: SCALE_Q,
    trees: Array.from(
      { length: trees },
      (_, treeIndex) => deterministicTree(depth, treeIndex),
    ),
  });
}

function makeXorShift32(seed) {
  let state = seed >>> 0;
  if (state === 0) state = 0x6d2b79f5;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state;
  };
}

function makeVectors(count) {
  const next = makeXorShift32(VECTOR_SEED);
  return Array.from({ length: count }, () => [
    Number(next() % 6_001) - 3_000,
    Number(next() % 6_001) - 3_000,
  ]);
}

function gasSummary(values) {
  assert.ok(values.length > 0);
  const sorted = [...values].sort((left, right) => (left < right ? -1 : left > right ? 1 : 0));
  const sum = sorted.reduce((total, value) => total + value, 0n);
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2
    ? Number(sorted[middle])
    : Number(sorted[middle - 1] + sorted[middle]) / 2;
  return {
    raw: values.map(String),
    min: String(sorted[0]),
    median,
    mean: Number(sum) / sorted.length,
    max: String(sorted.at(-1)),
  };
}

function linearFit(rows, xKey) {
  const points = rows.map((row) => ({
    x: Number(row.work[xKey]),
    y: Number(row.estimatedGas.mean),
  }));
  const n = points.length;
  const meanX = points.reduce((sum, point) => sum + point.x, 0) / n;
  const meanY = points.reduce((sum, point) => sum + point.y, 0) / n;
  const covariance = points.reduce(
    (sum, point) => sum + ((point.x - meanX) * (point.y - meanY)),
    0,
  );
  const varianceX = points.reduce(
    (sum, point) => sum + ((point.x - meanX) ** 2),
    0,
  );
  const slope = covariance / varianceX;
  const intercept = meanY - (slope * meanX);
  const residuals = points.map((point) => point.y - (intercept + (slope * point.x)));
  const ssResidual = residuals.reduce((sum, value) => sum + (value ** 2), 0);
  const ssTotal = points.reduce((sum, point) => sum + ((point.y - meanY) ** 2), 0);
  return {
    x: xKey,
    interceptGas: Number(intercept.toFixed(3)),
    slopeGasPerUnit: Number(slope.toFixed(6)),
    rSquared: Number((1 - (ssResidual / ssTotal)).toFixed(8)),
    maxAbsoluteRelativeResidual: Number(
      Math.max(
        ...residuals.map((residual, index) => Math.abs(residual) / points[index].y),
      ).toFixed(8),
    ),
  };
}

function gitValue(args, fallback = "unavailable") {
  try {
    return execFileSync("git", args, { encoding: "utf8" }).trim();
  } catch {
    return fallback;
  }
}

function renderMarkdown(result) {
  const lines = [
    "# GL1F local-EVM scalar scaling benchmark",
    "",
    `**Status:** ${result.status}`,
    "",
    `**Recorded:** ${result.recordedAt}`,
    "",
    `**Source revision:** \`${result.source.revision}\`${
      result.source.dirty ? " (dirty working tree)" : ""
    }`,
    "",
    `**Compiler:** ${result.compiler.version}; viaIR; optimizer 200; EVM target Istanbul`,
    "",
    `**Execution client:** ${result.chain.engine}; hardfork ${result.chain.hardfork}`,
    "",
    `**Comparisons:** ${result.totalComparisons} exact reference/EVM profile-vector comparisons; `
      + `${result.mismatches} mismatches`,
    "",
    "These observations describe one local compiler/client/profile. They are not",
    "a live-network fee forecast, a marginal storage price, or a proof that every",
    "encoded model fits a transaction limit.",
    "",
    "| Trees | Depth | Bytes | Chunks | Decisions | Tree-body primitive reads | Rounded mean estimated gas | Min-max |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
  ];
  for (const row of result.profiles) {
    lines.push(
      `| ${row.trees} | ${row.depth} | ${row.coreBytes.toLocaleString("en-US")} `
      + `| ${row.chunks} | ${row.work.decisions.toLocaleString("en-US")} `
      + `| ${row.work.primitiveReads.toLocaleString("en-US")} `
      + `| ${Math.round(row.estimatedGas.mean).toLocaleString("en-US")} `
      + `| ${Number(row.estimatedGas.min).toLocaleString("en-US")}-`
      + `${Number(row.estimatedGas.max).toLocaleString("en-US")} |`,
    );
  }
  lines.push(
    "",
    "## Descriptive fits",
    "",
    ...result.fits.map(
      (fit) => `- Against \`${fit.x}\`: gas = ${fit.interceptGas.toLocaleString("en-US")} `
        + `+ ${fit.slopeGasPerUnit.toLocaleString("en-US")} x work; `
        + `R^2 = ${fit.rSquared}; maximum absolute relative residual = `
        + `${(fit.maxAbsoluteRelativeResidual * 100).toFixed(2)}%.`,
    ),
    "",
    "Decision and leaf-read counts follow exactly from canonical tree geometry.",
    "The tree-body primitive-read count is exact for the benchmarked scalar decoder.",
    "Fitted coefficients are empirical and can change with compiler, client,",
    "hardfork, chunk geometry, calldata, address warmth, and model layout.",
    "",
    "## Reproduction",
    "",
    "```bash",
    "npm ci",
    "npm run benchmark:evm",
    "```",
    "",
    "The JSON record retains every per-vector gas estimate, model identifier,",
    "write/registration gas observation, compiler profile, and host description.",
  );
  return `${lines.join("\n")}\n`;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const started = performance.now();
  const compileStarted = performance.now();
  const artifacts = compileContracts();
  const compileMs = performance.now() - compileStarted;

  const eip1193 = ganache.provider({
    logging: { quiet: true },
    wallet: {
      totalAccounts: 3,
      defaultBalance: 1_000,
      deterministic: true,
    },
    chain: {
      chainId: 1337,
      hardfork: "shanghai",
    },
    miner: {
      instamine: "eager",
    },
  });

  try {
    const provider = new BrowserProvider(eip1193);
    const owner = await provider.getSigner(0);
    const creator = await provider.getSigner(1);
    const ownerAddress = await owner.getAddress();

    const storeDeployment = await deploy(artifacts.ModelStore, owner);
    const registryDeployment = await deploy(
      artifacts.ModelRegistry,
      owner,
      [ownerAddress, "GL1F EVM scaling benchmark terms"],
    );
    const registryAddress = await registryDeployment.contract.getAddress();
    const nftDeployment = await deploy(
      artifacts.ModelNFT,
      owner,
      [registryAddress, "GL1F EVM Benchmark Model", "GL1FB"],
    );
    await (
      await registryDeployment.contract.setModelNFT(await nftDeployment.contract.getAddress())
    ).wait();
    const runtimeDeployment = await deploy(
      artifacts.ForestRuntime,
      owner,
      [registryAddress],
    );

    const vectors = makeVectors(VECTORS_PER_PROFILE);
    const rows = [];
    let comparisons = 0;
    let mismatches = 0;

    for (const profile of PROFILES) {
      const bytes = deterministicModel(profile);
      const published = await publishCanonicalModel({
        store: storeDeployment.contract,
        registry: registryDeployment.contract,
        provider,
        creator,
        bytes,
        chunkSize: CHUNK_SIZE,
        metadata: {
          nFeatures: 2,
          nTrees: profile.trees,
          depth: profile.depth,
          baseQ: 137,
          scaleQ: SCALE_Q,
          title: `EVM scaling M${profile.trees} d${profile.depth}`,
          description: "Deterministic scalar fixture for local EVM scaling measurement.",
          featuresPacked: "task=regression;features=f0,f1",
          versionLabel: `m${profile.trees}-d${profile.depth}`,
        },
      });

      const gasValues = [];
      for (const vector of vectors) {
        const packed = packFeatures(vector);
        const expected = referenceV1(bytes, vector);
        const observed = await runtimeDeployment.contract.predictView(
          published.modelId,
          packed,
        );
        comparisons += 1;
        if (observed !== expected) mismatches += 1;
        assert.equal(
          observed,
          expected,
          `reference/EVM mismatch for M=${profile.trees}, d=${profile.depth}, x=${vector}`,
        );
        gasValues.push(
          await runtimeDeployment.contract.predictView.estimateGas(
            published.modelId,
            packed,
          ),
        );
      }

      rows.push({
        ...profile,
        modelId: published.modelId,
        coreBytes: bytes.length,
        chunkSize: published.chunkSize,
        chunks: published.numChunks,
        work: {
          decisions: profile.trees * profile.depth,
          leafReads: profile.trees,
          primitiveReads: profile.trees * ((2 * profile.depth) + 1),
        },
        publicationGas: {
          chunkWrites: published.gas.chunks.map(String),
          tableWrite: String(published.gas.table),
          registration: String(published.gas.register),
        },
        estimatedGas: gasSummary(gasValues),
      });
    }

    const cpu = os.cpus()[0] ?? { model: "unavailable" };
    const result = {
      schemaVersion: 1,
      status: mismatches === 0 ? "PASS" : "FAIL",
      recordedAt: new Date().toISOString(),
      source: {
        revision: gitValue(["rev-parse", "HEAD"]),
        dirty: gitValue(["status", "--porcelain"], "") !== "",
      },
      environment: {
        platform: os.platform(),
        release: os.release(),
        architecture: os.arch(),
        cpu: cpu.model.trim(),
        logicalCpuCount: os.cpus().length,
        node: process.version,
      },
      compiler: {
        version: solc.version(),
        viaIR: true,
        optimizerRuns: 200,
        evmVersion: "istanbul",
        compileMs: Math.round(compileMs),
        runtimeBytes: Object.fromEntries(
          Object.entries(artifacts).map(([name, artifact]) => [name, artifact.runtimeBytes]),
        ),
      },
      chain: {
        engine: `ganache ${ganache.version ?? "7.9.2"}`,
        chainId: 1337,
        hardfork: "shanghai",
      },
      deploymentGas: {
        ModelStore: String(storeDeployment.gasUsed),
        ModelRegistry: String(registryDeployment.gasUsed),
        ModelNFT: String(nftDeployment.gasUsed),
        ForestRuntime: String(runtimeDeployment.gasUsed),
      },
      design: {
        chunkSize: CHUNK_SIZE,
        uniqueSharedVectors: VECTORS_PER_PROFILE,
        comparisonsPerProfile: VECTORS_PER_PROFILE,
        vectorSeed: VECTOR_SEED,
        featureRangeQInclusive: [-3_000, 3_000],
        featureCount: 2,
        scaleQ: SCALE_Q,
      },
      profiles: rows,
      totalComparisons: comparisons,
      mismatches,
      fits: [
        linearFit(rows, "decisions"),
        linearFit(rows, "primitiveReads"),
      ],
      totalRuntimeMs: Math.round(performance.now() - started),
      limitations: [
        "One host, compiler, client, hardfork, chunk geometry, and synthetic model family.",
        "eth_estimateGas observations are not mined inference receipts or live-network fees.",
        "The fit is descriptive over six points and is not a protocol gas theorem.",
        "Actual gas can vary with selected chunks and address warmth even when decision counts are fixed.",
      ],
    };

    assert.equal(result.totalComparisons, PROFILES.length * VECTORS_PER_PROFILE);
    assert.equal(result.mismatches, 0);

    const jsonPath = path.resolve(options.out);
    const markdownPath = path.resolve(options.markdown);
    mkdirSync(path.dirname(jsonPath), { recursive: true });
    mkdirSync(path.dirname(markdownPath), { recursive: true });
    writeFileSync(jsonPath, `${JSON.stringify(result, null, 2)}\n`);
    writeFileSync(markdownPath, renderMarkdown(result));
    process.stdout.write(
      `PASS: ${result.totalComparisons}/${result.totalComparisons} reference/EVM comparisons; `
      + `wrote ${options.out} and ${options.markdown}\n`,
    );
  } finally {
    await eip1193.disconnect();
  }
}

await main();

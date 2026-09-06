#!/usr/bin/env node

/** Controlled local-EVM comparison of identical canonical scalar core bytes.
 * Run from the repository root: node benchmarks/storage_comparison.mjs
 * Production code chunks are compared with a packed Solidity `bytes` mapping.
 * Registry metadata and feature vectors are shared; no public RPC is used.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { BrowserProvider, hexlify } from "ethers";
import ganache from "ganache";
import solc from "solc";
import {
  compileContracts, deploy, packFeatures, publishCanonicalModel, referenceV1, serializeV1,
} from "../tests/contracts_publication/test_evm_integration.mjs";

const PROFILES = [
  { trees: 20, depth: 3 }, { trees: 50, depth: 3 }, { trees: 50, depth: 4 },
  { trees: 100, depth: 4 }, { trees: 200, depth: 4 }, { trees: 50, depth: 6 },
];
const CHUNK_SIZE = 20_000;
const VECTOR_SEED = 20_260_724;
const VECTOR_COUNT = 12;
const SOURCE_FILES = [
  "benchmarks/storage_comparison.mjs", "benchmarks/contracts/PackedStorageBaseline.sol",
  "tests/contracts_publication/test_evm_integration.mjs", "contracts/ForestRuntime.sol",
  "contracts/ModelStore.sol", "contracts/ModelRegistry.sol", "contracts/ModelNFT.sol",
  "contracts/SimpleOwnable.sol", "package-lock.json", ".nvmrc",
];

function compileBaseline() {
  const files = ["benchmarks/contracts/PackedStorageBaseline.sol", "contracts/ForestRuntime.sol"];
  const input = {
    language: "Solidity",
    sources: Object.fromEntries(files.map((name) => [name, { content: readFileSync(name, "utf8") }])),
    settings: {
      optimizer: { enabled: true, runs: 200 }, viaIR: true, evmVersion: "istanbul",
      outputSelection: { "*": { "*": ["abi", "evm.bytecode.object", "evm.deployedBytecode.object"] } },
    },
  };
  const output = JSON.parse(solc.compile(JSON.stringify(input)));
  const errors = (output.errors ?? []).filter((item) => item.severity === "error");
  assert.equal(errors.length, 0, errors.map((item) => item.formattedMessage).join("\n"));
  const result = output.contracts[files[0]].PackedStorageBaseline;
  return {
    abi: result.abi, bytecode: `0x${result.evm.bytecode.object}`,
    runtimeBytes: result.evm.deployedBytecode.object.length / 2,
  };
}

// Same model family and vectors as evm_scaling_benchmark.mjs.
function makeCore({ trees, depth }) {
  return serializeV1({
    nFeatures: 2, depth, baseQ: 137, scaleQ: 1_000,
    trees: Array.from({ length: trees }, (_, treeIndex) => ({
      nodes: Array.from({ length: 2 ** depth - 1 }, (_, index) => ({
        feature: (index + treeIndex) % 2,
        threshold: (((index + 1) * 97 + (treeIndex + 1) * 53) % 6_001) - 3_000,
      })),
      leaves: Array.from({ length: 2 ** depth }, (_, index) => (
        (((index + 1) * 31 + (treeIndex + 1) * 17) % 2_001) - 1_000
      )),
    })),
  });
}

function makeVectors() {
  let state = VECTOR_SEED >>> 0;
  const next = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return Number(state % 6_001) - 3_000;
  };
  return Array.from({ length: VECTOR_COUNT }, () => [next(), next()]);
}

function summary(values) {
  const numbers = values.map(Number).sort((a, b) => a - b);
  return {
    min: numbers[0], max: numbers.at(-1),
    mean: numbers.reduce((a, b) => a + b, 0) / numbers.length,
    median: (numbers[5] + numbers[6]) / 2,
  };
}

function git(args) {
  return execFileSync("git", args, { encoding: "utf8" }).trim();
}

function parseArgs() {
  let out = "benchmarks/results/storage_comparison.json";
  const args = process.argv.slice(2);
  if (args.length) {
    assert.equal(args.length, 2, "usage: node benchmarks/storage_comparison.mjs [--out FILE]");
    assert.equal(args[0], "--out");
    out = args[1];
  }
  return out;
}

async function main() {
  const out = parseArgs();
  const source = {
    revision: git(["rev-parse", "HEAD"]), dirty: git(["status", "--porcelain"]) !== "",
    sha256: Object.fromEntries(SOURCE_FILES.map((name) => [
      name, createHash("sha256").update(readFileSync(name)).digest("hex"),
    ])),
  };
  const artifacts = compileContracts();
  artifacts.PackedStorageBaseline = compileBaseline();
  const chain = ganache.provider({
    logging: { quiet: true },
    wallet: { totalAccounts: 3, defaultBalance: 1_000, deterministic: true },
    chain: { chainId: 1337, hardfork: "shanghai" },
    miner: { instamine: "eager", blockGasLimit: 30_000_000 },
  });
  try {
    const provider = new BrowserProvider(chain);
    const owner = await provider.getSigner(0);
    const creator = await provider.getSigner(1);
    const ownerAddress = await owner.getAddress();
    const store = await deploy(artifacts.ModelStore, owner);
    const registry = await deploy(artifacts.ModelRegistry, owner, [ownerAddress, "Storage benchmark"]);
    const registryAddress = await registry.contract.getAddress();
    const nft = await deploy(artifacts.ModelNFT, owner, [registryAddress, "Storage benchmark", "SB"]);
    const linkReceipt = await (await registry.contract.setModelNFT(await nft.contract.getAddress())).wait();
    const codeRuntime = await deploy(artifacts.ForestRuntime, owner, [registryAddress]);
    const storageRuntime = await deploy(artifacts.PackedStorageBaseline, owner, [registryAddress]);
    const vectors = makeVectors();
    const profiles = [];

    for (const profile of PROFILES) {
      const core = makeCore(profile);
      const published = await publishCanonicalModel({
        store: store.contract, registry: registry.contract, provider, creator, bytes: core,
        chunkSize: CHUNK_SIZE,
        metadata: {
          nFeatures: 2, nTrees: profile.trees, depth: profile.depth, baseQ: 137, scaleQ: 1_000,
          title: `Storage comparison M${profile.trees} d${profile.depth}`,
          description: "Deterministic local-EVM storage comparison fixture.",
          featuresPacked: "task=regression;features=f0,f1",
          versionLabel: `m${profile.trees}-d${profile.depth}`,
        },
      });
      // Explicit gas limit avoids repeatedly simulating the largest storage write
      // during estimation. The mined receipt, not this limit, is the measurement.
      const write = await storageRuntime.contract.connect(creator).write(
        published.modelId, hexlify(core), { gasLimit: 30_000_000 },
      );
      const writeReceipt = await write.wait();
      assert.equal(await storageRuntime.contract.readModel(published.modelId), hexlify(core));
      const observations = [];
      for (const featuresQ of vectors) {
        const packed = packFeatures(featuresQ);
        const expected = referenceV1(core, featuresQ);
        const codeOutput = await codeRuntime.contract.predictView(published.modelId, packed);
        const storageOutput = await storageRuntime.contract.predictView(published.modelId, packed);
        assert.equal(codeOutput, expected, "code/reference output mismatch");
        assert.equal(storageOutput, expected, "storage/reference output mismatch");
        const codeGas = await codeRuntime.contract.predictView.estimateGas(published.modelId, packed);
        const storageGas = await storageRuntime.contract.predictView.estimateGas(published.modelId, packed);
        observations.push({
          featuresQ, packedFeatures: packed, referenceOutputQ: String(expected),
          codeOutputQ: String(codeOutput), storageOutputQ: String(storageOutput),
          codeEstimatedGas: String(codeGas), storageEstimatedGas: String(storageGas),
        });
      }
      const codeWriteGas = published.gas.chunks.reduce((a, b) => a + b, published.gas.table);
      const codeGas = summary(observations.map((row) => row.codeEstimatedGas));
      const storageGas = summary(observations.map((row) => row.storageEstimatedGas));
      const row = {
        ...profile, modelId: published.modelId, coreBytes: core.length,
        coreSha256: createHash("sha256").update(core).digest("hex"),
        chunks: published.numChunks,
        materializedData: {
          // Excludes shared executable code, account/trie overhead and registry.
          codeBytesIncludingMagicAndPointerTable: core.length + published.numChunks * 4 + 4 + published.numChunks * 32,
          solidityStorageSlots: 1 + Math.ceil(core.length / 32),
          solidityStorageBytes: (1 + Math.ceil(core.length / 32)) * 32,
        },
        perModelGas: {
          codeChunkWriteReceipts: published.gas.chunks.map(String),
          codeTableWriteReceipt: String(published.gas.table), codeMaterialization: String(codeWriteGas),
          storageMaterialization: String(writeReceipt.gasUsed),
          commonRegistryRegistration: String(published.gas.register),
          storageToCodeMaterializationRatio: Number(writeReceipt.gasUsed) / Number(codeWriteGas),
        },
        inferenceEstimates: { code: codeGas, storage: storageGas, storageToCodeMeanRatio: storageGas.mean / codeGas.mean },
        observations,
      };
      profiles.push(row);
      process.stdout.write(`M=${profile.trees} d=${profile.depth}: exact parity ${VECTOR_COUNT}/${VECTOR_COUNT}; `
        + `materialize code/storage ${codeWriteGas}/${writeReceipt.gasUsed}; `
        + `mean inference code/storage ${codeGas.mean.toFixed(1)}/${storageGas.mean.toFixed(1)}\n`);
    }
    const result = {
      schemaVersion: 1, status: "PASS", recordedAt: new Date().toISOString(), source,
      environment: { node: process.version, platform: os.platform(), architecture: os.arch(), cpu: os.cpus()[0]?.model },
      compiler: {
        version: solc.version(), viaIR: true, optimizerRuns: 200, evmVersion: "istanbul",
        runtimeBytes: Object.fromEntries(Object.entries(artifacts).map(([name, artifact]) => [name, artifact.runtimeBytes])),
      },
      chain: { engine: "ganache 7.9.2", chainId: 1337, hardfork: "shanghai", blockGasLimit: 30_000_000 },
      design: {
        chunkSize: CHUNK_SIZE, vectorSeed: VECTOR_SEED, vectorsPerProfile: VECTOR_COUNT,
        featureCount: 2, scaleQ: 1_000, baseQ: 137,
        baseline: "Exact canonical core bytes in a Solidity mapping(bytes32 => bytes); packed word SLOAD decoder.",
        sharedMetadata: "Both paths query the same registered model twice, matching the production scalar entry point.",
        gasScope: "Materialization uses mined local transaction receipts. Inference uses independent eth_estimateGas requests; access warmth resets for each request.",
        storageAccounting: "Code includes all chunk magic and the pointer table. Storage includes one length slot and ceil(coreBytes/32) data slots. Both exclude common registry metadata, shared executable code, account and trie overhead.",
      },
      sharedDeploymentGas: {
        ModelStore: String(store.gasUsed), ModelRegistry: String(registry.gasUsed), ModelNFT: String(nft.gasUsed),
        RegistryNFTLink: String(linkReceipt.gasUsed), ForestRuntime: String(codeRuntime.gasUsed),
        PackedStorageBaseline: String(storageRuntime.gasUsed),
      },
      profiles, threeWayComparisons: profiles.length * VECTOR_COUNT, referenceEvmComparisons: profiles.length * VECTOR_COUNT * 2,
      mismatches: 0,
      limitations: [
        "One synthetic scalar family, compiler, local EVM client, hardfork and chunk size; not a live-network fee forecast.",
        "This is a controlled implemented-backend comparison, not a universal gas lower bound or a comparison with all storage encodings.",
        "Both paths retain canonical byte packing. The baseline does not use per-node Solidity structs or sparse model compression.",
        "The baseline implements only free scalar inference and storage writes; production shared deployment includes additional functionality, so shared deployment costs are not used for a relative efficiency claim.",
        "The common registry still carries code-backend pointer metadata. Its registration receipt is reported separately, not attributed to either materialization path.",
        "The benchmark verifies bytes and outputs, not access control, malformed-input handling, model quality or production readiness of the comparator.",
      ],
    };
    const filename = path.resolve(out);
    mkdirSync(path.dirname(filename), { recursive: true });
    writeFileSync(filename, `${JSON.stringify(result, null, 2)}\n`);
    process.stdout.write(`PASS: ${result.threeWayComparisons} three-way comparisons; wrote ${out}\n`);
  } finally {
    await chain.disconnect();
  }
}

await main();

#!/usr/bin/env node

/**
 * End-to-end GL1F EVM publication test.
 *
 * This test compiles the unmodified contracts with the documented deployment
 * profile, starts an in-process Ganache chain, wires the core contracts,
 * publishes canonical v1 and v2 models as GL1C code objects, and compares EVM
 * view inference with an independent integer reference interpreter.
 */

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import * as ethers from "ethers";
import ganache from "ganache";
import solc from "solc";

const { BrowserProvider, ContractFactory, getAddress, getBytes, hexlify, id, keccak256 } = ethers;
globalThis.ethers = ethers;
const { loadModelBytesFromChain } = await import(
  pathToFileURL(resolve("src/local_infer.js")).href
);

const SOLC_PREFIX = "0.8.20";
const CHUNK_MAGIC = Buffer.from("GL1C", "ascii");
const MODEL_MAGIC = Buffer.from("GL1F", "ascii");
export const CHUNK_SIZE = 17; // >= 4; deliberately makes several i32 fields cross a chunk boundary.

const SOURCE_FILES = [
  "contracts/ForestRuntime.sol",
  "contracts/ModelStore.sol",
  "contracts/ModelRegistry.sol",
  "contracts/ModelNFT.sol",
  "contracts/SimpleOwnable.sol",
];
const EVIDENCE_SOURCE_FILES = [
  ...SOURCE_FILES,
  ".nvmrc",
  "package-lock.json",
  "src/local_infer.js",
  "tests/contracts_publication/test_evm_integration.mjs",
];

function parseArgs(argv) {
  const options = { out: null };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--out") {
      options.out = argv[++index];
      assert.ok(options.out, "--out requires a path");
    } else {
      throw new Error(`unknown argument: ${argv[index]}`);
    }
  }
  return options;
}

function sourceDigests() {
  return Object.fromEntries(
    [...EVIDENCE_SOURCE_FILES].sort().map((name) => [
      name,
      createHash("sha256").update(readFileSync(name)).digest("hex"),
    ]),
  );
}

export function compileContracts() {
  assert.ok(
    solc.version().startsWith(SOLC_PREFIX),
    `expected solc ${SOLC_PREFIX}.x, received ${solc.version()}`,
  );

  const sources = Object.fromEntries(
    SOURCE_FILES.map((name) => [name, { content: readFileSync(name, "utf8") }]),
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
          "*": ["abi", "evm.bytecode.object", "evm.deployedBytecode.object"],
        },
      },
    },
  };

  const output = JSON.parse(solc.compile(JSON.stringify(input)));
  const diagnostics = output.errors ?? [];
  const errors = diagnostics.filter(({ severity }) => severity === "error");
  if (errors.length > 0) {
    throw new Error(errors.map(({ formattedMessage }) => formattedMessage).join("\n"));
  }

  function artifact(sourceName, contractName) {
    const compiled = output.contracts?.[sourceName]?.[contractName];
    assert.ok(compiled, `missing compiler artifact ${sourceName}:${contractName}`);
    assert.ok(compiled.evm.bytecode.object, `empty creation bytecode for ${contractName}`);
    return {
      abi: compiled.abi,
      bytecode: `0x${compiled.evm.bytecode.object}`,
      runtimeBytes: compiled.evm.deployedBytecode.object.length / 2,
    };
  }

  return {
    ForestRuntime: artifact("contracts/ForestRuntime.sol", "ForestRuntime"),
    ModelStore: artifact("contracts/ModelStore.sol", "ModelStore"),
    ModelRegistry: artifact("contracts/ModelRegistry.sol", "ModelRegistry"),
    ModelNFT: artifact("contracts/ModelNFT.sol", "ModelNFT"),
  };
}

function writeU16LE(buffer, offset, value) {
  buffer.writeUInt16LE(value, offset);
}

function writeU32LE(buffer, offset, value) {
  buffer.writeUInt32LE(value, offset);
}

function writeI32LE(buffer, offset, value) {
  buffer.writeInt32LE(value, offset);
}

export function serializeTree(depth, nodes, leaves) {
  const internalNodes = (2 ** depth) - 1;
  const leafCount = 2 ** depth;
  assert.equal(nodes.length, internalNodes, "complete tree requires 2^depth - 1 nodes");
  assert.equal(leaves.length, leafCount, "complete tree requires 2^depth leaves");

  const result = Buffer.alloc((internalNodes * 8) + (leafCount * 4));
  for (let index = 0; index < nodes.length; index += 1) {
    const offset = index * 8;
    writeU16LE(result, offset, nodes[index].feature);
    writeI32LE(result, offset + 2, nodes[index].threshold);
    writeU16LE(result, offset + 6, 0);
  }
  const leafOffset = internalNodes * 8;
  leaves.forEach((leaf, index) => writeI32LE(result, leafOffset + (index * 4), leaf));
  return result;
}

export function serializeV1({ nFeatures, depth, baseQ, scaleQ, trees }) {
  const header = Buffer.alloc(24);
  MODEL_MAGIC.copy(header, 0);
  header[4] = 1;
  header[5] = 0;
  writeU16LE(header, 6, nFeatures);
  writeU16LE(header, 8, depth);
  writeU32LE(header, 10, trees.length);
  writeI32LE(header, 14, baseQ);
  writeU32LE(header, 18, scaleQ);
  writeU16LE(header, 22, 0);
  return Buffer.concat([header, ...trees.map((tree) => serializeTree(depth, tree.nodes, tree.leaves))]);
}

function serializeV2({ nFeatures, depth, baseQ, scaleQ, treesByOutput }) {
  assert.ok(treesByOutput.length >= 2, "v2 requires at least two outputs");
  const treesPerOutput = treesByOutput[0].length;
  assert.ok(treesPerOutput > 0, "v2 requires at least one tree per output");
  assert.ok(
    treesByOutput.every((trees) => trees.length === treesPerOutput),
    "every v2 output must have the same number of trees",
  );
  assert.equal(baseQ.length, treesByOutput.length);

  const header = Buffer.alloc(24 + (4 * treesByOutput.length));
  MODEL_MAGIC.copy(header, 0);
  header[4] = 2;
  header[5] = 0;
  writeU16LE(header, 6, nFeatures);
  writeU16LE(header, 8, depth);
  writeU32LE(header, 10, treesPerOutput);
  writeI32LE(header, 14, 0);
  writeU32LE(header, 18, scaleQ);
  writeU16LE(header, 22, treesByOutput.length);
  baseQ.forEach((base, output) => writeI32LE(header, 24 + (4 * output), base));

  const trees = treesByOutput.flatMap((outputTrees) => (
    outputTrees.map((tree) => serializeTree(depth, tree.nodes, tree.leaves))
  ));
  return Buffer.concat([header, ...trees]);
}

function readTreeIncrement(bytes, treeOffset, depth, features) {
  const internalNodes = (2 ** depth) - 1;
  let index = 0;
  for (let level = 0; level < depth; level += 1) {
    const nodeOffset = treeOffset + (index * 8);
    const feature = bytes.readUInt16LE(nodeOffset);
    assert.ok(feature < features.length, "reference interpreter encountered an invalid feature index");
    const threshold = bytes.readInt32LE(nodeOffset + 2);
    index = features[feature] > threshold ? (index * 2) + 2 : (index * 2) + 1;
  }
  const leafIndex = index - internalNodes;
  return bytes.readInt32LE(treeOffset + (internalNodes * 8) + (leafIndex * 4));
}

export function referenceV1(bytes, features) {
  assert.equal(bytes.subarray(0, 4).toString("ascii"), "GL1F");
  assert.equal(bytes[4], 1);
  const nFeatures = bytes.readUInt16LE(6);
  const depth = bytes.readUInt16LE(8);
  const nTrees = bytes.readUInt32LE(10);
  assert.equal(features.length, nFeatures);
  const perTree = (((2 ** depth) - 1) * 8) + ((2 ** depth) * 4);
  let accumulator = BigInt(bytes.readInt32LE(14));
  for (let tree = 0; tree < nTrees; tree += 1) {
    accumulator += BigInt(readTreeIncrement(bytes, 24 + (tree * perTree), depth, features));
  }
  return accumulator;
}

function referenceV2(bytes, features) {
  assert.equal(bytes.subarray(0, 4).toString("ascii"), "GL1F");
  assert.equal(bytes[4], 2);
  const nFeatures = bytes.readUInt16LE(6);
  const depth = bytes.readUInt16LE(8);
  const treesPerOutput = bytes.readUInt32LE(10);
  const nOutputs = bytes.readUInt16LE(22);
  assert.equal(features.length, nFeatures);
  const perTree = (((2 ** depth) - 1) * 8) + ((2 ** depth) * 4);
  const treeRegion = 24 + (4 * nOutputs);

  return Array.from({ length: nOutputs }, (_, output) => {
    let accumulator = BigInt(bytes.readInt32LE(24 + (4 * output)));
    const outputBase = treeRegion + (output * treesPerOutput * perTree);
    for (let tree = 0; tree < treesPerOutput; tree += 1) {
      accumulator += BigInt(
        readTreeIncrement(bytes, outputBase + (tree * perTree), depth, features),
      );
    }
    return accumulator;
  });
}

function referenceClass(logits) {
  let bestClass = 0;
  for (let output = 1; output < logits.length; output += 1) {
    if (logits[output] > logits[bestClass]) bestClass = output;
  }
  return [BigInt(bestClass), logits[bestClass]];
}

export function packFeatures(features) {
  const packed = Buffer.alloc(features.length * 4);
  features.forEach((feature, index) => writeI32LE(packed, index * 4, feature));
  return hexlify(packed);
}

function parseEvent(contract, receipt, eventName) {
  for (const log of receipt.logs) {
    try {
      const parsed = contract.interface.parseLog(log);
      if (parsed?.name === eventName) return parsed;
    } catch {
      // The receipt can contain logs from another contract.
    }
  }
  throw new Error(`event ${eventName} not found`);
}

export async function deploy(factoryArtifact, signer, args = []) {
  const factory = new ContractFactory(factoryArtifact.abi, factoryArtifact.bytecode, signer);
  const contract = await factory.deploy(...args);
  const receipt = await contract.deploymentTransaction().wait();
  return { contract, gasUsed: receipt.gasUsed };
}

async function writeCodeObject(store, provider, signer, payload) {
  const transaction = await store.connect(signer).write(hexlify(payload));
  const receipt = await transaction.wait();
  const event = parseEvent(store, receipt, "ChunkWritten");
  const pointer = getAddress(event.args.pointer);
  assert.equal(event.args.size, BigInt(payload.length));

  const code = Buffer.from(getBytes(await provider.getCode(pointer)));
  assert.deepEqual(code.subarray(0, 4), CHUNK_MAGIC);
  assert.deepEqual(code.subarray(4), payload);
  return { pointer, gasUsed: receipt.gasUsed };
}

export async function publishCanonicalModel({
  store,
  registry,
  provider,
  creator,
  bytes,
  metadata,
  chunkSize = CHUNK_SIZE,
}) {
  const chunks = [];
  const writeGas = [];
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const stored = await writeCodeObject(
      store,
      provider,
      creator,
      bytes.subarray(offset, Math.min(bytes.length, offset + chunkSize)),
    );
    chunks.push(stored.pointer);
    writeGas.push(stored.gasUsed);
  }

  const tablePayload = Buffer.alloc(chunks.length * 32);
  chunks.forEach((pointer, index) => {
    Buffer.from(getBytes(pointer)).copy(tablePayload, (index * 32) + 12);
  });
  const table = await writeCodeObject(store, provider, creator, tablePayload);

  // Reconstruct using only the pointer-table and GL1C runtime code.
  const tableCode = Buffer.from(getBytes(await provider.getCode(table.pointer)));
  assert.deepEqual(tableCode.subarray(0, 4), CHUNK_MAGIC);
  const reconstructedParts = [];
  for (let index = 0; index < chunks.length; index += 1) {
    const slot = tableCode.subarray(4 + (index * 32), 4 + ((index + 1) * 32));
    assert.equal(slot.length, 32);
    const pointer = getAddress(hexlify(slot.subarray(12)));
    assert.equal(pointer, chunks[index]);
    const chunkCode = Buffer.from(getBytes(await provider.getCode(pointer)));
    assert.deepEqual(chunkCode.subarray(0, 4), CHUNK_MAGIC);
    reconstructedParts.push(chunkCode.subarray(4));
  }
  const reconstructed = Buffer.concat(reconstructedParts).subarray(0, bytes.length);
  assert.deepEqual(reconstructed, bytes);

  const modelId = keccak256(hexlify(bytes));
  assert.equal(keccak256(hexlify(reconstructed)), modelId);
  const creatorAddress = await creator.getAddress();
  const requiredFee = await registry.requiredDeployFeeWei(bytes.length);
  const register = await registry.connect(creator).registerModel(
    modelId,
    table.pointer,
    chunkSize,
    chunks.length,
    bytes.length,
    metadata.nFeatures,
    metadata.nTrees,
    metadata.depth,
    metadata.baseQ,
    metadata.scaleQ,
    metadata.title,
    metadata.description,
    "0x89504e470d0a1a0a",
    metadata.featuresPacked,
    [id("publication"), id(metadata.versionLabel)],
    0,
    0,
    creatorAddress,
    1,
    1,
    creatorAddress,
    { value: requiredFee },
  );
  const registerReceipt = await register.wait();
  const registered = parseEvent(registry, registerReceipt, "ModelRegistered");
  assert.equal(registered.args.modelId, modelId);
  assert.equal(registered.args.creator, creatorAddress);

  const [registeredTable, registeredChunkSize, numChunks, totalBytes] = (
    await registry.getModelBytesInfo(modelId)
  );
  assert.equal(registeredTable, table.pointer);
  assert.equal(registeredChunkSize, BigInt(chunkSize));
  assert.equal(numChunks, BigInt(chunks.length));
  assert.equal(totalBytes, BigInt(bytes.length));

  return {
    modelId,
    tokenId: registered.args.tokenId,
    tablePtr: table.pointer,
    numChunks: chunks.length,
    chunkSize,
    gas: {
      chunks: writeGas,
      table: table.gasUsed,
      register: registerReceipt.gasUsed,
    },
  };
}

export function summarizeGas(values) {
  const sorted = [...values].sort((a, b) => (a < b ? -1 : 1));
  const sum = sorted.reduce((accumulator, value) => accumulator + value, 0n);
  return {
    min: sorted[0].toString(),
    max: sorted.at(-1).toString(),
    mean: (sum / BigInt(sorted.length)).toString(),
  };
}

export async function main(options = parseArgs(process.argv.slice(2))) {
  const artifacts = compileContracts();

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

    const deployed = {};
    const storeDeployment = await deploy(artifacts.ModelStore, owner);
    deployed.ModelStore = storeDeployment.gasUsed;
    const registryDeployment = await deploy(
      artifacts.ModelRegistry,
      owner,
      [ownerAddress, "GL1F publication integration-test terms"],
    );
    deployed.ModelRegistry = registryDeployment.gasUsed;
    const registryAddress = await registryDeployment.contract.getAddress();
    const nftDeployment = await deploy(
      artifacts.ModelNFT,
      owner,
      [registryAddress, "GenesisL1 Forest Model", "GL1FM"],
    );
    deployed.ModelNFT = nftDeployment.gasUsed;
    const setNftReceipt = await (
      await registryDeployment.contract.setModelNFT(await nftDeployment.contract.getAddress())
    ).wait();
    deployed.setModelNFT = setNftReceipt.gasUsed;
    const runtimeDeployment = await deploy(
      artifacts.ForestRuntime,
      owner,
      [registryAddress],
    );
    deployed.ForestRuntime = runtimeDeployment.gasUsed;

    assert.equal(await nftDeployment.contract.registry(), registryAddress);
    assert.equal(await registryDeployment.contract.modelNFT(), await nftDeployment.contract.getAddress());
    assert.equal(await runtimeDeployment.contract.registry(), registryAddress);

    const commonNodes = [
      { feature: 0, threshold: 0 },
      { feature: 1, threshold: -10 },
      { feature: 1, threshold: 10 },
    ];
    const v1Bytes = serializeV1({
      nFeatures: 2,
      depth: 2,
      baseQ: 1_000,
      scaleQ: 1_000,
      trees: [
        {
          nodes: commonNodes,
          leaves: [-100, -20, 30, 200],
        },
        {
          nodes: [
            { feature: 1, threshold: 0 },
            { feature: 0, threshold: -5 },
            { feature: 0, threshold: 5 },
          ],
          leaves: [-7, 11, 13, 17],
        },
      ],
    });

    const v2Bytes = serializeV2({
      nFeatures: 2,
      depth: 2,
      baseQ: [100, -50, 25],
      scaleQ: 1_000,
      treesByOutput: [
        [{ nodes: commonNodes, leaves: [-90, 20, 30, 40] }],
        [{ nodes: commonNodes, leaves: [60, 100, -200, 200] }],
        [{ nodes: commonNodes, leaves: [-15, -10, 130, 0] }],
      ],
    });

    const v1 = await publishCanonicalModel({
      store: storeDeployment.contract,
      registry: registryDeployment.contract,
      provider,
      creator,
      bytes: v1Bytes,
      metadata: {
        nFeatures: 2,
        nTrees: 2,
        depth: 2,
        baseQ: 1_000,
        scaleQ: 1_000,
        title: "Publication v1 scalar witness",
        description: "Canonical scalar model used by the local EVM integration test.",
        featuresPacked: "task=regression;features=f0,f1",
        versionLabel: "v1",
      },
    });
    const v2 = await publishCanonicalModel({
      store: storeDeployment.contract,
      registry: registryDeployment.contract,
      provider,
      creator,
      bytes: v2Bytes,
      metadata: {
        nFeatures: 2,
        nTrees: 3,
        depth: 2,
        baseQ: 0,
        scaleQ: 1_000,
        title: "Publication v2 vector witness",
        description: "Canonical vector model used by the local EVM integration test.",
        featuresPacked: "task=multiclass;features=f0,f1;classes=c0,c1,c2",
        versionLabel: "v2",
      },
    });

    // Exercise the registry's assurance boundary on the isolated chain. The
    // interface accepts model identity and storage geometry independently.
    const canonicalV1ModelId = keccak256(hexlify(v1Bytes));
    assert.equal(canonicalV1ModelId, v1.modelId);
    const mismatchedModelId = id("gl1f-publication-mismatched-registry-id");
    assert.notEqual(mismatchedModelId, canonicalV1ModelId);
    const creatorAddress = await creator.getAddress();
    const mismatchRequiredFee = await registryDeployment.contract.requiredDeployFeeWei(
      v1Bytes.length,
    );
    const mismatchRegisterReceipt = await (
      await registryDeployment.contract.connect(creator).registerModel(
        mismatchedModelId,
        v1.tablePtr,
        v1.chunkSize,
        v1.numChunks,
        v1Bytes.length,
        2,
        2,
        2,
        1_000,
        1_000,
        "Publication mismatched-ID witness",
        "Local counterexample showing that registry identity needs independent verification.",
        "0x89504e470d0a1a0a",
        "task=regression;features=f0,f1",
        [id("publication"), id("mismatched-id-witness")],
        0,
        0,
        creatorAddress,
        1,
        1,
        creatorAddress,
        { value: mismatchRequiredFee },
      )
    ).wait();
    const mismatchRegistered = parseEvent(
      registryDeployment.contract,
      mismatchRegisterReceipt,
      "ModelRegistered",
    );
    assert.equal(mismatchRegistered.args.modelId, mismatchedModelId);
    const mismatchInfo = await registryDeployment.contract.getModelBytesInfo(
      mismatchedModelId,
    );
    assert.equal(mismatchInfo.tablePtr, v1.tablePtr);
    assert.equal(mismatchInfo.chunkSize, BigInt(v1.chunkSize));
    assert.equal(mismatchInfo.numChunks, BigInt(v1.numChunks));
    assert.equal(mismatchInfo.totalBytes, BigInt(v1Bytes.length));
    const mismatchInput = packFeatures([1, 10]);
    const mismatchPredictionQ = await runtimeDeployment.contract.predictView(
      mismatchedModelId,
      mismatchInput,
    );
    assert.equal(mismatchPredictionQ, referenceV1(v1Bytes, [1, 10]));
    await assert.rejects(
      loadModelBytesFromChain({
        provider,
        store: storeDeployment.contract,
        registry: registryDeployment.contract,
        modelId: mismatchedModelId,
      }),
      /Model ID mismatch/,
    );

    const boundaryVectors = [
      [-2, -10],                 // exact left-child threshold; v2 three-way tie
      [0, -9],                   // exact root threshold
      [1, 10],                   // exact right-child threshold
      [6, 11],                   // strictly greater than both positive thresholds
      [-2_147_483_648, 2_147_483_647],
      [2_147_483_647, -2_147_483_648],
    ];
    const v1ViewGas = [];
    const v2MultiViewGas = [];
    const v2ClassViewGas = [];

    for (const vector of boundaryVectors) {
      const packed = packFeatures(vector);

      const expectedScalar = referenceV1(v1Bytes, vector);
      const evmScalar = await runtimeDeployment.contract.predictView(v1.modelId, packed);
      assert.equal(evmScalar, expectedScalar, `v1 mismatch for [${vector}]`);
      v1ViewGas.push(
        await runtimeDeployment.contract.predictView.estimateGas(v1.modelId, packed),
      );

      const expectedLogits = referenceV2(v2Bytes, vector);
      const evmLogits = Array.from(
        await runtimeDeployment.contract.predictMultiView(v2.modelId, packed),
      );
      assert.deepEqual(evmLogits, expectedLogits, `v2 logits mismatch for [${vector}]`);
      v2MultiViewGas.push(
        await runtimeDeployment.contract.predictMultiView.estimateGas(v2.modelId, packed),
      );

      const expectedClass = referenceClass(expectedLogits);
      const evmClass = await runtimeDeployment.contract.predictClassView(v2.modelId, packed);
      assert.deepEqual(
        [evmClass.classIndex, evmClass.bestScoreQ],
        expectedClass,
        `v2 class mismatch for [${vector}]`,
      );
      v2ClassViewGas.push(
        await runtimeDeployment.contract.predictClassView.estimateGas(v2.modelId, packed),
      );
    }

    // Equality follows the specified <= branch, and equal logits retain the
    // lowest output index because the on-chain argmax updates only on `>`.
    assert.equal(referenceV1(v1Bytes, [0, -10]), 911n);
    assert.deepEqual(referenceV2(v2Bytes, [-2, -10]), [10n, 10n, 10n]);
    const tiedClass = await runtimeDeployment.contract.predictClassView(
      v2.modelId,
      packFeatures([-2, -10]),
    );
    assert.equal(tiedClass.classIndex, 0n);
    assert.equal(tiedClass.bestScoreQ, 10n);

    // Transaction paths execute the same integer core and provide measured
    // gas from mined receipts, in addition to view-call estimates.
    const v1TxInput = packFeatures([1, 10]);
    assert.equal(
      await runtimeDeployment.contract.connect(creator).predictTx.staticCall(v1.modelId, v1TxInput),
      referenceV1(v1Bytes, [1, 10]),
    );
    const v1TxReceipt = await (
      await runtimeDeployment.contract.connect(creator).predictTx(v1.modelId, v1TxInput)
    ).wait();
    const v1Inference = parseEvent(runtimeDeployment.contract, v1TxReceipt, "Inference");
    assert.equal(v1Inference.args.scoreQ, referenceV1(v1Bytes, [1, 10]));

    const v2TxInput = packFeatures([6, 11]);
    assert.deepEqual(
      Array.from(
        await runtimeDeployment.contract.connect(creator).predictMultiTx.staticCall(
          v2.modelId,
          v2TxInput,
        ),
      ),
      referenceV2(v2Bytes, [6, 11]),
    );
    const v2TxReceipt = await (
      await runtimeDeployment.contract.connect(creator).predictMultiTx(v2.modelId, v2TxInput)
    ).wait();
    const v2Inference = parseEvent(runtimeDeployment.contract, v2TxReceipt, "InferenceMulti");
    assert.deepEqual(
      Array.from(v2Inference.args.logitsQ),
      referenceV2(v2Bytes, [6, 11]),
    );

    await assert.rejects(
      runtimeDeployment.contract.predictView(v1.modelId, packFeatures([0])),
      /FEAT_LEN|execution reverted/,
    );

    const summary = {
      schema: "gl1f-local-evm-integration/v1",
      status: "PASS",
      command: "node tests/contracts_publication/test_evm_integration.mjs "
        + "--out benchmarks/results/evm_integration.json",
      scope: "Deterministic isolated Ganache conformance evidence; not a public-chain observation.",
      sourceDigests: sourceDigests(),
      software: {
        node: process.version,
        ethers: ethers.version,
        ganache: ganache.version ?? "7.9.2",
      },
      compiler: {
        version: solc.version(),
        viaIR: true,
        optimizerRuns: 200,
        evmVersion: "istanbul",
        runtimeBytes: Object.fromEntries(
          Object.entries(artifacts).map(([name, artifact]) => [name, artifact.runtimeBytes]),
        ),
      },
      chain: {
        engine: `ganache ${ganache.version ?? "7.9.2"}`,
        chainId: 1337,
        hardfork: "shanghai",
      },
      deploymentGas: Object.fromEntries(
        Object.entries(deployed).map(([name, value]) => [name, value.toString()]),
      ),
      models: {
        v1: {
          modelId: v1.modelId,
          bytes: v1Bytes.length,
          chunks: v1.numChunks,
          chunkSize: CHUNK_SIZE,
          tokenId: v1.tokenId.toString(),
          chunkWriteGas: summarizeGas(v1.gas.chunks),
          tableWriteGas: v1.gas.table.toString(),
          registrationGas: v1.gas.register.toString(),
          predictViewEstimatedGas: summarizeGas(v1ViewGas),
          predictTxGas: v1TxReceipt.gasUsed.toString(),
        },
        v2: {
          modelId: v2.modelId,
          bytes: v2Bytes.length,
          chunks: v2.numChunks,
          chunkSize: CHUNK_SIZE,
          tokenId: v2.tokenId.toString(),
          chunkWriteGas: summarizeGas(v2.gas.chunks),
          tableWriteGas: v2.gas.table.toString(),
          registrationGas: v2.gas.register.toString(),
          predictMultiViewEstimatedGas: summarizeGas(v2MultiViewGas),
          predictClassViewEstimatedGas: summarizeGas(v2ClassViewGas),
          predictMultiTxGas: v2TxReceipt.gasUsed.toString(),
        },
      },
      assuranceBoundaryWitness: {
        registryAcceptedMismatchedModelId: true,
        registeredModelId: mismatchedModelId,
        reconstructedModelId: canonicalV1ModelId,
        tokenId: mismatchRegistered.args.tokenId.toString(),
        predictViewReachedReferencedBytes: true,
        strictLoaderRejectedMismatchedModelId: true,
        predictionQ: mismatchPredictionQ.toString(),
      },
      directByteChecks: {
        chunkPayloadEqualsInputSlices: true,
        reconstructedCoreEqualsInputCore: true,
        reconstructedDigestEqualsModelId: true,
      },
      comparisons: {
        boundaryVectors: boundaryVectors.length,
        viewResults: boundaryVectors.length * 3,
        transactionResults: 2,
        mismatches: 0,
      },
    };

    if (options.out) {
      writeFileSync(options.out, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
    }
    process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
  } finally {
    await eip1193.disconnect();
  }
}

if (import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main(parseArgs(process.argv.slice(2)));
}

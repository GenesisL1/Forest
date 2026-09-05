#!/usr/bin/env node

/**
 * Reconstruct every active GenesisL1 Forest model at one number-pinned block,
 * verify its content commitment and every applicable registry/header
 * consistency field, and compare a local zero-vector prediction with the
 * read-call runtime where access settings permit. The explicit --extended
 * mode additionally executes deterministic exact-int32 conformance vectors,
 * attempts historical gas estimates, and instruments path-dependent model
 * reads. It writes separate versioned artifacts so the v1 witness remains
 * immutable. The script rechecks the recorded block hash after all calls.
 * The resulting JSON is evidence, not a claim that any registered model is
 * scientifically valid.
 */

import { mkdir, writeFile } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import {
  Contract,
  JsonRpcProvider,
  concat,
  getAddress,
  getBytes,
  hexlify,
  keccak256,
} from "ethers";

globalThis.ethers = { getAddress, getBytes };
const { decodeModel, predictMultiQ, predictQ } = await import("../src/local_infer.js");

const RPC_URL = process.env.GL1F_RPC_URL || "https://rpc.genesisl1.org";
const REGISTRY_ADDRESS = "0x33c9844F77a07e36B98f0FFf8201B8A8b02c2a69";
const NFT_ADDRESS = "0x44Dc1c54B8D579B42d78cC21cf8260DC0A3279fA";
const RUNTIME_ADDRESS = "0xD2fD0cf461a6cb56Fc08d9aEc120833D8E79044E";
const STORE_ADDRESS = "0x9CdbC23392648Bd27B4A5eD09c0fEa9452454B54";
const MARKET_ADDRESS = "0xA9bfa0a719b7F73cE85CA1E7f23af626D383fB46";
const DEFAULT_OUTPUT = "benchmarks/results/live_chain_witness.json";
const EXTENDED_OUTPUT = "benchmarks/results/live_chain_witness_extended_v2.json";
const EXTENDED_MARKDOWN = "benchmarks/results/LIVE_CHAIN_WITNESS_EXTENDED_V2.md";
const EXTENDED_PINNED_BLOCK = 13_342_043;
const INT32_MIN = -2_147_483_648;
const INT32_MAX = 2_147_483_647;

const rawArgs = process.argv.slice(2);
const EXTENDED_MODE = (
  rawArgs.includes("--extended")
  || process.env.GL1F_WITNESS_MODE?.toLowerCase() === "extended"
);
const HELP_MODE = rawArgs.includes("--help") || rawArgs.includes("-h");
const positionalArgs = rawArgs.filter((arg) => !["--extended", "--help", "-h"].includes(arg));

const registryAbi = [
  "function getModelSummary(uint256 tokenId) view returns (bool exists, bytes32 modelId, address tablePtr, uint16 nFeatures, uint16 nTrees, uint16 depth, int32 baseQ, uint8 pricingMode, uint256 feeWei, address feeRecipient, bool inferenceEnabled, address creator, uint32 tosVersionAccepted, string title, string description)",
  "function getModelBytesInfo(bytes32 modelId) view returns (address tablePtr, uint32 chunkSize, uint32 numChunks, uint32 totalBytes)",
  "function getModelRuntime(bytes32 modelId) view returns (address tablePtr, uint32 chunkSize, uint32 numChunks, uint32 totalBytes, uint16 nFeatures, uint16 nTrees, uint16 depth, int32 baseQ, uint32 scaleQ, bool inferenceEnabled, uint8 pricingMode, uint256 feeWei, address feeRecipient)",
  "function deployFeeWei() view returns (uint256)",
  "function sizeFeeWeiPerByte() view returns (uint256)",
  "function tosVersion() view returns (uint256)",
  "function activeLicenseId() view returns (uint256)",
  "function modelNFT() view returns (address)",
];
const nftAbi = ["function totalMinted() view returns (uint256)"];
const runtimeAbi = [
  "function predictView(bytes32 modelId, bytes packedFeaturesQ) view returns (int256)",
  "function predictMultiView(bytes32 modelId, bytes packedFeaturesQ) view returns (int256[] logitsQ)",
];

function stripChunkMagic(codeHex, context) {
  const bytes = getBytes(codeHex);
  if (bytes.length < 4 || hexlify(bytes.slice(0, 4)) !== "0x474c3143") {
    throw new Error(`${context}: runtime code does not begin with GL1C`);
  }
  return bytes.slice(4);
}

function pointerTable(codeHex, expectedPointers) {
  const data = stripChunkMagic(codeHex, "pointer table");
  if (data.length % 32 !== 0) throw new Error("pointer table payload is not 32-byte aligned");
  if (data.length !== expectedPointers * 32) {
    throw new Error(
      `pointer table has ${data.length / 32} entries; expected exactly ${expectedPointers}`,
    );
  }
  const pointers = [];
  for (let offset = 0; offset < data.length; offset += 32) {
    if (data.slice(offset, offset + 12).some((byte) => byte !== 0)) {
      throw new Error(
        `pointer table entry ${offset / 32} has non-zero high-order bytes`,
      );
    }
    pointers.push(getAddress(hexlify(data.slice(offset + 12, offset + 32))));
  }
  return pointers;
}

async function mapLimit(items, limit, fn) {
  const results = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const index = next++;
      results[index] = await fn(items[index], index);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

async function getCodeWithRetry(provider, address, blockTag, context) {
  const attempts = Number(process.env.GL1F_RPC_RETRIES || 6);
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      return await provider.getCode(address, blockTag);
    } catch (error) {
      lastError = error;
      if (attempt + 1 >= attempts) break;
      const delayMs = Math.min(4_000, 250 * (2 ** attempt));
      process.stderr.write(
        `${context}: RPC read failed; retry ${attempt + 2}/${attempts} in ${delayMs} ms\n`,
      );
      await new Promise((resolveDelay) => setTimeout(resolveDelay, delayMs));
    }
  }
  throw lastError;
}

async function reconstruct(provider, tablePtr, chunkSize, numChunks, totalBytes, blockTag) {
  if (chunkSize < 4 || chunkSize > 24_572) {
    throw new Error(`chunkSize ${chunkSize} is outside the canonical storage range`);
  }
  if (numChunks !== Math.ceil(totalBytes / chunkSize)) {
    throw new Error(
      `numChunks ${numChunks} does not equal ceil(${totalBytes}/${chunkSize})`,
    );
  }
  const tableCode = await getCodeWithRetry(
    provider,
    tablePtr,
    blockTag,
    "pointer table",
  );
  const pointers = pointerTable(tableCode, numChunks);
  const pieces = await mapLimit(
    pointers,
    Number(process.env.GL1F_RPC_CONCURRENCY || 16),
    async (pointer, index) => stripChunkMagic(
      await getCodeWithRetry(
        provider,
        pointer,
        blockTag,
        `chunk ${index + 1}`,
      ),
      `chunk ${index + 1}`,
    ),
  );
  for (let index = 0; index < pieces.length; index++) {
    const expected = index + 1 < pieces.length
      ? chunkSize
      : totalBytes - chunkSize * (pieces.length - 1);
    if (pieces[index].length !== expected) {
      throw new Error(
        `chunk ${index + 1} has ${pieces[index].length} payload bytes; expected ${expected}`,
      );
    }
  }
  const joined = getBytes(concat(pieces));
  if (joined.length !== totalBytes) {
    throw new Error(`reconstructed ${joined.length} bytes; registry declares exactly ${totalBytes}`);
  }
  return {
    bytes: joined.slice(0, totalBytes),
    pointers,
    storedPayloadBytes: pieces.reduce((sum, part) => sum + part.length, 0),
  };
}

function headerRecord(model) {
  return {
    version: model.version,
    nFeatures: model.nFeatures,
    depth: model.depth,
    totalTrees: model.nTrees,
    treesPerOutput: model.version === 1 ? model.nTrees : model.treesPerClass,
    outputs: model.version === 1 ? 1 : model.nClasses,
    baseQ: model.version === 1 ? [model.baseQ] : Array.from(model.baseLogitsQ),
    scaleQ: model.scaleQ,
    coreBytes: model.version === 1
      ? 24 + model.nTrees * model.perTree
      : model.treesOff + model.nTrees * model.perTree,
  };
}

function stringifyError(error) {
  return String(error?.shortMessage || error?.reason || error?.message || error);
}

function errorRecord(error) {
  const message = stringifyError(error).replace(/\s+/g, " ").slice(0, 800);
  const record = { message };
  if (error?.code !== undefined) record.code = String(error.code);
  if (error?.error?.code !== undefined) record.rpcCode = String(error.error.code);
  return record;
}

function blockQuantity(blockNumber) {
  return `0x${blockNumber.toString(16)}`;
}

function packInt32Vector(featuresQ) {
  const bytes = new Uint8Array(featuresQ.length * 4);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < featuresQ.length; index++) {
    const value = featuresQ[index];
    if (!Number.isInteger(value) || value < INT32_MIN || value > INT32_MAX) {
      throw new Error(`feature ${index} is not an int32: ${value}`);
    }
    view.setInt32(index * 4, value, true);
  }
  return hexlify(bytes);
}

function createModelReadMetrics(chunkSize) {
  const touchedChunks = new Set();
  const readKinds = {
    header: 0,
    outputBase: 0,
    featureIndex: 0,
    threshold: 0,
    leaf: 0,
  };
  let total = 0;
  let crossing = 0;

  return {
    record(offset, length, kind) {
      total += 1;
      readKinds[kind] += 1;
      const firstChunk = Math.floor(offset / chunkSize);
      const lastChunk = Math.floor((offset + length - 1) / chunkSize);
      touchedChunks.add(firstChunk);
      touchedChunks.add(lastChunk);
      if (firstChunk !== lastChunk) crossing += 1;
    },
    finish(pathBits, leftDecisions, rightDecisions, treesTraversed, outputsEvaluated) {
      return {
        outputsEvaluated,
        treesTraversed,
        pathDecisions: pathBits.length,
        leftDecisions,
        rightDecisions,
        pathDecisionDigest: keccak256(Uint8Array.from(pathBits)),
        pathDecisionDigestEncoding:
          "one byte per decision in output-major/tree-major/level order; 0=left, 1=right",
        modelCodeReadCalls: total,
        modelCodeReadCallsByKind: readKinds,
        crossChunkReadCalls: crossing,
        boundaryTemporaryAllocations: crossing,
        boundaryAllocationBasis:
          "ForestRuntime._readBytes allocates one temporary bytes object only for a cross-chunk read",
        dataChunkPointerLookups: total + crossing,
        uniqueDataChunksTouched: touchedChunks.size,
      };
    },
  };
}

/**
 * Independent quantized evaluator used by the live study.
 *
 * Unlike the browser-facing helpers, this routine accepts already-quantized
 * int32 values. It therefore archives the exact EVM input domain without a
 * float-to-Q conversion and instruments the model-byte offsets read along
 * every path.
 */
function evaluateQuantized(model, featuresQ, chunkSize) {
  if (featuresQ.length !== model.nFeatures) {
    throw new Error(`Need ${model.nFeatures} quantized features`);
  }
  for (let index = 0; index < featuresQ.length; index++) {
    if (
      !Number.isInteger(featuresQ[index])
      || featuresQ[index] < INT32_MIN
      || featuresQ[index] > INT32_MAX
    ) {
      throw new Error(`Quantized feature ${index} is outside int32`);
    }
  }

  const metrics = createModelReadMetrics(chunkSize);
  const pathBits = [];
  let leftDecisions = 0;
  let rightDecisions = 0;

  const traverseTree = (treeBase) => {
    let nodeIndex = 0;
    for (let level = 0; level < model.depth; level++) {
      const nodeOffset = treeBase + nodeIndex * 8;
      metrics.record(nodeOffset, 2, "featureIndex");
      metrics.record(nodeOffset + 2, 4, "threshold");
      const feature = model.dv.getUint16(nodeOffset, true);
      const thresholdQ = model.dv.getInt32(nodeOffset + 2, true);
      const goRight = featuresQ[feature] > thresholdQ;
      pathBits.push(goRight ? 1 : 0);
      if (goRight) {
        rightDecisions += 1;
        nodeIndex = nodeIndex * 2 + 2;
      } else {
        leftDecisions += 1;
        nodeIndex = nodeIndex * 2 + 1;
      }
    }
    const leafIndex = nodeIndex - model.internal;
    const leafOffset = treeBase + model.internal * 8 + leafIndex * 4;
    metrics.record(leafOffset, 4, "leaf");
    return model.dv.getInt32(leafOffset, true);
  };

  if (model.version === 1) {
    let accumulator = BigInt(model.baseQ);
    for (let treeIndex = 0; treeIndex < model.nTrees; treeIndex++) {
      const treeBase = model.treesOff + treeIndex * model.perTree;
      accumulator += BigInt(traverseTree(treeBase));
    }
    return {
      predictionQ: [String(accumulator)],
      metrics: metrics.finish(
        pathBits,
        leftDecisions,
        rightDecisions,
        model.nTrees,
        1,
      ),
    };
  }

  // These are the model-code header/base reads performed by
  // ForestRuntime._predictMultiFromChunks before tree traversal.
  for (const [offset, length] of [
    [0, 4],
    [4, 1],
    [6, 2],
    [8, 2],
    [10, 4],
    [18, 4],
    [22, 2],
  ]) {
    metrics.record(offset, length, "header");
  }

  const predictionQ = [];
  for (let outputIndex = 0; outputIndex < model.nClasses; outputIndex++) {
    metrics.record(24 + outputIndex * 4, 4, "outputBase");
    let accumulator = BigInt(model.baseLogitsQ[outputIndex]);
    const outputTreeBase = (
      model.treesOff + outputIndex * model.treesPerClass * model.perTree
    );
    for (let treeIndex = 0; treeIndex < model.treesPerClass; treeIndex++) {
      accumulator += BigInt(traverseTree(outputTreeBase + treeIndex * model.perTree));
    }
    predictionQ.push(String(accumulator));
  }
  return {
    predictionQ,
    metrics: metrics.finish(
      pathBits,
      leftDecisions,
      rightDecisions,
      model.nTrees,
      model.nClasses,
    ),
  };
}

function collectRootThresholdCandidates(model) {
  const candidates = [];
  const seen = new Set();
  for (let treeIndex = 0; treeIndex < model.nTrees; treeIndex++) {
    const rootOffset = model.treesOff + treeIndex * model.perTree;
    const featureIndex = model.dv.getUint16(rootOffset, true);
    const thresholdQ = model.dv.getInt32(rootOffset + 2, true);
    if (thresholdQ <= INT32_MIN || thresholdQ >= INT32_MAX) continue;
    const key = `${featureIndex}:${thresholdQ}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const outputIndex = model.version === 1
      ? 0
      : Math.floor(treeIndex / model.treesPerClass);
    candidates.push({
      featureIndex,
      thresholdQ,
      serializedTreeIndex: treeIndex,
      outputIndex,
      treeWithinOutput: model.version === 1
        ? treeIndex
        : treeIndex % model.treesPerClass,
      nodeIndex: 0,
      modelByteOffset: rootOffset,
      targetNodeIsAlwaysVisited: true,
    });
  }
  return candidates;
}

function stratifiedCandidateOrder(candidates) {
  if (candidates.length <= 1) return candidates.slice();
  const indices = [];
  const used = new Set();
  const add = (index) => {
    const bounded = Math.max(0, Math.min(candidates.length - 1, index));
    if (!used.has(bounded)) {
      used.add(bounded);
      indices.push(bounded);
    }
  };

  // Endpoints and recursive dyadic midpoints provide deterministic coverage
  // of the serialized ensemble without sampling randomness.
  add(0);
  add(candidates.length - 1);
  for (let denominator = 2; indices.length < candidates.length; denominator *= 2) {
    for (let numerator = 1; numerator < denominator; numerator += 2) {
      add(Math.round((numerator * (candidates.length - 1)) / denominator));
    }
    if (denominator > candidates.length * 4) break;
  }
  for (let index = 0; index < candidates.length; index++) add(index);
  return indices.map((index) => candidates[index]);
}

function generateConformanceVectors(model, cap) {
  if (!Number.isInteger(cap) || cap < 3 || cap > 64) {
    throw new Error(`GL1F_EXTENDED_VECTOR_CAP must be an integer in 3..64, got ${cap}`);
  }
  const vectors = [];
  const packedSeen = new Set();

  const addVector = (featuresQ, sourceRule, thresholdProbe = null) => {
    const packedFeaturesQHex = packInt32Vector(featuresQ);
    if (packedSeen.has(packedFeaturesQHex)) return false;
    packedSeen.add(packedFeaturesQHex);
    vectors.push({
      vectorId: `v${String(vectors.length + 1).padStart(2, "0")}`,
      sourceRule,
      thresholdProbe,
      featuresQ,
      packedFeaturesQHex,
      packedBytes: featuresQ.length * 4,
    });
    return true;
  };

  addVector(new Array(model.nFeatures).fill(0), "all-zero");
  addVector(new Array(model.nFeatures).fill(INT32_MIN), "all-int32-min");
  addVector(new Array(model.nFeatures).fill(INT32_MAX), "all-int32-max");

  const candidates = collectRootThresholdCandidates(model);
  for (const candidate of stratifiedCandidateOrder(candidates)) {
    const proposed = [-1, 0, 1].map((deltaQ) => {
      const featuresQ = new Array(model.nFeatures).fill(0);
      featuresQ[candidate.featureIndex] = candidate.thresholdQ + deltaQ;
      return {
        featuresQ,
        sourceRule: deltaQ === 0
          ? "root-threshold-equality"
          : `root-threshold-${deltaQ < 0 ? "minus" : "plus"}-one`,
        thresholdProbe: { ...candidate, deltaQ },
        packed: packInt32Vector(featuresQ),
      };
    });
    const newCount = proposed.filter((item) => !packedSeen.has(item.packed)).length;
    if (newCount === 0) continue;
    // Keep every selected threshold probe as a complete {-1, 0, +1}
    // neighbourhood after deduplication. A default cap of 10 consequently
    // yields about ten vectors (normally nine), without a partial triplet.
    if (vectors.length + newCount > cap) continue;
    for (const item of proposed) {
      addVector(item.featuresQ, item.sourceRule, item.thresholdProbe);
    }
  }

  const limitations = [];
  const thresholdVectors = vectors.filter((vector) => vector.thresholdProbe !== null);
  if (thresholdVectors.length === 0) {
    limitations.push(
      "No representable non-sentinel root threshold produced a distinct probe vector.",
    );
  }
  if (vectors.length < Math.min(9, cap)) {
    limitations.push(
      `Deduplication and complete-triplet selection produced ${vectors.length} vectors below the nominal nine-vector target.`,
    );
  }

  return {
    cap,
    vectorCount: vectors.length,
    deduplicationKey: "exact packed little-endian int32 feature bytes",
    baselineRules: ["all-zero", "all-int32-min", "all-int32-max"],
    thresholdCandidateRule:
      "unique, non-sentinel root (feature, threshold) pairs in serialized tree order",
    thresholdSelectionRule:
      "endpoints followed by recursive dyadic midpoints; complete {-1, equality, +1} triplets only",
    rootThresholdCandidates: candidates.length,
    generationLimitations: limitations,
    vectors,
  };
}

function isDeterministicExecutionError(error) {
  const message = stringifyError(error).toLowerCase();
  return (
    error?.code === "CALL_EXCEPTION"
    || message.includes("execution reverted")
    || message.includes("revert")
  );
}

async function rpcWithRetry(operation, context) {
  const maxAttempts = Number(process.env.GL1F_RPC_RETRIES || 6);
  let lastError;
  let attemptsUsed = 0;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    attemptsUsed = attempt;
    try {
      return { ok: true, value: await operation(), attempts: attempt };
    } catch (error) {
      lastError = error;
      if (isDeterministicExecutionError(error) || attempt === maxAttempts) break;
      const delayMs = Math.min(4_000, 250 * (2 ** (attempt - 1)));
      process.stderr.write(
        `${context}: RPC execution failed; retry ${attempt + 1}/${maxAttempts} in ${delayMs} ms\n`,
      );
      await new Promise((resolveDelay) => setTimeout(resolveDelay, delayMs));
    }
  }
  return { ok: false, error: errorRecord(lastError), attempts: attemptsUsed };
}

async function executeConformanceVector({
  provider,
  runtime,
  model,
  modelId,
  vector,
  blockTag,
  inferenceEnabled,
  pricingMode,
  chunkSize,
}) {
  const local = evaluateQuantized(model, vector.featuresQ, chunkSize);
  const method = model.version === 1 ? "predictView" : "predictMultiView";
  const calldata = runtime.interface.encodeFunctionData(
    method,
    [modelId, vector.packedFeaturesQHex],
  );
  const requestRecord = {
    to: RUNTIME_ADDRESS,
    method,
    blockTag: blockQuantity(blockTag),
    calldataBytes: getBytes(calldata).length,
    calldataKeccak256: keccak256(calldata),
  };

  let evmRead;
  if (!inferenceEnabled) {
    evmRead = {
      status: "not-applicable",
      reason: "registry inferenceEnabled=false at the pinned block",
      outputQ: null,
      exactMatch: null,
      attempts: 0,
    };
  } else if (pricingMode === 2) {
    evmRead = {
      status: "not-applicable",
      reason: "predictView/predictMultiView rejects paid-required pricing mode 2",
      outputQ: null,
      exactMatch: null,
      attempts: 0,
    };
  } else {
    const callResult = await rpcWithRetry(
      () => provider.call({ to: RUNTIME_ADDRESS, data: calldata, blockTag }),
      `token vector ${vector.vectorId} eth_call`,
    );
    if (callResult.ok) {
      const decoded = runtime.interface.decodeFunctionResult(method, callResult.value);
      const outputQ = model.version === 1
        ? [String(decoded[0])]
        : decoded[0].map(String);
      evmRead = {
        status: "compared",
        outputQ,
        exactMatch: JSON.stringify(outputQ) === JSON.stringify(local.predictionQ),
        attempts: callResult.attempts,
      };
    } else {
      evmRead = {
        status: "rpc-error",
        outputQ: null,
        exactMatch: null,
        attempts: callResult.attempts,
        error: callResult.error,
      };
    }
  }

  let historicalGasEstimate;
  if (!inferenceEnabled || pricingMode === 2) {
    historicalGasEstimate = {
      status: "not-applicable",
      reason: !inferenceEnabled
        ? "registry inferenceEnabled=false at the pinned block"
        : "the unrestricted view method rejects paid-required pricing mode 2",
      gas: null,
      attempts: 0,
    };
  } else {
    const gasResult = await rpcWithRetry(
      () => provider.send(
        "eth_estimateGas",
        [{ to: RUNTIME_ADDRESS, data: calldata }, blockQuantity(blockTag)],
      ),
      `token vector ${vector.vectorId} historical eth_estimateGas`,
    );
    if (gasResult.ok) {
      historicalGasEstimate = {
        status: "estimated",
        gas: String(BigInt(gasResult.value)),
        attempts: gasResult.attempts,
      };
    } else {
      historicalGasEstimate = {
        status: "rpc-error",
        gas: null,
        attempts: gasResult.attempts,
        error: gasResult.error,
      };
    }
  }

  return {
    ...vector,
    localPredictionQ: local.predictionQ,
    evmRead,
    historicalGasEstimate: {
      rpcMethod: "eth_estimateGas",
      historicalBlockParameterSupplied: true,
      blockTag: blockQuantity(blockTag),
      ...historicalGasEstimate,
    },
    rpcRequest: requestRecord,
    executionMetrics: local.metrics,
  };
}

function markdownEscape(value) {
  return String(value).replace(/\|/g, "\\|").replace(/\r?\n/g, " ");
}

function decimalRatio(numerator, denominator, decimalPlaces) {
  if (denominator <= 0n) throw new Error("ratio denominator must be positive");
  const scale = 10n ** BigInt(decimalPlaces);
  const rounded = (numerator * scale + denominator / 2n) / denominator;
  const whole = rounded / scale;
  const fractional = String(rounded % scale).padStart(decimalPlaces, "0");
  return decimalPlaces === 0 ? String(whole) : `${whole}.${fractional}`;
}

function gasUtilizationRecord(gas, blockGasLimit) {
  return {
    numeratorGas: String(gas),
    denominatorBlockGasLimit: String(blockGasLimit),
    fractionDecimal: decimalRatio(gas, blockGasLimit, 9),
    percent: decimalRatio(gas * 100n, blockGasLimit, 7),
  };
}

function gasSummaryForVectors(vectors, blockGasLimit) {
  const estimates = vectors
    .filter((vector) => vector.historicalGasEstimate.status === "estimated")
    .map((vector) => ({
      vectorId: vector.vectorId,
      gas: BigInt(vector.historicalGasEstimate.gas),
    }));
  if (estimates.length === 0) {
    return {
      estimatesReturned: 0,
      minimumGas: null,
      maximumGas: null,
      maximumVectorId: null,
      maximumUtilizationOfPinnedBlockGasLimit: null,
    };
  }
  let minimum = estimates[0];
  let maximum = estimates[0];
  for (const estimate of estimates.slice(1)) {
    if (estimate.gas < minimum.gas) minimum = estimate;
    if (estimate.gas > maximum.gas) maximum = estimate;
  }
  return {
    estimatesReturned: estimates.length,
    minimumGas: String(minimum.gas),
    maximumGas: String(maximum.gas),
    maximumVectorId: maximum.vectorId,
    maximumUtilizationOfPinnedBlockGasLimit:
      gasUtilizationRecord(maximum.gas, blockGasLimit),
  };
}

function formatWhole(value) {
  return BigInt(value).toLocaleString("en-US");
}

function gasRange(vectors) {
  const gas = vectors
    .filter((vector) => vector.historicalGasEstimate.status === "estimated")
    .map((vector) => BigInt(vector.historicalGasEstimate.gas));
  if (gas.length === 0) return "—";
  const low = gas.reduce((left, right) => left < right ? left : right);
  const high = gas.reduce((left, right) => left > right ? left : right);
  return low === high
    ? formatWhole(low)
    : `${formatWhole(low)}–${formatWhole(high)}`;
}

function renderExtendedMarkdown(result, jsonOutput) {
  const lines = [
    "# GL1F extended live-chain conformance and gas witness",
    "",
    `**Chain:** GenesisL1, chain ID ${result.chainId}  `,
    `**Number-pinned block:** ${result.block.number.toLocaleString("en-US")}  `,
    `**Block hash:** \`${result.block.hash}\`  `,
    `**Block time:** ${result.block.isoTimestamp}  `,
    `**Pinned block gas limit:** ${formatWhole(result.block.gasLimit)} (\`${result.block.gasLimitHex}\`)  `,
    `**RPC used:** \`${result.rpcUrl}\`  `,
    `**Reorganization guard:** block hash re-read and unchanged after all calls  `,
    `**Machine record:** [\`${basename(jsonOutput)}\`](${basename(jsonOutput)})  `,
    "**Reproducer:** [`../live_chain_witness.mjs`](../live_chain_witness.mjs)",
    "",
    "## Result",
    "",
    "| Check | Result |",
    "|---|---:|",
    `| Active registered models | ${result.summary.activeModels} |`,
    `| Reconstructed core bytes | ${result.summary.totalRegisteredModelBytes.toLocaleString("en-US")} |`,
    `| Content commitments matching | ${result.summary.contentCommitmentsMatched}/${result.summary.activeModels} |`,
    `| Extended conformance vectors | ${result.summary.extendedConformanceVectors} |`,
    `| Exact local/provider-returned vector results | ${result.summary.extendedReadCallsExactlyMatched}/${result.summary.extendedReadCallsCompared} |`,
    `| Historical gas estimates returned | ${result.summary.historicalGasEstimatesSucceeded}/${result.summary.historicalGasEstimatesAttempted} |`,
    `| Historical gas-estimation RPC errors | ${result.summary.historicalGasEstimateErrors} |`,
    `| Maximum historical gas estimate | ${formatWhole(result.summary.maximumHistoricalGasEstimate)} |`,
    `| Maximum estimate / pinned block limit | ${result.summary.maximumHistoricalGasUtilizationOfPinnedBlockLimit.percent}% |`,
    `| Instrumented model-code reads | ${result.summary.instrumentedModelCodeReadCalls.toLocaleString("en-US")} |`,
    `| Cross-chunk reads / derived temporary allocations | ${result.summary.instrumentedCrossChunkReadCalls.toLocaleString("en-US")} |`,
    "",
    "## Per-model evidence",
    "",
    "| Token | Title | Vectors | Exact provider/local | Gas returned | Historical gas range | Max / block limit | Cross-chunk reads |",
    "|---:|---|---:|---:|---:|---:|---:|---:|",
  ];

  for (const model of result.models.filter((item) => item.active)) {
    const vectors = model.conformanceStudy.vectors;
    const compared = vectors.filter((vector) => vector.evmRead.status === "compared");
    const exact = compared.filter((vector) => vector.evmRead.exactMatch);
    const gas = vectors.filter(
      (vector) => vector.historicalGasEstimate.status === "estimated",
    );
    const crossing = vectors.reduce(
      (sum, vector) => sum + vector.executionMetrics.crossChunkReadCalls,
      0,
    );
    const maximumPercent = (
      model.conformanceStudy.historicalGasSummary
        .maximumUtilizationOfPinnedBlockGasLimit?.percent
      ?? "—"
    );
    lines.push(
      `| ${model.tokenId} | ${markdownEscape(model.title)} | ${vectors.length} | `
      + `${exact.length}/${compared.length} | ${gas.length}/${vectors.length} | `
      + `${gasRange(vectors)} | ${maximumPercent === "—" ? maximumPercent : `${maximumPercent}%`} | `
      + `${crossing.toLocaleString("en-US")} |`,
    );
  }

  lines.push(
    "",
    "## Deterministic vector protocol",
    "",
    "Each model receives three exact int32 baselines: the all-zero vector, the",
    "all-`INT32_MIN` vector, and the all-`INT32_MAX` vector. The script then",
    "extracts unique, non-sentinel root `(feature, threshold)` pairs from the",
    "serialized ensemble. Roots are always visited, so each retained",
    "`threshold-1`, equality, and `threshold+1` triplet directly exercises the",
    "runtime's strict-greater-than branch at the named split. Candidates are",
    "selected deterministically across serialized order; exact packed inputs",
    "are deduplicated and complete triplets are retained up to the configured",
    "cap. The machine record stores every int32 vector, packed little-endian",
    "hex string, source rule, split provenance, local result, EVM result, gas",
    "outcome, path digest, read count, and chunk-boundary count.",
    "",
    "The local evaluator consumes int32 Q-values directly and accumulates with",
    "arbitrary-precision integers. This avoids floating-point input conversion.",
    "The EVM comparison uses only historical `eth_call`; gas is requested with",
    "historical `eth_estimateGas(transaction, blockTag)`. Unsupported or failed",
    "estimates remain explicit error records and are never replaced by a latest-",
    "block estimate. Maximum utilization records retain the estimate numerator,",
    "the pinned header's block-gas-limit denominator, and rounded decimal/percent",
    "renderings, so the reported percentages are independently recomputable.",
    "",
    "## Interpretation boundary",
    "",
    "This witness establishes public-byte reconstruction and execution agreement",
    "for the listed deployment state and inputs. Gas values are RPC simulations,",
    "not transaction receipts, and may be subject to the selected node's",
    "estimation policy. Read/boundary metrics are exact source-level",
    "instrumentation of model-code reads for the archived paths; they are not an",
    "opcode trace or a decomposition of total gas. The study does not establish",
    "training provenance, predictive accuracy, calibration, fairness, safety,",
    "authorship, or fitness for use.",
    "",
  );
  return `${lines.join("\n")}\n`;
}

async function main() {
  if (HELP_MODE) {
    process.stdout.write(
      "Usage: node benchmarks/live_chain_witness.mjs [OUTPUT.json] [BLOCK] [--extended]\n"
      + "\n"
      + "Default mode preserves the v1 zero-vector witness. Extended mode defaults\n"
      + `to publication block ${EXTENDED_PINNED_BLOCK} and writes a separate v2 JSON/Markdown study.\n`,
    );
    return;
  }
  const provider = new JsonRpcProvider(RPC_URL, undefined, { staticNetwork: false });
  const network = await provider.getNetwork();
  if (Number(network.chainId) !== 29) {
    throw new Error(`unexpected chain ID ${network.chainId}; expected GenesisL1 chain ID 29`);
  }
  const requested = process.env.GL1F_WITNESS_BLOCK || positionalArgs[1];
  const blockNumber = requested
    ? Number(requested)
    : EXTENDED_MODE
      ? EXTENDED_PINNED_BLOCK
      : await provider.getBlockNumber();
  if (!Number.isSafeInteger(blockNumber) || blockNumber < 0) {
    throw new Error(`invalid GL1F_WITNESS_BLOCK=${requested}`);
  }
  const block = await provider.getBlock(blockNumber);
  if (!block) throw new Error(`block ${blockNumber} not found`);
  const blockTag = blockNumber;

  const registry = new Contract(REGISTRY_ADDRESS, registryAbi, provider);
  const nft = new Contract(NFT_ADDRESS, nftAbi, provider);
  const runtime = new Contract(RUNTIME_ADDRESS, runtimeAbi, provider);
  const call = { blockTag };
  const totalMinted = Number(await nft.totalMinted(call));
  const extendedVectorCap = Number(process.env.GL1F_EXTENDED_VECTOR_CAP || 10);

  const contractCodeBytes = {};
  for (const [name, address] of Object.entries({
    store: STORE_ADDRESS,
    registry: REGISTRY_ADDRESS,
    nft: NFT_ADDRESS,
    runtime: RUNTIME_ADDRESS,
    marketplace: MARKET_ADDRESS,
  })) {
    contractCodeBytes[name] = (
      getBytes(await getCodeWithRetry(provider, address, blockTag, `${name} runtime`))
    ).length;
  }

  const models = [];
  for (let tokenId = 1; tokenId <= totalMinted; tokenId++) {
    const summary = await registry.getModelSummary(tokenId, call);
    if (!summary.exists) {
      models.push({ tokenId, active: false });
      continue;
    }
    const modelId = summary.modelId;
    const info = await registry.getModelBytesInfo(modelId, call);
    const runtimeMeta = await registry.getModelRuntime(modelId, call);
    const tablePtr = info.tablePtr;
    const chunkSize = Number(info.chunkSize);
    const numChunks = Number(info.numChunks);
    const totalBytes = Number(info.totalBytes);
    const rebuilt = await reconstruct(
      provider,
      tablePtr,
      chunkSize,
      numChunks,
      totalBytes,
      blockTag,
    );
    const computedModelId = keccak256(rebuilt.bytes);
    const decoded = decodeModel(rebuilt.bytes);
    const header = headerRecord(decoded);
    if (header.coreBytes !== totalBytes) {
      throw new Error(
        `token ${tokenId}: strict header length ${header.coreBytes} != registry ${totalBytes}`,
      );
    }
    const packedZero = hexlify(new Uint8Array(decoded.nFeatures * 4));
    const localPrediction = decoded.version === 1
      ? [String(predictQ(decoded, new Array(decoded.nFeatures).fill(0)))]
      : predictMultiQ(decoded, new Array(decoded.nFeatures).fill(0)).map(String);

    let chainPrediction = null;
    let predictionStatus = "not-attempted";
    if (!runtimeMeta.inferenceEnabled) {
      predictionStatus = "inference-disabled";
    } else if (Number(runtimeMeta.pricingMode) === 2) {
      predictionStatus = "paid-view-restricted";
    } else {
      try {
        chainPrediction = decoded.version === 1
          ? [String(await runtime.predictView(modelId, packedZero, call))]
          : (await runtime.predictMultiView(modelId, packedZero, call)).map(String);
        predictionStatus = "compared";
      } catch (error) {
        predictionStatus = `read-call-error: ${stringifyError(error)}`;
      }
    }

    const registryHeaderAgreement = {
      nFeatures: Number(runtimeMeta.nFeatures) === header.nFeatures,
      nTrees: Number(runtimeMeta.nTrees) === header.totalTrees,
      depth: Number(runtimeMeta.depth) === header.depth,
      baseQ: decoded.version === 1
        ? Number(runtimeMeta.baseQ) === decoded.baseQ
        : Number(runtimeMeta.baseQ) === 0,
      scaleQ: Number(runtimeMeta.scaleQ) === header.scaleQ,
      tablePtr: getAddress(runtimeMeta.tablePtr) === getAddress(tablePtr),
      totalBytes: Number(runtimeMeta.totalBytes) === totalBytes,
      numChunks: Number(runtimeMeta.numChunks) === numChunks,
      chunkSize: Number(runtimeMeta.chunkSize) === chunkSize,
    };

    let conformanceStudy;
    if (EXTENDED_MODE) {
      conformanceStudy = generateConformanceVectors(decoded, extendedVectorCap);
      const executedVectors = [];
      for (const vector of conformanceStudy.vectors) {
        executedVectors.push(await executeConformanceVector({
          provider,
          runtime,
          model: decoded,
          modelId,
          vector,
          blockTag,
          inferenceEnabled: Boolean(runtimeMeta.inferenceEnabled),
          pricingMode: Number(runtimeMeta.pricingMode),
          chunkSize,
        }));
        process.stderr.write(
          `Token ${tokenId} extended vector ${vector.vectorId}/${conformanceStudy.vectorCount}\n`,
        );
      }
      conformanceStudy = {
        ...conformanceStudy,
        historicalGasSummary: gasSummaryForVectors(executedVectors, block.gasLimit),
        vectors: executedVectors,
      };
    }

    const modelRecord = {
      tokenId,
      active: true,
      title: summary.title,
      creator: getAddress(summary.creator),
      modelId,
      computedModelId,
      contentCommitmentMatches: computedModelId.toLowerCase() === modelId.toLowerCase(),
      tablePtr: getAddress(tablePtr),
      chunkSize,
      numChunks,
      totalBytes,
      storedPayloadBytes: rebuilt.storedPayloadBytes,
      pointerCount: rebuilt.pointers.length,
      canonicalStorageShape: true,
      header,
      registryHeaderAgreement,
      inferenceEnabled: Boolean(runtimeMeta.inferenceEnabled),
      pricingMode: Number(runtimeMeta.pricingMode),
      zeroVector: {
        packedBytes: decoded.nFeatures * 4,
        localPredictionQ: localPrediction,
        chainPredictionQ: chainPrediction,
        status: predictionStatus,
        exactMatch: chainPrediction === null
          ? null
          : JSON.stringify(chainPrediction) === JSON.stringify(localPrediction),
      },
    };
    if (EXTENDED_MODE) modelRecord.conformanceStudy = conformanceStudy;
    models.push(modelRecord);
    process.stderr.write(`Verified token ${tokenId}/${totalMinted}: ${summary.title}\n`);
  }

  const active = models.filter((model) => model.active);
  const compared = active.filter((model) => model.zeroVector.status === "compared");
  const creators = [...new Set(active.map((model) => model.creator))];
  const [
    registryConfiguredNft,
    deployFeeWei,
    sizeFeeWeiPerByte,
    tosVersion,
    activeLicenseId,
  ] = await Promise.all([
    registry.modelNFT(call),
    registry.deployFeeWei(call),
    registry.sizeFeeWeiPerByte(call),
    registry.tosVersion(call),
    registry.activeLicenseId(call),
  ]);

  const extendedVectors = EXTENDED_MODE
    ? active.flatMap((model) => model.conformanceStudy.vectors)
    : [];
  const extendedCompared = extendedVectors.filter(
    (vector) => vector.evmRead.status === "compared",
  );
  const gasAttempted = extendedVectors.filter(
    (vector) => ["estimated", "rpc-error"].includes(vector.historicalGasEstimate.status),
  );
  const gasSucceeded = extendedVectors.filter(
    (vector) => vector.historicalGasEstimate.status === "estimated",
  );
  const gasObservations = EXTENDED_MODE
    ? active.flatMap((model) => model.conformanceStudy.vectors
      .filter((vector) => vector.historicalGasEstimate.status === "estimated")
      .map((vector) => ({
        tokenId: model.tokenId,
        vectorId: vector.vectorId,
        gas: BigInt(vector.historicalGasEstimate.gas),
      })))
    : [];
  const maximumGasObservation = gasObservations.reduce(
    (maximum, observation) => (
      maximum === null || observation.gas > maximum.gas ? observation : maximum
    ),
    null,
  );

  // This is deliberately the final RPC operation in the witness.
  const finalBlock = await provider.getBlock(blockNumber);
  if (!finalBlock || finalBlock.hash?.toLowerCase() !== block.hash?.toLowerCase()) {
    throw new Error(
      `block ${blockNumber} hash changed during witness: ${block.hash} -> ${finalBlock?.hash}`,
    );
  }
  const result = {
    schema: EXTENDED_MODE
      ? "gl1f-live-chain-witness/v2-extended"
      : "gl1f-live-chain-witness/v1",
    generatedAt: new Date().toISOString(),
    rpcUrl: RPC_URL,
    chainId: Number(network.chainId),
    ...(EXTENDED_MODE ? {
      studyMode: "extended-read-only",
      readOnlyRpcMethods: [
        "eth_chainId",
        "eth_blockNumber/eth_getBlockByNumber",
        "eth_getCode",
        "eth_call",
        "eth_estimateGas",
      ],
      prohibitedAndUnusedMethods: [
        "eth_sendTransaction",
        "eth_sendRawTransaction",
      ],
      conformanceProtocol: {
        featureDomain: "signed int32 Q-units, packed little-endian",
        vectorCapPerModel: extendedVectorCap,
        localEvaluator:
          "independent direct-int32 traversal with BigInt accumulation and model-byte read instrumentation",
        branchPredicate: "go right iff featureQ > thresholdQ; equality goes left",
        gasMethod:
          "eth_estimateGas with an explicit historical block-number parameter; no latest-block fallback",
      },
    } : {}),
    block: {
      number: block.number,
      hash: block.hash,
      timestamp: Number(block.timestamp),
      isoTimestamp: new Date(Number(block.timestamp) * 1000).toISOString(),
      ...(EXTENDED_MODE ? {
        gasLimit: String(block.gasLimit),
        gasLimitHex: blockQuantity(block.gasLimit),
      } : {}),
      selection: "number-pinned",
      hashRecheckedAfterCalls: true,
    },
    contracts: {
      store: STORE_ADDRESS,
      registry: REGISTRY_ADDRESS,
      nft: NFT_ADDRESS,
      runtime: RUNTIME_ADDRESS,
      marketplace: MARKET_ADDRESS,
      runtimeCodeBytes: contractCodeBytes,
      registryConfiguredNft,
      deployFeeWei: String(deployFeeWei),
      sizeFeeWeiPerByte: String(sizeFeeWeiPerByte),
      tosVersion: Number(tosVersion),
      activeLicenseId: Number(activeLicenseId),
    },
    summary: {
      totalMinted,
      activeModels: active.length,
      distinctCreators: creators.length,
      totalRegisteredModelBytes: active.reduce((sum, model) => sum + model.totalBytes, 0),
      contentCommitmentsMatched: active.filter((model) => model.contentCommitmentMatches).length,
      applicableRegistryHeaderChecksPassed: active.filter((model) =>
        Object.values(model.registryHeaderAgreement).every(Boolean)).length,
      readCallsCompared: compared.length,
      readCallsExactlyMatched: compared.filter((model) => model.zeroVector.exactMatch).length,
      minModelBytes: Math.min(...active.map((model) => model.totalBytes)),
      maxModelBytes: Math.max(...active.map((model) => model.totalBytes)),
      maxChunks: Math.max(...active.map((model) => model.numChunks)),
      ...(EXTENDED_MODE ? {
        extendedConformanceVectors: extendedVectors.length,
        extendedReadCallsCompared: extendedCompared.length,
        extendedReadCallsExactlyMatched: extendedCompared.filter(
          (vector) => vector.evmRead.exactMatch,
        ).length,
        extendedReadCallErrors: extendedVectors.filter(
          (vector) => vector.evmRead.status === "rpc-error",
        ).length,
        historicalGasEstimatesAttempted: gasAttempted.length,
        historicalGasEstimatesSucceeded: gasSucceeded.length,
        historicalGasEstimateErrors: extendedVectors.filter(
          (vector) => vector.historicalGasEstimate.status === "rpc-error",
        ).length,
        historicalGasEstimatesNotApplicable: extendedVectors.filter(
          (vector) => vector.historicalGasEstimate.status === "not-applicable",
        ).length,
        maximumHistoricalGasEstimate: maximumGasObservation === null
          ? null
          : String(maximumGasObservation.gas),
        maximumHistoricalGasEstimateTokenId: maximumGasObservation?.tokenId ?? null,
        maximumHistoricalGasEstimateVectorId: maximumGasObservation?.vectorId ?? null,
        maximumHistoricalGasUtilizationOfPinnedBlockLimit:
          maximumGasObservation === null
            ? null
            : gasUtilizationRecord(maximumGasObservation.gas, block.gasLimit),
        instrumentedModelCodeReadCalls: extendedVectors.reduce(
          (sum, vector) => sum + vector.executionMetrics.modelCodeReadCalls,
          0,
        ),
        instrumentedCrossChunkReadCalls: extendedVectors.reduce(
          (sum, vector) => sum + vector.executionMetrics.crossChunkReadCalls,
          0,
        ),
        derivedBoundaryTemporaryAllocations: extendedVectors.reduce(
          (sum, vector) => sum + vector.executionMetrics.boundaryTemporaryAllocations,
          0,
        ),
      } : {}),
    },
    limitations: [
      "The witness establishes public-byte reconstruction and execution agreement at one number-pinned block whose recorded hash was rechecked after all calls; it does not validate training data, accuracy, authorship, fairness, safety, or fitness for use.",
      "An eth_call is executed by the selected RPC provider and is not itself a consensus receipt.",
      "Only unrestricted view inference is compared; disabled or paid-view models require another authorization path.",
      ...(EXTENDED_MODE ? [
        "Historical eth_estimateGas values are node simulations rather than transaction receipts and can depend on RPC estimation policy; every unavailable or failed estimate is retained as an explicit outcome.",
        "Path/read/boundary counts instrument the checked source-level access pattern for each input. They are not opcode traces and do not isolate total gas by operation.",
        "The deterministic conformance vectors test exact execution agreement and split boundaries; they are not sampled from any model's training or target population and do not estimate predictive quality.",
      ] : []),
    ],
    models,
  };

  const output = resolve(
    positionalArgs[0] || (EXTENDED_MODE ? EXTENDED_OUTPUT : DEFAULT_OUTPUT),
  );
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  if (EXTENDED_MODE) {
    const derivedMarkdown = output.toLowerCase().endsWith(".json")
      ? `${output.slice(0, -5)}.md`
      : `${output}.md`;
    const markdownOutput = resolve(
      process.env.GL1F_WITNESS_MARKDOWN
      || (positionalArgs[0] ? derivedMarkdown : EXTENDED_MARKDOWN),
    );
    await mkdir(dirname(markdownOutput), { recursive: true });
    await writeFile(markdownOutput, renderExtendedMarkdown(result, output), "utf8");
    process.stdout.write(`Wrote ${markdownOutput}\n`);
  }
  process.stdout.write(`${JSON.stringify(result.summary, null, 2)}\n`);
  process.stdout.write(`Wrote ${output}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});

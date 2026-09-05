#!/usr/bin/env node

/**
 * Build a read-only, offline-replayable archive of the public GL1F deployment
 * at the publication block.
 *
 * The frozen live-chain witnesses are inputs, not outputs: this collector does
 * not modify them. It obtains registry state and runtime bytecode directly by
 * JSON-RPC, stores exact raw code bytes, reconstructs every active GL1F core,
 * checks each content commitment and canonical header, validates the archived
 * conformance corpus locally, and records the raw request/response bodies.
 *
 * Only read-only JSON-RPC methods are used:
 *   eth_chainId, eth_getBlockByNumber, eth_call, eth_getCode, eth_getProof.
 */

import {
  access,
  mkdir,
  readFile,
  readdir,
  rename,
  stat,
  writeFile,
} from "node:fs/promises";
import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";
import { basename, dirname, join, relative, resolve } from "node:path";
import {
  Interface,
  concat,
  getAddress,
  getBytes,
  hexlify,
  keccak256,
} from "ethers";

const CHAIN_ID = 29;
const BLOCK_NUMBER = 13_342_043;
const BLOCK_HASH = "0xffd825db1bb2534052a604db9584d361111d8bc9e19d753b0ee3861bf320d1b9";
const ARCHIVE_ID = "gl1f-genesisl1-29-block-13342043-ffd825db-v3";
const RPC_URL = process.env.GL1F_RPC_URL || "https://rpc.genesisl1.org";
const DEFAULT_OUTPUT = "benchmarks/results/live_chain_archive_v3";
const DEFAULT_WITNESS = "benchmarks/results/live_chain_witness_extended_v2.json";
const TRANSCRIPT_FILE = "rpc/collection-transcript.ndjson.gz";
const CHUNK_MAGIC = "0x474c3143";
const MAX_CHUNK_PAYLOAD = 24_572;
const INT32_MIN = -2_147_483_648;
const INT32_MAX = 2_147_483_647;

const ADDRESSES = Object.freeze({
  store: "0x9CdbC23392648Bd27B4A5eD09c0fEa9452454B54",
  registry: "0x33c9844F77a07e36B98f0FFf8201B8A8b02c2a69",
  nft: "0x44Dc1c54B8D579B42d78cC21cf8260DC0A3279fA",
  runtime: "0xD2fD0cf461a6cb56Fc08d9aEc120833D8E79044E",
  marketplace: "0xA9bfa0a719b7F73cE85CA1E7f23af626D383fB46",
});

const CONTRACT_SOURCES = Object.freeze({
  store: "contracts/ModelStore.sol",
  registry: "contracts/ModelRegistry.sol",
  nft: "contracts/ModelNFT.sol",
  runtime: "contracts/ForestRuntime.sol",
  marketplace: "contracts/ModelMarketplace.sol",
});

const registryInterface = new Interface([
  "function getModelSummary(uint256 tokenId) view returns (bool exists, bytes32 modelId, address tablePtr, uint16 nFeatures, uint16 nTrees, uint16 depth, int32 baseQ, uint8 pricingMode, uint256 feeWei, address feeRecipient, bool inferenceEnabled, address creator, uint32 tosVersionAccepted, string title, string description)",
  "function getModelBytesInfo(bytes32 modelId) view returns (address tablePtr, uint32 chunkSize, uint32 numChunks, uint32 totalBytes)",
  "function getModelRuntime(bytes32 modelId) view returns (address tablePtr, uint32 chunkSize, uint32 numChunks, uint32 totalBytes, uint16 nFeatures, uint16 nTrees, uint16 depth, int32 baseQ, uint32 scaleQ, bool inferenceEnabled, uint8 pricingMode, uint256 feeWei, address feeRecipient)",
  "function deployFeeWei() view returns (uint256)",
  "function sizeFeeWeiPerByte() view returns (uint256)",
  "function tosVersion() view returns (uint256)",
  "function activeLicenseId() view returns (uint256)",
  "function modelNFT() view returns (address)",
]);
const nftInterface = new Interface([
  "function totalMinted() view returns (uint256)",
]);
const runtimeInterface = new Interface([
  "function predictView(bytes32 modelId, bytes packedFeaturesQ) view returns (int256)",
  "function predictMultiView(bytes32 modelId, bytes packedFeaturesQ) view returns (int256[] logitsQ)",
]);

const argv = process.argv.slice(2);
const help = argv.includes("--help") || argv.includes("-h");
const outputArgument = argv.find((item) => !item.startsWith("-"));
const proofMode = !argv.includes("--no-proofs")
  && process.env.GL1F_ARCHIVE_PROOFS !== "0";
const outputDir = resolve(outputArgument || DEFAULT_OUTPUT);
const witnessPath = resolve(process.env.GL1F_ARCHIVE_WITNESS || DEFAULT_WITNESS);
const blockTag = `0x${BLOCK_NUMBER.toString(16)}`;
const batchSize = parsePositiveInteger(
  process.env.GL1F_RPC_BATCH_SIZE || "32",
  "GL1F_RPC_BATCH_SIZE",
  1,
  100,
);
const retryCount = parsePositiveInteger(
  process.env.GL1F_RPC_RETRIES || "6",
  "GL1F_RPC_RETRIES",
  1,
  12,
);
const timeoutMs = parsePositiveInteger(
  process.env.GL1F_RPC_TIMEOUT_MS || "90000",
  "GL1F_RPC_TIMEOUT_MS",
  1_000,
  600_000,
);

let nextRpcId = 1;
const transcript = [];

function parsePositiveInteger(value, name, minimum, maximum) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw new Error(`${name} must be an integer in ${minimum}..${maximum}`);
  }
  return parsed;
}

function sha256Hex(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function normalizeHex(value) {
  return String(value).toLowerCase();
}

function asNumber(value, context) {
  const number = Number(value);
  if (!Number.isSafeInteger(number)) {
    throw new Error(`${context} is not a safe integer: ${value}`);
  }
  return number;
}

function checkedEqual(actual, expected, context) {
  if (actual !== expected) {
    throw new Error(`${context}: got ${actual}; expected ${expected}`);
  }
}

function checkedHexEqual(actual, expected, context) {
  if (normalizeHex(actual) !== normalizeHex(expected)) {
    throw new Error(`${context}: got ${actual}; expected ${expected}`);
  }
}

function codeBytes(codeHex, context) {
  if (typeof codeHex !== "string" || !/^0x[0-9a-fA-F]*$/.test(codeHex)) {
    throw new Error(`${context}: invalid hex bytecode`);
  }
  const bytes = getBytes(codeHex);
  if (bytes.length === 0) throw new Error(`${context}: empty runtime bytecode`);
  return bytes;
}

function artifactDigest(bytes) {
  return {
    size: bytes.length,
    sha256: sha256Hex(bytes),
    keccak256: keccak256(bytes),
  };
}

function runtimeSlice(file, offset, bytes) {
  return {
    file,
    offset,
    ...artifactDigest(bytes),
  };
}

async function pathExists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function sleep(milliseconds) {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, milliseconds));
}

async function rpc(method, params, context, { allowRpcError = false } = {}) {
  let lastError;
  for (let attempt = 1; attempt <= retryCount; attempt++) {
    const id = nextRpcId++;
    const request = { jsonrpc: "2.0", id, method, params };
    const requestBody = JSON.stringify(request);
    let responseBody = null;
    let httpStatus = null;
    let transportError = null;
    try {
      const response = await fetch(RPC_URL, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: requestBody,
        signal: AbortSignal.timeout(timeoutMs),
      });
      httpStatus = response.status;
      responseBody = await response.text();
      transcript.push({
        sequence: id,
        attempt,
        context,
        httpStatus,
        requestBody,
        responseBody,
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      let decoded;
      try {
        decoded = JSON.parse(responseBody);
      } catch (error) {
        throw new Error(`invalid JSON-RPC response: ${error.message}`);
      }
      if (decoded.id !== id || decoded.jsonrpc !== "2.0") {
        throw new Error(`JSON-RPC envelope mismatch for request ${id}`);
      }
      if (decoded.error) {
        if (allowRpcError) return { ok: false, error: decoded.error };
        throw new Error(
          `JSON-RPC ${decoded.error.code}: ${decoded.error.message || "unknown error"}`,
        );
      }
      if (!Object.hasOwn(decoded, "result")) {
        throw new Error("JSON-RPC response has neither result nor error");
      }
      return { ok: true, result: decoded.result };
    } catch (error) {
      lastError = error;
      if (responseBody === null) {
        transportError = String(error?.message || error);
        transcript.push({
          sequence: id,
          attempt,
          context,
          httpStatus,
          requestBody,
          responseBody,
          transportError,
        });
      }
      if (attempt === retryCount) break;
      const delay = Math.min(4_000, 250 * (2 ** (attempt - 1)));
      process.stderr.write(
        `${context}: ${String(error?.message || error)}; retry `
        + `${attempt + 1}/${retryCount} in ${delay} ms\n`,
      );
      await sleep(delay);
    }
  }
  throw new Error(`${context}: ${String(lastError?.message || lastError)}`);
}

async function rpcResult(method, params, context) {
  const response = await rpc(method, params, context);
  return response.result;
}

async function rpcBatch(calls, context) {
  if (calls.length === 0) return [];
  let lastError;
  for (let attempt = 1; attempt <= retryCount; attempt++) {
    const requests = calls.map((call) => ({
      jsonrpc: "2.0",
      id: nextRpcId++,
      method: call.method,
      params: call.params,
    }));
    const requestBody = JSON.stringify(requests);
    const sequence = requests[0].id;
    let responseBody = null;
    let httpStatus = null;
    try {
      const response = await fetch(RPC_URL, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: requestBody,
        signal: AbortSignal.timeout(timeoutMs),
      });
      httpStatus = response.status;
      responseBody = await response.text();
      transcript.push({
        sequence,
        requestIds: requests.map((request) => request.id),
        attempt,
        context,
        httpStatus,
        requestBody,
        responseBody,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      let decoded;
      try {
        decoded = JSON.parse(responseBody);
      } catch (error) {
        throw new Error(`invalid batch JSON-RPC response: ${error.message}`);
      }
      if (!Array.isArray(decoded)) {
        throw new Error("batch JSON-RPC response is not an array");
      }
      const byId = new Map(decoded.map((item) => [item.id, item]));
      return requests.map((request, index) => {
        const item = byId.get(request.id);
        if (!item || item.jsonrpc !== "2.0") {
          throw new Error(`missing batch response for request ${request.id}`);
        }
        if (item.error) {
          throw new Error(
            `JSON-RPC ${item.error.code}: ${item.error.message || "unknown error"}`,
          );
        }
        if (!Object.hasOwn(item, "result")) {
          throw new Error(`batch response ${request.id} has no result`);
        }
        return {
          result: item.result,
          requestId: request.id,
          responseId: item.id,
          transcriptSequence: sequence,
          batchIndex: index,
        };
      });
    } catch (error) {
      lastError = error;
      if (responseBody === null) {
        transcript.push({
          sequence,
          requestIds: requests.map((request) => request.id),
          attempt,
          context,
          httpStatus,
          requestBody,
          responseBody,
          transportError: String(error?.message || error),
        });
      }
      if (attempt === retryCount) break;
      const delay = Math.min(8_000, 500 * (2 ** (attempt - 1)));
      process.stderr.write(
        `${context}: ${String(error?.message || error)}; retry `
        + `${attempt + 1}/${retryCount} in ${delay} ms\n`,
      );
      await sleep(delay);
    }
  }
  throw new Error(`${context}: ${String(lastError?.message || lastError)}`);
}

async function rpcBatches(calls, context) {
  const output = [];
  const totalBatches = Math.ceil(calls.length / batchSize);
  for (let offset = 0; offset < calls.length; offset += batchSize) {
    const batch = calls.slice(offset, offset + batchSize);
    const batchNumber = Math.floor(offset / batchSize) + 1;
    output.push(...await rpcBatch(
      batch,
      `${context} batch ${batchNumber}/${totalBatches}`,
    ));
  }
  return output;
}

async function rpcBatchResults(calls, context) {
  return (await rpcBatches(calls, context)).map((item) => item.result);
}

async function ethCall(address, contractInterface, functionName, args, context) {
  const data = contractInterface.encodeFunctionData(functionName, args);
  const result = await rpcResult(
    "eth_call",
    [{ to: address, data }, blockTag],
    context,
  );
  return contractInterface.decodeFunctionResult(functionName, result);
}

function stripChunkMagic(runtimeBytes, context) {
  if (runtimeBytes.length < 4 || hexlify(runtimeBytes.slice(0, 4)) !== CHUNK_MAGIC) {
    throw new Error(`${context}: runtime code does not begin with GL1C`);
  }
  return runtimeBytes.slice(4);
}

function decodePointerTable(runtimeBytes, expectedCount) {
  const payload = stripChunkMagic(runtimeBytes, "pointer table");
  checkedEqual(
    payload.length,
    expectedCount * 32,
    "pointer table payload length",
  );
  const pointers = [];
  for (let index = 0; index < expectedCount; index++) {
    const word = payload.slice(index * 32, (index + 1) * 32);
    if (word.slice(0, 12).some((byte) => byte !== 0)) {
      throw new Error(`pointer table word ${index} has non-zero high bytes`);
    }
    const pointer = getAddress(hexlify(word.slice(12)));
    if (pointer === "0x0000000000000000000000000000000000000000") {
      throw new Error(`pointer table word ${index} is the zero address`);
    }
    pointers.push(pointer);
  }
  return pointers;
}

function readU16(view, offset) {
  return view.getUint16(offset, true);
}

function readU32(view, offset) {
  return view.getUint32(offset, true);
}

function readI32(view, offset) {
  return view.getInt32(offset, true);
}

function parseAndValidateCore(core, context) {
  if (core.length < 24) throw new Error(`${context}: core shorter than 24 bytes`);
  if (Buffer.from(core.slice(0, 4)).toString("ascii") !== "GL1F") {
    throw new Error(`${context}: missing GL1F magic`);
  }
  const view = new DataView(core.buffer, core.byteOffset, core.byteLength);
  const version = core[4];
  checkedEqual(core[5], 0, `${context}: header reserved byte`);
  const nFeatures = readU16(view, 6);
  const depth = readU16(view, 8);
  if (nFeatures < 1) throw new Error(`${context}: nFeatures must be positive`);
  if (depth < 1 || depth > 12) {
    throw new Error(
      `${context}: depth ${depth} is outside publication/deployment profile 1..12`,
    );
  }
  const leaves = 2 ** depth;
  const internal = leaves - 1;
  const perTreeBytes = internal * 8 + leaves * 4;
  const scaleQ = readU32(view, 18);
  if (scaleQ < 1) throw new Error(`${context}: scaleQ must be positive`);

  let treesPerOutput;
  let outputs;
  let baseQ;
  let treesOffset;
  if (version === 1) {
    treesPerOutput = readU32(view, 10);
    outputs = 1;
    baseQ = [readI32(view, 14)];
    treesOffset = 24;
    checkedEqual(readU16(view, 22), 0, `${context}: v1 reserved field`);
  } else if (version === 2) {
    treesPerOutput = readU32(view, 10);
    if (treesPerOutput < 1) {
      throw new Error(`${context}: v2 treesPerOutput must be positive`);
    }
    checkedEqual(readI32(view, 14), 0, `${context}: v2 reserved field`);
    outputs = readU16(view, 22);
    if (outputs < 2) throw new Error(`${context}: v2 outputs must be at least two`);
    treesOffset = 24 + outputs * 4;
    if (core.length < treesOffset) {
      throw new Error(`${context}: truncated v2 base vector`);
    }
    baseQ = Array.from(
      { length: outputs },
      (_, index) => readI32(view, 24 + index * 4),
    );
  } else {
    throw new Error(`${context}: unsupported GL1F version ${version}`);
  }

  const totalTrees = treesPerOutput * outputs;
  const expectedBytes = treesOffset + totalTrees * perTreeBytes;
  checkedEqual(core.length, expectedBytes, `${context}: exact core length`);

  let offset = treesOffset;
  for (let tree = 0; tree < totalTrees; tree++) {
    for (let node = 0; node < internal; node++) {
      const feature = readU16(view, offset);
      if (feature >= nFeatures) {
        throw new Error(
          `${context}: tree ${tree}, node ${node} references feature ${feature}`,
        );
      }
      checkedEqual(
        readU16(view, offset + 6),
        0,
        `${context}: tree ${tree}, node ${node} reserved field`,
      );
      offset += 8;
    }
    offset += leaves * 4;
  }
  checkedEqual(offset, core.length, `${context}: validated tree cursor`);

  return {
    version,
    nFeatures,
    depth,
    nTrees: totalTrees,
    totalTrees,
    treesPerOutput,
    outputs,
    baseQ,
    scaleQ,
    treesOffset,
    perTreeBytes,
    coreBytes: core.length,
  };
}

function unpackInt32Vector(packedHex, nFeatures, context) {
  const bytes = getBytes(packedHex);
  checkedEqual(bytes.length, nFeatures * 4, `${context}: packed input length`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from({ length: nFeatures }, (_, index) =>
    view.getInt32(index * 4, true));
}

function evaluateCore(core, header, featuresQ) {
  const view = new DataView(core.buffer, core.byteOffset, core.byteLength);
  const leaves = 2 ** header.depth;
  const internal = leaves - 1;
  const traverse = (treeOffset) => {
    let node = 0;
    for (let level = 0; level < header.depth; level++) {
      const offset = treeOffset + node * 8;
      const feature = readU16(view, offset);
      const thresholdQ = readI32(view, offset + 2);
      node = featuresQ[feature] > thresholdQ ? node * 2 + 2 : node * 2 + 1;
    }
    const leafIndex = node - internal;
    return readI32(
      view,
      treeOffset + internal * 8 + leafIndex * 4,
    );
  };

  const output = [];
  for (let outputIndex = 0; outputIndex < header.outputs; outputIndex++) {
    let accumulator = BigInt(header.baseQ[outputIndex]);
    for (let tree = 0; tree < header.treesPerOutput; tree++) {
      const serializedTree = outputIndex * header.treesPerOutput + tree;
      accumulator += BigInt(
        traverse(header.treesOffset + serializedTree * header.perTreeBytes),
      );
    }
    output.push(String(accumulator));
  }
  return output;
}

function normalizedVectors(witnessModel, core, header, context) {
  if (!witnessModel?.conformanceStudy?.vectors) {
    throw new Error(`${context}: extended witness has no conformance vectors`);
  }
  return witnessModel.conformanceStudy.vectors.map((vector) => {
    const packedBytes = getBytes(vector.packedFeaturesQHex).length;
    checkedEqual(packedBytes, vector.packedBytes, `${context}/${vector.vectorId}: packedBytes`);
    const featuresQ = unpackInt32Vector(
      vector.packedFeaturesQHex,
      header.nFeatures,
      `${context}/${vector.vectorId}`,
    );
    for (const value of featuresQ) {
      if (value < INT32_MIN || value > INT32_MAX) {
        throw new Error(`${context}/${vector.vectorId}: feature outside int32`);
      }
    }
    const localPredictionQ = evaluateCore(core, header, featuresQ);
    if (JSON.stringify(localPredictionQ) !== JSON.stringify(vector.localPredictionQ)) {
      throw new Error(`${context}/${vector.vectorId}: local witness replay mismatch`);
    }
    const chainPredictionQ = vector.evmRead?.outputQ ?? null;
    const exactMatch = chainPredictionQ === null
      ? null
      : JSON.stringify(localPredictionQ) === JSON.stringify(chainPredictionQ);
    if (exactMatch !== vector.evmRead?.exactMatch) {
      throw new Error(`${context}/${vector.vectorId}: exactMatch witness mismatch`);
    }
    return {
      vectorId: vector.vectorId,
      sourceRule: vector.sourceRule,
      thresholdProbe: vector.thresholdProbe,
      packedFeaturesQHex: vector.packedFeaturesQHex,
      packedBytes,
      expectedOutputQ: localPredictionQ,
      localPredictionQ,
      sourceWitnessChainPredictionQ: chainPredictionQ,
      sourceWitnessStatus: vector.evmRead?.status ?? "missing",
      sourceWitnessExactMatch: exactMatch,
    };
  });
}

async function executeArchivedChainVectors(vectors, modelId, header, context) {
  const method = header.version === 1 ? "predictView" : "predictMultiView";
  const calls = vectors.map((vector) => {
    const calldata = runtimeInterface.encodeFunctionData(
      method,
      [modelId, vector.packedFeaturesQHex],
    );
    return {
      method: "eth_call",
      params: [{ to: ADDRESSES.runtime, data: calldata }, blockTag],
      calldata,
    };
  });
  const responses = await rpcBatches(
    calls.map((call) => ({ method: call.method, params: call.params })),
    `${context} archived conformance eth_call`,
  );
  checkedEqual(
    responses.length,
    vectors.length,
    `${context}: archived conformance response count`,
  );

  return vectors.map((vector, index) => {
    const call = calls[index];
    const response = responses[index];
    const decoded = runtimeInterface.decodeFunctionResult(method, response.result);
    const chainPredictionQ = header.version === 1
      ? [String(decoded[0])]
      : decoded[0].map(String);
    const exactMatch =
      JSON.stringify(chainPredictionQ) === JSON.stringify(vector.localPredictionQ);
    if (!exactMatch) {
      throw new Error(`${context}/${vector.vectorId}: archived chain output mismatch`);
    }
    const sourceWitnessMatches =
      JSON.stringify(chainPredictionQ)
      === JSON.stringify(vector.sourceWitnessChainPredictionQ);
    if (!sourceWitnessMatches) {
      throw new Error(
        `${context}/${vector.vectorId}: archived call differs from source witness`,
      );
    }
    return {
      ...vector,
      chainPredictionQ,
      status: "compared",
      exactMatch,
      archiveChainCall: {
        transcriptFile: TRANSCRIPT_FILE,
        transcriptSequence: response.transcriptSequence,
        requestId: response.requestId,
        responseId: response.responseId,
        batchIndex: response.batchIndex,
        rpcMethod: "eth_call",
        contractMethod: method,
        to: ADDRESSES.runtime,
        blockTag,
        calldataBytes: getBytes(call.calldata).length,
        calldataKeccak256: keccak256(call.calldata),
        rawResultHex: response.result,
        decodedOutputQ: chainPredictionQ,
        matchesLocalPrediction: exactMatch,
        matchesEmbeddedSourceWitness: sourceWitnessMatches,
      },
    };
  });
}

async function writeArtifact(stagingDir, relativePath, bytes) {
  const destination = join(stagingDir, relativePath);
  await mkdir(dirname(destination), { recursive: true });
  await writeFile(destination, bytes);
  return artifactDigest(bytes);
}

async function listFiles(root) {
  const files = [];
  async function walk(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      const fullPath = join(directory, entry.name);
      if (entry.isDirectory()) {
        await walk(fullPath);
      } else if (entry.isFile()) {
        files.push(relative(root, fullPath).replaceAll("\\", "/"));
      }
    }
  }
  await walk(root);
  return files;
}

async function fileInventory(root, excluded = new Set()) {
  const inventory = [];
  for (const file of await listFiles(root)) {
    if (excluded.has(file)) continue;
    const bytes = await readFile(join(root, file));
    inventory.push({ path: file, size: bytes.length, sha256: sha256Hex(bytes) });
  }
  return inventory;
}

function renderReadme(manifest) {
  return `# GL1F pinned live-chain state archive v3

This directory is a read-only evidence archive for GenesisL1 chain ID
\`${manifest.chain.chainId}\` at block \`${manifest.chain.blockNumber}\`
(\`${manifest.chain.blockHash}\`). It contains the exact reconstructed GL1F
cores, exact runtime code for every pointer table and data chunk, exact runtime
code for the five named application contracts, a complete
\`eth_getBlockByNumber(..., true)\` result, a deterministic replay corpus, and
a gzip-compressed raw JSON-RPC request/response-body transcript. The frozen
extended witness supplying the recorded 108 chain outputs is embedded under
\`source/\`. The collector independently recomputes every corresponding local
output from the archived cores, reissues all 108 historical runtime calls,
binds their exact request and result bytes to transcript entries, and requires
both observations to agree.

The exact collector source is embedded at
\`${manifest.collector.archivedSourceFile}\` and bound by SHA-256
\`${manifest.collector.archivedSourceSha256}\`. The repository revision and
worktree state, when recorded, describe collection-time provenance; the
embedded file is authoritative if those fields report a dirty or unavailable
revision.

## Offline reconstruction

For each model, read the byte range named by every ordered \`chunks[]\` entry
from its \`storageRuntimeFile\`. Each range is exact EVM runtime code and must
begin with ASCII \`GL1C\`; concatenate the bytes after those four-byte prefixes
in chunk order. The result must have \`totalBytes\` bytes and match both the
archived \`core.file\` and \`modelId = keccak256(core)\`. The pointer-table
range starts with \`GL1C\` and its payload contains exactly the ordered chunk
addresses as zero-extended 32-byte words.

## Trust boundary

The archive is provider-attested historical state obtained from
\`${manifest.chain.rpcEndpoint}\`. The block hash is checked before and after
collection. The provider exposes \`eth_getProof\`, so account-proof objects are
archived for every contract/table/chunk address and their reported
\`codeHash\` values are compared with the collected bytecode. This collector
does not implement verification of GenesisL1's returned proof encoding against
the block state root, and no contract storage-slot proofs are requested.
Consequently, the archive removes dependence on future archive-node
availability and enables exact offline replay, but does not by itself replace
independent consensus/header verification or proof verification.

No transaction-submission method is used. See \`manifest.json\` for the
machine-readable schema and \`SHA256SUMS\` for a whole-directory checksum
inventory.
`;
}

async function main() {
  if (help) {
    process.stdout.write(
      "Usage: node benchmarks/archive_live_chain_state.mjs [OUTPUT_DIR] [--no-proofs]\n"
      + "\n"
      + `Default output: ${DEFAULT_OUTPUT}\n`
      + `Pinned state: chain ${CHAIN_ID}, block ${BLOCK_NUMBER} (${BLOCK_HASH})\n`,
    );
    return;
  }

  if (await pathExists(outputDir)) {
    throw new Error(
      `refusing to overwrite existing archive directory ${outputDir}; choose another path`,
    );
  }
  const parent = dirname(outputDir);
  await mkdir(parent, { recursive: true });
  const stagingDir = join(parent, `.${basename(outputDir)}.collecting-${process.pid}`);
  if (await pathExists(stagingDir)) {
    throw new Error(`staging directory already exists: ${stagingDir}`);
  }
  await mkdir(stagingDir, { recursive: false });

  const collectorSourceBytes = await readFile(new URL(import.meta.url));
  const archivedCollectorFile = "source/archive_live_chain_state.mjs";
  const collectorSourceDigest = await writeArtifact(
    stagingDir,
    archivedCollectorFile,
    collectorSourceBytes,
  );

  const witnessBytes = await readFile(witnessPath);
  const witness = JSON.parse(witnessBytes.toString("utf8"));
  checkedEqual(witness.chainId, CHAIN_ID, "extended witness chain ID");
  checkedEqual(witness.block.number, BLOCK_NUMBER, "extended witness block number");
  checkedHexEqual(witness.block.hash, BLOCK_HASH, "extended witness block hash");
  checkedEqual(witness.summary.activeModels, 12, "extended witness active model count");
  const archivedWitnessFile = "source/live_chain_witness_extended_v2.json";
  const archivedWitnessDigest = await writeArtifact(
    stagingDir,
    archivedWitnessFile,
    witnessBytes,
  );
  checkedEqual(
    archivedWitnessDigest.sha256,
    sha256Hex(witnessBytes),
    "embedded source witness SHA-256",
  );

  const chainIdHex = await rpcResult("eth_chainId", [], "chain ID");
  checkedEqual(Number(BigInt(chainIdHex)), CHAIN_ID, "RPC chain ID");
  const selectedBlock = await rpcResult(
    "eth_getBlockByNumber",
    [blockTag, true],
    "full selected block",
  );
  if (!selectedBlock) throw new Error(`block ${BLOCK_NUMBER} was not returned`);
  checkedHexEqual(selectedBlock.hash, BLOCK_HASH, "selected block hash");
  const blockFile = "block/full.json";
  await writeArtifact(stagingDir, blockFile, jsonBytes(selectedBlock));

  const contractRecords = [];
  const runtimeByAddress = new Map();
  for (const [role, address] of Object.entries(ADDRESSES)) {
    const runtimeHex = await rpcResult(
      "eth_getCode",
      [address, blockTag],
      `${role} contract runtime`,
    );
    const runtime = codeBytes(runtimeHex, `${role} contract runtime`);
    const file = `contracts/${role}.runtime.bin`;
    await writeArtifact(stagingDir, file, runtime);
    runtimeByAddress.set(normalizeHex(address), runtime);
    contractRecords.push({
      role,
      address: getAddress(address),
      source: CONTRACT_SOURCES[role],
      runtime: { file, ...artifactDigest(runtime) },
    });
  }

  const [totalMintedResult] = await ethCall(
    ADDRESSES.nft,
    nftInterface,
    "totalMinted",
    [],
    "NFT totalMinted",
  );
  const totalMinted = asNumber(totalMintedResult, "totalMinted");
  checkedEqual(totalMinted, 12, "pinned totalMinted");

  const models = [];
  const allCodeAddresses = new Set(
    Object.values(ADDRESSES).map((address) => normalizeHex(address)),
  );
  let totalCoreBytes = 0;
  let totalChunkCount = 0;
  let totalVectorCount = 0;
  for (let tokenId = 1; tokenId <= totalMinted; tokenId++) {
    const context = `token ${tokenId}`;
    const summary = await ethCall(
      ADDRESSES.registry,
      registryInterface,
      "getModelSummary",
      [tokenId],
      `${context} summary`,
    );
    if (!summary.exists) throw new Error(`${context}: inactive at pinned block`);
    const modelId = String(summary.modelId);
    const info = await ethCall(
      ADDRESSES.registry,
      registryInterface,
      "getModelBytesInfo",
      [modelId],
      `${context} byte manifest`,
    );
    const runtimeMeta = await ethCall(
      ADDRESSES.registry,
      registryInterface,
      "getModelRuntime",
      [modelId],
      `${context} runtime metadata`,
    );

    const tablePtr = getAddress(info.tablePtr);
    const chunkSize = asNumber(info.chunkSize, `${context} chunkSize`);
    const numChunks = asNumber(info.numChunks, `${context} numChunks`);
    const totalBytes = asNumber(info.totalBytes, `${context} totalBytes`);
    if (chunkSize < 4 || chunkSize > MAX_CHUNK_PAYLOAD) {
      throw new Error(`${context}: noncanonical chunkSize ${chunkSize}`);
    }
    checkedEqual(
      numChunks,
      Math.ceil(totalBytes / chunkSize),
      `${context}: canonical numChunks`,
    );
    const tableRuntimeHex = await rpcResult(
      "eth_getCode",
      [tablePtr, blockTag],
      `${context} pointer table runtime`,
    );
    const tableRuntime = codeBytes(tableRuntimeHex, `${context} pointer table`);
    const chunkPointers = decodePointerTable(tableRuntime, numChunks);
    const chunkRuntimeHexes = await rpcBatchResults(
      chunkPointers.map((address) => ({
        method: "eth_getCode",
        params: [address, blockTag],
      })),
      `${context} chunk runtimes`,
    );
    const chunkRuntimes = chunkRuntimeHexes.map((runtimeHex, index) =>
      codeBytes(runtimeHex, `${context} chunk ${index}`));

    const payloads = chunkRuntimes.map((runtime, index) => {
      const payload = stripChunkMagic(runtime, `${context} chunk ${index}`);
      const expected = index + 1 < numChunks
        ? chunkSize
        : totalBytes - chunkSize * (numChunks - 1);
      checkedEqual(payload.length, expected, `${context} chunk ${index} payload length`);
      return payload;
    });
    const core = getBytes(concat(payloads));
    checkedEqual(core.length, totalBytes, `${context} reconstructed bytes`);
    const computedModelId = keccak256(core);
    checkedHexEqual(computedModelId, modelId, `${context} content commitment`);
    const header = parseAndValidateCore(core, context);

    const registry = {
      modelId,
      tablePtr,
      chunkSize,
      numChunks,
      totalBytes,
      nFeatures: asNumber(runtimeMeta.nFeatures, `${context} registry nFeatures`),
      nTrees: asNumber(runtimeMeta.nTrees, `${context} registry nTrees`),
      depth: asNumber(runtimeMeta.depth, `${context} registry depth`),
      baseQ: asNumber(runtimeMeta.baseQ, `${context} registry baseQ`),
      scaleQ: asNumber(runtimeMeta.scaleQ, `${context} registry scaleQ`),
      inferenceEnabled: Boolean(runtimeMeta.inferenceEnabled),
      pricingMode: asNumber(runtimeMeta.pricingMode, `${context} pricingMode`),
      feeWei: String(runtimeMeta.feeWei),
      feeRecipient: getAddress(runtimeMeta.feeRecipient),
      creator: getAddress(summary.creator),
      tosVersionAccepted: asNumber(
        summary.tosVersionAccepted,
        `${context} tosVersionAccepted`,
      ),
      title: summary.title,
      description: summary.description,
    };
    const registryHeaderAgreement = {
      nFeatures: registry.nFeatures === header.nFeatures,
      nTrees: registry.nTrees === header.totalTrees,
      depth: registry.depth === header.depth,
      baseQ: header.version === 1
        ? registry.baseQ === header.baseQ[0]
        : registry.baseQ === 0,
      scaleQ: registry.scaleQ === header.scaleQ,
      tablePtr: normalizeHex(runtimeMeta.tablePtr) === normalizeHex(tablePtr),
      totalBytes: asNumber(runtimeMeta.totalBytes, `${context} runtime totalBytes`) === totalBytes,
      numChunks: asNumber(runtimeMeta.numChunks, `${context} runtime numChunks`) === numChunks,
      chunkSize: asNumber(runtimeMeta.chunkSize, `${context} runtime chunkSize`) === chunkSize,
    };
    if (!Object.values(registryHeaderAgreement).every(Boolean)) {
      throw new Error(`${context}: registry/header relation failed`);
    }

    const witnessModel = witness.models.find((model) => model.tokenId === tokenId);
    if (!witnessModel?.active) throw new Error(`${context}: absent from extended witness`);
    checkedHexEqual(witnessModel.modelId, modelId, `${context}: witness modelId`);
    checkedHexEqual(witnessModel.tablePtr, tablePtr, `${context}: witness tablePtr`);
    checkedEqual(witnessModel.totalBytes, totalBytes, `${context}: witness totalBytes`);
    checkedEqual(witnessModel.numChunks, numChunks, `${context}: witness numChunks`);
    checkedEqual(witnessModel.chunkSize, chunkSize, `${context}: witness chunkSize`);
    const sourceVectors = normalizedVectors(witnessModel, core, header, context);
    const vectors = await executeArchivedChainVectors(
      sourceVectors,
      modelId,
      header,
      context,
    );

    const tokenDirectory = `models/token-${String(tokenId).padStart(2, "0")}`;
    const coreFile = `${tokenDirectory}/core.gl1f`;
    await writeArtifact(stagingDir, coreFile, core);
    const storageRuntimeFile = `${tokenDirectory}/storage-runtime.bin`;
    const storageRuntime = getBytes(concat([tableRuntime, ...chunkRuntimes]));
    await writeArtifact(stagingDir, storageRuntimeFile, storageRuntime);

    let offset = 0;
    const tableRuntimeRecord = runtimeSlice(
      storageRuntimeFile,
      offset,
      tableRuntime,
    );
    offset += tableRuntime.length;
    const chunks = chunkRuntimes.map((runtime, index) => {
      const record = {
        index,
        address: chunkPointers[index],
        payloadSize: runtime.length - 4,
        runtime: runtimeSlice(storageRuntimeFile, offset, runtime),
      };
      offset += runtime.length;
      return record;
    });
    checkedEqual(offset, storageRuntime.length, `${context}: storage container index`);

    runtimeByAddress.set(normalizeHex(tablePtr), tableRuntime);
    allCodeAddresses.add(normalizeHex(tablePtr));
    for (let index = 0; index < chunkPointers.length; index++) {
      runtimeByAddress.set(normalizeHex(chunkPointers[index]), chunkRuntimes[index]);
      allCodeAddresses.add(normalizeHex(chunkPointers[index]));
    }

    models.push({
      tokenId,
      active: true,
      modelId,
      computedModelId,
      contentCommitmentMatches: true,
      coreFile,
      core: { file: coreFile, ...artifactDigest(core) },
      tablePtr,
      orderedChunkPointers: chunkPointers,
      chunkPointers,
      chunkSize,
      numChunks,
      totalBytes,
      storageRuntimeFile,
      tables: [{
        index: 0,
        address: tablePtr,
        pointerCount: chunkPointers.length,
        runtime: tableRuntimeRecord,
      }],
      chunks,
      registry,
      header,
      registryHeaderAgreement,
      vectors,
    });
    totalCoreBytes += core.length;
    totalChunkCount += chunks.length;
    totalVectorCount += vectors.length;
    process.stderr.write(
      `Archived token ${tokenId}/${totalMinted}: ${core.length} bytes, `
      + `${chunks.length} chunks, ${vectors.length} replay vectors\n`,
    );
  }

  let proofAvailability = {
    requested: proofMode,
    available: false,
    collection: "not requested",
    accountProofs: 0,
    storageProofsRequested: 0,
    codeHashesMatched: 0,
    file: null,
  };
  const proofByAddress = new Map();
  if (proofMode) {
    const proofProbe = await rpc(
      "eth_getProof",
      [ADDRESSES.registry, [], blockTag],
      "eth_getProof availability probe",
      { allowRpcError: true },
    );
    if (!proofProbe.ok) {
      proofAvailability = {
        ...proofAvailability,
        collection: "RPC method unavailable",
        rpcError: proofProbe.error,
      };
    } else {
      proofByAddress.set(normalizeHex(ADDRESSES.registry), proofProbe.result);
      const remaining = [...allCodeAddresses]
        .filter((address) => address !== normalizeHex(ADDRESSES.registry))
        .sort();
      const proofs = await rpcBatchResults(
        remaining.map((address) => ({
          method: "eth_getProof",
          params: [address, [], blockTag],
        })),
        "account proofs",
      );
      for (let index = 0; index < remaining.length; index++) {
        proofByAddress.set(remaining[index], proofs[index]);
      }
      let matched = 0;
      for (const [address, proof] of proofByAddress.entries()) {
        const runtime = runtimeByAddress.get(address);
        if (!runtime) throw new Error(`proof has no runtime bytes for ${address}`);
        checkedHexEqual(
          proof.codeHash,
          keccak256(runtime),
          `account proof codeHash ${address}`,
        );
        if ((proof.storageProof || []).length !== 0) {
          throw new Error(`unexpected storageProof entries for ${address}`);
        }
        matched += 1;
      }
      const proofFile = "proofs/account-proofs.json.gz";
      const proofRecords = [...proofByAddress.entries()]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([address, proof]) => ({ address: getAddress(address), proof }));
      await writeArtifact(
        stagingDir,
        proofFile,
        gzipSync(jsonBytes(proofRecords), { level: 9, mtime: 0 }),
      );
      proofAvailability = {
        requested: true,
        available: true,
        collection:
          "one empty-storage-key eth_getProof result per unique code address",
        accountProofs: proofRecords.length,
        storageProofsRequested: 0,
        codeHashesMatched: matched,
        file: proofFile,
        compression: "gzip level 9, mtime 0",
        verification:
          "reported codeHash compared to keccak256(runtime); proof path not independently verified",
      };
    }
  }

  const [
    registryConfiguredNft,
    deployFeeWei,
    sizeFeeWeiPerByte,
    tosVersion,
    activeLicenseId,
  ] = await Promise.all([
    ethCall(
      ADDRESSES.registry,
      registryInterface,
      "modelNFT",
      [],
      "registry configured NFT",
    ),
    ethCall(
      ADDRESSES.registry,
      registryInterface,
      "deployFeeWei",
      [],
      "registry deploy fee",
    ),
    ethCall(
      ADDRESSES.registry,
      registryInterface,
      "sizeFeeWeiPerByte",
      [],
      "registry size fee",
    ),
    ethCall(
      ADDRESSES.registry,
      registryInterface,
      "tosVersion",
      [],
      "registry ToS version",
    ),
    ethCall(
      ADDRESSES.registry,
      registryInterface,
      "activeLicenseId",
      [],
      "registry active license",
    ),
  ]);
  const registryConfiguredNftAddress = getAddress(registryConfiguredNft[0]);
  checkedHexEqual(
    registryConfiguredNftAddress,
    ADDRESSES.nft,
    "registry modelNFT topology",
  );

  // This is deliberately the final RPC operation in the collection.
  const finalBlock = await rpcResult(
    "eth_getBlockByNumber",
    [blockTag, false],
    "final block-hash recheck",
  );
  checkedHexEqual(finalBlock.hash, BLOCK_HASH, "final block hash");

  transcript.sort((left, right) => left.sequence - right.sequence);
  const transcriptLines =
    transcript.map((entry) => JSON.stringify(entry)).join("\n") + "\n";
  await writeArtifact(
    stagingDir,
    TRANSCRIPT_FILE,
    gzipSync(Buffer.from(transcriptLines, "utf8"), { level: 9, mtime: 0 }),
  );
  const methodCounts = {};
  for (const entry of transcript) {
    const request = JSON.parse(entry.requestBody);
    for (const item of Array.isArray(request) ? request : [request]) {
      methodCounts[item.method] = (methodCounts[item.method] || 0) + 1;
    }
  }

  const manifestSkeleton = {
    schema: "gl1f-live-chain-archive/v3",
    archiveId: ARCHIVE_ID,
    generatedAt: new Date().toISOString(),
    collector: {
      script: "benchmarks/archive_live_chain_state.mjs",
      archivedSourceFile: archivedCollectorFile,
      archivedSourceBytes: collectorSourceDigest.size,
      archivedSourceSha256: collectorSourceDigest.sha256,
      repositoryRevision:
        process.env.GL1F_COLLECTOR_SOURCE_REVISION || null,
      repositoryWorktreeState:
        process.env.GL1F_COLLECTOR_WORKTREE_STATE
        || "not recorded; the embedded source file and SHA-256 are authoritative",
      mode: "read-only historical-state archive",
      node: process.version,
      rpcBatchSize: batchSize,
      rpcRetries: retryCount,
      rpcTimeoutMs: timeoutMs,
    },
    chain: {
      name: "GenesisL1",
      chainId: CHAIN_ID,
      blockNumber: BLOCK_NUMBER,
      blockTag,
      blockHash: BLOCK_HASH,
      blockTimestamp: Number(BigInt(selectedBlock.timestamp)),
      blockStateRoot: selectedBlock.stateRoot,
      fullBlockFile: blockFile,
      fullTransactionsRequested: true,
      rpcEndpoint: RPC_URL,
      blockHashCheckedBeforeAndAfterCollection: true,
      trustModel:
        "provider-attested archive; exact bytes and provider proofs retained, "
        + "but returned proof paths and consensus headers are not independently verified",
    },
    readOnlyRpc: {
      methodsUsed: Object.keys(methodCounts).sort(),
      methodRequestCounts: Object.fromEntries(
        Object.entries(methodCounts).sort(([left], [right]) => left.localeCompare(right)),
      ),
      prohibitedAndUnusedMethods: [
        "eth_sendTransaction",
        "eth_sendRawTransaction",
      ],
      rawTranscript: {
        file: TRANSCRIPT_FILE,
        format:
          "gzip-compressed NDJSON; each line stores exact JSON requestBody and responseBody strings",
        compression: "gzip level 9, mtime 0",
        requestAttempts: transcript.length,
      },
    },
    sourceWitness: {
      file: archivedWitnessFile,
      originalRepositoryPath:
        relative(resolve("."), witnessPath).replaceAll("\\", "/"),
      schema: witness.schema,
      sha256: sha256Hex(witnessBytes),
      role:
        "source of the deterministic vector corpus and a prior historical-chain "
        + "observation; the source is embedded, every local expected output is "
        + "recomputed, and all 108 chain calls are reissued into this archive's transcript",
    },
    contracts: contractRecords,
    registrySnapshot: {
      totalMinted,
      activeModels: models.length,
      modelNFT: registryConfiguredNftAddress,
      modelNFTMatchesDeploymentAddress: true,
      deployFeeWei: String(deployFeeWei[0]),
      sizeFeeWeiPerByte: String(sizeFeeWeiPerByte[0]),
      tosVersion: asNumber(tosVersion[0], "tosVersion"),
      activeLicenseId: asNumber(activeLicenseId[0], "activeLicenseId"),
    },
    proofEvidence: proofAvailability,
    summary: {
      activeModels: models.length,
      reconstructedCoreBytes: totalCoreBytes,
      pointerTables: models.length,
      dataChunks: totalChunkCount,
      uniqueCodeAddresses: allCodeAddresses.size,
      contentCommitmentsMatched: models.filter(
        (model) => model.contentCommitmentMatches,
      ).length,
      registryHeaderRelationsMatched: models.filter((model) =>
        Object.values(model.registryHeaderAgreement).every(Boolean)).length,
      replayVectors: totalVectorCount,
      historicalRuntimeCallsReissued: totalVectorCount,
      locallyRecomputedVectorOutputsMatched: models.reduce(
        (sum, model) => sum + model.vectors.filter(
          (vector) => JSON.stringify(vector.expectedOutputQ)
            === JSON.stringify(vector.localPredictionQ),
        ).length,
        0,
      ),
      archivedChainOutputsExactlyMatched: models.reduce(
        (sum, model) => sum + model.vectors.filter(
          (vector) => vector.exactMatch === true,
        ).length,
        0,
      ),
      archivedChainOutputsMatchingEmbeddedWitness: models.reduce(
        (sum, model) => sum + model.vectors.filter(
          (vector) => vector.archiveChainCall.matchesEmbeddedSourceWitness,
        ).length,
        0,
      ),
    },
    reconstruction:
      "For each model, parse its one table runtime and ordered chunk runtime "
      + "ranges from storageRuntimeFile; require GL1C prefixes, concatenate "
      + "chunk payloads, require totalBytes, then compare core and Keccak-256.",
    limitations: [
      "The historical state and block JSON are responses from the named RPC provider.",
      "Account proofs are retained when eth_getProof is available, but this collector does not verify the provider-specific proof encoding or consensus header chain.",
      "No contract storage-slot proof is requested; registry values are archived as raw historical eth_call responses in the transcript.",
      "The embedded witness's earlier run did not retain raw response bodies; this archive therefore reissues all 108 historical runtime calls and retains their exact request/response bodies.",
      "The replay corpus establishes exact agreement only for the archived inputs, not predictive validity.",
    ],
    models,
    files: {},
  };

  const readmeFile = "README.md";
  await writeArtifact(
    stagingDir,
    readmeFile,
    Buffer.from(renderReadme(manifestSkeleton), "utf8"),
  );
  manifestSkeleton.files = Object.fromEntries(
    (await fileInventory(
      stagingDir,
      new Set(["manifest.json", "SHA256SUMS"]),
    )).map((file) => [
      file.path,
      { size: file.size, sha256: file.sha256 },
    ]),
  );

  const manifestFile = "manifest.json";
  await writeFile(join(stagingDir, manifestFile), jsonBytes(manifestSkeleton));
  const checksummedFiles = await fileInventory(
    stagingDir,
    new Set(["SHA256SUMS"]),
  );
  const sums = checksummedFiles
    .map((file) => `${file.sha256}  ${file.path}`)
    .join("\n") + "\n";
  await writeFile(join(stagingDir, "SHA256SUMS"), sums, "utf8");

  // Verify the sidecar before publishing the staging directory.
  for (const record of checksummedFiles) {
    const bytes = await readFile(join(stagingDir, record.path));
    checkedEqual(sha256Hex(bytes), record.sha256, `final SHA-256 ${record.path}`);
  }
  const stagingStats = await stat(stagingDir);
  if (!stagingStats.isDirectory()) throw new Error("staging path is not a directory");
  await rename(stagingDir, outputDir);

  const archiveFiles = await fileInventory(outputDir);
  const archiveBytes = archiveFiles.reduce((sum, file) => sum + file.size, 0);
  const manifestBytes = await readFile(join(outputDir, manifestFile));
  process.stdout.write(`${JSON.stringify({
    output: outputDir,
    schema: manifestSkeleton.schema,
    blockHash: BLOCK_HASH,
    models: models.length,
    reconstructedCoreBytes: totalCoreBytes,
    chunks: totalChunkCount,
    vectors: totalVectorCount,
    proofEvidence: proofAvailability,
    rpcRequestAttempts: transcript.length,
    files: archiveFiles.length,
    bytes: archiveBytes,
    manifestSha256: sha256Hex(manifestBytes),
  }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});

/*
MIT License

Copyright (c) 2026 Decentralized Science Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Local decoding + inference for GenesisL1 Forest model format:
// - v1: regression / binary classification (single scoreQ)
// - v2: multiclass classification (argmax over class logits)
// plus helpers to load model bytes from on-chain chunk contracts.

const ethers = globalThis.ethers;

const MAGIC = "GL1F"; // model magic
const CHUNK_MAGIC = "GL1C"; // chunk/table magic
const PACKAGE_MAGIC = "GL1X"; // optional JSON footer
const MAX_SAFE_DEPTH = 20;
const MAX_CORE_BYTES = 0x7fffffff;
const MAX_CHUNK_PAYLOAD_BYTES = 24_572;
const POINTER_WORD_BYTES = 32;
const MAX_POINTERS = Math.floor(MAX_CHUNK_PAYLOAD_BYTES / POINTER_WORD_BYTES);
const ZERO_ADDRESS = "0x0000000000000000000000000000000000000000";


function readU16(dv, off) { return dv.getUint16(off, true); }
function readU32(dv, off) { return dv.getUint32(off, true); }
function readI32(dv, off) { return dv.getInt32(off, true); }

function _u8From(bytesU8){
  return (bytesU8 instanceof Uint8Array) ? bytesU8 : new Uint8Array(bytesU8);
}

function checkedTreeGeometry(depth) {
  if (!Number.isInteger(depth) || depth < 1 || depth > MAX_SAFE_DEPTH) {
    throw new Error(`GL1F depth ${depth} is outside the supported range 1..${MAX_SAFE_DEPTH}`);
  }
  const leaves = 2 ** depth;
  const internal = leaves - 1;
  const perTree = internal * 8 + leaves * 4;
  return { leaves, internal, perTree };
}

function checkedCoreLength(headerBytes, treeCount, perTree) {
  const coreLength = headerBytes + treeCount * perTree;
  if (!Number.isSafeInteger(coreLength) || coreLength > MAX_CORE_BYTES) {
    throw new Error(`Declared GL1F core length ${coreLength} exceeds the decoder safety cap`);
  }
  return coreLength;
}

// Compute the exact GL1F model byte length (without any trailing package footer).
export function gl1fModelLength(bytesU8){
  const u8 = _u8From(bytesU8);
  if (u8.length < 24) throw new Error("GL1F bytes too short");
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const m = String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
  if (m !== MAGIC) throw new Error("Missing GL1F magic");
  const ver = u8[4];

  const nFeatures = readU16(dv, 6);
  if (nFeatures < 1) throw new Error("GL1F nFeatures must be positive");
  const depth = readU16(dv, 8);
  const { perTree } = checkedTreeGeometry(depth);

  if (ver === 1){
    const nTrees = readU32(dv, 10);
    const expect = checkedCoreLength(24, nTrees, perTree);
    if (u8.length < expect) throw new Error(`Model bytes truncated (${u8.length} < ${expect})`);
    return expect;
  }
  if (ver === 2){
    const treesPerClass = readU32(dv, 10);
    if (treesPerClass < 1) throw new Error("GL1F v2 treesPerOutput must be positive");
    const nClasses = readU16(dv, 22);
    if (nClasses < 2) throw new Error(`Bad nClasses=${nClasses}`);
    const treesOff = 24 + nClasses * 4;
    const expect = checkedCoreLength(treesOff, treesPerClass * nClasses, perTree);
    if (u8.length < expect) throw new Error(`Model bytes truncated (${u8.length} < ${expect})`);
    return expect;
  }
  throw new Error(`Unsupported GL1F version ${ver}`);
}

// Parse an optional GL1X footer from a .gl1f file.
// Returns: { modelBytes:Uint8Array, pkg:Object|null, hasFooter:boolean }
export function parseGl1fPackage(bytesU8){
  const u8 = _u8From(bytesU8);
  const modelLen = gl1fModelLength(u8);

  if (u8.length === modelLen) {
    return { modelBytes: u8.slice(0, modelLen), pkg: null, hasFooter: false };
  }
  if (u8.length < modelLen + 12) {
    throw new Error("Trailing bytes are too short to be a canonical GL1X footer");
  }

  const magic = String.fromCharCode(u8[modelLen], u8[modelLen+1], u8[modelLen+2], u8[modelLen+3]);
  if (magic !== PACKAGE_MAGIC) throw new Error("Trailing bytes do not begin with GL1X");
  if (u8[modelLen + 4] !== 1) {
    throw new Error(`Unsupported GL1X version ${u8[modelLen + 4]}`);
  }
  if (u8[modelLen + 5] !== 0 || u8[modelLen + 6] !== 0 || u8[modelLen + 7] !== 0) {
    throw new Error("Non-canonical GL1X reserved bytes");
  }

  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const jsonLen = dv.getUint32(modelLen + 8, true);
  const start = modelLen + 12;
  const end = start + Number(jsonLen);
  if (end !== u8.length) {
    throw new Error(`GL1X length mismatch (${u8.length - modelLen} != ${12 + jsonLen})`);
  }

  let pkg;
  try {
    const raw = u8.slice(start, end);
    const str = new TextDecoder("utf-8", { fatal: true }).decode(raw);
    pkg = JSON.parse(str);
  } catch (error) {
    throw new Error(`Invalid GL1X JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (!pkg || Array.isArray(pkg) || typeof pkg !== "object") {
    throw new Error("GL1X JSON root must be an object");
  }
  return { modelBytes: u8.slice(0, modelLen), pkg, hasFooter: true };
}

export function buildGl1xFooter(pkgObj){
  const payload = new TextEncoder().encode(JSON.stringify(pkgObj));
  const header = new Uint8Array(12);
  header[0] = 0x47; // G
  header[1] = 0x4c; // L
  header[2] = 0x31; // 1
  header[3] = 0x58; // X
  header[4] = 1; header[5] = 0; header[6] = 0; header[7] = 0;
  new DataView(header.buffer).setUint32(8, payload.length >>> 0, true);

  const out = new Uint8Array(header.length + payload.length);
  out.set(header, 0);
  out.set(payload, header.length);
  return out;
}

export function attachGl1xFooter(modelBytesU8, pkgObj){
  const modelBytes = _u8From(modelBytesU8);
  const footer = buildGl1xFooter(pkgObj);
  const out = new Uint8Array(modelBytes.length + footer.length);
  out.set(modelBytes, 0);
  out.set(footer, modelBytes.length);
  return out;
}


export function decodeModel(bytesU8) {
  const u8 = bytesU8 instanceof Uint8Array ? bytesU8 : new Uint8Array(bytesU8);
  if (u8.length < 24) throw new Error("Model buffer too short");

  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const m = String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
  if (m !== MAGIC) throw new Error(`Bad model magic (got ${m})`);

  const version = u8[4];
  if (u8[5] !== 0) throw new Error("Non-canonical GL1F header reserved byte");

  const validateTrees = ({ treesOff, totalTrees, internal, leaves, perTree, nFeatures, baseByOutput, treesPerOutput }) => {
    const maxSafe = Number.MAX_SAFE_INTEGER;
    const accumulatorBounds = baseByOutput.map((base) => Math.abs(Number(base)));
    let off = treesOff;
    for (let tree = 0; tree < totalTrees; tree++) {
      for (let node = 0; node < internal; node++) {
        const feature = readU16(dv, off);
        if (feature >= nFeatures) {
          throw new Error(`Tree ${tree} node ${node} references feature ${feature} >= nFeatures ${nFeatures}`);
        }
        if (readU16(dv, off + 6) !== 0) {
          throw new Error(`Tree ${tree} node ${node} has non-zero reserved bytes`);
        }
        off += 8;
      }
      let maxAbsLeaf = 0;
      for (let leaf = 0; leaf < leaves; leaf++) {
        maxAbsLeaf = Math.max(maxAbsLeaf, Math.abs(readI32(dv, off)));
        off += 4;
      }
      const output = Math.floor(tree / treesPerOutput);
      accumulatorBounds[output] += maxAbsLeaf;
      if (accumulatorBounds[output] > maxSafe) {
        throw new Error(`Output ${output} can exceed JavaScript's exact integer range`);
      }
    }
    if (off !== treesOff + totalTrees * perTree) {
      throw new Error("Internal GL1F decoder length mismatch");
    }
  };

  // =========================
  // v1: scalar output (regression / binary)
  // =========================
  if (version === 1) {
    const nFeatures = readU16(dv, 6);
    const depth = readU16(dv, 8);
    const nTrees = readU32(dv, 10);
    const baseQ = readI32(dv, 14);
    const scaleQ = readU32(dv, 18);
    if (nFeatures < 1) throw new Error("GL1F nFeatures must be positive");
    if (scaleQ < 1) throw new Error("GL1F scaleQ must be positive");
    if (readU16(dv, 22) !== 0) throw new Error("Non-canonical GL1F v1 reserved field");

    const headerSize = 24;
    const { leaves: pow, internal, perTree } = checkedTreeGeometry(depth);

    const expect = checkedCoreLength(headerSize, nTrees, perTree);
    if (u8.length !== expect) {
      const relation = u8.length < expect ? "truncated" : "has undeclared trailing bytes";
      throw new Error(`Model ${relation} (${u8.length} != ${expect})`);
    }

    const treesOff = headerSize;
    validateTrees({
      treesOff,
      totalTrees: nTrees,
      internal,
      leaves: pow,
      perTree,
      nFeatures,
      baseByOutput: [baseQ],
      treesPerOutput: nTrees || 1,
    });
    return { u8, dv, version, nFeatures, depth, nTrees, baseQ, scaleQ, treesOff, internal, leaves: pow, perTree };
  }

  // =========================
  // v2: multiclass (stores base logits + treesPerClass*nClasses trees)
  // Header (little-endian):
  //   0..3  magic "GL1F"
  //   4     version=2
  //   5     reserved
  //   6..7  nFeatures (u16)
  //   8..9  depth (u16)
  //   10..13 treesPerClass (u32)
  //   14..17 reserved (i32)
  //   18..21 scaleQ (u32)
  //   22..23 nClasses (u16)
  //   24..  baseLogitsQ[int32]*nClasses
  //   then trees (class-major)
  // =========================
  if (version === 2) {
    const nFeatures = readU16(dv, 6);
    const depth = readU16(dv, 8);
    const treesPerClass = readU32(dv, 10);
    const scaleQ = readU32(dv, 18);
    const nClasses = readU16(dv, 22);
    if (nFeatures < 1) throw new Error("GL1F nFeatures must be positive");
    if (treesPerClass < 1) throw new Error("GL1F v2 treesPerOutput must be positive");
    if (scaleQ < 1) throw new Error("GL1F scaleQ must be positive");
    if (!nClasses || nClasses < 2) throw new Error(`Bad nClasses=${nClasses}`);
    if (readI32(dv, 14) !== 0) throw new Error("Non-canonical GL1F v2 reserved field");

    const baseOff = 24;
    const treesOff = baseOff + nClasses * 4;
    if (u8.length < treesOff) throw new Error(`Model bytes truncated (${u8.length} < ${treesOff})`);
    const baseLogitsQ = new Int32Array(nClasses);
    for (let k = 0; k < nClasses; k++) {
      baseLogitsQ[k] = readI32(dv, baseOff + k * 4);
    }

    const { leaves: pow, internal, perTree } = checkedTreeGeometry(depth);
    const totalTrees = treesPerClass * nClasses;
    const expect = checkedCoreLength(treesOff, totalTrees, perTree);
    if (u8.length !== expect) {
      const relation = u8.length < expect ? "truncated" : "has undeclared trailing bytes";
      throw new Error(`Model ${relation} (${u8.length} != ${expect})`);
    }
    validateTrees({
      treesOff,
      totalTrees,
      internal,
      leaves: pow,
      perTree,
      nFeatures,
      baseByOutput: Array.from(baseLogitsQ),
      treesPerOutput: treesPerClass,
    });

    // For back-compat with existing UI that expects nTrees/baseQ fields.
    const nTrees = totalTrees;
    const baseQ = 0;

    return {
      u8,
      dv,
      version,
      nFeatures,
      depth,
      nTrees,
      baseQ,
      scaleQ,
      treesOff,
      internal,
      leaves: pow,
      perTree,
      // v2 extras
      nClasses,
      treesPerClass,
      baseLogitsQ,
    };
  }

  throw new Error(`Unsupported model version ${version}`);
}

export function predictQ(model, featuresFloat) {
  if (model?.version !== 1) throw new Error("predictQ only supports v1 (scalar) models");
  const { dv, nFeatures, nTrees, depth, baseQ, scaleQ, treesOff, internal, leaves, perTree } = model;
  if (featuresFloat.length !== nFeatures) throw new Error(`Need ${nFeatures} features`);

  const featQ = new Int32Array(nFeatures);
  for (let i = 0; i < nFeatures; i++) {
    const x = Number(featuresFloat[i]);
    if (!Number.isFinite(x)) throw new Error(`Feature ${i} must be finite`);
    const q = Math.round(x * scaleQ);
    featQ[i] = Math.max(-2147483648, Math.min(2147483647, q));
  }

  let acc = baseQ;

  for (let t = 0; t < nTrees; t++) {
    const base = treesOff + t * perTree;

    let idx = 0;
    for (let level = 0; level < depth; level++) {
      const nodeOff = base + idx * 8;
      const f = dv.getUint16(nodeOff, true);
      const thr = dv.getInt32(nodeOff + 2, true);

      const xq = featQ[f];
      const goRight = xq > thr;
      idx = goRight ? (2 * idx + 2) : (2 * idx + 1);
    }

    const leafIndex = idx - internal;
    const leafBase = base + internal * 8;
    const leafOff = leafBase + leafIndex * 4;
    const valQ = dv.getInt32(leafOff, true);
    acc += valQ;
  }

  return acc;
}

// Predict all class/label logits for a v2 vector-output model.
// Returns: number[] logitsQ[k] in Q-units (integer).
export function predictMultiQ(model, featuresFloat) {
  if (model?.version !== 2) throw new Error("predictMultiQ only supports v2 models");
  const { dv, nFeatures, depth, scaleQ, treesOff, internal, perTree, nClasses, treesPerClass, baseLogitsQ } = model;
  if (featuresFloat.length !== nFeatures) throw new Error(`Need ${nFeatures} features`);

  const featQ = new Int32Array(nFeatures);
  for (let i = 0; i < nFeatures; i++) {
    const x = Number(featuresFloat[i]);
    if (!Number.isFinite(x)) throw new Error(`Feature ${i} must be finite`);
    const q = Math.round(x * scaleQ);
    featQ[i] = Math.max(-2147483648, Math.min(2147483647, q));
  }

  // IMPORTANT: On-chain runtime accumulates logits in int256. Using Int32Array (and bitwise casts)
  // can overflow/wrap for large models. Use JS numbers (safe up to 2^53 for typical scaleQ/tree counts).
  const logits = new Array(nClasses);
  for (let k = 0; k < nClasses; k++) {
    let acc = Number(baseLogitsQ?.[k] ?? 0);
    const classTreeBase = treesOff + (k * treesPerClass) * perTree;

    for (let t = 0; t < treesPerClass; t++) {
      const base = classTreeBase + t * perTree;
      let idx = 0;
      for (let level = 0; level < depth; level++) {
        const nodeOff = base + idx * 8;
        const f = dv.getUint16(nodeOff, true);
        const thr = dv.getInt32(nodeOff + 2, true);
        const xq = featQ[f];
        idx = (xq > thr) ? (2 * idx + 2) : (2 * idx + 1);
      }
      const leafIndex = idx - internal;
      const leafBase = base + internal * 8;
      const leafOff = leafBase + leafIndex * 4;
      const valQ = dv.getInt32(leafOff, true);
      acc += valQ;
    }

    logits[k] = acc;
  }

  return logits;
}

// Predict the argmax class for a v2 multiclass model.
// Returns: { classIndex, bestLogitQ, logitsQ:number[] }
export function predictClassQ(model, featuresFloat) {
  if (model?.version !== 2) throw new Error("predictClassQ only supports v2 multiclass models");
  const logits = predictMultiQ(model, featuresFloat);
  let bestK = 0;
  let best = Number(logits[0] ?? 0);
  for (let k = 1; k < logits.length; k++) {
    const v = Number(logits[k]);
    if (v > best) { best = v; bestK = k; }
  }
  return { classIndex: bestK, bestLogitQ: best, logitsQ: logits };
}

function stripChunkMagic(codeHex, context = "chunk") {
  const bytes = ethers.getBytes(codeHex);
  if (bytes.length < 4) throw new Error(`${context} code too short`);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== CHUNK_MAGIC) throw new Error(`Bad ${context} magic (${magic})`);
  return bytes.slice(4);
}

function parsePointerTable(tableCodeHex, expectedPointers) {
  const data = stripChunkMagic(tableCodeHex, "pointer table");
  const expectedBytes = expectedPointers * POINTER_WORD_BYTES;
  if (data.length !== expectedBytes) {
    throw new Error(
      `Pointer table has ${data.length} payload bytes; expected exactly ${expectedBytes}`,
    );
  }
  const ptrs = [];
  for (let i = 0; i < expectedPointers; i++) {
    const slot = data.slice(i * POINTER_WORD_BYTES, (i + 1) * POINTER_WORD_BYTES);
    if (slot.slice(0, 12).some((byte) => byte !== 0)) {
      throw new Error(`Pointer table entry ${i} has non-zero high-order bytes`);
    }
    const addrBytes = slot.slice(12, POINTER_WORD_BYTES);
    const addr = ethers.getAddress("0x" + Array.from(addrBytes).map(b => b.toString(16).padStart(2, "0")).join(""));
    if (addr.toLowerCase() === ZERO_ADDRESS) {
      throw new Error(`Pointer table entry ${i} is the zero address`);
    }
    ptrs.push(addr);
  }
  return ptrs;
}

function checkedManifestInteger(value, name, minimum, maximum) {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < minimum || number > maximum) {
    throw new Error(
      `${name} must be an integer in ${minimum}..${maximum} (got ${String(value)})`,
    );
  }
  return number;
}

export async function loadModelBytesFromChain({ provider, store, registry, modelId, log = () => {} }) {
  const info = await registry.getModelBytesInfo(modelId, { gasLimit: 2_000_000_000 });
  const tablePtr = ethers.getAddress(String(info[0]));
  if (tablePtr.toLowerCase() === ZERO_ADDRESS) {
    throw new Error("Pointer table address is zero");
  }
  const chunkSize = checkedManifestInteger(
    info[1],
    "chunkSize",
    4,
    MAX_CHUNK_PAYLOAD_BYTES,
  );
  const numChunks = checkedManifestInteger(info[2], "numChunks", 1, MAX_POINTERS);
  const totalBytes = checkedManifestInteger(info[3], "totalBytes", 1, MAX_CORE_BYTES);
  const expectedChunks = Math.ceil(totalBytes / chunkSize);
  if (numChunks !== expectedChunks) {
    throw new Error(
      `numChunks ${numChunks} does not equal ceil(${totalBytes}/${chunkSize}) = ${expectedChunks}`,
    );
  }

  log(`Model bytes info: table=${tablePtr} chunkSize=${chunkSize} numChunks=${numChunks} total=${totalBytes}`);

  const tableCode = await provider.getCode(tablePtr);
  const ptrs = parsePointerTable(tableCode, numChunks);

  const out = new Uint8Array(totalBytes);
  let off = 0;

  for (let i = 0; i < numChunks; i++) {
    const ptr = ptrs[i];
    const code = await provider.getCode(ptr);
    const data = stripChunkMagic(code, `chunk ${i + 1}`);
    const expectedLength = Math.min(chunkSize, totalBytes - off);
    if (data.length !== expectedLength) {
      throw new Error(
        `Chunk ${i + 1} has ${data.length} payload bytes; expected exactly ${expectedLength}`,
      );
    }

    out.set(data, off);
    off += data.length;
    log(`Loaded chunk ${i+1}/${numChunks}: ${data.length} bytes (ptr=${ptr})`);
  }

  if (off !== totalBytes) throw new Error(`Model bytes incomplete (${off}/${totalBytes})`);
  const expectedModelId = ethers.hexlify(modelId);
  const reconstructedModelId = ethers.keccak256(out);
  if (reconstructedModelId.toLowerCase() !== expectedModelId.toLowerCase()) {
    throw new Error(
      `Model ID mismatch: registry=${expectedModelId} reconstructed=${reconstructedModelId}`,
    );
  }
  return out;
}

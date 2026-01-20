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

function readU16(dv, off) { return dv.getUint16(off, true); }
function readU32(dv, off) { return dv.getUint32(off, true); }
function readI32(dv, off) { return dv.getInt32(off, true); }

export function decodeModel(bytesU8) {
  const u8 = bytesU8 instanceof Uint8Array ? bytesU8 : new Uint8Array(bytesU8);
  if (u8.length < 24) throw new Error("Model buffer too short");

  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const m = String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
  if (m !== MAGIC) throw new Error(`Bad model magic (got ${m})`);

  const version = u8[4];

  // =========================
  // v1: scalar output (regression / binary)
  // =========================
  if (version === 1) {
    const nFeatures = readU16(dv, 6);
    const depth = readU16(dv, 8);
    const nTrees = readU32(dv, 10);
    const baseQ = readI32(dv, 14);
    const scaleQ = readU32(dv, 18);

    const headerSize = 24;
    const pow = 1 << depth;
    const internal = pow - 1;
    const perTree = internal * 8 + pow * 4;

    const expect = headerSize + nTrees * perTree;
    if (u8.length < expect) throw new Error(`Model bytes truncated (${u8.length} < ${expect})`);

    const treesOff = headerSize;
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
    if (!nClasses || nClasses < 2) throw new Error(`Bad nClasses=${nClasses}`);

    const baseOff = 24;
    const baseLogitsQ = new Int32Array(nClasses);
    for (let k = 0; k < nClasses; k++) {
      baseLogitsQ[k] = readI32(dv, baseOff + k * 4);
    }

    const treesOff = baseOff + nClasses * 4;

    const pow = 1 << depth;
    const internal = pow - 1;
    const perTree = internal * 8 + pow * 4;
    const totalTrees = treesPerClass * nClasses;
    const expect = treesOff + totalTrees * perTree;
    if (u8.length < expect) throw new Error(`Model bytes truncated (${u8.length} < ${expect})`);

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
// Returns: Int32Array logitsQ[k] in Q-units.
export function predictMultiQ(model, featuresFloat) {
  if (model?.version !== 2) throw new Error("predictMultiQ only supports v2 models");
  const { dv, nFeatures, depth, scaleQ, treesOff, internal, perTree, nClasses, treesPerClass, baseLogitsQ } = model;
  if (featuresFloat.length !== nFeatures) throw new Error(`Need ${nFeatures} features`);

  const featQ = new Int32Array(nFeatures);
  for (let i = 0; i < nFeatures; i++) {
    const x = Number(featuresFloat[i]);
    const q = Math.round(x * scaleQ);
    featQ[i] = Math.max(-2147483648, Math.min(2147483647, q));
  }

  const logits = new Int32Array(nClasses);
  for (let k = 0; k < nClasses; k++) {
    let acc = baseLogitsQ[k] | 0;
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
// Returns: { classIndex, bestLogitQ, logitsQ:Int32Array }
export function predictClassQ(model, featuresFloat) {
  if (model?.version !== 2) throw new Error("predictClassQ only supports v2 multiclass models");
  const logits = predictMultiQ(model, featuresFloat);
  let bestK = 0;
  let best = logits[0] | 0;
  for (let k = 1; k < logits.length; k++) {
    const v = logits[k] | 0;
    if (v > best) { best = v; bestK = k; }
  }
  return { classIndex: bestK, bestLogitQ: best, logitsQ: logits };
}

function stripChunkMagic(codeHex) {
  const bytes = ethers.getBytes(codeHex);
  if (bytes.length < 4) throw new Error("Chunk code too short");
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== CHUNK_MAGIC) throw new Error(`Bad chunk magic (${magic})`);
  return bytes.slice(4);
}

function parsePointerTable(tableCodeHex) {
  const data = stripChunkMagic(tableCodeHex);
  if (data.length % 32 !== 0) throw new Error("Pointer table misaligned");
  const n = data.length / 32;
  const ptrs = [];
  for (let i = 0; i < n; i++) {
    const slot = data.slice(i * 32, i * 32 + 32);
    const addrBytes = slot.slice(12, 32);
    const addr = ethers.getAddress("0x" + Array.from(addrBytes).map(b => b.toString(16).padStart(2, "0")).join(""));
    ptrs.push(addr);
  }
  return ptrs;
}

export async function loadModelBytesFromChain({ provider, store, registry, modelId, log = () => {} }) {
  const info = await registry.getModelBytesInfo(modelId, { gasLimit: 2_000_000_000 });
  const tablePtr = info[0];
  const chunkSize = Number(info[1]);
  const numChunks = Number(info[2]);
  const totalBytes = Number(info[3]);

  log(`Model bytes info: table=${tablePtr} chunkSize=${chunkSize} numChunks=${numChunks} total=${totalBytes}`);

  const tableCode = await provider.getCode(tablePtr);
  const ptrs = parsePointerTable(tableCode);
  if (ptrs.length < numChunks) throw new Error("Pointer table has fewer pointers than expected");

  const out = new Uint8Array(totalBytes);
  let off = 0;

  for (let i = 0; i < numChunks; i++) {
    const ptr = ptrs[i];
    const code = await provider.getCode(ptr);
    const data = stripChunkMagic(code);

    const take = Math.min(data.length, totalBytes - off);
    out.set(data.slice(0, take), off);
    off += take;
    log(`Loaded chunk ${i+1}/${numChunks}: ${take} bytes (ptr=${ptr})`);
    if (off >= totalBytes) break;
  }

  if (off !== totalBytes) throw new Error(`Model bytes incomplete (${off}/${totalBytes})`);
  return out;
}

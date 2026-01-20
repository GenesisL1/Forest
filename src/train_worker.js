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


// GenesisL1 Forest — Browser GBDT trainer
// - Regression (squared loss)
// - Binary classification (log loss)
// - Multiclass classification (softmax cross-entropy)
// - Multilabel classification (independent sigmoid per label; micro-averaged log loss + accuracy)
// Split 0.7/0.2/0.1 with seeded shuffle. Fixed-depth trees.
// Outputs GenesisL1 Forest model bytes ("GL1F" v1).

let stopFlag = false;

const INT32_MAX = 2147483647;
const INT32_MIN = -2147483648;

function clampI32(x) {
  if (x > INT32_MAX) return INT32_MAX;
  if (x < INT32_MIN) return INT32_MIN;
  return x | 0;
}

function xorshift32(seed) {
  let x = (seed | 0) || 123456789;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return x >>> 0;
  };
}

function shuffledIndices(n, seed) {
  const rng = xorshift32(seed);
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = rng() % (i + 1);
    const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
  }
  return idx;
}

function splitIdx(idx, fracTrain = 0.7, fracVal = 0.2) {
  const n = idx.length;
  let nTrain = Math.floor(n * fracTrain);
  let nVal = Math.floor(n * fracVal);
  if (nTrain < 1) nTrain = 1;
  if (nVal < 1) nVal = 1;
  if (nTrain + nVal >= n) nVal = Math.max(1, n - nTrain - 1);
  const nTest = Math.max(1, n - nTrain - nVal);
  const train = idx.slice(0, nTrain);
  const val = idx.slice(nTrain, nTrain + nVal);
  const test = idx.slice(nTrain + nVal, nTrain + nVal + nTest);
  return { train, val, test };
}

// Stratified split for single-label classification (binary/multiclass).
// Falls back to random split if it cannot produce non-empty splits.
function splitIdxStratifiedByClass(idx, yK, nClasses, fracTrain = 0.7, fracVal = 0.2) {
  const buckets = Array.from({ length: nClasses }, () => []);
  for (let i = 0; i < idx.length; i++) {
    const r = idx[i];
    let k = yK[r] | 0;
    if (!(k >= 0 && k < nClasses)) k = 0;
    buckets[k].push(r);
  }

  const train = [];
  const val = [];
  const test = [];

  for (let k = 0; k < nClasses; k++) {
    const arr = buckets[k];
    const n = arr.length;
    if (n <= 0) continue;

    let nTrain = Math.floor(n * fracTrain);
    let nVal = Math.floor(n * fracVal);
    // Ensure at least 1 sample remains for test if possible.
    if (nTrain + nVal >= n) {
      nVal = Math.max(0, n - nTrain - 1);
      if (nTrain + nVal >= n) nTrain = Math.max(0, n - nVal - 1);
    }

    // No strict per-class min constraints; only global splits must be non-empty.
    for (let i = 0; i < nTrain; i++) train.push(arr[i]);
    for (let i = nTrain; i < nTrain + nVal; i++) val.push(arr[i]);
    for (let i = nTrain + nVal; i < n; i++) test.push(arr[i]);
  }

  if (train.length < 1 || val.length < 1 || test.length < 1) {
    return splitIdx(idx, fracTrain, fracVal);
  }

  return {
    train: Uint32Array.from(train),
    val: Uint32Array.from(val),
    test: Uint32Array.from(test)
  };
}


function mse(yQ, predQ, indices, scaleQ) {
  let s = 0;
  for (let i = 0; i < indices.length; i++) {
    const r = indices[i];
    const diff = (yQ[r] - predQ[r]) / scaleQ;
    s += diff * diff;
  }
  return s / Math.max(1, indices.length);
}

// Stable sigmoid for classification (avoids exp overflow for large |z|).
function sigmoid(z) {
  const x = Number(z);
  if (!Number.isFinite(x)) return 0.5;
  if (x >= 0) {
    const ez = Math.exp(-x);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(x);
  return ez / (1 + ez);
}

function loglossAcc(y01, predQ, indices, scaleQ, wRow = null) {
  let loss = 0;
  let correct = 0;
  let wSum = 0;
  // Numerically safe clamp
  const EPS = 1e-12;
  for (let i = 0; i < indices.length; i++) {
    const r = indices[i];
    const y = (y01[r] >= 0.5) ? 1 : 0;
    const w = (wRow && Number.isFinite(wRow[r])) ? wRow[r] : 1;
    if (!(w > 0)) continue;
    wSum += w;
    const logit = predQ[r] / scaleQ;
    let p = sigmoid(logit);
    if (p < EPS) p = EPS;
    else if (p > 1 - EPS) p = 1 - EPS;
    loss += w * (-(y * Math.log(p) + (1 - y) * Math.log(1 - p)));
    const pred = (p >= 0.5) ? 1 : 0;
    if (pred === y) correct += w;
  }
  const denom = wSum || 1;
  return { loss: loss / denom, acc: correct / denom };
}

// Softmax probabilities for multiclass classification.
// predQ layout: row-major (r*nClasses + k) in Q-units.
function softmaxProbs(predQ, nRows, nClasses, scaleQ, outProb) {
  for (let r = 0; r < nRows; r++) {
    const base = r * nClasses;
    // Find max logit for numerical stability
    let maxZ = -Infinity;
    for (let k = 0; k < nClasses; k++) {
      const z = predQ[base + k] / scaleQ;
      if (z > maxZ) maxZ = z;
    }
    let sum = 0;
    for (let k = 0; k < nClasses; k++) {
      const z = predQ[base + k] / scaleQ;
      const e = Math.exp(z - maxZ);
      outProb[base + k] = e;
      sum += e;
    }
    const inv = 1 / (sum || 1);
    for (let k = 0; k < nClasses; k++) outProb[base + k] *= inv;
  }
}

function loglossAccMulti(yK, prob, indices, nClasses, wRow = null) {
  let loss = 0;
  let correct = 0;
  let wSum = 0;
  const EPS = 1e-12;
  for (let i = 0; i < indices.length; i++) {
    const r = indices[i];
    const y = yK[r] | 0;
    const w = (wRow && Number.isFinite(wRow[r])) ? wRow[r] : 1;
    if (!(w > 0)) continue;
    wSum += w;

    const base = r * nClasses;
    let bestK = 0;
    let bestP = prob[base] || 0;
    for (let k = 1; k < nClasses; k++) {
      const p = prob[base + k] || 0;
      if (p > bestP) { bestP = p; bestK = k; }
    }
    if (bestK === y) correct += w;

    let py = prob[base + y];
    if (!(py > 0)) py = EPS;
    else if (py > 1 - EPS) py = 1 - EPS;
    loss += w * (-Math.log(py));
  }
  const denom = wSum || 1;
  return { loss: loss / denom, acc: correct / denom };
}

// Micro-averaged logloss + accuracy for multilabel classification.
// yFlat layout: row-major (r*nLabels + k) with 0/1 values.
// predQ layout: row-major (r*nLabels + k) in Q-units (logits).
function loglossAccMultiLabel(yFlat, predQ, indices, nLabels, scaleQ, posW = null, wScale = 1) {
  let loss = 0;
  let correct = 0;
  let wSum = 0;
  const EPS = 1e-12;
  for (let i = 0; i < indices.length; i++) {
    const r = indices[i];
    const base = r * nLabels;
    for (let k = 0; k < nLabels; k++) {
      const y = (yFlat[base + k] >= 0.5) ? 1 : 0;
      const w0 = posW ? (y ? (posW[k] || 1) : 1) : 1;
      const w = wScale * w0;
      if (!(w > 0)) continue;
      wSum += w;

      const logit = predQ[base + k] / scaleQ;
      let p = sigmoid(logit);
      if (p < EPS) p = EPS;
      else if (p > 1 - EPS) p = 1 - EPS;
      loss += w * (-(y * Math.log(p) + (1 - y) * Math.log(1 - p)));
      const pred = (p >= 0.5) ? 1 : 0;
      if (pred === y) correct += w;
    }
  }
  const denom = wSum || 1;
  return { loss: loss / denom, acc: correct / denom };
}

function serializeModel({ nFeatures, depth, nTrees, baseQ, scaleQ, trees }) {
  const pow = 1 << depth;
  const internal = pow - 1;
  const perTree = internal * 8 + pow * 4;
  const totalBytes = 24 + nTrees * perTree;

  const out = new Uint8Array(totalBytes);
  const dv = new DataView(out.buffer);

  out[0] = "G".charCodeAt(0);
  out[1] = "L".charCodeAt(0);
  out[2] = "1".charCodeAt(0);
  out[3] = "F".charCodeAt(0);
  out[4] = 1;
  out[5] = 0;

  dv.setUint16(6, nFeatures, true);
  dv.setUint16(8, depth, true);
  dv.setUint32(10, nTrees, true);
  dv.setInt32(14, baseQ, true);
  dv.setUint32(18, scaleQ, true);
  out[22] = 0;
  out[23] = 0;

  let off = 24;
  for (let t = 0; t < nTrees; t++) {
    const tr = trees[t];
    const feat = tr.feat;
    const thr = tr.thr;
    const leaf = tr.leaf;
    for (let i = 0; i < internal; i++) {
      dv.setUint16(off, feat[i], true); off += 2;
      dv.setInt32(off, thr[i], true); off += 4;
      dv.setUint16(off, 0, true); off += 2;
    }
    for (let i = 0; i < pow; i++) {
      dv.setInt32(off, leaf[i], true); off += 4;
    }
  }

  return out;
}

function serializeModelV2({ nFeatures, depth, nClasses, treesPerClass, baseLogitsQ, scaleQ, treesByClass }) {
  const pow = 1 << depth;
  const internal = pow - 1;
  const perTree = internal * 8 + pow * 4;

  const headerSize = 24 + nClasses * 4;
  const totalTrees = treesPerClass * nClasses;
  const totalBytes = headerSize + totalTrees * perTree;

  const out = new Uint8Array(totalBytes);
  const dv = new DataView(out.buffer);

  out[0] = "G".charCodeAt(0);
  out[1] = "L".charCodeAt(0);
  out[2] = "1".charCodeAt(0);
  out[3] = "F".charCodeAt(0);
  out[4] = 2; // version
  out[5] = 0;

  dv.setUint16(6, nFeatures, true);
  dv.setUint16(8, depth, true);
  dv.setUint32(10, treesPerClass, true);
  dv.setInt32(14, 0, true); // reserved
  dv.setUint32(18, scaleQ, true);
  dv.setUint16(22, nClasses, true);

  // base logits
  let off = 24;
  for (let k = 0; k < nClasses; k++) {
    dv.setInt32(off, baseLogitsQ[k] | 0, true);
    off += 4;
  }

  // Trees: class-major (all trees for class0, then class1, ...)
  for (let k = 0; k < nClasses; k++) {
    const clsTrees = treesByClass[k] || [];
    for (let t = 0; t < treesPerClass; t++) {
      const tr = clsTrees[t];
      const feat = tr.feat;
      const thr = tr.thr;
      const leaf = tr.leaf;
      for (let i = 0; i < internal; i++) {
        dv.setUint16(off, feat[i], true); off += 2;
        dv.setInt32(off, thr[i], true); off += 4;
        dv.setUint16(off, 0, true); off += 2;
      }
      for (let i = 0; i < pow; i++) {
        dv.setInt32(off, leaf[i], true); off += 4;
      }
    }
  }

  return out;
}

function sampleFeatures(nFeatures, k, rng) {
  k = Math.max(1, Math.min(nFeatures, k));
  const used = new Uint8Array(nFeatures);
  const out = [];
  while (out.length < k) {
    const f = rng() % nFeatures;
    if (used[f]) continue;
    used[f] = 1;
    out.push(f);
  }
  return out;
}

function meanResidual(residual, samples) {
  let s = 0;
  for (let i = 0; i < samples.length; i++) s += residual[samples[i]];
  return s / Math.max(1, samples.length);
}

function buildTreeRegression({
  X, nRows, nFeatures, trainSamples, residual,
  featMin, featRange, depth, minLeaf, lr, scaleQ, rng,
  bins = 32, binning = "linear", qThr = null
}) {
  // Split-candidate histogram binning is training-only.
  // On-chain format (tree structure + quantized thresholds/leaves) stays unchanged.
  const BINS = Math.max(8, bins | 0);
  const isQuantile = String(binning || "").toLowerCase() === "quantile";

  const pow = 1 << depth;
  const internal = pow - 1;

  const featU16 = new Uint16Array(internal);
  const thrI32 = new Int32Array(internal);
  const leafI32 = new Int32Array(pow);

  function fillForced(nodeIdx, level, leafValQ) {
    if (level === depth) {
      leafI32[nodeIdx - internal] = leafValQ;
      return;
    }
    featU16[nodeIdx] = 0;
    thrI32[nodeIdx] = INT32_MAX;
    fillForced(nodeIdx * 2 + 1, level + 1, leafValQ);
    fillForced(nodeIdx * 2 + 2, level + 1, leafValQ);
  }

  function computeLeafQ(samples) {
    const m = meanResidual(residual, samples);
    const v = lr * m;
    return clampI32(Math.round(v * scaleQ));
  }

  function nodeSplit(nodeIdx, level, samples) {
    if (stopFlag) return;

    if (samples.length === 0) { fillForced(nodeIdx, level, 0); return; }
    if (level === depth) { leafI32[nodeIdx - internal] = computeLeafQ(samples); return; }
    if (samples.length < 2 * minLeaf) { fillForced(nodeIdx, level, computeLeafQ(samples)); return; }

    const colsample = Math.max(1, Math.round(Math.sqrt(nFeatures)));
    const feats = sampleFeatures(nFeatures, colsample, rng);

    let bestF = -1;
    let bestThrQ = 0;
    let bestSSE = Infinity;

    const cnt = new Int32Array(BINS);
    const sum = new Float64Array(BINS);
    const sum2 = new Float64Array(BINS);

    for (let fi = 0; fi < feats.length; fi++) {
      const f = feats[fi];
      const range = featRange[f];
      if (!(range > 0)) continue;

      const thrArr = isQuantile ? (qThr ? qThr[f] : null) : null;
      if (isQuantile) {
        if (!thrArr || (thrArr.length | 0) !== (BINS - 1)) continue;
      }

      cnt.fill(0); sum.fill(0); sum2.fill(0);

      const minF = featMin[f];
      const inv = 1 / range;

      let totalCount = 0;
      let totalSum = 0;
      let totalSum2 = 0;

      for (let i = 0; i < samples.length; i++) {
        const r = samples[i];
        const x = X[r * nFeatures + f];
        const rr = residual[r];

        let b = 0;
        if (isQuantile) {
          // Lower-bound: first threshold >= x. Returns [0..BINS-1].
          let lo = 0, hi = thrArr.length;
          while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (x <= thrArr[mid]) hi = mid;
            else lo = mid + 1;
          }
          b = lo;
        } else {
          b = Math.floor(((x - minF) * inv) * BINS);
          if (b < 0) b = 0;
          else if (b >= BINS) b = BINS - 1;
        }

        cnt[b] += 1;
        sum[b] += rr;
        sum2[b] += rr * rr;

        totalCount += 1;
        totalSum += rr;
        totalSum2 += rr * rr;
      }

      if (totalCount < 2 * minLeaf) continue;

      let leftCount = 0;
      let leftSum = 0;
      let leftSum2 = 0;

      for (let b = 0; b < BINS - 1; b++) {
        leftCount += cnt[b];
        leftSum += sum[b];
        leftSum2 += sum2[b];

        const rightCount = totalCount - leftCount;
        if (leftCount < minLeaf || rightCount < minLeaf) continue;

        const rightSum = totalSum - leftSum;
        const rightSum2 = totalSum2 - leftSum2;

        const leftSSE = leftSum2 - (leftSum * leftSum) / leftCount;
        const rightSSE = rightSum2 - (rightSum * rightSum) / rightCount;
        const sse = leftSSE + rightSSE;

        if (sse < bestSSE) {
          bestSSE = sse;
          bestF = f;
          const thrF = isQuantile ? thrArr[b] : (minF + range * ((b + 1) / BINS));
          bestThrQ = clampI32(Math.round(thrF * scaleQ));
        }
      }
    }

    if (bestF < 0) { fillForced(nodeIdx, level, computeLeafQ(samples)); return; }

    const left = [];
    const right = [];
    for (let i = 0; i < samples.length; i++) {
      const r = samples[i];
      const x = X[r * nFeatures + bestF];
      const xQ = clampI32(Math.round(x * scaleQ));
      if (xQ > bestThrQ) right.push(r);
      else left.push(r);
    }

    if (left.length < minLeaf || right.length < minLeaf) { fillForced(nodeIdx, level, computeLeafQ(samples)); return; }

    featU16[nodeIdx] = bestF;
    thrI32[nodeIdx] = bestThrQ;

    nodeSplit(nodeIdx * 2 + 1, level + 1, left);
    nodeSplit(nodeIdx * 2 + 2, level + 1, right);
  }

  nodeSplit(0, 0, Array.from(trainSamples));
  return { feat: featU16, thr: thrI32, leaf: leafI32 };
}
function buildTreeBinary({
  X, nRows, nFeatures, trainSamples, grad, hess,
  featMin, featRange, depth, minLeaf, lr, scaleQ, rng,
  bins = 32, binning = "linear", qThr = null
}) {
  // Classic gradient boosting for logistic loss using a Newton-style leaf weight:
  //   w = lr * sum(grad) / (sum(hess) + lambda)
  // Split criterion uses gain based on sums of grad/hess.
  const BINS = Math.max(8, bins | 0);
  const isQuantile = String(binning || "").toLowerCase() === "quantile";

  const pow = 1 << depth;
  const internal = pow - 1;

  const featU16 = new Uint16Array(internal);
  const thrI32 = new Int32Array(internal);
  const leafI32 = new Int32Array(pow);

  const LAMBDA = 1.0;

  function fillForced(nodeIdx, level, leafValQ) {
    if (level === depth) {
      leafI32[nodeIdx - internal] = leafValQ;
      return;
    }
    featU16[nodeIdx] = 0;
    thrI32[nodeIdx] = INT32_MAX;
    fillForced(nodeIdx * 2 + 1, level + 1, leafValQ);
    fillForced(nodeIdx * 2 + 2, level + 1, leafValQ);
  }

  function computeLeafQ(samples) {
    let G = 0;
    let H = 0;
    for (let i = 0; i < samples.length; i++) {
      const r = samples[i];
      G += grad[r];
      H += hess[r];
    }
    const w = lr * (G / (H + LAMBDA));
    return clampI32(Math.round(w * scaleQ));
  }

  function nodeSplit(nodeIdx, level, samples) {
    if (stopFlag) return;

    if (samples.length === 0) { fillForced(nodeIdx, level, 0); return; }
    if (level === depth) { leafI32[nodeIdx - internal] = computeLeafQ(samples); return; }
    if (samples.length < 2 * minLeaf) { fillForced(nodeIdx, level, computeLeafQ(samples)); return; }

    const colsample = Math.max(1, Math.round(Math.sqrt(nFeatures)));
    const feats = sampleFeatures(nFeatures, colsample, rng);

    let bestF = -1;
    let bestThrQ = 0;
    let bestGain = 0;

    const cnt = new Int32Array(BINS);
    const sumG = new Float64Array(BINS);
    const sumH = new Float64Array(BINS);

    for (let fi = 0; fi < feats.length; fi++) {
      const f = feats[fi];
      const range = featRange[f];
      if (!(range > 0)) continue;

      const thrArr = isQuantile ? (qThr ? qThr[f] : null) : null;
      if (isQuantile) {
        if (!thrArr || (thrArr.length | 0) !== (BINS - 1)) continue;
      }

      cnt.fill(0); sumG.fill(0); sumH.fill(0);
      const minF = featMin[f];
      const inv = 1 / range;

      let totalCount = 0;
      let totalG = 0;
      let totalH = 0;

      for (let i = 0; i < samples.length; i++) {
        const r = samples[i];
        const x = X[r * nFeatures + f];

        let b = 0;
        if (isQuantile) {
          let lo = 0, hi = thrArr.length;
          while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (x <= thrArr[mid]) hi = mid;
            else lo = mid + 1;
          }
          b = lo;
        } else {
          b = Math.floor(((x - minF) * inv) * BINS);
          if (b < 0) b = 0;
          else if (b >= BINS) b = BINS - 1;
        }

        const g = grad[r];
        const h = hess[r];

        cnt[b] += 1;
        sumG[b] += g;
        sumH[b] += h;
        totalCount += 1;
        totalG += g;
        totalH += h;
      }

      if (totalCount < 2 * minLeaf) continue;

      const parentScore = (totalG * totalG) / (totalH + LAMBDA);

      let leftCount = 0;
      let leftG = 0;
      let leftH = 0;

      for (let b = 0; b < BINS - 1; b++) {
        leftCount += cnt[b];
        leftG += sumG[b];
        leftH += sumH[b];

        const rightCount = totalCount - leftCount;
        if (leftCount < minLeaf || rightCount < minLeaf) continue;

        const rightG = totalG - leftG;
        const rightH = totalH - leftH;

        const gain = (leftG * leftG) / (leftH + LAMBDA)
          + (rightG * rightG) / (rightH + LAMBDA)
          - parentScore;

        if (gain > bestGain) {
          bestGain = gain;
          bestF = f;
          const thrF = isQuantile ? thrArr[b] : (minF + range * ((b + 1) / BINS));
          bestThrQ = clampI32(Math.round(thrF * scaleQ));
        }
      }
    }

    if (bestF < 0) { fillForced(nodeIdx, level, computeLeafQ(samples)); return; }

    const left = [];
    const right = [];
    for (let i = 0; i < samples.length; i++) {
      const r = samples[i];
      const x = X[r * nFeatures + bestF];
      const xQ = clampI32(Math.round(x * scaleQ));
      if (xQ > bestThrQ) right.push(r);
      else left.push(r);
    }

    if (left.length < minLeaf || right.length < minLeaf) { fillForced(nodeIdx, level, computeLeafQ(samples)); return; }

    featU16[nodeIdx] = bestF;
    thrI32[nodeIdx] = bestThrQ;

    nodeSplit(nodeIdx * 2 + 1, level + 1, left);
    nodeSplit(nodeIdx * 2 + 2, level + 1, right);
  }

  nodeSplit(0, 0, Array.from(trainSamples));
  return { feat: featU16, thr: thrI32, leaf: leafI32 };
}
self.onmessage = async (ev) => {
  const msg = ev.data;
  if (msg?.type === "stop") { stopFlag = true; return; }
  if (msg?.type !== "train") return;
  stopFlag = false;

  try {
    const X = new Float32Array(msg.X);
    const y = new Float32Array(msg.y);
    const nRows = msg.nRows | 0;
    const nFeatures = msg.nFeatures | 0;
    const p = msg.params || {};

    const maxTrees = Math.max(1, p.trees | 0);
    const depth = Math.max(1, p.depth | 0);
    const lrBase = Number(p.lr ?? 0.05);
    const minLeaf = Math.max(1, p.minLeaf | 0);
    const seed = (p.seed | 0) || 42;
    const earlyStop = !!p.earlyStop;
    const patience = Math.max(1, p.patience | 0);
    const scaleQ = Math.max(1, p.scaleQ | 0);

    const taskRaw = String(p.task || "regression").trim();
    const task = (taskRaw === "binary_classification" || taskRaw === "binary" || taskRaw === "classification")
      ? "binary_classification"
      : (taskRaw === "multiclass_classification" || taskRaw === "multiclass")
        ? "multiclass_classification"
        : (taskRaw === "multilabel_classification" || taskRaw === "multilabel")
          ? "multilabel_classification"
          : "regression";

    const idx = shuffledIndices(nRows, seed);

    // Optional class-imbalance settings (weights + stratified split)
    const imbalance = (p && typeof p.imbalance === "object" && p.imbalance) ? p.imbalance : null;
    const stratify = !!imbalance?.stratify;

    // Dataset split (train/val/test). Use params provided by UI (p).
    // NOTE: a previous version accidentally referenced an undefined `params` symbol.
    const fracTrain = (typeof p.splitTrain === "number") ? p.splitTrain : 0.7;
    const fracVal = (typeof p.splitVal === "number") ? p.splitVal : 0.2;

    let train, val, test;
    if (stratify && (task === "binary_classification" || task === "multiclass_classification")) {
      const nClassesForSplit = (task === "multiclass_classification") ? Math.max(2, (p.nClasses | 0) || 2) : 2;
      ({ train, val, test } = splitIdxStratifiedByClass(idx, y, nClassesForSplit, fracTrain, fracVal));
    } else {
      ({ train, val, test } = splitIdx(idx, fracTrain, fracVal));
    }

    function concatIdx(a, b) {
      const out = new Uint32Array(a.length + b.length);
      out.set(a, 0);
      out.set(b, a.length);
      return out;
    }

    // Optional refit stage: train on Train+Val for a fixed tree budget.
    const refitTrainVal = !!p.refitTrainVal;
    const trainFit = refitTrainVal ? concatIdx(train, val) : train;

    const trainIdx = trainFit;

    // Feature min/max on train
    const featMin = new Float32Array(nFeatures);
    const featMax = new Float32Array(nFeatures);
    for (let f = 0; f < nFeatures; f++) { featMin[f] = Infinity; featMax[f] = -Infinity; }
    for (let i = 0; i < trainFit.length; i++) {
      const r = trainFit[i];
      const base = r * nFeatures;
      for (let f = 0; f < nFeatures; f++) {
        const x = X[base + f];
        if (x < featMin[f]) featMin[f] = x;
        if (x > featMax[f]) featMax[f] = x;
      }
    }
    const featRange = new Float32Array(nFeatures);
    for (let f = 0; f < nFeatures; f++) {
      const r = featMax[f] - featMin[f];
      featRange[f] = r > 0 ? r : 0;
    }

    // Split-candidate binning (training-only; on-chain bytes are unchanged).
    const bins = Math.max(8, Math.min(512, (p.bins | 0) || 32));
    let binning = String(p.binning || "linear").trim().toLowerCase();
    if (binning !== "quantile") binning = "linear";

    // For quantile binning we precompute (bins-1) thresholds per feature from the train split
    // (using a random-like sample because `train` is already shuffled).
    let qThr = null;
    if (binning === "quantile") {
      const sampleN0 = (p.quantileSamples | 0) || 50000;
      const sampleN = Math.min(trainFit.length, Math.max(256, sampleN0));
      self.postMessage({
        type: "log",
        line: `Quantile binning: computing thresholds (bins=${bins}, sample=${sampleN.toLocaleString()} rows)…`
      });

      qThr = new Array(nFeatures);
      for (let f = 0; f < nFeatures; f++) {
        if (!(featRange[f] > 0)) { qThr[f] = null; continue; }

        const vals = new Float32Array(sampleN);
        for (let i = 0; i < sampleN; i++) {
          const r = trainFit[i];
          vals[i] = X[r * nFeatures + f];
        }
        vals.sort();

        const thr = new Float32Array(Math.max(1, bins - 1));
        const nV = vals.length;
        let prev = -Infinity;
        for (let j = 1; j < bins; j++) {
          const q = j / bins;
          const pos = q * (nV - 1);
          const lo = Math.floor(pos);
          const hi = Math.min(nV - 1, lo + 1);
          let t = vals[lo];
          if (hi !== lo) t = t + (vals[hi] - vals[lo]) * (pos - lo);
          if (!Number.isFinite(t)) t = featMin[f] + featRange[f] * (j / bins);
          if (t < prev) t = prev;
          thr[j - 1] = t;
          prev = t;
        }
        qThr[f] = thr;
      }

      self.postMessage({ type: "log", line: "Quantile thresholds ready." });
    }

    self.postMessage({ type: "log", line: `Binning: ${binning} bins=${bins}` });

    const rng = xorshift32(seed ^ 0x9e3779b9);
    const trees = [];

    // Shared leaf-walk for both tasks.
    function treePredictLeafQ(tree, row) {
      const pow = 1 << depth;
      const internal = pow - 1;
      let idxNode = 0;
      for (let lvl = 0; lvl < depth; lvl++) {
        const f = tree.feat[idxNode];
        const thrQ = tree.thr[idxNode];
        const x = X[row * nFeatures + f];
        const xQ = clampI32(Math.round(x * scaleQ));
        idxNode = (xQ > thrQ) ? (idxNode * 2 + 2) : (idxNode * 2 + 1);
      }
      const leafIndex = idxNode - internal;
      return tree.leaf[leafIndex] | 0;
    }

    function applyTree(tree, indices, predQ) {
      for (let i = 0; i < indices.length; i++) {
        const r = indices[i];
        predQ[r] += treePredictLeafQ(tree, r);
      }
    }

    self.postMessage({ type: "log", line: `Split: train=${train.length} val=${val.length} test=${test.length}` });

    // =========================================================
    // Learning-rate schedule (optional)
    // =========================================================
    const lrSched = (p && typeof p.lrSchedule === "object" && p.lrSchedule) ? p.lrSchedule : null;
    let lrMode = lrSched ? String(lrSched.mode || "none") : "none";
    let lrCur = lrBase;
    let lrMin = 0;
    // Plateau schedule state
    let plateauPatience = 0;
    let plateauFactor = 1.0;
    let plateauSince = 0;
    // Piecewise schedule state
    let piecewiseSegs = null;
    let piecewiseLastLR = NaN;

    if (lrMode === "plateau") {
      plateauPatience = parseInt(lrSched?.patience ?? lrSched?.n ?? 0, 10);
      if (!Number.isFinite(plateauPatience) || plateauPatience < 1) {
        // Simple heuristic default: ~10% of maxTrees, clamped.
        plateauPatience = Math.max(5, Math.min(100, Math.round(maxTrees * 0.1)));
      }
      const pct = Number(lrSched?.dropPct ?? 10);
      const factor = 1 - (pct / 100);
      if (!(factor > 0 && factor < 1)) {
        plateauFactor = 0.9;
      } else {
        plateauFactor = factor;
      }
      lrMin = Number(lrSched?.minLR ?? 0);
      if (!Number.isFinite(lrMin) || lrMin < 0) lrMin = 0;
      self.postMessage({ type: "log", line: `LR schedule: plateau patience=${plateauPatience} reduce=${((1 - plateauFactor) * 100).toFixed(0)}% minLR=${lrMin}` });
    } else if (lrMode === "piecewise") {
      const segsIn = Array.isArray(lrSched?.segments) ? lrSched.segments : [];
      const segs = [];
      for (const s of segsIn) {
        const start = parseInt(s?.start ?? 0, 10);
        const end = parseInt(s?.end ?? s?.start ?? 0, 10);
        const lr = Number(s?.lr);
        if (!Number.isFinite(start) || start < 1) throw new Error(`LR schedule: invalid start ${s?.start}`);
        if (!Number.isFinite(end) || end < start) throw new Error(`LR schedule: invalid end ${s?.end}`);
        if (!Number.isFinite(lr) || lr <= 0) throw new Error(`LR schedule: invalid lr ${s?.lr}`);
        segs.push({ start, end, lr });
      }
      segs.sort((a, b) => (a.start - b.start) || (a.end - b.end));
      for (let i = 1; i < segs.length; i++) {
        if (segs[i].start <= segs[i - 1].end) {
          throw new Error(`LR schedule ranges overlap: ${segs[i - 1].start}-${segs[i - 1].end} and ${segs[i].start}-${segs[i].end}`);
        }
      }
      piecewiseSegs = segs;
      if (piecewiseSegs.length === 0) {
        lrMode = "none";
      } else {
        self.postMessage({ type: "log", line: `LR schedule: piecewise segments=${piecewiseSegs.length}` });
      }
    } else {
      lrMode = "none";
    }

    function lrForIter(t) {
      const iter = t | 0;
      if (lrMode === "piecewise" && Array.isArray(piecewiseSegs) && piecewiseSegs.length) {
        // Find the matching segment (ranges are 1-indexed inclusive).
        let lr = lrBase;
        for (let i = 0; i < piecewiseSegs.length; i++) {
          const s = piecewiseSegs[i];
          if (iter < s.start) break;
          if (iter >= s.start && iter <= s.end) { lr = s.lr; break; }
        }
        if (!(lr > 0)) lr = lrBase;
        if (lr !== piecewiseLastLR) {
          piecewiseLastLR = lr;
          self.postMessage({ type: "log", line: `LR=${lr} at tree ${iter}` });
        }
        return lr;
      }
      if (lrMode === "plateau") return lrCur;
      return lrBase;
    }

    function lrAfterMetric(improved, t) {
      if (lrMode !== "plateau") return;
      plateauSince = improved ? 0 : (plateauSince + 1);
      if (plateauSince >= plateauPatience) {
        const old = lrCur;
        lrCur = lrCur * plateauFactor;
        if (lrMin > 0 && lrCur < lrMin) lrCur = lrMin;
        // Prevent denormals/zero.
        if (lrCur < 1e-12) lrCur = 1e-12;
        plateauSince = 0;
        self.postMessage({ type: "log", line: `LR plateau: ${old} → ${lrCur} at tree ${t}` });
      }
    }

    if (task === "binary_classification") {
      // =========================
      // Binary classification
      // =========================
      const y01 = y; // expected 0/1

      // Optional class-weighting (imbalance handling)
      const imbMode = String(imbalance?.mode || "none").trim().toLowerCase();
      const imbCapRaw = Number(imbalance?.cap);
      const imbCap = (Number.isFinite(imbCapRaw) && imbCapRaw > 0) ? imbCapRaw : 20;
      const imbNormalize = !!imbalance?.normalize;

      let wRow = null;
      let w0 = 1;
      let w1 = 1;
      if (imbMode === "auto" || imbMode === "manual") {
        let c0 = 0;
        let c1 = 0;
        for (let i = 0; i < trainIdx.length; i++) {
          const r = trainIdx[i];
          if (y01[r] >= 0.5) c1++; else c0++;
        }
        const N = c0 + c1;

        if (imbMode === "manual") {
          w0 = Number(imbalance?.w0);
          w1 = Number(imbalance?.w1);
          if (!Number.isFinite(w0) || w0 <= 0) w0 = 1;
          if (!Number.isFinite(w1) || w1 <= 0) w1 = 1;
        } else {
          if (c0 > 0) w0 = N / (2 * c0);
          if (c1 > 0) w1 = N / (2 * c1);
        }

        // Cap very large weights to keep training stable
        if (w0 > imbCap) w0 = imbCap;
        if (w1 > imbCap) w1 = imbCap;

        // Optional normalization keeps average weight ~1 (after capping).
        if (imbNormalize && N > 0) {
          const avg = (w0 * c0 + w1 * c1) / N;
          if (avg > 0) { w0 = w0 / avg; w1 = w1 / avg; }
        }

        wRow = new Float32Array(nRows);
        for (let r = 0; r < nRows; r++) wRow[r] = (y01[r] >= 0.5) ? w1 : w0;

        self.postMessage({ type: "log", line: `Imbalance: binary mode=${imbMode} w0=${w0.toFixed(3)} w1=${w1.toFixed(3)} cap=${imbCap} normalize=${imbNormalize}` });
      }

      // Base score is log-odds of the (optionally weighted) training positive rate.
      let sumW = 0;
      let sumWPos = 0;
      for (let i = 0; i < trainIdx.length; i++) {
        const r = trainIdx[i];
        const w = wRow ? wRow[r] : 1;
        sumW += w;
        sumWPos += w * ((y01[r] >= 0.5) ? 1 : 0);
      }
      let p0 = sumWPos / Math.max(1e-12, sumW);
      const EPS = 1e-6;
      if (p0 < EPS) p0 = EPS;
      else if (p0 > 1 - EPS) p0 = 1 - EPS;

      const baseLogit = Math.log(p0 / (1 - p0));
      const baseQ = clampI32(Math.round(baseLogit * scaleQ));

      const predQ = new Float64Array(nRows);
      predQ.fill(baseQ);

      const grad = new Float32Array(nRows);
      const hess = new Float32Array(nRows);

      let bestValMetric = Infinity;
      let bestTrainMetric = Infinity;
      let bestTestMetric = Infinity;
      let bestTrainAcc = 0;
      let bestValAcc = 0;
      let bestTestAcc = 0;
      let bestIter = 0;
      let sinceBest = 0;

      function refreshGradHessTrain() {
        for (let i = 0; i < trainIdx.length; i++) {
          const r = trainIdx[i];
          const logit = predQ[r] / scaleQ;
          const p = sigmoid(logit);
          const w = wRow ? wRow[r] : 1;
          grad[r] = (y01[r] - p) * w;
          hess[r] = (p * (1 - p)) * w;
        }
      }

      for (let t = 1; t <= maxTrees; t++) {
        if (stopFlag) break;

	    refreshGradHessTrain();
	    const lrUsed = lrForIter(t);
	    const tree = buildTreeBinary({ X, nRows, nFeatures, trainSamples: trainIdx, grad, hess, featMin, featRange, depth, minLeaf, lr: lrUsed, scaleQ, rng, bins, binning, qThr });
        trees.push(tree);

        applyTree(tree, trainIdx, predQ);
        applyTree(tree, val, predQ);
        applyTree(tree, test, predQ);

        const tr = loglossAcc(y01, predQ, trainIdx, scaleQ, wRow);
        const va = loglossAcc(y01, predQ, val, scaleQ, wRow);
        const te = loglossAcc(y01, predQ, test, scaleQ, wRow);

        const trainLoss = tr.loss;
        const valLoss = va.loss;
        const testLoss = te.loss;
        const trainAcc = tr.acc;
        const valAcc = va.acc;
        const testAcc = te.acc;

        let improved = false;
	    if (valLoss + 1e-12 < bestValMetric) {
          bestValMetric = valLoss;
          bestTrainMetric = trainLoss;
          bestTestMetric = testLoss;
          bestTrainAcc = trainAcc;
          bestValAcc = valAcc;
          bestTestAcc = testAcc;
          bestIter = t;
          sinceBest = 0;
          improved = true;
        } else {
          sinceBest += 1;
        }

	    // Reduce-on-plateau schedule update (uses validation improvement signal).
	    lrAfterMetric(improved, t);

        if (t % 5 === 0 || t === maxTrees || improved) {
          self.postMessage({
            type: "progress",
            task,
            metricName: "LogLoss",
            done: t,
            total: maxTrees,
            trainMetric: trainLoss,
            valMetric: valLoss,
            testMetric: testLoss,
            trainAcc,
            valAcc,
            testAcc,
            bestValMetric,
	        bestIter,
	        lr: lrUsed
          });
        }
        if (t % 5 === 0) {
          self.postMessage({
            type: "log",
            line: `Train ${t}/${maxTrees}: trainLoss=${trainLoss.toFixed(6)} valLoss=${valLoss.toFixed(6)} best=${bestValMetric.toFixed(6)}${improved ? " ★" : ""}`
          });
        }

        if (earlyStop && sinceBest >= patience) {
          self.postMessage({ type: "log", line: `Early stop at tree ${t} (best=${bestIter}, patience=${patience})` });
          break;
        }
      }

      const usedTrees = earlyStop ? Math.max(1, bestIter) : Math.max(1, trees.length);
      const finalTrees = trees.slice(0, usedTrees);
      const modelBytes = serializeModel({ nFeatures, depth, nTrees: usedTrees, baseQ, scaleQ, trees: finalTrees });

      const meta = {
        task,
        metricName: "LogLoss",
        nFeatures, depth, maxTrees, usedTrees,
        baseQ, scaleQ,
        bins, binning,
        bestIter: earlyStop ? bestIter : usedTrees,
        bestTrainMetric,
        bestValMetric,
        bestTestMetric,
        bestTrainLoss: bestTrainMetric,
        bestValLoss: bestValMetric,
        bestTestLoss: bestTestMetric,
        bestTrainAcc,
        bestValAcc,
        bestTestAcc,
        earlyStop
      };

      self.postMessage({ type: "done", modelBytes: modelBytes.buffer, meta }, [modelBytes.buffer]);
    } else if (task === "multiclass_classification") {
      // =========================
      // Multiclass classification (softmax)
      // =========================
      const nClasses = Math.max(2, p.nClasses | 0);
      const yK = y; // expected integer class indices in [0..nClasses-1]


      // Optional class-weighting (imbalance handling)
      const imbMode = String(imbalance?.mode || "none").trim().toLowerCase();
      const imbCap = Number.isFinite(Number(imbalance?.cap)) ? Number(imbalance.cap) : 20;
      const imbNormalize = !!imbalance?.normalize;

      let wRow = null;
      let wClass = null;
      let capUsed = 20;

      if (imbMode === "auto" || imbMode === "manual") {
        // Class counts on train split
        const counts = new Int32Array(nClasses);
        for (let i = 0; i < trainIdx.length; i++) {
          const r = trainIdx[i];
          const cls = yK[r] | 0;
          if (cls >= 0 && cls < nClasses) counts[cls] += 1;
        }
        const N = trainIdx.length;

        wClass = new Float32Array(nClasses);

        if (imbMode === "manual") {
          const manual = Array.isArray(imbalance?.classWeights) ? imbalance.classWeights : [];
          for (let k = 0; k < nClasses; k++) {
            let w = Number(manual[k]);
            if (!Number.isFinite(w) || w <= 0) w = 1;
            wClass[k] = w;
          }
        } else {
          for (let k = 0; k < nClasses; k++) {
            const c = counts[k];
            let w = (c > 0) ? (N / (nClasses * c)) : 1;
            wClass[k] = w;
          }
        }

        capUsed = (Number.isFinite(imbCap) && imbCap > 0) ? imbCap : 20;
        for (let k = 0; k < nClasses; k++) {
          if (wClass[k] > capUsed) wClass[k] = capUsed;
        }

        if (imbNormalize && N > 0) {
          let avg = 0;
          for (let k = 0; k < nClasses; k++) avg += wClass[k] * counts[k];
          avg /= N;
          if (avg > 0) {
            for (let k = 0; k < nClasses; k++) wClass[k] = wClass[k] / avg;
          }
        }

        wRow = new Float32Array(nRows);
        for (let r = 0; r < nRows; r++) {
          const cls = yK[r] | 0;
          wRow[r] = (cls >= 0 && cls < nClasses) ? wClass[cls] : 1;
        }

        let minW = Infinity;
        let maxW = 0;
        for (let k = 0; k < nClasses; k++) {
          const w = wClass[k];
          if (w < minW) minW = w;
          if (w > maxW) maxW = w;
        }

        self.postMessage({
          type: "log",
          line: `Imbalance: multiclass mode=${imbMode} minW=${minW.toFixed(3)} maxW=${maxW.toFixed(3)} cap=${capUsed} normalize=${imbNormalize}`
        });
      }

      // Base logits are log(prior) from training split (weighted if enabled).
      const smooth = 1e-3;
      const baseLogitsQ = new Int32Array(nClasses);
      const sumWClass = new Float64Array(nClasses);
      let sumW = 0;
      for (let i = 0; i < trainIdx.length; i++) {
        const r = trainIdx[i];
        const cls = yK[r] | 0;
        const w = wRow ? wRow[r] : 1;
        sumW += w;
        if (cls >= 0 && cls < nClasses) sumWClass[cls] += w;
      }
      const denom = Math.max(1e-9, sumW + smooth * nClasses);
      for (let k = 0; k < nClasses; k++) {
        let pk = (sumWClass[k] + smooth) / denom;
        if (pk < 1e-9) pk = 1e-9;
        baseLogitsQ[k] = clampI32(Math.round(Math.log(pk) * scaleQ));
      }

      // predQ is row-major [r*nClasses+k] in Q units
      const predQ = new Float64Array(nRows * nClasses);
      for (let r = 0; r < nRows; r++) {
        const base = r * nClasses;
        for (let k = 0; k < nClasses; k++) predQ[base + k] = baseLogitsQ[k];
      }

      const prob = new Float32Array(nRows * nClasses);
      softmaxProbs(predQ, nRows, nClasses, scaleQ, prob);

      const grad = new Float32Array(nRows);
      const hess = new Float32Array(nRows);

      const treesByClass = Array.from({ length: nClasses }, () => []);

      let bestValMetric = Infinity;
      let bestTrainMetric = Infinity;
      let bestTestMetric = Infinity;
      let bestTrainAcc = 0;
      let bestValAcc = 0;
      let bestTestAcc = 0;
      let bestIter = 0;
      let sinceBest = 0;

      function applyTreeClass(tree, indices, cls) {
        const k = cls | 0;
        for (let i = 0; i < indices.length; i++) {
          const r = indices[i];
          predQ[r * nClasses + k] += treePredictLeafQ(tree, r);
        }
      }

      for (let t = 1; t <= maxTrees; t++) {
        if (stopFlag) break;

        const lrUsed = lrForIter(t);

        for (let k = 0; k < nClasses; k++) {
          // Refresh grad/hess for this class on the train split using current probabilities.
          for (let i = 0; i < trainIdx.length; i++) {
            const r = trainIdx[i];
            const p_k = prob[r * nClasses + k];
            const yk = (yK[r] | 0) === k ? 1 : 0;
            const w = wRow ? wRow[r] : 1;
            grad[r] = (yk - p_k) * w;
            hess[r] = (p_k * (1 - p_k)) * w;
          }

	      const tree = buildTreeBinary({ X, nRows, nFeatures, trainSamples: trainIdx, grad, hess, featMin, featRange, depth, minLeaf, lr: lrUsed, scaleQ, rng, bins, binning, qThr });
          treesByClass[k].push(tree);
          applyTreeClass(tree, trainIdx, k);
          applyTreeClass(tree, val, k);
          applyTreeClass(tree, test, k);
        }

        // Update probabilities after this boosting round.
        softmaxProbs(predQ, nRows, nClasses, scaleQ, prob);

        const tr = loglossAccMulti(yK, prob, trainIdx, nClasses, wRow);
        const va = loglossAccMulti(yK, prob, val, nClasses, wRow);
        const te = loglossAccMulti(yK, prob, test, nClasses, wRow);

        const trainLoss = tr.loss;
        const valLoss = va.loss;
        const testLoss = te.loss;
        const trainAcc = tr.acc;
        const valAcc = va.acc;
        const testAcc = te.acc;

        let improved = false;
        if (valLoss + 1e-12 < bestValMetric) {
          bestValMetric = valLoss;
          bestTrainMetric = trainLoss;
          bestTestMetric = testLoss;
          bestTrainAcc = trainAcc;
          bestValAcc = valAcc;
          bestTestAcc = testAcc;
          bestIter = t;
          sinceBest = 0;
          improved = true;
        } else {
          sinceBest += 1;
        }

        // Reduce-on-plateau schedule update (uses validation improvement signal).
        lrAfterMetric(improved, t);

        if (t % 5 === 0 || t === maxTrees || improved) {
          self.postMessage({
            type: "progress",
            task,
            metricName: "LogLoss",
            done: t,
            total: maxTrees,
            trainMetric: trainLoss,
            valMetric: valLoss,
            testMetric: testLoss,
            trainAcc,
            valAcc,
            testAcc,
            bestValMetric,
            bestIter
          });
        }
        if (t % 5 === 0) {
          self.postMessage({
            type: "log",
            line: `Train ${t}/${maxTrees}: trainLoss=${trainLoss.toFixed(6)} valLoss=${valLoss.toFixed(6)} best=${bestValMetric.toFixed(6)}${improved ? " ★" : ""}`
          });
        }

        if (earlyStop && sinceBest >= patience) {
          self.postMessage({ type: "log", line: `Early stop at round ${t} (best=${bestIter}, patience=${patience})` });
          break;
        }
      }

      const itersDone = (treesByClass[0] || []).length;
      const usedTreesPerClass = earlyStop ? Math.max(1, bestIter) : Math.max(1, itersDone);
      const finalTreesByClass = treesByClass.map(arr => arr.slice(0, usedTreesPerClass));

      const modelBytes = serializeModelV2({
        nFeatures,
        depth,
        nClasses,
        treesPerClass: usedTreesPerClass,
        baseLogitsQ,
        scaleQ,
        treesByClass: finalTreesByClass
      });

      const meta = {
        task,
        metricName: "LogLoss",
        nFeatures,
        depth,
        scaleQ,
        bins,
        binning,
        nClasses,
        maxTrees,
        usedTrees: usedTreesPerClass,
        treesPerClass: usedTreesPerClass,
        totalTrees: usedTreesPerClass * nClasses,
        bestIter: earlyStop ? bestIter : usedTreesPerClass,
        bestTrainMetric,
        bestValMetric,
        bestTestMetric,
        bestTrainLoss: bestTrainMetric,
        bestValLoss: bestValMetric,
        bestTestLoss: bestTestMetric,
        bestTrainAcc,
        bestValAcc,
        bestTestAcc,
        earlyStop
      };

      self.postMessage({ type: "done", modelBytes: modelBytes.buffer, meta }, [modelBytes.buffer]);
    } else if (task === "multilabel_classification") {
      // =========================
      // Multilabel classification (independent sigmoid per label)
      // =========================
      const nLabels = Math.max(2, p.nClasses | 0);
      const yFlat = y; // expected 0/1 row-major (r*nLabels + k)

      // Count positives per label on train split (rows used for training).
      const posCount = new Int32Array(nLabels);
      for (let i = 0; i < trainIdx.length; i++) {
        const r = trainIdx[i];
        const base = r * nLabels;
        for (let k = 0; k < nLabels; k++) {
          if (yFlat[base + k] >= 0.5) posCount[k] += 1;
        }
      }

      // Optional per-label positive weights (pos_weight-style) for imbalance.
      const imbMode = String(imbalance?.mode || "none").trim().toLowerCase();
      const imbCap = Number.isFinite(Number(imbalance?.cap)) ? Number(imbalance.cap) : 20;
      const imbNormalize = !!imbalance?.normalize;

      let posW = null;
      let wScale = 1;
      if (imbMode === "auto" || imbMode === "manual") {
        posW = new Float32Array(nLabels);
        if (imbMode === "manual") {
          const arr = Array.isArray(imbalance?.posWeights) ? imbalance.posWeights : [];
          for (let k = 0; k < nLabels; k++) {
            let w = Number(arr[k]);
            if (!Number.isFinite(w) || w <= 0) w = 1;
            posW[k] = w;
          }
        } else {
          for (let k = 0; k < nLabels; k++) {
            const pos = posCount[k];
            const neg = trainIdx.length - pos;
            let w = 1;
            if (pos > 0) w = neg / pos;
            posW[k] = w;
          }
        }

        const cap = (Number.isFinite(imbCap) && imbCap > 0) ? imbCap : 20;
        for (let k = 0; k < nLabels; k++) {
          if (posW[k] > cap) posW[k] = cap;
        }

        if (imbNormalize && trainIdx.length > 0) {
          // Global scaling to keep average weight ~1 across (row,label) pairs.
          let wSum = 0;
          for (let k = 0; k < nLabels; k++) {
            const pos = posCount[k];
            const neg = trainIdx.length - pos;
            wSum += neg + posW[k] * pos;
          }
          const avg = wSum / (trainIdx.length * nLabels);
          if (avg > 0) wScale = 1 / avg;
        }

        // Log a small summary.
        let minW = Infinity;
        let maxW = 0;
        for (let k = 0; k < nLabels; k++) {
          const w = posW[k];
          if (w < minW) minW = w;
          if (w > maxW) maxW = w;
        }
        self.postMessage({ type: "log", line: `Imbalance: multilabel mode=${imbMode} posW[min,max]=${minW.toFixed(3)},${maxW.toFixed(3)} cap=${cap} normalize=${imbNormalize}` });
      }

      // Base logit per label is log-odds of the (weighted) training positive rate for that label.
      const baseLogitsQ = new Int32Array(nLabels);
      for (let k = 0; k < nLabels; k++) {
        const pos = posCount[k];
        const neg = trainIdx.length - pos;

        let p0 = 0.5;
        if (trainIdx.length > 0) {
          if (posW) {
            const num = posW[k] * pos;
            const den = neg + posW[k] * pos;
            p0 = num / Math.max(1e-12, den);
          } else {
            p0 = pos / Math.max(1, trainIdx.length);
          }
        }

        const EPS = 1e-6;
        if (p0 < EPS) p0 = EPS;
        else if (p0 > 1 - EPS) p0 = 1 - EPS;

        const baseLogit = Math.log(p0 / (1 - p0));
        baseLogitsQ[k] = clampI32(Math.round(baseLogit * scaleQ));
      }

      // predQ is row-major [r*nLabels+k] in Q units (logits).
      const predQ = new Float64Array(nRows * nLabels);
      for (let r = 0; r < nRows; r++) {
        const base = r * nLabels;
        for (let k = 0; k < nLabels; k++) predQ[base + k] = baseLogitsQ[k];
      }

      const grad = new Float32Array(nRows);
      const hess = new Float32Array(nRows);

      const treesByClass = Array.from({ length: nLabels }, () => []);

      let bestValMetric = Infinity;
      let bestTrainMetric = Infinity;
      let bestTestMetric = Infinity;
      let bestTrainAcc = 0;
      let bestValAcc = 0;
      let bestTestAcc = 0;
      let bestIter = 0;
      let sinceBest = 0;

      function applyTreeLabel(tree, indices, k) {
        const kk = k | 0;
        for (let i = 0; i < indices.length; i++) {
          const r = indices[i];
          predQ[r * nLabels + kk] += treePredictLeafQ(tree, r);
        }
      }

      for (let t = 1; t <= maxTrees; t++) {
        if (stopFlag) break;

        const lrUsed = lrForIter(t);

        for (let k = 0; k < nLabels; k++) {
          // Refresh grad/hess for this label on the train split using current logits.
          for (let i = 0; i < trainIdx.length; i++) {
            const r = trainIdx[i];
            const logit = predQ[r * nLabels + k] / scaleQ;
            const p = sigmoid(logit);
            const yk = (yFlat[r * nLabels + k] >= 0.5) ? 1 : 0;
            const w = posW ? ((yk ? posW[k] : 1) * wScale) : 1;
            grad[r] = (yk - p) * w;
            hess[r] = (p * (1 - p)) * w;
          }

	      const tree = buildTreeBinary({ X, nRows, nFeatures, trainSamples: trainIdx, grad, hess, featMin, featRange, depth, minLeaf, lr: lrUsed, scaleQ, rng, bins, binning, qThr });
          treesByClass[k].push(tree);
          applyTreeLabel(tree, trainIdx, k);
          applyTreeLabel(tree, val, k);
          applyTreeLabel(tree, test, k);
        }

        const tr = loglossAccMultiLabel(yFlat, predQ, trainIdx, nLabels, scaleQ, posW, wScale);
        const va = loglossAccMultiLabel(yFlat, predQ, val, nLabels, scaleQ, posW, wScale);
        const te = loglossAccMultiLabel(yFlat, predQ, test, nLabels, scaleQ, posW, wScale);

        const trainLoss = tr.loss;
        const valLoss = va.loss;
        const testLoss = te.loss;
        const trainAcc = tr.acc;
        const valAcc = va.acc;
        const testAcc = te.acc;

        let improved = false;
        if (valLoss + 1e-12 < bestValMetric) {
          bestValMetric = valLoss;
          bestTrainMetric = trainLoss;
          bestTestMetric = testLoss;
          bestTrainAcc = trainAcc;
          bestValAcc = valAcc;
          bestTestAcc = testAcc;
          bestIter = t;
          sinceBest = 0;
          improved = true;
        } else {
          sinceBest += 1;
        }

        // Reduce-on-plateau schedule update (uses validation improvement signal).
        lrAfterMetric(improved, t);

        if (t % 5 === 0 || t === maxTrees || improved) {
          self.postMessage({
            type: "progress",
            task,
            metricName: "LogLoss",
            done: t,
            total: maxTrees,
            trainMetric: trainLoss,
            valMetric: valLoss,
            testMetric: testLoss,
            trainAcc,
            valAcc,
            testAcc,
            bestValMetric,
            bestIter
          });
        }
        if (t % 5 === 0) {
          self.postMessage({
            type: "log",
            line: `Train ${t}/${maxTrees}: trainLoss=${trainLoss.toFixed(6)} valLoss=${valLoss.toFixed(6)} best=${bestValMetric.toFixed(6)}${improved ? " ★" : ""}`
          });
        }

        if (earlyStop && sinceBest >= patience) {
          self.postMessage({ type: "log", line: `Early stop at round ${t} (best=${bestIter}, patience=${patience})` });
          break;
        }
      }

      const itersDone = (treesByClass[0] || []).length;
      const usedTreesPerLabel = earlyStop ? Math.max(1, bestIter) : Math.max(1, itersDone);
      const finalTreesByLabel = treesByClass.map(arr => arr.slice(0, usedTreesPerLabel));

      const modelBytes = serializeModelV2({
        nFeatures,
        depth,
        nClasses: nLabels,
        treesPerClass: usedTreesPerLabel,
        baseLogitsQ,
        scaleQ,
        treesByClass: finalTreesByLabel
      });

      const meta = {
        task,
        metricName: "LogLoss",
        nFeatures,
        depth,
        scaleQ,
        bins,
        binning,
        nClasses: nLabels,
        maxTrees,
        usedTrees: usedTreesPerLabel,
        treesPerClass: usedTreesPerLabel,
        totalTrees: usedTreesPerLabel * nLabels,
        bestIter: earlyStop ? bestIter : usedTreesPerLabel,
        bestTrainMetric,
        bestValMetric,
        bestTestMetric,
        bestTrainLoss: bestTrainMetric,
        bestValLoss: bestValMetric,
        bestTestLoss: bestTestMetric,
        bestTrainAcc,
        bestValAcc,
        bestTestAcc,
        earlyStop
      };

      self.postMessage({ type: "done", modelBytes: modelBytes.buffer, meta }, [modelBytes.buffer]);
    } else {
      // =========================
      // Regression
      // =========================
      const yQ = new Float64Array(nRows);
      for (let i = 0; i < nRows; i++) yQ[i] = clampI32(Math.round(y[i] * scaleQ));

      let sumY = 0;
      for (let i = 0; i < trainIdx.length; i++) sumY += yQ[trainIdx[i]];
      const baseQ = clampI32(Math.round(sumY / Math.max(1, trainIdx.length)));

      const predQ = new Float64Array(nRows);
      predQ.fill(baseQ);

      const residual = new Float32Array(nRows);

      let bestValMSE = Infinity;
      let bestTrainMSE = Infinity;
      let bestTestMSE = Infinity;
      let bestIter = 0;
      let sinceBest = 0;

      function refreshResidualTrain() {
        for (let i = 0; i < trainIdx.length; i++) {
          const r = trainIdx[i];
          residual[r] = (yQ[r] - predQ[r]) / scaleQ;
        }
      }

      for (let t = 1; t <= maxTrees; t++) {
        if (stopFlag) break;

        refreshResidualTrain();
        const lrUsed = lrForIter(t);
        const tree = buildTreeRegression({ X, nRows, nFeatures, trainSamples: trainIdx, residual, featMin, featRange, depth, minLeaf, lr: lrUsed, scaleQ, rng, bins, binning, qThr });
        trees.push(tree);

        applyTree(tree, trainIdx, predQ);
        applyTree(tree, val, predQ);
        applyTree(tree, test, predQ);

        const trainMSE = mse(yQ, predQ, trainIdx, scaleQ);
        const valMSE = mse(yQ, predQ, val, scaleQ);
        const testMSE = mse(yQ, predQ, test, scaleQ);

        let improved = false;
        if (valMSE + 1e-12 < bestValMSE) {
          bestValMSE = valMSE;
          bestTrainMSE = trainMSE;
          bestTestMSE = testMSE;
          bestIter = t;
          sinceBest = 0;
          improved = true;
        } else {
          sinceBest += 1;
        }

        // Reduce-on-plateau schedule update (uses validation improvement signal).
        lrAfterMetric(improved, t);

        if (t % 5 === 0 || t === maxTrees || improved) {
          self.postMessage({
            type: "progress",
            task,
            metricName: "MSE",
            done: t,
            total: maxTrees,
            trainMetric: trainMSE,
            valMetric: valMSE,
            testMetric: testMSE,
            bestValMetric: bestValMSE,
            bestIter,
            // Back-compat field names
            trainMSE,
            valMSE,
            testMSE,
            bestValMSE
          });
        }
        if (t % 5 === 0) {
          self.postMessage({ type: "log", line: `Train ${t}/${maxTrees}: trainMSE=${trainMSE.toFixed(6)} valMSE=${valMSE.toFixed(6)} best=${bestValMSE.toFixed(6)}${improved ? " ★" : ""}` });
        }

        if (earlyStop && sinceBest >= patience) {
          self.postMessage({ type: "log", line: `Early stop at tree ${t} (best=${bestIter}, patience=${patience})` });
          break;
        }
      }

      const usedTrees = earlyStop ? Math.max(1, bestIter) : Math.max(1, trees.length);
      const finalTrees = trees.slice(0, usedTrees);
      const modelBytes = serializeModel({ nFeatures, depth, nTrees: usedTrees, baseQ, scaleQ, trees: finalTrees });

      const meta = {
        task,
        metricName: "MSE",
        nFeatures, depth, maxTrees, usedTrees,
        baseQ, scaleQ,
        bins, binning,
        bestIter: earlyStop ? bestIter : usedTrees,
        bestTrainMetric: bestTrainMSE,
        bestValMetric: bestValMSE,
        bestTestMetric: bestTestMSE,
        // Back-compat field names
        bestTrainMSE,
        bestValMSE,
        bestTestMSE,
        earlyStop
      };

      self.postMessage({ type: "done", modelBytes: modelBytes.buffer, meta }, [modelBytes.buffer]);
    }

  } catch (e) {
    self.postMessage({ type: "error", message: e?.message || String(e) });
  }
};

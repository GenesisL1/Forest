<!--
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
-->

# GenesisL1 Forest (GL1F)
Link: https://GL1F.com

A browser-only studio for:
- training GBDT models (in-browser): regression + binary + multiclass + multilabel classification
- deploying models on GenesisL1 as ERC-721 Model NFTs
- on-chain inference (view call + optional paid tx)
- AI store listing/buying
- on-chain Terms + Creative Commons license enforcement
- on-chain title-word search index

## Run locally
You must serve this folder over HTTP (not file://) so module imports work.

### Option A: Python
```bash
python3 -m http.server 8080
```
Open:
- http://localhost:8080/
- http://localhost:8080/forest.html

## Network
GenesisL1:
- chainId: 29
- default RPC: https://rpc.genesisl1.org

## Notes
- Model bytes are stored as chunk contracts. Pointer-table and chunks use runtime magic "GL1C".
- Model format "GL1F" v1 is used for regression + binary classification.
- Model format "GL1F" v2 is used for multiclass + multilabel classification (vector-output logits).
- Model metadata (title/description/icon/features) is stored on-chain in the NFT.

# GenesisL1 Forest Model (GL1F) — Architecture & Design

This document describes the **model itself** (not the UI or the trainer servers): the GL1F formats, inference rules, and the training procedure used to produce deterministic, on-chain-friendly Gradient Boosted Decision Trees (GBDT).

---

## 1) Design goals

### Deterministic execution
The model must produce identical predictions across:
- browser (WebWorker + local inference),
- Python trainer,
- C++ trainer,
- EVM on-chain runtime.

Determinism is achieved by:
- seeded RNG (`xorshift32`) for all randomized choices,
- fixed split/search procedure,
- explicit rounding rules compatible with JavaScript `Math.round`,
- fixed model layout (no pointer-heavy structures).

### On-chain friendliness
The on-chain runtime must be:
- bounded-time (no unbounded recursion, no dynamic tree growth at inference),
- integer-only operations (no floats on-chain),
- compact enough to store as EVM bytecode chunks.

Solutions:
- **fixed-depth** complete binary trees,
- **int32 quantization** (`Q` units) for thresholds and leaf values,
- simple `xQ > thrQ` branching,
- model bytes chunking for on-chain storage (GL1C chunks + pointer table).

---

## 2) What the model is

GL1F is a **GBDT ensemble** of fixed-depth, axis-aligned decision trees.

There are two model versions:

### v1 (scalar output)
Used for:
- regression
- binary classification (outputs a single **logit**)

Output: a single integer value `scoreQ` in **Q-units**.

### v2 (vector output)
Used for:
- multiclass classification (outputs logits per class)
- multilabel classification (outputs independent logits per label)

Output: an array of integers `logitsQ[k]` in **Q-units** (one per class/label).

---

## 3) Numeric representation (Q-units)

The model stores all thresholds and leaf values as signed 32-bit integers (`int32`) representing:

```
valueQ = round(valueFloat * scaleQ)
valueFloat ≈ valueQ / scaleQ
```

Where:
- `scaleQ` is stored in the model header (`uint32`).
- Features at inference are quantized the same way:
  - `xQ = round(x * scaleQ)` clamped into int32 range.

### Rounding rule (important)
All implementations (JS/Python/C++) match:

```
Math.round(x)  <=>  floor(x + 0.5)   for finite x
```

This matches JavaScript `Math.round`, including negative half cases (e.g. -1.5 → -1).

### Choosing scaleQ
Higher `scaleQ` increases precision but risks overflow in:
- quantized inputs `xQ`
- accumulated logits during inference

Practical guidance:
- keep `abs(x * scaleQ)` safely within `≈ 2.147e9` (int32 range),
- keep accumulated logits within JS safe integer range (≈ 2^53) if doing off-chain inference in JS numbers,
- on-chain inference uses `int256` accumulators, so it is typically safe for larger totals.

---

## 4) Tree structure

All trees are **complete binary trees** of fixed `depth`.

Let:
- `L = 2^depth` leaves
- `I = 2^depth - 1` internal nodes

The tree is stored in arrays:
- internal nodes indexed `idx = 0 .. I-1`
- leaves indexed `idx = I .. I+L-1`

Traversal starts at `idx=0` (root). At each level:

```
goRight = (xQ[feature] > thresholdQ)
idx = goRight ? (2*idx + 2) : (2*idx + 1)
```

After `depth` decisions, `idx` points to a leaf node:
```
leafIndex = idx - I
leafValueQ = leaf[leafIndex]
```

### “Forced leaf” nodes
If the trainer cannot find a valid split (or leaf constraints prevent splitting),
it fills the remaining subtree with:
- `feature = 0`
- `threshold = INT32_MAX`
- identical leaf values copied downwards

This preserves fixed-depth layout and deterministic inference.

---

## 5) GL1F binary formats

All fields are **little-endian**.

### 5.1) GL1F v1 header (24 bytes)

| Offset | Size | Type   | Meaning |
|-------:|-----:|--------|---------|
| 0      | 4    | bytes  | magic `"GL1F"` |
| 4      | 1    | u8     | version = 1 |
| 5      | 1    | u8     | reserved |
| 6      | 2    | u16    | `nFeatures` |
| 8      | 2    | u16    | `depth` |
| 10     | 4    | u32    | `nTrees` |
| 14     | 4    | i32    | `baseQ` (base prediction) |
| 18     | 4    | u32    | `scaleQ` |
| 22     | 2    | u16    | reserved |

After the header come `nTrees` trees.

### 5.2) GL1F v1 tree layout

For each tree:

1) Internal nodes: `I * 8` bytes  
Each internal node is 8 bytes:
- `u16 featureIndex` at offset +0
- `i32 thresholdQ` at offset +2
- 2 bytes padding/reserved at offset +6

2) Leaves: `L * 4` bytes  
Each leaf:
- `i32 leafValueQ`

So each tree is:
```
perTreeBytes = I*8 + L*4
```

### 5.3) GL1F v2 header

v2 is similar but adds:
- vector output size (`nClasses`)
- base logits per class/label
- trees arranged class-major

Header:

| Offset | Size | Type | Meaning |
|-------:|-----:|------|---------|
| 0      | 4    | bytes | magic `"GL1F"` |
| 4      | 1    | u8   | version = 2 |
| 5      | 1    | u8   | reserved |
| 6      | 2    | u16  | `nFeatures` |
| 8      | 2    | u16  | `depth` |
| 10     | 4    | u32  | `treesPerClass` |
| 14     | 4    | i32  | reserved |
| 18     | 4    | u32  | `scaleQ` |
| 22     | 2    | u16  | `nClasses` (>=2) |
| 24..   | 4*nClasses | i32[] | `baseLogitsQ[k]` |

After base logits come `treesPerClass * nClasses` trees, stored **class-major**:

```
for class k in 0..nClasses-1:
  for t in 0..treesPerClass-1:
    write tree(k,t)
```

Tree layout is identical to v1 (internal nodes + leaves).

---

## 6) Inference rules

### 6.1) v1 inference (regression / binary logit)
Compute:

```
accQ = baseQ
for each tree:
  accQ += leafValueQ(tree, featuresQ)
return accQ
```

Interpretation:
- Regression: `y ≈ accQ / scaleQ`
- Binary classification:
  - logit = `accQ / scaleQ`
  - probability = `sigmoid(logit)`
  - class = `prob >= 0.5` (equivalently `logit >= 0`)

### 6.2) v2 inference (multiclass / multilabel)
For each class/label `k`:

```
logitsQ[k] = baseLogitsQ[k] + sum_t leafValueQ(tree(k,t), featuresQ)
```

Interpretation:
- Multiclass: `argmax_k logitsQ[k]`
  - optional probabilities: `softmax(logitsQ/scaleQ)`
- Multilabel: independent per label:
  - `p_k = sigmoid(logitsQ[k] / scaleQ)`
  - label active if `p_k >= threshold` (often 0.5 => logit >= 0)

### 6.3) Overflow considerations
- Stored values are int32.
- Accumulation:
  - Browser local inference uses **JS numbers** for v2 logits to avoid int32 wrap.
  - On-chain runtime should use `int256` accumulators.

---

## 7) Training procedure (GBDT)

Training produces GL1F bytes by boosting fixed-depth trees.

### 7.1) Dataset split
Rows are shuffled by a seeded RNG (`xorshift32`) and split into:
- train
- validation
- test

A stratified split is used for single-label classification when enabled.

### 7.2) Feature sub-sampling
At each node split, a random subset of features is sampled:
- `colsample = round(sqrt(nFeatures))` (at least 1)

This reduces correlation and speeds up search while keeping determinism via seeded RNG.

### 7.3) Candidate thresholds via histogram binning
For each candidate feature:
- compute a bin index per row using either:

**Linear binning**
```
b = floor( ((x - min) / range) * BINS )
```

**Quantile binning**
- precompute `(BINS-1)` thresholds per feature from a deterministic sample of the train split,
- bin is found by binary-searching thresholds.

For each bin boundary `b = 0..BINS-2`, the threshold used is:
- linear: `min + range * ((b+1)/BINS)`
- quantile: `thr[b]`

The stored `thresholdQ` is `round(threshold * scaleQ)`.

### 7.4) Regression objective (squared loss)
- Base prediction: mean of `y` on training rows:
  ```
  baseQ = round(mean(y_train) * scaleQ)
  ```
- Residual: `r = y - pred`
- Leaf weight for a node: mean residual scaled by learning rate:
  ```
  leaf = lr * mean(residuals_in_leaf)
  leafQ = round(leaf * scaleQ)
  ```
- Split score: minimize SSE (sum of squared errors) of residuals:
  ```
  SSE = SSE_left + SSE_right
  ```

### 7.5) Binary classification objective (log loss)
- Base logit: log-odds of (optionally weighted) positive rate:
  ```
  p0 = clamp(mean(y_train), [1e-6, 1-1e-6])
  baseLogit = log(p0/(1-p0))
  baseQ = round(baseLogit * scaleQ)
  ```
- Uses Newton-style updates:
  ```
  grad = (y - p)
  hess = p*(1-p)
  ```
  (optionally multiplied by per-row weights)
- Leaf weight:
  ```
  w = lr * sum(grad) / (sum(hess) + lambda)
  ```
  with `lambda = 1.0`.
- Split gain:
  ```
  gain = G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G_P^2/(H_P+λ)
  ```
  where `G` is sum of gradients and `H` sum of hessians for the node.

A split is accepted only if `gain > 0` and both sides have at least `minLeaf` rows.

### 7.6) Multiclass classification
Vector-output boosting (v2).

- Base logits: log of class priors (with numerical safeguards).
- Each boosting round trains one tree **per class** using class-specific gradients/hessians
computed from current softmax probabilities.

Trees are stored class-major:
- `treesPerClass` rounds,
- total trees = `treesPerClass * nClasses`.

### 7.7) Multilabel classification
Also v2.

- Each label is treated as an independent logistic head.
- Base logits per label from label-wise positive rates.
- Each boosting round trains one tree per label using the same binary Newton scheme.

### 7.8) Early stopping and LR schedules
Training can optionally:
- early-stop based on validation metric (patience),
- adjust learning rate (e.g., plateau schedule).

Important: because early stopping decisions depend on floating-point metric values,
absolute bit-identical results across machines require:
- the same metric computations,
- and no ties/near-ties where extremely tiny float differences flip the “best” iteration.

In practice, the implementation is designed so that JS/Python/C++ produce **bit-identical model bytes**
given identical inputs and parameters (validated in parity tests).

---

## 8) Reproducibility & cross-implementation parity

If you want **byte-identical** GL1F outputs across:
- WebWorker (JS),
- Python,
- C++,

ensure:
1) same feature order and same selected columns,
2) same class/label ordering (binary/multiclass),
3) same `seed`, `splitTrain/splitVal`, `refitTrainVal`, etc.,
4) same binning mode and `bins`/`quantileSamples`,
5) compare **core model bytes** (exclude any optional GL1X footer).

---

## 9) Optional packaging: GL1X footer

A `.gl1f` file may contain:
- core GL1F model bytes
- followed by an optional `GL1X` footer with JSON metadata (mint info, feature names, etc.)

This footer is **not used for inference**, and should be stripped before computing `modelId`
(so deployment matches the on-chain runtime expectations).

---

## 10) Limitations (model-level)
- Numeric features only (no native categorical handling).
- Missing values are typically dropped at preprocessing time.
- Fixed depth (no leaf-wise growth like LightGBM).
- No monotonic constraints (can be added later if desired).
- No per-feature normalization stored in-model (must be handled by the data pipeline).

---

## 11) Why this architecture works well on-chain
- Fixed-depth trees => predictable gas and runtime complexity.
- Integer thresholds/leaf values => deterministic behavior across EVM nodes.
- Compact contiguous layout => efficient bytecode storage and fast decoding.
- Vector-output v2 => multiclass/multilabel without separate model assets.


## License

This project is released under the MIT License. See the `LICENSE` file for details.

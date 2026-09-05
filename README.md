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
Website: <https://gl1f.com>

A browser studio and command-line toolchain for:

- training GBDT models for regression, binary, multiclass, and multilabel tasks;
- serializing models into the canonical GL1F format;
- publishing model bytes on GenesisL1 with custom transferable,
  ERC-721-like ownership records;
- local or EVM integer inference; and
- optional marketplace discovery and paid contract calls.

## Research paper v0.2.3

The canonical [research paper](GL1F.pdf) is a single-column technical report
released on 5 September 2026. It has not been peer reviewed. The
[reproducibility guide](REPRODUCIBILITY.md), [format specification](FORMAT_SPEC.md),
tests, and checked-in result records define the public evidence boundary.

Version 0.2.3 does not change or redeploy any contract.

```bash
npm ci
make verify
```

## Run locally

Serve this folder over HTTP (not `file://`) so module imports work.

### Python

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

---

# Web-exact hyperparameter search — `gl1f_search.py`

`gl1f_search.py` is the **command-line equivalent of the Forest studio's heuristic search button**. It is intended for users who want the same search behavior as the browser UI, but want to run it from a terminal against the local Python or C++ GL1F trainer.

This script is deliberately different from a general-purpose optimizer. It does **not** run random search, Optuna, simulated annealing, or an evolutionary strategy. Instead, it mirrors the web UI's small deterministic heuristic loop:

1. **Round 1 trains exactly the current/base UI parameters.**
2. **Rounds 2..N generate one heuristic candidate at a time.**
3. Each new candidate is generated from the current best parameters 75% of the time, otherwise from the original base parameters.
4. The search keeps whichever completed round is best according to the selected footer metric.
5. If requested, a final `--refit-train-val` pass is run only after the best round is chosen.

Use this script when you want CLI convenience while keeping the same first-run semantics, perturbation formulas, size clamp, seeded RNG, and best-round selection style as the web UI.

## What problem it solves

The browser studio is excellent for interactive training, but a hyperparameter search can be inconvenient in the browser when:

- a dataset is large,
- the C++ trainer is faster locally,
- you want reproducible shell commands,
- you want a saved leaderboard,
- you want to run exact first-run parameters from a JSON object,
- you want to choose a target metric such as validation loss, test loss, or validation accuracy.

`gl1f_search.py` solves that by driving the same trainer CLI repeatedly and reading each resulting `.gl1f` file's `GL1X` footer to decide which round is best.

## Web parity: what is matched

The script matches the web heuristic search behavior in the places that matter for reproducing the UI search sequence:

- **Round 1 is the exact base/current UI parameter object.**
- The search RNG is `xorshift32`.
- The search RNG seed is:

  ```text
  seed ^ 0x9e3779b9
  ```

- Round `1` uses `baseParams` directly.
- Rounds `2..N` use the web-style heuristic candidate generator.
- Candidate pivot is selected as:

  ```text
  pivot = bestParams with probability 0.75, otherwise baseParams
  ```

- Search changes only the web-search hyperparameters:
  - `trees`
  - `depth`
  - `lr`
  - `minLeaf`
  - `patience` when early stopping is enabled
  - plateau LR schedule fields when a plateau schedule is enabled

- Search keeps these fields fixed for comparability:
  - `task`
  - `seed`
  - `splitTrain`
  - `splitVal`
  - `bins`
  - `binning`
  - `earlyStop`
  - `nClasses`
  - class-imbalance settings
  - selected columns and labels
  - dataset input

- It uses JavaScript-compatible rounding for candidate generation. Python's built-in `round()` uses banker's rounding, so the script uses its own `js_round()` equivalent of JavaScript `Math.round`.

## Round 1: the initial parameters

In the web UI, the first round is whatever the user has currently set in the training form. The CLI version has the same concept: **round 1 is the initial/base parameter object**.

There are two ways to provide it.

### Option A: normal CLI flags

If you pass normal flags, those flags form the round-1 parameter object:

```bash
python gl1f_search.py \
    --engine cpp \
    --task regression \
    --input data/btc_vol.csv \
    --label-col y \
    --trees 300 \
    --depth 5 \
    --lr 0.08 \
    --min-leaf 7 \
    --bins 64 \
    --binning quantile \
    --seed 123 \
    --early-stop \
    --patience 30 \
    --split-train 0.75 \
    --split-val 0.15 \
    --trials 20 \
    --best-by bestValMetric \
    --out best.gl1f \
    --work runs/web_exact
```

Round 1 trains those values, after the same web-style clamping described below.

### Option B: exact web-style JSON object

Use `--initial-params` when you want to paste the same object shape used by the web state. This is the most explicit way to say: “start exactly from these UI params.”

```bash
python gl1f_search.py \
    --engine cpp \
    --task regression \
    --input data/btc_vol.csv \
    --label-col y \
    --initial-params '{"trees":300,"depth":5,"lr":0.08,"minLeaf":7,"bins":64,"binning":"quantile","seed":123,"earlyStop":true,"patience":30,"splitTrain":0.75,"splitVal":0.15}' \
    --trials 20 \
    --best-by bestValMetric \
    --out best.gl1f \
    --work runs/web_exact
```

`--initial-params` accepts:

- inline JSON:

  ```bash
  --initial-params '{"trees":250,"depth":4,"lr":0.05}'
  ```

- a direct file path:

  ```bash
  --initial-params params.json
  ```

- an `@file` path:

  ```bash
  --initial-params @params.json
  ```

It also accepts a leaderboard-style object containing `params`, or an object containing `best.params`, so you can reuse previous search output.

### Default initial parameters

If you pass neither `--initial-params` nor explicit hyperparameter flags, round 1 uses the same practical defaults as the web training form:

| Parameter | Default | Meaning |
|---|---:|---|
| `trees` | `250` | Number of boosting trees, or trees per class/head for v2 tasks. |
| `depth` | `4` | Fixed tree depth. |
| `lr` | `0.05` | Learning rate. |
| `minLeaf` / `--min-leaf` | `10` | Minimum rows per leaf. |
| `bins` | `32` | Histogram bins for split search. |
| `binning` | `linear` | `linear` or `quantile`. |
| `seed` | `42` | Training split/trainer seed and basis for search RNG. |
| `earlyStop` | `true` | Enable early stopping. |
| `patience` | `25` | Early-stopping patience. |
| `splitTrain` | `0.7` | Train split fraction. |
| `splitVal` | `0.2` | Validation split fraction. |
| `nClasses` | `2` | Class/label count used for multiclass/multilabel size clamping. |
| `scaleQ` | `auto` | Trainer quantization scale selection. |
| `chainId` | `29` | GenesisL1 chain id recorded by the trainer package. |

To print the actual clamped round-1 object without training, use:

```bash
python gl1f_search.py \
    --task regression \
    --input data/btc_vol.csv \
    --label-col y \
    --initial-params '{"trees":300,"depth":5,"lr":0.08,"minLeaf":7}' \
    --print-initial-params
```

This is useful because the script intentionally applies web-style clamps before it trains.

## Web-style clamping

Before round 1, and again after each generated candidate, the script clamps parameters into the same UI-safe ranges:

| Parameter | Clamp |
|---|---:|
| `trees` | `10..5000` for v1 tasks; at least `1` for multiclass/multilabel internal trees-per-class logic. |
| `depth` | `1..12` |
| `lr` | `0.001..1.0` |
| `minLeaf` | `1..1000` |
| `bins` | `8..512` |
| `patience` | `1..500` |
| `seed` | `1..2147483647` |
| `splitTrain` | `50%..90%` |
| `splitVal` | `5%..40%`, adjusted so train + validation leaves a non-trivial test split. |

It also applies the web model-size safety clamp. The estimated core model size must fit below the configured web size limit. If a candidate is too large, the script reduces `trees` in steps of 25 and then reduces `depth` if needed.

## Candidate-generation formulas

For round `r >= 2`, a candidate is generated from either the current best parameters or the base parameters. The formulas are intentionally small local moves, not a broad global search.

### Trees

```text
treesFactor = 2 ** ((rand() - 0.5) * 1.4)
trees = round((pivot.trees * treesFactor) / 25) * 25
trees = clamp(trees, 10, 5000)
```

This gives approximately a `×0.62..×1.62` move and snaps tree counts to multiples of 25.

### Depth

```text
depthStep = Math.round((rand() - 0.5) * 4)
depth = clamp(pivot.depth + depthStep, 1, 12)
```

This usually moves depth by about `-2..+2`.

### Learning rate

```text
lrFactor = 10 ** ((rand() - 0.5) * 0.8)
lr = clamp(pivot.lr * lrFactor, 0.001, 1.0)
lr = Math.round(lr * 1_000_000) / 1_000_000
```

This gives approximately a `×0.40..×2.51` move and rounds to six decimals.

### Minimum leaf size

```text
minLeafFactor = 2 ** ((rand() - 0.5) * 2.0)
minLeaf = clamp(Math.round(pivot.minLeaf * minLeafFactor), 1, 1000)
```

This gives approximately a `×0.5..×2.0` move.

### Early-stopping patience

Patience is perturbed only when early stopping is enabled:

```text
patienceFactor = 2 ** ((rand() - 0.5) * 1.6)
patience = clamp(Math.round((pivot.patience * patienceFactor) / 5) * 5, 1, 500)
```

This gives approximately a `×0.57..×1.74` move and snaps patience to multiples of 5.

### Plateau LR schedule

If `lrSchedule.mode == "plateau"`, the script also perturbs:

- plateau patience,
- drop percentage,
- minimum learning rate.

Piecewise schedules are passed through, but the candidate generator does not rewrite the segment string.

## Choosing the metric used to select the best model

By default, the script chooses the best round using:

```text
bestValMetric
```

This mirrors the usual training objective:

- regression: validation MSE, lower is better,
- classification: validation log loss, lower is better.

You can select the objective with any of these equivalent flags:

```bash
--metric-key bestValMetric
--best-by bestValMetric
--objective-metric bestValMetric
--target-metric bestValMetric
```

The script reads the chosen key from:

```text
local.trainMeta.<metricKey>
```

inside the `.gl1f` file's optional `GL1X` footer.

### Metric aliases

Convenient aliases are supported:

| Alias | Footer key | Direction |
|---|---|---|
| `val`, `validation`, `val_loss` | `bestValMetric` | `min` |
| `test`, `test_loss` | `bestTestMetric` | `min` |
| `train`, `train_loss` | `bestTrainMetric` | `min` |
| `val_acc`, `val_accuracy` | `bestValAcc` | `max` |
| `test_acc`, `test_accuracy` | `bestTestAcc` | `max` |
| `train_acc`, `train_accuracy` | `bestTrainAcc` | `max` |

Raw footer keys also work. For example:

```bash
--best-by bestTestMetric
--best-by bestValAcc
```

### Direction: min or max

Direction is automatic:

- accuracy-like metrics use `max`,
- loss/error/metric keys use `min`.

You can override this:

```bash
--best-by bestValAcc --direction max
--best-by bestValMetric --direction min
```

The leaderboard stores both the raw metric and an internal score. For `min`, score is the raw value. For `max`, score is `-raw`, so sorting still puts the best trial first.

### Target stopping

`--target` is evaluated against the selected metric and selected direction.

For loss/error metrics:

```bash
# stop once bestValMetric <= 0.01
--best-by bestValMetric --target 0.01
```

For accuracy metrics:

```bash
# stop once validation accuracy >= 0.95
--best-by val_acc --target 0.95
```

## Refit behavior

`--refit-train-val` follows the web pattern: it is **not used during the search rounds**. Search rounds use the fixed train/validation/test split so scores are comparable.

After the best round is selected, the script optionally trains one final model on Train+Val:

1. copy the best round's parameters,
2. read `usedTrees` from the best round's `GL1X` footer when available,
3. set `trees = usedTrees`,
4. set `earlyStop = false`,
5. pass `--refit-train-val` to the trainer,
6. copy the refit model to `--out` if refit succeeds.

The leaderboard records both the best search trial and the optional refit trial.

## Engines

The script can drive either local trainer:

| Engine | Command used | Notes |
|---|---|---|
| `python` | `python train_gl1f.py ...` | Supports CSV and the Python trainer's NPZ/NPY options. |
| `cpp` | `./train_gl1f_cpp ...` | CSV only; usually faster for large tabular datasets. |
| `auto` | C++ if executable exists, otherwise Python | Default. |

Examples:

```bash
--engine python --train-script train_gl1f.py
```

```bash
--engine cpp --cpp-bin ./train_gl1f_cpp
```

For `cpp`, the script rejects NPZ/NPY inputs because the C++ trainer expects CSV.

## Data and task flags

The data/task flags are fixed across all rounds:

| Flag | Purpose |
|---|---|
| `--task` | Required. One of `regression`, `binary_classification`, `multiclass_classification`, `multilabel_classification`. |
| `--input` | CSV path, or NPZ path when using `--npz` with the Python engine. |
| `--label-col` | Single label column for regression, binary, or multiclass. |
| `--label-cols` | Multiple label columns for multilabel tasks. |
| `--feature-cols` | Optional comma-separated feature column list. |
| `--delimiter` | CSV delimiter, default `auto`. |
| `--no-header` | Treat CSV as headerless. |
| `--limit-rows` | Optional row cap forwarded to the trainer. |
| `--neg-label`, `--pos-label` | Explicit binary-class label mapping. |
| `--class-labels` | Explicit class ordering for multiclass. |
| `--n-classes` | Class/label count used by the web size clamp for v2 tasks. |

Python-engine array inputs:

| Flag | Purpose |
|---|---|
| `--npz` | Interpret `--input` as an NPZ file. |
| `--npz-x-key` | Feature array key, default `X`. |
| `--npz-y-key` | Label array key, default `y`. |
| `--npy-x` | Feature `.npy` path. |
| `--npy-y` | Label `.npy` path. |
| `--mmap` | Forward memory-mapping option to the trainer. |

## Class imbalance options

The script can keep the web/class-weighting settings fixed across all rounds:

| Flag | Meaning |
|---|---|
| `--imbalance-mode none` | No special weighting. |
| `--imbalance-mode auto` | Let the trainer compute automatic class/label weights. |
| `--imbalance-mode manual` | Use explicit weights. |
| `--imbalance-cap` | Cap automatic weights. |
| `--imbalance-normalize` / `--no-imbalance-normalize` | Normalize or do not normalize weights. |
| `--stratify` | Stratify train/val/test split for single-label classification. |
| `--w0`, `--w1` | Manual binary negative/positive weights. |
| `--class-weights` | Manual multiclass weights. |
| `--pos-weights` | Manual multilabel positive weights. |

These values are part of the base parameter object and remain fixed throughout the heuristic search.

## Learning-rate schedule options

The base learning-rate schedule is also part of the initial parameter object.

| Flag | Meaning |
|---|---|
| `--lr-schedule none` | No schedule. |
| `--lr-schedule plateau` | Enable plateau schedule. |
| `--lr-patience` | Plateau patience. |
| `--lr-drop-pct` | Plateau drop percentage. |
| `--lr-min` | Minimum learning rate. |
| `--lr-schedule piecewise` | Use piecewise schedule. |
| `--lr-segments` | Trainer segment format, for example `0:100:0.05,100:250:0.01`. |

When plateau mode is active, the heuristic search also perturbs the plateau schedule fields. When piecewise mode is active, the segment string is held fixed.

## Outputs

Each run writes:

| Output | Purpose |
|---|---|
| `--out` model file | The best `.gl1f`, or the refit `.gl1f` if `--refit-train-val` succeeds. Default: `best_model.gl1f`. |
| `--work` directory | Stores all per-round `.gl1f` artifacts and `leaderboard.json`. Default: `gl1f_search_runs`. |
| `leaderboard.json` | Full machine-readable search record. |

`leaderboard.json` contains:

- mode: `web-exact`,
- chosen metric,
- direction,
- engine,
- elapsed time,
- best round,
- optional refit round,
- all trials in chronological order,
- all successful trials sorted by objective,
- failed trial count,
- exact reproduction command per trial,
- trainer metadata parsed from each `GL1X` footer.

## Usage examples

### Basic regression search with C++ trainer

```bash
python gl1f_search.py \
    --engine cpp \
    --task regression \
    --input data/btc_vol.csv \
    --label-col y \
    --trials 10 \
    --best-by bestValMetric \
    --out best_btc_vol.gl1f \
    --work runs/btcvol_web_exact
```

### Use exact web initial parameters

```bash
python gl1f_search.py \
    --engine cpp \
    --task regression \
    --input data/btc_vol.csv \
    --label-col y \
    --initial-params '{"trees":250,"depth":4,"lr":0.05,"minLeaf":10,"bins":32,"binning":"linear","seed":42,"earlyStop":true,"patience":25,"splitTrain":0.7,"splitVal":0.2}' \
    --trials 25 \
    --best-by val \
    --out best.gl1f \
    --work runs/web_exact
```

### Select validation accuracy as the objective

```bash
python gl1f_search.py \
    --engine cpp \
    --task binary_classification \
    --input data/signals.csv \
    --label-col breakout \
    --pos-label yes \
    --neg-label no \
    --trials 30 \
    --best-by val_acc \
    --target 0.95 \
    --out best_acc.gl1f \
    --work runs/signal_acc
```

Because `val_acc` maps to `bestValAcc`, the direction is automatically `max`.

### Stop when validation loss reaches a target

```bash
python gl1f_search.py \
    --engine cpp \
    --task regression \
    --input data/train.csv \
    --label-col target \
    --trials 100 \
    --best-by bestValMetric \
    --target 0.001 \
    --out best_target.gl1f
```

Because `bestValMetric` is loss/error-like, the direction is automatically `min`.

### Search, then refit on Train+Val

```bash
python gl1f_search.py \
    --engine cpp \
    --task regression \
    --input data/train.csv \
    --label-col target \
    --trials 30 \
    --best-by val \
    --refit-train-val \
    --out best_refit.gl1f \
    --work runs/refit_web_exact
```

The search leaderboard still chooses the best round from validation metrics. The final output file is the refit model if the refit succeeds.

### Python trainer with NPZ input

```bash
python gl1f_search.py \
    --engine python \
    --task regression \
    --input data/train_arrays.npz \
    --npz \
    --npz-x-key X \
    --npz-y-key y \
    --trials 20 \
    --best-by val \
    --out best_npz.gl1f
```

### Multiclass with explicit class count for size clamp

```bash
python gl1f_search.py \
    --engine cpp \
    --task multiclass_classification \
    --input data/classes.csv \
    --label-col label \
    --class-labels red,green,blue,yellow \
    --n-classes 4 \
    --trials 20 \
    --best-by val_acc \
    --out best_multiclass.gl1f
```

For multiclass and multilabel tasks, `--n-classes` is important because the model-size clamp must account for vector-output trees.

## Full CLI reference

### Engine

| Flag | Default | Description |
|---|---|---|
| `--engine` | `auto` | `python`, `cpp`, or `auto`. |
| `--python-exe` | current Python | Python executable for the Python trainer. |
| `--train-script` | `train_gl1f.py` | Python trainer script path. |
| `--cpp-bin` | `./train_gl1f_cpp` | C++ trainer executable path. |

### Base hyperparameters

| Flag | Default | Description |
|---|---:|---|
| `--trees` | `250` | Round-1 tree count. |
| `--depth` | `4` | Round-1 tree depth. |
| `--lr` | `0.05` | Round-1 learning rate. |
| `--min-leaf` | `10` | Round-1 minimum leaf size. |
| `--bins` | `32` | Fixed histogram bin count. |
| `--binning` | `linear` | Fixed binning mode. |
| `--seed` | `42` | Fixed training seed and search RNG basis. |
| `--split-train` | `0.7` | Fixed training split fraction. |
| `--split-val` | `0.2` | Fixed validation split fraction. |
| `--scaleQ` | `auto` | Trainer quantization scale. |
| `--chain-id` | `29` | Chain id recorded by trainer package. |
| `--early-stop` | enabled | Enable early stopping. |
| `--no-early-stop` | — | Disable early stopping. |
| `--patience` | `25` | Early-stopping patience. |
| `--initial-params` | — | Exact round-1 JSON object; overrides base hyperparameter flags. |
| `--print-initial-params` | — | Print clamped round-1 params and exit. |
| `--refit-train-val` | — | Final web-style refit after search. |

### Objective and stopping

| Flag | Default | Description |
|---|---|---|
| `--metric-key` / `--best-by` | `bestValMetric` | Footer metric used to choose best. |
| `--direction` | `auto` | `auto`, `min`, or `max`. |
| `--target` | — | Stop when selected metric reaches this threshold. |
| `--no-improve-patience` | `0` | Stop after this many rounds without improvement; `0` disables. |

### Search control

| Flag | Default | Description |
|---|---:|---|
| `--trials` | `10` | Exact web heuristic rounds. Round 1 is the base params. Clamped to `1..1000`. |
| `--time-budget` | `0` | Optional seconds budget. `0` disables. |
| `--trial-timeout` | `1800` | Per-trainer-process timeout in seconds. |
| `--extra` | empty | Verbatim passthrough flags sent to the trainer. |

### Metadata and output

| Flag | Default | Description |
|---|---|---|
| `--title` | empty | Optional trainer package title. |
| `--description` | empty | Optional trainer package description. |
| `--out` | `best_model.gl1f` | Final best model path. |
| `--work` | `gl1f_search_runs` | Working directory and leaderboard location. |

## Filename contract

The command-line web-exact search script is named:

```text
gl1f_search.py
```

This README documents `gl1f_search.py` as the browser-matching search script. Round 1 is the exact initial parameter set from `--initial-params` or explicit CLI flags; later rounds follow the Forest web UI heuristic; the best model is chosen by `--best-by`; and `--target` is evaluated against that same chosen metric.

## Reproducibility checklist

To reproduce a run exactly:

1. Keep the same script version.
2. Keep the same trainer implementation and build.
3. Keep the same input file and selected columns.
4. Keep the same `--task`, label mapping, and class/label ordering.
5. Keep the same round-1 params, either via flags or `--initial-params`.
6. Keep the same `seed`, `splitTrain`, `splitVal`, `bins`, and `binning`.
7. Keep the same selected `--best-by` metric and direction.
8. Keep the full `leaderboard.json`; every trial stores the exact command that produced it.

The `.gl1f` output may include a `GL1X` footer with local training metadata. That footer is useful for leaderboard parsing and mint metadata, but it is not part of core inference. As with all GL1F packaging, strip the footer before computing a deployment `modelId`.

## Practical notes

- `--trials` means exact web rounds. `--trials 10` runs at most 10 trainer processes before optional refit; round 1 is the initial/base parameter set.
- The script is sequential by design to match the web loop. It does not submit parallel candidates.
- `bins` and `binning` are intentionally fixed during search. This keeps scores comparable to the UI behavior.
- For classification accuracy objectives, choose `--best-by val_acc`, `--best-by test_acc`, or `--best-by train_acc`.
- For loss/error objectives, choose `--best-by val`, `--best-by test`, or `--best-by train`.
- If a selected metric is missing from the `GL1X` footer, the trial is marked failed and the error message lists the available `trainMeta` keys when possible.
- For C++ runs, build `train_gl1f_cpp` first and use CSV input.
- For NPZ/NPY input, use the Python engine.

---

# Headless minting — `mint_model.py`

`mint_model.py` (repo root) is the **command-line equivalent of the Forest studio's Mint tab**. It mirrors the browser/MetaMask flow byte-for-byte, but signs every transaction with a private key from `.env` — so a large model that would otherwise require hundreds of MetaMask confirmations can be deployed unattended in a single run.

It is the right tool when:
- **The model is large** (hundreds of KB to several MB), where the browser path means clicking through 100–600+ MetaMask popups.
- **You need a machine-readable deployment record** — every completed run emits `S1.json` with the model digest, contract pointers, transaction identifiers, and recorded settings.
- **The mint may be interrupted** — `mint_state.json` allows resumption from the last completed chunk.
- **You want a pre-flight check** — `--dry-run` validates the whole flow without sending any transaction.

## What it does (UI parity)

The script replicates `src/create_page.js` (the `deployBtn` handler) exactly:

1. **Load** the `.gl1f` file, strip the optional `GL1X` JSON footer.
2. **Compute** `modelId = keccak256(core_bytes)` — over the post-footer core only, matching the on-chain runtime.
3. **Read registry state**: `deployFeeWei`, `sizeFeeWeiPerByte`, `requiredDeployFeeWei(totalBytes)`, `activeLicenseId`, `tosVersion`.
4. **Chunk-deploy the model**: split the core into `CHUNK_SIZE = 24000`-byte chunks. For each chunk:
   - call `store.write(chunk)` → wait for receipt → parse `ChunkWritten` event → record the chunk-contract pointer.
5. **Build the pointer table**: 32 bytes per chunk pointer (right-aligned in the slot, padding zeros on the left), then `store.write(table)` to deploy it.
6. **Register the model**: a single `registry.registerModel(...)` call (payable, value = `requiredDeployFeeWei`) which also mints the custom transferable model token.
7. **Emit artifacts** (see below).

One transaction per block, sequentially — same ordering as the UI. No batching, no parallel sending.

## Inputs

### CLI flags

| Flag | Required | Default | Purpose |
|---|:---:|---|---|
| `--gl1f` | ✓ | — | Path to `.gl1f` file. |
| `--env` |   | `./.env` | Path to `.env` (see below). |
| `--rpc` |   | `RPC_URL` from env, else `https://rpc.genesisl1.org` | Override RPC endpoint. |
| `--resume` |   | — | Resume from `mint_state.json`, skipping completed chunks. |
| `--dry-run` |   | — | Validate and build everything, send NO transactions. |
| `--pricing-mode` |   | `0` | `0 = free`, `1 = tips`, `2 = paid required`. |
| `--pricing-fee-eth` |   | `0.001` | Per-inference fee in L1, used when mode ≠ 0. |
| `--pricing-recipient` |   | signer address | Address to receive inference fees. |
| `--task` |   | `regression` | Task type. *Only consulted if `.gl1f` has no `GL1X` footer.* |
| `--label-name` |   | `target` | Label-column name. *Only if no footer.* |
| `--feature-names-file` |   | — | Newline-separated file of feature names. *Only if no footer.* |
| `--store-addr` |   | live address | Override `ModelStore` contract address. |
| `--registry-addr` |   | live address | Override `Registry` contract address. |
| `--nft-addr` |   | live address | Override `NFT` contract address. |

### `.env` file

```bash
PRIVATE_KEY=0x...                       # wallet with L1 for gas + deploy fee
GAS_PRICE_GWEI=1                        # gas price for every transaction
RPC_URL=https://rpc.genesisl1.org       # optional; default if not set
```

### Interactive prompts

Whatever isn't auto-filled from the `GL1X` footer is asked at the terminal:

- **Title** — ≥ 3 chars, must contain ≥ 1 word ≥ 2 chars (used to build the on-chain title-word search index via `keccak256(lowercased word)`).
- **Description** — ≥ 8 chars.
- **Icon** — path to a PNG, validated as exactly **128×128** with a correct PNG signature.
- **Pricing mode + fee + recipient** — confirms `--pricing-mode` settings.
- **License + ToS acceptance** — must type the *exact* license name and acceptance phrase, no `Y/N` shortcut (mirrors the UI's "I have read and accept" affordance — the on-chain record stores both `licenseIdAccepted` and `tosVersionAccepted`).

### Footer-aware metadata

When the `.gl1f` carries a `GL1X` JSON footer — which both the Python and C++ trainers in this repo emit by default, only `--no-package` suppresses it:

- The on-chain `featuresPacked` string is taken **verbatim** from `pkg.nft.featuresPacked`, so the deployed metadata is byte-identical to what the trainer constructed (same task, same feature-name order, same label-name and class-label ordering).
- `pkg.nft.title`, `pkg.nft.description`, and `pkg.nft.iconPngB64` (if present) are offered as **defaults in the prompts** — press Enter to accept, type a new value to override.
- A sanity check verifies the footer's feature-name count matches the model header's `nFeatures`; mismatched footers are rejected outright, not silently overridden.

The corresponding `--task` / `--label-name` / `--feature-names-file` flags are only consulted when the footer is absent or its `featuresPacked` is missing/malformed. If neither is available, feature names fall back to placeholders `feat_0..feat_{n-1}` (a warning is printed; these placeholders are permanently embedded in the NFT).

## Outputs (artifacts)

Three files written to the current working directory:

| File | Purpose |
|---|---|
| `mint_state.json` | Per-chunk progress: pointer for each completed chunk, table pointer, register-tx hash. Used by `--resume`. **Do not delete mid-mint.** |
| `owner_key.txt` | Freshly-generated owner API keypair. Required for off-chain ownership operations on the model. **Back this up immediately** — it is not recoverable. |
| `S1.json` | Machine-readable deployment record containing `modelId`, `tokenId`, chunk pointers, table pointer, register-transaction hash and block number, deployer address, owner-key address, model header fields, core SHA-256, accepted license and ToS identifiers, pricing configuration, chain ID, RPC label, and feature names. |

`S1.json` supplies the identifiers and digests needed to compare a local
`.gl1f` core and query the referenced receipts and contract state. The file is
a record produced by the client; it is not, by itself, proof of the chain
claims it contains.

## Reliability

- **Per-tx retry with gas bump.** If a transaction is not confirmed within `TX_TIMEOUT_SECONDS = 90 s`, the script resubmits with `gasPrice × 1.25`, up to 5 times per chunk.
- **Receipt polling at 2 s intervals.**
- **Resumable.** State is persisted after every successful chunk; `--resume` skips already-confirmed chunks by pointer.
- **Chain-id guard.** Refuses to send anything if connected to a chain other than `chainId = 29` (GenesisL1).
- **Wei-exact deploy fee.** `requiredDeployFeeWei(totalBytes)` is read fresh from the registry on every run — no stale fee math.

## Usage

### Mint a `.gl1f` with a `GL1X` footer (typical case)

```bash
python mint_model.py --gl1f path/to/model.gl1f
```

Task, feature names, label name, and optional title/description/icon are auto-extracted from the footer. Only fee and license-acceptance prompts remain.

### Mint a paid-inference model

```bash
python mint_model.py \
    --gl1f path/to/model.gl1f \
    --pricing-mode 2 \
    --pricing-fee-eth 0.005 \
    --pricing-recipient 0xYourCollectorAddress
```

### Resume an interrupted mint

```bash
python mint_model.py --gl1f path/to/model.gl1f --resume
```

Reads `mint_state.json` in cwd, skips chunks whose pointers are already recorded, and picks up at the first missing chunk.

### Dry run (CI / pre-flight)

```bash
python mint_model.py --gl1f path/to/model.gl1f --dry-run
```

Reads the model, computes `modelId`, validates the icon and metadata, reads registry state, encodes every transaction — but sends none. Use this to confirm a model is mintable before paying gas.

### Footerless model with explicit metadata

```bash
python mint_model.py \
    --gl1f path/to/legacy.gl1f \
    --task binary_classification \
    --label-name will_break_resistance \
    --feature-names-file features.txt
```

## Dependencies

```bash
pip install web3 eth-account python-dotenv pillow
```

## Default contracts (GenesisL1 mainnet, chainId 29)

| Contract | Address |
|---|---|
| ModelStore | `0x9CdbC23392648Bd27B4A5eD09c0fEa9452454B54` |
| Registry   | `0x33c9844F77a07e36B98f0FFf8201B8A8b02c2a69` |
| NFT        | `0x44Dc1c54B8D579B42d78cC21cf8260DC0A3279fA` |
| Runtime    | `0xD2fD0cf461a6cb56Fc08d9aEc120833D8E79044E` |
| Market     | `0xA9bfa0a719b7F73cE85CA1E7f23af626D383fB46` |

All override-able via `--store-addr` / `--registry-addr` / `--nft-addr`.

---

## License

Original GL1F software is released under the MIT License. See
[`LICENSE`](LICENSE). Third-party and separately licensed material retains its
upstream terms; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

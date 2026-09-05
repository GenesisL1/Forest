# GL1F canonical binary and publication specification

**Specification version:** 1.0\
**Core magics:** `GL1F` model, `GL1C` code-as-data object\
**Optional envelope magic:** `GL1X`\
**Byte order:** little-endian for every multibyte numeric field\
**Reference implementation:** GL1F Forest v0.2.3

## 1. Status and normative language

This document is the canonical publication profile for the GL1F artifacts in
this repository. It consolidates the implemented wire layout, the strict
publication parser, the formal assumptions, the cross-engine parity profile,
and the EVM storage manifest.

The words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**, and
**MAY** are normative. A “canonical” artifact satisfies all applicable MUST
and MUST NOT rules. A parser MAY support a separate explicitly named
compatibility mode, but tolerant behavior does not make a noncanonical
artifact canonical.

The format has four layers:

| Layer | Magic | Role | Included in model identity |
|---|---|---|---|
| GL1F core | `GL1F` | Inference-relevant header and trees | Yes |
| GL1X envelope | `GL1X` | Optional UTF-8 JSON application metadata | No |
| GL1C chunk | `GL1C` | EVM runtime-code prefix for a model payload | No |
| GL1C pointer table | `GL1C` | EVM runtime-code prefix for ordered chunk addresses | No |

## 2. Design boundary

GL1F represents fixed-depth, complete, axis-aligned integer decision-tree
ensembles. It defines:

- feature count and order;
- tree depth and count;
- signed integer thresholds;
- signed integer base terms;
- signed integer leaf increments;
- one common positive quantization scale;
- strict branch semantics; and
- scalar or vector integer outputs.

The core does **not** define:

- whether a v1 score is regression output or a binary logit;
- whether v2 outputs are multiclass logits or independent multilabel logits;
- feature names, units, transformations, normalization, or missing-value
  policy;
- class/label names;
- sigmoid, softmax, calibration, or application thresholds;
- dataset identity, train/test membership, metrics, or intended use;
- licensing, ownership, pricing, access, or marketplace state; or
- chain addresses, chunks, registry records, or transactions.

Those facts MUST be carried by a versioned reproducibility manifest or GL1X
metadata and reconciled with the core before scientific use. Task metadata is
not allowed to change core inference semantics.

## 3. Primitive encodings

| Name | Bytes | Encoding |
|---|---:|---|
| `u8` | 1 | Unsigned integer |
| `u16le` | 2 | Unsigned 16-bit little-endian integer |
| `u32le` | 4 | Unsigned 32-bit little-endian integer |
| `i32le` | 4 | Two's-complement signed 32-bit little-endian integer |
| magic | 4 | Four literal ASCII bytes |
| address slot | 32 | Twelve zero high bytes followed by a 20-byte EVM address |

Canonical reserved bytes MUST be zero. Decoders MUST reject an unknown GL1F
or GL1X version rather than guessing a layout.

Integer domains are:

\[
\mathbb U_{16}=[0,65535],
\quad
\mathbb U_{32}=[0,2^{32}-1],
\quad
\mathbb I_{32}=[-2^{31},2^{31}-1].
\]

## 4. Tree geometry

For a tree of depth \(d\):

\[
L(d)=2^d
\]

is the number of leaves, and:

\[
I(d)=2^d-1
\]

is the number of internal nodes.

Each internal node occupies eight bytes and each leaf occupies four bytes.
One tree therefore occupies:

\[
b(d)=8I(d)+4L(d)=12\cdot2^d-8.
\]

Implementations MUST calculate geometry with checked arithmetic before
allocation or offset calculation. They MUST NOT rely on a language bit-shift
whose count wraps, truncates, or changes sign.

### 4.1 Profile depth limits

The 16-bit wire field could describe impractical values. Canonical publication
profiles narrow it:

| Profile | Required depth |
|---|---|
| First-party publication/deployment | \(1\le d\le12\) |
| Hardened local decoder safety ceiling | \(1\le d\le20\) |
| Arbitrary wire value | Not sufficient for canonical acceptance |

The first-party paper and chain publication claims MUST use the 1–12 profile.
A decoder MAY accept depths 13–20 under its resource cap, but that does not
make such a model part of the tested first-party deployment profile.

## 5. GL1F v1 scalar core

Version 1 stores one base term and one scalar ensemble. The same bytes can
represent either regression or a binary-classification logit; task
interpretation is external metadata.

### 5.1 Header

The v1 header is exactly 24 bytes:

| Offset | Size | Type | Canonical value or meaning |
|---:|---:|---|---|
| 0 | 4 | magic | Literal `GL1F` |
| 4 | 1 | `u8` | Version `1` |
| 5 | 1 | `u8` | Reserved; MUST be `0` |
| 6 | 2 | `u16le` | `nFeatures = F`; MUST be positive |
| 8 | 2 | `u16le` | `depth = d` |
| 10 | 4 | `u32le` | `nTrees = T` |
| 14 | 4 | `i32le` | Scalar `baseQ = a_0` |
| 18 | 4 | `u32le` | `scaleQ = Q`; MUST be positive |
| 22 | 2 | `u16le` | Reserved; MUST be `0` |

The exact v1 core length is:

\[
\ell_1=24+T\,b(d).
\]

The wire format permits \(T=0\), which defines a base-only scalar. A trained
publication artifact MUST use \(T\ge1\). For the deployed registry ABI,
on-chain publication MUST additionally use \(T\le65535\), because registry
metadata stores the count as `uint16`.

### 5.2 Tree offsets

Tree \(t\), zero-based, begins at:

\[
o_t=24+t\,b(d),
\quad 0\le t<T.
\]

No padding exists between trees.

## 6. GL1F v2 vector core

Version 2 stores \(K\) output bases and \(R\) trees per output. The same wire
layout supports multiclass logits or independent multilabel logits; task
meaning and label names are external metadata.

### 6.1 Fixed header and base vector

| Offset | Size | Type | Canonical value or meaning |
|---:|---:|---|---|
| 0 | 4 | magic | Literal `GL1F` |
| 4 | 1 | `u8` | Version `2` |
| 5 | 1 | `u8` | Reserved; MUST be `0` |
| 6 | 2 | `u16le` | `nFeatures = F`; MUST be positive |
| 8 | 2 | `u16le` | `depth = d` |
| 10 | 4 | `u32le` | `treesPerOutput = R`; MUST be positive |
| 14 | 4 | `i32le` | Reserved; MUST be `0` |
| 18 | 4 | `u32le` | `scaleQ = Q`; MUST be positive |
| 22 | 2 | `u16le` | `nOutputs = K`; MUST be at least `2` |
| 24 | `4K` | `i32le[K]` | Base vector `baseQ[k]` |

The historical field name `nClasses` is equivalent to `nOutputs`. The latter
is more precise because v2 also represents multilabel heads.

Trees begin at:

\[
o_{\mathrm{trees}}=24+4K.
\]

The exact v2 core length is:

\[
\ell_2=24+4K+KR\,b(d).
\]

### 6.2 Tree order

Trees are output-major, also called class-major:

```text
for output k = 0 .. K-1:
    for tree t = 0 .. R-1:
        serialize tree(k, t)
```

Tree \((k,t)\) begins at:

\[
o_{k,t}=24+4K+(kR+t)b(d).
\]

A conforming decoder MUST NOT interpret the tree region as round-major.

### 6.3 Registry count restriction

The deployed registry stores `nTrees` in a `uint16` even though v2 stores
`R` as `u32`. A publication entry that requires exact registry/header
agreement MUST set:

\[
KR\le65535
\]

and register the total tree count \(KR\). This restriction belongs to the
documented deployment profile, not the general GL1F v2 wire grammar.

## 7. Per-tree layout

Every v1 and v2 tree has the same representation.

### 7.1 Internal-node array

The first \(8I(d)\) bytes contain internal nodes in breadth-first binary-heap
order. Internal node \(j\) begins at:

\[
o_{\mathrm{node}}=o_{\mathrm{tree}}+8j,
\quad 0\le j<I(d).
\]

Its eight bytes are:

| Relative offset | Size | Type | Meaning |
|---:|---:|---|---|
| 0 | 2 | `u16le` | `featureIndex`; MUST be `< F` |
| 2 | 4 | `i32le` | `thresholdQ` |
| 6 | 2 | `u16le` | Reserved; MUST be `0` |

The signed threshold is intentionally unaligned at relative offset 2.
Implementations MUST use byte-order-aware reads and MUST NOT assume native
struct alignment without an explicitly packed representation.

### 7.2 Leaf array

The next \(4L(d)\) bytes contain signed leaf increments:

\[
o_{\mathrm{leaf}}(l)
=o_{\mathrm{tree}}+8I(d)+4l,
\quad 0\le l<L(d).
\]

Each leaf is one `i32le`.

### 7.3 Forced subtrees

The trainer may represent a node with no accepted split by filling the
remaining fixed-depth subtree. The first-party convention uses:

- `featureIndex = 0`;
- `thresholdQ = INT32_MAX`; and
- identical descendant leaf values.

This is a trainer convention, not a separate wire tag. Decoders MUST execute
the stored comparisons normally.

## 8. Core length and framing

A parser MUST:

1. read the minimum 24-byte header;
2. validate magic, version, reserved fields, positive dimensions, and profile
   depth;
3. derive checked tree geometry;
4. derive checked total tree count and exact core length;
5. reject a core length above its declared resource cap;
6. reject truncation;
7. validate every feature index and node reserved field; and
8. treat bytes after the exact core length only as a strictly framed GL1X
   envelope.

A canonical bare core MUST end exactly at its derived core length. Unframed
trailing bytes are invalid.

The hardened first-party JavaScript decoder and strict publication oracle cap
the declared core at:

\[
2^{31}-1\text{ bytes}.
\]

The deployed one-level ModelStore pointer profile imposes a much smaller
practical bound, described in [Section 15](#15-gl1c-evm-storage).

## 9. Quantization

For finite real \(x\) and integer scale \(Q>0\), define:

\[
\operatorname{round}_{JS}(z)=\left\lfloor z+\frac12\right\rfloor
\]

and:

\[
q_Q(x)=
\operatorname{clamp}_{\mathbb I_{32}}
\left(\operatorname{round}_{JS}(Qx)\right).
\]

The negative-half behavior is:

```text
roundJS(-1.5) = -1
roundJS(-0.5) = 0
roundJS( 0.5) = 1
```

The formula above uses exact-real arithmetic. For a binary64 operand `z`,
implementations MUST match `Math.round(z)` before int32 saturation. Evaluating
`floor(z + 0.5)` in binary64 is not equivalent: for
`z = 0.49999999999999994`, the addition rounds to `1`, whereas
`Math.round(z)` returns `0`. The Python and C++ implementations compare the
fractional part `z - floor(z)` with `0.5` to avoid this extra rounding step.

Implementations MUST reject NaN and positive/negative infinity at the
scientific input boundary. The mathematical clamp defines saturation for
finite values whose scaled magnitude is outside int32.

### 9.1 Scale profiles

| Layer | Allowed range |
|---|---|
| GL1F wire | \(1\le Q\le2^{32}-1\) |
| Tested three-trainer parity | \(1\le Q\le2^{31}-1\) |
| Typical automatic first-party choice | At most 1,000,000 |

Values above `INT32_MAX` are not in the current cross-trainer parity profile
because browser and C++ operational paths use signed 32-bit behavior even
though the wire field is unsigned 32-bit.

### 9.2 Quantization error

If quantization does not saturate:

\[
\left|\frac{q_Q(x)}Q-x\right|\le\frac1{2Q}.
\]

For a real split \(x>\tau\), a sufficient condition for preserving the branch
after quantizing both sides is:

\[
|x-\tau|>\frac1Q.
\]

This is sufficient, not necessary. Near a split, a different leaf may be
selected, so a bound involving only rounding error is invalid without a path
margin or a bound on alternate leaves.

## 10. Packed inference input

For a model with \(F\) features, the packed input is exactly \(4F\) bytes:

```text
i32le qQ(x[0])
i32le qQ(x[1])
...
i32le qQ(x[F-1])
```

Feature order is part of the scientific model contract. A runtime MUST reject
a byte length other than \(4F\). Feature names or metadata MUST NOT be used to
silently reorder the packed vector unless a separately versioned,
deterministic mapping is applied and recorded.

Missing values are not represented. A caller MUST either reject them or apply
a declared preprocessing/imputation rule before quantization. Silent blank to
zero conversion is noncanonical for publication.

For the strongest conformance evidence, a manifest SHOULD retain:

- original finite feature values;
- `scaleQ`;
- the exact packed byte string;
- feature order and units; and
- the integer result.

## 11. Traversal

Traversal begins with heap index \(i_0=0\). At level \(j\):

1. read node `featureIndex` and `thresholdQ`;
2. select the corresponding packed feature \(q_f\);
3. go right if and only if \(q_f>\text{thresholdQ}\); and
4. update:

\[
i_{j+1}=
\begin{cases}
2i_j+2,&q_f>\text{thresholdQ},\\
2i_j+1,&q_f\le\text{thresholdQ}.
\end{cases}
\]

After exactly \(d\) decisions:

\[
l=i_d-I(d)
\]

is the zero-based leaf index. Equality MUST go left.

The traversal invariant is:

\[
2^j-1\le i_j\le2^{j+1}-2.
\]

Therefore a canonical tree selects exactly one leaf in
\([0,L(d)-1]\).

## 12. Scalar inference

For a v1 model:

\[
s_Q=a_0+\sum_{t=0}^{T-1}v_{t,\pi_t(q)},
\]

where \(\pi_t(q)\) is the selected leaf in tree \(t\).

The canonical result is the integer \(s_Q\). Application interpretation is:

- regression estimate: \(s_Q/Q\); or
- binary logit: \(s_Q/Q\).

Sigmoid computation and a decision threshold are outside the GL1F core.
Implementations SHOULD preserve the integer score in reproducibility output
even when displaying a floating interpretation.

## 13. Vector inference

For output \(k\) of a v2 model:

\[
s_{Q,k}=a_{0,k}+\sum_{t=0}^{R-1}v_{k,t,\pi_{k,t}(q)}.
\]

The canonical result is the ordered integer vector:

\[
(s_{Q,0},\ldots,s_{Q,K-1}).
\]

For multiclass use, canonical class selection is integer argmax. Equal logits
MUST resolve to the lowest output index; this follows from replacing the
incumbent only on strict improvement.

Softmax, per-label sigmoid, calibration, and thresholds are application
operations. They are not necessary for exact local/EVM integer equivalence.

## 14. Accumulation

Stored bases and leaves are int32, but accumulation MUST use a wider exact
integer type.

- The EVM runtime uses `int256`.
- JavaScript uses `Number`, which is exact for integers with magnitude at most
  \(2^{53}-1\).
- Python integers are unbounded.
- C++ inference implementations SHOULD use at least signed 64-bit arithmetic
  after proving the relevant model bound.

Under the documented canonical deployment profile:

- v1 has at most 65,535 registered trees, so the conservative bound is
  \(2^{47}\);
- the one-level storage table and minimum v2 geometry imply fewer than 589,000
  trees per output, so the conservative per-output bound is below \(2^{51}\).

Both are below JavaScript's exact range. A standalone decoder accepting bytes
outside the deployment profile MUST compute a value-level worst-case bound and
reject a model that can exceed its exact accumulator.

## 15. GL1C EVM storage

### 15.1 Code-as-data object

`ModelStore.write(data)` deploys runtime code:

```text
"GL1C" || data
```

The literal four-byte prefix is:

```text
47 4c 31 43
```

The payload MUST contain at most 24,572 bytes, leaving a total runtime code
size at or below the EIP-170 limit of 24,576 bytes.

GL1C magic identifies a storage object type; it is not content
authentication. Arbitrary code beginning with the same four bytes can satisfy
a magic-only check.

### 15.2 Core chunks

Let:

- \(B\) be the exact GL1F core;
- \(\ell=|B|\);
- \(c\) be the constant logical chunk payload size; and
- \(N=\lceil\ell/c\rceil\).

Canonical chunk \(j\) contains:

\[
\texttt{GL1C}\Vert
B[jc:\min((j+1)c,\ell)].
\]

The formal storage profile requires:

\[
4\le c\le24572.
\]

The lower bound exists because the runtime primitive reader handles fields of
up to four bytes and consults at most two chunks. If \(c<4\), a four-byte field
can span three logical chunks and reconstruction is not exact.

The first-party publisher uses:

```text
c = 24,000 bytes
```

### 15.3 Pointer table

The pointer table is another GL1C object:

```text
"GL1C" || slot[0] || ... || slot[N-1]
```

Each slot is exactly 32 bytes:

```text
12 zero bytes || 20-byte chunk address
```

Canonical slots MUST have zero high bytes and a nonzero address. Pointer order
MUST equal chunk order.

Because the table payload is limited to 24,572 bytes:

\[
N\le\left\lfloor\frac{24572}{32}\right\rfloor=767.
\]

### 15.4 Well-formed manifest

A manifest `(tablePtr, chunkSize, numChunks, totalBytes)` is well formed for
core \(B\) only if:

1. `4 <= chunkSize <= 24572`;
2. `totalBytes == len(B)`;
3. `numChunks == ceil(totalBytes / chunkSize)`;
4. `1 <= numChunks <= 767`;
5. table runtime code is exactly `4 + 32*numChunks` bytes;
6. table magic is GL1C;
7. every slot is canonical and resolves to a nonzero address;
8. every nonfinal chunk runtime code is exactly
   `GL1C || chunkSize bytes`;
9. the final chunk runtime code is exactly `GL1C || remaining bytes`;
10. concatenating ordered payloads yields exactly \(B\); and
11. all registry fields equal the validated values.

Extra table slots, trailing payload bytes, zero-padded missing data, and
underdeclared `totalBytes` are noncanonical.

## 16. Content identity

The canonical on-chain model identifier is:

\[
\texttt{modelId}=\operatorname{keccak256}(B),
\]

where \(B\) is the exact GL1F core and excludes GL1X.

An accompanying artifact record SHOULD also record:

\[
\operatorname{SHA256}(B)
\]

for conventional artifact verification.

Neither digest includes:

- GL1X metadata;
- GL1C prefixes;
- chunk addresses;
- pointer table;
- registry/NFT fields; or
- mutable service settings.

The deployed registry accepts `modelId` and the pointer manifest as separate
caller-supplied values. Registry existence therefore does **not** imply
`modelId == keccak256(reconstructedCore)`. A conforming verifier MUST
reconstruct the exact core and perform that comparison independently until a
hardened registry enforces it.

[`benchmarks/live_chain_witness.mjs`](benchmarks/live_chain_witness.mjs)
performs pinned retrieval, exact pointer-table and chunk-shape validation,
strict core-length parsing, content hashing, header/registry comparison, and
local/EVM prediction comparison. The public v0.2.3 repository contains a
provider-attested summary for the 12 entries observed at block 13,342,043 in
[`benchmarks/results/LIVE_CHAIN_WITNESS.md`](benchmarks/results/LIVE_CHAIN_WITNESS.md).
The raw provider responses and reconstructed deployed bytes are not included,
so the historical observation cannot be replayed offline from this repository.
The summary does not certify later entries or model scientific quality.

## 17. GL1X optional metadata envelope

If present, GL1X begins immediately at the exact derived GL1F core length.

### 17.1 Frame

| Relative offset | Size | Type | Canonical value or meaning |
|---:|---:|---|---|
| 0 | 4 | magic | Literal `GL1X` |
| 4 | 1 | `u8` | Version `1` |
| 5 | 3 | bytes | Reserved; MUST all be `0` |
| 8 | 4 | `u32le` | `jsonLength = J` |
| 12 | `J` | bytes | UTF-8 JSON payload |

A canonical GL1X envelope MUST:

- have an exact total length of \(12+J\);
- end exactly at end-of-file;
- contain valid UTF-8;
- parse as valid JSON; and
- have a JSON object as its root.

An empty object is valid. A non-object JSON root is invalid. Trailing bytes
after the declared JSON are invalid.

### 17.2 Semantic status

GL1X:

- MAY contain feature names, class/label order, dataset/run metadata, training
  parameters, metrics, mint defaults, and deployment witnesses;
- MUST NOT alter GL1F traversal or integer outputs;
- MUST be stripped before computing `modelId`;
- is excluded from cross-trainer core-byte parity; and
- SHOULD have its own SHA-256 recorded when cited.

Creation timestamps and frontend-specific JSON construction mean GL1X bytes
are not currently expected to be identical across trainers.

### 17.3 Production tolerance

The production browser parser historically tolerated malformed GL1X JSON and
could report a detected footer with a null package object. That behavior is a
compatibility feature, not canonical validation. Publication tools MUST use
the strict rules above, implemented independently in
[`tests/publication/model_format.py`](tests/publication/model_format.py).

## 18. Canonical validation algorithm

A publication validator SHOULD report every checked invariant and MUST fail
closed.

### 18.1 Local file

```text
read raw bytes
require length >= 24
require magic == GL1F
require version in {1, 2}
require canonical reserved bytes
read F, d, counts, Q
require F > 0, Q > 0, profile depth
compute I, L, bytesPerTree with checked arithmetic
if v1:
    derive coreLength = 24 + T*bytesPerTree
if v2:
    require K >= 2 and R > 0
    require complete base vector
    derive coreLength = 24 + 4*K + K*R*bytesPerTree
require resource cap and raw length >= coreLength
for every internal node:
    require featureIndex < F
    require node reserved == 0
if raw length == coreLength:
    accept bare core
else:
    require exact canonical GL1X frame and object payload
emit parsed header, core length, SHA-256, and Keccak-256
```

### 18.2 Chain entry

In addition to local validation:

```text
pin block number and verify chain id
read table/chunk/runtime code at that block
validate exact manifest and all GL1C objects
reconstruct exactly totalBytes
strictly parse reconstructed core
require keccak256(core) == modelId
require registry/header equality
require metadata feature count and task/output kind agree
record contract code hashes and block hash
compare local and contract integer conformance vectors
```

Magic checks alone are insufficient.

## 19. Training publication profile

Training-byte identity is empirical and applies only when every item below is
recorded and equal.

### 19.1 Data

- All feature values MUST be finite.
- All engines MUST consume the same numeric matrix after conversion to
  float32.
- Row and feature order MUST be identical.
- Targets/class indices/label vectors MUST have identical values and order.
- Class and label ordering MUST be explicit.
- Preprocessing MUST be completed before the parity boundary.
- Raw CSV equivalence MUST NOT be assumed.

### 19.2 Parameters

- Task aliases MUST normalize identically.
- All hyperparameters MUST be the same after frontend clamping/defaulting.
- Split fractions MUST produce nonempty required partitions.
- Counts, dimensions, and scale MUST satisfy the selected profile.
- Linear binning MUST compute:

  ```text
  floor(((x - minimum) / range) * bins)
  ```

  in that operation order.

- Quantile threshold sampling/order MUST be identical.
- Feature subsampling and every random consumer MUST occur in the same order.
- Early-stopping and schedule comparisons MUST use the matched implementation
  behavior.

### 19.3 Random generator

The shared generator is xorshift32. Seed is reduced modulo \(2^{32}\). An
all-zero internal state is replaced with:

```text
123456789
```

Each step is:

```text
x ^= x << 13
x ^= x >>> 17
x ^= x << 5
return x modulo 2^32
```

Shuffle uses descending Fisher–Yates and:

```text
j = nextU32() % (i + 1)
```

An explicit seed `0` MUST remain explicit until the xorshift fallback is
applied; it MUST NOT be replaced by a UI default such as 42.

### 19.4 Numeric environment

- The environment MUST provide ordinary IEEE-754 behavior.
- C++ MUST NOT use `-ffast-math` or algebraic reassociation.
- `scaleQ` MUST be in \(1\ldots2^{31}-1\) for three-engine parity.
- The core, not GL1X, MUST be the equality unit.
- Near-tie early-stopping/split outcomes MUST be interpreted as tested
  observations, not cross-platform mathematical guarantees.

### 19.5 Refit rule

`refitTrainVal` changes the fitting indices to `Train ∪ Val`. It does not
preserve Val as an independent early-stopping set. Publication runs MUST
either:

- use refit with a fixed tree budget and early stopping disabled; or
- perform selection first, freeze the selected budget, and execute a separate
  fixed-budget refit.

## 20. Scientific publication profile

A GL1F artifact is scientifically reproducible only when the byte format is
paired with:

- source revision and dirty state;
- trainer/runtime/compiler identities and flags;
- original and canonicalized dataset digests;
- preprocessing and dropped-row accounting;
- feature names, order, units, ranges, and missingness policy;
- target construction and output mapping;
- exact split membership or deterministic split witness;
- model-selection and refit policy;
- final untouched evaluation design;
- task-appropriate metrics, denominators, uncertainty, and calibration;
- intended use and invalid-use statement;
- core SHA-256 and Keccak-256;
- strict-parser report;
- fixed packed-input and integer-output conformance vectors; and
- deployment block, code hashes, manifest, and mutable settings when chain
  claims are made.

[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) defines the full run record.

## 21. Formal guarantees and non-guarantees

### 21.1 Guarantees under canonical preconditions

| Property | Result |
|---|---|
| Tree storage | Exactly \(12\cdot2^d-8\) bytes |
| v1 storage | Exactly \(24+Tb(d)\) bytes |
| v2 storage | Exactly \(24+4K+KRb(d)\) bytes |
| Traversal | Exactly \(d\) decisions and one in-range leaf per tree |
| Scalar work | Exactly \(Td\) decisions and \(T\) leaf reads |
| Vector work | Exactly \(KRd\) decisions and \(KR\) leaf reads |
| Storage read | Exact for 1/2/4-byte fields under a well-formed manifest with `chunkSize >= 4` |
| Integer equivalence | Exact local/EVM scalar or vector for the same core and packed input |
| Abstract one-value quantization | Error at most \(1/(2Q)\) without saturation when \(Qx\) is formed in exact-real arithmetic |
| Abstract stable-path scalar output | Error at most \((T+1)/(2Q)\) without saturation |
| Abstract split preservation | Guaranteed by the sufficient margin \(|x-\tau|>1/Q\) |

Production binary64 preprocessing first forms
\(\operatorname{fl}(Qx)=Qx+\delta_Q(x)\). Unless it is shown to emit the same
integer as the abstract quantizer, replace a one-value half-unit error by
\((1/2+|\delta_Q(x)|)/Q\), provided the rounded result remains in the int32
range. With neither operand saturating, the corresponding sufficient split
margin is \((1+|\delta_Q(x)|+|\delta_Q(\tau)|)/Q\). No half-unit-style
error bound is claimed for a saturated result.

### 21.2 Not guaranteed by the format

- universal cross-platform training-byte identity;
- prediction accuracy or calibration;
- robustness, fairness, causality, or safety;
- input, output, or model confidentiality;
- legal ownership or licence enforceability;
- registry/hash correctness in the deployed contract;
- affordable gas for every syntactically valid model;
- availability of chain, RPC, UI, or archive;
- one-time semantics for replayable view signatures; or
- full ERC-721 interoperability of the deployed custom NFT.

The proofs and counterexamples are in
[`paper/formal_results.tex`](paper/formal_results.tex), with executable
boundary cases in `tests/contracts_publication/`.

## 22. Worked v1 conformance vector

This 40-byte v1 model has:

- one feature;
- depth one;
- one tree;
- base `3`;
- scale `10`;
- threshold `5`;
- left leaf `-7`; and
- right leaf `11`.

Hex:

```text
474c314601000100010001000000030000000a00000000000000050000000000f9ffffff0b000000
```

Field decomposition:

| Bytes | Meaning |
|---|---|
| `474c3146` | `GL1F` |
| `01 00` | v1, reserved zero |
| `0100` | one feature |
| `0100` | depth one |
| `01000000` | one tree |
| `03000000` | base `3` |
| `0a000000` | scale `10` |
| `0000` | reserved |
| `0000` | feature index `0` |
| `05000000` | threshold `5` |
| `0000` | node reserved |
| `f9ffffff` | left leaf `-7` |
| `0b000000` | right leaf `11` |

For input \(x=0.5\):

```text
q = roundJS(0.5 * 10) = 5
5 > 5 is false
scoreQ = 3 + (-7) = -4
```

For input \(x=0.55\):

```text
q = roundJS(0.55 * 10) = 6
6 > 5 is true
scoreQ = 3 + 11 = 14
```

This vector is exercised in
[`tests/publication/test_publication.py`](tests/publication/test_publication.py).

## 23. Size examples

For depth four:

\[
b(4)=12\cdot16-8=184.
\]

A v1 model with 60 trees is therefore:

\[
24+60\cdot184=11064\text{ bytes}.
\]

This is the exact size of both reference benchmark cores.

For a v2 model with three outputs, 18 trees per output, and depth three:

\[
b(3)=12\cdot8-8=88,
\]

\[
24+4\cdot3+3\cdot18\cdot88=4788\text{ bytes}.
\]

These calculations SHOULD be performed before allocation, upload, or fee
estimation and then checked against actual bytes.

## 24. Compatibility and version evolution

Future changes MUST preserve explicit version boundaries.

- New core semantics require a new GL1F version.
- Existing reserved fields MUST NOT gain meaning without a new version.
- Parsers MUST reject unsupported versions.
- A new GL1X schema SHOULD carry its own schema identifier inside the JSON in
  addition to the frame version.
- A new storage-manifest scheme SHOULD be domain-separated and MUST state
  whether it preserves `keccak256(core)` as model identity.
- Decoders SHOULD expose the exact version and profile used.
- Writers SHOULD emit only canonical reserved zeros.
- Golden conformance vectors MUST be retained across releases.

Changing tree order, rounding, comparison strictness, task interpretation,
quantization scale semantics, or core identity is a breaking change.

## 25. Security requirements for hostile inputs

A decoder processing untrusted bytes MUST:

- impose file/core/depth/output/tree-count caps before allocation;
- use checked multiplication and addition;
- validate every offset before reading;
- validate feature indices before inference;
- reject noncanonical reserved fields in strict mode;
- reject non-finite feature inputs;
- prove accumulator safety for its integer representation;
- reject malformed GL1X lengths and text;
- avoid recursive traversal;
- bound chain-fetch concurrency and total retrieved bytes; and
- report invalid models as errors rather than silently coercing fields.

A chain verifier MUST also treat:

- zero addresses;
- missing code;
- short code;
- non-GL1C code;
- extra/missing pointers;
- duplicate pointers;
- incorrect chunk lengths;
- underdeclared/overdeclared totals;
- header/registry mismatches; and
- content-hash mismatch

as explicit failures. Duplicate pointers are not intrinsically impossible, but
they MUST reproduce the exact expected byte sequence and SHOULD be flagged for
manual inspection because they are unusual in a first-party upload.

## 26. Conformance commands

Run:

```bash
python3 -m unittest -v tests.publication.test_publication
python3 -m unittest -v tests.contracts_publication.test_formal_properties
```

The first command validates production three-engine behavior and strict
format/inference cases. The second validates formal boundary witnesses and
checked-in artifact equality.

For a chain entry:

```bash
GL1F_WITNESS_BLOCK=13342043 \
node benchmarks/live_chain_witness.mjs \
  /tmp/gl1f-live-chain-witness.json
```

The full environment and expected results are in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## 27. Implementation references

| Role | File |
|---|---|
| Python writer/trainer | [`train_gl1f.py`](train_gl1f.py) |
| C++ writer/trainer | [`cpp/train_gl1f_cpp.cpp`](cpp/train_gl1f_cpp.cpp) |
| Browser writer/trainer | [`src/train_worker.js`](src/train_worker.js) |
| Hardened browser decoder | [`src/local_infer.js`](src/local_infer.js) |
| Independent strict parser/oracle | [`tests/publication/model_format.py`](tests/publication/model_format.py) |
| Code-as-data store | [`contracts/ModelStore.sol`](contracts/ModelStore.sol) |
| EVM evaluator | [`contracts/ForestRuntime.sol`](contracts/ForestRuntime.sol) |
| Registry | [`contracts/ModelRegistry.sol`](contracts/ModelRegistry.sol) |
| Browser publisher | [`src/create_page.js`](src/create_page.js) |
| Python publisher | [`mint_model.py`](mint_model.py) |
| Chain witness scaffold | [`benchmarks/live_chain_witness.mjs`](benchmarks/live_chain_witness.mjs) |

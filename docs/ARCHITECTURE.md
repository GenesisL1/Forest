# GL1F architecture

GL1F is an open protocol and software stack for serializing integer-quantized,
fixed-depth gradient-boosted decision trees and executing them through browser,
Python, C++, and EVM paths. The deployed system is live on GenesisL1, chain ID
29. Its five contract addresses are recorded in
[`deployments/genesisl1.json`](../deployments/genesisl1.json).

## System boundary

GL1F separates model construction from model execution:

1. A trainer produces a canonical GL1F byte sequence.
2. The byte sequence is divided into bounded GL1C payloads.
3. `ModelStore` deploys each payload as immutable contract runtime code.
4. A GL1C pointer table records the chunk addresses in order.
5. `ModelRegistry` records the pointer table, dimensions, settings, metadata,
   and the model identifier.
6. `ModelNFT` represents the registry entry as a custom transferable
   model-token record.
7. `ForestRuntime` reconstructs the required fields and evaluates integer tree
   traversal.
8. The browser, Python, and C++ paths can reconstruct the same bytes and
   evaluate them independently.

```text
CSV / arrays
     │
     ▼
browser · Python · C++ trainer
     │ canonical GL1F bytes
     ├────────────────────────────► local strict decoder and inference
     │
     ▼
GL1C immutable chunks ─► pointer table ─► registry + model-token record
                                           │
                                           ▼
                                    ForestRuntime
                                 view call or transaction
```

The chain records bytes and state transitions. It does not establish whether a
training dataset is correct, whether a model is scientifically valid, or
whether a prediction is suitable for a particular decision.

## Canonical model representation

The model core begins with the four-byte ASCII magic `GL1F`.

- Version 1 represents scalar-output regression and binary-logit ensembles.
- Version 2 represents vector-output multiclass and multilabel ensembles.
- Features, thresholds, base scores, and leaf contributions are represented in
  integer Q-units.
- Each complete binary tree has a fixed depth, so its traversal and serialized
  size are bounded by the header fields.
- Optional GL1X data follows the core and carries packaging metadata; it is not
  part of the model identifier.

The normative byte layout and validation requirements are defined in
[`FORMAT_SPEC.md`](../FORMAT_SPEC.md).

## Storage layer

`ModelStore.write(bytes)` deploys a pointer contract whose runtime bytecode is
`GL1C || payload`. Payloads are bounded to 24,572 bytes. The pointer table is
itself a GL1C record containing each ordered 20-byte chunk address right-aligned
in a canonical 32-byte slot.

The stored model is reconstructed by:

1. reading the pointer table runtime code;
2. validating its `GL1C` prefix and exact pointer count;
3. reading every referenced chunk;
4. validating each prefix and expected payload length;
5. concatenating exactly `totalBytes`; and
6. checking `keccak256(core) == modelId`.

The deployed registry stores a manifest for this reconstruction, but clients
that require scientific or archival assurance should perform every validation
step rather than trusting registry fields in isolation.

## Execution layer

`ForestRuntime` accepts one signed little-endian `int32` per feature. For every
tree it follows the left or right branch using the comparison
`featureQ > thresholdQ`, selects one leaf, and adds its quantized contribution
to the accumulator.

- v1 returns a scalar accumulator.
- v2 vector inference returns one accumulator per output.
- v2 class inference returns the lowest-index maximum and its score.

There is no floating-point arithmetic in the EVM traversal. Any conversion from
real-valued features to Q-units happens before the runtime call and is part of
the caller's reproducibility record.

## Access and transaction modes

Unrestricted view functions can be invoked with `eth_call`. Paid-required
models use transaction functions, while signed owner/access view functions
support EIP-712 authorization. These gates govern canonical runtime entry
points; they are not confidentiality mechanisms. Published GL1F bytes are
intentionally public and can be reconstructed and evaluated locally.

An `eth_call` is re-executed by the selected RPC provider and does not produce a
consensus receipt. A transaction is ordered and executed by the network and
produces a receipt. Consequential research records should pin the block number
and hash and independently replay public bytes and inputs.

## Implementation surfaces

| Surface | Primary files | Responsibility |
|---|---|---|
| Browser trainer | `src/train_worker.js`, `src/train_gbdt.js` | Train and serialize models |
| Browser inference | `src/local_infer.js` | Strict decode and local evaluation |
| Python | `train_gl1f.py` | Headless training and packaging |
| C++ | `cpp/train_gl1f_cpp.cpp` | Native training and packaging |
| Contracts | `contracts/*.sol` | Storage, registry, NFT, runtime, marketplace |
| Reproducer | `benchmarks/live_chain_witness.mjs` | Pinned-chain reconstruction and output comparison |
| Optional archive collector | `benchmarks/archive_live_chain_state.mjs` | Create a local archive of provider-returned pinned code and calls; no such archive is checked in |
| Offline-verifier tooling | `benchmarks/independent_archive_verify.py` | Parse, reconstruct, hash, locally evaluate, and compare a supplied archive |

## Evidence boundary

The repository tests establish behavior only for their stated profiles and
fixtures. The recorded provider-attested summary reports reconstruction,
commitment, manifest, and execution agreement for the 12 active records at
block 13,342,043. Its raw provider responses and reconstructed deployed bytes
are not checked in. Neither evidence class establishes predictive accuracy,
data provenance, fairness, causal validity, safety, or future registry
integrity.

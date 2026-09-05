# GL1F on-chain API

This guide describes the deployed GenesisL1 interfaces used by the live GL1F
UI. It is intentionally narrower than the Solidity sources. Treat the sources
in [`contracts/`](../contracts/) and the deployed bytecode as authoritative.

## Network and addresses

- Network: GenesisL1
- Chain ID: `29`
- Default JSON-RPC: `https://rpc.genesisl1.org`
- Deployment manifest:
  [`deployments/genesisl1.json`](../deployments/genesisl1.json)

| Component | Address |
|---|---|
| `ModelStore` | `0x9CdbC23392648Bd27B4A5eD09c0fEa9452454B54` |
| `ModelRegistry` | `0x33c9844F77a07e36B98f0FFf8201B8A8b02c2a69` |
| `ModelNFT` | `0x44Dc1c54B8D579B42d78cC21cf8260DC0A3279fA` |
| `ForestRuntime` | `0xD2fD0cf461a6cb56Fc08d9aEc120833D8E79044E` |
| `ModelMarketplace` | `0xA9bfa0a719b7F73cE85CA1E7f23af626D383fB46` |

The repository client ABIs are exported by [`src/abis.js`](../src/abis.js).
`ModelNFT` exposes the transfer, ownership, enumeration, and approval functions
used by the client, but this document does not assert full ERC-721 compliance.

## Read a model manifest

Call:

```solidity
getModelRuntime(bytes32 modelId)
```

The registry returns:

```text
tablePtr, chunkSize, numChunks, totalBytes,
nFeatures, nTrees, depth, baseQ, scaleQ,
inferenceEnabled, pricingMode, feeWei, feeRecipient
```

For display metadata and its token identifier, use:

```solidity
getModelSummary(uint256 tokenId)
```

For storage-only fields, use:

```solidity
getModelBytesInfo(bytes32 modelId)
```

The registry fields are discovery data. To establish artifact integrity,
reconstruct the exact bytes from GL1C code and verify the model commitment.

The v1 core stores `nTrees` as unsigned 32-bit. The deployed registry exposes
and accepts `nTrees` as unsigned 16-bit, so the live deployment supports only
a narrower subset of otherwise canonical v1 cores. For v2, the registry
reports total trees, while the core stores `R` trees per output and `K`
outputs; the consistency relation is therefore `nTrees = K*R`.

The core does not contain a task discriminator. A v1 core can be interpreted
as a regression score or binary logit, and a v2 core as multiclass or
multilabel logits. That interpretation is metadata outside
`keccak256(core)`. A matching model commitment therefore authenticates the
execution artifact, not its application-level task label.

## Pack feature values

The runtime expects exactly `4 * nFeatures` bytes. Each feature is a signed
32-bit Q-unit integer encoded little-endian in declared feature order.

For a finite real-valued feature `x` and positive scale `scaleQ`, the
repository encoders apply JavaScript-compatible rounding and signed-`int32`
saturation before packing:

```text
xQ = clamp_int32(floor(x * scaleQ + 1/2))
```

The rounding rule, feature order, scale, input units, missing-value policy, and
preprocessing must be recorded by the caller. The EVM receives only packed
integers and cannot recover or validate those upstream choices.

JavaScript packing example:

```js
function packI32LE(values) {
  const out = new Uint8Array(values.length * 4);
  const view = new DataView(out.buffer);
  values.forEach((value, index) => {
    if (!Number.isInteger(value) || value < -2147483648 || value > 2147483647) {
      throw new RangeError(`feature ${index} is not a signed int32`);
    }
    view.setInt32(index * 4, value, true);
  });
  return `0x${Array.from(out, (b) => b.toString(16).padStart(2, "0")).join("")}`;
}
```

Reject non-finite real inputs. If quantization saturates, record that fact:
saturation can alter a model path and invalidates ordinary unsaturated
quantization-error bounds. When packing already-quantized integers directly,
reject non-integers and values outside the signed `int32` range.

## Scalar inference

For a v1 model with unrestricted view inference:

```solidity
predictView(bytes32 modelId, bytes packedFeaturesQ)
    returns (int256 scoreQ)
```

For a transaction path:

```solidity
predictTx(bytes32 modelId, bytes packedFeaturesQ)
    payable returns (int256 scoreQ)
```

The scalar is in Q-units. Regression callers divide by `scaleQ`. Binary
classification callers normally apply a sigmoid to `scoreQ / scaleQ` outside
the EVM.

## Multiclass inference

For the selected class and its quantized logit:

```solidity
predictClassView(bytes32 modelId, bytes packedFeaturesQ)
    returns (uint16 classIndex, int256 bestScoreQ)
```

Transaction and signed authorization variants are:

```text
predictClassTx
predictClassOwnerView
predictClassAccessView
```

When multiple outputs share the maximum score, the implementation returns the
lowest index.

## Vector inference

For all v2 output logits:

```solidity
predictMultiView(bytes32 modelId, bytes packedFeaturesQ)
    returns (int256[] logitsQ)
```

Transaction and signed authorization variants are:

```text
predictMultiTx
predictMultiOwnerView
predictMultiAccessView
```

For multilabel output, apply a sigmoid to each `logitQ / scaleQ` outside the
EVM. For multiclass probabilities, apply a numerically stable softmax outside
the EVM if probabilities are required.

## Signed read calls

Owner and access read functions accept an EIP-712 signature over:

```text
modelId, keccak256(packedFeaturesQ), deadline
```

Domain:

```text
name: GenesisL1 Forest
version: 1
chainId: 29
verifyingContract: ForestRuntime address
```

Signatures authorize a runtime entry point. They do not conceal the public
model core.

## Register a model

The browser studio and `mint_model.py` perform this sequence:

1. serialize and validate the GL1F core;
2. compute `modelId = keccak256(core)`;
3. split the core into chunks;
4. call `ModelStore.write` for every chunk;
5. construct and store the pointer table;
6. read current registry fee, terms, and license state;
7. call `registerModel` with the storage manifest, header fields, settings,
   metadata, title-word hashes, and exact required value; and
8. record transaction receipts and deployment artifacts.

Use `mint_model.py --dry-run` before sending transactions. An interrupted
deployment can be resumed from its generated state.

## Reconstruct and verify

For a research-grade record:

1. select a block number and obtain its block hash;
2. pass that block number to every getter and `eth_getCode` request;
3. validate exact GL1C pointer-table shape;
4. validate every chunk prefix and payload length;
5. concatenate exactly `totalBytes`;
6. strictly decode the GL1F core;
7. check the nine version-aware registry/header/storage relations:
   `nFeatures = F`;
   `nTrees = T` for v1 and `nTrees = K*R` for v2;
   `depth = d`;
   `baseQ = a0` for v1 and the v2 reserved-field convention `baseQ = 0`;
   `scaleQ = Q`;
   `tablePtr = P`;
   `totalBytes = ell`;
   `numChunks = N`; and
   `chunkSize = c`;
8. compare `keccak256(core)` with `modelId`;
9. evaluate the same packed input locally and through the runtime; and
10. re-read the selected block hash after all calls.

The repository implements this protocol in
[`benchmarks/live_chain_witness.mjs`](../benchmarks/live_chain_witness.mjs).
The production browser loader also reconstructs the bytes and rejects a
commitment mismatch before returning a model.

The public repository does not include the raw provider responses or
reconstructed deployed bytes underlying the pinned historical summary. The
deployment summaries record the reported results and limitations, while
[`benchmarks/live_chain_witness.mjs`](../benchmarks/live_chain_witness.mjs)
generates a fresh read-only record from an archival RPC endpoint.

The deployed registry does not itself perform step 8. The isolated integration
study demonstrates this limitation: it accepted registered identifier
`0x60c713e19362bcb16b04fb6fc611821a8a3fb90e1698490bf4819c2c44838875`
for a 104-byte v1 core whose reconstructed identifier is
`0xeb0ee33a46399a8592b6676035a7e580c1f90e7761a2eb1f4a56d0f9f0e4ab18`.
The runtime returned `predictionQ = 1043` under the supplied registration
metadata, whereas the strict production loader rejected the mismatched
commitment. This is a registration-integrity limitation.

## RPC and evidence semantics

An `eth_call` or `eth_estimateGas` result is supplied by the chosen RPC
provider. It is reproducible against a pinned state but is not itself a mined
receipt. A historical gas estimate remains a simulation under the node's
estimation policy. For independent evidence, retain:

- chain ID, block number, block hash, and block timestamp;
- contract addresses;
- model identifier and reconstructed core hash;
- exact packed input bytes;
- local output and EVM output;
- RPC endpoint and software versions; and
- transaction hash and receipt when a transaction path is used.

# Deployed GL1F system

The GL1F contracts are deployed on GenesisL1 and the browser application is
live at [gl1f.com](https://gl1f.com). Version 0.2.3 changes client software,
tests, and documentation; it does not redeploy the contracts or replace their
addresses.

## Canonical manifest

[`deployments/genesisl1.json`](../deployments/genesisl1.json) is the
machine-readable deployment record consumed by the Research page.

| Component | GenesisL1 address | Observed runtime bytes |
|---|---|---:|
| ModelStore | `0x9CdbC23392648Bd27B4A5eD09c0fEa9452454B54` | 714 |
| ModelRegistry | `0x33c9844F77a07e36B98f0FFf8201B8A8b02c2a69` | 12,725 |
| ModelNFT | `0x44Dc1c54B8D579B42d78cC21cf8260DC0A3279fA` | 5,968 |
| ForestRuntime | `0xD2fD0cf461a6cb56Fc08d9aEc120833D8E79044E` | 17,794 |
| ModelMarketplace | `0xA9bfa0a719b7F73cE85CA1E7f23af626D383fB46` | 3,828 |

Runtime sizes were observed at GenesisL1 block 13,342,043. They are chain
observations, not a claim that the local source compiles to byte-identical
deployed code. The local compile suite separately checks source compilation
with Solidity 0.8.20, IR optimization, 200 optimizer runs, and Istanbul EVM
targeting.

## Replayable deployment evidence

A fresh observation at block **13,602,838**, dated **6 September 2026,
17:20:41 UTC**, retained all 12 model cores (31,185,324 bytes), 1,306 data
chunks, 12 address tables, five deployed contract runtimes, and the raw
request/response transcript. The block hash is
`0xfd3da1020c37ee3c1fe7cd0a6060dbc5ec3ec5fb90c0b812256dc33e467dace3`.

All 108 conformance outputs matched the local evaluator. The compressed
[archive](../benchmarks/results/live_chain_archive_13602838.tar.gz) and
[replay record](../benchmarks/results/live_chain_replay_13602838.json)
support offline checking with a separate Python implementation. The block
and responses remain provider-reported evidence, without authenticated
consensus or account-proof verification. Exact commands are in
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

## Historical deployment summaries

The recorded witness selected:

- block number: `13,342,043`
- block hash:
  `0xffd825db1bb2534052a604db9584d361111d8bc9e19d753b0ee3861bf320d1b9`
- block time: `2026-07-24T17:23:05Z`

The block hash was re-read after all calls and remained unchanged. At that
state, the witness observed:

- 12 active model-token records from two creator addresses;
- 31,185,324 reconstructed model bytes;
- 12/12 content commitments matching `keccak256(core)`;
- all nine version-aware registry/header/storage relations passing for every
  record; and
- 12/12 exact local/provider-returned all-zero-input integer results.

See the
[`human-readable witness`](../benchmarks/results/LIVE_CHAIN_WITNESS.md) and
the [`reproducer`](../benchmarks/live_chain_witness.mjs). The raw provider
responses and reconstructed deployed bytes for that July observation are not
included in this repository.

That record is the frozen original witness: it uses one all-zero vector per
model and contains no live-model gas study. It is retained unchanged for
provenance.

The separate
[`extended witness`](../benchmarks/results/LIVE_CHAIN_WITNESS_EXTENDED_V2.md)
keeps the same block and 31,185,324-byte corpus and applies nine deterministic
exact-int32 vectors per model. It reports:

- 108/108 exact local/provider-returned read-call outputs;
- 108/108 historical `eth_estimateGas` results, with zero estimation errors;
- a maximum estimate of 86,178,402, or 8.6178402% of the pinned block's
  1,000,000,000 gas limit;
- 1,604,970 source-instrumented model-code reads; and
- 17 cross-chunk reads and derived boundary-path temporary allocations.

The vectors include the all-zero and both int32-extreme vectors plus two
non-sentinel root-threshold triplets at \(\theta-1\), \(\theta\), and
\(\theta+1\). Roots are necessarily visited, so these triplets directly exercise
strict-greater-than behavior. All 17 reported crossings occurred in token 1's
v2 paths; token 9 had none in those executed paths. Instrumented reads are
source-level metrics, not opcode traces. Historical gas estimates are RPC
simulations, not transaction receipts, inclusion evidence, affordability
guarantees, or portable fee forecasts.

The values above are retained as a provider-attested historical summary. The
repository does not include the raw July responses needed for offline
reproduction of those summaries. The separate September archive above
preserves a fresh observation and does not replace their provenance.

The nine checked relations are:

1. `nFeatures = F`;
2. `nTrees = T` for v1 and `nTrees = K*R` for v2;
3. `depth = d`;
4. `baseQ = a0` for v1 and the v2 reserved-field convention `baseQ = 0`;
5. `scaleQ = Q`;
6. `tablePtr = P`;
7. `totalBytes = ell`;
8. `numChunks = N`; and
9. `chunkSize = c`.

## Compatibility rule

The live UI defaults in `src/common.js`, command-line minter defaults,
reproducer, and deployment manifest use the same chain ID and addresses.

To check address consistency:

```bash
python3 - <<'PY'
import json
import pathlib
import re

root = pathlib.Path(".")
manifest = json.loads((root / "deployments/genesisl1.json").read_text())
addresses = {
    entry["address"].lower()
    for entry in manifest["contracts"].values()
}
common = (root / "src/common.js").read_text()
configured = {
    value.lower()
    for value in re.findall(r"0x[a-fA-F0-9]{40}", common)
}
assert addresses == configured
print("deployment addresses agree")
PY
```

## Operational boundary

- Model cores stored through GL1F are intentionally public.
- The v1 wire format stores `nTrees` as unsigned 32-bit; the deployed registry
  imposes a narrower unsigned 16-bit cap.
- Task meaning is outside the v1/v2 core hash: the commitment does not
  distinguish regression from binary or multiclass from multilabel.
- `ModelNFT` is a custom transferable model-token record; this release does
  not assert full ERC-721 compliance.
- Paid and signed modes gate calls to deployed runtime functions; they do not
  provide model confidentiality.
- A selected RPC can omit, alter, or lag responses. Pin state and compare
  independent endpoints for consequential observations.
- The chain establishes bytes, ordering, and execution state. It does not
  establish scientific truth, data quality, model accuracy, authorship,
  fairness, safety, or fitness for a decision.

## Repeating the witness

Install the pinned Node dependencies and run:

```bash
npm ci
GL1F_RPC_URL=https://rpc.genesisl1.org \
node benchmarks/live_chain_witness.mjs \
  /tmp/gl1f-live-chain-witness.json \
  13342043
```

Run the separate extended read-only protocol with:

```bash
GL1F_RPC_URL=https://rpc.genesisl1.org \
GL1F_WITNESS_BLOCK=13342043 \
node benchmarks/live_chain_witness.mjs \
  /tmp/gl1f-live-chain-witness-extended.json \
  --extended
```

Both modes are read-only and write their observations to the selected local
output file. See
[`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for the complete protocol.

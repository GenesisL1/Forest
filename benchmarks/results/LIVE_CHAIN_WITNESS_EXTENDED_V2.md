# GL1F extended live-chain conformance and gas witness

**Chain:** GenesisL1, chain ID 29\
**Number-pinned block:** 13,342,043\
**Block hash:** `0xffd825db1bb2534052a604db9584d361111d8bc9e19d753b0ee3861bf320d1b9`\
**Block time:** 2026-07-24T17:23:05.000Z\
**Pinned block gas limit:** 1,000,000,000 (`0x3b9aca00`)\
**RPC used:** `https://rpc.genesisl1.org`\
**Reorganization guard:** block hash re-read and unchanged after all calls\
**Public evidence form:** historical summary only; the raw machine record is not included\
**Reproducer:** [`../live_chain_witness.mjs`](../live_chain_witness.mjs)

## Result

This repository retains the summary and read-only reproducer. Raw vectors,
model/runtime bytes, and provider responses are not included, so these results
are not presented as a
self-contained public replay archive.

| Check | Result |
|---|---:|
| Active registered models | 12 |
| Reconstructed core bytes | 31,185,324 |
| Content commitments matching | 12/12 |
| Extended conformance vectors | 108 |
| Exact local/provider-returned vector results | 108/108 |
| Historical gas estimates returned | 108/108 |
| Historical gas-estimation RPC errors | 0 |
| Maximum historical gas estimate | 86,178,402 |
| Maximum estimate / pinned block limit | 8.6178402% |
| Instrumented model-code reads | 1,604,970 |
| Cross-chunk reads / derived temporary allocations | 17 |

## Per-model evidence

| Token | Title | Vectors | Exact provider/local | Gas returned | Historical gas range | Max / block limit | Cross-chunk reads |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Iris model | 9 | 9/9 | 9/9 | 22,782,413–22,784,217 | 2.2784217% | 17 |
| 2 | Raisin classification | 9 | 9/9 | 9/9 | 3,213,782–3,213,872 | 0.3213872% | 0 |
| 3 | Boston housing prices | 9 | 9/9 | 9/9 | 4,397,999–4,398,166 | 0.4398166% | 0 |
| 4 | PV_efficiency_57degree_NorthL_Weights | 9 | 9/9 | 9/9 | 3,355,295–3,355,955 | 0.3355955% | 0 |
| 5 | Bitcoin 24h volatility model | 9 | 9/9 | 9/9 | 419,789–422,265 | 0.0422265% | 0 |
| 6 | AI predicting $ETH +2% in 5h | 9 | 9/9 | 9/9 | 48,187,258–48,195,195 | 4.8195195% | 0 |
| 7 | AI predicting $ETH -2% in 5h | 9 | 9/9 | 9/9 | 86,163,819–86,178,402 | 8.6178402% | 0 |
| 8 | AI predicting $XRP +2% in 5h | 9 | 9/9 | 9/9 | 31,373,837–31,379,108 | 3.1379108% | 0 |
| 9 | AI predicting $XRP -2% in 5h | 9 | 9/9 | 9/9 | 57,181,535–57,190,348 | 5.7190348% | 0 |
| 10 | AI predicting $ZEC +4% in 5h | 9 | 9/9 | 9/9 | 50,986,502–50,993,025 | 5.0993025% | 0 |
| 11 | AI predicting $ZEC -4% in 5h | 9 | 9/9 | 9/9 | 32,514,162–32,515,585 | 3.2515585% | 0 |
| 12 | AI gating ZEC ±4% vol-sufficiency in 5h | 9 | 9/9 | 9/9 | 155,671–157,902 | 0.0157902% | 0 |

## Deterministic vector protocol

Each model receives three exact int32 baselines: the all-zero vector, the
all-`INT32_MIN` vector, and the all-`INT32_MAX` vector. The script then
extracts unique, non-sentinel root `(feature, threshold)` pairs from the
serialized ensemble. Roots are always visited, so each retained
`threshold-1`, equality, and `threshold+1` triplet directly exercises the
runtime's strict-greater-than branch at the named split. Candidates are
selected deterministically across serialized order; exact packed inputs
are deduplicated and complete triplets are retained up to the configured
cap. The machine record stores every int32 vector, packed little-endian
hex string, source rule, split provenance, local result, EVM result, gas
outcome, path digest, read count, and chunk-boundary count.

The local evaluator consumes int32 Q-values directly and accumulates with
arbitrary-precision integers. This avoids floating-point input conversion.
The EVM comparison uses only historical `eth_call`; gas is requested with
historical `eth_estimateGas(transaction, blockTag)`. Unsupported or failed
estimates remain explicit error records and are never replaced by a latest-
block estimate. Maximum utilization records retain the estimate numerator,
the pinned header's block-gas-limit denominator, and rounded decimal/percent
renderings, so the reported percentages are independently recomputable.

## Interpretation boundary

The recorded provider-attested summary reports public-byte reconstruction and
execution agreement for the listed deployment state and inputs. The raw
responses are unavailable in this repository. Gas values are RPC simulations,
not transaction receipts, and may be subject to the selected node's
estimation policy. Read/boundary metrics are exact source-level
instrumentation of model-code reads for the archived paths; they are not an
opcode trace or a decomposition of total gas. The study does not establish
training provenance, predictive accuracy, calibration, fairness, safety,
authorship, or fitness for use.

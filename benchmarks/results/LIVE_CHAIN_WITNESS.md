# GL1F live-chain reconstruction witness

**Chain:** GenesisL1, chain ID 29\
**Number-pinned block:** 13,342,043\
**Block hash:** `0xffd825db1bb2534052a604db9584d361111d8bc9e19d753b0ee3861bf320d1b9`\
**Block time:** 2026-07-24 17:23:05 UTC\
**RPC used for the archived record:** `https://29.rpc.thirdweb.com`\
**Reorganization guard:** block hash re-read and unchanged after all witness calls\
**Public evidence form:** historical summary only; the raw machine record is not included\
**Reproducer:** [`../live_chain_witness.mjs`](../live_chain_witness.mjs)

## Result

This repository preserves the numerical summary and the read-only reproducer.
Raw model/runtime bytes and provider-returned JSON are not included; the table
below is therefore an observation report,
not a self-contained replay archive.

The witness reconstructed all model cores from immutable GL1C runtime code at
one pinned historical state:

| Check | Result |
|---|---:|
| Active registered models | 12 |
| Distinct creator addresses | 2 |
| Reconstructed core bytes | 31,185,324 |
| Content commitments matching `keccak256(core)` | 12/12 |
| Canonical storage shape and applicable registry/header checks | 12/12 |
| Exact local/provider-returned zero-vector integer results | 12/12 |
| Model size range | 816–9,523,096 bytes |
| Largest pointer table | 397 chunks |

The recorded summary reports that every active model observed at the
number-pinned block satisfied the artifact, manifest, and execution checks. It
does not identify which client workflow produced those records. It also reports
that a 9.52 MB, 1,552-tree,
depth-9 ensemble can be reconstructed from 397 immutable code chunks and
produce the same integer result locally and through the deployed runtime.

## Per-model record

| Token | Title | Ver. | Features | Depth | Total trees | Bytes | Chunks | Exact checks |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Iris model | 2 | 4 | 3 | 1,800 | 158,436 | 7 | hash, header, local/provider |
| 2 | Raisin classification | 1 | 7 | 5 | 169 | 63,568 | 3 | hash, header, local/provider |
| 3 | Boston housing prices | 1 | 13 | 6 | 196 | 148,984 | 7 | hash, header, local/provider |
| 4 | PV efficiency, 57° North | 1 | 4 | 3 | 275 | 24,224 | 2 | hash, header, local/provider |
| 5 | Bitcoin 24 h volatility model | 1 | 53 | 5 | 20 | 7,544 | 1 | hash, header, local/provider |
| 6 | ETH +2% in 5 h | 1 | 26 | 8 | 1,496 | 4,583,768 | 191 | hash, header, local/provider |
| 7 | ETH −2% in 5 h | 1 | 27 | 8 | 2,500 | 7,660,024 | 320 | hash, header, local/provider |
| 8 | XRP +2% in 5 h | 1 | 33 | 8 | 1,008 | 3,088,536 | 129 | hash, header, local/provider |
| 9 | XRP −2% in 5 h | 1 | 33 | 9 | 1,552 | 9,523,096 | 397 | hash, header, local/provider |
| 10 | ZEC +4% in 5 h | 1 | 33 | 7 | 1,789 | 2,733,616 | 114 | hash, header, local/provider |
| 11 | ZEC −4% in 5 h | 1 | 33 | 8 | 1,042 | 3,192,712 | 134 | hash, header, local/provider |
| 12 | ZEC ±4% volatility gate | 1 | 47 | 3 | 9 | 816 | 1 | hash, header, local/provider |

## Exact procedure

For each active token, the script:

1. requires GenesisL1 chain ID 29, pins every getter and `eth_getCode` request
   to the stated block number, and rechecks that block's recorded hash after
   all calls;
2. reads the exact GL1C pointer table, requires exactly `numChunks` slots, and
   requires the high 12 bytes of every 32-byte address slot to be zero;
3. requires `numChunks == ceil(totalBytes/chunkSize)`;
4. checks every GL1C prefix and exact canonical chunk payload length;
5. reconstructs exactly `totalBytes`, with no undeclared trailing payload;
6. strictly decodes the GL1F header and tree structure;
7. checks the core length, storage fields, and every header field that has an
   applicable registry consistency counterpart;
8. compares `keccak256(core)` with the registered model identifier; and
9. packs a zero int32 vector, evaluates the same core locally, and compares
   the exact integer output with the EVM read call.

## Interpretation boundary

The summary reports byte-level reconstruction and execution agreement for the
listed deployment state. The raw responses are unavailable in this repository,
so this is provider-attested rather than independently reproducible evidence.
It does not establish the accuracy, calibration, causal
meaning, authorship, training provenance, fairness, safety, or future
performance of any model. The selected RPC supplies historical state and
`eth_call` responses; a read call is not itself a transaction receipt. The
registry does not enforce the entire binding for arbitrary future entries.

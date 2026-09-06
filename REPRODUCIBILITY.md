# Reproducing GL1F Forest v0.2.3

This document covers the executable evidence for the GL1F paper and software.
All commands run from the repository root. No command in the default
verification suite connects to GenesisL1 or submits a live-chain transaction.

## Environment

The verified Linux evidence profile uses:

- Python 3.12.13;
- NumPy 2.3.5;
- setuptools 84.0.0, wheel 0.48.0, and packaging 26.3 for the offline
  wheel-build test;
- Node.js 24.19.0;
- g++ 13.3.0 with C++17, `-ffp-contract=off`, and `-fno-fast-math`;
- Solidity 0.8.20, optimizer 200, `viaIR=true`, `evmVersion=istanbul`;
- ethers 6.15.0 and Ganache 7.9.2.

Python 3.10+ and Node.js 20+ are supported for normal use. The runtime profile
is pinned by `.nvmrc`, `package-lock.json`, and
`requirements-evidence-linux-x86_64.txt`; the wheel-build tools are fixed in
the install command below and in CI.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --only-binary=:all: \
  setuptools==84.0.0 wheel==0.48.0 packaging==26.3
python -m pip install --require-hashes \
  -r requirements-evidence-linux-x86_64.txt
npm ci
```

## Verification

Run the complete local suite:

```bash
make verify
```

The individual evidence commands are:

| Evidence | Command |
|---|---|
| Cross-engine training and format parity | `python3 -m unittest -v tests.publication.test_publication` |
| Mint-input validation | `python3 -m unittest -v tests.publication.test_mint_validation` |
| Resumable mint workflow | `python3 -m unittest -v tests.publication.test_mint_workflow` |
| Deployment identity checks | `python3 -m unittest -v tests.publication.test_deployment_identity` |
| Local trainer server | `python3 -m unittest -v tests.test_local_trainer_server` |
| Python wheel build and contents | `python3 -m unittest -v tests.publication.test_wheel_package` |
| Formal boundary witnesses | `python3 -m unittest -v tests.contracts_publication.test_formal_properties` |
| Offline archive verifier fixtures | `python3 -m unittest -v tests.publication.test_independent_archive` |
| Paper/result invariants | `python3 tests/scientific_invariants.py --public` |
| Static UI checks | `python3 tests/ui_static_check.py` |
| Pinned Solidity compilation | `node tests/contracts_publication/compile_contracts.mjs` |
| Isolated EVM integration | `node tests/contracts_publication/test_evm_integration.mjs` |

The isolated-EVM test uses an in-process Ganache chain. It does not deploy to,
write to, or alter GenesisL1.

## Checked-in evidence

| Record | Contents |
|---|---|
| `benchmarks/results/parity_matrix.json` | 25 distinct JavaScript/Python/C++ training profiles, five auxiliary controls, and one standalone IEEE-754 operation-order witness; status `PASS` |
| `benchmarks/results/evm_integration.json` | Local deployment, chunk reconstruction, content-hash checks, scalar/vector/class inference, 18 view comparisons, two isolated transactions, and zero mismatches |
| `benchmarks/results/evm_scaling_benchmark.json` | Six scalar shapes and 72 reference/runtime comparisons with zero mismatches |
| `benchmarks/results/storage_comparison.json` | Six identical scalar cores in code-as-data and packed Solidity storage; 72 three-way output comparisons, raw gas observations and source digests |
| `benchmarks/results/publication_benchmark.json` | Thirty measured runs per engine and workload, including environment and model digests |
| `deployments/genesisl1.json` | Chain ID, pinned block, and deployed contract addresses used by the paper |

The JSON parity and local-EVM records contain SHA-256 digests of their source
inputs. `tests/scientific_invariants.py --public` verifies those digests and
the numerical claims used by the paper.

Regenerate the deterministic evidence records with:

```bash
python3 benchmarks/generate_parity_evidence.py \
  --out benchmarks/results/parity_matrix.json
node tests/contracts_publication/test_evm_integration.mjs \
  --out benchmarks/results/evm_integration.json
```

## Benchmarks

Regenerate the CPU timing study:

```bash
python3 benchmarks/publication_benchmark.py \
  --rows 3000 --features 12 --trees 60 --repeats 30 \
  --out benchmarks/results/publication_benchmark.json
```

Regenerate the isolated-EVM scaling study:

```bash
npm run benchmark:evm
```

Reproduce the controlled storage comparison:

```bash
node benchmarks/storage_comparison.mjs
```

This local experiment stores identical canonical bytes through the production
code-as-data backend and a packed Solidity-storage comparator. It records
per-model write receipts separately from common registry and shared deployment
costs, and measures view-gas estimates on the same inputs. The comparator is
benchmark code, not a production replacement contract.

Model digests and integer comparison results are conformance checks. Wall
times and gas estimates depend on the host, compiler, EVM implementation, and
execution path and are not portable performance guarantees.

## Paper

Build the paper and formal supplement with:

```bash
make pdfs
```

The outputs are `GL1F.pdf` and `paper/GL1F_Formal_Supplement.pdf`.
The main manuscript uses the Ledger author template with publication-status
metadata omitted; both documents remain unreviewed author materials.

## Replayable pinned-chain observation

The main article uses a fresh observation at GenesisL1 block **13,602,838**,
with timestamp **2026-09-06 17:20:41 UTC** and hash
`0xfd3da1020c37ee3c1fe7cd0a6060dbc5ec3ec5fb90c0b812256dc33e467dace3`.
The compressed archive is
[`benchmarks/results/live_chain_archive_13602838.tar.gz`](benchmarks/results/live_chain_archive_13602838.tar.gz).
Its [result record](benchmarks/results/live_chain_replay_13602838.json)
contains the archive checksum and independent verification counts.

The archive contains the 12 model cores, exact storage runtimes and tables,
five deployed contract runtimes, pinned block records, raw RPC transcript,
108 packed conformance vectors and outputs, collector source, and file digests.
It preserves 31,185,324 model bytes reconstructed from 1,306 chunks. The raw
transcript records the provider's responses; it does not authenticate chain
consensus or verify account-proof paths.

Reproduce the checks offline with the Python standard library:

```bash
mkdir -p tmp/chain-replay
tar -xzf benchmarks/results/live_chain_archive_13602838.tar.gz \
  -C tmp/chain-replay
python3 benchmarks/independent_archive_verify.py \
  --manifest tmp/chain-replay/live_chain_archive_13602838/manifest.json
```

The expected result is `verified`: 12 models, 31,185,324 core bytes, and 108
vectors. This checks raw-observation consistency, file digests, strict parsing,
reconstruction, content commitments, registry relations, and integer outputs.
It requires no RPC endpoint and submits no transactions.

A new collection requires a fresh extended witness for the same explicitly
selected block. The collector accepts `GL1F_ARCHIVE_BLOCK`,
`GL1F_ARCHIVE_BLOCK_HASH`, and `GL1F_ARCHIVE_WITNESS`; run
`node benchmarks/archive_live_chain_state.mjs --help` for collection options.
An unavailable historical state is an error, never a fallback to latest state.

## Historical summary

Earlier factual summaries report a provider-attested observation at GenesisL1 block
13,342,043. The method is implemented by
`benchmarks/live_chain_witness.mjs`; the factual summaries are stored in
`benchmarks/results/LIVE_CHAIN_WITNESS.md` and
`benchmarks/results/LIVE_CHAIN_WITNESS_EXTENDED_V2.md`.

A fresh read-only run requires an RPC endpoint that serves historical state:

```bash
GL1F_RPC_URL=https://rpc.genesisl1.org \
node benchmarks/live_chain_witness.mjs \
  /tmp/gl1f-live-chain-witness.json 13342043 --extended
```

The repository does not contain the raw provider responses or reconstructed
deployed model/runtime bytes behind the recorded historical summaries.
Consequently, the public bundle supports inspection of the method and a fresh
RPC-assisted rerun, but not an offline replay of those historical responses.
An RPC response is provider-attested data, not independent proof of chain
consensus or finality.

## Scope

These procedures test serialization, deterministic execution, cross-engine
conformance, and the stated deployment observations. They do not establish
predictive accuracy, dataset provenance, calibration, causality, fairness,
privacy, authorship of a model, or fitness for a particular application.

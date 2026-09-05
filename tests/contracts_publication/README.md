# Contract/formal publication checks

Run from the repository root.

```bash
python3 -m unittest -v tests.contracts_publication.test_formal_properties
node tests/contracts_publication/compile_contracts.mjs
npm run test:evm
```

The Python suite supplies executable boundary witnesses for:

- complete-tree traversal and leaf-index closure;
- exact one-, two-, and four-byte reads across chunk boundaries when
  `chunkSize >= 4`;
- the concrete failure mode when a four-byte field spans three chunks;
- pointer-table capacity;
- fixed-point rounding bounds;
- JavaScript exact-integer accumulator bounds; and
- checked-in Python/C++ core-byte parity.

The compiler check intentionally pins the source-compatible profile:
Solidity 0.8.20, optimizer 200, `viaIR=true`, and `evmVersion=istanbul`.
Override only for diagnostic testing:

```bash
GL1F_SOLC_VERSION=0.8.30 node tests/contracts_publication/compile_contracts.mjs
```

At the tested revision, sampled newer versions fail in the IR/Yul compiler.

## Local EVM integration

`npm run test:evm` is the executable end-to-end publication witness. It:

- compiles the unmodified contracts with local `solc` 0.8.20,
  `viaIR=true`, optimizer 200, and `evmVersion=istanbul`;
- starts an isolated in-process Ganache chain and deploys `ModelStore`,
  `ModelRegistry`, `ModelNFT`, and `ForestRuntime`;
- verifies registry/NFT/runtime wiring;
- serializes minimal canonical v1 scalar and v2 three-output models;
- writes 17-byte GL1C chunks plus the address table, reconstructs the model
  from deployed runtime code, and checks `keccak256(reconstructed) == modelId`;
- registers both models through the public registry API;
- compares an independent bigint reference interpreter with `predictView`,
  `predictMultiView`, and `predictClassView` on threshold equality,
  strict-greater, signed-int32 extreme, and equal-logit boundary cases; and
- executes the transaction inference paths and reports measured deployment,
  publication, and inference gas.

The deliberately small 17-byte test chunks force multi-byte fields to cross
chunk boundaries. They are a boundary witness, not a recommended production
chunk size. Gas figures printed by the test are Ganache measurements for
these tiny fixtures and compiler settings; they are reproducible regression
evidence, not estimates of a particular public network's fees. The protocol
test concerns byte reconstruction and execution agreement.

The LaTeX theorem fragment has a standalone smoke wrapper:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -output-directory=/tmp \
  tests/contracts_publication/formal_results_smoke.tex
```

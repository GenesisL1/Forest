# GL1F publication validation

This directory contains an independent structural parser/reference inference
implementation and an executable parity matrix for the production Python,
C++, and browser JavaScript trainers.

Run from the repository root:

```bash
python3 -m unittest -v tests.publication.test_publication
```

The suite:

- compiles `cpp/train_gl1f_cpp.cpp` with a C++17 compiler in a temporary
  directory;
- invokes the production `train_gl1f.py` CLI;
- runs the production `src/train_worker.js` behind a minimal Node WebWorker
  adapter;
- tests regression, binary, multiclass, and multilabel tasks with linear and
  quantile binning;
- retains a generated 4,000-row, four-class profile that detects whether the
  softmax denominator is accumulated from binary64 rather than prematurely
  rounded binary32 exponentials;
- exercises class weighting, stratification, Train+Val refit,
  repeated-run determinism, GL1X framing,
  inference, JavaScript-compatible rounding, and malformed byte streams;
- includes a one-tree regression control whose learning rate places a leaf
  immediately below a half-integer, plus scalar and saturation witnesses;
- enables early-stopping and plateau-schedule configurations; in the frozen
  fixtures they reach the full tree budget and do not demonstrate an actual
  early stop or plateau learning-rate reduction;
- compares only core GL1F bytes. GL1X packages include timestamps and
  front-end-specific metadata and are therefore not expected to be identical.

The test profile uses finite float32 input matrices, `1 <= scaleQ <= 2^31-1`,
positive dimensions/counts, a fixed feature/output order, shared normalization
of an explicit zero RNG seed, and no
platform-specific fast-math compiler flags. Those conditions are part of the
defensible parity claim; arbitrary CSV preprocessing and malformed/unbounded
parameters are outside it.

For timing:

```bash
python3 benchmarks/publication_benchmark.py
```

Benchmark output is operational end-to-end wall time. Python and C++ consume
the same CSV; the JavaScript worker consumes equivalent JSON, so its timing is
reported separately and must not be presented as a pure kernel comparison.

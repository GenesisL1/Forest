# GL1F publication benchmark

**Recorded:** 2026-09-05T22:23:15.350070Z

**Source revision:** `72e91a5ba6d335fa5c8cc805c2ac40eb94bbc0fb` (clean)

## Result

On the recorded Linux/AMD EPYC environment, the C++ CLI completed the
end-to-end regression case in a median **0.0365 s** versus **0.3046 s** for
Python (a ratio of medians of **8.35×**), and binary classification in
**0.0437 s** versus **0.3149 s** (**7.20×**). All three engines emitted
SHA-256-identical core
GL1F bytes in both cases.

| Task | Core bytes | Python median (IQR width), s | C++ median (IQR width), s | Ratio of medians | Node median (IQR width), s |
|---|---:|---:|---:|---:|---:|
| Regression | 11,064 | 0.3046 (0.0124) | 0.0365 (0.0017) | 8.35× | 0.1384 (0.0088) |
| Binary classification | 11,064 | 0.3149 (0.0177) | 0.0437 (0.0027) | 7.20× | 0.1610 (0.0130) |

These are measurements from one disclosed environment, not universal speed
claims. Each bracketed value is the interquartile range over 30 timed
observations. The host was neither isolated nor CPU-pinned, so the ratios do
not define a cross-host speedup distribution.

## Workload

- 3,000 rows and 12 numeric float32-compatible features
- 60 fixed-depth trees, depth 4
- learning rate 0.075; minimum leaf size 8
- 32-bin linear histograms
- `scaleQ = 100000`; seed `20260724`
- one warm-up followed by 30 timed repetitions per engine
- round-robin rotated execution order
- C++ flags:
  `-O3 -DNDEBUG -std=c++17 -ffp-contract=off -fno-fast-math`

Each observation includes process startup, input parsing, training,
serialization, and output writing. Python and C++ read the same CSV. The
browser worker runs under Node and reads an equivalent JSON numeric matrix;
therefore its result is operational and must not be treated as a pure
language-kernel comparison.

## Environment

- Linux 6.18.35 x86-64, glibc 2.39
- AMD EPYC 9V74 host; 9 logical CPUs visible
- Python 3.12.13; NumPy 2.3.5
- Node 24.19.0
- g++ 13.3.0

## Reproduce

From the repository root:

```bash
python3 benchmarks/publication_benchmark.py \
  --rows 3000 \
  --features 12 \
  --trees 60 \
  --repeats 30 \
  --out benchmarks/results/publication_benchmark.json
```

The machine-readable raw times, quartiles, IQR, median absolute deviation,
hashes, source revision, parameters, and environment are in
`publication_benchmark.json`.

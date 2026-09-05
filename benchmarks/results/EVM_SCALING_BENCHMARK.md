# GL1F local-EVM scalar scaling benchmark

**Status:** PASS

**Recorded:** 2026-09-05T21:28:04.536Z

**Source revision:** `d27e1f70fa88d12b53dc8ead49cf9df6174c254d`

**Compiler:** 0.8.20+commit.a1b79de6.Emscripten.clang; viaIR; optimizer 200; EVM target Istanbul

**Execution client:** ganache 7.9.2; hardfork shanghai

**Comparisons:** 72 exact reference/EVM profile-vector comparisons; 0 mismatches

These observations describe one local compiler/client/profile. They are not
a live-network fee forecast, a marginal storage price, or a proof that every
encoded model fits a transaction limit.

| Trees | Depth | Bytes | Chunks | Decisions | Tree-body primitive reads | Rounded mean estimated gas | Min-max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 3 | 1,784 | 1 | 60 | 140 | 286,247 | 286,211-286,280 |
| 50 | 3 | 4,424 | 1 | 150 | 350 | 644,454 | 644,382-644,535 |
| 50 | 4 | 9,224 | 1 | 200 | 450 | 810,777 | 810,689-810,888 |
| 100 | 4 | 18,424 | 1 | 400 | 900 | 1,576,945 | 1,576,732-1,577,114 |
| 200 | 4 | 36,824 | 2 | 800 | 1,800 | 3,121,210 | 3,120,837-3,121,528 |
| 50 | 6 | 38,024 | 2 | 300 | 650 | 1,146,414 | 1,146,275-1,146,566 |

## Descriptive fits

- Against `decisions`: gas = 45,474.153 + 3,828.902 x work; R^2 = 0.99937982; maximum absolute relative residual = 4.16%.
- Against `primitiveReads`: gas = 42,923.392 + 1,708.277 x work; R^2 = 0.99998002; maximum absolute relative residual = 1.45%.

Decision and leaf-read counts follow exactly from canonical tree geometry.
The tree-body primitive-read count is exact for the benchmarked scalar decoder.
Fitted coefficients are empirical and can change with compiler, client,
hardfork, chunk geometry, calldata, address warmth, and model layout.

## Reproduction

```bash
npm ci
npm run benchmark:evm
```

The JSON record retains every per-vector gas estimate, model identifier,
write/registration gas observation, compiler profile, and host description.

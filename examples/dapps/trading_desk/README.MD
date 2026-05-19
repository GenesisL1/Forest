# Forest Trading Desk

Unified single-page trading desk for all 6 GenesisL1 GBDT models across 3 assets.

## Quick deploy

Drop `trading_desk.html` in the same folder as your existing CSV files:

```
/var/www/forest/dapps/
├── trading_desk.html        ← this file
├── ethup2_predictions.csv   (from ethup2_predictor_daemon.py)
├── ethdn2_predictions.csv   (from ethdn2_predictor_daemon.py)
├── xrpup_predictions.csv    (from xrpup_predictor_daemon.py)
├── xrpdn_predictions.csv    (from xrpdn_predictor_daemon.py)
├── zecup_predictions.csv    (from zecup_predictor_daemon.py)
├── zecdn_predictions.csv    (from zecdn_predictor_daemon.py)
└── (existing dual dApps stay where they are — this is additive)
```

No build step, no dependencies beyond ethers.js (loaded from CDN).

## Tabs

| Tab        | Purpose                                                      |
|-----------|--------------------------------------------------------------|
| Overview   | All 3 assets in one view; latest snapshot from CSVs · ticker tape · "Execute Live (×6)" runs all 6 forests on-chain in parallel |
| Backtest   | Per-asset historical signal chart + Binance OHLC scoring (L1/L2/L3/X grading) |
| History    | Per-asset inference history table (inner-joined CSVs)        |
| Inference  | Per-asset historical-candle inspector — pick any past UTC 15m bar, run both forests at that timepoint |
| Models     | Per-model info cards · 6 models · filterable by individual model |

## Models

| Token | Asset | Side | Target  | Test acc | Test logloss | Trees |
|-------|-------|------|---------|----------|--------------|-------|
| #6    | ETH   | UP   | +2.0%   | 95.66%   | 0.1094       | 1,496 |
| #7    | ETH   | DOWN | −2.0%   | 95.62%   | 0.1122       | 2,500 |
| #8    | XRP   | UP   | +2.0%   | 94.63%   | 0.1403       | 1,008 |
| #9    | XRP   | DOWN | −2.0%   | 94.51%   | 0.1387       | 1,552 |
| #10   | ZEC   | UP   | +4.0%   | 93.54%   | 0.1592       | 1,789 |
| #11   | ZEC   | DOWN | −4.0%   | 94.58%   | 0.1417       | 1,042 |

Feature schemas: ETH UP uses 26 features, ETH DOWN uses 27 (different subsets of
the 33-feature superset). XRP and ZEC models all use the full 33-feature schema.
The Overview Execute button computes the union once and packs each model's
subset for its individual predictView call.

## Notes

- The Overview tab pulls latest predictions from the CSV files on each refresh
  (cheap, no on-chain calls). The Execute Live button is the only thing that
  hits the chain; it runs 6 parallel `predictView` eth_calls.
- The Backtest tab fetches Binance Futures candles for outcome scoring.
- The Historical Inference tab fetches the surrounding window for a chosen
  past candle and runs both models against it on-chain.
- Theme toggle (sun/moon icon, top right) persists in localStorage.
- All times are UTC throughout.

## License & disclaimer

CC BY-SA 4.0. Experimental research artifact. Probabilities are not investment
advice. Out-of-sample test accuracy on a fixed window does not guarantee live
performance.

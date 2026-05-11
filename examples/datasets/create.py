#!/usr/bin/env python3
"""
create.py — Unified Binance dataset builder with triple-barrier labels
======================================================================

Combines the feature catalogues of three source scripts:
    1) btc_features.py                  (full BTC indicator set, hours-based windows)
    2) btc_features_gl1f_vol_trimmed.py (vol-prediction features, hours-based)
    3) create_live_dataset_eth.py       (15m ETH live features, BAR-based windows)

Features whose names already appear in an earlier source are computed once
(no duplicate columns).

Strict no-leak guarantee
------------------------
At each row N the dataset stores:
    OHLCV[N], features[N]  using only data from rows [0..N]  (past + present)
    label[N]               using only data from rows [N+1..N+horizon] (future)

Labelling (triple-barrier vs. an EMA baseline)
----------------------------------------------
baseline_N = ema{P}(close)[N]   (computed with information available up to N)

Class "up"   : target = baseline_N · (1 + move/100), stop = baseline_N · (1 − retrace/100)
               label = 1 iff some future bar k∈[N+1..N+horizon] hits the target
                       BEFORE any bar in [N+1..k] hits the stop.
                       Same-bar conflicts are resolved conservatively in favour of the stop.
               label = 0 otherwise (timed out, or stopped out).

Class "down" : target = baseline_N · (1 − move/100), stop = baseline_N · (1 + retrace/100)
               symmetric definition with target hit on the LOW and stop on the HIGH.

Multi-value flags
-----------------
--ticker, --candle, --class, --base, --move-pct, --no-retrace, --horizon
all accept comma-separated values and produce a Cartesian product of output files.

Output filename
---------------
{TICKER}_{candle}_{class}_{base}_move{move}_retrace{retrace}_horizon{horizon}_start_{start-date}.csv

Example
-------
    python create.py \
        --ticker ETH \
        --candle 15m \
        --class up,down \
        --base ema5,ema7 \
        --move-pct 1.5 \
        --no-retrace 0.5 \
        --horizon 4h \
        --start-date 01-01-2024
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# =============================================================================
# Constants
# =============================================================================

BASE_URL = "https://fapi.binance.com"   # Binance USDT-M perpetual futures

INTERVAL_TO_MIN: dict[str, int] = {
    "1m":   1, "3m":   3, "5m":  5, "15m":  15, "30m":  30,
    "1h":  60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440,
}

# Hours-based features rely on this maximum lookback (504h ≈ 21 days)
MAX_LOOKBACK_HOURS = 504
# ETH script uses up to 100-bar aggregates; pad a little
MAX_LOOKBACK_BARS  = 200
# Extra warmup buffer to absorb any min-period quirks
WARMUP_PAD_BARS    = 50

INT_CAP = 2_147_480_000  # GL1F-style integer cap, used only to print scaleQ hint

# Slim feature set used in GL1F training. With --small, the output CSV
# contains exactly these columns + label.
#
# 20 original features (R1, 2.0/0.7/5h ETH model, test logloss 0.1254):
#   - returns / scale: r1_log, r12_log
#   - oscillators: RSI14
#   - volatility: ATR_norm14, ATR_ratio100, BollBW50, BW_CHOP100, ATR_HL_ratio100
#   - candle: body_range_ratio
#   - regime: TrendConsist100
#   - returns @ multi-bar: return_5_3, return_60_3, volatility_5, mom_5
#   - calendar: hour, dow
#   - funding: funding_rate, funding_rate_ma_24h, funding_cum_24h
#   - label-aligned: close_to_ema5
#
# +11 new features (Section D in build_features), targeting:
#   - barrier-relative position (dist_ema5_atr, rv6_over_rv24)
#   - bar-level rejection (body_signed_3, wick_imbalance_3)
#   - flow surges (volume_surge_3, taker_imb_sum_6)
#   - directional momentum vs noise (ret_skew_24h, sign_persist_12)
#   - microstructure efficiency (range_efficiency_6)
#   - trend strength (trend_r2_12)
#   - extreme breakouts (dist_high24_atr, dist_low24_atr) [counts as 2]
SMALL_FEATURE_LIST: tuple[str, ...] = (
    # original 20
    "r1_log", "r12_log",
    "RSI14", "ATR_norm14", "body_range_ratio",
    "ATR_ratio100", "BollBW50", "BW_CHOP100",
    "ATR_HL_ratio100", "TrendConsist100",
    "return_5_3", "return_60_3",
    "volatility_5", "mom_5",
    "hour", "dow",
    "funding_rate", "funding_rate_ma_24h", "funding_cum_24h",
    "close_to_ema5",
    # new 11
    "dist_ema5_atr", "rv6_over_rv24",
    "body_signed_3", "wick_imbalance_3",
    "volume_surge_3", "taker_imb_sum_6",
    "ret_skew_24h", "sign_persist_12",
    "range_efficiency_6",
    "trend_r2_12",
    "dist_high24_atr", "dist_low24_atr",
)


# =============================================================================
# CLI
# =============================================================================

def _split_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--ticker",     required=True,
                   help="Comma-separated tickers, e.g. ETH or ETH,BTC. "
                        "Quote currency USDT is assumed (Binance USDT-M futures).")
    p.add_argument("--candle",     required=True,
                   help=f"Comma-separated candle resolutions, "
                        f"any of: {', '.join(INTERVAL_TO_MIN)}.")
    p.add_argument("--class",      required=True, dest="cls",
                   help="Comma-separated class directions: up, down.")
    p.add_argument("--base",       required=True,
                   help="Comma-separated EMA baselines, e.g. ema5,ema7. "
                        "EMA is computed on close with span=N (adjust=False).")
    p.add_argument("--move-pct",   required=True, dest="move_pct",
                   help="Comma-separated move targets in percent, e.g. 1.5,2.0.")
    p.add_argument("--no-retrace", required=True, dest="no_retrace",
                   help="Comma-separated max retraces in percent, e.g. 0.3,0.5.")
    p.add_argument("--horizon",    required=True,
                   help="Comma-separated lookahead horizons, e.g. 4h,2h,30m,1d. "
                        "Each horizon must be an integer multiple of the candle size.")
    p.add_argument("--start-date", required=True, dest="start_date",
                   help="Start date DD-MM-YYYY or YYYY-MM-DD (UTC).")
    p.add_argument("--end-date",   default=None, dest="end_date",
                   help="End date DD-MM-YYYY or YYYY-MM-DD (UTC). Default: now.")
    p.add_argument("--output-dir", default=".",
                   help="Directory to write CSV files into (created if missing).")
    p.add_argument("--no-funding", action="store_true",
                   help="Skip funding-rate features (one less HTTP call per ticker).")
    p.add_argument("--small", action="store_true",
                   help="Emit only the 19-feature slim set "
                        "(R1 / GL1F training; see SMALL_FEATURE_LIST).")
    return p.parse_args()


def parse_date(s: str) -> datetime:
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse date {s!r} (expected DD-MM-YYYY or YYYY-MM-DD)")


def parse_duration_minutes(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("h"):
        return int(s[:-1]) * 60
    if s.endswith("d"):
        return int(s[:-1]) * 1440
    if s.endswith("m"):
        return int(s[:-1])
    raise ValueError(f"Cannot parse duration {s!r} (expected like '4h', '30m', '1d')")


def parse_ema_period(s: str) -> int:
    s = s.strip().lower()
    if not s.startswith("ema"):
        raise ValueError(f"--base must look like 'emaN', got {s!r}")
    return int(s[3:])


# =============================================================================
# Binance fetchers (paginated, with backoff)
# =============================================================================

def _http_get(endpoint: str, params: dict, retries: int = 6) -> list | dict:
    last = None
    for i in range(retries):
        try:
            r = requests.get(BASE_URL + endpoint, params=params, timeout=30)
            if r.status_code in (418, 429):
                wait = 2 ** (i + 1)
                print(f"  rate-limited ({r.status_code}); sleeping {wait}s", file=sys.stderr)
                time.sleep(wait); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"GET {endpoint} failed after {retries} retries: {last}")


def fetch_klines(symbol: str, interval: str,
                 start_ms: int, end_ms: int) -> pd.DataFrame:
    """Paginated fetch of perpetual-futures klines."""
    candle_ms = INTERVAL_TO_MIN[interval] * 60_000
    rows: list[list] = []
    cur = start_ms
    while cur < end_ms:
        data = _http_get("/fapi/v1/klines", {
            "symbol": symbol, "interval": interval,
            "startTime": cur, "endTime": end_ms, "limit": 1500,
        })
        if not data:
            break
        rows.extend(data)
        cur = data[-1][0] + candle_ms
        if len(data) < 1500:
            break
        time.sleep(0.20)

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols).drop(columns=["ignore"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return (df.drop_duplicates("open_time")
              .sort_values("open_time")
              .reset_index(drop=True))


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = []
    cur = start_ms
    while cur < end_ms:
        data = _http_get("/fapi/v1/fundingRate", {
            "symbol": symbol,
            "startTime": cur, "endTime": end_ms, "limit": 1000,
        })
        if not data:
            break
        rows.extend(data)
        cur = data[-1]["fundingTime"] + 1
        if len(data) < 1000:
            break
        time.sleep(0.20)
    if not rows:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])
    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return (df[["fundingTime", "fundingRate"]]
            .drop_duplicates("fundingTime")
            .sort_values("fundingTime")
            .reset_index(drop=True))


# =============================================================================
# Indicator primitives  (NO LEAK: all use rolling/ewm on past+present only)
# =============================================================================

def rsi_ewm(series: pd.Series, period: int) -> pd.Series:
    """Wilder/EWM RSI as in btc_features.py."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def rsi_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple-MA RSI as in create_live_dataset_eth.py (kept identical for parity)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.rolling(period).mean()
    avg_l = loss.rolling(period).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([(df["high"] - df["low"]).abs(),
                      (df["high"] - pc).abs(),
                      (df["low"]  - pc).abs()], axis=1).max(axis=1)


def atr_ewm(df: pd.DataFrame, period: int) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def atr_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """SMA-of-TR ATR (matches create_live_dataset_eth.py's ATR_norm14)."""
    return true_range(df).rolling(period).mean()


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm  = pd.Series(np.where((up > dn) & (up > 0),  up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0),  dn, 0.0), index=df.index)
    atr_ = true_range(df).ewm(alpha=1 / period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm( alpha=1 / period, adjust=False).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    line = ef - es
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper, lower = ma + n_std * sd, ma - n_std * sd
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    bw = (upper - lower) / ma.replace(0, np.nan)
    return pct_b, bw


def parkinson_vol(df: pd.DataFrame, period: int) -> pd.Series:
    return np.sqrt((np.log(df["high"] / df["low"]) ** 2)
                   .rolling(period).mean() / (4 * np.log(2)))


def garman_klass_vol(df: pd.DataFrame, period: int) -> pd.Series:
    ln_hl = np.log(df["high"] / df["low"])  ** 2
    ln_co = np.log(df["close"] / df["open"]) ** 2
    return np.sqrt((0.5 * ln_hl - (2 * np.log(2) - 1) * ln_co)
                   .rolling(period).mean())


def rogers_satchell_vol(df: pd.DataFrame, period: int) -> pd.Series:
    ho = np.log(df["high"]  / df["open"])
    hc = np.log(df["high"]  / df["close"])
    lo = np.log(df["low"]   / df["open"])
    lc = np.log(df["low"]   / df["close"])
    return np.sqrt((ho * hc + lo * lc).rolling(period).mean())


# =============================================================================
# ETH-script "100-bar aggregate" helpers (vectorised over a rolling window)
# =============================================================================
#
# These mirror the per-snapshot helpers in create_live_dataset_eth.py
# (compute_atr, compute_boll_bw, compute_chop, compute_hl_range,
# compute_trend_consistency) but produce a full series via rolling apply.
# All use only past+present data → no leak.

def rolling_chop(df: pd.DataFrame, window: int) -> pd.Series:
    """CHOP100 from ETH script: 100·log10(sum_TR / (HHmax-LLmin)) / log10(window)."""
    tr = true_range(df)                       # bar 0 of TR is NaN by definition
    sum_tr = tr.rolling(window).sum()
    hh = df["high"].rolling(window).max()
    ll = df["low"].rolling(window).min()
    denom = (hh - ll).replace(0, np.nan)
    out = 100 * np.log10(sum_tr / denom) / math.log10(window)
    return out


def rolling_hl_range(df: pd.DataFrame, window: int) -> pd.Series:
    return df["high"].rolling(window).max() - df["low"].rolling(window).min()


def rolling_trend_consistency(close: pd.Series, window: int, seg: int = 10) -> pd.Series:
    """std-of-segment-mean-diffs over a rolling `window` of bars."""
    n_segs = window // seg
    if n_segs < 2:
        return pd.Series(np.nan, index=close.index)
    arr = close.to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan)
    for end in range(window, arr.shape[0] + 1):
        win = arr[end - window:end]
        avgs = win.reshape(n_segs, seg).mean(axis=1)
        if avgs.size > 1:
            out[end - 1] = np.std(np.diff(avgs), ddof=1)
    return pd.Series(out, index=close.index)


# =============================================================================
# Feature builder
# =============================================================================

def build_features(k: pd.DataFrame,
                   funding: Optional[pd.DataFrame],
                   candle: str) -> pd.DataFrame:
    """
    Compute the full union of features. All formulas use only data up to t.

    Implementation note: features are accumulated in a dict and concatenated
    once at the end. This avoids the O(K²) cost of column-by-column inserts
    on a wide DataFrame and the associated `PerformanceWarning` chatter.
    """
    candle_min    = INTERVAL_TO_MIN[candle]
    bars_per_hour = 60 / candle_min

    def H(hours: float) -> int:
        """Convert hours → number of candles for the current resolution."""
        return max(1, int(round(hours * bars_per_hour)))

    df = k.copy().set_index("open_time")
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    tbase, trades = df["taker_buy_base"], df["trades"]

    feats: dict[str, pd.Series] = {}

    # -- 1-bar log return (used by many features below; not exported) --
    r1 = np.log(c / c.shift(1))
    abs_r1 = r1.abs()

    # =========================================================================
    # SECTION A — features from btc_features.py  (HOUR-based windows)
    # =========================================================================

    # A.1 multi-horizon log returns
    for hours in [1, 2, 4, 8, 12, 24, 48, 72, 168]:
        feats[f"ret_{hours}h"] = np.log(c / c.shift(H(hours)))

    # A.2 lagged 1-bar returns
    for lag in [1, 2, 3, 4, 6, 12, 24]:
        feats[f"ret_1h_lag{lag}"] = r1.shift(lag)

    # A.3 realized vol (close-to-close)
    for hours in [6, 12, 24, 48, 168]:
        feats[f"realized_vol_{hours}h"] = r1.rolling(H(hours)).std()

    # A.4 OHLC vol estimators
    feats["parkinson_24"] = parkinson_vol(df,    H(24))
    feats["gk_24"]        = garman_klass_vol(df, H(24))
    feats["atr_14"]       = atr_ewm(df, H(14))
    feats["atr_24"]       = atr_ewm(df, H(24))
    feats["atr_pct_14"]   = feats["atr_14"] / c
    feats["tr_pct"]       = true_range(df) / c

    # A.5 momentum / oscillators
    feats["rsi_14h"] = rsi_ewm(c, H(14))
    feats["rsi_24h"] = rsi_ewm(c, H(24))
    m, s, hist = macd(c)
    feats["macd"]        = m    / c
    feats["macd_signal"] = s    / c
    feats["macd_hist"]   = hist / c
    for hours in [6, 12, 24, 48]:
        feats[f"roc_{hours}h"] = c.pct_change(H(hours))

    for hours in [14, 24]:
        n = H(hours)
        hh = h.rolling(n).max()
        ll = l.rolling(n).min()
        rng = (hh - ll).replace(0, np.nan)
        feats[f"willr_{hours}h"]   = -100 * (hh - c) / rng
        feats[f"stoch_k_{hours}h"] =  100 * (c - ll) / rng

    tp = (h + l + c) / 3
    ma_tp = tp.rolling(H(20)).mean()
    md_tp = (tp - ma_tp).abs().rolling(H(20)).mean()
    feats["cci_20h"] = (tp - ma_tp) / (0.015 * md_tp.replace(0, np.nan))

    # A.6 trend / EMA ratios
    for hours in [9, 21, 50, 100, 200]:
        ema = c.ewm(span=H(hours), adjust=False).mean()
        feats[f"ema_ratio_{hours}h"] = c / ema - 1
    # close vs the 5-BAR EMA used as the triple-barrier baseline (--base ema5).
    # Encodes "where is current price relative to the entry reference point"
    # in the same coordinate system as the label. Causal: ewm(adjust=False)
    # uses only data up to and including the current bar.
    ema5_close = c.ewm(span=5, adjust=False).mean()
    feats["close_to_ema5"] = c / ema5_close - 1.0
    feats["adx_14h"] = adx(df, H(14))

    # A.7 distance from recent extremes
    for hours in [24, 72, 168]:
        n = H(hours)
        feats[f"dist_high_{hours}h"] = c / h.rolling(n).max() - 1
        feats[f"dist_low_{hours}h"]  = c / l.rolling(n).min() - 1

    # A.8 Bollinger (20 bars in original; mapped to H(20))
    pctb, bw = bollinger(c, H(20), 2)
    feats["bb_pctb_20h"]  = pctb
    feats["bb_width_20h"] = bw

    # A.9 volume / order flow
    feats["log_volume"] = np.log1p(v)
    for hours in [24, 168]:
        n = H(hours)
        mv = v.rolling(n).mean()
        sv = v.rolling(n).std().replace(0, np.nan)
        feats[f"vol_z_{hours}"] = ((v - mv) / sv).clip(-10, 10)
    feats["vol_ma_ratio_24h"] = v / v.rolling(H(24)).mean()
    feats["taker_buy_ratio"]  = tbase / v.replace(0, np.nan)
    taker_imbalance = (2 * tbase - v) / v.replace(0, np.nan)
    feats["taker_imbalance"]  = taker_imbalance
    for hours in [6, 24]:
        feats[f"taker_imbalance_ma_{hours}h"] = taker_imbalance.rolling(H(hours)).mean()

    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    feats["obv"] = obv
    feats["obv_z_168h"] = (obv - obv.rolling(H(168)).mean()) / \
                          obv.rolling(H(168)).std().replace(0, np.nan)

    mf = tp * v
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    mfr = pos.rolling(H(14)).sum() / neg.rolling(H(14)).sum().replace(0, np.nan)
    feats["mfi_14h"] = 100 - 100 / (1 + mfr)

    for hours in [24, 72]:
        n = H(hours)
        vwap = (tp * v).rolling(n).sum() / v.rolling(n).sum().replace(0, np.nan)
        feats[f"vwap_dev_{hours}h"] = c / vwap - 1

    # A.10 candle / microstructure
    rng_hl = (h - l).replace(0, np.nan)
    feats["body_ratio"]     = (c - o) / rng_hl
    feats["upper_wick"]     = (h - np.maximum(c, o)) / rng_hl
    feats["lower_wick"]     = (np.minimum(c, o) - l) / rng_hl
    feats["gap_open"]       = np.log(o / c.shift(1))
    feats["log_trades"]     = np.log1p(trades)
    feats["avg_trade_size"] = (v / trades.replace(0, np.nan)).clip(0, 1e6)

    # A.11 funding
    if funding is not None and not funding.empty:
        f = funding.set_index("fundingTime").sort_index()
        merged = df.index.union(f.index).sort_values()
        f = f.reindex(merged).ffill().loc[df.index]
        fr = f["fundingRate"]
        feats["funding_rate"]              = fr
        feats["funding_rate_ma_24h"]       = fr.rolling(H(24)).mean()
        feats["funding_rate_ma_72h"]       = fr.rolling(H(72)).mean()
        feats["funding_rate_change"]       = fr.diff()
        feats["funding_cum_24h"]           = fr.rolling(H(24)).sum()
        feats["funding_positive_frac_168h"] = (fr > 0).rolling(H(168)).mean()

    # A.12 calendar
    hour = df.index.hour
    dow  = df.index.dayofweek
    feats["hour_sin"]   = pd.Series(np.sin(2 * np.pi * hour / 24), index=df.index)
    feats["hour_cos"]   = pd.Series(np.cos(2 * np.pi * hour / 24), index=df.index)
    feats["dow_sin"]    = pd.Series(np.sin(2 * np.pi * dow  / 7),  index=df.index)
    feats["dow_cos"]    = pd.Series(np.cos(2 * np.pi * dow  / 7),  index=df.index)
    feats["is_weekend"] = pd.Series((dow >= 5).astype(np.int8),    index=df.index)
    feats["is_friday"]  = pd.Series((dow == 4).astype(np.int8),    index=df.index)

    # =========================================================================
    # SECTION B — features from btc_features_gl1f_vol_trimmed.py
    # =========================================================================

    for hours in [6, 24, 48, 168, 336]:
        feats[f"rv_{hours}h"] = r1.rolling(H(hours)).std()

    for hours in [6, 48, 72, 168]:
        feats[f"park_{hours}h"] = parkinson_vol(df, H(hours))

    for hours in [6, 48, 168]:
        feats[f"gk_{hours}h"] = garman_klass_vol(df, H(hours))

    for hours in [12, 48, 168]:
        feats[f"rs_{hours}h"] = rogers_satchell_vol(df, H(hours))

    feats["atr_14_pct"] = atr_ewm(df, H(14)) / c
    feats["atr_48_pct"] = atr_ewm(df, H(48)) / c

    rv24  = r1.rolling(H(24)).std()
    rv168 = r1.rolling(H(168)).std()
    rv504 = r1.rolling(H(504)).std()
    feats["rv_short_vs_long_504"] = rv24  / rv504
    feats["rv_medium_vs_long"]    = rv168 / rv504
    feats["rv24_rank_168"] = rv24.rolling(H(168)).apply(
        lambda w: (w[-1] > w[:-1]).mean() if len(w) > 1 else np.nan, raw=True)
    feats["rv24_rank_504"] = rv24.rolling(H(504)).apply(
        lambda w: (w[-1] > w[:-1]).mean() if len(w) > 1 else np.nan, raw=True)
    feats["vol_of_vol_168"] = rv24.rolling(H(168)).std()
    feats["vol_of_vol_504"] = rv24.rolling(H(504)).std()
    for hours in [12, 24, 72]:
        feats[f"rv_log_change_{hours}h"] = np.log(rv24 / rv24.shift(H(hours)))

    sigma_168 = r1.rolling(H(168)).std()
    feats["jumps_2sigma_168h"] = (
        (abs_r1 > 2 * sigma_168).rolling(H(168)).sum()) / H(168)
    for hours in [24, 168]:
        n = H(hours)
        feats[f"max_abs_ret_scaled_{hours}h"] = (
            abs_r1.rolling(n).max() / r1.rolling(n).std().replace(0, np.nan))
    for hours in [24, 72]:
        n = H(hours)
        bv  = (np.pi / 2) * (abs_r1 * abs_r1.shift(1)).rolling(n).sum() / n
        rv2 = (r1 ** 2).rolling(n).sum() / n
        feats[f"jump_ratio_{hours}h"] = ((rv2 - bv) / rv2.replace(0, np.nan)).clip(-1, 1)

    for hours in [24, 72, 168]:
        n = H(hours)
        neg_var = (r1.where(r1 < 0, 0) ** 2).rolling(n).sum() / n
        pos_var = (r1.where(r1 > 0, 0) ** 2).rolling(n).sum() / n
        if hours in (24, 72):
            feats[f"neg_var_{hours}h"] = np.sqrt(neg_var)
        if hours in (24, 168):
            feats[f"pos_var_{hours}h"] = np.sqrt(pos_var)
        if hours in (72, 168):
            feats[f"semivol_ratio_{hours}h"] = np.sqrt(
                neg_var / pos_var.replace(0, np.nan)).clip(0, 10)

    hl_log = np.log(h / l)
    feats["hl_range_mean_24h"]  = hl_log.rolling(H(24)).mean()
    feats["hl_range_mean_168h"] = hl_log.rolling(H(168)).mean()
    feats["hl_range_max_72h"]   = hl_log.rolling(H(72)).max()
    feats["hl_range_max_168h"]  = hl_log.rolling(H(168)).max()

    feats["taker_imbalance_abs_ma_24h"] = taker_imbalance.abs().rolling(H(24)).mean()

    # =========================================================================
    # SECTION C — features from create_live_dataset_eth.py (BAR-based)
    # All windows are kept in BARS to match the original ETH model exactly.
    # =========================================================================

    feats["r1_log"]  = np.log(c / c.shift(1))
    feats["r12_log"] = np.log(c / c.shift(12))

    feats["RSI14"]      = rsi_sma(c, 14)
    feats["ATR_norm14"] = atr_sma(df, 14) / c

    feats["body_range_ratio"] = pd.Series(
        np.where(h != l, (c - o).abs() / (h - l), 0.0),
        index=df.index)

    BAR100 = 100
    atr100        = atr_sma(df, BAR100)
    hl_range100   = rolling_hl_range(df, BAR100)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    sma50 = c.rolling(50).mean()
    std50 = c.rolling(50).std()
    bollbw20 = ((sma20 + 2 * std20) - (sma20 - 2 * std20)) / sma20.replace(0, np.nan) * 100.0
    bollbw50 = ((sma50 + 2 * std50) - (sma50 - 2 * std50)) / sma50.replace(0, np.nan) * 100.0
    chop100      = rolling_chop(df, BAR100)
    trendcons100 = rolling_trend_consistency(c, BAR100, seg=10)

    feats["BollBW50"]        = bollbw50
    feats["ATR_ratio100"]    = atr100 / c.replace(0, np.nan)
    feats["BW_CHOP100"]      = bollbw20 / chop100.replace(0, np.nan)
    feats["ATR_HL_ratio100"] = atr100 / hl_range100.replace(0, np.nan)
    feats["TrendConsist100"] = trendcons100

    feats["return_5_3"]   = c.pct_change(3)
    feats["return_60_3"]  = c.pct_change(36)
    feats["volatility_5"] = c.pct_change().rolling(12).std()
    feats["mom_5"]        = c.pct_change().rolling(5).sum()

    feats["hour"] = pd.Series(df.index.hour.astype(np.int16),       index=df.index)
    feats["dow"]  = pd.Series(df.index.dayofweek.astype(np.int16),  index=df.index)

    # =========================================================================
    # SECTION D — label-aligned & microstructure additions
    # All causal: every value at row N uses only data ≤ row N.
    # =========================================================================

    # --- D.1 distance to barriers in volatility units ---------------------
    # `close_to_ema5` already encodes signed distance to baseline. These
    # express the SAME distance scaled by current ATR — i.e. "how many
    # ATRs is price above/below the entry baseline?". Strongly label-aligned.
    atr14 = atr_sma(df, 14)                        # bar-level ATR(14)
    ema5_close_loc = c.ewm(span=5, adjust=False).mean()
    dist_to_ema5 = c - ema5_close_loc
    feats["dist_ema5_atr"]   = dist_to_ema5 / atr14.replace(0, np.nan)
    # vol-of-vol position: where does current short-vol sit vs medium-vol?
    feats["rv6_over_rv24"]   = r1.rolling(H(6)).std() / r1.rolling(H(24)).std()

    # --- D.2 wick / candle directionality (3-bar smoothed) ----------------
    rng_hl_loc   = (h - l).replace(0, np.nan)
    body_signed  = (c - o) / rng_hl_loc            # signed body fraction in [-1,1]
    upper_w      = (h - np.maximum(c, o)) / rng_hl_loc
    lower_w      = (np.minimum(c, o) - l) / rng_hl_loc
    feats["body_signed_3"]    = body_signed.rolling(3).mean()
    # Net rejection: positive = lower wicks dominate (buyers absorbing dips)
    feats["wick_imbalance_3"] = (lower_w - upper_w).rolling(3).mean()

    # --- D.3 short-term volume / flow surges -----------------------------
    # Captures "is current bar volume exploding vs recent mean" — proxy for
    # institutional entry / liquidation cascades.
    feats["volume_surge_3"]   = v / v.rolling(H(6)).mean().replace(0, np.nan)
    # Buy-pressure persistence: did the last 6 bars net-buy or net-sell?
    # taker_imbalance is (2*tbase - v)/v ∈ [-1,1]; rolling sum gives momentum
    feats["taker_imb_sum_6"]  = taker_imbalance.rolling(H(6)).sum()

    # --- D.4 directional momentum vs noise -------------------------------
    # Skewness of last 24 bar-returns: positive = up-tail, negative = down-tail.
    # Distinguishes "trending up" from "choppy with up-bias".
    feats["ret_skew_24h"]     = r1.rolling(H(24)).skew()
    # Sign-persistence of last 12 bar-returns ∈ [-1, +1]:
    # +1 = all up, −1 = all down, 0 = balanced. Different from mom_5 because
    # it ignores magnitude and captures consistency.
    feats["sign_persist_12"]  = np.sign(r1).rolling(12).mean()

    # --- D.5 microstructure efficiency -----------------------------------
    # Fraction of bar high-low range traversed by the close from open.
    # Near 1 = clean directional bar; near 0 = doji / whippy bar.
    # Smoothed over 6 bars to denoise.
    range_eff = (c - o).abs() / rng_hl_loc
    feats["range_efficiency_6"] = range_eff.rolling(6).mean()

    # --- D.6 trend strength via R² of recent close ----------------------
    # R² of close vs time over last 12 bars. High = clean trend (up or down),
    # low = chop. Combined with sign of slope it tells direction; alone, it
    # captures "is now a momentum regime regardless of direction".
    def _trend_r2(window):
        if window.isna().any():
            return np.nan
        n = len(window)
        x = np.arange(n)
        # variance of x is constant; only need correlation²
        sx = x.std()
        sy = window.std()
        if sx == 0 or sy == 0:
            return 0.0
        return float(((x - x.mean()) * (window.values - window.mean())).mean()
                     / (sx * sy)) ** 2
    feats["trend_r2_12"] = c.rolling(12).apply(_trend_r2, raw=False)

    # --- D.7 distance to recent extremes in vol units --------------------
    # close vs last-24h max/min, normalized by atr14. Catches breakouts.
    feats["dist_high24_atr"] = (c - h.rolling(H(24)).max()) / atr14.replace(0, np.nan)
    feats["dist_low24_atr"]  = (c - l.rolling(H(24)).min()) / atr14.replace(0, np.nan)

    # =========================================================================
    # Single concat at the end → contiguous frame, no fragmentation
    # Returns ONLY derived features (raw OHLCV is intentionally excluded).
    # =========================================================================
    feature_frame = pd.DataFrame(feats, index=df.index)
    return feature_frame


# =============================================================================
# Triple-barrier label generator (vectorised, no leak)
# =============================================================================

@dataclass
class LabelSpec:
    direction:  str    # "up" or "down"
    base_period: int   # EMA span in BARS
    move_pct:   float  # e.g. 1.5
    retrace_pct: float # e.g. 0.5
    horizon_bars: int  # number of forward bars to scan
    # raw user strings for filename
    base_str:    str
    move_str:    str
    retrace_str: str
    horizon_str: str


def compute_triple_barrier_labels(klines: pd.DataFrame, spec: LabelSpec
                                  ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (label, baseline, valid_mask) aligned to klines.index.

    `klines` must have columns ``high``, ``low``, ``close`` and be indexed by
    open_time. Pass the raw kline frame here — features are NOT consulted.

    label[N] examines bars (N+1 .. N+horizon_bars) only.
    baseline[N] = ema{P}(close)[N]   (uses only data up to and including N)
    valid_mask[N] = True iff there are `horizon_bars` future bars available.

    Same-bar conflict rule: if both target and stop touch on the same future bar,
    the stop is considered hit first → label=0 (conservative for long trades).
    """
    H = klines["high"].to_numpy(dtype=float)
    L = klines["low"].to_numpy(dtype=float)
    N = H.shape[0]
    horizon = spec.horizon_bars

    # ---- baseline: EMA of close, computed using ONLY past+present (ewm is causal) ----
    baseline = klines["close"].ewm(span=spec.base_period, adjust=False).mean()
    base_arr = baseline.to_numpy(dtype=float)

    # ---- target / stop levels per row ----
    if spec.direction == "up":
        target = base_arr * (1.0 + spec.move_pct    / 100.0)   # break above
        stop   = base_arr * (1.0 - spec.retrace_pct / 100.0)   # break below
    elif spec.direction == "down":
        target = base_arr * (1.0 - spec.move_pct    / 100.0)   # break below
        stop   = base_arr * (1.0 + spec.retrace_pct / 100.0)   # break above
    else:
        raise ValueError(f"unknown direction {spec.direction!r}")

    # ---- build (N, horizon) future high/low matrices ----
    # future_high[i, k] = high[i + 1 + k]   for k = 0..horizon-1
    fut_h = np.full((N, horizon), np.inf,  dtype=float)   # inf so missing never trips a target
    fut_l = np.full((N, horizon), -np.inf, dtype=float)   # -inf so missing never trips a stop
    valid = np.zeros(N, dtype=bool)
    for k in range(horizon):
        src_start = k + 1
        if src_start >= N:
            break
        n_copy = N - src_start
        fut_h[:n_copy, k] = H[src_start:src_start + n_copy]
        fut_l[:n_copy, k] = L[src_start:src_start + n_copy]
    # row i is valid iff i + horizon < N, i.e. all `horizon` future bars exist
    valid[: max(0, N - horizon)] = True

    # ---- find first hit indices (vectorised) ----
    if spec.direction == "up":
        target_hit = fut_h >= target[:, None]
        stop_hit   = fut_l <= stop[:,   None]
    else:
        target_hit = fut_l <= target[:, None]
        stop_hit   = fut_h >= stop[:,   None]

    # argmax of bool returns 0 when no True; mask with .any()
    th_any = target_hit.any(axis=1)
    sh_any = stop_hit.any(axis=1)
    # fill with `horizon` sentinel when no hit (so comparisons treat as "never")
    th_first = np.where(th_any, target_hit.argmax(axis=1), horizon)
    sh_first = np.where(sh_any, stop_hit.argmax(axis=1),   horizon)

    # label: 1 iff target strictly precedes stop AND target was actually hit
    label = ((th_first < sh_first) & (th_first < horizon)).astype(np.int8)

    # rows without a full forward window: invalidate the label
    label_series = pd.Series(label, index=klines.index, name="label")
    label_series[~valid] = -1   # sentinel; will be dropped before save
    valid_series = pd.Series(valid, index=klines.index, name="valid")
    base_series  = pd.Series(base_arr, index=klines.index, name="baseline")

    return label_series, base_series, valid_series


# =============================================================================
# Output assembly
# =============================================================================

def output_filename(ticker: str, candle: str, spec: LabelSpec, start_str: str) -> str:
    return (f"{ticker.upper()}_{candle}_{spec.direction}_{spec.base_str}"
            f"_move{spec.move_str}_retrace{spec.retrace_str}"
            f"_horizon{spec.horizon_str}_start_{start_str}.csv")


def write_dataset(features: pd.DataFrame,
                  klines: pd.DataFrame,
                  spec: LabelSpec,
                  start_dt: datetime,
                  out_path: Path) -> dict:
    """
    Trim to [start_dt, end), drop warmup-NaN rows and rows without a full
    forward window, then save.

    The output CSV contains ONLY the derived feature columns + a single
    ``label`` column (and ``open_time`` as the index). Raw OHLCV and the
    EMA baseline are intentionally excluded — those are construction-time
    artefacts, not features.

    `features` must contain only the derived feature columns (no raw klines).
    `klines` must contain ``high``, ``low``, ``close`` indexed by open_time;
    its index must equal ``features.index``.

    Returns a stats dict.
    """
    # ---- alignment guard ----
    if not features.index.equals(klines.index):
        raise ValueError("features.index and klines.index must be identical")
    # ---- guard: no raw OHLCV columns must have leaked into features ----
    forbidden = {"open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "taker_buy_base", "taker_buy_quote",
                 "open_time", "baseline", "label"}
    intruders = sorted(set(features.columns) & forbidden)
    if intruders:
        raise ValueError(
            f"build_features must not return raw kline / label / baseline "
            f"columns; got: {intruders}")

    label, _baseline, _valid = compute_triple_barrier_labels(klines, spec)

    df = features.copy()
    df["label"] = label.astype(np.int8)

    # 1) drop rows before user-requested start_dt (warmup region)
    df = df[df.index >= start_dt].copy()

    n_before_warmup = len(df)

    # 2) drop rows where any feature is NaN (warmup tail extending into start window)
    feat_cols = [c for c in df.columns if c != "label"]
    nan_mask = df[feat_cols].isna().any(axis=1)
    df = df[~nan_mask].copy()
    n_after_nan = len(df)

    # 3) drop rows that have no full forward window (label == -1)
    df = df[df["label"] != -1].copy()
    n_after_label = len(df)

    # ---- save: ONLY features + label, with open_time as index ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index_label="open_time", float_format="%.8g")

    # ---- final post-write verification: read back column set ----
    written_cols = pd.read_csv(out_path, nrows=0).columns.tolist()
    bad = [c for c in written_cols if c in forbidden and c not in ("open_time", "label")]
    if bad:
        raise RuntimeError(f"output CSV has forbidden columns: {bad}")
    if "label" not in written_cols:
        raise RuntimeError("output CSV missing 'label' column")
    if "open_time" not in written_cols:
        raise RuntimeError("output CSV missing 'open_time' index column")

    # ---- stats ----
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    total = pos + neg
    return {
        "file":         str(out_path),
        "rows":         total,
        "n_features":   len(feat_cols),
        "pos":          pos,
        "neg":          neg,
        "pos_pct":      (pos / total * 100.0) if total else 0.0,
        "neg_pct":      (neg / total * 100.0) if total else 0.0,
        "dropped_nan":  n_before_warmup - n_after_nan,
        "dropped_no_future": n_after_nan - n_after_label,
    }


# =============================================================================
# Driver
# =============================================================================

def hours_to_warmup_bars(candle: str) -> int:
    bars_per_hour = 60 / INTERVAL_TO_MIN[candle]
    return max(int(round(MAX_LOOKBACK_HOURS * bars_per_hour)) + WARMUP_PAD_BARS,
               MAX_LOOKBACK_BARS + WARMUP_PAD_BARS)


def main() -> None:
    args = parse_args()

    tickers      = _split_csv(args.ticker)
    candles      = _split_csv(args.candle)
    classes      = _split_csv(args.cls)
    bases        = _split_csv(args.base)
    move_pcts    = _split_csv(args.move_pct)
    retraces     = _split_csv(args.no_retrace)
    horizons     = _split_csv(args.horizon)

    for cls in classes:
        if cls not in ("up", "down"):
            sys.exit(f"--class must be 'up' or 'down' (got {cls!r})")
    for cd in candles:
        if cd not in INTERVAL_TO_MIN:
            sys.exit(f"--candle {cd!r} not supported. "
                     f"Choose from: {', '.join(INTERVAL_TO_MIN)}")

    start_dt = parse_date(args.start_date)
    end_dt   = parse_date(args.end_date) if args.end_date else \
               datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    if end_dt <= start_dt:
        sys.exit("--end-date must be after --start-date")

    out_root = Path(args.output_dir)
    all_stats: list[dict] = []

    # Outer loop: per (ticker, candle) we fetch + build features ONCE, then iterate
    # all label combinations on that single feature frame.
    for ticker, candle in product(tickers, candles):
        symbol = ticker.upper() + "USDT"

        warmup_bars = hours_to_warmup_bars(candle)
        candle_min  = INTERVAL_TO_MIN[candle]
        warmup_dt   = start_dt - timedelta(minutes=warmup_bars * candle_min)

        print()
        print("=" * 78)
        print(f"  {symbol}  candle={candle}")
        print(f"  fetch range  : {warmup_dt.isoformat()}  →  {end_dt.isoformat()}")
        print(f"  warmup bars  : {warmup_bars}  ({warmup_bars * candle_min / 60:.1f} h)")
        print("=" * 78)

        start_ms = int(warmup_dt.timestamp() * 1000)
        end_ms   = int(end_dt.timestamp() * 1000)

        print("  fetching klines ...")
        k = fetch_klines(symbol, candle, start_ms, end_ms)
        print(f"    {len(k):,} candles")

        funding = None
        if not args.no_funding:
            print("  fetching funding rate ...")
            funding = fetch_funding(symbol, start_ms - 8 * 3_600_000, end_ms)
            print(f"    {len(funding):,} funding events")

        print("  building features ...")
        feats = build_features(k, funding, candle)
        print(f"    feature frame  : {feats.shape[0]:,} rows × {feats.shape[1]} cols")

        if args.small:
            missing = [c for c in SMALL_FEATURE_LIST if c not in feats.columns]
            if missing:
                if args.no_funding and all(m.startswith("funding_") for m in missing):
                    sys.exit(f"--small needs funding features but --no-funding was set "
                             f"(missing: {missing})")
                sys.exit(f"--small: feature(s) not produced by build_features: {missing}")
            feats = feats[list(SMALL_FEATURE_LIST)].copy()
            print(f"    --small filter : {feats.shape[1]} features kept "
                  f"({', '.join(SMALL_FEATURE_LIST[:3])} … {SMALL_FEATURE_LIST[-1]})")

        # klines view used by labeling only (high/low/close indexed by open_time).
        # Kept separate so it never enters the feature matrix.
        k_indexed = k.set_index("open_time")[["high", "low", "close"]]
        if not k_indexed.index.equals(feats.index):
            sys.exit("internal error: klines and features index disagree")

        # Pre-validate / normalize horizons.
        # Accepted forms:
        #   "4h", "30m", "1d"   → duration → must be integer multiple of candle
        #   "43"                → bare integer = number of bars (always valid)
        horizon_bars = {}
        for h_str in horizons:
            s = h_str.strip()
            if s.isdigit():
                horizon_bars[h_str] = int(s)
            else:
                h_min = parse_duration_minutes(s)
                if h_min % candle_min:
                    sys.exit(f"horizon {h_str!r} is not an integer multiple "
                             f"of candle {candle!r}")
                horizon_bars[h_str] = h_min // candle_min

        # Inner loop: every combination of label specs on the same features
        for cls, base, mv, rt, h_str in product(classes, bases, move_pcts, retraces, horizons):
            spec = LabelSpec(
                direction    = cls,
                base_period  = parse_ema_period(base),
                move_pct     = float(mv),
                retrace_pct  = float(rt),
                horizon_bars = horizon_bars[h_str],
                base_str     = base,
                move_str     = mv,
                retrace_str  = rt,
                horizon_str  = h_str,
            )
            fname = output_filename(ticker, candle, spec, args.start_date)
            out_path = out_root / fname

            print(f"  → {fname}")
            stats = write_dataset(feats, k_indexed, spec, start_dt, out_path)
            all_stats.append({**stats,
                              "ticker": ticker.upper(),
                              "candle": candle,
                              "class":  cls,
                              "base":   base,
                              "move":   mv,
                              "retrace": rt,
                              "horizon": h_str})

    # ---- final summary table ----
    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    header = f"{'file':<60s} {'rows':>8s} {'pos%':>6s} {'neg%':>6s} {'feats':>6s}"
    print(header)
    print("-" * len(header))
    for s in all_stats:
        name = Path(s["file"]).name
        if len(name) > 58:
            name = name[:55] + "..."
        print(f"{name:<60s} {s['rows']:>8d} "
              f"{s['pos_pct']:>5.1f}% {s['neg_pct']:>5.1f}% "
              f"{s['n_features']:>6d}")

    print()
    print("  Detailed per-file stats:")
    for s in all_stats:
        print(f"    {Path(s['file']).name}")
        print(f"      rows kept              : {s['rows']:,}")
        print(f"      class 1 (event)        : {s['pos']:,}  ({s['pos_pct']:.2f}%)")
        print(f"      class 0 (no event)     : {s['neg']:,}  ({s['neg_pct']:.2f}%)")
        print(f"      features               : {s['n_features']}  (+ label)")
        print(f"      dropped (warmup NaN)   : {s['dropped_nan']:,}")
        print(f"      dropped (no future win): {s['dropped_no_future']:,}")

    print()
    print(f"  {len(all_stats)} dataset(s) written.")


if __name__ == "__main__":
    main()

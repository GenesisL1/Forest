#!/usr/bin/env python3
"""
zecup_predictor_daemon.py
=========================

Runs every 15-minute candle close (+ 3 second buffer for Binance to settle the
candle and propagate the funding rate) and:

  1. Fetches the most recent closed 15m ZECUSDT candle + ~250 bars of context
  2. Computes the same 33 causal features the dApp computes
  3. Quantises to int32 at scaleQ, packs little-endian
  4. Calls ForestRuntime.predictView(modelId, packed) on GenesisL1 (tokenId 10)
  5. Computes probability = sigmoid(rawLogit / scaleQ)
  6. Appends a row to zecup_predictions.csv with the candle OHLC + score

Designed to be run as a long-lived process under systemd / supervisor / tmux.

Dependencies:
    pip install requests web3 eth-abi

Run:
    python3 zecup_predictor_daemon.py

The CSV grows append-only; downstream consumers (web page, analytics) can
re-read it cheaply. To rotate, move the file aside and the daemon will create
a new one with header on next bar.
"""

import csv
import json
import math
import os
import signal
import struct
import sys
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from web3 import Web3

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# All paths default to the directory containing this script — perfect for
# dropping into an existing /html/forest directory and running it there.
# Override any of these via environment variables if you want.
SCRIPT_DIR = Path(__file__).resolve().parent

CSV_PATH = Path(os.environ.get(
    'ZECUP_CSV', str(SCRIPT_DIR / 'zecup_predictions.csv')
))

# Single HTML file — read, splice, atomic-rewrite, in place. No separate
# template + output. The sentinel comments <!-- HISTORY:BEGIN ... HISTORY:END -->
# inside the file are how we find the section to replace each tick.
HTML_PATH = Path(os.environ.get(
    'ZECUP_HTML', str(SCRIPT_DIR / 'zecup_5h_scrying.html')
))

# Maximum rows to render in the history table.
HISTORY_MAX_ROWS = int(os.environ.get('ZECUP_HISTORY_ROWS', '500'))

# GenesisL1
RPC_URL  = 'https://rpc.genesisl1.org'
CHAIN_ID = 29
REGISTRY = '0x33c9844F77a07e36B98f0FFf8201B8A8b02c2a69'
RUNTIME  = '0xD2fD0cf461a6cb56Fc08d9aEc120833D8E79044E'
TOKEN_ID = 10

# Binance
BINANCE_FAPI = 'https://fapi.binance.com'
SYMBOL       = 'ZECUSDT'
INTERVAL     = '15m'

# Timing
INTERVAL_SECONDS = 15 * 60
BUFFER_SECONDS   = 3        # Wait this long after candle close before fetching

# Feature ordering — extracted from .gl1f GL1X footer. DO NOT REORDER.
# Both ZEC models use the full 33-feature small set + open_time.
FEATURE_ORDER = [
    'open_time', 'r1_log', 'r12_log', 'RSI14', 'ATR_norm14',
    'body_range_ratio', 'ATR_ratio100', 'BollBW50', 'BW_CHOP100',
    'ATR_HL_ratio100', 'TrendConsist100',
    'return_5_3', 'return_60_3', 'volatility_5', 'mom_5',
    'hour', 'dow',
    'funding_rate', 'funding_rate_ma_24h', 'funding_cum_24h',
    'close_to_ema5', 'dist_ema5_atr', 'rv6_over_rv24',
    'body_signed_3', 'wick_imbalance_3',
    'volume_surge_3', 'taker_imb_sum_6',
    'ret_skew_24h', 'sign_persist_12',
    'range_efficiency_6', 'trend_r2_12',
    'dist_high24_atr', 'dist_low24_atr'
]

ABI_REGISTRY = json.loads('''[
  {"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],
   "name":"getModelSummary",
   "outputs":[
     {"internalType":"bool","name":"exists","type":"bool"},
     {"internalType":"bytes32","name":"modelId","type":"bytes32"},
     {"internalType":"address","name":"tablePtr","type":"address"},
     {"internalType":"uint16","name":"nFeatures","type":"uint16"},
     {"internalType":"uint16","name":"nTrees","type":"uint16"},
     {"internalType":"uint16","name":"depth","type":"uint16"},
     {"internalType":"int32","name":"baseQ","type":"int32"},
     {"internalType":"uint8","name":"pricingMode","type":"uint8"},
     {"internalType":"uint256","name":"feeWei","type":"uint256"},
     {"internalType":"address","name":"feeRecipient","type":"address"},
     {"internalType":"bool","name":"inferenceEnabled","type":"bool"},
     {"internalType":"address","name":"creator","type":"address"},
     {"internalType":"uint32","name":"tosVersionAccepted","type":"uint32"},
     {"internalType":"string","name":"title","type":"string"},
     {"internalType":"string","name":"description","type":"string"}
   ],"stateMutability":"view","type":"function"},
  {"inputs":[{"internalType":"bytes32","name":"modelId","type":"bytes32"}],
   "name":"getModelRuntime",
   "outputs":[
     {"internalType":"address","name":"tablePtr","type":"address"},
     {"internalType":"uint32","name":"chunkSize","type":"uint32"},
     {"internalType":"uint32","name":"numChunks","type":"uint32"},
     {"internalType":"uint32","name":"totalBytes","type":"uint32"},
     {"internalType":"uint16","name":"nFeatures","type":"uint16"},
     {"internalType":"uint16","name":"nTrees","type":"uint16"},
     {"internalType":"uint16","name":"depth","type":"uint16"},
     {"internalType":"int32","name":"baseQ","type":"int32"},
     {"internalType":"uint32","name":"scaleQ","type":"uint32"},
     {"internalType":"bool","name":"inferenceEnabled","type":"bool"},
     {"internalType":"uint8","name":"pricingMode","type":"uint8"},
     {"internalType":"uint256","name":"feeWei","type":"uint256"},
     {"internalType":"address","name":"feeRecipient","type":"address"}
   ],"stateMutability":"view","type":"function"}
]''')

ABI_RUNTIME = json.loads('''[
  {"inputs":[
     {"internalType":"bytes32","name":"modelId","type":"bytes32"},
     {"internalType":"bytes","name":"packedFeaturesQ","type":"bytes"}
   ],"name":"predictView",
   "outputs":[{"internalType":"int256","name":"","type":"int256"}],
   "stateMutability":"view","type":"function"}
]''')


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log(msg, level='INFO'):
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts} UTC] [{level}] {msg}', flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# BINANCE
# ─────────────────────────────────────────────────────────────────────────────

def fetch_klines(limit=251, retries=8):
    """Binance returns the in-flight candle as the last element. Caller drops it."""
    url = f'{BINANCE_FAPI}/fapi/v1/klines?symbol={SYMBOL}&interval={INTERVAL}&limit={limit}'
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                raise RuntimeError('empty klines response')
            return [{
                't':   k[0],
                'o':   float(k[1]),
                'h':   float(k[2]),
                'l':   float(k[3]),
                'c':   float(k[4]),
                'v':   float(k[5]),
                'tr':  int(k[8]),
                'tbb': float(k[9])
            } for k in data]
        except Exception as e:
            last_err = e
            sleep = 0.25 * (1.6 ** (attempt - 1))
            log(f'klines retry {attempt}/{retries} after error: {e} (sleeping {sleep:.2f}s)', 'WARN')
            time.sleep(sleep)
    raise RuntimeError(f'klines failed after {retries} attempts: {last_err}')


def fetch_funding(retries=8):
    url = f'{BINANCE_FAPI}/fapi/v1/fundingRate?symbol={SYMBOL}&limit=50'
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                raise RuntimeError('bad funding response')
            return [{'t': f['fundingTime'], 'r': float(f['fundingRate'])} for f in data]
        except Exception as e:
            last_err = e
            sleep = 0.25 * (1.6 ** (attempt - 1))
            log(f'funding retry {attempt}/{retries} after error: {e}', 'WARN')
            time.sleep(sleep)
    raise RuntimeError(f'funding failed after {retries} attempts: {last_err}')


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING-WINDOW HELPERS (mirror create.py / dApp JS exactly)
# ─────────────────────────────────────────────────────────────────────────────

def roll_mean(arr, i, n):
    if i + 1 < n: return float('nan')
    return sum(arr[i - n + 1: i + 1]) / n

def roll_std(arr, i, n):
    if i + 1 < n: return float('nan')
    window = arr[i - n + 1: i + 1]
    m = sum(window) / n
    return math.sqrt(sum((x - m) ** 2 for x in window) / (n - 1))

def roll_sum(arr, i, n):
    if i + 1 < n: return float('nan')
    return sum(arr[i - n + 1: i + 1])

def roll_min(arr, i, n):
    if i + 1 < n: return float('nan')
    return min(arr[i - n + 1: i + 1])

def roll_max(arr, i, n):
    if i + 1 < n: return float('nan')
    return max(arr[i - n + 1: i + 1])

def true_range(klines):
    n = len(klines)
    tr = [float('nan')] * n
    for j in range(1, n):
        pc = klines[j-1]['c']
        tr[j] = max(abs(klines[j]['h'] - klines[j]['l']),
                    abs(klines[j]['h'] - pc),
                    abs(klines[j]['l'] - pc))
    return tr

def atr_sma(tr, i, n):
    return roll_mean(tr, i, n)

def rsi_sma(closes, i, n):
    if i + 1 < n + 1: return float('nan')
    avg_g = avg_l = 0.0
    for k in range(i - n + 1, i + 1):
        d = closes[k] - closes[k-1]
        if d > 0: avg_g += d
        else:     avg_l += -d
    avg_g /= n; avg_l /= n
    if avg_l == 0: return 100.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)

def chop(klines, tr, i, w):
    sum_tr = roll_sum(tr, i, w)
    hh = roll_max([k['h'] for k in klines], i, w)
    ll = roll_min([k['l'] for k in klines], i, w)
    denom = hh - ll
    if not (denom > 0) or not (sum_tr > 0): return float('nan')
    return 100.0 * math.log10(sum_tr / denom) / math.log10(w)

def boll_bw(closes, i, n):
    m = roll_mean(closes, i, n)
    s = roll_std(closes, i, n)
    if not (m > 0): return float('nan')
    return 4.0 * s / m * 100.0

def trend_consist(closes, i, w, seg=10):
    n_segs = w // seg
    if n_segs < 2 or i + 1 < w: return float('nan')
    start = i - w + 1
    seg_means = [sum(closes[start + s*seg : start + (s+1)*seg]) / seg for s in range(n_segs)]
    diffs = [seg_means[s+1] - seg_means[s] for s in range(n_segs - 1)]
    if len(diffs) < 2: return 0.0
    m = sum(diffs) / len(diffs)
    return math.sqrt(sum((d - m)**2 for d in diffs) / (len(diffs) - 1))

def skewness(arr, i, n):
    """Fisher-Pearson bias-corrected skew (matches pandas Series.skew)."""
    if i + 1 < n or n < 3: return float('nan')
    window = arr[i - n + 1: i + 1]
    mean = sum(window) / n
    m2 = sum((x - mean)**2 for x in window) / n
    m3 = sum((x - mean)**3 for x in window) / n
    if m2 <= 0: return 0.0
    g1 = m3 / (m2 ** 1.5)
    return math.sqrt(n * (n - 1)) / (n - 2) * g1

def sign_persist_mean(arr, i, n):
    if i + 1 < n: return float('nan')
    s = 0
    for k in range(i - n + 1, i + 1):
        if arr[k] > 0: s += 1
        elif arr[k] < 0: s -= 1
    return s / n

def range_eff_mean(klines, i, n):
    if i + 1 < n: return float('nan')
    s = 0.0
    for k in range(i - n + 1, i + 1):
        rng = klines[k]['h'] - klines[k]['l']
        s += abs(klines[k]['c'] - klines[k]['o']) / rng if rng > 0 else 0.0
    return s / n

def trend_r2(closes, i, n):
    if i + 1 < n: return float('nan')
    start = i - n + 1
    mx = (n - 1) / 2
    my = sum(closes[start: start + n]) / n
    sxx = syy = sxy = 0.0
    for k in range(n):
        dx = k - mx
        dy = closes[start + k] - my
        sxx += dx*dx; syy += dy*dy; sxy += dx*dy
    if sxx <= 0 or syy <= 0: return 0.0
    r = sxy * (n - 1) / (n * math.sqrt(sxx * syy))
    return r * r

def ema_series(arr, span):
    alpha = 2 / (span + 1)
    out = [arr[0]]
    for j in range(1, len(arr)):
        out.append(alpha * arr[j] + (1 - alpha) * out[-1])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build_features(klines, funding, i=None):
    """Mirrors the dApp's buildFeatures() and create.py's small-feature set.

    `i` is the target candle index within `klines` (defaults to the last one).
    Passing an explicit `i` lets us compute features for any historical candle
    in the fetched window — that's how we back-fill candles that were missed
    during transient outages. EMA / ATR / etc. only need values 0..i so they
    are stable for any choice of i, given enough lookback (≥ 105 bars).
    """
    n = len(klines)
    if i is None:
        i = n - 1
    if i < 0 or i >= n:
        raise RuntimeError(f'build_features: i={i} out of range (n={n})')
    if i < 104:
        raise RuntimeError(f'build_features: i={i} has too little lookback (need ≥ 104)')
    c = klines[i]['c']
    closes = [k['c'] for k in klines]
    highs  = [k['h'] for k in klines]
    lows   = [k['l'] for k in klines]
    vols   = [k['v'] for k in klines]

    # log returns (used for r1_log, r12_log, ret_skew_24h, rv6_over_rv24, sign_persist_12)
    r1 = [0.0] * n
    for j in range(1, n):
        r1[j] = math.log(klines[j]['c'] / klines[j-1]['c'])

    # simple percentage returns (used for volatility_5, mom_5 — match pandas pct_change)
    r1pct = [0.0] * n
    for j in range(1, n):
        r1pct[j] = klines[j]['c'] / klines[j-1]['c'] - 1.0

    tr = true_range(klines)
    atr14 = atr_sma(tr, i, 14)

    feats = {}

    feats['r1_log']  = math.log(klines[i]['c'] / klines[i-1]['c'])
    feats['r12_log'] = math.log(klines[i]['c'] / klines[i-12]['c'])
    feats['RSI14']   = rsi_sma(closes, i, 14)
    feats['ATR_norm14'] = atr14 / c
    rng = klines[i]['h'] - klines[i]['l']
    feats['body_range_ratio'] = abs(klines[i]['c'] - klines[i]['o']) / rng if rng > 0 else 0.0

    atr100 = atr_sma(tr, i, 100)
    hh100  = roll_max(highs, i, 100)
    ll100  = roll_min(lows,  i, 100)
    hl100  = hh100 - ll100
    bb20   = boll_bw(closes, i, 20)
    bb50   = boll_bw(closes, i, 50)
    ch100  = chop(klines, tr, i, 100)
    tc100  = trend_consist(closes, i, 100, 10)

    feats['BollBW50']        = bb50
    feats['ATR_ratio100']    = atr100 / c
    feats['BW_CHOP100']      = (bb20 / ch100) if (ch100 != 0 and not math.isnan(ch100)) else 0.0
    feats['ATR_HL_ratio100'] = atr100 / hl100 if hl100 > 0 else 0.0
    feats['TrendConsist100'] = tc100

    # ZEC-specific (ZEC uses the full feature set; ETH UP omits these)
    feats['return_5_3']      = (klines[i]['c'] / klines[i-3]['c']) - 1.0
    feats['return_60_3']     = (klines[i]['c'] / klines[i-36]['c']) - 1.0
    feats['volatility_5']    = roll_std(r1pct, i, 12)
    feats['mom_5']           = roll_sum(r1pct, i, 5)

    d = datetime.fromtimestamp(klines[i]['t'] / 1000, tz=timezone.utc)
    feats['hour'] = d.hour
    feats['dow']  = d.weekday()  # Mon=0..Sun=6 (pandas convention)

    fr = fr_ma24 = fr_cum24 = 0.0
    if funding:
        sorted_f = sorted(funding, key=lambda x: x['t'])
        cur = 0.0
        for f in sorted_f:
            if f['t'] <= klines[i]['t']:
                cur = f['r']
            else: break
        fr = cur
        s = 0.0; cnt = 0
        for off in range(95, -1, -1):
            t = klines[i - off]['t']
            v = 0.0
            for f in sorted_f:
                if f['t'] <= t: v = f['r']
                else: break
            s += v; cnt += 1
        fr_ma24 = s / cnt if cnt > 0 else 0.0
        fr_cum24 = s
    feats['funding_rate']        = fr
    feats['funding_rate_ma_24h'] = fr_ma24
    feats['funding_cum_24h']     = fr_cum24

    ema5_arr = ema_series(closes, 5)
    ema5 = ema5_arr[i]
    feats['close_to_ema5'] = c / ema5 - 1.0
    feats['dist_ema5_atr'] = (c - ema5) / atr14 if atr14 > 0 else 0.0

    rv6  = roll_std(r1, i, 24)
    rv24 = roll_std(r1, i, 96)
    feats['rv6_over_rv24'] = rv6 / rv24 if rv24 > 0 else 0.0

    # body_signed_3 = mean over last 3 bars of (c-o)/(h-l)
    bs_sum = 0.0
    for j in range(i - 2, i + 1):
        rng_j = klines[j]['h'] - klines[j]['l']
        bs_sum += (klines[j]['c'] - klines[j]['o']) / rng_j if rng_j > 0 else 0.0
    feats['body_signed_3'] = bs_sum / 3.0

    # wick_imbalance_3 = mean over last 3 bars of (lower_wick - upper_wick)/(h-l)
    wi_sum = 0.0
    for j in range(i - 2, i + 1):
        rng_j = klines[j]['h'] - klines[j]['l']
        if rng_j > 0:
            upper_w = (klines[j]['h'] - max(klines[j]['c'], klines[j]['o'])) / rng_j
            lower_w = (min(klines[j]['c'], klines[j]['o']) - klines[j]['l']) / rng_j
            wi_sum += lower_w - upper_w
    feats['wick_imbalance_3'] = wi_sum / 3.0

    # volume_surge_3 = current bar volume / mean over last 24 bars (H(6) at 15m)
    vol_mean_24 = roll_mean(vols, i, 24)
    feats['volume_surge_3'] = (vols[i] / vol_mean_24) if vol_mean_24 > 0 else 0.0

    # taker_imb_sum_6 = sum over last 24 bars (H(6) at 15m) of (2*tbb-v)/v
    imb_sum = 0.0
    for j in range(i - 23, i + 1):
        if klines[j]['v'] > 0:
            imb_sum += (2 * klines[j]['tbb'] - klines[j]['v']) / klines[j]['v']
    feats['taker_imb_sum_6'] = imb_sum

    feats['ret_skew_24h']       = skewness(r1, i, 96)
    feats['sign_persist_12']    = sign_persist_mean(r1, i, 12)
    feats['range_efficiency_6'] = range_eff_mean(klines, i, 6)
    feats['trend_r2_12']        = trend_r2(closes, i, 12)

    # dist_high24_atr / dist_low24_atr in ATR units, over last 96 bars (H(24) at 15m)
    hh24 = roll_max(highs, i, 96)
    ll24 = roll_min(lows,  i, 96)
    feats['dist_high24_atr'] = (c - hh24) / atr14 if atr14 > 0 else 0.0
    feats['dist_low24_atr']  = (c - ll24) / atr14 if atr14 > 0 else 0.0

    feats['open_time'] = d.year

    # Sanity check
    for name in FEATURE_ORDER:
        if name not in feats or not math.isfinite(feats[name]):
            raise RuntimeError(f'feature not finite: {name} = {feats.get(name)}')

    return feats, ema5


# ─────────────────────────────────────────────────────────────────────────────
# PACKING + ON-CHAIN
# ─────────────────────────────────────────────────────────────────────────────

def pack_features(values, scale_q):
    """Float → int32 Q → little-endian bytes (mirrors dApp packFeatures)."""
    INT32_MIN = -2147483648
    INT32_MAX =  2147483647
    out = bytearray(len(values) * 4)
    for j, v in enumerate(values):
        q = int(round(v * scale_q))
        if q < INT32_MIN: q = INT32_MIN
        if q > INT32_MAX: q = INT32_MAX
        struct.pack_into('<i', out, j * 4, q)
    return bytes(out)


_meta_cache = None

def resolve_model_meta(w3, retries=8):
    """Resolve & cache the model metadata. Retries with exponential backoff on
    RPC errors so a transient blip at startup doesn't bring down the daemon."""
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            registry = w3.eth.contract(address=Web3.to_checksum_address(REGISTRY), abi=ABI_REGISTRY)
            summary = registry.functions.getModelSummary(TOKEN_ID).call()
            exists = summary[0]
            model_id = summary[1]
            inference_enabled = summary[10]
            if not exists:
                raise RuntimeError(f'tokenId {TOKEN_ID} does not exist')
            if not inference_enabled:
                raise RuntimeError(f'inference disabled for tokenId {TOKEN_ID}')
            rt = registry.functions.getModelRuntime(model_id).call()
            n_features = rt[4]
            scale_q    = rt[8]
            if not scale_q or not n_features:
                raise RuntimeError('invalid runtime metadata')
            if n_features != len(FEATURE_ORDER):
                raise RuntimeError(f'feature count mismatch: chain={n_features} script={len(FEATURE_ORDER)}')
            _meta_cache = {
                'model_id': model_id,
                'scale_q':  scale_q,
                'n_features': n_features,
            }
            return _meta_cache
        except Exception as e:
            last_err = e
            sleep = 0.5 * (1.6 ** (attempt - 1))
            log(f'resolve_model_meta retry {attempt}/{retries} after error: {e} (sleep {sleep:.2f}s)', 'WARN')
            time.sleep(sleep)
    raise RuntimeError(f'resolve_model_meta failed after {retries} attempts: {last_err}')


def call_predict(w3, model_id, packed_bytes, retries=8):
    runtime = w3.eth.contract(address=Web3.to_checksum_address(RUNTIME), abi=ABI_RUNTIME)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            score_q = runtime.functions.predictView(model_id, packed_bytes).call()
            return score_q
        except Exception as e:
            last_err = e
            sleep = 0.5 * (1.6 ** (attempt - 1))
            log(f'predictView retry {attempt}/{retries} after error: {e} (sleep {sleep:.2f}s)', 'WARN')
            time.sleep(sleep)
    raise RuntimeError(f'predictView failed after {retries} attempts: {last_err}')


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

CSV_HEADERS = [
    'open_time_iso',         # ISO timestamp of the candle open
    'open_time_ms',          # Same as int ms
    'open',                  # Closed candle OHLC
    'high',
    'low',
    'close',
    'volume',
    'ema5',                  # EMA5 baseline at this candle
    'target_price',          # ema5 * 1.040
    'stop_price',            # ema5 * 0.990
    'raw_logit',             # Score from chain / scaleQ
    'score_q_hex',           # Raw int as hex (deterministic seed)
    'probability',           # sigmoid(raw_logit) — in [0,1]
    'probability_pct',       # probability * 100, 4dp
    'classification',        # below/weak/indeter/elevated/high
    'inferred_at_iso',       # When this row was written
    'fetch_ms',
    'features_ms',
    'inference_ms',
]


def classify_label(prob_pct):
    if prob_pct < 20:  return 'below'
    if prob_pct < 40:  return 'weak'
    if prob_pct < 60:  return 'indeter'
    if prob_pct < 80:  return 'elevated'
    return 'high'


def ensure_csv():
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open('w', newline='') as f:
            csv.writer(f).writerow(CSV_HEADERS)
        log(f'created {CSV_PATH}')


def dedupe_existing_csv():
    """One-shot cleanup: collapse rows with identical open_time_ms.

    For each duplicate group we keep the LATEST row (highest inferred_at_iso),
    discarding earlier writes for the same candle. Preserves the file's mode
    and ownership, matching the same atomic-rewrite pattern as the trim path.
    """
    if not CSV_PATH.exists():
        return
    with CSV_PATH.open('r', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return
        rows = list(reader)
    if len(rows) < 2:
        return

    # Find index of open_time_ms and inferred_at_iso columns
    try:
        idx_open = header.index('open_time_ms')
        idx_when = header.index('inferred_at_iso')
    except ValueError:
        return

    # Walk rows, keep newest write per candle key
    by_candle = {}
    order = []  # preserve original order of first-seen candles
    for r in rows:
        if len(r) != len(header):
            continue
        key = r[idx_open]
        when = r[idx_when]
        if key not in by_candle:
            by_candle[key] = r
            order.append(key)
        else:
            if when > by_candle[key][idx_when]:
                by_candle[key] = r

    deduped = [by_candle[k] for k in order]
    if len(deduped) == len(rows):
        return  # no duplicates found

    n_dropped = len(rows) - len(deduped)

    try:
        orig_stat = CSV_PATH.stat()
    except FileNotFoundError:
        orig_stat = None

    fd, tmp_path = tempfile.mkstemp(
        prefix='.zecup_', suffix='.csv.tmp', dir=str(CSV_PATH.parent)
    )
    try:
        with os.fdopen(fd, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(deduped)
        if orig_stat is not None:
            os.chmod(tmp_path, orig_stat.st_mode & 0o777)
            try:
                os.chown(tmp_path, orig_stat.st_uid, orig_stat.st_gid)
            except (PermissionError, OSError):
                pass
        else:
            os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, CSV_PATH)
    except Exception:
        try: os.unlink(tmp_path)
        except OSError: pass
        raise
    log(f'CSV deduped on boot: {n_dropped} duplicate rows removed, {len(deduped)} kept')


def append_row(row):
    with CSV_PATH.open('a', newline='') as f:
        csv.writer(f).writerow(row)
    _enforce_csv_cap()


# Keep at most this many rows in the CSV. The daemon trims older rows after
# every append. The HTML page paginates the survivors at 50/page.
CSV_MAX_ROWS = int(os.environ.get('ZECUP_CSV_MAX_ROWS', '500'))


def _enforce_csv_cap():
    """Trim oldest rows from the CSV if it exceeds CSV_MAX_ROWS.

    After back-filling older candles, file order is no longer the same as
    chronological order, so we sort by open_time_ms before deciding what to
    keep — we always keep the most chronologically recent rows regardless
    of where they sit in the file.

    Rewrites the file in place via a temp + rename. Safe under concurrent
    readers — they always see either the old file or the new one.
    """
    if not CSV_PATH.exists():
        return
    with CSV_PATH.open('r', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return
        rows = list(reader)
    if len(rows) <= CSV_MAX_ROWS:
        return

    # Find open_time_ms column.
    try:
        idx_ms = header.index('open_time_ms')
    except ValueError:
        # Fall back to file-order trim if the column is missing.
        kept = rows[-CSV_MAX_ROWS:]
    else:
        def _ms(r):
            try:
                return int(r[idx_ms])
            except (ValueError, IndexError, TypeError):
                return 0
        # Sort chronologically and keep the newest CSV_MAX_ROWS.
        sorted_rows = sorted(rows, key=_ms)
        kept = sorted_rows[-CSV_MAX_ROWS:]

    # Capture the original CSV's mode + ownership for the same reason as
    # the HTML rewrite — mkstemp gives us 0600, which would 403 the file.
    try:
        orig_stat = CSV_PATH.stat()
    except FileNotFoundError:
        orig_stat = None

    fd, tmp_path = tempfile.mkstemp(
        prefix='.zecup_', suffix='.csv.tmp', dir=str(CSV_PATH.parent)
    )
    try:
        with os.fdopen(fd, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(kept)
        if orig_stat is not None:
            os.chmod(tmp_path, orig_stat.st_mode & 0o777)
            try:
                os.chown(tmp_path, orig_stat.st_uid, orig_stat.st_gid)
            except (PermissionError, OSError):
                pass
        else:
            os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, CSV_PATH)
    except Exception:
        try: os.unlink(tmp_path)
        except OSError: pass
        raise
    log(f'CSV trimmed: dropped {len(rows) - len(kept)} oldest, kept {len(kept)}')


# ─────────────────────────────────────────────────────────────────────────────
# HTML PAGE REGENERATION — bake history into the static page
# ─────────────────────────────────────────────────────────────────────────────

import html as _html_escape
import tempfile

HISTORY_BEGIN_MARKER = '<!-- HISTORY:BEGIN'
HISTORY_END_MARKER   = '<!-- HISTORY:END -->'

_MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


def _fmt_candle_ts(iso_str):
    """ISO timestamp → 'May 08 · 13:45' (UTC)."""
    if not iso_str:
        return '—'
    try:
        # Strip trailing tz suffix variants, then parse
        d = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        d = d.astimezone(timezone.utc)
        return f'{_MONTHS[d.month - 1]} {d.day:02d} · {d.hour:02d}:{d.minute:02d}'
    except Exception:
        return iso_str


def _fmt_dollar(s):
    try:
        n = float(s)
    except (ValueError, TypeError):
        return '—'
    if not math.isfinite(n):
        return '—'
    # ZEC trades sub-$10 most of the time; 4 dp keeps EMA / target / stop legible.
    return f'${n:,.4f}'


def _fmt_age(iso_str):
    if not iso_str:
        return '—'
    try:
        d = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        d = d.astimezone(timezone.utc)
    except Exception:
        return iso_str
    now = datetime.now(timezone.utc)
    delta = (now - d).total_seconds()
    if delta < 60:    return f'{int(delta)}s ago'
    if delta < 3600:  return f'{int(delta // 60)}m ago'
    if delta < 86400: return f'{int(delta // 3600)}h ago'
    return f'{int(delta // 86400)}d ago'


def _read_csv_rows():
    """Read all logged predictions back. Returns list of dict rows."""
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open('r', newline='') as f:
        return list(csv.DictReader(f))


HISTORY_PAGE_SIZE = int(os.environ.get('ZECUP_PAGE_SIZE', '50'))


def _render_history_section(rows):
    """Render the <section class="card history-card"> block — full HTML."""
    if not rows:
        return (
            '  <!-- HISTORY:BEGIN — content below is regenerated by zecup_predictor_daemon.py -->\n'
            '  <section class="card history-card" id="historyCard">\n'
            '    <header class="card-head">\n'
            '      <span class="card-title">Inference history</span>\n'
            '      <span class="card-sub">Updates every 15m candle close</span>\n'
            '    </header>\n'
            '    <div class="history-meta">\n'
            '      <span><span class="meta-val">0</span> records</span>\n'
            '      <span>Last bar: <span class="meta-val">—</span></span>\n'
            '      <span>Updated: <span class="meta-val">—</span></span>\n'
            '    </div>\n'
            '    <div class="history-empty">\n'
            '      Awaiting first inference. The daemon writes one row per 15-minute candle close.\n'
            '    </div>\n'
            '  </section>\n'
            '  <!-- HISTORY:END -->'
        )

    # Newest first — sort by open_time_ms numerically
    sorted_rows = sorted(
        rows,
        key=lambda r: int(r.get('open_time_ms', 0) or 0),
        reverse=True
    )

    total_count = len(sorted_rows)
    head        = sorted_rows[0]
    last_bar    = _fmt_candle_ts(head.get('open_time_iso'))
    updated     = _fmt_age(head.get('inferred_at_iso'))

    # Stale flag: if last bar is older than 18 minutes, badge it
    stale = False
    try:
        last_open_ms = int(head.get('open_time_ms', 0) or 0)
        if last_open_ms > 0:
            age_min = (time.time() - last_open_ms / 1000) / 60
            if age_min > 18:
                stale = True
    except (ValueError, TypeError):
        pass

    if stale:
        updated_html = f'<span class="meta-val meta-stale">{_html_escape.escape(updated)} · daemon may be down</span>'
    else:
        updated_html = f'<span class="meta-val">{_html_escape.escape(updated)}</span>'

    # Render up to HISTORY_MAX_ROWS rows, paginated at HISTORY_PAGE_SIZE per page
    capped     = sorted_rows[:HISTORY_MAX_ROWS]
    n_pages    = max(1, (len(capped) + HISTORY_PAGE_SIZE - 1) // HISTORY_PAGE_SIZE)

    body_rows = []
    for idx, r in enumerate(capped):
        page_num = (idx // HISTORY_PAGE_SIZE) + 1
        cls = (r.get('classification') or 'below').strip()
        if cls not in ('below', 'weak', 'indeter', 'elevated', 'high'):
            cls = 'below'

        try:
            prob_pct = float(r.get('probability_pct', ''))
            prob_str = f'{prob_pct:.2f}%'
        except (ValueError, TypeError):
            prob_str = '—'

        try:
            logit = float(r.get('raw_logit', ''))
            logit_str = f'{logit:.3f}'
        except (ValueError, TypeError):
            logit_str = '—'

        # Hide all rows except page 1 by default. Pager script un-hides as needed.
        hidden_attr = '' if page_num == 1 else ' hidden'

        body_rows.append(
            f'            <tr data-page="{page_num}"{hidden_attr}>\n'
            f'              <td class="time">{_html_escape.escape(_fmt_candle_ts(r.get("open_time_iso")))}</td>\n'
            f'              <td class="num prob {cls}">{prob_str}</td>\n'
            f'              <td><span class="chip {cls}">{cls}</span></td>\n'
            f'              <td class="num">{logit_str}</td>\n'
            f'              <td class="num">{_fmt_dollar(r.get("close"))}</td>\n'
            f'              <td class="num">{_fmt_dollar(r.get("ema5"))}</td>\n'
            f'              <td class="num">{_fmt_dollar(r.get("target_price"))}</td>\n'
            f'              <td class="num">{_fmt_dollar(r.get("stop_price"))}</td>\n'
            '            </tr>'
        )
    tbody = '\n'.join(body_rows)

    # Pager UI — only rendered when there's more than one page
    if n_pages > 1:
        page_range_text = f'1–{min(HISTORY_PAGE_SIZE, len(capped))} of {len(capped)}'
        pager_html = (
            '    <div class="history-pager" id="historyPager" data-total-pages="' + str(n_pages) + '" data-page-size="' + str(HISTORY_PAGE_SIZE) + '" data-total-rows="' + str(len(capped)) + '">\n'
            '      <button class="pager-btn pager-prev" disabled aria-label="Previous page">←</button>\n'
            '      <span class="pager-status">Showing <span class="pager-range">' + page_range_text + '</span></span>\n'
            '      <span class="pager-pages" id="pagerPages"></span>\n'
            '      <button class="pager-btn pager-next" aria-label="Next page">→</button>\n'
            '    </div>\n'
            '    <script>\n'
            '      (function() {\n'
            '        var card  = document.getElementById("historyCard"); if (!card) return;\n'
            '        var pager = card.querySelector("#historyPager"); if (!pager) return;\n'
            '        var nPages   = parseInt(pager.dataset.totalPages, 10);\n'
            '        var pageSize = parseInt(pager.dataset.pageSize, 10);\n'
            '        var totalRows= parseInt(pager.dataset.totalRows, 10);\n'
            '        var rows  = card.querySelectorAll("tbody tr[data-page]");\n'
            '        var prev  = pager.querySelector(".pager-prev");\n'
            '        var next  = pager.querySelector(".pager-next");\n'
            '        var rangeEl = pager.querySelector(".pager-range");\n'
            '        var pagesEl = pager.querySelector("#pagerPages");\n'
            '        var current = 1;\n'
            '        function renderPagesNumbers() {\n'
            '          // Compact pager: show 1, current-1, current, current+1, last (with ellipses)\n'
            '          var nums = new Set([1, nPages, current-1, current, current+1]);\n'
            '          var visible = Array.from(nums).filter(function(n){return n>=1 && n<=nPages;}).sort(function(a,b){return a-b;});\n'
            '          var html = "";\n'
            '          for (var i = 0; i < visible.length; i++) {\n'
            '            if (i > 0 && visible[i] - visible[i-1] > 1) html += "<span class=\\"pager-ellipsis\\">…</span>";\n'
            '            var n = visible[i];\n'
            '            var isCurrent = (n === current);\n'
            '            html += \'<button class="pager-num\' + (isCurrent ? " active" : "") + \'" data-page="\' + n + \'"\' + (isCurrent ? \' aria-current="page"\' : \'\') + \'>\' + n + \'</button>\';\n'
            '          }\n'
            '          pagesEl.innerHTML = html;\n'
            '        }\n'
            '        function showPage(p) {\n'
            '          if (p < 1 || p > nPages) return;\n'
            '          current = p;\n'
            '          for (var i = 0; i < rows.length; i++) {\n'
            '            var rp = parseInt(rows[i].dataset.page, 10);\n'
            '            if (rp === p) rows[i].removeAttribute("hidden");\n'
            '            else rows[i].setAttribute("hidden", "");\n'
            '          }\n'
            '          var firstIdx = (p - 1) * pageSize + 1;\n'
            '          var lastIdx  = Math.min(p * pageSize, totalRows);\n'
            '          rangeEl.textContent = firstIdx + "–" + lastIdx + " of " + totalRows;\n'
            '          prev.disabled = (p === 1);\n'
            '          next.disabled = (p === nPages);\n'
            '          renderPagesNumbers();\n'
            '          // Keep table top in view when paging\n'
            '          var scroll = card.querySelector(".history-scroll");\n'
            '          if (scroll) scroll.scrollTop = 0;\n'
            '        }\n'
            '        prev.addEventListener("click", function(){ showPage(current - 1); });\n'
            '        next.addEventListener("click", function(){ showPage(current + 1); });\n'
            '        pagesEl.addEventListener("click", function(e){\n'
            '          var btn = e.target.closest(".pager-num"); if (!btn) return;\n'
            '          showPage(parseInt(btn.dataset.page, 10));\n'
            '        });\n'
            '        renderPagesNumbers();\n'
            '      })();\n'
            '    </script>\n'
        )
    else:
        pager_html = ''

    return (
        '  <!-- HISTORY:BEGIN — content below is regenerated by zecup_predictor_daemon.py -->\n'
        '  <section class="card history-card" id="historyCard">\n'
        '    <header class="card-head">\n'
        '      <span class="card-title">Inference history</span>\n'
        '      <span class="card-sub">Updates every 15m candle close</span>\n'
        '    </header>\n'
        '    <div class="history-meta">\n'
        f'      <span><span class="meta-val">{total_count:,}</span> records</span>\n'
        f'      <span>Last bar: <span class="meta-val">{_html_escape.escape(last_bar)}</span></span>\n'
        f'      <span>Updated: {updated_html}</span>\n'
        '    </div>\n'
        '    <div class="history-scroll">\n'
        '      <table class="hist-table">\n'
        '        <thead>\n'
        '          <tr>\n'
        '            <th>Candle (UTC)</th>\n'
        '            <th class="num">Probability</th>\n'
        '            <th>Class</th>\n'
        '            <th class="num">Logit</th>\n'
        '            <th class="num">Close</th>\n'
        '            <th class="num">EMA<sub>5</sub></th>\n'
        '            <th class="num">Target +4.0%</th>\n'
        '            <th class="num">Stop −1.0%</th>\n'
        '          </tr>\n'
        '        </thead>\n'
        '        <tbody>\n'
        f'{tbody}\n'
        '        </tbody>\n'
        '      </table>\n'
        '    </div>\n'
        f'{pager_html}'
        '  </section>\n'
        '  <!-- HISTORY:END -->'
    )


def regenerate_html(rows):
    """Read the HTML, splice the history section in place, atomically write back.

    The file is rewritten via temp + rename in the same directory. POSIX
    guarantees readers either see the entire old file or the entire new one,
    never a partial — so a webserver can keep serving the page while we
    rebuild it every 15 minutes.

    The sentinel <!-- HISTORY:BEGIN ... HISTORY:END --> markers are preserved
    in the output so subsequent rebuilds find them again. Idempotent: rebuilding
    with the same rows produces byte-identical output.
    """
    if not HTML_PATH.exists():
        log(f'HTML file missing: {HTML_PATH} — skipping page regeneration', 'WARN')
        return

    current = HTML_PATH.read_text(encoding='utf-8')

    begin = current.find(HISTORY_BEGIN_MARKER)
    end   = current.find(HISTORY_END_MARKER)
    if begin == -1 or end == -1 or end <= begin:
        log(f'HTML missing HISTORY markers — cannot splice', 'WARN')
        return
    end_complete = end + len(HISTORY_END_MARKER)

    # Find the start of the line containing HISTORY:BEGIN to preserve indentation.
    line_start = current.rfind('\n', 0, begin) + 1

    section = _render_history_section(rows)
    new_html = current[:line_start] + section + current[end_complete:]

    # Skip the write entirely if nothing actually changed (avoid touching mtime).
    if new_html == current:
        return

    # Capture the existing file's mode + ownership BEFORE writing the temp.
    # mkstemp creates files with mode 0600 — without this fix-up, nginx
    # (www-data) would lose read access after the rename and serve 403.
    try:
        orig_stat = HTML_PATH.stat()
    except FileNotFoundError:
        orig_stat = None

    # Atomic rewrite: temp file in same directory → rename onto final path.
    fd, tmp_path = tempfile.mkstemp(
        prefix='.zecup_', suffix='.html.tmp', dir=str(HTML_PATH.parent)
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(new_html)
        # Restore mode (and ownership when we have permission). World-readable
        # 0644 fallback covers fresh installs where the file didn't exist yet.
        if orig_stat is not None:
            os.chmod(tmp_path, orig_stat.st_mode & 0o777)
            try:
                os.chown(tmp_path, orig_stat.st_uid, orig_stat.st_gid)
            except (PermissionError, OSError):
                pass  # Non-fatal: only matters when running as root.
        else:
            os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, HTML_PATH)
    except Exception:
        try: os.unlink(tmp_path)
        except OSError: pass
        raise


# ─────────────────────────────────────────────────────────────────────────────
# CORE TICK
# ─────────────────────────────────────────────────────────────────────────────

def _last_logged_open_ms():
    """Return the open_time_ms of the most recent CSV row, or 0 if none."""
    rows = _read_csv_rows()
    if not rows:
        return 0
    try:
        # Rows may have been written out of order during backfill — pick the max.
        ms = 0
        for r in rows:
            try:
                v = int(r.get('open_time_ms', 0) or 0)
                if v > ms:
                    ms = v
            except (ValueError, TypeError):
                continue
        return ms
    except Exception:
        return 0


def _logged_open_ms_set():
    """Return a set of every open_time_ms currently in the CSV.

    Used to detect which candles in our fetched klines window we still need to
    log, so a one-off failure doesn't permanently drop a candle — the next
    cycle picks the missed candle up as 'missing' and back-fills it.
    """
    out = set()
    for r in _read_csv_rows():
        try:
            ms = int(r.get('open_time_ms', 0) or 0)
        except (ValueError, TypeError):
            continue
        if ms > 0:
            out.add(ms)
    return out


def _log_one_candle(w3, klines, funding, idx, meta):
    """Compute features at klines[idx], call predictView, append row to CSV.

    Returns (open_time_ms, prob_pct) on success. Raises on any failure — the
    caller decides whether one bad candle should abort the whole cycle or just
    be skipped. We deliberately do NOT regenerate the HTML here; that's done
    once at the end of the cycle by the caller.
    """
    bar = klines[idx]
    open_ms = bar['t']
    open_dt = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc)

    t_feat0 = time.monotonic()
    feats, ema5 = build_features(klines, funding, i=idx)
    values = [feats[name] for name in FEATURE_ORDER]
    features_ms = (time.monotonic() - t_feat0) * 1000

    t_inf0 = time.monotonic()
    packed = pack_features(values, meta['scale_q'])
    score_q = call_predict(w3, meta['model_id'], packed)
    inference_ms = (time.monotonic() - t_inf0) * 1000

    raw_logit = score_q / meta['scale_q']
    probability = 1.0 / (1.0 + math.exp(-raw_logit))
    prob_pct = probability * 100

    target = ema5 * 1.040
    stop   = ema5 * 0.990
    score_q_hex = '0x' + format(score_q & ((1 << 256) - 1), 'x') \
        if score_q >= 0 else '-0x' + format(-score_q, 'x')

    row = [
        open_dt.isoformat(),
        open_ms,
        f'{bar["o"]:.2f}',
        f'{bar["h"]:.2f}',
        f'{bar["l"]:.2f}',
        f'{bar["c"]:.2f}',
        f'{bar["v"]:.2f}',
        f'{ema5:.4f}',
        f'{target:.4f}',
        f'{stop:.4f}',
        f'{raw_logit:.6f}',
        score_q_hex,
        f'{probability:.6f}',
        f'{prob_pct:.4f}',
        classify_label(prob_pct),
        datetime.now(timezone.utc).isoformat(),
        '0.0',                       # fetch_ms — accounted for at cycle level
        f'{features_ms:.1f}',
        f'{inference_ms:.1f}',
    ]
    append_row(row)
    return open_ms, prob_pct


# Default: on a normal tick we fetch 251 bars (drop the in-flight one → 250).
# On startup or when explicit deep back-fill is wanted, we can fetch up to 1500
# bars per Binance call (~15.6 days of 15m candles).
KLINES_FETCH_LIMIT       = int(os.environ.get('ZECUP_KLINES_LIMIT',       '251'))
KLINES_FETCH_LIMIT_BOOT  = int(os.environ.get('ZECUP_KLINES_LIMIT_BOOT',  '500'))


def perform_inference_cycle(w3, fetch_limit=None):
    """One cycle: fetch market data, then log every candle in the fetched
    window that is still missing from the CSV.

    Returns (n_new_rows, latest_logged_ms). The caller can compare
    `latest_logged_ms` to the latest closed candle to know whether the most
    recent candle was successfully logged this cycle.

    Each candle is logged independently — a single failure does NOT abort the
    cycle for the other candles. The cycle re-renders the HTML page exactly
    once at the end, if any new rows were written.
    """
    if fetch_limit is None:
        fetch_limit = KLINES_FETCH_LIMIT

    t0 = time.monotonic()
    klines_raw = fetch_klines(limit=fetch_limit)
    funding    = fetch_funding()
    klines = klines_raw[:-1]  # all CLOSED bars (drop in-flight)
    if len(klines) < 105:
        raise RuntimeError(f'only {len(klines)} closed bars, need ≥ 105')
    fetch_ms = (time.monotonic() - t0) * 1000

    latest = klines[-1]
    latest_ms = latest['t']
    latest_dt = datetime.fromtimestamp(latest_ms / 1000, tz=timezone.utc)

    # Resolve model metadata first (cached after first success).
    meta = resolve_model_meta(w3)

    # Identify every candle in our fetched window that isn't logged yet AND
    # has enough lookback (i ≥ 104). Build a list of indices to log, newest
    # last so the CSV gets appended in chronological order.
    logged = _logged_open_ms_set()
    to_log = []  # list of indices into `klines`
    for i in range(104, len(klines)):
        if klines[i]['t'] not in logged:
            to_log.append(i)

    if not to_log:
        log(f'cycle: no new candles to log (latest closed: {latest_dt.isoformat()} '
            f'· fetched {len(klines)} closed bars · fetch {fetch_ms:.0f}ms)')
        # Still refresh the HTML in case it's stale.
        try:
            rows = _read_csv_rows()
            regenerate_html(rows)
        except Exception as e:
            log(f'page regeneration failed: {e}', 'WARN')
        return 0, _last_logged_open_ms()

    n_to_log = len(to_log)
    if n_to_log > 1:
        first_dt = datetime.fromtimestamp(klines[to_log[0]]['t']/1000, tz=timezone.utc)
        log(f'cycle: {n_to_log} candle(s) to log — back-filling from '
            f'{first_dt.isoformat()} through {latest_dt.isoformat()} · '
            f'fetch {fetch_ms:.0f}ms')
    else:
        log(f'cycle: 1 candle to log — {latest_dt.isoformat()} · fetch {fetch_ms:.0f}ms')

    n_logged = 0
    latest_logged_ms = _last_logged_open_ms()
    last_err = None
    for idx in to_log:
        bar_ms = klines[idx]['t']
        bar_dt = datetime.fromtimestamp(bar_ms / 1000, tz=timezone.utc)
        # Defensive: re-check the CSV set in case another instance raced us.
        if bar_ms in _logged_open_ms_set():
            log(f'  skip {bar_dt.isoformat()} — already present (race)', 'WARN')
            continue
        try:
            ms, prob_pct = _log_one_candle(w3, klines, funding, idx, meta)
            n_logged += 1
            if ms > latest_logged_ms:
                latest_logged_ms = ms
            tag = '(latest)' if idx == len(klines) - 1 else '(backfill)'
            log(f'  ✓ {bar_dt.isoformat()} {tag} p = {prob_pct:.2f}% '
                f'class = {classify_label(prob_pct)}')
        except Exception as e:
            last_err = e
            log(f'  ✗ {bar_dt.isoformat()} failed: {e}', 'ERROR')
            # Continue with the next candle — they're independent.

    # Re-render once at the end if anything was added.
    if n_logged > 0:
        try:
            rows = _read_csv_rows()
            regenerate_html(rows)
            log(f'page regenerated: {HTML_PATH} ({len(rows)} records · {n_logged} new this cycle)')
        except Exception as e:
            log(f'page regeneration failed: {e}', 'WARN')

    # If we wrote SOMETHING but missed at least one, surface that to the
    # caller via last_err so the outer retry loop knows the cycle was partial.
    if n_logged < n_to_log and last_err is not None:
        raise RuntimeError(
            f'partial cycle: logged {n_logged}/{n_to_log}, last err: {last_err}'
        )

    return n_logged, latest_logged_ms


# Back-compat alias: anything that used to call perform_inference still works.
def perform_inference(w3):
    perform_inference_cycle(w3)


# ─────────────────────────────────────────────────────────────────────────────
# DEADLINE-AWARE RETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Stop retrying this many seconds before the next candle close, so we don't
# step on the next tick's heels. The next tick will pick up any candle we
# still missed via back-fill, so this is a safety margin not a deadline cliff.
RETRY_SAFETY_MARGIN_S = float(os.environ.get('ZECUP_RETRY_SAFETY_S', '20'))
# First retry waits this long; each subsequent wait doubles up to the cap.
RETRY_BACKOFF_INITIAL_S = float(os.environ.get('ZECUP_RETRY_BACKOFF_S', '5'))
RETRY_BACKOFF_CAP_S     = float(os.environ.get('ZECUP_RETRY_CAP_S',     '45'))


def _sleep_responsive(seconds):
    """Sleep that wakes promptly on SIGTERM/SIGINT."""
    slept = 0.0
    while _running and slept < seconds:
        chunk = min(0.5, seconds - slept)
        time.sleep(chunk)
        slept += chunk


def run_tick_with_retries(w3, deadline_monotonic, fetch_limit=None):
    """Call perform_inference_cycle() repeatedly, with exponential backoff,
    until either:
      - the cycle completes cleanly (every candle in the window is logged), OR
      - we reach `deadline_monotonic` (the next candle's close minus a margin).

    Returns True on full success.

    The cycle's contract: it returns normally if and only if every candle in
    its fetched window that needed logging was logged. A partial cycle raises.
    So this retry loop is simple: try, sleep, retry until clean or deadline.

    Any candle STILL missing when we give up is picked up by the NEXT tick's
    cycle as a back-fill target — combined with this in-window retry, that
    closes the loop on every transient failure mode short of an extended
    chain/Binance outage.
    """
    attempt = 0
    backoff = RETRY_BACKOFF_INITIAL_S
    while _running:
        attempt += 1
        try:
            perform_inference_cycle(w3, fetch_limit=fetch_limit)
            if attempt > 1:
                log(f'tick succeeded on attempt {attempt}')
            return True
        except Exception as e:
            log(f'tick attempt {attempt} failed: {e}', 'WARN')

        remaining = deadline_monotonic - time.monotonic()
        if remaining <= backoff + 2.0:
            log(f'tick giving up after {attempt} attempt(s) — {remaining:.1f}s '
                f'to deadline; any still-missing candle(s) will be back-filled '
                f'on the next tick', 'ERROR')
            return False

        log(f'  sleeping {backoff:.1f}s before retry '
            f'({remaining:.0f}s left in window)')
        _sleep_responsive(backoff)
        if not _running:
            return False
        backoff = min(RETRY_BACKOFF_CAP_S, backoff * 2)

    return False


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def seconds_to_next_tick():
    """Returns seconds until the next 15m boundary + BUFFER_SECONDS."""
    now = time.time()
    # Next bucket start (UTC), aligned to UTC midnight 15-minute slots
    secs_into_bucket = now % INTERVAL_SECONDS
    next_bucket = now + (INTERVAL_SECONDS - secs_into_bucket)
    target = next_bucket + BUFFER_SECONDS
    return max(0.0, target - now)


_running = True

def _sigterm(*_):
    global _running
    _running = False
    log('shutdown signal received, finishing current loop iter and exiting', 'INFO')


def main():
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT,  _sigterm)

    log(f'CSV path:  {CSV_PATH}')
    log(f'HTML path: {HTML_PATH} (in-place rewrite)')
    log(f'RPC: {RPC_URL} · runtime: {RUNTIME} · tokenId: {TOKEN_ID}')
    log(f'Schedule: every 15m candle close + {BUFFER_SECONDS}s buffer · '
        f'retry until next close − {RETRY_SAFETY_MARGIN_S:.0f}s')

    ensure_csv()
    dedupe_existing_csv()

    # Regenerate the page once on boot so the served HTML reflects whatever
    # is already in the CSV — useful after a daemon restart.
    try:
        existing_rows = _read_csv_rows()
        regenerate_html(existing_rows)
        log(f'page regenerated on boot: {len(existing_rows)} existing records')
    except Exception as e:
        log(f'boot-time page regeneration skipped: {e}', 'WARN')

    w3 = Web3(Web3.HTTPProvider(RPC_URL, request_kwargs={'timeout': 30}))
    if not w3.is_connected():
        log('failed to connect to RPC at startup — will retry on first tick', 'WARN')

    # Resolve & cache model metadata once at startup (has its own retry loop).
    try:
        meta = resolve_model_meta(w3)
        log(f'model #{TOKEN_ID} resolved · {meta["n_features"]} features · scaleQ={meta["scale_q"]}')
        log(f'modelId = 0x{meta["model_id"].hex()}')
    except Exception as e:
        log(f'could not pre-resolve model metadata: {e} (will retry per tick)', 'WARN')

    # On startup, fetch a LARGER window and back-fill any gaps caused by
    # daemon downtime. Default 500 bars ≈ 5.2 days of 15m candles. The retry
    # deadline is generous on boot since we're not racing the next bucket
    # yet — give it the rest of the current 15-minute window.
    if os.environ.get('ZECUP_INIT_NOW', '1') == '1':
        log(f'startup back-fill: fetching {KLINES_FETCH_LIMIT_BOOT} bars to '
            f'catch any candles missed during downtime …')
        boot_deadline = time.monotonic() + seconds_to_next_tick() \
                                          - RETRY_SAFETY_MARGIN_S
        try:
            run_tick_with_retries(
                w3, boot_deadline, fetch_limit=KLINES_FETCH_LIMIT_BOOT
            )
        except Exception as e:
            log(f'startup back-fill error: {e}', 'ERROR')

    while _running:
        wait = seconds_to_next_tick()
        log(f'sleeping {wait:.1f}s until next tick')
        _sleep_responsive(wait)
        if not _running:
            break

        # Deadline for this tick's retry loop: next candle close minus margin.
        # `seconds_to_next_tick()` just after the boundary is ≈ INTERVAL_SECONDS;
        # we want to stop retrying RETRY_SAFETY_MARGIN_S before that next close.
        deadline = time.monotonic() + INTERVAL_SECONDS - RETRY_SAFETY_MARGIN_S
        run_tick_with_retries(w3, deadline)

    log('exited cleanly')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""xrpdual_predictor_daemon.py — merge xrpup + xrpdn prediction CSVs into
the dual-model history block of xrpdual_5h_scrying.html.

This daemon does NO inference of its own. It reads:
  - xrpup_predictions.csv  (produced by xrpup_predictor_daemon.py)
  - xrpdn_predictions.csv  (produced by xrpdn_predictor_daemon.py)

Inner-joins them on `open_time_ms` (every candle where BOTH models have
inferred), computes the directional bias per row, and rewrites the section
between the <!-- HISTORY:BEGIN ... HISTORY:END --> sentinels in the dual
page with a paginated dual-direction table.

Run it alongside the two predictor daemons. It polls the source CSVs'
mtimes and regenerates the page when either source changes — so the dual
page stays in sync without coordination between the three processes.

Author : Bioinformatics LLC, 2025-2026.
License: CC BY-SA 4.0.
"""

from __future__ import annotations

import argparse
import csv
import html as _html_escape
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (env-overridable)
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()

CSV_UP_PATH = Path(os.environ.get(
    'XRPDUAL_CSV_UP', str(SCRIPT_DIR / 'xrpup_predictions.csv')
))
CSV_DN_PATH = Path(os.environ.get(
    'XRPDUAL_CSV_DN', str(SCRIPT_DIR / 'xrpdn_predictions.csv')
))
HTML_PATH = Path(os.environ.get(
    'XRPDUAL_HTML', str(SCRIPT_DIR / 'xrpdual_5h_scrying.html')
))

POLL_INTERVAL_S    = int(os.environ.get('XRPDUAL_POLL_S',    '20'))
HISTORY_MAX_ROWS   = int(os.environ.get('XRPDUAL_MAX_ROWS',  '500'))
HISTORY_PAGE_SIZE  = int(os.environ.get('XRPDUAL_PAGE_SIZE', '50'))

# Bias magnitude (as fraction in [0,1]) below which we render the bias as
# neutral rather than directional.
BIAS_NEUTRAL_BAND  = float(os.environ.get('XRPDUAL_BIAS_NEUTRAL', '0.05'))

HISTORY_BEGIN_MARKER = '<!-- HISTORY:BEGIN'
HISTORY_END_MARKER   = '<!-- HISTORY:END -->'

VALID_CLASSES = ('below', 'weak', 'indeter', 'elevated', 'high')

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str, level: str = 'INFO') -> None:
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {level:5s} {msg}', flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CSV HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open('r', newline='') as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error) as e:
        log(f'failed to read {path}: {e}', 'WARN')
        return []


def _index_by_ms(rows: list[dict]) -> dict[int, dict]:
    """Build {open_time_ms: row}. Skip rows with missing/non-numeric ms."""
    out: dict[int, dict] = {}
    for r in rows:
        try:
            ms = int(r.get('open_time_ms', 0) or 0)
        except (ValueError, TypeError):
            continue
        if ms <= 0:
            continue
        # If duplicates exist, last write wins (matches reader semantics for
        # an append-only CSV; the predictor daemons dedup but we don't rely
        # on that here).
        out[ms] = r
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _fmt_candle_ts(iso_str: str) -> str:
    """ISO timestamp → 'May 08 · 13:45' (UTC)."""
    if not iso_str:
        return '—'
    try:
        d = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        d = d.astimezone(timezone.utc)
        return f'{_MONTHS[d.month - 1]} {d.day:02d} · {d.hour:02d}:{d.minute:02d}'
    except Exception:
        return iso_str


def _fmt_dollar(s) -> str:
    try:
        n = float(s)
    except (ValueError, TypeError):
        return '—'
    if not math.isfinite(n):
        return '—'
    # XRP trades sub-$10 most of the time; 4 dp keeps EMA / close legible.
    return f'${n:,.4f}'


def _fmt_age(iso_str: str) -> str:
    if not iso_str:
        return '—'
    try:
        d = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        d = d.astimezone(timezone.utc)
    except Exception:
        return iso_str
    delta = (datetime.now(timezone.utc) - d).total_seconds()
    if delta < 60:    return f'{int(delta)}s ago'
    if delta < 3600:  return f'{int(delta // 60)}m ago'
    if delta < 86400: return f'{int(delta // 3600)}h ago'
    return f'{int(delta // 86400)}d ago'


def _safe_class(s: str) -> str:
    s = (s or 'below').strip()
    return s if s in VALID_CLASSES else 'below'


def _safe_pct(s) -> tuple[float, str]:
    """Return (numeric_pct, formatted_str) — first is 0 on error."""
    try:
        n = float(s)
        if not math.isfinite(n):
            return 0.0, '—'
        return n, f'{n:.2f}%'
    except (ValueError, TypeError):
        return 0.0, '—'


def _bias_kind(bias_frac: float) -> str:
    if bias_frac >  BIAS_NEUTRAL_BAND: return 'up'
    if bias_frac < -BIAS_NEUTRAL_BAND: return 'down'
    return 'neutral'


# ─────────────────────────────────────────────────────────────────────────────
# RENDERING — empty + populated history sections
# ─────────────────────────────────────────────────────────────────────────────

def _render_empty_section() -> str:
    return (
        '  <!-- HISTORY:BEGIN — content below is regenerated by xrpdual_predictor_daemon.py -->\n'
        '  <section class="card history-card" id="historyCard">\n'
        '    <header class="card-head">\n'
        '      <span class="card-title">Inference history</span>\n'
        '      <span class="card-sub">Dual-model · updates every 15m candle close</span>\n'
        '    </header>\n'
        '    <div class="history-meta">\n'
        '      <span><span class="meta-val">0</span> records</span>\n'
        '      <span>Last bar: <span class="meta-val">—</span></span>\n'
        '      <span>Updated: <span class="meta-val meta-stale">awaiting both daemons</span></span>\n'
        '    </div>\n'
        '    <div class="history-empty">\n'
        '      Inner-join of xrpup_predictions.csv + xrpdn_predictions.csv is empty. Both source daemons need to be running and to have produced predictions for at least one shared candle.\n'
        '    </div>\n'
        '  </section>\n'
        '  <!-- HISTORY:END -->'
    )


def _render_pager_html(n_pages: int, n_rows: int) -> str:
    """Pager UI + script. Identical pattern to the source daemons —
    it operates on the table inside #historyCard so column count is
    irrelevant."""
    if n_pages <= 1:
        return ''

    page_range_text = f'1–{min(HISTORY_PAGE_SIZE, n_rows)} of {n_rows}'

    return (
        '    <div class="history-pager" id="historyPager"'
        f' data-total-pages="{n_pages}" data-page-size="{HISTORY_PAGE_SIZE}" data-total-rows="{n_rows}">\n'
        '      <button class="pager-btn pager-prev" disabled aria-label="Previous page">←</button>\n'
        f'      <span class="pager-status">Showing <span class="pager-range">{page_range_text}</span></span>\n'
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


def _render_history_section(merged: list[tuple[dict, dict]],
                            n_up: int, n_dn: int) -> str:
    """Render the dual history block.

    `merged`  : list of (up_row, dn_row) tuples, sorted newest-first.
    `n_up`    : total rows in the up-source CSV (informational).
    `n_dn`    : total rows in the dn-source CSV (informational).
    """
    if not merged:
        return _render_empty_section()

    total_count = len(merged)
    head_up, head_dn = merged[0]
    last_bar      = _fmt_candle_ts(head_up.get('open_time_iso'))
    head_inferred = max(
        head_up.get('inferred_at_iso', '') or '',
        head_dn.get('inferred_at_iso', '') or '',
    )
    updated = _fmt_age(head_inferred)

    # Stale flag: most recent merged candle is older than 18 minutes
    stale = False
    try:
        last_ms = int(head_up.get('open_time_ms', 0) or 0)
        if last_ms > 0:
            age_min = (time.time() - last_ms / 1000) / 60
            if age_min > 18:
                stale = True
    except (ValueError, TypeError):
        pass

    if stale:
        updated_html = (f'<span class="meta-val meta-stale">'
                        f'{_html_escape.escape(updated)} · daemon may be down</span>')
    else:
        updated_html = f'<span class="meta-val">{_html_escape.escape(updated)}</span>'

    capped  = merged[:HISTORY_MAX_ROWS]
    n_pages = max(1, (len(capped) + HISTORY_PAGE_SIZE - 1) // HISTORY_PAGE_SIZE)

    body_rows = []
    for idx, (ru, rd) in enumerate(capped):
        page_num = (idx // HISTORY_PAGE_SIZE) + 1
        cls_up  = _safe_class(ru.get('classification'))
        cls_dn  = _safe_class(rd.get('classification'))

        pu_pct, pu_str = _safe_pct(ru.get('probability_pct'))
        pd_pct, pd_str = _safe_pct(rd.get('probability_pct'))

        # Bias is fraction-in-[-1,+1] for color thresholding, rendered as %.
        bias_frac = (pu_pct - pd_pct) / 100.0
        bias_kind = _bias_kind(bias_frac)
        bias_sign = '+' if bias_frac >= 0 else '−'
        bias_str  = f'{bias_sign}{abs(bias_frac) * 100:.1f}%'

        # Up-row close/ema5 are canonical (both daemons compute against the
        # same Binance feed so they agree, but we pick one consistently).
        close = ru.get('close')
        ema5  = ru.get('ema5')

        hidden_attr = '' if page_num == 1 else ' hidden'
        body_rows.append(
            f'            <tr data-page="{page_num}"{hidden_attr}>\n'
            f'              <td class="time">{_html_escape.escape(_fmt_candle_ts(ru.get("open_time_iso")))}</td>\n'
            f'              <td class="num prob {cls_up}">{pu_str}</td>\n'
            f'              <td><span class="chip chip-up {cls_up}">{cls_up}</span></td>\n'
            f'              <td class="num prob {cls_dn}">{pd_str}</td>\n'
            f'              <td><span class="chip chip-dn {cls_dn}">{cls_dn}</span></td>\n'
            f'              <td class="num bias {bias_kind}">{bias_str}</td>\n'
            f'              <td class="num">{_fmt_dollar(close)}</td>\n'
            f'              <td class="num">{_fmt_dollar(ema5)}</td>\n'
            '            </tr>'
        )
    tbody = '\n'.join(body_rows)
    pager_html = _render_pager_html(n_pages, len(capped))

    sub_text = f'Inner-join · ↑ {n_up:,} · ↓ {n_dn:,} · matched {total_count:,}'

    return (
        '  <!-- HISTORY:BEGIN — content below is regenerated by xrpdual_predictor_daemon.py -->\n'
        '  <section class="card history-card" id="historyCard">\n'
        '    <header class="card-head">\n'
        '      <span class="card-title">Inference history</span>\n'
        f'      <span class="card-sub">{_html_escape.escape(sub_text)}</span>\n'
        '    </header>\n'
        '    <div class="history-meta">\n'
        f'      <span><span class="meta-val">{total_count:,}</span> records</span>\n'
        f'      <span>Last bar: <span class="meta-val">{_html_escape.escape(last_bar)}</span></span>\n'
        f'      <span>Updated: {updated_html}</span>\n'
        '    </div>\n'
        '    <div class="history-scroll">\n'
        '      <table class="hist-table hist-table-dual">\n'
        '        <thead>\n'
        '          <tr>\n'
        '            <th>Candle (UTC)</th>\n'
        '            <th class="num">P↑</th>\n'
        '            <th>↑</th>\n'
        '            <th class="num">P↓</th>\n'
        '            <th>↓</th>\n'
        '            <th class="num">Bias</th>\n'
        '            <th class="num">Close</th>\n'
        '            <th class="num">EMA<sub>5</sub></th>\n'
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


# ─────────────────────────────────────────────────────────────────────────────
# HTML SPLICE — atomic rewrite preserving file mode/owner
# ─────────────────────────────────────────────────────────────────────────────

def regenerate_html(merged: list[tuple[dict, dict]],
                    n_up: int, n_dn: int) -> bool:
    """Splice the dual history section into HTML_PATH atomically.
    Returns True if the file actually changed, False otherwise."""
    if not HTML_PATH.exists():
        log(f'HTML file missing: {HTML_PATH} — skipping page regeneration', 'WARN')
        return False

    current = HTML_PATH.read_text(encoding='utf-8')
    begin = current.find(HISTORY_BEGIN_MARKER)
    end   = current.find(HISTORY_END_MARKER)
    if begin == -1 or end == -1 or end <= begin:
        log('HTML missing HISTORY markers — cannot splice', 'WARN')
        return False
    end_complete = end + len(HISTORY_END_MARKER)

    # Preserve indentation by snapping `begin` to the start of its line.
    line_start = current.rfind('\n', 0, begin) + 1

    section = _render_history_section(merged, n_up, n_dn)
    new_html = current[:line_start] + section + current[end_complete:]

    if new_html == current:
        return False

    # Capture original file mode/ownership BEFORE writing the temp, so the
    # webserver doesn't lose read access after the rename (mkstemp creates
    # 0600 files by default).
    try:
        orig_stat = HTML_PATH.stat()
    except FileNotFoundError:
        orig_stat = None

    fd, tmp_path = tempfile.mkstemp(
        prefix='.xrpdual_', suffix='.html.tmp', dir=str(HTML_PATH.parent)
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(new_html)
        if orig_stat is not None:
            os.chmod(tmp_path, orig_stat.st_mode & 0o777)
            try:
                os.chown(tmp_path, orig_stat.st_uid, orig_stat.st_gid)
            except (PermissionError, OSError):
                pass  # only matters when running as root
        else:
            os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, HTML_PATH)
    except Exception:
        try: os.unlink(tmp_path)
        except OSError: pass
        raise

    return True


# ─────────────────────────────────────────────────────────────────────────────
# MERGE TICK
# ─────────────────────────────────────────────────────────────────────────────

def merge_and_regenerate() -> tuple[int, int, int, bool]:
    """Read both source CSVs, inner-join on open_time_ms, regenerate page.
    Returns (n_up, n_dn, n_merged, changed)."""
    rows_up = _read_csv_rows(CSV_UP_PATH)
    rows_dn = _read_csv_rows(CSV_DN_PATH)
    idx_up  = _index_by_ms(rows_up)
    idx_dn  = _index_by_ms(rows_dn)
    common_ms = sorted(set(idx_up.keys()) & set(idx_dn.keys()), reverse=True)
    merged = [(idx_up[ms], idx_dn[ms]) for ms in common_ms]
    changed = regenerate_html(merged, len(idx_up), len(idx_dn))
    return len(idx_up), len(idx_dn), len(merged), changed


# ─────────────────────────────────────────────────────────────────────────────
# DAEMON LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0.0


def daemon_loop(poll_s: int) -> None:
    log(f'daemon mode · poll {poll_s}s · ↑ {CSV_UP_PATH} · ↓ {CSV_DN_PATH}')
    log(f'output   {HTML_PATH}')

    last_up = -1.0
    last_dn = -1.0
    last_html_check = 0.0

    # Prime once on startup so a cold start always rewrites.
    n_up, n_dn, n_m, changed = merge_and_regenerate()
    log(f'startup · ↑ {n_up} · ↓ {n_dn} · matched {n_m} · '
        f'{"updated" if changed else "no change"}')
    last_up = _mtime(CSV_UP_PATH)
    last_dn = _mtime(CSV_DN_PATH)

    while True:
        try:
            mt_up = _mtime(CSV_UP_PATH)
            mt_dn = _mtime(CSV_DN_PATH)
            if mt_up != last_up or mt_dn != last_dn:
                # NOTE: only advance the watermark AFTER a successful merge.
                # If merge raises, we keep last_{up,dn} unchanged so the next
                # poll iteration will see the mtime delta and retry — this is
                # how the daemon recovers from a transient I/O blip without
                # waiting for the next upstream CSV write.
                n_up, n_dn, n_m, changed = merge_and_regenerate()
                last_up, last_dn = mt_up, mt_dn
                if changed:
                    log(f'regenerated · ↑ {n_up} · ↓ {n_dn} · matched {n_m}')
                # If no change occurred but mtimes moved, log once.
                elif n_m == 0:
                    log(f'tick · ↑ {n_up} · ↓ {n_dn} · matched {n_m} (no inner-join yet)',
                        'INFO')
        except KeyboardInterrupt:
            log('caught SIGINT, exiting')
            return
        except Exception as e:
            log(f'tick error: {e} (will retry on next poll)', 'ERROR')

        try:
            time.sleep(poll_s)
        except KeyboardInterrupt:
            log('caught SIGINT, exiting')
            return


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description='Merge xrpup + xrpdn prediction CSVs into the dual '
                    'page history block.'
    )
    ap.add_argument('--once', action='store_true',
                    help='regenerate once and exit (no polling)')
    ap.add_argument('--poll', type=int, default=POLL_INTERVAL_S,
                    help=f'polling interval in seconds (default {POLL_INTERVAL_S})')
    args = ap.parse_args()

    log(f'sources: ↑ {CSV_UP_PATH}')
    log(f'         ↓ {CSV_DN_PATH}')
    log(f'output : {HTML_PATH}')

    if args.once:
        n_up, n_dn, n_m, changed = merge_and_regenerate()
        log(f'one-shot · ↑ {n_up} · ↓ {n_dn} · matched {n_m} · '
            f'{"updated" if changed else "no change"}')
        return 0

    daemon_loop(args.poll)
    return 0


if __name__ == '__main__':
    sys.exit(main())

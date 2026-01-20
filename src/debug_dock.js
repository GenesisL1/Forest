/*
MIT License

Copyright (c) 2026 Decentralized Science Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

import { makeLogger } from "./common.js";
import { getWalletState } from "./eth.js";

async function copyTextCompat(text) {
  const s = String(text ?? "");
  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(s);
      return true;
    }
  } catch {}
  // Fallback: execCommand
  try {
    const ta = document.createElement("textarea");
    ta.value = s;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    ta.style.top = "0";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return !!ok;
  } catch {
    return false;
  }
}

export function setupDebugDock({ state = "idle" } = {}) {
  const debugDockEl = document.getElementById("debugDock");
  const debugLinesEl = document.getElementById("debugLines");
  const debugFabEl = document.getElementById("debugFab");
  const dbgCollapseBtn = document.getElementById("dbgCollapse");
  const dbgCopyBtn = document.getElementById("dbgCopy");
  const dbgClearBtn = document.getElementById("dbgClear");
  const dbgStateEl = document.getElementById("dbgState");
  const dbgConnEl = document.getElementById("dbgConn");

  function setDockState(s) {
    if (dbgStateEl) dbgStateEl.textContent = String(s);
  }

  function setDockConn(s) {
    if (dbgConnEl) dbgConnEl.textContent = String(s);
  }

  // Controls
  if (dbgCollapseBtn && debugDockEl) {
    dbgCollapseBtn.addEventListener("click", () => {
      const collapsed = debugDockEl.classList.toggle("collapsed");
      dbgCollapseBtn.textContent = collapsed ? "Expand" : "Collapse";
    });
  }
  if (dbgClearBtn && debugLinesEl) {
    dbgClearBtn.addEventListener("click", () => {
      debugLinesEl.textContent = "";
    });
  }
  if (dbgCopyBtn && debugLinesEl) {
    dbgCopyBtn.addEventListener("click", async () => {
      await copyTextCompat(debugLinesEl.textContent || "");
    });
  }

  // Optional show/hide via FAB (create page ships FAB hidden, but keep behavior identical).
  if (debugFabEl && debugDockEl) {
    debugFabEl.addEventListener("click", () => {
      debugDockEl.hidden = false;
      debugFabEl.hidden = true;
    });
  }

  function refreshDockConn() {
    try {
      const w = getWalletState();
      if (w && w.address) {
        const a = String(w.address);
        const short = a.length > 12 ? `${a.slice(0, 6)}…${a.slice(-4)}` : a;
        setDockConn(`wallet ${short}${w.chainId != null ? ` (chainId=${w.chainId})` : ""}`);
      } else {
        setDockConn("wallet not connected");
      }
    } catch {
      setDockConn("wallet unknown");
    }
  }

  window.addEventListener("genesis_wallet_changed", refreshDockConn);
  refreshDockConn();
  setDockState(state);

  // Create-page style: FAB exists but is hidden by default.
  if (debugFabEl) debugFabEl.hidden = true;

  const log = makeLogger(debugLinesEl);

  return {
    log,
    debugLinesEl,
    debugDockEl,
    debugFabEl,
    setDockState,
    setDockConn,
    refreshDockConn,
  };
}

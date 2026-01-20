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

import { makeLogger, nowTs, shortAddr } from "./common.js";
import { connectWallet, getWalletState, tryAutoConnect } from "./eth.js";

export function setupNav({ active = null, logElId = "log" } = {}) {
  const logEl = document.getElementById(logElId);
  const log = makeLogger(logEl);

  document.querySelectorAll("[data-nav]").forEach((el) => {
    const is = el.getAttribute("data-nav") === active;
    el.classList.toggle("primary", is);
  });

  const connectBtn = document.getElementById("connectBtn");
  const walletPill = document.getElementById("walletPill");

  function refreshPill() {
    const w = getWalletState();
    if (w?.address) {
      walletPill.textContent = `${shortAddr(w.address)} · chainId=${w.chainId ?? "?"}`;
      connectBtn.textContent = "Connected";
    } else {
      walletPill.textContent = "Read-only (RPC)";
      connectBtn.textContent = "Connect Wallet";
    }
  }

  connectBtn?.addEventListener("click", async () => {
    try {
      await connectWallet(log);
      refreshPill();
    } catch (e) {
      log(`[${nowTs()}] [error] Connect failed: ${e.message || e}`);
    }
  });

  window.addEventListener("genesis_wallet_changed", refreshPill);
  refreshPill();

  // Best-effort: if the site is already authorized in MetaMask, reflect it immediately.
  // No pop-up / no prompt.
  tryAutoConnect(log).finally(() => refreshPill());

  return { log, refreshPill };
}

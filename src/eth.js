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

import { nowTs } from "./common.js";

const ethers = globalThis.ethers;

let wallet = {
  address: null,
  chainId: null
};

export function getWalletState() {
  return { ...wallet };
}

export function getReadProvider(rpcUrl) {
  // Provide explicit network so ethers doesn't try ENS.
  const network = { name: "genesisl1", chainId: 29 };
  return new ethers.JsonRpcProvider(rpcUrl, network);
}

// Attempts to detect an already-authorized MetaMask connection (no prompt).
// This makes the connection feel "persistent" across tabs/pages.
export async function tryAutoConnect(log = () => {}) {
  try {
    if (!window.ethereum) return null;
    const provider = new ethers.BrowserProvider(window.ethereum, { name: "genesisl1", chainId: 29 });

    // eth_accounts returns authorized accounts without popping MetaMask.
    const accounts = await provider.send("eth_accounts", []);
    if (!accounts || !accounts.length) return null;

    const addr = accounts[0];
    const net = await provider.getNetwork();
    wallet.address = addr;
    wallet.chainId = Number(net.chainId);

    // Subscribe once for updates.
    if (!window.__gl1f_wallet_subscribed) {
      window.__gl1f_wallet_subscribed = true;

      window.ethereum.on("accountsChanged", async (accs) => {
        wallet.address = accs && accs.length ? accs[0] : null;
        window.dispatchEvent(new Event("genesis_wallet_changed"));
      });

      window.ethereum.on("chainChanged", async (cidHex) => {
        try { wallet.chainId = parseInt(cidHex, 16); } catch { wallet.chainId = null; }
        window.dispatchEvent(new Event("genesis_wallet_changed"));
      });
    }

    log(`[${nowTs()}] Auto-connected: ${addr} on chainId=${wallet.chainId}`);
    window.dispatchEvent(new Event("genesis_wallet_changed"));
    return { provider, address: addr, chainId: wallet.chainId };
  } catch (e) {
    // Silent fail: auto-connect is best-effort.
    return null;
  }
}

export async function connectWallet(log = () => {}) {
  if (!window.ethereum) throw new Error("MetaMask not found (window.ethereum missing)");

  const provider = new ethers.BrowserProvider(window.ethereum, { name: "genesisl1", chainId: 29 });

  await provider.send("eth_requestAccounts", []);
  const signer = await provider.getSigner();
  const addr = await signer.getAddress();
  const net = await provider.getNetwork();

  wallet.address = addr;
  wallet.chainId = Number(net.chainId);

  log(`[${nowTs()}] Connected: ${addr} on chainId=${wallet.chainId}`);

  if (!window.__gl1f_wallet_subscribed) {
    window.__gl1f_wallet_subscribed = true;

    window.ethereum.on("accountsChanged", async (accs) => {
      wallet.address = accs && accs.length ? accs[0] : null;
      window.dispatchEvent(new Event("genesis_wallet_changed"));
    });

    window.ethereum.on("chainChanged", async (cidHex) => {
      try { wallet.chainId = parseInt(cidHex, 16); } catch { wallet.chainId = null; }
      window.dispatchEvent(new Event("genesis_wallet_changed"));
    });
  }

  window.dispatchEvent(new Event("genesis_wallet_changed"));
  return { provider, signer, address: addr, chainId: wallet.chainId };
}

export async function getSignerProvider() {
  if (!window.ethereum) throw new Error("MetaMask not found");
  const provider = new ethers.BrowserProvider(window.ethereum, { name: "genesisl1", chainId: 29 });
  const signer = await provider.getSigner();
  return { provider, signer };
}

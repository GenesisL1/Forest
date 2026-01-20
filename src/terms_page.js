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

import { setupNav } from "./ui_nav.js";
import { setupDebugDock } from "./debug_dock.js";
import { makeLogger, loadSystem, mustAddr, nowTs, weiToEth } from "./common.js";
import { getReadProvider } from "./eth.js";
import { ABI_REGISTRY, ABI_MARKET } from "./abis.js";

const ethers = globalThis.ethers;

function setKV(el, entries) {
  if (!el) return;
  el.innerHTML = "";
  for (const [k, v] of entries) {
    const kk = document.createElement("div");
    kk.className = "k";
    kk.textContent = k;

    const vv = document.createElement("div");
    vv.className = "v";
    vv.textContent = v;

    el.appendChild(kk);
    el.appendChild(vv);
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  setupNav({ active: "terms", logElId: "debugLines" });
  setupDebugDock({ state: "idle" });
  const log = makeLogger(document.getElementById("debugLines"));

  const licKV = document.getElementById("licKV");
  const feesKV = document.getElementById("feesKV");
  const feesNote = document.getElementById("feesNote");
  const tosKV = document.getElementById("tosKV");
  const tosText = document.getElementById("tosText");

  async function load() {
    const sys = loadSystem();
    if (!sys.rpc || !sys.registry) {
      log(`[${nowTs()}] [error] Missing system config (rpc, registry). Configure contracts in src/common.js defaults or provide deployed contract addresses (see src/common.js defaults).`);
      return;
    }

    const rp = getReadProvider(sys.rpc);
    const registry = new ethers.Contract(mustAddr(sys.registry), ABI_REGISTRY, rp);

    // Active license
    const licId = Number(await registry.activeLicenseId({ gasLimit: 2_000_000_000 }));
    const lic = await registry.getLicense(BigInt(licId), { gasLimit: 2_000_000_000 });
    setKV(licKV, [
      ["License ID", `#${licId}`],
      ["Name", lic[0]],
      ["URL", lic[1]],
    ]);

    // Terms of Service
    const tosVer = Number(await registry.tosVersion({ gasLimit: 2_000_000_000 }));
    const tosHash = await registry.tosHash({ gasLimit: 2_000_000_000 });
    const txt = await registry.tosText({ gasLimit: 2_000_000_000 });
    setKV(tosKV, [["Version", `#${tosVer}`], ["Hash", String(tosHash)]]);
    if (tosText) tosText.textContent = txt;

    // Fees
    try {
      const deployFeeWei = await registry.deployFeeWei({ gasLimit: 2_000_000_000 });
      const sizeFeeWeiPerByte = await registry.sizeFeeWeiPerByte({ gasLimit: 2_000_000_000 });

      let listingFeeWei = null;
      if (sys.market) {
        try {
          const market = new ethers.Contract(mustAddr(sys.market), ABI_MARKET, rp);
          listingFeeWei = await market.listingFeeWei({ gasLimit: 2_000_000_000 });
        } catch {
          listingFeeWei = null;
        }
      }

      setKV(feesKV, [
        ["Registry deploy fee (base)", `${weiToEth(deployFeeWei)} L1`],
        ["Registry size fee", `${weiToEth(sizeFeeWeiPerByte)} L1 / byte`],
        ["Required deploy value", "deployFee + (sizeFeePerByte × modelBytes)"],
        ["Ai store listing fee", listingFeeWei === null ? "—" : `${weiToEth(listingFeeWei)} L1`],
      ]);

      if (feesNote) {
        feesNote.textContent = "Inference fees are set per model by the model owner (see each model page).";
      }
    } catch (e) {
      log(`[${nowTs()}] [warn] Failed to load fees: ${e.message || e}`);
      setKV(feesKV, [["Fees", "Failed to load"]]);
    }
  }

  await load();
});

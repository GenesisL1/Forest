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
import { loadSystem, mustAddr, shortAddr, nowTs, clamp, ethToWei, weiToEth, unpackNftFeatures, sigmoid, taskLabel } from "./common.js";
import { getReadProvider, getSignerProvider, getWalletState, connectWallet } from "./eth.js";
import { ABI_REGISTRY, ABI_MODELNFT, ABI_RUNTIME, ABI_MARKET, ABI_STORE } from "./abis.js";

const ethers = globalThis.ethers;

function qp() { return new URLSearchParams(window.location.search); }

function setKV(el, entries) {
  if (!el) return;
  el.innerHTML = "";
  for (const [k, v] of entries) {
    const kk = document.createElement("div"); kk.className = "k"; kk.textContent = k;
    const vv = document.createElement("div"); vv.className = "v"; vv.textContent = v;
    el.appendChild(kk); el.appendChild(vv);
  }
}

function pricingLabel(enabled, mode, feeWei) {
  if (!enabled) return "Disabled";
  if (mode === 0) return "Free";
  if (mode === 1) return `Tips · ${weiToEth(feeWei)} L1`;
  return `Paid · ${weiToEth(feeWei)} L1`;
}

function packFeaturesQ(vals, scaleQ) {
  const out = new Uint8Array(vals.length * 4);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < vals.length; i++) {
    const q = Math.round(Number(vals[i]) * Number(scaleQ));
    const cl = Math.max(-2147483648, Math.min(2147483647, q));
    dv.setInt32(i*4, cl, true);
  }
  return out;
}

async function bytesToDataUrlPng(bytes) {
  try {
    const u8 = bytes instanceof Uint8Array ? bytes : ethers.getBytes(bytes);
    if (!u8?.length) return null;
    const blob = new Blob([u8], { type: "image/png" });
    return URL.createObjectURL(blob);
  } catch { return null; }
}

document.addEventListener("DOMContentLoaded", async () => {
  setupNav({ active: null, logElId: "debugLines" });
  const dbg = setupDebugDock({ state: "idle" });
  const log = dbg.log;

  // Local formatting / error helpers (kept here so they can use `log`).
  const fmtAddr = (a) => shortAddr(String(a || ""));
  const fmtL1 = (wei) => {
    try {
      return weiToEth(BigInt(wei));
    } catch (_) {
      return String(wei);
    }
  };
  const err = (msg) => log(`[${nowTs()}] [error] ${msg}`);

  const tokenId = Number(qp().get("tokenId") || "0");
  if (!tokenId) {
    log(`[${nowTs()}] [error] Missing tokenId in URL (?tokenId=123)`);
    return;
  }

  // elements
  const titleEl = document.getElementById("title");
  const descEl = document.getElementById("desc");
  const tokenPill = document.getElementById("tokenPill");
  const modePill = document.getElementById("modePill");
  const salePill = document.getElementById("salePill");
  const iconEl = document.getElementById("icon");
  const kvEl = document.getElementById("kv");
  const creatorLink = document.getElementById("creatorLink");
  const ownerAddrLink = document.getElementById("ownerAddrLink");
  const ownerLink = document.getElementById("ownerLink");
  const refreshBtn = document.getElementById("refreshBtn");

  const featNote = document.getElementById("featNote");
  const featList = document.getElementById("featList");

  const inferGrid = document.getElementById("inferGrid");
  const callBtn = document.getElementById("callBtn");
  const txBtn = document.getElementById("txBtn");
  const txBox = document.getElementById("txBox");
  const payAmt = document.getElementById("payAmt");
  const payHint = document.getElementById("payHint");
  const inferKV = document.getElementById("inferKV");

  const marketKV = document.getElementById("marketKV");
  const ownerMarketBox = document.getElementById("ownerMarketBox");
  const buyBox = document.getElementById("buyBox");
  const listPrice = document.getElementById("listPrice");
  const listBtn = document.getElementById("listBtn");
  const cancelBtn = document.getElementById("cancelBtn");
  const buyBtn = document.getElementById("buyBtn");

  const ownerSettings = document.getElementById("ownerSettings");
  const enabledSel = document.getElementById("enabledSel");
  const modeSel = document.getElementById("modeSel");
  const feeSel = document.getElementById("feeSel");
  const recSel = document.getElementById("recSel");
  const saveSettingsBtn = document.getElementById("saveSettingsBtn");
  const burnBtn = document.getElementById("burnBtn");
  let summary = null;
  let features = [];
  let modelMeta = null;
  let modelTask = "regression";
  let modelLabels = null; // for binary classification: [label0, label1]
  let callMode = "view"; // view | ownerSig | none


  // ===== API access / subscription logic (Paid-required models) =====
  let accessCtx = null;
  let accessHandlersWired = false;

  // Cached owner address for the current token, set by loadAll().
  let ownerAddr = null;

  let modelScaleQ = 1_000_000;

  function currentScaleQ() {
    const sq = Number(accessCtx?.scaleQ ?? modelScaleQ ?? 1);
    return (Number.isFinite(sq) && sq > 0) ? Math.floor(sq) : 1;
  }


  function renderRes(res, note = "") {
    const scaleQ = currentScaleQ();
    if (modelTask === "multilabel_classification") {
      // Vector logits per label (Q units). UI applies sigmoid(logitQ/scaleQ).
      let logits = null;
      if (Array.isArray(res)) logits = res;
      else if (res && typeof res === "object") logits = res.logitsQ ?? res[0] ?? null;

      if (!Array.isArray(logits) || logits.length === 0) {
        setKV(inferKV, [["Error", "Invalid multilabel result"], ["Source", note || "—"]]);
        return;
      }

      const K = logits.length;
      const qThr = 40n * BigInt(scaleQ); // sigmoid saturation threshold
      const rows = [];
      const pos = [];
      for (let k = 0; k < K; k++) {
        const q = (typeof logits[k] === "bigint") ? logits[k] : BigInt(logits[k]);
        let p;
        if (q >= qThr) p = 1;
        else if (q <= -qThr) p = 0;
        else p = sigmoid(Number(q) / scaleQ);

        const nm = Array.isArray(modelLabels) ? (modelLabels[k] ?? `y${k}`) : `y${k}`;
        rows.push({ k, nm, q, p });
        if (p >= 0.5) pos.push({ k, nm, p });
      }

      const posShown = pos
        .sort((a, b) => b.p - a.p)
        .slice(0, 12)
        .map(o => `${o.nm} (${o.p.toFixed(3)})`);
      const posStr = pos.length
        ? (posShown.join(", ") + (pos.length > posShown.length ? `, … (+${pos.length - posShown.length})` : ""))
        : "—";

      const topShown = rows
        .slice()
        .sort((a, b) => b.p - a.p)
        .slice(0, 8)
        .map(o => `${o.nm}: ${o.p.toFixed(4)}`)
        .join(", ");

      setKV(inferKV, [
        ["Labels", String(K)],
        ["Predicted positive (p≥0.5)", posStr],
        ["Top probabilities", topShown || "—"],
        ["Source", note || "—"],
      ]);
      return;
    }
    // Multiclass returns (classIndex, bestScoreQ)
    if (modelTask === "multiclass_classification") {
      let cls = null;
      let bestQ = null;
      if (Array.isArray(res) && res.length >= 2) {
        cls = res[0];
        bestQ = res[1];
      } else if (res && typeof res === "object") {
        cls = res.classIndex ?? res[0];
        bestQ = res.bestScoreQ ?? res.bestLogitQ ?? res[1];
      }

      if (cls === null || bestQ === null) {
        setKV(inferKV, [["Error", "Invalid multiclass result"], ["Source", note || "—"]]);
        return;
      }

      const clsNum = (typeof cls === "bigint") ? Number(cls) : Number(cls);
      const q = (typeof bestQ === "bigint") ? bestQ : BigInt(bestQ);
      const bestLogit = Number(q) / scaleQ;
      const lbl = Array.isArray(modelLabels) ? modelLabels[clsNum] : null;
      setKV(inferKV, [
        ["Class", lbl ? `${clsNum} (${lbl})` : String(clsNum)],
        ["Best logit (Q)", q.toString()],
        ["Best logit", bestLogit.toFixed(6)],
        ["Source", note || "—"],
      ]);
      return;
    }

    // v1 scalar return (regression/binary)
    const q = (typeof res === "bigint") ? res : BigInt(res);
    const logitOrValue = Number(q) / scaleQ;

    if (modelTask === "binary_classification") {
      const prob = sigmoid(logitOrValue);
      const pred01 = prob >= 0.5 ? 1 : 0;
      const lbl = Array.isArray(modelLabels) ? modelLabels[pred01] : null;
      setKV(inferKV, [
        ["Logit (Q)", q.toString()],
        ["Logit", logitOrValue.toFixed(6)],
        ["Probability", prob.toFixed(6)],
        ["Predicted", lbl ? `${pred01} (${lbl})` : String(pred01)],
        ["Source", note || "—"],
      ]);
      return;
    }

    setKV(inferKV, [
      ["Result (Q)", q.toString()],
      ["Value", logitOrValue.toFixed(6)],
      ["Source", note || "—"],
    ]);
  }

  async function copyText(text) {
    const t = String(text || "");
    if (!t) return;
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(t);
        return true;
      }
    } catch {
      // fall through
    }
    try {
      const ta = document.createElement("textarea");
      ta.value = t;
      ta.style.position = "fixed";
      ta.style.top = "-1000px";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch {
      return false;
    }
  }

  function _addrEq(a, b) {
    if (!a || !b) return false;
    return String(a).toLowerCase() === String(b).toLowerCase();
  }

  function _fmtDurationBlocks(blocks) {
    const b = Number(blocks);
    if (!Number.isFinite(b) || b <= 0) return String(blocks);
    const seconds = b * 15;
    const days = seconds / 86400;
    if (days >= 1) return `${b.toLocaleString()} blocks (~${days.toFixed(2)} days)`;
    const hours = seconds / 3600;
    if (hours >= 1) return `${b.toLocaleString()} blocks (~${hours.toFixed(2)} hours)`;
    const mins = seconds / 60;
    return `${b.toLocaleString()} blocks (~${mins.toFixed(2)} minutes)`;
  }

  function _renderPlans() {
    if (!accessPlansList || !planSelect) return;
    const plans = accessCtx?.plans || [];
    if (!plans.length) {
      accessPlansList.innerHTML = '<div class="muted">No access plans yet.</div>';
      planSelect.innerHTML = '';
      return;
    }

    accessPlansList.innerHTML = plans.map(p => {
      const priceL1 = fmtL1(p.priceWei);
      const dur = _fmtDurationBlocks(p.durationBlocks);
      const status = p.active ? '<span class="pill">Active</span>' : '<span class="pill">Inactive</span>';
      return `
        <div class="card">
          <div class="row spread">
            <div><b>Plan #${p.planId}</b></div>
            <div>${status}</div>
          </div>
          <div class="kv">
            <div class="k">Duration</div><div class="v mono">${dur}</div>
            <div class="k">Price</div><div class="v mono">${priceL1} L1</div>
          </div>
        </div>
      `;
    }).join("");

    const activePlans = plans.filter(p => p.active);
    planSelect.innerHTML = activePlans.map(p => {
      const priceL1 = fmtL1(p.priceWei);
      const dur = _fmtDurationBlocks(p.durationBlocks);
      return `<option value="${p.planId}">#${p.planId} — ${dur} — ${priceL1} L1</option>`;
    }).join("");
  }

  async function refreshExpiryInfoForKey() {
    if (!expiryInfo) return;
    const keyAddr = buyerKeyAddr?.value?.trim();
    if (!keyAddr || !ethers.isAddress(keyAddr)) {
      expiryInfo.textContent = "";
      return;
    }
    try {
      const exp = await accessCtx.registry.accessExpiry(accessCtx.modelId, keyAddr); // uint64 -> BigInt
      if (exp === 0n) {
        expiryInfo.textContent = "No active access for this key.";
        return;
      }
      const MAX_U64 = (2n ** 64n) - 1n;
      if (exp === MAX_U64) {
        expiryInfo.textContent = "Perpetual access (expiry: max uint64).";
        return;
      }
      const bn = BigInt(await accessCtx.rp.getBlockNumber());
      const remaining = exp > bn ? (exp - bn) : 0n;
      expiryInfo.textContent = `Access expiry: block ${exp.toString()} (≈ ${remaining.toString()} blocks remaining).`;
    } catch (e) {
      console.warn(e);
      expiryInfo.textContent = "Failed to load expiry.";
    }
  }

  async function refreshAccessUI() {
    if (!accessPanel) return;
    if (!accessCtx || accessCtx.pricingMode !== 2) {
      accessPanel.style.display = "none";
      return;
    }
    accessPanel.style.display = "";
    if (ownerPlansBox) ownerPlansBox.style.display = accessCtx.isOwner ? "" : "none";

    // Load plans
    try {
      const cnt = Number(await accessCtx.registry.accessPlanCount(accessCtx.modelId));
      const plans = [];
      for (let i = 1; i <= cnt; i++) {
        const [dur, price, active] = await accessCtx.registry.getAccessPlan(accessCtx.modelId, i);
        plans.push({ planId: i, durationBlocks: Number(dur), priceWei: price, active });
      }
      accessCtx.plans = plans;
    } catch (e) {
      console.warn(e);
      accessCtx.plans = [];
    }

    _renderPlans();
    await refreshExpiryInfoForKey();
  }

  function wireAccessHandlers() {
    if (accessHandlersWired) return;
    accessHandlersWired = true;

    if (buyerKeyAddr) buyerKeyAddr.addEventListener("input", () => {
      refreshExpiryInfoForKey();
    });

    if (genBuyerKeyBtn) genBuyerKeyBtn.addEventListener("click", () => {
      const w = ethers.Wallet.createRandom();
      if (buyerKeyAddr) buyerKeyAddr.value = w.address;
      if (buyerKeyPriv) buyerKeyPriv.value = w.privateKey;
      log("Generated a new buyer API key. Save the private key now.");
      refreshExpiryInfoForKey();
    });

    if (copyBuyerKeyBtn) copyBuyerKeyBtn.addEventListener("click", async () => {
      const v = buyerKeyPriv?.value?.trim();
      if (!v) return;
      const ok = await copyText(v);
      log(ok ? "Copied buyer private key to clipboard." : "Clipboard copy failed — please copy manually.");
    });

    if (buyPlanBtn) buyPlanBtn.addEventListener("click", async () => {
      if (!accessCtx) return;
      const planId = Number(planSelect?.value || 0);
      const keyAddr = buyerKeyAddr?.value?.trim();
      if (!planId) return err("Select a plan.");
      if (!keyAddr || !ethers.isAddress(keyAddr)) return err("Enter a valid API key address.");

      const plan = (accessCtx.plans || []).find(p => p.planId === planId);
      if (!plan) return err("Plan not found.");
      if (!plan.active) return err("Plan is inactive.");

      // Buyer pays once to grant access for a specific API key address.
      let w = getWalletState();
      if (!w?.address) {
        await connectWallet(log);
        w = getWalletState();
      }
      if (!w?.address) return err("Wallet not connected.");
      if (String(w.chainId) !== "29") return err("Switch to GenesisL1 (chainId=29)");

      const { signer } = await getSignerProvider();
      const reg = accessCtx.registry.connect(signer);
      log(`[tx] buyAccess planId=${planId} key=${fmtAddr(keyAddr)} price=${fmtL1(plan.priceWei)} L1`);
      const tx = await reg.buyAccess(accessCtx.modelId, planId, keyAddr, { value: plan.priceWei, gasLimit: 10_000_000 });
      log(`  tx.hash ${tx.hash}`);
      await tx.wait();
      log("✅ Access purchased.");
      await refreshExpiryInfoForKey();
    });

    if (createPlanBtn) createPlanBtn.addEventListener("click", async () => {
      let w = getWalletState();
      if (!w?.address) {
        await connectWallet(log);
        w = getWalletState();
      }
      if (!w?.address) return err("Wallet not connected.");
      if (String(w.chainId) !== "29") return err("Switch to GenesisL1 (chainId=29)");

      accessCtx.isOwner = _addrEq(w.address, accessCtx.owner);
      if (!accessCtx.isOwner) return err("Owner only.");

      const dur = Number(planDuration?.value || 0);
      if (!dur || dur <= 0) return err("Duration must be > 0 blocks.");
      const priceStr = String(planPrice?.value || "").trim();
      if (!priceStr) return err("Price missing.");
      const priceWei = ethers.parseUnits(priceStr, 18);
      const active = String(planActive?.value || "1") === "1";

      const { signer } = await getSignerProvider();
      const reg = accessCtx.registry.connect(signer);
      log(`[tx] createAccessPlan duration=${dur} price=${priceStr} active=${active}`);
      const tx = await reg.createAccessPlan(accessCtx.modelId, dur, priceWei, active, { gasLimit: 12_000_000 });
      log(`  tx.hash ${tx.hash}`);
      await tx.wait();
      log("✅ Plan created.");
      await refreshAccessUI();
    });

    if (setPlanBtn) setPlanBtn.addEventListener("click", async () => {
      let w = getWalletState();
      if (!w?.address) {
        await connectWallet(log);
        w = getWalletState();
      }
      if (!w?.address) return err("Wallet not connected.");
      if (String(w.chainId) !== "29") return err("Switch to GenesisL1 (chainId=29)");

      accessCtx.isOwner = _addrEq(w.address, accessCtx.owner);
      if (!accessCtx.isOwner) return err("Owner only.");

      const planId = Number(editPlanId?.value || 0);
      if (!planId) return err("Plan ID missing.");
      const dur = Number(editPlanDuration?.value || 0);
      if (!dur || dur <= 0) return err("Duration must be > 0 blocks.");
      const priceStr = String(editPlanPrice?.value || "").trim();
      if (!priceStr) return err("Price missing.");
      const priceWei = ethers.parseUnits(priceStr, 18);
      const active = String(editPlanActive?.value || "1") === "1";

      const { signer } = await getSignerProvider();
      const reg = accessCtx.registry.connect(signer);
      log(`[tx] setAccessPlan id=${planId} duration=${dur} price=${priceStr} active=${active}`);
      const tx = await reg.setAccessPlan(accessCtx.modelId, planId, dur, priceWei, active, { gasLimit: 12_000_000 });
      log(`  tx.hash ${tx.hash}`);
      await tx.wait();
      log("✅ Plan updated.");
      await refreshAccessUI();
    });

    if (genOwnerKey2Btn) genOwnerKey2Btn.addEventListener("click", () => {
      const w = ethers.Wallet.createRandom();
      if (ownerSetKeyAddr) ownerSetKeyAddr.value = w.address;
      if (ownerKey2Priv) ownerKey2Priv.value = w.privateKey;
      log("Generated a new owner API key. Save the private key now, then click 'Grant perpetual access'.");
    });

    if (copyOwnerKey2Btn) copyOwnerKey2Btn.addEventListener("click", async () => {
      const v = ownerKey2Priv?.value?.trim();
      if (!v) return;
      const ok = await copyText(v);
      log(ok ? "Copied owner private key to clipboard." : "Clipboard copy failed — please copy manually.");
    });

    if (setOwnerKeyBtn) setOwnerKeyBtn.addEventListener("click", async () => {
      const keyAddr = ownerSetKeyAddr?.value?.trim();
      if (!keyAddr || !ethers.isAddress(keyAddr)) return err("Enter a valid key address.");
      let w = getWalletState();
      if (!w?.address) {
        await connectWallet(log);
        w = getWalletState();
      }
      if (!w?.address) return err("Wallet not connected.");
      if (String(w.chainId) !== "29") return err("Switch to GenesisL1 (chainId=29)");

      accessCtx.isOwner = _addrEq(w.address, accessCtx.owner);
      if (!accessCtx.isOwner) return err("Owner only.");

      const { signer } = await getSignerProvider();
      const reg = accessCtx.registry.connect(signer);
      log(`[tx] setOwnerAccessKey key=${fmtAddr(keyAddr)}`);
      const tx = await reg.setOwnerAccessKey(accessCtx.modelId, keyAddr, { gasLimit: 12_000_000 });
      log(`  tx.hash ${tx.hash}`);
      await tx.wait();
      log("✅ Perpetual access granted.");
      // Optional: update expiry info if buyerKeyAddr equals this
      await refreshExpiryInfoForKey();
    });

    if (keyCallBtn) keyCallBtn.addEventListener("click", async () => {
      if (!accessCtx) return;
      const pk = keyPriv?.value?.trim();
      if (!pk) return err("Enter a private key.");
      if (pk.length < 10) return err("Private key looks too short.");

      // Pack current inputs from the main inference grid
      const packed = packInputsToBytes();
      const deadline = Math.floor(Date.now() / 1000) + 60;

      const wallet = new ethers.Wallet(pk);
      const net = await accessCtx.rp.getNetwork();
      const domain = {
        name: "GenesisL1 Forest",
        version: "1",
        chainId: Number(net.chainId),
        verifyingContract: accessCtx.runtime.target,
      };
      const types = {
        AccessView: [
          { name: "modelId", type: "bytes32" },
          { name: "packedHash", type: "bytes32" },
          { name: "deadline", type: "uint256" },
        ],
      };
      const value = {
        modelId: accessCtx.modelId,
        packedHash: ethers.keccak256(packed),
        deadline,
      };
      const sig = await wallet.signTypedData(domain, types, value);

      if (modelTask === "multiclass_classification") {
        log(`[call] runtime.predictClassAccessView (eth_call) modelId=${accessCtx.modelId} bytes=${packed.length}`);
        const res = await accessCtx.runtime.predictClassAccessView(accessCtx.modelId, packed, deadline, sig, { gasLimit: 2_000_000_000 });
        renderRes(res, "API key (read call)");
        const cls = Number(res[0]);
        log(`API key predict: class=${cls}`);
      } else if (modelTask === "multilabel_classification") {
        log(`[call] runtime.predictMultiAccessView (eth_call) modelId=${accessCtx.modelId} bytes=${packed.length}`);
        const res = await accessCtx.runtime.predictMultiAccessView(accessCtx.modelId, packed, deadline, sig, { gasLimit: 2_000_000_000 });
        renderRes(res, "API key (read call)");
        const k = Array.isArray(res) ? res.length : (res?.logitsQ?.length || res?.[0]?.length || 0);
        log(`API key predict: logitsK=${k}`);
      } else {
        log(`[call] runtime.predictAccessView (eth_call) modelId=${accessCtx.modelId} bytes=${packed.length}`);
        const scoreQ = await accessCtx.runtime.predictAccessView(accessCtx.modelId, packed, deadline, sig, { gasLimit: 2_000_000_000 });
        renderRes(scoreQ, "API key (read call)");
        const val = Number(scoreQ) / Number(accessCtx.scaleQ || 1);
        log(`API key predict: scoreQ=${scoreQ} value=${val}`);
      }
    });
  }

  // Pack feature inputs into bytes using the model metadata in accessCtx (scaleQ, nFeatures).
  function packInputsToBytes() {
    // Re-use the same input read path as tx / call inference
    const inputs = inferGrid.querySelectorAll("input");
    const arr = [];
    for (let i = 0; i < inputs.length; i++) {
      const s = (inputs[i].value || "").trim();
      const x = s === "" ? 0 : Number(s);
      arr.push(x);
    }
    const packed = packFeaturesQ(arr, accessCtx?.scaleQ || 1);
    return packed;
  }
  async function loadAll() {
    const sys = loadSystem();
    if (!sys.rpc || !sys.registry || !sys.nft || !sys.runtime) {
      log(`[${nowTs()}] [error] Missing system config (rpc/registry/nft/runtime).`);
      return;
    }

    const rp = getReadProvider(sys.rpc);
    const registry = new ethers.Contract(mustAddr(sys.registry), ABI_REGISTRY, rp);
    const nft = new ethers.Contract(mustAddr(sys.nft), ABI_MODELNFT, rp);
    const runtime = new ethers.Contract(mustAddr(sys.runtime), ABI_RUNTIME, rp);
    const market = sys.market ? new ethers.Contract(mustAddr(sys.market), ABI_MARKET, rp) : null;

    // model info
    summary = await registry.getModelSummary(BigInt(tokenId), { gasLimit: 2_000_000_000 });
    if (!summary[0]) {
      log(`[${nowTs()}] [error] Model not found in registry for tokenId=${tokenId}`);
      return;
    }
    const modelId = summary[1];

    // Load the model's quantization scale (scaleQ) from the registry runtime data.
    // This keeps input packing and output decoding consistent for models trained on
    // different-magnitude datasets.
    try {
      const rt = await registry.getModelRuntime(modelId, { gasLimit: 2_000_000_000 });
      const sq = Number(rt[8]);
      if (Number.isFinite(sq) && sq > 0) modelScaleQ = Math.floor(sq);
    } catch {
      // Fallback to default (legacy / older deployments)
      modelScaleQ = modelScaleQ || 1_000_000;
    }
    const nFeatures = Number(summary[3]);
    const nTrees = Number(summary[4]);
    const depth = Number(summary[5]);
    const baseQ = Number(summary[6]);
    const pricingMode = Number(summary[7]);
    const feeWei = BigInt(summary[8]);
    const feeRecipient = summary[9];
    const enabled = !!summary[10];
    const creator = summary[11];
    const tosVer = Number(summary[12]);
    const title = summary[13];
    const desc = summary[14];

    const owner = await nft.ownerOf(BigInt(tokenId), { gasLimit: 2_000_000_000 });
    // Cache owner for later actions (tx inference, etc.)
    ownerAddr = owner;

    titleEl.textContent = title || `Model #${tokenId}`;
    descEl.textContent = desc || "";
    tokenPill.textContent = `NFT ID: ${tokenId}`;
    modePill.textContent = `Inference: ${pricingLabel(enabled, pricingMode, feeWei)}`;
    creatorLink.textContent = creator;
    creatorLink.href = `./forest.html?creator=${encodeURIComponent(creator)}`;
    ownerAddrLink.textContent = owner;
    ownerAddrLink.href = `./forest.html?owner=${encodeURIComponent(owner)}`;
    ownerLink.href = `./forest.html?owner=${encodeURIComponent(owner)}`;

    // icon + labels
    try {
      const iconBytes = await nft.icon(BigInt(tokenId), { gasLimit: 2_000_000_000 });
      const url = await bytesToDataUrlPng(iconBytes);
      if (url) iconEl.src = url;
    } catch {}

    try {
      const ftxt = await nft.features(BigInt(tokenId), { gasLimit: 2_000_000_000 });
      const unpacked = unpackNftFeatures(ftxt);
      modelMeta = unpacked.meta;
      features = unpacked.features;
      const t = String(modelMeta?.task || "regression").trim();
      if (t === "binary_classification" || t === "binary" || t === "classification") {
        modelTask = "binary_classification";
      } else if (t === "multiclass_classification" || t === "multiclass") {
        modelTask = "multiclass_classification";
      } else if (t === "multilabel_classification" || t === "multilabel") {
        modelTask = "multilabel_classification";
      } else {
        modelTask = "regression";
      }
      // For binary/multiclass: meta.labels are class labels.
      // For multilabel: prefer meta.labelNames (output label names), else fall back to meta.labels (legacy v1).
      if (modelTask === "multilabel_classification") {
        modelLabels = Array.isArray(modelMeta?.labelNames)
          ? modelMeta.labelNames
          : (Array.isArray(modelMeta?.labels) ? modelMeta.labels : null);
      } else {
        modelLabels = Array.isArray(modelMeta?.labels) ? modelMeta.labels : null;
      }
    } catch {
      features = [];
      modelMeta = null;
      modelTask = "regression";
      modelLabels = null;
    }

    featList.innerHTML = "";
    if (features.length) {
      let note = `Stored on-chain in the NFT (${features.length} features). Task: ${taskLabel(modelTask)}`;
      if (modelTask === "binary_classification" && Array.isArray(modelLabels) && modelLabels.length >= 2) {
        note += ` · Labels: 0="${modelLabels[0]}", 1="${modelLabels[1]}"`;
      } else if (modelTask === "multiclass_classification" && Array.isArray(modelLabels) && modelLabels.length >= 2) {
        const shown = modelLabels.slice(0, 6).join(", ") + (modelLabels.length > 6 ? ", …" : "");
        note += ` · Classes: K=${modelLabels.length} [${shown}]`;
      } else if (modelTask === "multilabel_classification" && Array.isArray(modelLabels) && modelLabels.length >= 1) {
        const shown = modelLabels.slice(0, 6).join(", ") + (modelLabels.length > 6 ? ", …" : "");
        note += ` · Labels: K=${modelLabels.length} [${shown}]`;
      }
      featNote.textContent = note + ":";
      for (const f of features) {
        const div = document.createElement("div");
        div.textContent = f;
        featList.appendChild(div);
      }
    } else {
      featNote.textContent = "No feature labels found (NFT features string empty). Inputs fall back to f0,f1,…";
    }

    // details kv
    const kvEntries = [
      ["Task", taskLabel(modelTask)],
      ["Model ID", String(modelId)],
      ["Trees", String(nTrees)],
      ["Depth", String(depth)],
      ["Features", String(nFeatures)],
      ["BaseQ", String(baseQ)],
    ];
    if (modelTask === "binary_classification" && Array.isArray(modelLabels) && modelLabels.length >= 2) {
      kvEntries.push(["Labels", `0=${modelLabels[0]} · 1=${modelLabels[1]}`]);
    } else if (modelTask === "multiclass_classification" && Array.isArray(modelLabels) && modelLabels.length >= 2) {
      kvEntries.push(["Classes", String(modelLabels.length)]);
      kvEntries.push(["Class labels", modelLabels.slice(0, 8).join(", ") + (modelLabels.length > 8 ? ", …" : "")]);
    } else if (modelTask === "multilabel_classification" && Array.isArray(modelLabels) && modelLabels.length >= 1) {
      kvEntries.push(["Labels", String(modelLabels.length)]);
      kvEntries.push(["Label names", modelLabels.slice(0, 8).join(", ") + (modelLabels.length > 8 ? ", …" : "")]);
    }
    kvEntries.push(["Fee recipient", feeRecipient]);
    kvEntries.push(["ToS accepted", `v${tosVer}`]);
    setKV(kvEl, kvEntries);

    // inference inputs
    // Build labeled feature inputs (labels appear above each input).
    inferGrid.innerHTML = "";
    const labels = features.length ? features : Array.from({length: nFeatures}, (_,i)=>`f${i}`);
    labels.forEach((nm,i)=>{
      const cell = document.createElement("div");
      cell.className = "featCell";
      const lab = document.createElement("div");
      lab.className = "featName";
      lab.textContent = nm;
      const inp = document.createElement("input");
      inp.type="number";
      inp.step="any";
      inp.value="";
      inp.title = nm;
      inp.dataset.idx = String(i);
      cell.appendChild(lab);
      cell.appendChild(inp);
      inferGrid.appendChild(cell);
    });

    // Inference controls
    const w = getWalletState();
    const connected = !!w.address;
    const isOwner = connected && ownerAddr && w.address.toLowerCase() === ownerAddr.toLowerCase();

    // Access key / subscription context (Paid-required models)
    accessCtx = {
      registry,
      runtime,
      rp,
      modelId,
      pricingMode,
      owner: ownerAddr,
      isOwner,
      scaleQ: modelScaleQ,
      plans: []
    };
    wireAccessHandlers();
    await refreshAccessUI();

    // Reset defaults
    callBtn.disabled = false;
    callBtn.textContent = "On-chain Predict (read call)";
    callBtn.style.display = "inline-flex";

    // Pricing modes: 0=FREE, 1=TIP_OPTIONAL, 2=PAID_REQUIRED
    callMode = "view";
    if (!enabled) {
      callMode = "none";
      callBtn.disabled = true;
      callBtn.textContent = "Inference disabled";
      callBtn.style.display = "inline-flex";
      txBtn.style.display = "none";
      txBox.style.display = "none";
      payHint.textContent = "Inference is disabled for this model.";
    } else if (pricingMode === 2) {
      // Paid-required models: prevent fee bypass via read-call. Non-owners must use a paid tx.
      if (isOwner) {
        callMode = "ownerSig";
        callBtn.style.display = "inline-flex";
        callBtn.disabled = !connected;
        callBtn.textContent = "Owner Predict (no tx)";
      } else {
        callMode = "none";
        callBtn.style.display = "none";
      }

      txBox.style.display = "block";
      txBtn.style.display = connected ? "inline-flex" : "none";
      txBtn.textContent = isOwner ? "On-chain Predict (tx, optional tip)" : "On-chain Predict (paid tx)";
      payHint.textContent = !connected
        ? `Paid inference required. Fee: ${weiToEth(feeWei)} L1 per prediction. Connect a wallet to run on-chain inference.`
        : (isOwner
            ? `Owner detected. Use "Owner Predict (no tx)" for free read-only inference, or send an optional tip via tx.`
            : `Paid inference required. Fee: ${weiToEth(feeWei)} L1 per prediction.`);
      payAmt.value = isOwner ? "0" : weiToEth(feeWei);
    } else if (pricingMode === 1) {
      // Tips mode: free read-call inference is allowed; tx lets you tip
      callMode = "view";
      callBtn.style.display = "inline-flex";
      callBtn.textContent = "On-chain Predict (free read call)";
      txBox.style.display = connected ? "block" : "none";
      txBtn.style.display = connected ? "inline-flex" : "none";
      payHint.textContent = connected
        ? `Tips mode. You may optionally tip the owner (default ${weiToEth(feeWei)} L1).`
        : "Tips mode. Connect a wallet to send an optional tip.";
      payAmt.value = weiToEth(feeWei);
    } else {
      // Free
      callMode = "view";
      callBtn.style.display = "inline-flex";
      callBtn.textContent = "On-chain Predict (free read call)";
      txBtn.style.display = "none";
      txBox.style.display = "none";
      payHint.textContent = "Free inference (read call).";
    }

    // Ai store
    if (market) {
      try {
        const li = await market.getListing(BigInt(tokenId), { gasLimit: 2_000_000_000 });
        const listed = !!li[0];
        const priceWei = BigInt(li[1]);
        const seller = li[2];

        salePill.textContent = listed ? `Sale: Listed (${weiToEth(priceWei)} L1)` : "Sale: Not listed";
        setKV(marketKV, [
          ["Listed", String(listed)],
          ["Price (L1)", listed ? weiToEth(priceWei) : "—"],
          ["Seller", listed ? seller : "—"]
        ]);

        const isOwner = connected && ethers.getAddress(owner) === ethers.getAddress(w.address);
        ownerMarketBox.style.display = isOwner ? "block" : "none";
        ownerSettings.style.display = isOwner ? "block" : "none";

        if (listed) {
          // buyer visible if not owner
          buyBox.style.display = (!isOwner && connected) ? "block" : "none";
        } else {
          buyBox.style.display = "none";
        }

        // prefill owner settings
        if (isOwner) {
          enabledSel.value = enabled ? "1" : "0";
          modeSel.value = String(pricingMode);
          feeSel.value = pricingMode === 0 ? "0.001" : weiToEth(feeWei);
          recSel.value = feeRecipient && feeRecipient !== ethers.ZeroAddress ? feeRecipient : w.address;
        }
      } catch (e) {
        salePill.textContent = "Sale: (error)";
        setKV(marketKV, [["Error", e.message || String(e)]]);
      }
    } else {
      salePill.textContent = "Sale: (market not configured)";
      setKV(marketKV, [["Ai store", "Not configured in System"]]);
    }

    // reset inference kv
    if (modelTask === "multiclass_classification") {
      setKV(inferKV, [["Class", "—"], ["Best logit", "—"], ["Source", "—"]]);
    } else if (modelTask === "multilabel_classification") {
      setKV(inferKV, [["Predicted positive (p≥0.5)", "—"], ["Top probabilities", "—"], ["Source", "—"]]);
    } else if (modelTask === "binary_classification") {
      setKV(inferKV, [["Logit (Q)", "—"], ["Probability", "—"], ["Predicted", "—"], ["Source", "—"]]);
    } else {
      setKV(inferKV, [["Result (Q)", "—"], ["Value", "—"], ["Source", "—"]]);
    }
  }

  async function runCall() {
    if (callMode === "none") {
      throw new Error("Read-call inference is not available for this model.");
    }

    const sys = loadSystem();
    const rp = getReadProvider(sys.rpc);
    const runtime = new ethers.Contract(mustAddr(sys.runtime), ABI_RUNTIME, rp);

    const modelId = summary[1];
    const scaleQ = currentScaleQ();
    const vals = Array.from(inferGrid.querySelectorAll("input[data-idx]"))
      .sort((a, b) => Number(a.dataset.idx) - Number(b.dataset.idx))
      .map((el) => Number(el.value || "0"));
    const packed = packFeaturesQ(vals, scaleQ);

    if (callMode === "ownerSig") {
      const w = getWalletState();
      if (!w.address) throw new Error("Connect wallet to run owner view inference.");
      // Request an EIP-712 signature from the current owner (no transaction).
      const { signer } = await getSignerProvider();
      const deadline = Math.floor(Date.now() / 1000) + 300;
      const packedHash = ethers.keccak256(packed);

      const domain = {
        name: "GenesisL1 Forest",
        version: "1",
        chainId: Number(w.chainId || 29),
        verifyingContract: runtime.target,
      };
      const types = {
        OwnerView: [
          { name: "modelId", type: "bytes32" },
          { name: "packedHash", type: "bytes32" },
          { name: "deadline", type: "uint256" },
        ],
      };
      const value = { modelId, packedHash, deadline };

      log(`[${nowTs()}] [sig] OwnerView: modelId=${modelId} deadline=${deadline}`);
      const signature = await signer.signTypedData(domain, types, value);

      if (modelTask === "multiclass_classification") {
        log(`[${nowTs()}] [call] runtime.predictClassOwnerView (eth_call) modelId=${modelId} packedBytes=${packed.length}`);
        const res = await runtime.predictClassOwnerView(modelId, packed, deadline, signature, {
          gasLimit: 2_000_000_000,
        });
        renderRes(res, "Owner view (read call)");
        const cls = Number(res[0]);
        log(`[${nowTs()}] Owner view predict: class=${cls}`);
        return;
      } else if (modelTask === "multilabel_classification") {
        log(`[${nowTs()}] [call] runtime.predictMultiOwnerView (eth_call) modelId=${modelId} packedBytes=${packed.length}`);
        const res = await runtime.predictMultiOwnerView(modelId, packed, deadline, signature, {
          gasLimit: 2_000_000_000,
        });
        renderRes(res, "Owner view (read call)");
        const k = Array.isArray(res) ? res.length : (res?.logitsQ?.length || res?.[0]?.length || 0);
        log(`[${nowTs()}] Owner view predict: logitsK=${k}`);
        return;
      } else {
        log(`[${nowTs()}] [call] runtime.predictOwnerView (eth_call) modelId=${modelId} packedBytes=${packed.length}`);
        const res = await runtime.predictOwnerView(modelId, packed, deadline, signature, {
          gasLimit: 2_000_000_000,
        });
        const scoreQ = BigInt(res.toString());
        renderRes(scoreQ, "Owner view (read call)");
        const valueNum = Number(scoreQ) / scaleQ;
        log(`[${nowTs()}] Owner view predict: scoreQ=${scoreQ.toString()} value=${valueNum}`);
        return;
      }
    }

    // Default free read-call path
    if (modelTask === "multiclass_classification") {
      log(`[${nowTs()}] [call] runtime.predictClassView (eth_call) modelId=${modelId} packedBytes=${packed.length}`);
      const res = await runtime.predictClassView(modelId, packed, { gasLimit: 2_000_000_000 });
      renderRes(res, "Read call");
      const cls = Number(res[0]);
      log(`[${nowTs()}] On-chain predictClassView: class=${cls}`);
    } else if (modelTask === "multilabel_classification") {
      log(`[${nowTs()}] [call] runtime.predictMultiView (eth_call) modelId=${modelId} packedBytes=${packed.length}`);
      const res = await runtime.predictMultiView(modelId, packed, { gasLimit: 2_000_000_000 });
      renderRes(res, "Read call");
      const k = Array.isArray(res) ? res.length : (res?.logitsQ?.length || res?.[0]?.length || 0);
      log(`[${nowTs()}] On-chain predictMultiView: logitsK=${k}`);
    } else {
      log(`[${nowTs()}] [call] runtime.predictView (eth_call) modelId=${modelId} packedBytes=${packed.length}`);
      const res = await runtime.predictView(modelId, packed, { gasLimit: 2_000_000_000 });
      const scoreQ = BigInt(res.toString());
      renderRes(scoreQ, "Read call");
      const valueNum = Number(scoreQ) / scaleQ;
      log(`[${nowTs()}] On-chain predictView: scoreQ=${scoreQ.toString()} value=${valueNum}`);
    }
  }

  async function runTx() {
    const sys = loadSystem();
    const w = getWalletState();
    if (!w.address) throw new Error("Connect wallet first");
    if (String(w.chainId) !== "29") throw new Error("Switch to GenesisL1 (chainId=29)");
    const isOwner = ownerAddr && w.address && w.address.toLowerCase() === ownerAddr.toLowerCase();

    const { signer } = await getSignerProvider();
    const runtime = new ethers.Contract(mustAddr(sys.runtime), ABI_RUNTIME, signer);

    const pricingMode = Number(summary[7]);
    const feeWei = BigInt(summary[8]);
    const scaleQ = currentScaleQ();
    const vals = Array.from(inferGrid.querySelectorAll("input")).map(i => Number(i.value));
    const packed = packFeaturesQ(vals, scaleQ);

    let pay = 0n;
    if (pricingMode === 2) {
      if (isOwner) {
        // Owner can run fee-free inference (gas still applies). Allow optional tip via payAmt.
        pay = ethToWei(String(Math.max(0, Number(payAmt.value || "0"))));
      } else {
        // Non-owner must pay at least the model fee.
        pay = ethToWei(String(Math.max(Number(weiToEth(feeWei)), Number(payAmt.value || "0"))));
      }
    } else if (pricingMode === 1) {
      // Tips are optional and clamped client-side.
      pay = ethToWei(String(clamp(payAmt.value || "0", 0, 1)));
    }

    const mid = summary[1];
    if (modelTask === "multiclass_classification") {
      log(`[${nowTs()}] [tx] runtime.predictClassTx value=${weiToEth(pay)} L1`);
      const tx = await runtime.predictClassTx(mid, packed, { value: pay, gasLimit: 35_000_000 });
      log(`  tx.hash ${tx.hash}`);
      const rcpt = await tx.wait();
      log(`  mined status=${rcpt.status}`);
      if (rcpt.status !== 1) throw new Error("predictClassTx reverted");

      // Read result from tx logs
      try {
        const iface = new ethers.Interface(ABI_RUNTIME);
        for (const lg of rcpt.logs || []) {
          if ((lg.address || "").toLowerCase() !== runtime.target.toLowerCase()) continue;
          let parsed;
          try { parsed = iface.parseLog(lg); } catch (_) { continue; }
          if (!parsed || parsed.name !== "InferenceClass") continue;
          const evMid = parsed.args[0];
          const classIndex = parsed.args[2];
          const bestScoreQ = parsed.args[3];
          if (String(evMid).toLowerCase() !== String(mid).toLowerCase()) continue;
          log(`  event InferenceClass: class=${classIndex} bestScoreQ=${bestScoreQ}`);
          renderRes([classIndex, bestScoreQ], "Transaction");
          return;
        }
        log(`  (no InferenceClass event found in receipt; check explorer logs)`);
      } catch (e) {
        log(`[${nowTs()}] [warn] Could not parse InferenceClass event: ${e.message || e}`);
      }
      return;
    }

    if (modelTask === "multilabel_classification") {
      log(`[${nowTs()}] [tx] runtime.predictMultiTx value=${weiToEth(pay)} L1`);
      const tx = await runtime.predictMultiTx(mid, packed, { value: pay, gasLimit: 35_000_000 });
      log(`  tx.hash ${tx.hash}`);
      const rcpt = await tx.wait();
      log(`  mined status=${rcpt.status}`);
      if (rcpt.status !== 1) throw new Error("predictMultiTx reverted");

      // Read result from tx logs
      try {
        const iface = new ethers.Interface(ABI_RUNTIME);
        for (const lg of rcpt.logs || []) {
          if ((lg.address || "").toLowerCase() !== runtime.target.toLowerCase()) continue;
          let parsed;
          try { parsed = iface.parseLog(lg); } catch (_) { continue; }
          if (!parsed || parsed.name !== "InferenceMulti") continue;
          const evMid = parsed.args[0];
          const logitsQ = parsed.args[2];
          if (String(evMid).toLowerCase() !== String(mid).toLowerCase()) continue;
          const k = Array.isArray(logitsQ) ? logitsQ.length : (logitsQ?.length || 0);
          log(`  event InferenceMulti: logitsK=${k}`);
          renderRes(logitsQ, "Transaction");
          return;
        }
        log(`  (no InferenceMulti event found in receipt; check explorer logs)`);
      } catch (e) {
        log(`[${nowTs()}] [warn] Could not parse InferenceMulti event: ${e.message || e}`);
      }
      return;
    }

    log(`[${nowTs()}] [tx] runtime.predictTx value=${weiToEth(pay)} L1`);
    const tx = await runtime.predictTx(mid, packed, { value: pay, gasLimit: 35_000_000 });
    log(`  tx.hash ${tx.hash}`);
    const rcpt = await tx.wait();
    log(`  mined status=${rcpt.status}`);
    if (rcpt.status !== 1) throw new Error("predictTx reverted");

    // Read result from tx logs (predictView may be disabled for paid models)
    try {
      const iface = new ethers.Interface(ABI_RUNTIME);
      for (const lg of rcpt.logs || []) {
        if ((lg.address || "").toLowerCase() !== runtime.target.toLowerCase()) continue;
        let parsed;
        try { parsed = iface.parseLog(lg); } catch (_) { continue; }
        if (!parsed || parsed.name !== "Inference") continue;
        const evMid = parsed.args[0];
        const scoreQ = parsed.args[2];
        if (String(evMid).toLowerCase() !== String(mid).toLowerCase()) continue;
        log(`  event Inference: scoreQ=${scoreQ}`);
        renderRes(scoreQ, "Transaction");
        return;
      }
      log(`  (no Inference event found in receipt; check explorer logs)`);
    } catch (e) {
      log(`[${nowTs()}] [warn] Could not parse Inference event: ${e.message || e}`);
    }
  }

  callBtn.addEventListener("click", async () => {
    try { await runCall(); } catch (e) { log(`[${nowTs()}] [error] ${e.message || e}`); }
  });
  txBtn.addEventListener("click", async () => {
    try { await runTx(); } catch (e) { log(`[${nowTs()}] [error] ${e.message || e}`); }
  });

  // Ai store actions
  async function listForSale() {
    const sys = loadSystem();
    const w = getWalletState();
    if (!w.address) throw new Error("Connect wallet");
    if (String(w.chainId) !== "29") throw new Error("Switch to chainId=29");
    if (!sys.market || !sys.nft) throw new Error("Ai store not configured");

    const { signer } = await getSignerProvider();
    const marketAddr = mustAddr(sys.market);
    const market = new ethers.Contract(marketAddr, ABI_MARKET, signer);
    const nft = new ethers.Contract(mustAddr(sys.nft), ABI_MODELNFT, signer);

    const priceFloat = Number(listPrice.value || "0");
    if (!Number.isFinite(priceFloat) || priceFloat <= 0) throw new Error("Enter a price > 0");
    const priceWei = ethToWei(String(priceFloat));

    // Ensure the Ai store is approved to transfer this token.
    const owner = await nft.ownerOf(BigInt(tokenId));
    if (owner.toLowerCase() !== w.address.toLowerCase()) throw new Error("You are not the current owner");

    const approved = await nft.getApproved(BigInt(tokenId));
    const approvedForAll = await nft.isApprovedForAll(owner, marketAddr);
    if (!approvedForAll && approved.toLowerCase() !== marketAddr.toLowerCase()) {
      log(`[${nowTs()}] [tx] approve Ai store for tokenId=${tokenId}`);
      const txA = await nft.approve(marketAddr, BigInt(tokenId), { gasLimit: 1_000_000 });
      log(`  tx.hash ${txA.hash}`);
      const rA = await txA.wait();
      log(`  mined status=${rA.status}`);
      if (rA.status !== 1) throw new Error("approve reverted");
    }

    const fee = BigInt(await market.listingFeeWei());

    // Preflight for a useful revert reason before sending a transaction.
    try {
      await market.list.staticCall(BigInt(tokenId), priceWei, { value: fee });
    } catch (e) {
      throw new Error(`List would revert: ${prettyEthersError(e)}`);
    }

    log(`[${nowTs()}] [tx] list tokenId=${tokenId} price=${weiToEth(priceWei)} listingFee=${weiToEth(fee)}`);
    const tx = await market.list(BigInt(tokenId), priceWei, { value: fee, gasLimit: 2_000_000 });
    log(`  tx.hash ${tx.hash}`);
    const rcpt = await tx.wait();
    log(`  mined status=${rcpt.status}`);
    if (rcpt.status !== 1) throw new Error("list reverted");
  }

  async function cancelListing() {
    const sys = loadSystem();
    const w = getWalletState();
    if (!w.address) throw new Error("Connect wallet");
    if (String(w.chainId) !== "29") throw new Error("Switch to chainId=29");
    const { signer } = await getSignerProvider();
    const market = new ethers.Contract(mustAddr(sys.market), ABI_MARKET, signer);
    log(`[${nowTs()}] [tx] cancel listing tokenId=${tokenId}`);
    const tx = await market.cancel(BigInt(tokenId), { gasLimit: 10_000_000 });
    log(`  tx.hash ${tx.hash}`);
    const rcpt = await tx.wait();
    log(`  mined status=${rcpt.status}`);
    if (rcpt.status !== 1) throw new Error("cancel reverted");
  }

  async function buyNow() {
    const sys = loadSystem();
    const w = getWalletState();
    if (!w.address) throw new Error("Connect wallet");
    if (String(w.chainId) !== "29") throw new Error("Switch to chainId=29");
    const rp = getReadProvider(sys.rpc);
    const marketR = new ethers.Contract(mustAddr(sys.market), ABI_MARKET, rp);
    const li = await marketR.getListing(BigInt(tokenId), { gasLimit: 2_000_000_000 });
    if (!li[0]) throw new Error("Not listed");
    const priceWei = BigInt(li[1]);
    const { signer } = await getSignerProvider();
    const market = marketR.connect(signer);
    log(`[${nowTs()}] [tx] buy tokenId=${tokenId} value=${weiToEth(priceWei)}`);
    const tx = await market.buy(BigInt(tokenId), { value: priceWei, gasLimit: 20_000_000 });
    log(`  tx.hash ${tx.hash}`);
    const rcpt = await tx.wait();
    log(`  mined status=${rcpt.status}`);
    if (rcpt.status !== 1) throw new Error("buy reverted");
  }

  listBtn.addEventListener("click", async ()=>{ try { await listForSale(); await loadAll(); } catch(e){ log(`[${nowTs()}] [error] ${e.message||e}`);} });
  cancelBtn.addEventListener("click", async ()=>{ try { await cancelListing(); await loadAll(); } catch(e){ log(`[${nowTs()}] [error] ${e.message||e}`);} });
  buyBtn.addEventListener("click", async ()=>{ try { await buyNow(); await loadAll(); } catch(e){ log(`[${nowTs()}] [error] ${e.message||e}`);} });

  // owner settings
  saveSettingsBtn.addEventListener("click", async () => {
    try {
      const sys = loadSystem();
      const w = getWalletState();
      if (!w.address) throw new Error("Connect wallet");
      if (String(w.chainId) !== "29") throw new Error("Switch to chainId=29");

      const { signer } = await getSignerProvider();
      const registry = new ethers.Contract(mustAddr(sys.registry), ABI_REGISTRY, signer);

      const enabled = enabledSel.value === "1";
      const mode = Number(modeSel.value);
      const fee = mode === 0 ? 0n : ethToWei(String(clamp(feeSel.value || "0.001", 0.001, 1)));
      const rec = recSel.value.trim() ? mustAddr(recSel.value.trim()) : w.address;

      log(`[${nowTs()}] [tx] updateModelSettings enabled=${enabled} mode=${mode} fee=${weiToEth(fee)} rec=${rec}`);
      const tx = await registry.updateModelSettings(BigInt(tokenId), enabled, mode, fee, rec, { gasLimit: 12_000_000 });
      log(`  tx.hash ${tx.hash}`);
      const rcpt = await tx.wait();
      log(`  mined status=${rcpt.status}`);
      if (rcpt.status !== 1) throw new Error("update reverted");
      await loadAll();
    } catch (e) {
      log(`[${nowTs()}] [error] ${e.message || e}`);
    }
  });

  burnBtn.addEventListener("click", async () => {
    try {
      const sys = loadSystem();
      const w = getWalletState();
      if (!w.address) throw new Error("Connect wallet");
      if (String(w.chainId) !== "29") throw new Error("Switch to chainId=29");
      if (!confirm("Burn & delete this model? This cannot be undone.")) return;

      const { signer } = await getSignerProvider();
      const registry = new ethers.Contract(mustAddr(sys.registry), ABI_REGISTRY, signer);
      log(`[${nowTs()}] [tx] burnAndDelete tokenId=${tokenId}`);
      const tx = await registry.burnAndDelete(BigInt(tokenId), { gasLimit: 20_000_000 });
      log(`  tx.hash ${tx.hash}`);
      const rcpt = await tx.wait();
      log(`  mined status=${rcpt.status}`);
      if (rcpt.status !== 1) throw new Error("burn/delete reverted");
      window.location.href = "./my.html";
    } catch (e) {
      log(`[${nowTs()}] [error] ${e.message || e}`);
    }
  });

  refreshBtn.addEventListener("click", loadAll);

  window.addEventListener("genesis_wallet_changed", () => {
    // show tx controls when wallet connects
    loadAll().catch(()=>{});
  });

  await loadAll();
});

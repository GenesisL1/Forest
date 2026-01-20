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
import { makeLogger, loadSystem, mustAddr, shortAddr, nowTs, weiToEth } from "./common.js";
import { getReadProvider, getSignerProvider, getWalletState } from "./eth.js";
import { ABI_REGISTRY, ABI_MODELNFT, ABI_MARKET } from "./abis.js";

const ethers = globalThis.ethers;

const PAGE_SIZE = 25;

function qp() { return new URLSearchParams(window.location.search); }
function setQP(params) {
  const u = new URL(window.location.href);
  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined || v === "") u.searchParams.delete(k);
    else u.searchParams.set(k, String(v));
  }
  history.replaceState({}, "", u.toString());
}

function wordHashes(query) {
  const words = String(query || "")
    .toLowerCase()
    .split(/[\s,]+/)
    .map((w) => w.trim())
    .filter((w) => w.length >= 2);
  const uniq = Array.from(new Set(words));
  return uniq.map((w) => ethers.keccak256(ethers.toUtf8Bytes(w)));
}

async function bytesToDataUrlPng(bytes) {
  try {
    const u8 = bytes instanceof Uint8Array ? bytes : ethers.getBytes(bytes);
    if (!u8?.length) return null;
    const blob = new Blob([u8], { type: "image/png" });
    return URL.createObjectURL(blob);
  } catch {
    return null;
  }
}

function pricingLabel(enabled, mode, feeWei) {
  if (!enabled) return "Disabled";
  if (mode === 0) return "Free";
  if (mode === 1) return `Tips · ${weiToEth(feeWei)} L1`;
  return `Paid · ${weiToEth(feeWei)} L1`;
}

function cardEl({ tokenId, title, desc, iconUrl, nTrees, nFeatures, depth, owner, enabled, mode, feeWei, priceWei, onBuy }) {
  const card = document.createElement("div");
  card.className = "card";
  card.addEventListener("click", () => window.location.href = `./model.html?tokenId=${tokenId}`);

  const head = document.createElement("div");
  head.style.display = "flex";
  head.style.flexDirection = "column";
  head.style.alignItems = "center";
  head.style.gap = "12px";

  const t = document.createElement("h3");
  t.style.margin = "0";
  t.style.textAlign = "center";
  t.textContent = title || `Model #${tokenId}`;
  head.appendChild(t);

  const img = document.createElement("img");
  img.style.width = "84px";
  img.style.height = "84px";
  img.style.borderRadius = "14px";
  img.style.border = "1px solid var(--border-light)";
  img.style.background = "white";
  img.style.objectFit = "cover";
  img.alt = "icon";
  img.src =
    iconUrl ||
    "data:image/svg+xml;utf8," +
      encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="84" height="84">
        <rect width="84" height="84" rx="14" fill="#f1f5f9"/>
        <path d="M16 52 L33 35 L42 44 L68 18" stroke="#2563eb" stroke-width="6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>`);
  head.appendChild(img);

  card.appendChild(head);

  const p = document.createElement("div");
  p.className = "muted";
  p.style.marginTop = "12px";
  p.style.minHeight = "42px";
  p.textContent = desc || "";
  card.appendChild(p);

  const tags = document.createElement("div");
  tags.style.display = "flex";
  tags.style.flexWrap = "wrap";
  tags.style.gap = "8px";
  tags.style.marginTop = "14px";

  const tag = (txt, blue = false) => {
    const s = document.createElement("span");
    s.className = "tag" + (blue ? " blue" : "");
    s.textContent = txt;
    return s;
  };

  tags.appendChild(tag(`NFT #${tokenId}`, true));
  tags.appendChild(tag(`${nTrees} trees`));
  tags.appendChild(tag(`depth ${depth}`));
  tags.appendChild(tag(`${nFeatures} feats`));
  tags.appendChild(tag(`inference: ${pricingLabel(enabled, mode, feeWei)}`, true));
  tags.appendChild(tag(`price: ${weiToEth(priceWei)} L1`, true));

  card.appendChild(tags);

  const foot = document.createElement("div");
  foot.style.display = "flex";
  foot.style.justifyContent = "space-between";
  foot.style.alignItems = "center";
  foot.style.marginTop = "14px";
  foot.style.gap = "10px";

  const own = document.createElement("a");
  own.href = `./forest.html?owner=${encodeURIComponent(owner)}`;
  own.textContent = shortAddr(owner);
  own.style.textDecoration = "none";
  own.style.color = "var(--primary)";
  own.style.fontFamily = "'JetBrains Mono', monospace";
  own.style.fontSize = "0.85rem";
  own.addEventListener("click", (e) => e.stopPropagation());

  const buy = document.createElement("button");
  buy.className = "btn primary small";
  buy.textContent = "Buy";
  buy.addEventListener("click", async (e) => {
    e.stopPropagation();
    await onBuy?.();
  });

  foot.appendChild(own);
  foot.appendChild(buy);
  card.appendChild(foot);

  return card;
}

document.addEventListener("DOMContentLoaded", async () => {
  setupNav({ active: "market", logElId: "debugLines" });
  setupDebugDock({ state: "idle" });
  const logger = makeLogger(document.getElementById("debugLines"));

  const cards = document.getElementById("cards");
  const countPill = document.getElementById("countPill");
  const pagePill = document.getElementById("pagePill");

  const searchBox = document.getElementById("searchBox");
  const applyBtn = document.getElementById("applyBtn");
  const clearBtn = document.getElementById("clearBtn");
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");


  const q0 = qp().get("q") || "";
  const page0 = parseInt(qp().get("page") || "0", 10) || 0;
  let page = Math.max(0, page0);
  searchBox.value = q0;

  applyBtn.addEventListener("click", async () => {
    page = 0;
    setQP({ q: searchBox.value.trim(), page });
    await load();
  });

  clearBtn.addEventListener("click", async () => {
    searchBox.value = "";
    page = 0;
    setQP({ q: "", page });
    await load();
  });

  prevBtn.addEventListener("click", async () => {
    if (page <= 0) return;
    page -= 1;
    setQP({ page });
    await load();
  });

  nextBtn.addEventListener("click", async () => {
    page += 1;
    setQP({ page });
    await load();
  });

  async function buyToken(market, tokenId, priceWei) {
    const w = getWalletState();
    if (!w?.address) throw new Error("Connect wallet first");
    if (String(w.chainId) !== "29") throw new Error("Switch MetaMask to GenesisL1 (chainId=29)");
    const { signer } = await getSignerProvider();
    const m = market.connect(signer);
    logger(`[${nowTs()}] [tx] Buy tokenId=${tokenId} value=${weiToEth(priceWei)} L1`);
    const tx = await m.buy(BigInt(tokenId), { value: priceWei, gasLimit: 35_000_000 });
    logger(`  tx.hash ${tx.hash}`);
    const rcpt = await tx.wait();
    logger(`  mined status=${rcpt.status}`);
    if (rcpt.status !== 1) throw new Error("buy reverted");
  }

  async function load() {
    const sys = loadSystem();
    if (!sys.rpc || !sys.registry || !sys.nft || !sys.market) {
      logger(`[${nowTs()}] [error] Missing system config (rpc, registry, nft, market). Configure contracts in src/common.js defaults or provide deployed contract addresses (see src/common.js defaults).`);
      return;
    }

    const rp = getReadProvider(sys.rpc);
    const registry = new ethers.Contract(mustAddr(sys.registry), ABI_REGISTRY, rp);
    const nft = new ethers.Contract(mustAddr(sys.nft), ABI_MODELNFT, rp);
    const market = new ethers.Contract(mustAddr(sys.market), ABI_MARKET, rp);

    const q = searchBox.value.trim();

    cards.innerHTML = "";
    countPill.textContent = "Loading…";
    pagePill.textContent = `Page ${page + 1}`;

    let items = [];

    if (!q) {
      const cursor = BigInt(page * PAGE_SIZE);
      const res = await market.getListingsPage(cursor, BigInt(PAGE_SIZE), { gasLimit: 2_000_000_000 });
      const tids = res[0] || [];
      const prices = res[1] || [];
      const sellers = res[2] || [];
      items = tids.map((tid, i) => ({
        tokenId: Number(tid),
        priceWei: BigInt(prices[i] || 0),
        seller: sellers[i] || ethers.ZeroAddress
      }));
      countPill.textContent = `${items.length} listed`;
    } else {
      const words = wordHashes(q);
      let cursor = BigInt(page * PAGE_SIZE);
      let gathered = [];
      while (gathered.length < PAGE_SIZE) {
        const res = await registry.searchTitleWords(words, cursor, 120n, { gasLimit: 2_000_000_000 });
        const tids = (res[0] || []).map((x) => Number(x));
        const nextCursor = BigInt(res[1] || 0);
        if (!tids.length) break;

        for (const tid of tids) {
          try {
            const li = await market.getListing(BigInt(tid), { gasLimit: 2_000_000_000 });
            const listed = !!li[0];
            if (!listed) continue;
            gathered.push({ tokenId: tid, priceWei: BigInt(li[1]), seller: li[2] });
          } catch {}
          if (gathered.length >= PAGE_SIZE) break;
        }

        if (nextCursor === cursor) break;
        cursor = nextCursor;
      }
      items = gathered.slice(0, PAGE_SIZE);
      countPill.textContent = `Search-listed`;
    }

    const iconCache = new Map();

    for (const it of items) {
      const tid = it.tokenId;

      try {
        const sum = await registry.getModelSummary(BigInt(tid), { gasLimit: 2_000_000_000 });
        if (!sum[0]) continue;

        const nFeatures = Number(sum[3]);
        const nTrees = Number(sum[4]);
        const depth = Number(sum[5]);
        const mode = Number(sum[7]);
        const feeWei = BigInt(sum[8]);
        const enabled = !!sum[10];
        const title = sum[13];
        const desc = sum[14];

        const owner = await nft.ownerOf(BigInt(tid), { gasLimit: 2_000_000_000 });

        let iconUrl = iconCache.get(tid) || null;
        if (!iconUrl) {
          try {
            const raw = await nft.icon(BigInt(tid), { gasLimit: 2_000_000_000 });
            iconUrl = await bytesToDataUrlPng(raw);
            if (iconUrl) iconCache.set(tid, iconUrl);
          } catch {}
        }

        cards.appendChild(cardEl({
          tokenId: tid,
          title,
          desc,
          iconUrl,
          nTrees,
          nFeatures,
          depth,
          owner,
          enabled,
          mode,
          feeWei,
          priceWei: it.priceWei,
          onBuy: async () => {
            try {
              await buyToken(market, tid, it.priceWei);
              await load();
            } catch (e) {
              logger(`[${nowTs()}] [error] ${e.message || e}`);
            }
          }
        }));
      } catch (e) {
        logger(`[${nowTs()}] [warn] Failed to render tokenId=${tid}: ${e.message || e}`);
      }
    }

    if (!items.length) {
      const empty = document.createElement("div");
      empty.className = "muted";
      empty.textContent = "No listed models found on this page.";
      cards.appendChild(empty);
    }
  }

  await load();
});

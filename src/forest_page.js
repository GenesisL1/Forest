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
import { getReadProvider } from "./eth.js";
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
  const out = uniq.map((w) => ethers.keccak256(ethers.toUtf8Bytes(w)));
  return out;
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

function cardEl({ tokenId, title, desc, iconUrl, nTrees, nFeatures, depth, owner, enabled, mode, feeWei, listedPriceWei }) {
  const card = document.createElement("div");
  card.className = "card";

  card.addEventListener("click", () => {
    window.location.href = `./model.html?tokenId=${tokenId}`;
  });

  const h = document.createElement("div");
  h.style.display = "flex";
  h.style.flexDirection = "column";
  h.style.alignItems = "center";
  h.style.gap = "12px";

  const t = document.createElement("h3");
  t.style.margin = "0";
  t.style.textAlign = "center";
  t.textContent = title || `Model #${tokenId}`;
  h.appendChild(t);

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
  h.appendChild(img);

  card.appendChild(h);

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

  if (listedPriceWei !== null && listedPriceWei !== undefined) {
    tags.appendChild(tag(`price: ${weiToEth(listedPriceWei)} L1`, true));
  }

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

  const view = document.createElement("span");
  view.className = "pill";
  view.textContent = "Open";
  view.style.fontSize = "0.75rem";

  foot.appendChild(own);
  foot.appendChild(view);
  card.appendChild(foot);

  return card;
}

document.addEventListener("DOMContentLoaded", async () => {
  setupNav({ active: "forest", logElId: "debugLines" });
  setupDebugDock({ state: "idle" });
  const logger = makeLogger(document.getElementById("debugLines"));

  const cards = document.getElementById("cards");
  const countPill = document.getElementById("countPill");
  const pagePill = document.getElementById("pagePill");

  const searchBox = document.getElementById("searchBox");
  const ownerBox = document.getElementById("ownerBox");
  const creatorBox = document.getElementById("creatorBox");
  const applyBtn = document.getElementById("applyBtn");
  const clearBtn = document.getElementById("clearBtn");
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");


  // init filters from query params
  const q0 = qp().get("q") || "";
  const o0 = qp().get("owner") || "";
  const c0 = qp().get("creator") || "";
  const page0 = parseInt(qp().get("page") || "0", 10) || 0;
  let page = Math.max(0, page0);

  searchBox.value = q0;
  ownerBox.value = o0;
  creatorBox.value = c0;

  applyBtn.addEventListener("click", async () => {
    page = 0;
    setQP({ q: searchBox.value.trim(), owner: ownerBox.value.trim(), creator: creatorBox.value.trim(), page });
    await load();
  });

  clearBtn.addEventListener("click", async () => {
    searchBox.value = "";
    ownerBox.value = "";
    creatorBox.value = "";
    page = 0;
    setQP({ q: "", owner: "", creator: "", page });
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

  async function load() {
    const sys = loadSystem();
    if (!sys.rpc || !sys.registry || !sys.nft) {
      logger(`[${nowTs()}] [error] Missing system config (rpc, registry, nft). Configure contracts in src/common.js defaults or provide deployed contract addresses (see src/common.js defaults).`);
      return;
    }

    const rp = getReadProvider(sys.rpc);
    const registry = new ethers.Contract(mustAddr(sys.registry), ABI_REGISTRY, rp);
    const nft = new ethers.Contract(mustAddr(sys.nft), ABI_MODELNFT, rp);
    const market = sys.market ? new ethers.Contract(mustAddr(sys.market), ABI_MARKET, rp) : null;

    const q = searchBox.value.trim();
    const ownerFilterRaw = ownerBox.value.trim();
    let ownerFilter = null;
    try { if (ownerFilterRaw) ownerFilter = mustAddr(ownerFilterRaw); } catch { ownerFilter = null; }

    const creatorFilterRaw = creatorBox.value.trim();
    let creatorFilter = null;
    try { if (creatorFilterRaw) creatorFilter = mustAddr(creatorFilterRaw); } catch { creatorFilter = null; }

    cards.innerHTML = "";
    countPill.textContent = "Loading…";
    pagePill.textContent = `Page ${page + 1}`;

    // Choose token IDs
    let tokenIds = [];

    if (!q) {
      if (ownerFilter) {
        const bal = Number(await nft.balanceOf(ownerFilter, { gasLimit: 2_000_000_000 }));

        // If also filtering by creator, we can't use simple page slicing.
        if (creatorFilter) {
          const targetStart = page * PAGE_SIZE;
          let matched = 0;
          for (let i = 0; i < bal; i++) {
            const tid = Number(await nft.tokenOfOwnerByIndex(ownerFilter, i, { gasLimit: 2_000_000_000 }));
            try {
              const sum = await registry.getModelSummary(BigInt(tid), { gasLimit: 2_000_000_000 });
              if (!sum[0]) continue;
              const cr = ethers.getAddress(sum[11]);
              if (cr !== ethers.getAddress(creatorFilter)) continue;
              if (matched >= targetStart && tokenIds.length < PAGE_SIZE) tokenIds.push(tid);
              matched++;
              if (tokenIds.length >= PAGE_SIZE) break;
            } catch {}
            if (i > 1200 && tokenIds.length === 0) break; // prevent pathological scans
          }
          countPill.textContent = `${bal} owned (filtered by creator)`;
        } else {
          const start = page * PAGE_SIZE;
          const end = Math.min(bal, start + PAGE_SIZE);
          for (let i = start; i < end; i++) {
            const tid = await nft.tokenOfOwnerByIndex(ownerFilter, i, { gasLimit: 2_000_000_000 });
            tokenIds.push(Number(tid));
          }
          countPill.textContent = `${bal} models`;
        }
      } else if (creatorFilter) {
        // Creator filter: scan minted ids and pick those whose creator matches.
        const total = Number(await nft.totalMinted({ gasLimit: 2_000_000_000 }));
        const targetStart = page * PAGE_SIZE;
        let matched = 0;
        for (let tid = 1; tid <= total; tid++) {
          try {
            const sum = await registry.getModelSummary(BigInt(tid), { gasLimit: 2_000_000_000 });
            if (!sum[0]) continue;
            const cr = ethers.getAddress(sum[11]);
            if (cr !== ethers.getAddress(creatorFilter)) continue;
            if (matched >= targetStart && tokenIds.length < PAGE_SIZE) tokenIds.push(tid);
            matched++;
            if (tokenIds.length >= PAGE_SIZE) break;
          } catch {}
          if (tid > 2500 && tokenIds.length === 0) break;
        }
        countPill.textContent = `${total} minted (filtered by creator)`;
      } else {
        const total = Number(await nft.totalMinted({ gasLimit: 2_000_000_000 }));
        const startId = page * PAGE_SIZE + 1;
        let cur = startId;
        while (tokenIds.length < PAGE_SIZE && cur <= total) {
          try {
            const sum = await registry.getModelSummary(BigInt(cur), { gasLimit: 2_000_000_000 });
            if (sum[0]) tokenIds.push(cur);
          } catch {}
          cur++;
          if (cur - startId > 400) break;
        }
        countPill.textContent = `${total} minted (showing active)`;
      }
    } else {
      const words = wordHashes(q);
      let cursor = BigInt(page * PAGE_SIZE);
      let gathered = [];
      let nextCursor = cursor;
      while (gathered.length < PAGE_SIZE) {
        const limit = ownerFilter ? 120n : 25n;
        const res = await registry.searchTitleWords(words, cursor, limit, { gasLimit: 2_000_000_000 });
        const tids = (res[0] || []).map((x) => Number(x));
        nextCursor = BigInt(res[1] || 0);
        if (!tids.length) break;

        for (const tid of tids) {
          try {
            if (ownerFilter) {
              const own = await nft.ownerOf(BigInt(tid), { gasLimit: 2_000_000_000 });
              if (ethers.getAddress(own) !== ethers.getAddress(ownerFilter)) continue;
            }
            if (creatorFilter) {
              const sum = await registry.getModelSummary(BigInt(tid), { gasLimit: 2_000_000_000 });
              if (!sum[0]) continue;
              const cr = ethers.getAddress(sum[11]);
              if (cr !== ethers.getAddress(creatorFilter)) continue;
            }
            gathered.push(tid);
          } catch {}
          if (gathered.length >= PAGE_SIZE) break;
        }

        if (nextCursor === cursor) break;
        cursor = nextCursor;
        if (cursor > BigInt(1_000_000_000)) break;
      }
      tokenIds = gathered.slice(0, PAGE_SIZE);
      countPill.textContent = `Search results`;
    }

    const iconCache = new Map();

    for (const tid of tokenIds) {
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

        let listedPriceWei = null;
        if (market) {
          try {
            const li = await market.getListing(BigInt(tid), { gasLimit: 2_000_000_000 });
            const listed = !!li[0];
            if (listed) listedPriceWei = BigInt(li[1]);
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
          listedPriceWei
        }));
      } catch (e) {
        logger(`[${nowTs()}] [warn] Failed to render tokenId=${tid}: ${e.message || e}`);
      }
    }

    if (!tokenIds.length) {
      const empty = document.createElement("div");
      empty.className = "muted";
      empty.textContent = "No models found for this filter/search on this page.";
      cards.appendChild(empty);
    }
  }

  await load();
});

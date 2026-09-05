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

const MANIFEST_URL = "./deployments/genesisl1.json";

const contractLabels = {
  modelStore: "ModelStore",
  modelRegistry: "ModelRegistry",
  modelNft: "ModelNFT",
  forestRuntime: "ForestRuntime",
  modelMarketplace: "ModelMarketplace",
};

const numberFormatter = new Intl.NumberFormat("en-US");

function byId(id) {
  return document.getElementById(id);
}

function setText(id, value) {
  const element = byId(id);
  if (element) element.textContent = String(value);
}

function bytesLabel(bytes) {
  const numeric = Number(bytes);
  if (!Number.isFinite(numeric) || numeric < 0) return "—";
  if (numeric >= 1_000_000) return `${(numeric / 1_000_000).toFixed(2)} MB`;
  return `${numberFormatter.format(numeric)} bytes`;
}

function isAddress(value) {
  return /^0x[a-fA-F0-9]{40}$/.test(String(value));
}

function isHash(value) {
  return /^0x[a-fA-F0-9]{64}$/.test(String(value));
}

function isSha256(value) {
  return /^[a-fA-F0-9]{64}$/.test(String(value));
}

function isNonnegativeSafeInteger(value) {
  const numeric = Number(value);
  return Number.isSafeInteger(numeric) && numeric >= 0;
}

async function loadJson(url) {
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function renderManifest(manifest) {
  if (manifest?.manifest !== "gl1f-deployment/v1") {
    throw new Error("Unsupported deployment manifest");
  }
  if (Number(manifest?.network?.chainId) !== 29) {
    throw new Error("Unexpected deployment chain");
  }

  const entries = Object.entries(manifest?.contracts || {});
  if (entries.length !== 5) throw new Error("Incomplete deployment manifest");

  const tbody = byId("deploymentBody");
  if (!tbody) return;

  const rows = document.createDocumentFragment();
  for (const [key, contract] of entries) {
    if (
      !contractLabels[key]
      || !isAddress(contract?.address)
      || !isNonnegativeSafeInteger(contract?.runtimeCodeBytesAtWitnessBlock)
      || !isSha256(contract?.runtimeCodeSha256AtWitnessBlock)
      || !isHash(contract?.runtimeCodeKeccak256AtWitnessBlock)
    ) {
      throw new Error("Invalid deployment contract record");
    }

    const row = document.createElement("tr");
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = contractLabels[key];

    const addressCell = document.createElement("td");
    const address = document.createElement("code");
    address.textContent = contract.address;
    addressCell.append(address);

    const sizeCell = document.createElement("td");
    sizeCell.textContent = bytesLabel(contract.runtimeCodeBytesAtWitnessBlock);
    sizeCell.title = `SHA-256 ${contract.runtimeCodeSha256AtWitnessBlock}; Keccak-256 ${contract.runtimeCodeKeccak256AtWitnessBlock}`;

    const roleCell = document.createElement("td");
    roleCell.textContent = String(contract.role || "—");

    row.append(heading, addressCell, sizeCell, roleCell);
    rows.append(row);
  }

  tbody.replaceChildren(rows);

  const block = manifest?.evidence?.witnessBlock;
  if (Number.isSafeInteger(Number(block?.number))) {
    setText("manifestNote", `Runtime sizes and dual code digests were observed at block ${numberFormatter.format(Number(block.number))}. They identify deployed bytes, but do not assert that local source builds reproduce them byte-for-byte.`);
  }
}

function setupNavigation() {
  const topbar = document.querySelector(".topbar");
  const toggle = document.querySelector(".navToggle");
  const nav = byId("siteNav");
  if (!topbar || !toggle || !nav) return;

  const setOpen = (open) => {
    topbar.classList.toggle("menu-open", open);
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  };

  toggle.addEventListener("click", () => {
    setOpen(!topbar.classList.contains("menu-open"));
  });
  nav.addEventListener("click", (event) => {
    if (event.target.closest("a")) setOpen(false);
  });
  document.addEventListener("click", (event) => {
    if (!topbar.contains(event.target)) setOpen(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      setOpen(false);
      toggle.focus();
    }
  });
  window.addEventListener("resize", () => {
    if (window.innerWidth > 1200) setOpen(false);
  });
}

function setupCopyButtons() {
  document.querySelectorAll(".copy-command").forEach((button) => {
    button.addEventListener("click", async () => {
      const value = button.dataset.copy || "";
      const original = button.textContent;
      try {
        await navigator.clipboard.writeText(value);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Copy unavailable";
      }
      window.setTimeout(() => {
        button.textContent = original;
      }, 1800);
    });
  });
}

async function initializeEvidence() {
  const state = byId("evidenceState");
  try {
    const manifest = await loadJson(MANIFEST_URL);
    renderManifest(manifest);
    if (state) {
      state.textContent = "Deployment manifest loaded";
      state.dataset.state = "loaded";
    }
  } catch {
    // Keep the static historical summary when the optional manifest is unavailable.
    if (state) {
      state.textContent = "Recorded provider summary";
      state.dataset.state = "fallback";
    }
  }
}

setupNavigation();
setupCopyButtons();
initializeEvidence();

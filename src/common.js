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

const ethers = globalThis.ethers;

const SYS_KEY = "genesisl1_forest_system_v25";

export function nowTs() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function clamp(x, a, b) {
  const n = Number(x);
  if (!Number.isFinite(n)) return a;
  return Math.min(b, Math.max(a, n));
}

export function shortAddr(a) {
  try {
    const x = ethers.getAddress(a);
    return `${x.slice(0, 6)}…${x.slice(-4)}`;
  } catch {
    return String(a || "—");
  }
}

export function mustAddr(a) {
  if (!a || typeof a !== "string") throw new Error("Missing address");
  const s = a.trim();
  if (!s.startsWith("0x")) throw new Error(`Address must be 0x… (got "${s}")`);
  return ethers.getAddress(s);
}

export function loadSystem() {
  try {
    const raw = localStorage.getItem(SYS_KEY);
    const obj = raw ? JSON.parse(raw) : {};
    return {
      rpc: obj.rpc || "https://rpc.genesisl1.org",
      store: obj.store || "0x9CdbC23392648Bd27B4A5eD09c0fEa9452454B54",
      registry: obj.registry || "0x33c9844F77a07e36B98f0FFf8201B8A8b02c2a69",
      nft: obj.nft || "0x44Dc1c54B8D579B42d78cC21cf8260DC0A3279fA",
      runtime: obj.runtime || "0xD2fD0cf461a6cb56Fc08d9aEc120833D8E79044E",
      market: obj.market || "0xA9bfa0a719b7F73cE85CA1E7f23af626D383fB46"
    };
  } catch {
    return {
      rpc: "https://rpc.genesisl1.org",
      store: "0x9CdbC23392648Bd27B4A5eD09c0fEa9452454B54",
      registry: "0x33c9844F77a07e36B98f0FFf8201B8A8b02c2a69",
      nft: "0x44Dc1c54B8D579B42d78cC21cf8260DC0A3279fA",
      runtime: "0xD2fD0cf461a6cb56Fc08d9aEc120833D8E79044E",
      market: "0xA9bfa0a719b7F73cE85CA1E7f23af626D383fB46"
    };
  }
}

export function saveSystem(next) {
  const cur = loadSystem();
  const merged = { ...cur, ...next };
  localStorage.setItem(SYS_KEY, JSON.stringify(merged));
  return merged;
}

export function makeLogger(preEl) {
  return (line) => {
    if (!preEl) return;
    const msg = String(line ?? "");
    preEl.textContent += msg + "\n";
    preEl.scrollTop = preEl.scrollHeight;
  };
}

export function weiToEth(wei) {
  try {
    return ethers.formatEther(wei);
  } catch {
    return String(wei);
  }
}

export function ethToWei(eth) {
  const s = typeof eth === "string" ? eth : String(eth);
  return ethers.parseEther(s);
}

// ===== NFT features packing =====
//
// We store feature labels (and lightweight ML task metadata) in the ModelNFT `features` string.
// Back-compat: older models stored only newline-separated feature names.
// New format (one-line JSON header + newline-separated feature names):
//   #meta={"v":1,"task":"regression"}
//   age
//   bmi
//   ...
//
// For classification models we also store class labels:
//   #meta={"v":1,"task":"binary_classification","labels":["no","yes"],"labelName":"target"}
//   #meta={"v":1,"task":"multiclass_classification","labels":["setosa","versicolor","virginica"],"labelName":"species"}
//
// For multilabel classification models we store BOTH:
// - the output label names (one per column) in `labelNames`
// - the binary class labels in `labels` (defaults to ["0","1"])
// Example:
//   #meta={"v":1,"task":"multilabel_classification","labelNames":["spam","toxic"],"labels":["0","1"]}

export function packNftFeatures({ task, featureNames, labelName = null, labels = null, labelNames = null }) {
  const raw = String(task || "regression").trim();
  const t = (raw === "binary_classification" || raw === "binary")
    ? "binary_classification"
    : (raw === "multiclass_classification" || raw === "multiclass")
      ? "multiclass_classification"
      : (raw === "multilabel_classification" || raw === "multilabel")
        ? "multilabel_classification"
        : "regression";
  const meta = { v: 1, task: t };
  if (labelName) meta.labelName = String(labelName);

  if (t === "multilabel_classification") {
    // Output label names (one per column)
    if (Array.isArray(labelNames) && labelNames.length >= 1) {
      meta.labelNames = labelNames.map((x) => String(x));
    } else if (Array.isArray(labels) && labels.length >= 1) {
      // Back-compat: older callers passed multilabel output names via `labels`.
      meta.labelNames = labels.map((x) => String(x));
    }

    // Binary class labels (defaults to 0/1). Allows custom strings if provided.
    if (Array.isArray(labels) && labels.length >= 2) {
      meta.labels = [String(labels[0]), String(labels[1])];
    } else {
      meta.labels = ["0", "1"];
    }
  } else if ((t === "binary_classification" || t === "multiclass_classification") && Array.isArray(labels) && labels.length >= 2) {
    // For binary/multiclass: `labels` are the class labels.
    meta.labels = labels.map((x) => String(x));
  }
  const lines = [`#meta=${JSON.stringify(meta)}`];
  for (const f of (featureNames || [])) {
    const s = String(f ?? "").trim();
    if (s) lines.push(s);
  }
  return lines.join("\n");
}

export function unpackNftFeatures(featuresPacked) {
  const raw = String(featuresPacked || "");
  const lines = raw.split("\n").map(s => String(s || "").trim()).filter(Boolean);
  let meta = null;
  let start = 0;
  if (lines.length && lines[0].startsWith("#meta=")) {
    try {
      meta = JSON.parse(lines[0].slice(6));
      start = 1;
    } catch {
      meta = null;
      start = 0;
    }
  }
  const features = lines.slice(start);
  return { meta, features, raw };
}

export function taskLabel(task) {
  if (task === "binary_classification") return "Binary classification";
  if (task === "multiclass_classification") return "Multiclass classification";
  if (task === "multilabel_classification") return "Multilabel classification";
  return "Regression";
}

// Stable sigmoid (avoids overflow for large |x|).
export function sigmoid(x) {
  const z = Number(x);
  if (!Number.isFinite(z)) return 0.5;
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}


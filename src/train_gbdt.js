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

// Byte-size estimation for the GenesisL1 Forest model format v1.
export function estimateModelBytes(nTrees, depth) {
  const d = Math.max(1, depth | 0);
  const T = Math.max(0, nTrees | 0);
  const pow = 1 << d; // 2^depth
  const internal = pow - 1;
  const perTree = internal * 8 + pow * 4;
  return 24 + T * perTree;
}

// Byte-size estimation for model format v2 (multiclass).
// v2 header: 24 bytes + 4*nClasses base logits, then (treesPerClass*nClasses) trees.
export function estimateModelBytesV2(treesPerClass, depth, nClasses) {
  const d = Math.max(1, depth | 0);
  const T = Math.max(0, treesPerClass | 0);
  const K = Math.max(0, nClasses | 0);
  const pow = 1 << d;
  const internal = pow - 1;
  const perTree = internal * 8 + pow * 4;
  const header = 24 + Math.max(0, K) * 4;
  return header + (T * K) * perTree;
}

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

// Minimal ABIs used by the UI. Keep consistent with /contracts/*.sol.
export const ABI_STORE = [
  "function write(bytes data) external returns (address pointer)",
  "event ChunkWritten(address indexed pointer, uint256 size)"
];

export const ABI_REGISTRY = [
  "function deployFeeWei() view returns (uint256)",
  "function sizeFeeWeiPerByte() view returns (uint256)",
  "function requiredDeployFeeWei(uint32 totalBytes) view returns (uint256)",
  "function activeLicenseId() view returns (uint256)",
  "function getLicense(uint256 id) view returns (string name, string url)",
  "function tosVersion() view returns (uint256)",
  "function tosHash() view returns (bytes32)",
  "function tosText() view returns (string)",
  "function creatorOf(uint256 tokenId) view returns (address)",
  "function getModelSummary(uint256 tokenId) view returns (bool exists, bytes32 modelId, address tablePtr, uint16 nFeatures, uint16 nTrees, uint16 depth, int32 baseQ, uint8 pricingMode, uint256 feeWei, address feeRecipient, bool inferenceEnabled, address creator, uint32 tosVersionAccepted, string title, string description)",
  "function getModelBytesInfo(bytes32 modelId) view returns (address tablePtr, uint32 chunkSize, uint32 numChunks, uint32 totalBytes)",
  "function getModelRuntime(bytes32 modelId) view returns (address tablePtr, uint32 chunkSize, uint32 numChunks, uint32 totalBytes, uint16 nFeatures, uint16 nTrees, uint16 depth, int32 baseQ, uint32 scaleQ, bool inferenceEnabled, uint8 pricingMode, uint256 feeWei, address feeRecipient)",
  "function searchTitleWords(bytes32[] words, uint256 cursor, uint256 limit) view returns (uint256[] tokenIds, uint256 nextCursor)",
  "function updateModelSettings(uint256 tokenId, bool enabled, uint8 pricingMode, uint256 feeWei, address recipient) external",
  "function burnAndDelete(uint256 tokenId) external",
  "function registerModel(bytes32 modelId, address tablePtr, uint32 chunkSize, uint32 numChunks, uint32 totalBytes, uint16 nFeatures, uint16 nTrees, uint16 depth, int32 baseQ, uint32 scaleQ, string title, string description, bytes iconPng32, string featuresPacked, bytes32[] titleWordHashes, uint8 pricingMode, uint256 feeWei, address recipient, uint32 tosVersionAccepted, uint32 licenseIdAccepted, address ownerKey) payable external",
  "function accessPlanCount(bytes32 modelId) view returns (uint8)",
  "function accessExpiry(bytes32 modelId, address key) view returns (uint64)",
  "function getAccessPlan(bytes32 modelId, uint8 planId) view returns (uint32 durationBlocks, uint256 priceWei, bool active)",
  "function createAccessPlan(bytes32 modelId, uint32 durationBlocks, uint256 priceWei, bool active) external returns (uint8 planId)",
  "function setAccessPlan(bytes32 modelId, uint8 planId, uint32 durationBlocks, uint256 priceWei, bool active) external",
  "function buyAccess(bytes32 modelId, uint8 planId, address key) payable external returns (uint64 newExpiry)",
  "function setOwnerAccessKey(bytes32 modelId, address key) external",
  "function revokeAccessKey(bytes32 modelId, address key) external",
];

export const ABI_MODELNFT = [
  "function ownerOf(uint256 tokenId) view returns (address)",
  "function balanceOf(address owner) view returns (uint256)",
  "function tokenOfOwnerByIndex(address owner, uint256 index) view returns (uint256)",
  "function totalMinted() view returns (uint256)",
  // --- ERC-721 approvals (required for Ai store listings) ---
  "function getApproved(uint256 tokenId) view returns (address)",
  "function isApprovedForAll(address owner, address operator) view returns (bool)",
  "function approve(address to, uint256 tokenId) external",
  "function setApprovalForAll(address operator, bool approved) external",
  "function icon(uint256 tokenId) view returns (bytes)",
  "function title(uint256 tokenId) view returns (string)",
  "function description(uint256 tokenId) view returns (string)",
  "function features(uint256 tokenId) view returns (string)"
];

export const ABI_RUNTIME = [
  "function predictView(bytes32 modelId, bytes packedFeaturesQ) view returns (int256)",
  "function predictOwnerView(bytes32 modelId, bytes packedFeaturesQ, uint256 deadline, bytes signature) view returns (int256)",
  "function predictTx(bytes32 modelId, bytes packedFeaturesQ) payable returns (int256)",
  "event Inference(bytes32 indexed modelId, address indexed caller, int256 scoreQ, uint256 valueWei)",
  "function predictAccessView(bytes32 modelId, bytes packedFeaturesQ, uint256 deadline, bytes sig) view returns (int256)",

  // Multiclass classification (model format v2)
  "function predictClassView(bytes32 modelId, bytes packedFeaturesQ) view returns (uint16 classIndex, int256 bestScoreQ)",
  "function predictClassOwnerView(bytes32 modelId, bytes packedFeaturesQ, uint256 deadline, bytes signature) view returns (uint16 classIndex, int256 bestScoreQ)",
  "function predictClassTx(bytes32 modelId, bytes packedFeaturesQ) payable returns (uint16 classIndex, int256 bestScoreQ)",
  "event InferenceClass(bytes32 indexed modelId, address indexed caller, uint16 classIndex, int256 bestScoreQ, uint256 valueWei)",
  "function predictClassAccessView(bytes32 modelId, bytes packedFeaturesQ, uint256 deadline, bytes sig) view returns (uint16 classIndex, int256 bestScoreQ)",

  // Vector-output (model format v2). Used for multilabel classification.
  // Returns logitsQ per label; UI applies sigmoid(logitQ/scaleQ) for probabilities.
  "function predictMultiView(bytes32 modelId, bytes packedFeaturesQ) view returns (int256[] logitsQ)",
  "function predictMultiOwnerView(bytes32 modelId, bytes packedFeaturesQ, uint256 deadline, bytes signature) view returns (int256[] logitsQ)",
  "function predictMultiTx(bytes32 modelId, bytes packedFeaturesQ) payable returns (int256[] logitsQ)",
  "event InferenceMulti(bytes32 indexed modelId, address indexed caller, int256[] logitsQ, uint256 valueWei)",
  "function predictMultiAccessView(bytes32 modelId, bytes packedFeaturesQ, uint256 deadline, bytes sig) view returns (int256[] logitsQ)",
];

export const ABI_MARKET = [
  "function listingFeeWei() view returns (uint256)",
  "function getListing(uint256 tokenId) view returns (bool listed, uint256 priceWei, address seller)",
  "function getListingsPage(uint256 cursor, uint256 limit) view returns (uint256[] tokenIds, uint256[] prices, address[] sellers, uint256 nextCursor)",
  "function list(uint256 tokenId, uint256 priceWei) payable external",
  "function cancel(uint256 tokenId) external",
  "function buy(uint256 tokenId) payable external"
];

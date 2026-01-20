// SPDX-License-Identifier: MIT
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
pragma solidity ^0.8.20;

import "./SimpleOwnable.sol";
import "./ModelNFT.sol";

contract ModelRegistry is SimpleOwnable {
    // ===== Legal =====
    struct License { string name; string url; }

    uint32 public licenseCount;
    uint32 public activeLicenseId;

    mapping(uint32 => License) private _licenses;

    uint32 public tosVersion;
    bytes32 public tosHash;
    string public tosText;

    // ===== Fees =====
    uint256 public deployFeeWei;
    uint256 public sizeFeeWeiPerByte;

    // ===== Model storage =====
    struct Model {
        bool exists;
        bool active;

        bytes32 modelId;

        address tablePtr;
        uint32 chunkSize;
        uint32 numChunks;
        uint32 totalBytes;

        uint16 nFeatures;
        uint16 nTrees;
        uint16 depth;
        int32 baseQ;
        uint32 scaleQ;

        bool inferenceEnabled;
        uint8 pricingMode; // 0 free, 1 tips, 2 paid
        uint256 feeWei;
        address feeRecipient;

        address creator;
        uint32 tosVersionAccepted;
        uint32 licenseIdAccepted;

        uint256 tokenId;
    }

    // modelId => model
    mapping(bytes32 => Model) public models;

    // tokenId => modelId
    mapping(uint256 => bytes32) public modelIdByTokenId;

    // modelId => tokenId
    mapping(bytes32 => uint256) public tokenIdByModelId;

    // ===== API Access Keys / Subscriptions =====
    struct AccessPlan {
        uint32 durationBlocks;
        uint256 priceWei;
        bool active;
    }

    // modelId => planId (1..count) => plan
    mapping(bytes32 => mapping(uint8 => AccessPlan)) private _accessPlans;
    // modelId => number of plans
    mapping(bytes32 => uint8) public accessPlanCount;

    // modelId => access key => expiry block. 0 = none. type(uint64).max = never expires.
    mapping(bytes32 => mapping(address => uint64)) public accessExpiry;


    // Title-word index: wordHash => tokenId[] and membership mapping for fast AND search
    mapping(bytes32 => uint256[]) private _wordTokens;
    mapping(bytes32 => mapping(uint256 => bool)) private _wordHasToken;
    mapping(uint256 => bytes32[]) private _tokenWords;

    ModelNFT public modelNFT;

    event ModelNFTSet(address indexed nft);
    event LicenseAdded(uint32 indexed id, string name, string url);
    event ActiveLicenseSet(uint32 indexed id);
    event ToSUpdated(uint32 indexed version, bytes32 hash);
    event DeployFeeSet(uint256 feeWei);
    event SizeFeeSet(uint256 weiPerByte);

    event ModelRegistered(uint256 indexed tokenId, bytes32 indexed modelId, address indexed creator);
    event ModelSettingsUpdated(uint256 indexed tokenId, bool enabled, uint8 mode, uint256 feeWei, address recipient);
    event ModelBurned(uint256 indexed tokenId, bytes32 indexed modelId);
    event AccessPlanSet(bytes32 indexed modelId, uint8 indexed planId, uint32 durationBlocks, uint256 priceWei, bool active);
    event AccessPurchased(bytes32 indexed modelId, address indexed buyer, address indexed key, uint8 planId, uint64 newExpiry);
    event OwnerAccessKeySet(bytes32 indexed modelId, address indexed key, uint64 expiry);
    event AccessRevoked(bytes32 indexed modelId, address indexed key);


    constructor(address owner_, string memory initialToS) SimpleOwnable(owner_) {
        // default: 10 L1
        deployFeeWei = 10 ether;
        emit DeployFeeSet(deployFeeWei);
        sizeFeeWeiPerByte = 0;
        emit SizeFeeSet(sizeFeeWeiPerByte);

        // ToS v1
        tosVersion = 1;
        tosText = initialToS;
        tosHash = keccak256(bytes(initialToS));
        emit ToSUpdated(tosVersion, tosHash);

        // License #1: CC BY-SA 4.0
        _addLicense("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/");
        activeLicenseId = 1;
        emit ActiveLicenseSet(activeLicenseId);
    }

    // ===== Admin =====
    function setModelNFT(address nftAddr) external onlyOwner {
        require(nftAddr != address(0), "NFT0");
        modelNFT = ModelNFT(nftAddr);
        emit ModelNFTSet(nftAddr);
    }

    function setDeployFeeWei(uint256 feeWei) external onlyOwner {
        deployFeeWei = feeWei;
        emit DeployFeeSet(feeWei);
    }

    function setSizeFeeWeiPerByte(uint256 weiPerByte) external onlyOwner {
        sizeFeeWeiPerByte = weiPerByte;
        emit SizeFeeSet(weiPerByte);
    }

    function requiredDeployFeeWei(uint32 totalBytes) public view returns (uint256) {
        return deployFeeWei + (sizeFeeWeiPerByte * uint256(totalBytes));
    }

    function addLicense(string calldata name, string calldata url) external onlyOwner returns (uint32 id) {
        id = _addLicense(name, url);
    }

    function _addLicense(string memory name, string memory url) internal returns (uint32 id) {
        require(bytes(name).length > 0, "LIC_NAME");
        require(bytes(url).length > 0, "LIC_URL");
        id = ++licenseCount;
        _licenses[id] = License(name, url);
        emit LicenseAdded(id, name, url);
    }

    function setActiveLicenseId(uint32 id) external onlyOwner {
        require(id >= 1 && id <= licenseCount, "LIC_ID");
        activeLicenseId = id;
        emit ActiveLicenseSet(id);
    }

    function getLicense(uint256 id) external view returns (string memory name, string memory url) {
        License memory l = _licenses[uint32(id)];
        return (l.name, l.url);
    }

    function setToS(string calldata text) external onlyOwner {
        require(bytes(text).length > 0, "TOS_EMPTY");
        tosVersion += 1;
        tosText = text;
        tosHash = keccak256(bytes(text));
        emit ToSUpdated(tosVersion, tosHash);
    }

    // ===== Views for UI =====
    function creatorOf(uint256 tokenId) external view returns (address) {
        bytes32 mid = modelIdByTokenId[tokenId];
        if (mid == bytes32(0)) return address(0);
        return models[mid].creator;
    }

    function getModelSummary(uint256 tokenId) external view returns (
        bool exists,
        bytes32 modelId,
        address tablePtr,
        uint16 nFeatures,
        uint16 nTrees,
        uint16 depth,
        int32 baseQ,
        uint8 pricingMode,
        uint256 feeWei,
        address feeRecipient,
        bool inferenceEnabled,
        address creator,
        uint32 tosVersionAccepted,
        string memory title,
        string memory description
    ) {
        bytes32 mid = modelIdByTokenId[tokenId];
        if (mid == bytes32(0)) return (false, 0, address(0), 0, 0, 0, 0, 0, 0, address(0), false, address(0), 0, "", "");
        Model storage m = models[mid];
        if (!m.exists || !m.active) return (false, 0, address(0), 0, 0, 0, 0, 0, 0, address(0), false, address(0), 0, "", "");

        title = modelNFT.title(tokenId);
        description = modelNFT.description(tokenId);

        return (true, m.modelId, m.tablePtr, m.nFeatures, m.nTrees, m.depth, m.baseQ, m.pricingMode, m.feeWei, m.feeRecipient, m.inferenceEnabled, m.creator, m.tosVersionAccepted, title, description);
    }

    function getModelBytesInfo(bytes32 modelId) external view returns (address tablePtr, uint32 chunkSize, uint32 numChunks, uint32 totalBytes) {
        Model storage m = models[modelId];
        require(m.exists && m.active, "NF");
        return (m.tablePtr, m.chunkSize, m.numChunks, m.totalBytes);
    }

    function getModelRuntime(bytes32 modelId) external view returns (
        address tablePtr,
        uint32 chunkSize,
        uint32 numChunks,
        uint32 totalBytes,
        uint16 nFeatures,
        uint16 nTrees,
        uint16 depth,
        int32 baseQ,
        uint32 scaleQ,
        bool inferenceEnabled,
        uint8 pricingMode,
        uint256 feeWei,
        address feeRecipient
    ) {
        Model storage m = models[modelId];
        require(m.exists && m.active, "NF");
        return (m.tablePtr, m.chunkSize, m.numChunks, m.totalBytes, m.nFeatures, m.nTrees, m.depth, m.baseQ, m.scaleQ, m.inferenceEnabled, m.pricingMode, m.feeWei, m.feeRecipient);
    }

    // AND search on words (exact hash match), paginated over the first word list.
    function searchTitleWords(bytes32[] calldata words, uint256 cursor, uint256 limit) external view returns (uint256[] memory tokenIds, uint256 nextCursor) {
        if (words.length == 0) return (new uint256[](0), 0);

        uint256[] storage baseList = _wordTokens[words[0]];
        uint256 n = baseList.length;
        if (cursor >= n) return (new uint256[](0), 0);

        uint256[] memory tmp = new uint256[](limit);
        uint256 found = 0;
        uint256 i = cursor;

        for (; i < n && found < limit; i++) {
            uint256 tid = baseList[i];
            bytes32 mid = modelIdByTokenId[tid];
            if (mid == bytes32(0)) continue;
            Model storage m = models[mid];
            if (!m.exists || !m.active) continue;

            bool ok = true;
            for (uint256 w = 1; w < words.length; w++) {
                if (!_wordHasToken[words[w]][tid]) { ok = false; break; }
            }
            if (!ok) continue;

            tmp[found++] = tid;
        }

        tokenIds = new uint256[](found);
        for (uint256 k = 0; k < found; k++) tokenIds[k] = tmp[k];
        nextCursor = (i >= n) ? 0 : i;
    }

    // ===== Mutations =====
    function registerModel(
        bytes32 modelId,
        address tablePtr,
        uint32 chunkSize,
        uint32 numChunks,
        uint32 totalBytes,
        uint16 nFeatures,
        uint16 nTrees,
        uint16 depth,
        int32 baseQ,
        uint32 scaleQ,
        string calldata title_,
        string calldata description_,
        bytes calldata iconPng32,
        string calldata featuresPacked,
        bytes32[] calldata titleWordHashes,
        uint8 pricingMode,
        uint256 feeWei,
        address recipient,
        uint32 tosVersionAccepted_,
        uint32 licenseIdAccepted_,
        address ownerKey
    ) external payable returns (uint256 tokenId) {
        require(address(modelNFT) != address(0), "NFT_NOT_SET");
        require(modelId != bytes32(0), "MID0");
        require(!models[modelId].exists, "EXISTS");
        uint256 requiredFee = requiredDeployFeeWei(totalBytes);
        require(msg.value == requiredFee, "DEPLOY_FEE");

        require(tosVersionAccepted_ == tosVersion, "TOS");
        require(licenseIdAccepted_ == activeLicenseId, "LIC");
        require(ownerKey != address(0), "OWNER_KEY");

        require(bytes(title_).length > 0, "TITLE");
        require(bytes(description_).length > 0, "DESC");
        require(iconPng32.length > 0, "ICON");
        require(numChunks > 0, "NO_CHUNKS");
        require(chunkSize > 0, "CHUNK0");

        // fee rules
        if (pricingMode == 0) {
            feeWei = 0;
        } else {
            require(feeWei > 0, "FEE_ZERO");
        }
        if (recipient == address(0)) recipient = msg.sender;

        // mint NFT
        tokenId = modelNFT.mintTo(msg.sender, title_, description_, iconPng32, featuresPacked);
        // Grant the model owner a perpetual API access key.
        accessExpiry[modelId][ownerKey] = type(uint64).max;
        emit OwnerAccessKeySet(modelId, ownerKey, type(uint64).max);

        Model storage m = models[modelId];
        m.exists = true;
        m.active = true;
        m.modelId = modelId;

        m.tablePtr = tablePtr;
        m.chunkSize = chunkSize;
        m.numChunks = numChunks;
        m.totalBytes = totalBytes;

        m.nFeatures = nFeatures;
        m.nTrees = nTrees;
        m.depth = depth;
        m.baseQ = baseQ;
        m.scaleQ = scaleQ;

        m.inferenceEnabled = true;
        m.pricingMode = pricingMode;
        m.feeWei = feeWei;
        m.feeRecipient = recipient;

        m.creator = msg.sender;
        m.tosVersionAccepted = tosVersionAccepted_;
        m.licenseIdAccepted = licenseIdAccepted_;
        m.tokenId = tokenId;

        modelIdByTokenId[tokenId] = modelId;
        tokenIdByModelId[modelId] = tokenId;

        // title index
        if (titleWordHashes.length > 0) {
            bytes32[] storage arr = _tokenWords[tokenId];
            for (uint256 i = 0; i < titleWordHashes.length; i++) {
                bytes32 wh = titleWordHashes[i];
                if (wh == bytes32(0)) continue;
                if (_wordHasToken[wh][tokenId]) continue;
                _wordHasToken[wh][tokenId] = true;
                _wordTokens[wh].push(tokenId);
                arr.push(wh);
            }
        }

        // forward deploy fee to owner
        if (requiredFee > 0) {
            (bool ok,) = owner.call{value: requiredFee}("");
            require(ok, "FEE_SEND");
        }

        emit ModelRegistered(tokenId, modelId, msg.sender);
    }

    function _requireTokenOwner(uint256 tokenId) internal view returns (address o) {
        o = modelNFT.ownerOf(tokenId);
        require(o == msg.sender, "NOT_OWNER");
    }

    function updateModelSettings(uint256 tokenId, bool enabled, uint8 pricingMode, uint256 feeWei, address recipient) external {
        address o = modelNFT.ownerOf(tokenId);
        require(o == msg.sender, "NOT_OWNER");
        bytes32 mid = modelIdByTokenId[tokenId];
        require(mid != bytes32(0), "NF");
        Model storage m = models[mid];
        require(m.exists && m.active, "NF");

        if (pricingMode == 0) feeWei = 0;
        else require(feeWei > 0, "FEE_ZERO");
        if (recipient == address(0)) recipient = o;

        m.inferenceEnabled = enabled;
        m.pricingMode = pricingMode;
        m.feeWei = feeWei;
        m.feeRecipient = recipient;


    }
    // ===== API Access Key Plans =====

    function createAccessPlan(bytes32 modelId, uint32 durationBlocks, uint256 priceWei, bool active) external returns (uint8 planId) {
        _requireTokenOwnerByModelId(modelId);
        require(models[modelId].pricingMode == 2, "MODE");
        require(durationBlocks > 0, "DUR0");
        planId = accessPlanCount[modelId] + 1;
        require(planId != 0, "PLAN_OVERFLOW"); // uint8 overflow
        accessPlanCount[modelId] = planId;
        _accessPlans[modelId][planId] = AccessPlan({durationBlocks: durationBlocks, priceWei: priceWei, active: active});
        emit AccessPlanSet(modelId, planId, durationBlocks, priceWei, active);
    }

    function setAccessPlan(bytes32 modelId, uint8 planId, uint32 durationBlocks, uint256 priceWei, bool active) external {
        _requireTokenOwnerByModelId(modelId);
        require(models[modelId].pricingMode == 2, "MODE");
        require(planId > 0 && planId <= accessPlanCount[modelId], "PLAN_ID");
        require(durationBlocks > 0, "DUR0");
        _accessPlans[modelId][planId] = AccessPlan({durationBlocks: durationBlocks, priceWei: priceWei, active: active});
        emit AccessPlanSet(modelId, planId, durationBlocks, priceWei, active);
    }

    function getAccessPlan(bytes32 modelId, uint8 planId) external view returns (uint32 durationBlocks, uint256 priceWei, bool active) {
        AccessPlan memory p = _accessPlans[modelId][planId];
        return (p.durationBlocks, p.priceWei, p.active);
    }

    function buyAccess(bytes32 modelId, uint8 planId, address key) external payable returns (uint64 newExpiry) {
        require(key != address(0), "KEY0");
        Model storage m = models[modelId];
        require(m.exists && m.active, "NF");
        require(m.pricingMode == 2, "MODE");

        AccessPlan memory p = _accessPlans[modelId][planId];
        require(p.active, "PLAN_OFF");
        require(msg.value == p.priceWei, "PRICE");

        uint64 cur = accessExpiry[modelId][key];
        uint64 start = cur > uint64(block.number) ? cur : uint64(block.number);
        newExpiry = start + uint64(p.durationBlocks);
        accessExpiry[modelId][key] = newExpiry;

        // payout to current owner / recipient
        address payTo = m.feeRecipient;
        if (payTo == address(0)) {
            payTo = modelNFT.ownerOf(m.tokenId);
        }
        if (msg.value > 0) {
            (bool ok,) = payTo.call{value: msg.value}("");
            require(ok, "PAY_FAIL");
        }

        emit AccessPurchased(modelId, msg.sender, key, planId, newExpiry);
    }

    function setOwnerAccessKey(bytes32 modelId, address key) external {
        _requireTokenOwnerByModelId(modelId);
        require(models[modelId].pricingMode == 2, "MODE");
        require(key != address(0), "KEY0");
        accessExpiry[modelId][key] = type(uint64).max;
        emit OwnerAccessKeySet(modelId, key, type(uint64).max);
    }

    function revokeAccessKey(bytes32 modelId, address key) external {
        _requireTokenOwnerByModelId(modelId);
        require(models[modelId].pricingMode == 2, "MODE");
        require(key != address(0), "KEY0");
        accessExpiry[modelId][key] = 0;
        emit AccessRevoked(modelId, key);
    }

    function _requireTokenOwnerByModelId(bytes32 modelId) internal view returns (uint256 tokenId, address owner) {
        require(address(modelNFT) != address(0), "NFT_NOT_SET");
        tokenId = tokenIdByModelId[modelId];
        require(tokenId != 0, "NO_TOKEN");
        owner = modelNFT.ownerOf(tokenId);
        require(owner == msg.sender, "NOT_OWNER");
    }


    function burnAndDelete(uint256 tokenId) external {
        address o = modelNFT.ownerOf(tokenId);
        require(o == msg.sender, "NOT_OWNER");
        _burnAndDelete(tokenId);
    }

    function adminBurnAndDelete(uint256 tokenId) external onlyOwner {
        // emergency
        _burnAndDelete(tokenId);
    }

    function _burnAndDelete(uint256 tokenId) internal {
        bytes32 mid = modelIdByTokenId[tokenId];
        require(mid != bytes32(0), "NF");
        Model storage m = models[mid];
        require(m.exists && m.active, "NF");

        // burn NFT
        modelNFT.burn(tokenId);

        // clear index membership flags (arrays remain, but membership false)
        bytes32[] storage words = _tokenWords[tokenId];
        for (uint256 i = 0; i < words.length; i++) {
            _wordHasToken[words[i]][tokenId] = false;
        }
        delete _tokenWords[tokenId];

        // clear mappings
        delete modelIdByTokenId[tokenId];
        delete tokenIdByModelId[mid];

        // deactivate model
        m.active = false;
        m.inferenceEnabled = false;
        m.tablePtr = address(0);

        emit ModelBurned(tokenId, mid);
    }
}
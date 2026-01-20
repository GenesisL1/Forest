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

interface IModelRegistryRuntime {
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
    );

    function tokenIdByModelId(bytes32 modelId) external view returns (uint256);
    function accessExpiry(bytes32 modelId, address key) external view returns (uint64);
    function modelNFT() external view returns (address);
}

// On-chain inference for GenesisL1 Forest model bytes.
// Model bytes are stored across chunks; the pointer-table and chunks are contracts whose runtime code starts with "GL1C".

interface IModelNFT {
    function ownerOf(uint256 tokenId) external view returns (address);
}

contract ForestRuntime {
    bytes4 internal constant CHUNK_MAGIC = 0x474c3143; // "GL1C"
    IModelRegistryRuntime public immutable registry;
    // ---- EIP-712: owner-gated view inference (no-tx) ----
    // We cannot safely allow free view inference for paid models based on msg.sender alone,
    // because eth_call can spoof the caller address. Instead, the current NFT owner signs
    // an EIP-712 message and anyone can relay it in a read-call.
    bytes32 private constant _EIP712DOMAIN_TYPEHASH = keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)");
    bytes32 private constant _NAME_HASH = keccak256(bytes("GenesisL1 Forest"));
    bytes32 private constant _VERSION_HASH = keccak256(bytes("1"));
    bytes32 private constant _OWNER_VIEW_TYPEHASH = keccak256("OwnerView(bytes32 modelId,bytes32 packedHash,uint256 deadline)");
    bytes32 private constant _ACCESS_VIEW_TYPEHASH = keccak256("AccessView(bytes32 modelId,bytes32 packedHash,uint256 deadline)");


    event Inference(bytes32 indexed modelId, address indexed caller, int256 scoreQ, uint256 valueWei);
    event InferenceClass(bytes32 indexed modelId, address indexed caller, uint16 classIndex, int256 bestScoreQ, uint256 valueWei);
    // Vector-output inference (model format v2): returns logitsQ per label/class.
    // NOTE: We emit int256[] to preserve the full accumulator range (can exceed int32 for many trees).
    event InferenceMulti(bytes32 indexed modelId, address indexed caller, int256[] logitsQ, uint256 valueWei);

    constructor(address registryAddr) {
        require(registryAddr != address(0), "REG0");
        registry = IModelRegistryRuntime(registryAddr);
    }

    // Read-call inference.
    //
    // IMPORTANT: If a model is configured as "paid required" (mode=2), this function reverts.
    // Paid inference must be performed through predictTx() so fees can be enforced.
    function predictView(bytes32 modelId, bytes calldata packedFeaturesQ) external view returns (int256 scoreQ) {
        // Read the model settings to block free inference for pay-required models.
        // We destructure all 13 fields for cross-solc stability (no blank tuple slots).
        (
            address _tablePtr,
            uint32 _chunkSize,
            uint32 _numChunks,
            uint32 _totalBytes,
            uint16 _nFeatures,
            uint16 _nTrees,
            uint16 _depth,
            int32 _baseQ,
            uint32 _scaleQ,
            bool enabled,
            uint8 mode,
            uint256 _feeWei,
            address _recipient
        ) = registry.getModelRuntime(modelId);

        // Silence unused-variable warnings.
        _sinkA(_tablePtr);
        _sinkU(_chunkSize);
        _sinkU(_numChunks);
        _sinkU(_totalBytes);
        _sinkU(_nFeatures);
        _sinkU(_nTrees);
        _sinkU(_depth);
        _sinkI(_baseQ);
        _sinkU(_scaleQ);
        _sinkU(_feeWei);
        _sinkA(_recipient);

        require(enabled, "INF_DISABLED");
        require(mode != 2, "PAID_ONLY");

        scoreQ = _predict(modelId, packedFeaturesQ);
    }

    // Read-call inference for multiclass classification (model format v2).
    //
    // IMPORTANT: If a model is configured as "paid required" (mode=2), this function reverts.
    // Paid inference must be performed through predictClassTx() so fees can be enforced.
    function predictClassView(bytes32 modelId, bytes calldata packedFeaturesQ)
        external
        view
        returns (uint16 classIndex, int256 bestScoreQ)
    {
        (
            address tablePtr,
            uint32 chunkSize,
            uint32 numChunks,
            uint32 totalBytes,
            uint16 nFeatures,
            uint16 _nTrees,
            uint16 _depth,
            int32 _baseQ,
            uint32 _scaleQ,
            bool enabled,
            uint8 mode,
            uint256 _feeWei,
            address _recipient
        ) = registry.getModelRuntime(modelId);

        // Silence unused-variable warnings.
        _sinkU(_nTrees);
        _sinkU(_depth);
        _sinkI(_baseQ);
        _sinkU(_scaleQ);
        _sinkU(_feeWei);
        _sinkA(_recipient);

        require(enabled, "INF_DISABLED");
        require(mode != 2, "PAID_ONLY");

        (classIndex, bestScoreQ) = _predictClassFromChunks(modelId, packedFeaturesQ, tablePtr, chunkSize, numChunks, totalBytes, nFeatures);
    }

    // Read-call inference for vector-output v2 models (multiclass/multilabel).
    // For multilabel classification, the caller should apply sigmoid(logitQ/scaleQ) per label.
    //
    // IMPORTANT: If a model is configured as "paid required" (mode=2), this function reverts.
    // Paid inference must be performed through predictMultiTx() so fees can be enforced.
    function predictMultiView(bytes32 modelId, bytes calldata packedFeaturesQ)
        external
        view
        returns (int256[] memory logitsQ)
    {
        (
            address tablePtr,
            uint32 chunkSize,
            uint32 numChunks,
            uint32 totalBytes,
            uint16 nFeatures,
            uint16 _nTrees,
            uint16 _depth,
            int32 _baseQ,
            uint32 _scaleQ,
            bool enabled,
            uint8 mode,
            uint256 _feeWei,
            address _recipient
        ) = registry.getModelRuntime(modelId);

        // Silence unused-variable warnings.
        _sinkU(_nTrees);
        _sinkU(_depth);
        _sinkI(_baseQ);
        _sinkU(_scaleQ);
        _sinkU(_feeWei);
        _sinkA(_recipient);

        require(enabled, "INF_DISABLED");
        require(mode != 2, "PAID_ONLY");

        logitsQ = _predictMultiFromChunks(modelId, packedFeaturesQ, tablePtr, chunkSize, numChunks, totalBytes, nFeatures);
    }

    // Owner-only read-call inference (no transaction) gated by an EIP-712 signature.
    //
    // This enables the current NFT owner to run paid-required models without paying and without a tx,
    // while still preventing non-owners from bypassing fees via spoofed eth_call 'from' addresses.
    //
    // The owner signs typed data:
    //   OwnerView(modelId, keccak256(packedFeaturesQ), deadline)
    // with domain:
    //   name="GenesisL1 Forest", version="1", chainId, verifyingContract=this
    function predictOwnerView(
        bytes32 modelId,
        bytes calldata packedFeaturesQ,
        uint256 deadline,
        bytes calldata signature
    ) external view returns (int256 scoreQ) {
        require(block.timestamp <= deadline, "SIG_EXPIRED");
        uint256 tokenId = registry.tokenIdByModelId(modelId);
        require(tokenId != 0, "NO_TOKEN");
        address nftAddr = registry.modelNFT();
        require(nftAddr != address(0), "NO_NFT");
        address owner = IModelNFT(nftAddr).ownerOf(tokenId);

        bytes32 structHash = keccak256(abi.encode(
            _OWNER_VIEW_TYPEHASH,
            modelId,
            keccak256(packedFeaturesQ),
            deadline
        ));
        bytes32 digest = _hashTypedDataV4(structHash);
        require(_recover(digest, signature) == owner, "BAD_SIG");

        scoreQ = _predict(modelId, packedFeaturesQ);
    }

    function predictClassOwnerView(
        bytes32 modelId,
        bytes calldata packedFeaturesQ,
        uint256 deadline,
        bytes calldata signature
    ) external view returns (uint16 classIndex, int256 bestScoreQ) {
        require(block.timestamp <= deadline, "SIG_EXPIRED");
        uint256 tokenId = registry.tokenIdByModelId(modelId);
        require(tokenId != 0, "NO_TOKEN");
        address nftAddr = registry.modelNFT();
        require(nftAddr != address(0), "NO_NFT");
        address owner = IModelNFT(nftAddr).ownerOf(tokenId);

        bytes32 structHash = keccak256(abi.encode(
            _OWNER_VIEW_TYPEHASH,
            modelId,
            keccak256(packedFeaturesQ),
            deadline
        ));
        bytes32 digest = _hashTypedDataV4(structHash);
        require(_recover(digest, signature) == owner, "BAD_SIG");

        (classIndex, bestScoreQ) = _predictClass(modelId, packedFeaturesQ);
    }

    function predictMultiOwnerView(
        bytes32 modelId,
        bytes calldata packedFeaturesQ,
        uint256 deadline,
        bytes calldata signature
    ) external view returns (int256[] memory logitsQ) {
        require(block.timestamp <= deadline, "SIG_EXPIRED");
        uint256 tokenId = registry.tokenIdByModelId(modelId);
        require(tokenId != 0, "NO_TOKEN");
        address nftAddr = registry.modelNFT();
        require(nftAddr != address(0), "NO_NFT");
        address owner = IModelNFT(nftAddr).ownerOf(tokenId);

        bytes32 structHash = keccak256(abi.encode(
            _OWNER_VIEW_TYPEHASH,
            modelId,
            keccak256(packedFeaturesQ),
            deadline
        ));
        bytes32 digest = _hashTypedDataV4(structHash);
        require(_recover(digest, signature) == owner, "BAD_SIG");

        logitsQ = _predictMulti(modelId, packedFeaturesQ);
    }


    // Optional paid/tips inference in a transaction.

    // View inference for *subscribed API keys* (no transaction, but requires an EIP-712 signature from the key).
    // Intended for Paid-required models (pricingMode=2).
    function predictAccessView(bytes32 modelId, bytes calldata packedFeaturesQ, uint256 deadline, bytes calldata sig)
        external
        view
        returns (int256 scoreQ)
    {
        require(block.timestamp <= deadline, "DEADLINE");
        require(sig.length == 65, "SIG");

        (
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
            address recipient
        ) = registry.getModelRuntime(modelId);

        require(inferenceEnabled, "INF_OFF");
        require(pricingMode == 2, "MODE"); // paid-required only

        bytes32 packedHash = keccak256(packedFeaturesQ);
        bytes32 structHash = keccak256(abi.encode(_ACCESS_VIEW_TYPEHASH, modelId, packedHash, deadline));
        bytes32 digest = keccak256(abi.encodePacked(hex"1901", _domainSeparatorV4(), structHash));
        address signer = _recover(digest, sig);
        require(signer != address(0), "SIG0");

        uint64 exp = registry.accessExpiry(modelId, signer);
        require(exp == type(uint64).max || exp >= uint64(block.number), "NO_ACCESS");

        // Silence unused warnings (viaIR builds) while keeping tuple ABI stable.
        _sinkU(feeWei);
        _sinkA(recipient);

        scoreQ = _predictFromChunks(modelId, packedFeaturesQ, tablePtr, chunkSize, numChunks, totalBytes, nFeatures, nTrees, depth, baseQ, scaleQ);
    }

    // View inference for *subscribed API keys* (multiclass, model format v2).
    // Intended for Paid-required models (pricingMode=2).
    function predictClassAccessView(bytes32 modelId, bytes calldata packedFeaturesQ, uint256 deadline, bytes calldata sig)
        external
        view
        returns (uint16 classIndex, int256 bestScoreQ)
    {
        require(block.timestamp <= deadline, "DEADLINE");
        require(sig.length == 65, "SIG");

        (
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
            address recipient
        ) = registry.getModelRuntime(modelId);

        require(inferenceEnabled, "INF_OFF");
        require(pricingMode == 2, "MODE"); // paid-required only

        bytes32 packedHash = keccak256(packedFeaturesQ);
        bytes32 structHash = keccak256(abi.encode(_ACCESS_VIEW_TYPEHASH, modelId, packedHash, deadline));
        bytes32 digest = keccak256(abi.encodePacked(hex"1901", _domainSeparatorV4(), structHash));
        address signer = _recover(digest, sig);
        require(signer != address(0), "SIG0");

        uint64 exp = registry.accessExpiry(modelId, signer);
        require(exp == type(uint64).max || exp >= uint64(block.number), "NO_ACCESS");

        // Silence unused warnings (viaIR builds) while keeping tuple ABI stable.
        _sinkU(feeWei);
        _sinkA(recipient);
        _sinkU(nTrees);
        _sinkU(depth);
        _sinkI(baseQ);
        _sinkU(scaleQ);

        (classIndex, bestScoreQ) = _predictClassFromChunks(modelId, packedFeaturesQ, tablePtr, chunkSize, numChunks, totalBytes, nFeatures);
    }

    // View inference for *subscribed API keys* (vector-output v2).
    // Intended for Paid-required models (pricingMode=2).
    function predictMultiAccessView(bytes32 modelId, bytes calldata packedFeaturesQ, uint256 deadline, bytes calldata sig)
        external
        view
        returns (int256[] memory logitsQ)
    {
        require(block.timestamp <= deadline, "DEADLINE");
        require(sig.length == 65, "SIG");

        (
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
            address recipient
        ) = registry.getModelRuntime(modelId);

        require(inferenceEnabled, "INF_OFF");
        require(pricingMode == 2, "MODE"); // paid-required only

        bytes32 packedHash = keccak256(packedFeaturesQ);
        bytes32 structHash = keccak256(abi.encode(_ACCESS_VIEW_TYPEHASH, modelId, packedHash, deadline));
        bytes32 digest = keccak256(abi.encodePacked(hex"1901", _domainSeparatorV4(), structHash));
        address signer = _recover(digest, sig);
        require(signer != address(0), "SIG0");

        uint64 exp = registry.accessExpiry(modelId, signer);
        require(exp == type(uint64).max || exp >= uint64(block.number), "NO_ACCESS");

        // Silence unused warnings (viaIR builds) while keeping tuple ABI stable.
        _sinkU(feeWei);
        _sinkA(recipient);
        _sinkU(nTrees);
        _sinkU(depth);
        _sinkI(baseQ);
        _sinkU(scaleQ);

        logitsQ = _predictMultiFromChunks(modelId, packedFeaturesQ, tablePtr, chunkSize, numChunks, totalBytes, nFeatures);
    }



    function predictTx(bytes32 modelId, bytes calldata packedFeaturesQ) external payable returns (int256 scoreQ) {
        // Note: avoid blank tuple slots (",") because different solc versions can
        // interpret trailing commas differently. Destructure all 13 components.
        (
            address _tablePtr,
            uint32 _chunkSize,
            uint32 _numChunks,
            uint32 _totalBytes,
            uint16 _nFeatures,
            uint16 _nTrees,
            uint16 _depth,
            int32 _baseQ,
            uint32 _scaleQ,
            bool enabled,
            uint8 mode,
            uint256 feeWei,
            address recipient
        ) = registry.getModelRuntime(modelId);

        // We intentionally destructure all 13 fields for cross-solc stability (no blank tuple slots).
        // Silence unused-variable warnings for fields not needed by predictTx().
        _sinkA(_tablePtr);
        _sinkU(_chunkSize);
        _sinkU(_numChunks);
        _sinkU(_totalBytes);
        _sinkU(_nFeatures);
        _sinkU(_nTrees);
        _sinkU(_depth);
        _sinkI(_baseQ);
        _sinkU(_scaleQ);

        require(enabled, "INF_DISABLED");

        if (mode == 2 && msg.value < feeWei) {
            // Paid-required: allow the current NFT owner to run fee-free inference (gas still applies).
            // NOTE: We *must not* allow fee-free view calls for paid models; only this tx path can be safely gated.
            uint256 tokenId = registry.tokenIdByModelId(modelId);
            address nftAddr = registry.modelNFT();
            address owner = IModelNFT(nftAddr).ownerOf(tokenId);
            require(msg.sender == owner, "INF_FEE");
        }
        // tips mode: any value ok, including 0

        scoreQ = _predict(modelId, packedFeaturesQ);

        if (msg.value > 0) {
            if (recipient == address(0)) recipient = msg.sender;
            (bool ok,) = recipient.call{value: msg.value}("");
            require(ok, "PAY_FAIL");
        }

        emit Inference(modelId, msg.sender, scoreQ, msg.value);
    }

    function predictClassTx(bytes32 modelId, bytes calldata packedFeaturesQ)
        external
        payable
        returns (uint16 classIndex, int256 bestScoreQ)
    {
        (
            address tablePtr,
            uint32 chunkSize,
            uint32 numChunks,
            uint32 totalBytes,
            uint16 nFeatures,
            uint16 nTrees,
            uint16 depth,
            int32 baseQ,
            uint32 scaleQ,
            bool enabled,
            uint8 mode,
            uint256 feeWei,
            address recipient
        ) = registry.getModelRuntime(modelId);

        // Silence unused warnings (tuple ABI stability across solc versions).
        _sinkU(nTrees);
        _sinkU(depth);
        _sinkI(baseQ);
        _sinkU(scaleQ);

        require(enabled, "INF_DISABLED");

        if (mode == 2 && msg.value < feeWei) {
            // Paid-required: allow the current NFT owner to run fee-free inference (gas still applies).
            uint256 tokenId = registry.tokenIdByModelId(modelId);
            address nftAddr = registry.modelNFT();
            address owner = IModelNFT(nftAddr).ownerOf(tokenId);
            require(msg.sender == owner, "INF_FEE");
        }

        (classIndex, bestScoreQ) = _predictClassFromChunks(modelId, packedFeaturesQ, tablePtr, chunkSize, numChunks, totalBytes, nFeatures);

        if (msg.value > 0) {
            if (recipient == address(0)) recipient = msg.sender;
            (bool ok,) = recipient.call{value: msg.value}("");
            require(ok, "PAY_FAIL");
        }

        emit InferenceClass(modelId, msg.sender, classIndex, bestScoreQ, msg.value);
    }

    function predictMultiTx(bytes32 modelId, bytes calldata packedFeaturesQ)
        external
        payable
        returns (int256[] memory logitsQ)
    {
        (
            address tablePtr,
            uint32 chunkSize,
            uint32 numChunks,
            uint32 totalBytes,
            uint16 nFeatures,
            uint16 nTrees,
            uint16 depth,
            int32 baseQ,
            uint32 scaleQ,
            bool enabled,
            uint8 mode,
            uint256 feeWei,
            address recipient
        ) = registry.getModelRuntime(modelId);

        // Silence unused warnings (tuple ABI stability across solc versions).
        _sinkU(nTrees);
        _sinkU(depth);
        _sinkI(baseQ);
        _sinkU(scaleQ);

        require(enabled, "INF_DISABLED");

        if (mode == 2 && msg.value < feeWei) {
            // Paid-required: allow the current NFT owner to run fee-free inference (gas still applies).
            uint256 tokenId = registry.tokenIdByModelId(modelId);
            address nftAddr = registry.modelNFT();
            address owner = IModelNFT(nftAddr).ownerOf(tokenId);
            require(msg.sender == owner, "INF_FEE");
        }

        logitsQ = _predictMultiFromChunks(modelId, packedFeaturesQ, tablePtr, chunkSize, numChunks, totalBytes, nFeatures);

        if (msg.value > 0) {
            if (recipient == address(0)) recipient = msg.sender;
            (bool ok,) = recipient.call{value: msg.value}("");
            require(ok, "PAY_FAIL");
        }

        emit InferenceMulti(modelId, msg.sender, logitsQ, msg.value);
    }

    // ========= Internal prediction =========

    /// @dev Predict using already-fetched runtime parameters.
    ///      This avoids a second registry.getModelRuntime() call for view inference paths.
    ///      All parameters are used or explicitly sunk to keep builds warning-free.
    function _predictFromChunks(
        bytes32 modelId,
        bytes calldata packed,
        address tablePtr,
        uint32 chunkSize,
        uint32 numChunks,
        uint32 totalBytes,
        uint16 nFeatures,
        uint16 nTrees,
        uint16 depth,
        int32 baseQ,
        uint32 scaleQ
    ) internal view returns (int256) {
        // keep the modelId in the signature for UI/ABI stability; sink to avoid warnings.
        _sinkB32(modelId);
        _sinkU(totalBytes);
        _sinkU(scaleQ);

        require(packed.length == uint256(nFeatures) * 4, "FEAT_LEN");
        require(numChunks > 0, "NO_CHUNKS");

        // Check table magic
        _requireChunkMagic(tablePtr, "TABLE_CODE");

        int256 acc = int256(baseQ);

        uint256 pow2 = uint256(1) << uint256(depth);
        uint256 internalNodes = pow2 - 1;
        uint256 perTree = internalNodes * 8 + pow2 * 4;

        // trees start at offset 24
        uint256 treesOff = 24;

        for (uint256 t = 0; t < nTrees; t++) {
            uint256 treeBase = treesOff + t * perTree;
            uint256 idx = 0;

            for (uint256 lvl = 0; lvl < depth; lvl++) {
                uint256 nodeOff = treeBase + idx * 8;

                uint16 f = _readU16Model(tablePtr, chunkSize, nodeOff);
                int32 thr = _readI32Model(tablePtr, chunkSize, nodeOff + 2);

                int32 xq = _readI32LE(packed, uint256(f) * 4);
                if (xq > thr) idx = idx * 2 + 2;
                else idx = idx * 2 + 1;
            }

            uint256 leafIndex = idx - internalNodes;
            uint256 leafBase = treeBase + internalNodes * 8;
            uint256 leafOff = leafBase + leafIndex * 4;

            int32 leafQ = _readI32Model(tablePtr, chunkSize, leafOff);
            acc += int256(leafQ);
        }

        return acc;
    }

    // v2 vector-output prediction: returns logitsQ per class/label.
    // Used for multiclass (argmax off-chain) and multilabel (sigmoid per label off-chain).
    function _predictMultiFromChunks(
        bytes32 modelId,
        bytes calldata packed,
        address tablePtr,
        uint32 chunkSize,
        uint32 numChunks,
        uint32 totalBytes,
        uint16 nFeatures
    ) internal view returns (int256[] memory logitsQ) {
        // keep the modelId in the signature for UI/ABI stability; sink to avoid warnings.
        _sinkB32(modelId);

        require(packed.length == uint256(nFeatures) * 4, "FEAT_LEN");
        require(numChunks > 0, "NO_CHUNKS");

        // Check table magic
        _requireChunkMagic(tablePtr, "TABLE_CODE");

        // v2 header
        bytes4 magic = bytes4(_readBytes(tablePtr, chunkSize, 0, 4));
        require(magic == 0x474c3146, "MODEL_MAGIC"); // "GL1F"
        uint8 ver = _readU8Model(tablePtr, chunkSize, 4);
        require(ver == 2, "MODEL_VER");

        uint16 nf = _readU16Model(tablePtr, chunkSize, 6);
        require(nf == nFeatures, "NF_MISMATCH");

        uint16 depth = _readU16Model(tablePtr, chunkSize, 8);
        uint32 treesPerClass = _readU32Model(tablePtr, chunkSize, 10);
        // int32 reserved = _readI32Model(tablePtr, chunkSize, 14);
        uint32 _scaleQ = _readU32Model(tablePtr, chunkSize, 18);
        uint16 nClasses = _readU16Model(tablePtr, chunkSize, 22);

        // silence unused warnings
        _sinkU(_scaleQ);

        require(nClasses >= 2, "NCLS");
        require(treesPerClass > 0, "NTPC");

        uint256 headerSize = 24 + uint256(nClasses) * 4;
        uint256 pow2 = uint256(1) << uint256(depth);
        uint256 internalNodes = pow2 - 1;
        uint256 perTree = internalNodes * 8 + pow2 * 4;
        uint256 totalTrees = uint256(treesPerClass) * uint256(nClasses);
        uint256 expectedLen = headerSize + totalTrees * perTree;
        require(expectedLen <= uint256(totalBytes), "MODEL_LEN");

        uint256 treesOff = headerSize;

        logitsQ = new int256[](uint256(nClasses));

        for (uint256 c = 0; c < uint256(nClasses); c++) {
            int256 acc = int256(_readI32Model(tablePtr, chunkSize, 24 + c * 4));

            uint256 classBase = treesOff + c * uint256(treesPerClass) * perTree;
            for (uint256 t = 0; t < uint256(treesPerClass); t++) {
                uint256 treeBase = classBase + t * perTree;
                uint256 idx = 0;

                for (uint256 lvl = 0; lvl < uint256(depth); lvl++) {
                    uint256 nodeOff = treeBase + idx * 8;
                    uint16 f = _readU16Model(tablePtr, chunkSize, nodeOff);
                    int32 thr = _readI32Model(tablePtr, chunkSize, nodeOff + 2);
                    int32 xq = _readI32LE(packed, uint256(f) * 4);
                    if (xq > thr) idx = idx * 2 + 2;
                    else idx = idx * 2 + 1;
                }

                uint256 leafIndex = idx - internalNodes;
                uint256 leafBase = treeBase + internalNodes * 8;
                uint256 leafOff = leafBase + leafIndex * 4;
                int32 leafQ = _readI32Model(tablePtr, chunkSize, leafOff);
                acc += int256(leafQ);
            }

            logitsQ[c] = acc;
        }
    }

    function _predictClassFromChunks(
        bytes32 modelId,
        bytes calldata packed,
        address tablePtr,
        uint32 chunkSize,
        uint32 numChunks,
        uint32 totalBytes,
        uint16 nFeatures
    ) internal view returns (uint16 classIndex, int256 bestScoreQ) {
        // keep the modelId in the signature for UI/ABI stability; sink to avoid warnings.
        _sinkB32(modelId);

        require(packed.length == uint256(nFeatures) * 4, "FEAT_LEN");
        require(numChunks > 0, "NO_CHUNKS");

        // Check table magic
        _requireChunkMagic(tablePtr, "TABLE_CODE");

        // v2 header
        bytes4 magic = bytes4(_readBytes(tablePtr, chunkSize, 0, 4));
        require(magic == 0x474c3146, "MODEL_MAGIC"); // "GL1F"
        uint8 ver = _readU8Model(tablePtr, chunkSize, 4);
        require(ver == 2, "MODEL_VER");

        uint16 nf = _readU16Model(tablePtr, chunkSize, 6);
        require(nf == nFeatures, "NF_MISMATCH");

        uint16 depth = _readU16Model(tablePtr, chunkSize, 8);
        uint32 treesPerClass = _readU32Model(tablePtr, chunkSize, 10);
        // int32 reserved = _readI32Model(tablePtr, chunkSize, 14);
        uint32 _scaleQ = _readU32Model(tablePtr, chunkSize, 18);
        uint16 nClasses = _readU16Model(tablePtr, chunkSize, 22);

        // silence unused warnings
        _sinkU(_scaleQ);

        require(nClasses >= 2, "NCLS");
        require(treesPerClass > 0, "NTPC");

        uint256 headerSize = 24 + uint256(nClasses) * 4;
        uint256 pow2 = uint256(1) << uint256(depth);
        uint256 internalNodes = pow2 - 1;
        uint256 perTree = internalNodes * 8 + pow2 * 4;
        uint256 totalTrees = uint256(treesPerClass) * uint256(nClasses);
        uint256 expectedLen = headerSize + totalTrees * perTree;
        require(expectedLen <= uint256(totalBytes), "MODEL_LEN");

        uint256 treesOff = headerSize;

        int256 best = type(int256).min;
        uint16 bestC = 0;

        for (uint256 c = 0; c < uint256(nClasses); c++) {
            int256 acc = int256(_readI32Model(tablePtr, chunkSize, 24 + c * 4));

            uint256 classBase = treesOff + c * uint256(treesPerClass) * perTree;
            for (uint256 t = 0; t < uint256(treesPerClass); t++) {
                uint256 treeBase = classBase + t * perTree;
                uint256 idx = 0;

                for (uint256 lvl = 0; lvl < uint256(depth); lvl++) {
                    uint256 nodeOff = treeBase + idx * 8;
                    uint16 f = _readU16Model(tablePtr, chunkSize, nodeOff);
                    int32 thr = _readI32Model(tablePtr, chunkSize, nodeOff + 2);
                    int32 xq = _readI32LE(packed, uint256(f) * 4);
                    if (xq > thr) idx = idx * 2 + 2;
                    else idx = idx * 2 + 1;
                }

                uint256 leafIndex = idx - internalNodes;
                uint256 leafBase = treeBase + internalNodes * 8;
                uint256 leafOff = leafBase + leafIndex * 4;
                int32 leafQ = _readI32Model(tablePtr, chunkSize, leafOff);
                acc += int256(leafQ);
            }

            if (acc > best) {
                best = acc;
                bestC = uint16(c);
            }
        }

        classIndex = bestC;
        bestScoreQ = best;
    }

    function _predictMulti(bytes32 modelId, bytes calldata packed) internal view returns (int256[] memory logitsQ) {
        (
            address tablePtr,
            uint32 chunkSize,
            uint32 numChunks,
            uint32 totalBytes,
            uint16 nFeatures,
            uint16 _nTrees,
            uint16 _depth,
            int32 _baseQ,
            uint32 _scaleQ,
            bool enabled,
            uint8 _mode,
            uint256 _feeWei,
            address _recipient
        ) = registry.getModelRuntime(modelId);

        // Silence unused-variable warnings.
        _sinkU(_nTrees);
        _sinkU(_depth);
        _sinkI(_baseQ);
        _sinkU(_scaleQ);
        _sinkU(_mode);
        _sinkU(_feeWei);
        _sinkA(_recipient);

        require(enabled, "INF_DISABLED");
        logitsQ = _predictMultiFromChunks(modelId, packed, tablePtr, chunkSize, numChunks, totalBytes, nFeatures);
    }

    function _predictClass(bytes32 modelId, bytes calldata packed) internal view returns (uint16 classIndex, int256 bestScoreQ) {
        (
            address tablePtr,
            uint32 chunkSize,
            uint32 numChunks,
            uint32 totalBytes,
            uint16 nFeatures,
            uint16 _nTrees,
            uint16 _depth,
            int32 _baseQ,
            uint32 _scaleQ,
            bool enabled,
            uint8 _mode,
            uint256 _feeWei,
            address _recipient
        ) = registry.getModelRuntime(modelId);

        // Silence unused-variable warnings.
        _sinkU(_nTrees);
        _sinkU(_depth);
        _sinkI(_baseQ);
        _sinkU(_scaleQ);
        _sinkU(_mode);
        _sinkU(_feeWei);
        _sinkA(_recipient);

        require(enabled, "INF_DISABLED");
        (classIndex, bestScoreQ) = _predictClassFromChunks(modelId, packed, tablePtr, chunkSize, numChunks, totalBytes, nFeatures);
    }

    function _predict(bytes32 modelId, bytes calldata packed) internal view returns (int256) {
        // Same story here: always destructure all 13 components.
        (
            address tablePtr,
            uint32 chunkSize,
            uint32 numChunks,
            uint32 totalBytes,
            uint16 nFeatures,
            uint16 nTrees,
            uint16 depth,
            int32 baseQ,
            uint32 scaleQ,
            bool enabled,
            uint8 _mode,
            uint256 _feeWei,
            address _recipient
        ) = registry.getModelRuntime(modelId);

        // Silence unused-variable warnings (these fields are used by predictTx()).
        _sinkU(totalBytes);
        _sinkU(scaleQ);
        _sinkU(_mode);
        _sinkU(_feeWei);
        _sinkA(_recipient);

        // If the model owner/admin disabled inference, do not allow any inference calls.
        require(enabled, "INF_DISABLED");
        require(packed.length == uint256(nFeatures) * 4, "FEAT_LEN");
        require(numChunks > 0, "NO_CHUNKS");

        // Check table magic
        _requireChunkMagic(tablePtr, "TABLE_CODE");

        int256 acc = int256(baseQ);

        uint256 pow2 = uint256(1) << uint256(depth);
        uint256 internalNodes = pow2 - 1;
        uint256 perTree = internalNodes * 8 + pow2 * 4;

        // trees start at offset 24
        uint256 treesOff = 24;

        for (uint256 t = 0; t < nTrees; t++) {
            uint256 treeBase = treesOff + t * perTree;
            uint256 idx = 0;

            for (uint256 lvl = 0; lvl < depth; lvl++) {
                uint256 nodeOff = treeBase + idx * 8;

                uint16 f = _readU16Model(tablePtr, chunkSize, nodeOff);
                int32 thr = _readI32Model(tablePtr, chunkSize, nodeOff + 2);

                int32 xq = _readI32LE(packed, uint256(f) * 4);
                if (xq > thr) idx = idx * 2 + 2;
                else idx = idx * 2 + 1;
            }

            uint256 leafIndex = idx - internalNodes;
            uint256 leafBase = treeBase + internalNodes * 8;
            uint256 leafOff = leafBase + leafIndex * 4;

            int32 leafQ = _readI32Model(tablePtr, chunkSize, leafOff);
            acc += int256(leafQ);
        }

        return acc;
    }

    function _readI32LE(bytes calldata b, uint256 off) internal pure returns (int32) {
        require(off + 4 <= b.length, "OOB");
        uint32 v =
            uint32(uint8(b[off])) |
            (uint32(uint8(b[off + 1])) << 8) |
            (uint32(uint8(b[off + 2])) << 16) |
            (uint32(uint8(b[off + 3])) << 24);
        return int32(int256(uint256(v)));
    }

    function _requireChunkMagic(address ptr, string memory err) internal view {
        require(ptr.code.length >= 4, err);
        bytes4 m;
        assembly ("memory-safe") {
            let p := mload(0x40)
            extcodecopy(ptr, p, 0, 4)
            m := mload(p)
        }
        require(m == CHUNK_MAGIC, err);
    }

    function _chunkPtrAt(address tablePtr, uint256 chunkIdx) internal view returns (address ptr) {
        // read 32-byte slot from table runtime code at offset 4 + chunkIdx*32
        uint256 src = 4 + chunkIdx * 32;
        bytes32 word;
        assembly ("memory-safe") {
            let p := mload(0x40)
            extcodecopy(tablePtr, p, src, 32)
            word := mload(p)
        }
        ptr = address(uint160(uint256(word)));
        require(ptr != address(0), "BAD_PTR");
        _requireChunkMagic(ptr, "CHUNK_CODE");
    }

    function _readBytes(address tablePtr, uint32 chunkSize, uint256 off, uint256 n) internal view returns (bytes32 outWord) {
        // reads up to 32 bytes starting at off, returns in lowest bytes of outWord
        require(n > 0 && n <= 32, "READN");
        uint256 csz = uint256(chunkSize);

        uint256 chunkIdx = off / csz;
        uint256 inChunk = off % csz;

        address ptr = _chunkPtrAt(tablePtr, chunkIdx);

        // if within one chunk
        if (inChunk + n <= csz) {
            assembly ("memory-safe") {
                let p := mload(0x40)
                extcodecopy(ptr, p, add(4, inChunk), n)
                outWord := mload(p)
            }
        } else {
            // boundary: read first part then second part
            uint256 n1 = csz - inChunk;
            uint256 n2 = n - n1;

            bytes memory tmp = new bytes(n);
            assembly ("memory-safe") {
                extcodecopy(ptr, add(tmp, 32), add(4, inChunk), n1)
            }
            address ptr2 = _chunkPtrAt(tablePtr, chunkIdx + 1);
            assembly ("memory-safe") {
                extcodecopy(ptr2, add(add(tmp, 32), n1), 4, n2)
                outWord := mload(add(tmp, 32))
            }
        }
    }

    function _readU16Model(address tablePtr, uint32 chunkSize, uint256 off) internal view returns (uint16 v) {
        bytes32 w = _readBytes(tablePtr, chunkSize, off, 2);
        uint256 b0 = uint8(bytes1(w));
        uint256 b1 = uint8(bytes1(w << 8));
        v = uint16(b0 | (b1 << 8));
    }

    function _readU8Model(address tablePtr, uint32 chunkSize, uint256 off) internal view returns (uint8 v) {
        bytes32 w = _readBytes(tablePtr, chunkSize, off, 1);
        v = uint8(bytes1(w));
    }

    function _readU32Model(address tablePtr, uint32 chunkSize, uint256 off) internal view returns (uint32 v) {
        bytes32 w = _readBytes(tablePtr, chunkSize, off, 4);
        uint256 b0 = uint8(bytes1(w));
        uint256 b1 = uint8(bytes1(w << 8));
        uint256 b2 = uint8(bytes1(w << 16));
        uint256 b3 = uint8(bytes1(w << 24));
        v = uint32(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
    }

    function _readI32Model(address tablePtr, uint32 chunkSize, uint256 off) internal view returns (int32 v) {
        bytes32 w = _readBytes(tablePtr, chunkSize, off, 4);
        uint256 b0 = uint8(bytes1(w));
        uint256 b1 = uint8(bytes1(w << 8));
        uint256 b2 = uint8(bytes1(w << 16));
        uint256 b3 = uint8(bytes1(w << 24));
        uint32 u = uint32(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
        v = int32(int256(uint256(u)));
    }


    // ---- EIP-712 helpers ----
    function _domainSeparatorV4() internal view returns (bytes32) {
        return keccak256(abi.encode(
            _EIP712DOMAIN_TYPEHASH,
            _NAME_HASH,
            _VERSION_HASH,
            block.chainid,
            address(this)
        ));
    }

    function _hashTypedDataV4(bytes32 structHash) internal view returns (bytes32) {
        return keccak256(abi.encodePacked(hex"1901", _domainSeparatorV4(), structHash));
    }

    function _recover(bytes32 digest, bytes calldata sig) internal pure returns (address) {
        if (sig.length != 65) return address(0);
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly ("memory-safe") {
            r := calldataload(sig.offset)
            s := calldataload(add(sig.offset, 32))
            v := byte(0, calldataload(add(sig.offset, 64)))
        }
        if (v < 27) v += 27;
        if (v != 27 && v != 28) return address(0);
        // EIP-2: enforce lower-S malleability
        if (uint256(s) > 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0) return address(0);
        return ecrecover(digest, v, r, s);
    }

    // ---- no-op sinks (remove solc "unused" warnings without changing behavior) ----
    function _sinkA(address) private pure {}
    function _sinkU(uint256) private pure {}
    function _sinkI(int256) private pure {}
    function _sinkB32(bytes32) private pure {}
}
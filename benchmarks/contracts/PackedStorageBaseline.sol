// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../../contracts/ForestRuntime.sol";

/// @notice Local benchmark only: immutable-by-interface canonical bytes in SSTORE.
/// @dev The same ModelRegistry supplies the same scalar runtime metadata. This
/// contract is not a replacement for the production marketplace or v2 runtime.
contract PackedStorageBaseline {
    IModelRegistryRuntime public immutable registry;
    mapping(bytes32 => bytes) private models;

    constructor(address registryAddress) {
        registry = IModelRegistryRuntime(registryAddress);
    }

    function write(bytes32 modelId, bytes calldata core) external {
        require(models[modelId].length == 0, "EXISTS");
        require(core.length >= 24, "SHORT");
        models[modelId] = core;
    }

    function readModel(bytes32 modelId) external view returns (bytes memory) {
        return models[modelId];
    }

    function predictView(bytes32 modelId, bytes calldata packed) external view returns (int256) {
        // Match the production scalar entry point's first registry lookup and
        // access checks; _predict repeats the lookup as ForestRuntime does.
        (,,,,,,,,, bool enabled, uint8 mode,,) = registry.getModelRuntime(modelId);
        require(enabled, "INF_DISABLED");
        require(mode != 2, "PAID_ONLY");
        return _predict(modelId, packed);
    }

    function _predict(bytes32 modelId, bytes calldata packed) private view returns (int256) {
        (
            , , uint32 numChunks, uint32 totalBytes,
            uint16 nFeatures, uint16 nTrees, uint16 depth,
            int32 baseQ, , bool enabled, , ,
        ) = registry.getModelRuntime(modelId);
        require(enabled, "INF_DISABLED");
        require(packed.length == uint256(nFeatures) * 4, "FEAT_LEN");
        require(numChunks > 0, "NO_CHUNKS");
        bytes storage core = models[modelId];
        require(core.length == totalBytes && core.length >= 32, "MODEL_LEN");
        uint256 dataSlot;
        assembly ("memory-safe") {
            mstore(0, core.slot)
            dataSlot := keccak256(0, 32)
        }

        int256 acc = int256(baseQ);
        uint256 pow2 = uint256(1) << uint256(depth);
        uint256 internalNodes = pow2 - 1;
        uint256 perTree = internalNodes * 8 + pow2 * 4;
        for (uint256 t = 0; t < nTrees; t++) {
            uint256 treeBase = 24 + t * perTree;
            uint256 idx = 0;
            for (uint256 level = 0; level < depth; level++) {
                uint256 nodeOff = treeBase + idx * 8;
                uint16 feature = _readU16(dataSlot, nodeOff);
                int32 threshold = _readI32(dataSlot, nodeOff + 2);
                int32 xq = _readInput(packed, uint256(feature) * 4);
                if (xq > threshold) idx = idx * 2 + 2;
                else idx = idx * 2 + 1;
            }
            uint256 leafOff = treeBase + internalNodes * 8 + (idx - internalNodes) * 4;
            acc += int256(_readI32(dataSlot, leafOff));
        }
        return acc;
    }

    // A packed 32-byte word is fetched once per primitive read. Explicit SLOAD
    // prevents a deliberately inefficient byte-by-byte storage comparator.
    function _readWord(uint256 dataSlot, uint256 off, uint256 width) private view returns (bytes32 word) {
        assembly ("memory-safe") {
            let slot := add(dataSlot, div(off, 32))
            let shift := mul(mod(off, 32), 8)
            word := shl(shift, sload(slot))
            if gt(add(mod(off, 32), width), 32) {
                word := or(word, shr(sub(256, shift), sload(add(slot, 1))))
            }
        }
    }

    function _readU16(uint256 dataSlot, uint256 off) private view returns (uint16) {
        bytes32 word = _readWord(dataSlot, off, 2);
        return uint16(uint8(bytes1(word))) | (uint16(uint8(bytes1(word << 8))) << 8);
    }

    function _readI32(uint256 dataSlot, uint256 off) private view returns (int32) {
        bytes32 word = _readWord(dataSlot, off, 4);
        uint32 value = uint32(uint8(bytes1(word)))
            | (uint32(uint8(bytes1(word << 8))) << 8)
            | (uint32(uint8(bytes1(word << 16))) << 16)
            | (uint32(uint8(bytes1(word << 24))) << 24);
        return int32(int256(uint256(value)));
    }

    function _readInput(bytes calldata packed, uint256 off) private pure returns (int32) {
        require(off + 4 <= packed.length, "OOB");
        uint32 value = uint32(uint8(packed[off]))
            | (uint32(uint8(packed[off + 1])) << 8)
            | (uint32(uint8(packed[off + 2])) << 16)
            | (uint32(uint8(packed[off + 3])) << 24);
        return int32(int256(uint256(value)));
    }
}

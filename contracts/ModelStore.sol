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

/// @notice Stores arbitrary byte chunks on-chain by deploying a tiny pointer contract
///         whose runtime bytecode is: MAGIC(4 bytes) || DATA.
///
/// Design goals:
/// - On-chain writes avoid SSTORE-heavy storage (cheaper for medium blobs).
/// - Reads are easy off-chain via eth_getCode / extcodecopy.
/// - Compatible with evmVersion=istanbul (no post-Istanbul opcodes required).
contract ModelStore {
    // 0x47 0x4c 0x31 0x43 = "GL1C" (GenesisL1 Chunk)
    uint32 public constant MAGIC = 0x474c3143;

    event ChunkWritten(address indexed pointer, uint256 size);

    /// @notice Deploy a new pointer contract containing (MAGIC || data) as its runtime bytecode.
    /// @dev Runtime code size must respect EIP-170 (24,576 bytes). We reserve 4 bytes for MAGIC.
    function write(bytes calldata data) external returns (address pointer) {
        uint256 dlen = data.length;
        require(dlen <= 24_572, "CHUNK_TOO_LARGE");

        uint256 rlen = dlen + 4; // runtime length
        bytes memory init = new bytes(14 + rlen);

        assembly ("memory-safe") {
            let p := add(init, 32)

            // Minimal init-code:
            // PUSH2 rlen
            // PUSH1 0x0e
            // PUSH1 0x00
            // CODECOPY
            // PUSH2 rlen
            // PUSH1 0x00
            // RETURN
            mstore8(p, 0x61)
            mstore8(add(p, 1), shr(8, rlen))
            mstore8(add(p, 2), and(rlen, 0xff))
            mstore8(add(p, 3), 0x60)
            mstore8(add(p, 4), 0x0e)
            mstore8(add(p, 5), 0x60)
            mstore8(add(p, 6), 0x00)
            mstore8(add(p, 7), 0x39)
            mstore8(add(p, 8), 0x61)
            mstore8(add(p, 9), shr(8, rlen))
            mstore8(add(p, 10), and(rlen, 0xff))
            mstore8(add(p, 11), 0x60)
            mstore8(add(p, 12), 0x00)
            mstore8(add(p, 13), 0xf3)

            // Runtime start
            let r := add(p, 14)

            // MAGIC (4 bytes)
            mstore(r, shl(224, 0x474c3143))

            // DATA
            calldatacopy(add(r, 4), data.offset, dlen)

            // CREATE
            pointer := create(0, p, mload(init))
        }

        require(pointer != address(0), "CREATE_FAIL");
        emit ChunkWritten(pointer, dlen);
    }
}

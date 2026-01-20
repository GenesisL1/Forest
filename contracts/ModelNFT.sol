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

interface IERC721Receiver {
    function onERC721Received(address operator, address from, uint256 tokenId, bytes calldata data) external returns (bytes4);
}

// Minimal ERC-721 + enumerable ownership for GenesisL1 Forest Model NFTs.
// Metadata (title/description/icon/features) is stored fully on-chain.
contract ModelNFT {
    string public name;
    string public symbol;

    address public immutable registry;

    uint256 public totalMinted;

    mapping(uint256 => address) private _ownerOf;
    mapping(address => uint256) private _balanceOf;

    mapping(uint256 => address) public getApproved;
    mapping(address => mapping(address => bool)) public isApprovedForAll;

    // enumerable
    mapping(address => uint256[]) private _ownedTokens;
    mapping(uint256 => uint256) private _ownedTokensIndex;

    // metadata
    mapping(uint256 => string) public title;
    mapping(uint256 => string) public description;
    mapping(uint256 => string) public features;
    mapping(uint256 => bytes) public icon;

    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);
    event Approval(address indexed owner, address indexed spender, uint256 indexed tokenId);
    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);

    modifier onlyRegistry() {
        require(msg.sender == registry, "REG");
        _;
    }

    constructor(address _registry, string memory _name, string memory _symbol) {
        require(_registry != address(0), "REG0");
        registry = _registry;
        name = _name;
        symbol = _symbol;
    }

    function ownerOf(uint256 tokenId) public view returns (address) {
        address o = _ownerOf[tokenId];
        require(o != address(0), "NF");
        return o;
    }

    function balanceOf(address owner) external view returns (uint256) {
        require(owner != address(0), "ADDR0");
        return _balanceOf[owner];
    }

    function tokenOfOwnerByIndex(address owner, uint256 index) external view returns (uint256) {
        require(index < _ownedTokens[owner].length, "OOB");
        return _ownedTokens[owner][index];
    }

    function approve(address spender, uint256 tokenId) external {
        address o = ownerOf(tokenId);
        require(msg.sender == o || isApprovedForAll[o][msg.sender], "NOAUTH");
        getApproved[tokenId] = spender;
        emit Approval(o, spender, tokenId);
    }

    function setApprovalForAll(address operator, bool approved) external {
        isApprovedForAll[msg.sender][operator] = approved;
        emit ApprovalForAll(msg.sender, operator, approved);
    }

    function _isApprovedOrOwner(address spender, uint256 tokenId) internal view returns (bool) {
        address o = _ownerOf[tokenId];
        return (spender == o || getApproved[tokenId] == spender || isApprovedForAll[o][spender]);
    }

    function transferFrom(address from, address to, uint256 tokenId) public {
        require(to != address(0), "TO0");
        require(_ownerOf[tokenId] == from, "FROM");
        require(_isApprovedOrOwner(msg.sender, tokenId), "NOAUTH");
        _transfer(from, to, tokenId);
    }

    function safeTransferFrom(address from, address to, uint256 tokenId) external {
        safeTransferFrom(from, to, tokenId, "");
    }

    function safeTransferFrom(address from, address to, uint256 tokenId, bytes memory data) public {
        transferFrom(from, to, tokenId);
        if (to.code.length > 0) {
            bytes4 ret = IERC721Receiver(to).onERC721Received(msg.sender, from, tokenId, data);
            require(ret == IERC721Receiver.onERC721Received.selector, "UNSAFE");
        }
    }

    function _transfer(address from, address to, uint256 tokenId) internal {
        // clear approval
        if (getApproved[tokenId] != address(0)) {
            getApproved[tokenId] = address(0);
            emit Approval(from, address(0), tokenId);
        }

        // remove from old owner enumeration
        _removeOwnedToken(from, tokenId);

        _balanceOf[from] -= 1;
        _balanceOf[to] += 1;
        _ownerOf[tokenId] = to;

        _ownedTokensIndex[tokenId] = _ownedTokens[to].length;
        _ownedTokens[to].push(tokenId);

        emit Transfer(from, to, tokenId);
    }

    function _removeOwnedToken(address from, uint256 tokenId) internal {
        uint256 lastIdx = _ownedTokens[from].length - 1;
        uint256 idx = _ownedTokensIndex[tokenId];

        if (idx != lastIdx) {
            uint256 lastToken = _ownedTokens[from][lastIdx];
            _ownedTokens[from][idx] = lastToken;
            _ownedTokensIndex[lastToken] = idx;
        }
        _ownedTokens[from].pop();
        delete _ownedTokensIndex[tokenId];
    }

    function mintTo(
        address to,
        string memory _title,
        string memory _description,
        bytes memory _icon,
        string memory _features
    ) external onlyRegistry returns (uint256 tokenId) {
        require(to != address(0), "TO0");
        require(bytes(_title).length > 0, "TITLE");
        require(bytes(_description).length > 0, "DESC");
        require(_icon.length > 0, "ICON");

        tokenId = ++totalMinted;
        _ownerOf[tokenId] = to;
        _balanceOf[to] += 1;
        _ownedTokensIndex[tokenId] = _ownedTokens[to].length;
        _ownedTokens[to].push(tokenId);

        title[tokenId] = _title;
        description[tokenId] = _description;
        icon[tokenId] = _icon;
        features[tokenId] = _features;

        emit Transfer(address(0), to, tokenId);
    }

    function burn(uint256 tokenId) external onlyRegistry {
        address o = _ownerOf[tokenId];
        require(o != address(0), "NF");

        // clear approval
        if (getApproved[tokenId] != address(0)) {
            getApproved[tokenId] = address(0);
            emit Approval(o, address(0), tokenId);
        }

        _removeOwnedToken(o, tokenId);
        _balanceOf[o] -= 1;
        delete _ownerOf[tokenId];

        delete title[tokenId];
        delete description[tokenId];
        delete icon[tokenId];
        delete features[tokenId];

        emit Transfer(o, address(0), tokenId);
    }
}

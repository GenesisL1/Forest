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

interface IModelNFT {
    function ownerOf(uint256 tokenId) external view returns (address);
    function getApproved(uint256 tokenId) external view returns (address);
    function isApprovedForAll(address owner, address operator) external view returns (bool);
    function transferFrom(address from, address to, uint256 tokenId) external;
}

contract ModelMarketplace is SimpleOwnable {
    IModelNFT public immutable nft;

    struct Listing {
        bool listed;
        uint256 priceWei;
        address seller;
    }

    uint256 public listingFeeWei;

    mapping(uint256 => Listing) public listings;
    uint256[] private listedIds;
    mapping(uint256 => uint256) private listedIndex; // tokenId => idx+1

    event Listed(uint256 indexed tokenId, uint256 priceWei, address indexed seller);
    event Cancelled(uint256 indexed tokenId);
    event Bought(uint256 indexed tokenId, uint256 priceWei, address indexed seller, address indexed buyer);

    constructor(address nftAddr, address owner_) SimpleOwnable(owner_) {
        require(nftAddr != address(0), "NFT0");
        nft = IModelNFT(nftAddr);
    }

    function setListingFeeWei(uint256 feeWei) external onlyOwner {
        listingFeeWei = feeWei;
    }

    function getListing(uint256 tokenId) external view returns (bool listed, uint256 priceWei, address seller) {
        Listing memory l = listings[tokenId];
        return (l.listed, l.priceWei, l.seller);
    }

    function getListingsPage(uint256 cursor, uint256 limit) external view returns (uint256[] memory tokenIds, uint256[] memory prices, address[] memory sellers, uint256 nextCursor) {
        uint256 n = listedIds.length;
        if (cursor >= n) {
            return (new uint256[](0), new uint256[](0), new address[](0), 0);
        }
        uint256 end = cursor + limit;
        if (end > n) end = n;
        uint256 m = end - cursor;

        tokenIds = new uint256[](m);
        prices = new uint256[](m);
        sellers = new address[](m);

        for (uint256 i = 0; i < m; i++) {
            uint256 tid = listedIds[cursor + i];
            Listing memory l = listings[tid];
            tokenIds[i] = tid;
            prices[i] = l.priceWei;
            sellers[i] = l.seller;
        }

        nextCursor = (end >= n) ? 0 : end;
    }

    function list(uint256 tokenId, uint256 priceWei) external payable {
        require(msg.value == listingFeeWei, "LIST_FEE");
        require(priceWei > 0, "PRICE0");
        address seller = nft.ownerOf(tokenId);
        require(seller == msg.sender, "NOT_OWNER");

        bool approved = (nft.getApproved(tokenId) == address(this)) || nft.isApprovedForAll(seller, address(this));
        require(approved, "APPROVE_MARKET");

        Listing storage l = listings[tokenId];
        if (!l.listed) {
            l.listed = true;
            l.seller = seller;
            _addListed(tokenId);
        } else {
            // keep seller as original
            require(l.seller == seller, "SELLER_CHANGED");
        }
        l.priceWei = priceWei;

        if (listingFeeWei > 0) {
            (bool ok,) = owner.call{value: listingFeeWei}("");
            require(ok, "FEE_SEND");
        }

        emit Listed(tokenId, priceWei, seller);
    }

    function cancel(uint256 tokenId) external {
        Listing storage l = listings[tokenId];
        require(l.listed, "NOT_LISTED");
        address seller = l.seller;
        require(msg.sender == seller || msg.sender == owner, "NOAUTH");
        _removeListed(tokenId);
        delete listings[tokenId];
        emit Cancelled(tokenId);
    }

    function buy(uint256 tokenId) external payable {
        Listing storage l = listings[tokenId];
        require(l.listed, "NOT_LISTED");
        require(msg.value == l.priceWei, "BAD_PRICE");

        address seller = l.seller;
        // seller must still own it
        require(nft.ownerOf(tokenId) == seller, "SELLER_NOT_OWNER");

        // transfer NFT
        nft.transferFrom(seller, msg.sender, tokenId);

        // payout
        (bool ok,) = seller.call{value: msg.value}("");
        require(ok, "PAY_FAIL");

        _removeListed(tokenId);
        delete listings[tokenId];

        emit Bought(tokenId, msg.value, seller, msg.sender);
    }

    function _addListed(uint256 tokenId) internal {
        if (listedIndex[tokenId] != 0) return;
        listedIds.push(tokenId);
        listedIndex[tokenId] = listedIds.length; // idx+1
    }

    function _removeListed(uint256 tokenId) internal {
        uint256 idx1 = listedIndex[tokenId];
        if (idx1 == 0) return;
        uint256 idx = idx1 - 1;
        uint256 last = listedIds[listedIds.length - 1];
        if (idx != listedIds.length - 1) {
            listedIds[idx] = last;
            listedIndex[last] = idx + 1;
        }
        listedIds.pop();
        delete listedIndex[tokenId];
    }
}

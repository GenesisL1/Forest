#!/usr/bin/env python3
"""Guard release-source inventory and deployment/client manifest consistency."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE_CONTRACT_SOURCE_SHA256 = {
    # This release source includes an attribution header absent from the deployed
    # source-base commit. Solidity metadata can change when comments change, so
    # this digest freezes release source only; it is not a deployed-bytecode claim.
    "Base64.sol": "d6327e0fbb4dd22a4c6042b012254ebc0ccee0d5cf61c9d4924c78db9dadafc5",
    "ForestRuntime.sol": "72bfe83900ad7d74023a40945ec7bfe7ab123dbe274583acccd539886978a222",
    "ModelMarketplace.sol": "cb5dbe9e8a1d4db6cf104307e32e41f6f6857c97f522ab31d5d0b6588f418ef0",
    "ModelNFT.sol": "b190e8a4f363bc3050ab9d3d35f41c43b63bccef9670e6c51415713b7cbf9ff3",
    "ModelRegistry.sol": "bc1b5e02e28a98af780efc0f8331816e4acc995f0b0c209b1eebb90d35be0812",
    "ModelStore.sol": "0172f17dc9c3d549867f5fe02ff0e43ad09da37f5144120863c6e1ab9e003d61",
    "SimpleOwnable.sol": "03a4ea9b113a9d14ca8e4380ce017f0cc20c9b7b8962861fbbc7301355f83394",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DeploymentIdentityTests(unittest.TestCase):
    def test_release_contract_source_inventory_is_frozen(self) -> None:
        actual = {
            path.name: sha256(path)
            for path in sorted((ROOT / "contracts").glob("*.sol"))
        }
        self.assertEqual(actual, RELEASE_CONTRACT_SOURCE_SHA256)

    def test_manifest_and_client_defaults_agree(self) -> None:
        manifest = json.loads(
            (ROOT / "deployments" / "genesisl1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["network"]["chainId"], 29)
        self.assertEqual(
            manifest["network"]["rpcUrl"],
            "https://rpc.genesisl1.org",
        )
        expected = {
            record["address"].lower()
            for record in manifest["contracts"].values()
        }
        self.assertEqual(len(expected), 5)

        for relative in ("src/common.js", "mint_model.py"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            addresses = {
                match.lower()
                for match in re.findall(r"0x[0-9a-fA-F]{40}", source)
            }
            self.assertTrue(
                expected.issubset(addresses),
                f"{relative} does not contain every manifest address",
            )

    def test_publisher_validation_and_secret_exclusions_are_wired(self) -> None:
        source = (ROOT / "mint_model.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("parse_gl1f_package(raw)"),
            source.index('log(f"Deploying {n_chunks} chunks'),
        )
        ignore = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
        for required in (".env", "owner_key.txt", "mint_state.json", "S1.json"):
            self.assertIn(required, ignore)
        self.assertNotIn(
            "model_mint.py",
            (ROOT / "README.md").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

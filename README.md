<!--
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
-->

# GenesisL1 Forest (GL1F)
Link: https://GL1F.com

A browser-only studio for:
- training GBDT models (in-browser): regression + binary + multiclass + multilabel classification
- deploying models on GenesisL1 as ERC-721 Model NFTs
- on-chain inference (view call + optional paid tx)
- AI store listing/buying
- on-chain Terms + Creative Commons license enforcement
- on-chain title-word search index

## Run locally
You must serve this folder over HTTP (not file://) so module imports work.

### Option A: Python
```bash
python3 -m http.server 8080
```
Open:
- http://localhost:8080/
- http://localhost:8080/forest.html

## Network
GenesisL1:
- chainId: 29
- default RPC: https://rpc.genesisl1.org

## Notes
- Model bytes are stored as chunk contracts. Pointer-table and chunks use runtime magic "GL1C".
- Model format "GL1F" v1 is used for regression + binary classification.
- Model format "GL1F" v2 is used for multiclass + multilabel classification (vector-output logits).
- Model metadata (title/description/icon/features) is stored on-chain in the NFT.

## License

This project is released under the MIT License. See the `LICENSE` file for details.

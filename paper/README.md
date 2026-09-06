# GL1F research paper

**GL1F: Reproducible Integer Tree-Ensemble Inference on the EVM**
Mikhail Fedorov. Author manuscript, 6 September 2026; not peer reviewed.

The manuscript is [`../GL1F.pdf`](../GL1F.pdf). Supplementary methods and
proofs are in [`GL1F_Formal_Supplement.pdf`](GL1F_Formal_Supplement.pdf).
The manuscript uses Ledger's author-template typography and bibliography
style. Publication metadata are omitted because the article has not been
accepted or published by the journal. Template provenance is recorded in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Build

From the repository root, with TeX Live and latexmk installed:

```bash
make pdfs
```

`main.tex`, `architecture.tex`, `references.bib`, `ledger.cls`,
`ledger-manuscript.sty`, and `ledgerbib.bst` build the main article.
`supplement.tex` and `formal_results.tex` build the supplement.

## Reproduce the evidence

```bash
npm ci
make verify
node benchmarks/storage_comparison.mjs
```

See [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for the pinned environment,
recorded measurements, and offline deployment-archive verification.

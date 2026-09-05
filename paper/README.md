# GL1F research paper

This directory contains the source of:

**Artifact-to-Execution Assurance for Canonical Integer GBDTs on the EVM**
Mikhail Fedorov, technical report v0.2.3, 5 September 2026.

The canonical paper is [`../GL1F.pdf`](../GL1F.pdf). The formal supplement is
[`GL1F_Formal_Supplement.pdf`](GL1F_Formal_Supplement.pdf).

## Build

From the repository root:

```bash
make pdfs
```

The build uses `main.tex`, `architecture.tex`, `references.bib`,
`supplement.tex`, and `formal_results.tex`. It writes the two PDFs named above.

## Executable evidence

```bash
npm ci
make verify
```

See [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for the environment,
individual commands, checked-in result records, and evidence limits.

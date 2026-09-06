# Third-party notices

The root `LICENSE` applies to original GL1F software except where a file or
directory states otherwise.

## Base64 encoder

`contracts/Base64.sol` contains an encoding routine adapted from Brecht
Devos's MIT-licensed `Base64.sol`:

- source: <https://github.com/Brechtpd/base64/blob/e78d9fd951e7b0977ddca77d92dc85183770daf4/base64.sol>
- license: <https://github.com/Brechtpd/base64/blob/e78d9fd951e7b0977ddca77d92dc85183770daf4/LICENSE>

The upstream attribution and MIT terms remain in force.

## Trading-desk examples

Files under `examples/dapps/trading_desk/` are distributed under the terms
stated in that directory. See
[`examples/dapps/trading_desk/LICENSE.md`](examples/dapps/trading_desk/LICENSE.md).

## Ledger manuscript template

`paper/ledger.cls` and `paper/ledgerbib.bst` come from
[Ledger's LaTeX author template](https://ledger.pitt.edu/ojs/ledger/libraryFiles/downloadPublic/2),
linked in the journal's
[submission instructions](https://ledgerjournal.org/ojs/ledger/about/submissions)
and retrieved on 6 September 2026. They are third-party formatting resources,
not original GL1F software. The upstream class does not contain a separate
license declaration. The bibliography style retains its Patrick W. Daly
copyright notice and LaTeX Project Public License terms. The bibliography style
is unchanged. The class has two compatibility corrections: the obsolete
`caption` option `compatibility=true` is replaced by `compatibility=false`,
and a misplaced `\qedsymbol` token is removed from the `myproof` environment
definition to prevent execution of `\endproof` while loading the class.

`paper/ledger-manuscript.sty` supplies GL1F's manuscript settings separately,
including omission of the template's sample publication and review metadata.

## Dependencies

NumPy, ethers, Ganache, Solidity, and other dependencies are separate projects
distributed under their respective licenses. They are not relicensed by this
repository.

#!/usr/bin/env bash
set -euo pipefail

export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1788652800}"
export FORCE_SOURCE_DATE=1

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
paper_dir="$repo_dir/paper"
build_root="$(mktemp -d /tmp/gl1f-pdf-build.XXXXXX)"
trap 'rm -rf "$build_root"' EXIT

main_build_dir="$build_root/main"
supplement_build_dir="$build_root/supplement"
mkdir -p "$main_build_dir" "$supplement_build_dir"

(
  cd "$paper_dir"
  latexmk -C main.tex >/dev/null
  TEXINPUTS="$main_build_dir:${TEXINPUTS:-}" \
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -output-directory="$main_build_dir" main.tex
)

(
  cd "$paper_dir"
  latexmk -C supplement.tex >/dev/null
  TEXINPUTS="$supplement_build_dir:${TEXINPUTS:-}" \
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -output-directory="$supplement_build_dir" supplement.tex
  TEXINPUTS="$supplement_build_dir:${TEXINPUTS:-}" \
    latexmk -g -pdf -interaction=nonstopmode -halt-on-error \
      -output-directory="$supplement_build_dir" supplement.tex
)

cp "$main_build_dir/main.pdf" "$repo_dir/GL1F.pdf"
cp "$supplement_build_dir/supplement.pdf" \
  "$paper_dir/GL1F_Formal_Supplement.pdf"

pdfinfo "$repo_dir/GL1F.pdf" >/dev/null
pdfinfo "$paper_dir/GL1F_Formal_Supplement.pdf" >/dev/null

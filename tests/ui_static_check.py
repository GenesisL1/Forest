#!/usr/bin/env python3
"""Dependency-free structural checks for the GL1F browser interface.

The script deliberately separates hard structural errors from publication
warnings.  It does not claim WCAG conformance and it does not execute wallet,
RPC, training, or model-inference code.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML_FILES = sorted(ROOT.glob("*.html"))
CSS_FILE = ROOT / "style.css"


class StaticCheckParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.images_without_alt: list[int] = []
        self.blank_targets_without_rel: list[int] = []
        self.external_scripts_without_integrity: list[tuple[int, str]] = []
        self.buttons_without_type: list[int] = []
        self.label_for_values: set[str] = set()
        self.wrapped_label_controls: set[str] = set()
        self.control_ids: list[tuple[int, str, str]] = []
        self.live_region_ids: set[str] = set()
        self.progressbar_ids: set[str] = set()
        self.label_depth = 0
        self.title_depth = 0
        self.title_text: list[str] = []
        self.lang = ""
        self.has_viewport = False
        self.has_description = False
        self.has_main = False
        self.has_nav = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}
        line, _ = self.getpos()

        if tag == "html":
            self.lang = attrs_dict.get("lang", "").strip()
        if tag == "meta":
            name = attrs_dict.get("name", "").lower()
            if name == "viewport":
                self.has_viewport = True
            elif name == "description" and attrs_dict.get("content", "").strip():
                self.has_description = True
        if tag == "title":
            self.title_depth += 1
        if tag == "main":
            self.has_main = True
        if tag == "nav":
            self.has_nav = True

        element_id = attrs_dict.get("id", "").strip()
        if element_id:
            self.ids.append(element_id)
            if attrs_dict.get("aria-live", "").strip():
                self.live_region_ids.add(element_id)
            if attrs_dict.get("role", "").strip().lower() == "progressbar":
                self.progressbar_ids.add(element_id)

        if tag == "img" and "alt" not in attrs_dict:
            self.images_without_alt.append(line)

        if attrs_dict.get("target", "").lower() == "_blank":
            rel_tokens = set(attrs_dict.get("rel", "").lower().split())
            if not ({"noopener", "noreferrer"} & rel_tokens):
                self.blank_targets_without_rel.append(line)

        if tag == "script":
            src = attrs_dict.get("src", "")
            if src.startswith(("https://", "http://")) and not attrs_dict.get("integrity"):
                self.external_scripts_without_integrity.append((line, src))

        if tag == "button" and "type" not in attrs_dict:
            self.buttons_without_type.append(line)

        if tag == "label" and attrs_dict.get("for", "").strip():
            self.label_for_values.add(attrs_dict["for"].strip())
        if tag == "label":
            self.label_depth += 1

        if tag in {"input", "select", "textarea"}:
            control_id = attrs_dict.get("id", "").strip()
            if control_id and self.label_depth:
                self.wrapped_label_controls.add(control_id)
            accessible_name = (
                attrs_dict.get("aria-label", "").strip()
                or attrs_dict.get("aria-labelledby", "").strip()
                or attrs_dict.get("title", "").strip()
            )
            if control_id and not accessible_name:
                self.control_ids.append((line, tag, control_id))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self.title_depth:
            self.title_depth -= 1
        if tag == "label" and self.label_depth:
            self.label_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.title_depth:
            self.title_text.append(data)


def css_braces_balanced(text: str) -> bool:
    """Check braces while ignoring comments and quoted strings."""
    i = 0
    depth = 0
    quote = ""
    in_comment = False
    while i < len(text):
        pair = text[i : i + 2]
        char = text[i]
        if in_comment:
            if pair == "*/":
                in_comment = False
                i += 2
                continue
        elif quote:
            if char == "\\":
                i += 2
                continue
            if char == quote:
                quote = ""
        elif pair == "/*":
            in_comment = True
            i += 2
            continue
        elif char in {'"', "'"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0 and not quote and not in_comment


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    if not HTML_FILES:
        errors.append("No root HTML files found.")
    if not CSS_FILE.is_file():
        errors.append("style.css is missing.")
    else:
        css = CSS_FILE.read_text(encoding="utf-8")
        if not css_braces_balanced(css):
            errors.append("style.css has unbalanced braces/comments/quotes.")
        for required in (":focus-visible", "prefers-reduced-motion", "@media print"):
            if required not in css:
                errors.append(f"style.css is missing required resilience rule: {required}")

    for path in HTML_FILES:
        parser = StaticCheckParser()
        source = path.read_text(encoding="utf-8")
        parser.feed(source)
        parser.close()
        name = path.name

        duplicate_ids = [value for value, count in Counter(parser.ids).items() if count > 1]
        if duplicate_ids:
            errors.append(f"{name}: duplicate literal id(s): {', '.join(duplicate_ids)}")
        missing_label_targets = sorted(parser.label_for_values - set(parser.ids))
        if missing_label_targets:
            errors.append(
                f"{name}: label for= target(s) do not exist: "
                f"{', '.join(missing_label_targets)}."
            )
        if not parser.lang:
            errors.append(f"{name}: <html> is missing lang.")
        if not "".join(parser.title_text).strip():
            errors.append(f"{name}: non-empty <title> is missing.")
        if not parser.has_viewport:
            errors.append(f"{name}: viewport metadata is missing.")
        if parser.images_without_alt:
            errors.append(f"{name}: image(s) without alt at lines {parser.images_without_alt}.")
        if parser.blank_targets_without_rel:
            errors.append(
                f"{name}: target=_blank without noopener/noreferrer at lines "
                f"{parser.blank_targets_without_rel}."
            )

        if not parser.has_description:
            warnings.append(f"{name}: meta description is missing.")
        if not parser.has_main:
            warnings.append(f"{name}: no <main> landmark.")
        if not parser.has_nav:
            warnings.append(f"{name}: no <nav> landmark.")
        if parser.external_scripts_without_integrity:
            refs = ", ".join(f"line {line}" for line, _ in parser.external_scripts_without_integrity)
            warnings.append(f"{name}: external script(s) without SRI ({refs}).")
        if parser.buttons_without_type:
            warnings.append(
                f"{name}: {len(parser.buttons_without_type)} button(s) omit type=button."
            )

        unassociated = [
            (line, tag, control_id)
            for line, tag, control_id in parser.control_ids
            if (
                control_id not in parser.label_for_values
                and control_id not in parser.wrapped_label_controls
            )
        ]
        if unassociated:
            warnings.append(
                f"{name}: {len(unassociated)} controls have no explicit label association "
                "(wrapping labels and dynamic names require manual review)."
            )

        if name == "create.html":
            required_live_regions = {
                "walletPill",
                "trainStatusAnnouncer",
                "gl1fPill",
                "deployPill",
                "deployLog",
            }
            missing_live_regions = sorted(required_live_regions - parser.live_region_ids)
            if missing_live_regions:
                errors.append(
                    f"{name}: status/log region(s) are not live: "
                    f"{', '.join(missing_live_regions)}."
                )
            if "trainProgress" not in parser.progressbar_ids:
                errors.append(f"{name}: training progress is not exposed as a progressbar.")

        if name == "model.html":
            for required_id in ("verifyCoreBtn", "verifyCoreStatus"):
                if required_id not in parser.ids:
                    errors.append(f"{name}: missing model-verification control #{required_id}.")
            if "verifyCoreStatus" not in parser.live_region_ids:
                errors.append(f"{name}: model verification status is not a live region.")

        # Catch accidental reintroduction of the publication-facing spelling errors
        # without failing the structural check.
        for pattern, replacement in (
            (r"\bInteligence\b", "Intelligence"),
            (r"\bpowerfull\b", "powerful"),
            (r"\bper per inference\b", "per inference"),
        ):
            if re.search(pattern, source, flags=re.IGNORECASE):
                warnings.append(f"{name}: visible copy should use “{replacement}”.")

    model_page = ROOT / "src" / "model_page.js"
    if not model_page.is_file():
        errors.append("src/model_page.js is missing.")
    else:
        source = model_page.read_text(encoding="utf-8")
        for required in (
            "loadModelBytesFromChain",
            "decodeModel",
            "Number(network.chainId) !== 29",
            "Feature ${i} must be a finite number",
        ):
            if required not in source:
                errors.append(
                    f"src/model_page.js is missing verification boundary: {required}"
                )

    print(f"Checked {len(HTML_FILES)} HTML files and {CSS_FILE.name}.")
    for item in errors:
        print(f"ERROR: {item}")
    for item in warnings:
        print(f"WARN:  {item}")
    print(f"Result: {len(errors)} error(s), {len(warnings)} warning(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

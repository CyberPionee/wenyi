"""Pure consistency sweeps for residual translation defects.

These helpers only inspect source/target text and glossary mappings. They never
call models, never write state, and never publish targets. Callers may turn
findings into Autofix candidates through the existing shadow + before_hash chain.
"""

from __future__ import annotations

import re
from typing import Any

from ..glossary.store import GlossaryTerm, source_matches_text, term_match_sources

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text or "")


def scan_number_residue(source: str, target: str) -> dict[str, Any] | None:
    """Report digits present in source but missing from target (deterministic)."""
    if not (source or "").strip() or not (target or "").strip():
        return None
    target_numbers = set(_numbers(target))
    missing = [n for n in _numbers(source) if n not in target_numbers]
    if not missing:
        return None
    return {
        "kind": "number_residue",
        "missing_numbers": missing,
        "detail": f"Source numbers missing from translation: {', '.join(missing)}",
    }


def scan_untranslated_residue(source: str, target: str) -> dict[str, Any] | None:
    """Report source-language CJK runs left inside a non-CJK target (deterministic)."""
    if not (source or "").strip() or not (target or "").strip():
        return None
    # Only flag when the target is primarily Latin-script; CJK targets legitimately reuse CJK.
    latin = sum(1 for ch in target if ch.isascii() and ch.isalpha())
    cjk = sum(1 for ch in target if "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿")
    if latin < 10 or cjk == 0:
        return None
    leftover = re.findall(r"[぀-ヿ一-鿿]{2,}", target)
    if not leftover:
        return None
    sample = " / ".join(leftover[:3])
    return {
        "kind": "untranslated_residue",
        "leftover": leftover,
        "detail": f"Possible untranslated source text remains in translation: {sample}",
    }


def scan_term_drift(
    source: str,
    target: str,
    terms: list[GlossaryTerm],
) -> dict[str, Any] | None:
    """Report glossary source hits whose fixed target mapping is absent from the translation."""
    if not (source or "").strip() or not (target or "").strip() or not terms:
        return None
    for term in terms:
        if not term.target or not term.target.strip():
            continue
        keys = term_match_sources(term)
        if not any(source_matches_text(key, source) for key in keys):
            continue
        if term.target in target:
            continue
        return {
            "kind": "term_drift",
            "source_term": term.source,
            "expected_target": term.target,
            "detail": f"Glossary mapping not used in translation: {term.source} → {term.target}",
        }
    return None


def scan_segment(
    source: str,
    target: str,
    terms: list[GlossaryTerm] | None = None,
) -> list[dict[str, Any]]:
    """Run all deterministic residual scans for one segment."""
    findings: list[dict[str, Any]] = []
    for finding in (
        scan_number_residue(source, target),
        scan_untranslated_residue(source, target),
        scan_term_drift(source, target, terms or []),
    ):
        if finding is not None:
            findings.append(finding)
    return findings

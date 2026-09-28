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

# Letter-run detectors per script family. Length gates live in scan_untranslated_residue.
_SCRIPT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cjk", re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]+")),
    ("latin", re.compile(r"[A-Za-zÀ-ɏ]+")),
    ("cyrillic", re.compile(r"[Ѐ-ӿ]+")),
    ("greek", re.compile(r"[Ͱ-Ͽ]+")),
    ("arabic", re.compile(r"[؀-ۿ]+")),
    ("hebrew", re.compile(r"[֐-׿]+")),
    ("thai", re.compile(r"[฀-๿]+")),
    ("devanagari", re.compile(r"[ऀ-ॿ]+")),
)


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text or "")


def _script_letter_counts(text: str) -> dict[str, int]:
    counts = {name: 0 for name, _ in _SCRIPT_PATTERNS}
    for name, pattern in _SCRIPT_PATTERNS:
        counts[name] = sum(len(match.group()) for match in pattern.finditer(text))
    return counts


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
    """Report source-script runs left inside a target dominated by another script.

    Only non-dominant script runs that also appear verbatim in the source are
    flagged, so intentional loanwords that never appear in the source stay
    unflagged. Same-script translations stay unflagged.
    """
    if not (source or "").strip() or not (target or "").strip():
        return None
    target_counts = _script_letter_counts(target)
    dominant = max(target_counts, key=target_counts.get)  # type: ignore[arg-type]
    if target_counts[dominant] < 5:
        return None
    leftover: list[str] = []
    for name, pattern in _SCRIPT_PATTERNS:
        if name == dominant or target_counts[name] <= 0:
            continue
        min_len = 2 if name == "cjk" else 3
        for match in pattern.finditer(target):
            run = match.group()
            if len(run) >= min_len and run in source:
                leftover.append(run)
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

"""Pure consistency sweeps for residual translation defects.

These helpers only inspect source/target text and glossary mappings. They never
call models, never write state, and never publish targets. Callers may turn
findings into Autofix candidates through the existing shadow + before_hash chain.
"""

from __future__ import annotations

import re
from typing import Any

from ..glossary.store import (
    GlossaryTerm,
    _match_text,
    _source_pattern,
    term_match_sources,
)

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

# TLD / URL fragments that must not be treated as leftover prose.
_URL_TOKENS = frozenset(
    {
        "com",
        "org",
        "net",
        "edu",
        "gov",
        "www",
        "http",
        "https",
        "html",
        "htm",
        "php",
        "asp",
        "aspx",
        "jsp",
        "json",
        "xml",
        "css",
        "js",
        "pdf",
        "jpg",
        "png",
        "gif",
        "svg",
        "zip",
        "exe",
        "cgi",
        "shtml",
        "cfm",
        "mailto",
        "ftp",
    }
)


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text or "")


def _script_letter_counts(text: str) -> dict[str, int]:
    counts = {name: 0 for name, _ in _SCRIPT_PATTERNS}
    for name, pattern in _SCRIPT_PATTERNS:
        counts[name] = sum(len(match.group()) for match in pattern.finditer(text))
    return counts


def _looks_like_url_context(target: str, start: int, end: int) -> bool:
    """True when the match sits in URL/domain-like punctuation."""
    left = target[max(0, start - 3) : start]
    right = target[end : end + 3]
    if any(ch in left or ch in right for ch in ("/", ":", "@", "#", "?")):
        return True
    return left.endswith(".") or right.startswith(".")


def _is_proper_noun_or_url_token(run: str) -> bool:
    """Skip URL fragments and Title Case / ALLCAPS names that stay untranslated."""
    if run.lower() in _URL_TOKENS:
        return True
    letters = [c for c in run if c.isalpha()]
    if not letters:
        return True
    if run.isupper() and len(letters) >= 2:
        return True
    return bool(run[0].isupper() and run[1:].islower())


def _foreign_block_end(target: str, end: int, pattern: re.Pattern[str]) -> int:
    """Extend a script run across single spaces separating same-script words.

    Latin prose arrives as several letter runs split on spaces; the bracketed
    translation follows the whole passage, not each word, so the check must look
    at the end of the block.
    """
    while end < len(target) and target[end] == " ":
        nxt = pattern.match(target, end + 1)
        if nxt is None:
            break
        end = nxt.end()
    return end


def _has_parenthesized_translation(target: str, end: int) -> bool:
    """True when a bracketed translation immediately follows the run ending at end.

    Allows at most one space, accepts half- and full-width bracket pairs, and requires
    non-empty content. The rule this checks is the shared foreign-text guidance: keep the
    original wording and add the target-language translation in parentheses right after it.
    """
    i = end
    if i < len(target) and target[i] == " ":
        i += 1
    if i >= len(target):
        return False
    opener = target[i]
    closer = {"(": ")", "（": "）"}.get(opener)
    if closer is None:
        return False
    close_at = target.find(closer, i + 1)
    return close_at != -1 and close_at > i + 1


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

    Skip URLs, domain labels and proper nouns (Title Case / ALLCAPS) so copyright
    pages, publisher names and web addresses are not treated as untranslated prose.
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
            if len(run) < min_len:
                continue
            if run not in source:
                continue
            if name == "latin" and (
                _looks_like_url_context(target, match.start(), match.end())
                or _is_proper_noun_or_url_token(run)
            ):
                continue
            # A passage that already carries its required bracketed translation complies
            # with the foreign-text rule and is not untranslated residue.
            block_end = _foreign_block_end(target, match.end(), pattern)
            if _has_parenthesized_translation(target, block_end):
                continue
            leftover.append(run)
    if not leftover:
        return None
    sample = " / ".join(leftover[:3])
    return {
        "kind": "untranslated_residue",
        "leftover": leftover,
        "detail": f"Possible untranslated source text remains in translation: {sample}",
    }


def scan_foreign_unbracketed(source: str, target: str) -> dict[str, Any] | None:
    """Report source-present foreign runs that lack the required bracketed translation.

    The foreign-text rule keeps the original wording and appends its target-language
    translation in parentheses immediately after it. Runs that already follow the rule are
    compliant; isolated proper nouns and labels never require brackets. This is the inverse
    of scan_untranslated_residue, which reports runs that are missing entirely.
    """
    if not (source or "").strip() or not (target or "").strip():
        return None
    target_counts = _script_letter_counts(target)
    dominant = max(target_counts, key=target_counts.get)  # type: ignore[arg-type]
    if target_counts[dominant] < 5:
        return None
    missing: list[str] = []
    covered = 0
    for name, pattern in _SCRIPT_PATTERNS:
        if name == dominant or target_counts[name] <= 0:
            continue
        min_len = 2 if name == "cjk" else 3
        for match in pattern.finditer(target):
            run = match.group()
            if len(run) < min_len:
                continue
            if match.start() < covered:
                continue
            if run not in source:
                continue
            if name == "latin" and (
                _looks_like_url_context(target, match.start(), match.end())
                or _is_proper_noun_or_url_token(run)
            ):
                continue
            block_end = _foreign_block_end(target, match.end(), pattern)
            if _has_parenthesized_translation(target, block_end):
                continue
            # Report the whole word block, not the first word of it.
            missing.append(target[match.start() : block_end])
            covered = block_end
    if not missing:
        return None
    sample = " / ".join(missing[:3])
    return {
        "kind": "foreign_unbracketed",
        "unbracketed": missing,
        "detail": (
            "Source foreign passage lacks the required bracketed translation: {sample}"
        ).format(sample=sample),
    }


def scan_term_drift(
    source: str,
    target: str,
    terms: list[GlossaryTerm],
) -> dict[str, Any] | None:
    """Report glossary source hits whose fixed target mapping is absent from the translation."""
    if not (source or "").strip() or not (target or "").strip() or not terms:
        return None

    normalized_source = _match_text(source)
    if not normalized_source.strip():
        return None
    for term in terms:
        if not term.target or not term.target.strip():
            continue
        if term.target in target:
            continue
        keys = term_match_sources(term)
        hit = False
        for key in keys:
            norm_key = _match_text(key).strip()
            if not norm_key:
                continue
            if pattern := _source_pattern(norm_key):
                if pattern.search(normalized_source):
                    hit = True
                    break
            elif norm_key in normalized_source:
                hit = True
                break
        if not hit:
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
        scan_foreign_unbracketed(source, target),
        scan_term_drift(source, target, terms or []),
    ):
        if finding is not None:
            findings.append(finding)
    return findings

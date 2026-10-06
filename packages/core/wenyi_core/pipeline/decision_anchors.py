"""Optional target-side decision anchors distilled from finalized translations.

Anchors are a short must/avoid/example list used to stabilize naming, address
forms and register across later batches. They never replace style briefs,
synopses or glossaries.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ..storage.protocol import Storage

_ADDRESS_RE = re.compile(r"(先生|小姐|女士|大人|老师|阁下|桑|様|さま|君|ちゃん|さん)")
# Latin names are recognisable from the text itself. CJK ones are not: without an alignment
# between the source and target there is no way to tell which run of characters in the target
# renders which name, and matching any 2-3 CJK characters reported the opening words of a
# copyright page as name preferences. Curated CJK names live in the glossary, which already
# reaches the translation prompt.
_NAME_RE = re.compile(r"[A-Z][a-zA-Z\-']{2,}")


def _preview(text: str, limit: int = 40) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def distill_decision_anchors(
    pairs: list[tuple[str, str]],
    *,
    max_items: int = 8,
) -> dict[str, Any]:
    """Build a compact anchor dict from source/target pairs (pure)."""
    must: list[str] = []
    avoid: list[str] = []
    examples: list[dict[str, str]] = []

    address_counter: Counter[str] = Counter()
    for source, target in pairs:
        for match in _ADDRESS_RE.findall(target or ""):
            address_counter[match] += 1
    for form, count in address_counter.most_common(3):
        if count >= 2:
            must.append(f"称呼保持「{form}」类形式（出现 {count} 次）")

    # Prefer the first stable Latin name-like rendering from early pairs.
    seen_names: set[str] = set()
    for source, target in pairs[:20]:
        src_names = _NAME_RE.findall(source or "")
        tgt_names = _NAME_RE.findall(target or "")
        for name in src_names[:2]:
            if name in seen_names:
                continue
            mapped = None
            for candidate in tgt_names:
                if candidate != name:
                    mapped = candidate
                    break
            if mapped and mapped not in seen_names:
                seen_names.add(mapped)
                must.append(f"人名倾向「{mapped}」")
                examples.append({"source": _preview(source, 28), "target": _preview(target, 28)})
                break
        if len(examples) >= max_items:
            break

    # Avoid mixed full/half width punctuation spikes and trailing spaces heuristics.
    half_width = sum(1 for _s, t in pairs if re.search(r"[一-鿿][,.;:!?]", t or ""))
    if half_width >= max(2, len(pairs) // 4):
        avoid.append("避免中文句中夹半角标点（,.;:!?）")

    if not must and not avoid and not examples:
        return {"must": [], "avoid": [], "examples": [], "version": 1, "empty": True}
    return {
        "must": must[:max_items],
        "avoid": avoid[:max_items],
        "examples": examples[:max_items],
        "version": 1,
        "empty": False,
    }


def render_decision_anchors(anchors: dict[str, Any] | None) -> str:
    """Render anchors for prompt injection; empty when disabled or blank."""
    if not anchors or anchors.get("empty"):
        return ""
    must = anchors.get("must") or []
    avoid = anchors.get("avoid") or []
    examples = anchors.get("examples") or []
    if not must and not avoid and not examples:
        return ""
    lines: list[str] = ["Decision anchors (target-side conventions from earlier finalized text):"]
    for item in must:
        lines.append(f"- MUST: {item}")
    for item in avoid:
        lines.append(f"- AVOID: {item}")
    for item in examples[:3]:
        lines.append(f"- E.g. {item.get('source', '')} -> {item.get('target', '')}")
    return "\n".join(lines)


def load_decision_anchors(store: Storage) -> dict[str, Any]:
    analysis = store.load_analysis() or {}
    anchors = analysis.get("decision_anchors")
    return (
        anchors
        if isinstance(anchors, dict)
        else {"must": [], "avoid": [], "examples": [], "version": 1, "empty": True}
    )


def update_decision_anchors(
    store: Storage,
    *,
    mode: str = "off",
    min_pairs: int = 12,
    max_chapters: int = 3,
) -> dict[str, Any]:
    """Distill anchors from early finalized chapters and persist them in analysis."""
    if mode == "off":
        return load_decision_anchors(store)
    manifest = store.load_manifest()
    done = [
        row
        for row in manifest.get("chapters", [])
        if row.get("status") == "done" and isinstance(row.get("index"), int)
    ]
    if not done:
        return load_decision_anchors(store)
    pairs: list[tuple[str, str]] = []
    for row in done[:max_chapters]:
        chapter = store.load_chapter(row["index"])
        for segment in chapter.text_segments:
            target = (segment.target or "").strip()
            if not target:
                continue
            pairs.append((segment.source or "", target))
            if len(pairs) >= 120:
                break
        if len(pairs) >= 120:
            break
    if len(pairs) < min_pairs and mode != "risk":
        return load_decision_anchors(store)
    anchors = distill_decision_anchors(pairs)
    analysis = store.load_analysis() or {}
    analysis["decision_anchors"] = anchors
    store.save_analysis(analysis)
    store.log_event("decision_anchors_updated", mode=mode, empty=bool(anchors.get("empty")))
    return anchors

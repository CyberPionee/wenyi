"""Format glossary, annotation and segment payloads for agent prompts."""

from __future__ import annotations

import json

from ..glossary.store import GlossaryTerm


def render_glossary(
    terms: list[GlossaryTerm],
    *,
    include_note: bool = True,
    max_note_chars: int = 120,
    max_terms: int | None = None,
    source_lang: str = "",
) -> str:
    """Render references, exposing readings only for the resolved Japanese source."""
    if not terms:
        return "(none)"
    if max_terms is not None and max_terms >= 0:
        terms = terms[:max_terms]
        if not terms:
            return "(none)"
    lines = []
    for t in terms:
        extra = []
        if t.gender:
            extra.append(t.gender)
        if source_lang == "ja" and t.reading:
            extra.append(f"Pronunciation: {t.reading}")
        tag = f"({t.type}{(', ' + ', '.join(extra)) if extra else ''})"
        alias = f" [Aliases:  {', '.join(t.aliases)}]" if t.aliases else ""
        note = ""
        if include_note:
            raw_note = (t.note or "").strip()
            if raw_note:
                if max_note_chars > 0 and len(raw_note) > max_note_chars:
                    raw_note = raw_note[:max_note_chars]
                note = f" Note: {raw_note}"
        lines.append(f"- {t.source} → {t.target}{tag}{alias}{note}")
    return "\n".join(lines)


def render_annotation_contexts(contexts: list[list[dict[str, str]]]) -> str:
    """Deduplicate annotation data into stable JSON while retaining applicable batch indices."""
    rendered_by_key: dict[str, dict[str, object]] = {}
    for segment_index, items in enumerate(contexts):
        for item in items:
            target_key = item["target_key"]
            source = item["source"]
            rendered = rendered_by_key.get(target_key)
            if rendered is None:
                rendered_by_key[target_key] = {
                    "target_key": target_key,
                    "source": source,
                    "applies_to": [segment_index],
                }
                continue
            if rendered["source"] != source:
                raise ValueError(f"Inconsistent text for annotation target: {target_key}")
            applies_to = rendered["applies_to"]
            if isinstance(applies_to, list) and segment_index not in applies_to:
                applies_to.append(segment_index)
    return json.dumps(list(rendered_by_key.values()), ensure_ascii=False, indent=2)


def numbered(texts: list[str]) -> str:
    """Render text with zero-based indices in square brackets."""
    return "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))


def render_source_reference(source: str) -> str:
    """Quote one following source segment without adding numbered translation inputs."""
    return json.dumps(source, ensure_ascii=False) if source.strip() else "(none)"


def numbered_pairs(sources: list[str], targets: list[str]) -> str:
    """Render aligned source/target pairs for review prompts."""
    out = []
    for i, (s, t) in enumerate(zip(sources, targets)):
        out.append(f"[{i}] Source: {s}\n    Translation: {t}")
    return "\n".join(out)


def numbered_pairs_with_refs(
    sources: list[str],
    targets: list[str],
    refs: list[str],
) -> str:
    """Render source/target pairs with stable segment references for evidence review."""
    out = []
    for index, (source, target) in enumerate(zip(sources, targets)):
        ref = refs[index] if index < len(refs) else ""
        out.append(f"[{index}] ref={ref or '(none)'} Source: {source}\n    Translation: {target}")
    return "\n".join(out)


def strip_empty_sections(text: str) -> str:
    """Remove markdown ``##`` sections that carry no body content.

    A section is empty when everything between its heading and the next heading
    (or end of text) is whitespace.  Sections with content are kept unchanged.
    This trims digest/synopsis boilerplate before prompt injection.
    """
    raw = (text or "").strip()
    if not raw:
        return raw
    lines = raw.split("\n")
    sections: list[tuple[str, list[str]]] = []  # (heading, body_lines)
    current_heading = ""
    current_body: list[str] = []
    for line in lines:
        if line.startswith("## "):
            sections.append((current_heading, current_body))
            current_heading = line
            current_body = []
        else:
            current_body.append(line)
    sections.append((current_heading, current_body))
    kept: list[str] = []
    for heading, body in sections:
        if heading:
            if any(ln.strip() for ln in body):
                kept.append(heading)
                kept.extend(body)
        else:
            kept.extend(body)  # preamble before first heading
    return "\n".join(kept).strip()


def clip_context_block(value: str, max_chars: int) -> str:
    """Clip long style/synopsis/digest blocks for review prompts; empty becomes (none)."""
    text = strip_empty_sections(value or "")
    if not text:
        return "(none)"
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text

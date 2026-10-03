"""L0-L3 machine evaluation gate for autonomous acceptance.

L0 reuses assemble.report / residual sweep counts. L1 risk-gated back-translation
and L3 quality judge sample selected segments only. L2 measures whether established
glossary targets are actually used, as a rate with a configurable tolerance.
All selections and the gate are pure; model calls live behind injectable callables
for tests.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..glossary.store import GlossaryTerm, _match_text, _source_pattern, term_match_sources
from ..review.sweep import scan_segment
from ..storage.protocol import Storage

RiskFn = Callable[[str, str], list[dict[str, Any]]]

_DIALOG_RE = re.compile(r"[\"“”‘’「」『』]|—{2,}|\.{3,}|…")
_LONG_RE = re.compile(r"[^\n]{120,}")
_TERM_HINT_RE = re.compile(r"[A-Z][a-zA-Z\-']{2,}|[一-鿿]{2,4}(?:先生|小姐|老师|大人)")


@dataclass(frozen=True)
class RiskSegment:
    chapter: int
    index: int
    source_preview: str
    target_preview: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


@dataclass
class EvaluationResult:
    risk_segments: list[RiskSegment] = field(default_factory=list)
    back_translation: list[dict[str, Any]] = field(default_factory=list)
    judge_scores: list[dict[str, Any]] = field(default_factory=list)
    l0: dict[str, Any] = field(default_factory=dict)
    l2: dict[str, Any] = field(default_factory=dict)
    machine_gate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_segments": [item.to_dict() for item in self.risk_segments],
            "back_translation": list(self.back_translation),
            "judge_scores": list(self.judge_scores),
            "l0": dict(self.l0),
            "l2": dict(self.l2),
            "machine_gate": dict(self.machine_gate),
        }


def _preview(text: str, limit: int = 80) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def segment_risk_reasons(
    source: str, target: str, *, findings: Sequence[dict[str, Any]] = ()
) -> list[str]:
    """Return deterministic risk reasons for one source/target pair."""
    reasons: list[str] = []
    if not (target or "").strip():
        reasons.append("empty_target")
        return reasons
    if findings:
        reasons.append("sweep")
    if _DIALOG_RE.search(source or ""):
        reasons.append("dialog")
    if _LONG_RE.search(source or ""):
        reasons.append("long_segment")
    if _TERM_HINT_RE.search(source or ""):
        reasons.append("term_like")
    src_len = len((source or "").strip())
    tgt_len = len((target or "").strip())
    if src_len >= 20 and tgt_len > 0:
        ratio = tgt_len / src_len
        if ratio < 0.35 or ratio > 2.8:
            reasons.append("length_ratio")
    return reasons


def select_risk_segments(
    chapters: Sequence[Any],
    *,
    terms: Sequence[Any] = (),
    sample_ratio: float = 0.08,
    risk_scan: RiskFn | None = None,
) -> list[RiskSegment]:
    """Select high-risk segments plus a stable per-chapter sample of the rest."""
    scan = risk_scan or (lambda source, target: scan_segment(source, target, list(terms)))
    selected: list[RiskSegment] = []
    for chapter in chapters:
        chapter_index = int(getattr(chapter, "index", 0))
        segments = list(getattr(chapter, "text_segments", []) or [])
        if not segments:
            continue
        sampled_target = max(1, int(len(segments) * sample_ratio)) if sample_ratio > 0 else 0
        sample_slots: set[int] = set()
        if sampled_target:
            step = max(1, len(segments) // sampled_target)
            for position in range(0, len(segments), step):
                sample_slots.add(position)
                if len(sample_slots) >= sampled_target:
                    break
        for position, segment in enumerate(segments):
            source = segment.source or ""
            target = segment.target or ""
            findings = list(scan(source, target))
            reasons = segment_risk_reasons(source, target, findings=findings)
            if not reasons and position not in sample_slots:
                continue
            if not reasons:
                reasons = ["sample"]
            selected.append(
                RiskSegment(
                    chapter=chapter_index,
                    index=int(segment.index),
                    source_preview=_preview(source),
                    target_preview=_preview(target),
                    reasons=tuple(reasons),
                )
            )
    return selected


def _glossary_hits(source: str, terms: Sequence[GlossaryTerm]) -> list[GlossaryTerm]:
    """Terms whose source or alias occurs in the source segment."""
    normalized = _match_text(source or "")
    if not normalized.strip():
        return []
    hits: list[GlossaryTerm] = []
    for term in terms:
        for key in term_match_sources(term):
            norm_key = _match_text(key).strip()
            if not norm_key:
                continue
            if pattern := _source_pattern(norm_key):
                if pattern.search(normalized):
                    hits.append(term)
                    break
            elif norm_key in normalized:
                hits.append(term)
                break
    return hits


def scan_term_consistency(
    chapters: Sequence[Any],
    terms: Sequence[GlossaryTerm],
    *,
    max_items: int = 50,
) -> dict[str, Any]:
    """L2: measure how often established glossary targets reach the translation.

    A segment is *checked* when at least one glossary term with a non-empty target
    occurs in its source. It *drifts* when such a term's fixed target is absent from
    the translation. Empty targets belong to L0 and are skipped here.
    """
    checked = 0
    drifted = 0
    items: list[dict[str, Any]] = []
    if not terms:
        return {
            "checked": 0,
            "drifted": 0,
            "consistency_rate": 1.0,
            "items": [],
        }
    for chapter in chapters:
        chapter_index = int(getattr(chapter, "index", 0))
        for segment in getattr(chapter, "text_segments", []) or []:
            target = segment.target or ""
            if not target.strip():
                continue
            hits = [
                term for term in _glossary_hits(segment.source or "", terms) if term.target.strip()
            ]
            if not hits:
                continue
            checked += 1
            missing = [term for term in hits if term.target not in target]
            if not missing:
                continue
            drifted += 1
            if len(items) < max_items:
                first = missing[0]
                items.append(
                    {
                        "chapter": chapter_index,
                        "index": int(segment.index),
                        "source_term": first.source,
                        "expected_target": first.target,
                        "missing_targets": [term.target for term in missing],
                        "source_preview": _preview(segment.source or ""),
                        "target_preview": _preview(target),
                    }
                )
    rate = (checked - drifted) / checked if checked else 1.0
    return {
        "checked": checked,
        "drifted": drifted,
        "consistency_rate": round(rate, 4),
        "items": items,
    }


def back_translation_similarity(source: str, back: str) -> float:
    """Cheap character-trigram similarity in [0, 1] for offline L1 scoring."""
    src = re.findall(r"\w+", (source or "").lower())
    bak = re.findall(r"\w+", (back or "").lower())
    if not src or not bak:
        return 0.0

    def trigrams(tokens: Sequence[str]) -> set[str]:
        joined = " ".join(tokens)
        if len(joined) < 3:
            return {joined} if joined else set()
        return {joined[i : i + 3] for i in range(len(joined) - 2)}

    a, b = trigrams(src), trigrams(bak)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_back_translations(
    pairs: Sequence[tuple[int, int, str, str]],
    back_texts: Sequence[str],
) -> list[dict[str, Any]]:
    """Pair each located source with its back-translation score.

    ``pairs`` entries are ``(chapter, index, source, target)`` so low-scoring items can
    be routed back to the same paragraph for repair.
    """
    results: list[dict[str, Any]] = []
    for (chapter, index, source, _target), back in zip(pairs, back_texts):
        score = back_translation_similarity(source, back)
        results.append(
            {
                "chapter": chapter,
                "index": index,
                "source_preview": _preview(source),
                "back_preview": _preview(back),
                "score": round(score, 4),
            }
        )
    return results


# Autonomy tiers shape how strictly the machine gate is enforced and how densely
# evaluation samples the book. Explicit thresholds always win over tier defaults.
AUTONOMY_TIERS = ("off", "speed", "standard", "precise")


def apply_autonomy_tier(settings: dict[str, Any], tier: str) -> dict[str, Any]:
    """Return effective evaluation settings for an autonomy tier.

    - ``speed``: L0 must pass; L1-L3 stay informational and never block.
    - ``standard``: L0-L3 all green to pass, denser sampling than speed.
    - ``precise``: standard enforcement with tighter thresholds and sampling.
    """
    effective = {
        "tier": tier if tier in AUTONOMY_TIERS else "standard",
        "block_on_l0_only": False,
        "risk_sample_ratio": float(settings.get("risk_sample_ratio") or 0.08),
        "judge_sample_ratio": float(settings.get("judge_sample_ratio") or 0.05),
        "bt_score_min": float(settings.get("bt_score_min") or 0.45),
        "judge_score_min": float(settings.get("judge_score_min") or 3.5),
        "l2_min_consistency": float(settings.get("l2_min_consistency") or 1.0),
    }
    if effective["tier"] == "speed":
        effective["block_on_l0_only"] = True
    elif effective["tier"] == "precise":
        # Tighter acceptance plus denser sampling; never relax below the configured floor.
        effective["risk_sample_ratio"] = min(1.0, effective["risk_sample_ratio"] * 2)
        effective["judge_sample_ratio"] = min(1.0, effective["judge_sample_ratio"] * 2)
        effective["bt_score_min"] = max(effective["bt_score_min"], 0.6)
        effective["judge_score_min"] = max(effective["judge_score_min"], 4.0)
        effective["l2_min_consistency"] = 1.0
    return effective


def build_machine_gate(
    *,
    l0: dict[str, Any],
    back_translation: Sequence[dict[str, Any]] = (),
    judge_scores: Sequence[dict[str, Any]] = (),
    l2: dict[str, Any] | None = None,
    bt_score_min: float = 0.45,
    judge_score_min: float = 3.5,
    l2_min_consistency: float = 1.0,
    block_on_l0_only: bool = False,
) -> dict[str, Any]:
    """Aggregate L0-L3 into a single machine acceptance gate."""
    empty = int(l0.get("empty_target_count") or 0)
    conflicts = int(l0.get("open_conflict_count") or 0)
    residuals = int(l0.get("residual_finding_count") or 0)
    open_issues = int(l0.get("open_issue_count") or 0)

    l2_checked = int((l2 or {}).get("checked") or 0)
    l2_drifted = int((l2 or {}).get("drifted") or 0)
    l2_rate = float((l2 or {}).get("consistency_rate", 1.0))
    l2_strict = l2_min_consistency >= 1.0
    # Term drift also shows up in the residual sweep. When L2 owns the decision with a
    # tolerance below 1.0, do not let the same findings fail L0 twice.
    residuals_for_l0 = residuals if l2_strict else max(0, residuals - l2_drifted)
    l0_passed = empty == 0 and conflicts == 0 and residuals_for_l0 == 0 and open_issues == 0

    l2_passed = l2_checked == 0 or l2_rate >= l2_min_consistency

    bt_scores = [float(item.get("score") or 0.0) for item in back_translation]
    bt_low = sum(1 for score in bt_scores if score < bt_score_min)
    bt_passed = (not bt_scores) or (bt_low == 0)

    judge_values = [
        float(item.get("score") or 0.0) for item in judge_scores if item.get("score") is not None
    ]
    judge_avg = round(sum(judge_values) / len(judge_values), 4) if judge_values else None
    judge_low = sum(1 for score in judge_values if score < judge_score_min)
    judge_passed = (not judge_values) or (judge_avg is not None and judge_avg >= judge_score_min)

    passed = l0_passed and l2_passed and bt_passed and judge_passed
    # The speed tier keeps L1-L3 informational: only L0 can block export.
    blocking = not l0_passed if block_on_l0_only else not passed
    return {
        "passed": passed,
        "blocking": blocking,
        "block_on_l0_only": block_on_l0_only,
        "l0_passed": l0_passed,
        "l2_passed": l2_passed,
        "bt_passed": bt_passed,
        "judge_passed": judge_passed,
        "empty_target_count": empty,
        "open_conflict_count": conflicts,
        "residual_finding_count": residuals,
        "l0_residual_finding_count": residuals_for_l0,
        "open_issue_count": open_issues,
        "l2_checked_count": l2_checked,
        "l2_drift_count": l2_drifted,
        "l2_consistency_rate": l2_rate,
        "l2_min_consistency": l2_min_consistency,
        "bt_sample_count": len(bt_scores),
        "bt_low_count": bt_low,
        "bt_score_min": bt_score_min,
        "judge_sample_count": len(judge_values),
        "judge_avg": judge_avg,
        "judge_low_count": judge_low,
        "judge_score_min": judge_score_min,
    }


def quality_notes_to_autofix_issues(
    notes: Iterable[dict[str, Any]],
    *,
    origin: str = "quality",
) -> list[dict[str, Any]]:
    """Map C-batch / evaluation suggestions onto the Autofix issue contract."""
    issues: list[dict[str, Any]] = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        chapter = note.get("chapter")
        index = note.get("index")
        suggested = note.get("suggested")
        if not isinstance(chapter, int) or not isinstance(index, int):
            continue
        if not isinstance(suggested, str) or not suggested.strip():
            continue
        key = f"{origin}:{chapter}:{index}"
        issues.append(
            {
                "chapter": chapter,
                "index": index,
                "issue_key": key,
                "issue_id": key,
                "type": "fluency",
                "detail": str(note.get("detail") or f"{origin} suggested revision"),
                "suggestion": suggested,
            }
        )
    return issues


def quality_pass_notes_to_issues(
    quality: dict[str, Any], *, bt_min: float = 0.45
) -> list[dict[str, Any]]:
    """Convert C-batch notes into Autofix issues where a location is known.

    - self-revision / final-polish notes carry a replacement.
    - chapter self-check findings carry a detail only; the detail guides the fixer.
    - back-translation notes are candidates only when their offline score is low.
    - editorial notes are book-level and have no paragraph location, so they are skipped.
    """
    notes: list[dict[str, Any]] = []
    for key in ("self_revision_notes", "final_polish_notes"):
        values = quality.get(key)
        if isinstance(values, list):
            notes.extend(item for item in values if isinstance(item, dict))
    for item in quality.get("chapter_selfcheck_findings") or []:
        if not isinstance(item, dict):
            continue
        if not item.get("suggested"):
            item = {**item, "suggested": str(item.get("detail") or "").strip()}
        notes.append(item)
    for item in quality.get("back_translation_notes") or []:
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and float(score) >= bt_min
        ):
            continue
        if not item.get("suggested"):
            item = {
                **item,
                "suggested": "Revise this paragraph so it faithfully matches the source.",
            }
        notes.append(item)
    return quality_notes_to_autofix_issues(notes, origin="quality")


def evaluation_low_score_issues(
    evaluation: dict[str, Any], *, bt_min: float, judge_min: float
) -> list[dict[str, Any]]:
    """Map failing evaluation findings onto the Autofix issue contract.

    L2 drift and low-scoring L1/L3 paragraphs have no deterministic replacement here:
    L2 only knows the expected mapping, so the fixer revises the whole paragraph using
    that mapping as the suggestion.
    """
    issues: list[dict[str, Any]] = []

    def add(chapter: Any, index: Any, detail: str, suggestion: str, key: str) -> None:
        if not isinstance(chapter, int) or isinstance(chapter, bool):
            return
        if not isinstance(index, int) or isinstance(index, bool):
            return
        issues.append(
            {
                "chapter": chapter,
                "index": index,
                "issue_key": key,
                "issue_id": key,
                "type": "terminology" if suggestion and key.startswith("eval_l2") else "fluency",
                "detail": detail,
                "suggestion": suggestion,
            }
        )

    l2 = evaluation.get("l2") or {}
    for position, item in enumerate(l2.get("items") or []):
        missing = [value for value in (item.get("missing_targets") or []) if value]
        expected = item.get("expected_target") or ""
        suggestion = missing[0] if missing else expected
        add(
            item.get("chapter"),
            item.get("index"),
            f"Glossary mapping not used: {item.get('source_term') or ''} -> {suggestion}",
            suggestion,
            f"eval_l2:{item.get('chapter')}:{item.get('index')}:{position}",
        )

    for position, item in enumerate(evaluation.get("back_translation") or []):
        score = float(item.get("score") or 1.0)
        if score >= bt_min:
            continue
        add(
            item.get("chapter"),
            item.get("index"),
            f"Back-translation similarity {score:.2f} below {bt_min}",
            "",
            f"eval_bt:{item.get('chapter')}:{item.get('index')}:{position}",
        )

    for position, item in enumerate(evaluation.get("judge_scores") or []):
        score = item.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        if float(score) >= judge_min:
            continue
        note = str(item.get("note") or "").strip()
        detail = f"Quality judge score {score} below {judge_min}"
        if note:
            detail = f"{detail}: {note}"
        add(
            item.get("chapter"),
            item.get("index"),
            detail,
            "",
            f"eval_judge:{item.get('chapter')}:{item.get('index')}:{position}",
        )
    return issues


class EvaluationService:
    """Run L1/L3 sample evaluation and merge into the machine gate."""

    def __init__(
        self,
        store: Storage,
        *,
        risk_sample_ratio: float = 0.08,
        judge_sample_ratio: float = 0.05,
        bt_score_min: float = 0.45,
        judge_score_min: float = 3.5,
        l2_min_consistency: float = 1.0,
        block_on_l0_only: bool = False,
        risk_back_translation: bool = True,
        quality_judge: bool = True,
    ):
        self.store = store
        self.risk_sample_ratio = risk_sample_ratio
        self.judge_sample_ratio = judge_sample_ratio
        self.bt_score_min = bt_score_min
        self.judge_score_min = judge_score_min
        self.l2_min_consistency = l2_min_consistency
        self.block_on_l0_only = block_on_l0_only
        self.risk_back_translation = risk_back_translation
        self.quality_judge = quality_judge

    def collect_chapters(self) -> list[Any]:
        manifest = self.store.load_manifest()
        chapters = []
        for row in manifest.get("chapters", []):
            index = row.get("index")
            if not isinstance(index, int):
                continue
            if row.get("status") != "done":
                continue
            chapters.append(self.store.load_chapter(index))
        return chapters

    def run(
        self,
        *,
        l0: dict[str, Any],
        terms: Sequence[Any] = (),
        back_translate: Callable[[list[str]], list[str]] | None = None,
        judge: Callable[[list[tuple[str, str]]], list[dict[str, Any]]] | None = None,
    ) -> EvaluationResult:
        chapters = self.collect_chapters()
        risk = select_risk_segments(chapters, terms=terms, sample_ratio=self.risk_sample_ratio)
        result = EvaluationResult(risk_segments=risk, l0=dict(l0))
        # L2 is deterministic and cheap: always measure the real glossary coverage.
        result.l2 = scan_term_consistency(chapters, terms)
        # Keep chapter/segment identity so failing items can be routed back for repair.
        by_key = {
            (c.index, s.index): (c.index, s.index, s.source, s.target or "")
            for c in chapters
            for s in c.text_segments
        }
        pairs: list[tuple[int, int, str, str]] = []
        for item in risk:
            if "empty_target" in item.reasons:
                continue
            if not self.risk_back_translation and not self.quality_judge:
                break
            if "sample" in item.reasons and self.risk_sample_ratio <= 0:
                continue
            located = by_key.get((item.chapter, item.index))
            if located is not None:
                pairs.append(located)
        if pairs and self.risk_back_translation and back_translate is not None:
            targets = [target for _c, _i, _s, target in pairs]
            backs = back_translate(targets)
            result.back_translation = score_back_translations(pairs, backs)

        judge_pairs: list[tuple[int, int, str, str]] = []
        if self.quality_judge and judge is not None:
            step = max(1, int(1 / self.judge_sample_ratio)) if self.judge_sample_ratio > 0 else 1
            for position, pair in enumerate(pairs):
                if position % step == 0:
                    judge_pairs.append(pair)
            if judge_pairs:
                scored = judge([(source, target) for _c, _i, source, target in judge_pairs])
                located_scores: list[dict[str, Any]] = []
                for (chapter, index, _source, _target), item in zip(judge_pairs, scored):
                    record = dict(item) if isinstance(item, dict) else {}
                    record.setdefault("chapter", chapter)
                    record.setdefault("index", index)
                    located_scores.append(record)
                result.judge_scores = located_scores

        result.machine_gate = build_machine_gate(
            l0=l0,
            back_translation=result.back_translation,
            judge_scores=result.judge_scores,
            l2=result.l2,
            bt_score_min=self.bt_score_min,
            judge_score_min=self.judge_score_min,
            l2_min_consistency=self.l2_min_consistency,
            block_on_l0_only=self.block_on_l0_only,
        )
        return result

"""Summarize persisted translation issues that need human attention."""

from __future__ import annotations

from typing import Any

from ..glossary.store import GlossaryStore
from ..pipeline.runstore import STATUS_DONE
from ..review.sweep import scan_segment
from ..storage.protocol import Storage


def build_report(
    store: Storage,
    glossary: Storage | GlossaryStore,
    *,
    strict_auto_qa: bool = False,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize progress, empty translations, glossary conflicts and the latest review."""
    m = store.load_manifest()
    chapters_total = len(m["chapters"])
    chapters_done = sum(1 for c in m["chapters"] if c["status"] == STATUS_DONE)

    empty_targets: list[dict] = []
    blank_targets: list[dict] = []
    residual_findings: list[dict] = []
    terms = glossary.all_terms()

    for c in m["chapters"]:
        if c["status"] != STATUS_DONE:
            continue
        ch = store.load_chapter(c["index"])
        for s in ch.text_segments:
            if s.target is None:
                empty_targets.append(
                    {"chapter": c["index"], "index": s.index, "source": s.source[:60]}
                )
                continue
            if not s.target.strip():
                # MinerU keeps a blank target where it could not read text; that is a completed
                # segment rather than a pending one, so it is reported without failing the gate.
                blank_targets.append(
                    {"chapter": c["index"], "index": s.index, "source": s.source[:60]}
                )
                continue
            for finding in scan_segment(s.source, s.target, terms):
                residual_findings.append(
                    {
                        "chapter": c["index"],
                        "index": s.index,
                        "source": s.source[:60],
                        **finding,
                    }
                )

    conflicts = glossary.open_conflicts()
    gstats = glossary.stats()

    report: dict[str, Any] = {
        "summary": {
            "chapters_total": chapters_total,
            "chapters_done": chapters_done,
            "terms": gstats["terms"],
            "open_conflicts": len(conflicts),
            "empty_targets": len(empty_targets),
            "blank_targets": len(blank_targets),
        },
        "open_conflicts": conflicts,
        "empty_targets": empty_targets,
        "residual_findings": residual_findings,
    }
    review = store.load_latest_review_result()
    review_open_issues = 0
    if review is not None:
        review_summary = review.get("summary") or {}
        autofix = review.get("autofix") or {}
        applied_segments = int(autofix.get("applied_segment_count") or 0)
        review_open_issues = int(review_summary.get("issue_count") or 0)
        report["review"] = {
            "review_id": review.get("review_id"),
            "status": review.get("status"),
            "termination": review.get("termination"),
            "issue_count": review_open_issues,
            "change_count": int(review_summary.get("change_count") or 0),
            "read_only": applied_segments == 0,
            "autofix_status": autofix.get("status"),
            "autofix_applied_segment_count": applied_segments,
            "autofix_failed_issue_count": int(autofix.get("failed_issue_count") or 0),
        }
    # Aggregate residual/fixable state without redefining empty_targets/open_conflicts.
    auto_qa_passed = (
        not empty_targets and not conflicts and not residual_findings and review_open_issues == 0
    )
    report["auto_qa"] = {
        "passed": auto_qa_passed,
        "empty_target_count": len(empty_targets),
        "open_conflict_count": len(conflicts),
        "residual_finding_count": len(residual_findings),
        "open_issue_count": review_open_issues,
        "blocking": bool(strict_auto_qa) and not auto_qa_passed,
    }
    if evaluation is not None:
        report["evaluation"] = evaluation
        gate = evaluation.get("machine_gate") or {}
        report["machine_gate"] = gate
        if strict_auto_qa and gate.get("blocking"):
            report["auto_qa"]["blocking"] = True
    return report

"""Current progress/review/accounting reports for books and subtitles."""

from __future__ import annotations

from fastapi import APIRouter
from wenyi_core.assemble.report import build_report
from wenyi_core.srt.store import SrtRunStore

from ..project_service import project_write, require_project, storage_for
from ..schemas import ReportOut

router = APIRouter(prefix="/projects/{pid}/report", tags=["report"])


def _report(project: dict, storage) -> dict:
    if project.get("fmt") == "srt":
        with storage.state_lock():
            cues = SrtRunStore(storage.run_dir, storage=storage).load_cues()
            report = {
                "summary": {
                    "cues_total": len(cues),
                    "cues_done": sum(row.get("status") == "done" for row in cues.values()),
                    "empty_targets": sum(
                        not (row.get("target") or "").strip() for row in cues.values()
                    ),
                }
            }
    else:
        # Read-only scan; do not hold the state advisory lock for the full book walk.
        report = build_report(storage, storage)
    with storage.state_lock():
        if project.get("fmt") != "srt":
            saved = storage.load_report() or {}
            if saved.get("evaluation") is not None:
                report["evaluation"] = saved.get("evaluation")
            if saved.get("machine_gate") is not None:
                report["machine_gate"] = saved.get("machine_gate")
        report["usage"] = storage.load_usage() or {}
        report["timing"] = storage.read_artifact("timing.json") or {}
        return report


@router.get("", response_model=ReportOut)
def get_report(pid: str) -> dict:
    return _report(require_project(pid), storage_for(pid))


@router.post("", response_model=ReportOut)
def regenerate_report(pid: str) -> dict:
    with project_write(pid) as (project, storage):
        report = _report(project, storage)
        storage.save_report(report)
        storage.log_event("report_saved", artifact="report.json")
    return report

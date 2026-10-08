"""Glossary editing, source matching and explicit conflict resolution."""

from __future__ import annotations

import csv
import io
from dataclasses import replace
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from wenyi_core.glossary.resolver import keep_current_terms
from wenyi_core.glossary.store import MANUAL_STATUS, GlossaryTerm
from wenyi_core.glossary.writeback import apply_conflict_writeback, apply_term_writeback
from wenyi_core.i18n.metadata import normalize_term_type

from ..context import current_context
from ..project_service import project_write, require_book, require_project, storage_for
from ..schemas import (
    ConflictOut,
    GlossaryImport,
    KeptCurrentConflicts,
    Message,
    ResolveConflict,
    TermIn,
    TermOut,
)

router = APIRouter(prefix="/projects/{pid}/glossary", tags=["glossary"])


def _term(body: TermIn) -> GlossaryTerm:
    """Build a term from an operator's input."""
    if not body.source.strip() or not body.target.strip():
        raise HTTPException(422, "Term source and target must not be empty")
    return GlossaryTerm(**body.model_dump())


def _locked_term(body: TermIn) -> GlossaryTerm:
    """Build a term from an explicit single-term edit and lock it to the operator's choice.

    A target an operator types outranks automatic proposals for that source, which are dropped
    instead of recorded as a conflict. A bulk import does not lock: an imported disagreement
    still has to surface for a decision rather than silently overwrite a translation.
    """
    return replace(_term(body), status=MANUAL_STATUS)


@router.get("/terms", response_model=list[TermOut])
def list_terms(pid: str, q: str | None = Query(None), type: str | None = Query(None)) -> list[dict]:
    require_book(require_project(pid))
    terms = storage_for(pid).all_terms()
    if type:
        term_type = normalize_term_type(type)
        terms = [term for term in terms if term.type == term_type]
    if q:
        needle = q.strip().casefold()
        terms = [
            term
            for term in terms
            if any(
                needle in value.casefold()
                for value in [term.source, term.target, term.reading, term.note, *term.aliases]
            )
        ]
    return [vars(term) for term in terms]


@router.post("/terms", response_model=TermOut, status_code=201)
def add_term(pid: str, body: TermIn) -> dict:
    term = _locked_term(body)
    with project_write(pid) as (project, storage):
        require_book(project)
        if storage.get_term(term.source) is not None:
            raise HTTPException(409, "Term already exists; edit it or resolve its conflicts")
        storage.upsert_term(term)
        storage.log_event("glossary_term_added", source=term.source)
        return vars(storage.get_term(term.source))


@router.put("/terms/{source}", response_model=TermOut)
def update_term(pid: str, source: str, body: TermIn) -> dict:
    term = _locked_term(body)
    with project_write(pid) as (project, storage):
        require_book(project)
        existing = storage.get_term(source)
        if existing is None:
            raise HTTPException(404, "term not found")
        if term.source != source and storage.get_term(term.source) is not None:
            raise HTTPException(409, "Another term already uses that source")
        current_context().update_term(pid, storage, source, term)
        storage.log_event("glossary_term_edited", source=term.source, previous_source=source)
        writeback = None
        if existing.target and existing.target != term.target:
            writeback = apply_term_writeback(
                storage,
                term_source=source,
                term=term,
                old_target=existing.target,
                new_target=term.target,
            )
            storage.log_event(
                "glossary_target_written_back",
                source=source,
                old_target=existing.target,
                new_target=term.target,
                segments_replaced=writeback["segments_replaced"],
                chapters_touched=writeback["chapters_touched"],
            )
        result = vars(storage.get_term(term.source))
        result["writeback"] = writeback
        return result


@router.delete("/terms/{source}", response_model=Message)
def delete_term(pid: str, source: str) -> dict:
    with project_write(pid) as (project, storage):
        require_book(project)
        if not storage.delete_term(source):
            raise HTTPException(404, "term not found")
        storage.mark_conflicts_resolved(source)
        storage.log_event("glossary_term_deleted", source=source)
    return {"message": "deleted"}


@router.get("/conflicts", response_model=list[ConflictOut])
def list_conflicts(pid: str) -> list[dict]:
    require_book(require_project(pid))
    return storage_for(pid).open_conflicts()


@router.post("/conflicts/{cid}/resolve", response_model=Message)
def resolve_conflict(pid: str, cid: int, body: ResolveConflict) -> dict:
    with project_write(pid) as (project, storage):
        require_book(project)
        conflict = next((row for row in storage.open_conflicts() if row["id"] == cid), None)
        if conflict is None:
            raise HTTPException(404, "conflict not found")
        existing = storage.get_term(conflict["source"])
        if existing is None:
            raise HTTPException(404, "term not found")
        if body.decision == "current":
            target = existing.target
        elif body.decision == "proposed":
            target = conflict["proposed_target"]
        else:
            target = body.target
        if not isinstance(target, str) or not target.strip():
            raise HTTPException(422, "A nonempty target is required to resolve the conflict")
        with storage.state_lock():
            storage.resolve_term(existing.source, target)
            storage.mark_conflicts_resolved(existing.source)
            storage.log_event("glossary_conflict_resolved", source=existing.source, target=target)
        # Every rejected candidate has to leave the translation, including the proposal when the
        # established target wins: the passages translated with it are still wrong.
        rejected = [
            value
            for value in dict.fromkeys((existing.target, conflict["proposed_target"]))
            if isinstance(value, str) and value.strip() and value != target
        ]
        writeback = None
        if rejected:
            writeback = apply_conflict_writeback(
                storage,
                term_source=existing.source,
                term=GlossaryTerm(
                    source=existing.source,
                    target=target,
                    type=existing.type,
                    aliases=list(existing.aliases or []),
                ),
                rejected_targets=rejected,
                chosen_target=target,
            )
            storage.log_event(
                "glossary_target_written_back",
                source=existing.source,
                old_targets=rejected,
                new_target=target,
                segments_replaced=writeback["segments_replaced"],
                chapters_touched=writeback["chapters_touched"],
            )
    return {"message": "resolved", "detail": writeback}


@router.post("/conflicts/keep-current", response_model=KeptCurrentConflicts)
def keep_current_conflicts(pid: str) -> dict:
    """Close every open conflict in favour of each term's established target.

    A locked term's rows are already excluded from the open list, so this only settles
    disagreements nobody has decided yet. Each settlement is an operator's decision, so it
    locks the term and the same proposal cannot reopen a conflict later. Passages translated
    with a rejected proposal are rewritten to the target that now stands.
    """
    with project_write(pid) as (project, storage):
        require_book(project)
        with storage.state_lock():
            settled = keep_current_terms(storage)
            if settled:
                storage.log_event(
                    "glossary_conflicts_kept_current",
                    sources=[record["source"] for record in settled],
                )
        # Rewriting translations touches chapters, so it runs once the glossary lock is free.
        segments_replaced = 0
        for record in settled:
            writeback = apply_conflict_writeback(
                storage,
                term_source=record["source"],
                term=record["term"],
                rejected_targets=record["rejected"],
                chosen_target=record["target"],
            )
            segments_replaced += int(writeback["segments_replaced"])
            if writeback["segments_replaced"]:
                storage.log_event(
                    "glossary_target_written_back",
                    source=record["source"],
                    old_targets=record["rejected"],
                    new_target=record["target"],
                    segments_replaced=writeback["segments_replaced"],
                    chapters_touched=writeback["chapters_touched"],
                )
    return {
        "message": "resolved",
        "sources": [record["source"] for record in settled],
        "segments_replaced": segments_replaced,
    }


@router.get("/export")
def export_glossary(pid: str, format: Literal["json", "csv"] = "json"):
    require_book(require_project(pid))
    terms = storage_for(pid).all_terms()
    if format == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            ["source", "target", "reading", "type", "gender", "aliases", "note", "status"]
        )
        for term in terms:
            writer.writerow(
                [
                    term.source,
                    term.target,
                    term.reading,
                    term.type,
                    term.gender,
                    "|".join(term.aliases),
                    term.note,
                    term.status,
                ]
            )
        return Response(
            buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=glossary-{pid}.csv"},
        )
    return JSONResponse(
        [vars(term) for term in terms],
        headers={"Content-Disposition": f"attachment; filename=glossary-{pid}.json"},
    )


@router.post("/import")
def import_glossary(pid: str, body: GlossaryImport) -> dict:
    terms = [_term(item) for item in body.terms]
    with project_write(pid) as (project, storage):
        require_book(project)
        conflicts = 0
        with storage.state_lock():
            for term in terms:
                conflicts += storage.upsert_term(term) == "conflict"
            storage.log_event("glossary_imported", count=len(terms), conflicts=conflicts)
    return {"imported": len(terms), "conflicts": conflicts}

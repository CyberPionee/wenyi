"""SQLite glossary with two tables.
glossary stores one mapping per source. When another target is proposed, retain the current
translation and log the alternative in term_conflicts for human review. term_conflicts
records unresolved translation conflicts.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..i18n.metadata import normalize_gender, normalize_term_type

# Canonical glossary types.
TYPE_PERSON = "person"
TYPE_TERM = "term"
# An operator set this target by hand. It outranks automatic proposals, which stop
# offering alternatives for the source instead of recording conflicts against it.
MANUAL_STATUS = "manual"
TYPE_APPELLATION = "appellation"
TYPE_HONORIFIC = "honorific"
TYPE_SPEECH = "speech"
TYPE_FIXED_EXPR = "fixed_expression"

_SOURCE_ONLY_TYPES = {TYPE_APPELLATION, TYPE_HONORIFIC, TYPE_SPEECH, TYPE_FIXED_EXPR}


@dataclass
class GlossaryTerm:
    source: str
    target: str
    reading: str = ""
    type: str = TYPE_TERM
    gender: str = ""
    aliases: list[str] = field(default_factory=list)
    first_chapter: int | None = None
    note: str = ""
    status: str = "ok"

    def __post_init__(self) -> None:
        self.type = normalize_term_type(self.type)
        self.gender = normalize_gender(self.gender)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> GlossaryTerm:
        """Convert a SQLite row into a term and decode its JSON aliases."""
        return cls(
            source=row["source"],
            target=row["target"],
            reading=row["reading"] or "",
            type=row["type"] or TYPE_TERM,
            gender=row["gender"] or "",
            aliases=json.loads(row["aliases"] or "[]"),
            first_chapter=row["first_chapter"],
            note=row["note"] or "",
            status=row["status"] or "ok",
        )


_CREATE_GLOSSARY_TABLE = """
CREATE TABLE IF NOT EXISTS glossary (
    source        TEXT PRIMARY KEY,
    target        TEXT NOT NULL,
    reading       TEXT,
    type          TEXT,
    gender        TEXT,
    aliases       TEXT,
    first_chapter INTEGER,
    note          TEXT,
    status        TEXT DEFAULT 'ok',
    updated_at    REAL
)
"""

_SCHEMA = (
    _CREATE_GLOSSARY_TABLE
    + ";"
    + """
CREATE TABLE IF NOT EXISTS term_conflicts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,
    existing_target TEXT,
    proposed_target TEXT,
    chapter         INTEGER,
    note            TEXT,
    resolved        INTEGER DEFAULT 0,
    created_at      REAL
);
"""
)


# Match the reader's furigana markers in Segment.source; strip them before substring matching.
_RUBY_MARK_RE = re.compile(r"〘[^〙]*〙")


def _match_text(text: str) -> str:
    """Normalize width, compatibility forms and case for glossary matching.
    Strip embedded furigana so a reading inserted inside a word does not prevent that word
    from matching. The original ruby markup remains in the source template.
    """
    if "〘" in text:
        text = _RUBY_MARK_RE.sub("", text)
    return unicodedata.normalize("NFKC", text).casefold()


_WORD_BOUNDARY_SCRIPTS = ("LATIN", "GREEK", "CYRILLIC")


@functools.lru_cache(maxsize=4096)
def _source_pattern(key: str) -> re.Pattern[str] | None:
    """Build word boundaries for space-delimited scripts; return None for substring-matched
    scripts.
    """
    if key.isascii():
        return re.compile(rf"(?<![a-z0-9_]){re.escape(key)}(?![a-z0-9_])")

    letters = [char for char in key if char.isalpha()]
    if not letters or not all(
        any(script in unicodedata.name(char, "") for script in _WORD_BOUNDARY_SCRIPTS)
        for char in letters
    ):
        return None

    left_boundary = r"(?<!\w)" if key[0].isalnum() else ""
    right_boundary = r"(?!\w)" if key[-1].isalnum() else ""
    return re.compile(f"{left_boundary}{re.escape(key)}{right_boundary}")


def source_matches_text(source: str, text: str) -> bool:
    """Match source terms without matching inside longer words in space-delimited scripts.
    Use normalized substring matching for continuous scripts such as CJK. Check boundaries
    for ASCII, Latin, Greek and Cyrillic to avoid matching Ann inside Anna or a short
    Cyrillic word inside a longer one.
    """
    key = _match_text(source).strip()
    if not key:
        return False
    normalized_text = _match_text(text)
    if pattern := _source_pattern(key):
        return pattern.search(normalized_text) is not None
    return key in normalized_text


def term_match_sources(term: GlossaryTerm) -> list[str]:
    """Return source spellings allowed when matching a term.
    Match appellations, honorifics, speech habits and fixed expressions only by their
    complete source. A bare-name alias must not inject a context-specific derived
    translation into an ordinary address. Other entities can also match aliases.
    """
    if term.type in _SOURCE_ONLY_TYPES:
        return [term.source]
    return [term.source, *term.aliases]


def _source_occurrence_spans(source: str, normalized_text: str) -> list[tuple[int, int]]:
    """Return nonoverlapping source-term spans in normalized text."""
    key = _match_text(source).strip()
    if not key:
        return []
    if pattern := _source_pattern(key):
        return [match.span() for match in pattern.finditer(normalized_text)]

    spans: list[tuple[int, int]] = []
    start = 0
    while (index := normalized_text.find(key, start)) != -1:
        end = index + len(key)
        spans.append((index, end))
        start = end
    return spans


def _merged_occurrence_count(spans: set[tuple[int, int]]) -> int:
    """Merge overlapping source/alias matches at one location into a single mention."""
    count = 0
    active_end = -1
    for start, end in sorted(spans):
        if start >= active_end:
            count += 1
            active_end = end
        else:
            active_end = max(active_end, end)
    return count


class GlossaryOccurrenceMatcher:
    """Reuse one normalized corpus to detect repeated terms by source and aliases."""

    def __init__(self, text: str):
        self.normalized_text = _match_text(text)

    def recurring_terms(
        self,
        terms: list[GlossaryTerm],
        *,
        min_occurrences: int = 2,
    ) -> list[GlossaryTerm]:
        """Select terms whose source/aliases occur at least the requested number of times."""
        if min_occurrences <= 1:
            return GlossaryStore.terms_in(terms, self.normalized_text)

        recurring: list[GlossaryTerm] = []
        for term in terms:
            raw_keys = term_match_sources(term)
            keys = {normalized for key in raw_keys if (normalized := _match_text(key).strip())}
            spans: set[tuple[int, int]] = set()
            for key in keys:
                spans.update(_source_occurrence_spans(key, self.normalized_text))
            if _merged_occurrence_count(spans) >= min_occurrences:
                recurring.append(term)
        return recurring


# Branch an incoming term takes against an established mapping. Every storage adapter routes
# through classify_upsert so an operator's decision cannot be honoured by one backend and
# silently overwritten by another.
UPSERT_INSERT = "insert"
UPSERT_MANUAL_WRITE = "manual_write"
UPSERT_DROP = "drop"
UPSERT_FILL = "fill"
UPSERT_MERGE = "merge"
UPSERT_CONFLICT = "conflict"

# Public upsert_term results. Callers switch on these, so keep them stable.
_UPSERT_RESULTS = {
    UPSERT_INSERT: "inserted",
    UPSERT_MANUAL_WRITE: "updated",
    UPSERT_DROP: "unchanged",
    UPSERT_FILL: "updated",
    UPSERT_MERGE: "unchanged",
    UPSERT_CONFLICT: "conflict",
}


def classify_upsert(existing: GlossaryTerm | None, term: GlossaryTerm) -> str:
    """Return the branch an incoming term takes against an established mapping.

    A target an operator set by hand outranks every proposal: the alternative is discarded
    rather than recorded, so extraction cannot keep re-raising a decision already made. An
    incoming term that is itself marked manual writes through and becomes the authority.
    """
    if existing is None:
        return UPSERT_INSERT
    if term.status == MANUAL_STATUS:
        return UPSERT_MANUAL_WRITE
    if existing.status == MANUAL_STATUS and existing.target != term.target:
        return UPSERT_DROP
    if not (existing.target or "").strip() and (term.target or "").strip():
        return UPSERT_FILL
    if existing.target == term.target:
        return UPSERT_MERGE
    return UPSERT_CONFLICT


def upsert_result(branch: str) -> str:
    """Map a classify_upsert branch onto the public upsert_term result."""
    return _UPSERT_RESULTS[branch]


def conflict_already_recorded(
    recorded: Iterable[Sequence[str | None]],
    existing_target: str,
    proposed_target: str,
) -> bool:
    """Return whether an unresolved row already records this exact rejected proposal.

    ``recorded`` holds the (existing_target, proposed_target) pairs still open for the
    source. A source a later batch proposes again would otherwise add one row per batch: one
    recurring disagreement produced fifteen identical rows, and the operator had to settle
    the same decision repeatedly. Every storage adapter shares this test so the two backends
    cannot disagree about what the operator has already seen.
    """
    for row in recorded:
        if (row[0] or "") == existing_target and (row[1] or "") == proposed_target:
            return True
    return False


class GlossaryStore:
    def __init__(self, db_path: str):
        """Open the glossary database and initialize the current schema."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        # Wait for concurrent writes so editors and translation workers avoid database-is-locked errors.
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self._owns_connection = True

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> GlossaryStore:
        """Borrow an initialized connection; the caller owns transactions and its lifetime."""
        store = cls.__new__(cls)
        store.conn = conn
        store._owns_connection = False
        return store

    @staticmethod
    def initialize_schema(conn: sqlite3.Connection) -> None:
        """Create glossary tables without committing the caller's transaction."""
        for statement in _SCHEMA.split(";"):
            if statement.strip():
                conn.execute(statement)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._owns_connection:
            self.conn.close()

    @classmethod
    def load_terms_readonly(cls, db_path: str) -> list[GlossaryTerm]:
        """Read terms from a temporary snapshot without touching the formal database or its WAL
        files.
        immutable=1 avoids creating shm but ignores committed, uncheckpointed WAL data. Copy
        the database and existing WAL to a temporary directory and let SQLite recover the
        complete view there. Any resulting shm files or checkpoints remain outside formal
        book state.
        """
        with tempfile.TemporaryDirectory(
            prefix="wenyi-glossary-review-", ignore_cleanup_errors=True
        ) as directory:
            snapshot_path = f"{directory}/glossary.db"
            wal_path = f"{db_path}-wal"
            snapshot_wal_path = f"{snapshot_path}-wal"

            def signature(path: str) -> tuple[int, int, int, int] | None:
                """Return a lightweight signature sufficient to detect changes during copying."""
                try:
                    stat = os.stat(path)
                except FileNotFoundError:
                    return None
                return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

            # DB and WAL are separate files; sequential copying is not an atomic snapshot. Accept only
            # matching before/after signatures; discard and retry if a checkpoint or write intervenes.
            for _attempt in range(5):
                before = signature(db_path), signature(wal_path)
                try:
                    shutil.copy2(db_path, snapshot_path)
                    if before[1] is not None:
                        shutil.copy2(wal_path, snapshot_wal_path)
                    elif os.path.exists(snapshot_wal_path):
                        os.unlink(snapshot_wal_path)
                except FileNotFoundError:
                    continue
                after = signature(db_path), signature(wal_path)
                if before == after:
                    break
            else:
                raise RuntimeError(
                    "The glossary kept changing during snapshot capture; retry review later."
                )

            conn = sqlite3.connect(snapshot_path)
            conn.row_factory = sqlite3.Row
            try:
                # Return insertion order (rowid), not type/source sorting: inserting a new term in the middle
                # would shift the remaining prompt glossary and invalidate prefix caches.
                rows = conn.execute("SELECT * FROM glossary ORDER BY rowid").fetchall()
                return [GlossaryTerm.from_row(row) for row in rows]
            finally:
                conn.close()

    # Glossary terms.
    def get_term(self, source: str) -> GlossaryTerm | None:
        """Look up the exact source term; return None if absent."""
        row = self.conn.execute("SELECT * FROM glossary WHERE source = ?", (source,)).fetchone()
        return GlossaryTerm.from_row(row) if row else None

    def upsert_term(self, term: GlossaryTerm, chapter: int | None = None) -> str:
        """Apply one incoming term; return inserted, updated, unchanged or conflict.

        The branch and its rationale live in classify_upsert, which every storage backend
        shares so a locked operator decision cannot be honoured in one and lost in another.
        """
        try:
            # Acquire the lock before reading existing so two connections cannot decide from the same old view.
            if self._owns_connection:
                self.conn.execute("BEGIN IMMEDIATE")
            existing = self.get_term(term.source)
            now = time.time()
            branch = classify_upsert(existing, term)
            if branch == UPSERT_INSERT:
                self.conn.execute(
                    """INSERT INTO glossary
                       (source,target,reading,type,gender,aliases,first_chapter,note,
                        status,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        term.source,
                        term.target,
                        term.reading,
                        term.type,
                        term.gender,
                        json.dumps(term.aliases, ensure_ascii=False),
                        term.first_chapter if term.first_chapter is not None else chapter,
                        term.note,
                        term.status,
                        now,
                    ),
                )
            elif branch == UPSERT_MANUAL_WRITE:
                # An operator's edit writes through and clears any recorded conflict.
                self.conn.execute(
                    """UPDATE glossary SET target=?, reading=COALESCE(NULLIF(?,''),reading),
                       type=?, gender=COALESCE(NULLIF(?,''),gender), aliases=?,
                       note=COALESCE(NULLIF(?,''),note), status=?, updated_at=? WHERE source=?""",
                    (
                        term.target,
                        term.reading,
                        term.type,
                        term.gender,
                        json.dumps(
                            sorted(set(existing.aliases) | set(term.aliases)), ensure_ascii=False
                        ),
                        term.note,
                        MANUAL_STATUS,
                        now,
                        term.source,
                    ),
                )
                self.conn.execute(
                    "UPDATE term_conflicts SET resolved=1 WHERE source=?", (term.source,)
                )
            elif branch == UPSERT_FILL:
                # Fill a missing mapping without treating it as a conflict.
                merged_aliases = sorted(set(existing.aliases) | set(term.aliases))
                self.conn.execute(
                    """UPDATE glossary SET target=?, reading=COALESCE(NULLIF(?,''),reading),
                       gender=COALESCE(NULLIF(?,''),gender), aliases=?,
                       note=COALESCE(NULLIF(?,''),note), status='ok', updated_at=? WHERE source=?""",
                    (
                        term.target,
                        term.reading,
                        term.gender,
                        json.dumps(merged_aliases, ensure_ascii=False),
                        term.note,
                        now,
                        term.source,
                    ),
                )
            elif branch == UPSERT_MERGE:
                # Merging aliases or filling missing fields is not a conflict.
                merged_aliases = sorted(set(existing.aliases) | set(term.aliases))
                self.conn.execute(
                    """UPDATE glossary SET reading=COALESCE(NULLIF(?,''),reading),
                       gender=COALESCE(NULLIF(?,''),gender), aliases=?,
                       note=COALESCE(NULLIF(?,''),note), updated_at=? WHERE source=?""",
                    (
                        term.reading,
                        term.gender,
                        json.dumps(merged_aliases, ensure_ascii=False),
                        term.note,
                        now,
                        term.source,
                    ),
                )
            elif branch == UPSERT_CONFLICT:
                # Different target: keep the established mapping and record the candidate for human
                # resolution, once per disagreement rather than once per re-proposal.
                if not conflict_already_recorded(
                    self._open_conflict_rows(term.source),
                    existing.target or "",
                    term.target or "",
                ):
                    self._log_conflict(term.source, existing.target, term.target, chapter)
                self.conn.execute(
                    "UPDATE glossary SET status='conflict', updated_at=? WHERE source=?",
                    (now, term.source),
                )
            # UPSERT_DROP writes nothing: an operator's locked target already stands, so the
            # proposal that reached classify_upsert is discarded instead of recorded.
            if self._owns_connection:
                self.conn.commit()
            return upsert_result(branch)
        except Exception:
            if self._owns_connection:
                self.conn.rollback()
            raise

    def _open_conflict_rows(self, source: str) -> list[tuple]:
        """Return the still-open (existing, proposed) pairs for one source."""
        return self.conn.execute(
            """SELECT existing_target, proposed_target FROM term_conflicts
               WHERE source = ? AND resolved = 0""",
            (source,),
        ).fetchall()

    def _log_conflict(self, source, existing_target, proposed_target, chapter):
        """Record one candidate-translation conflict within the current transaction."""
        self.conn.execute(
            """INSERT INTO term_conflicts
               (source,existing_target,proposed_target,chapter,created_at)
               VALUES (?,?,?,?,?)""",
            (source, existing_target, proposed_target, chapter, time.time()),
        )

    def resolve_term(self, source: str, target: str) -> bool:
        """Apply a human decision on the final translation; report whether the term exists.

        Resolving is an operator's act, so the chosen target becomes authoritative and is
        locked against later proposals exactly like an explicit edit. Restoring ``ok`` here
        would let the next extraction pass re-open the conflict the operator just closed.
        """
        cur = self.conn.execute(
            "UPDATE glossary SET target=?, status=?, updated_at=? WHERE source=?",
            (target, MANUAL_STATUS, time.time(), source),
        )
        if self._owns_connection:
            self.conn.commit()
        return cur.rowcount > 0

    def all_terms(self) -> list[GlossaryTerm]:
        """Return all terms in insertion order (rowid).
        Translation, review and polishing prompts share this source. Existing positions must
        remain stable while new entries append at the end, preserving provider prefix-cache
        hits. Do not sort by type/source here: inserted entries would shift later text and
        invalidate the prefix. Alphabetical grouping belongs in the display layer, such as
        CLI sorted().
        """
        rows = self.conn.execute("SELECT * FROM glossary ORDER BY rowid").fetchall()
        return [GlossaryTerm.from_row(r) for r in rows]

    @staticmethod
    def terms_in(terms: list[GlossaryTerm], text: str) -> list[GlossaryTerm]:
        """Filter a prefetched term list by source/alias occurrences in text.
        Match evidence and local revision context without querying the database per term.
        """
        out: list[GlossaryTerm] = []
        normalized_text = _match_text(text)
        for term in terms:
            # Forms of address, verbal habits and fixed expressions carry tone or context. A bare-name alias
            # must not inject their derived translation into an ordinary address.
            keys = term_match_sources(term)
            if any(source_matches_text(k, normalized_text) for k in keys):
                out.append(term)
        return out

    @staticmethod
    def recurring_terms(
        terms: list[GlossaryTerm],
        text: str,
        *,
        min_occurrences: int = 2,
    ) -> list[GlossaryTerm]:
        """Select terms meeting the requested source/alias occurrence count.
        Count appellations, honorifics, speech habits and fixed expressions by source only
        so bare-name aliases cannot falsely make them frequent. For other types, merge
        source and deduplicated alias occurrences.
        """
        return GlossaryOccurrenceMatcher(text).recurring_terms(
            terms,
            min_occurrences=min_occurrences,
        )

    def mark_conflicts_resolved(self, source: str) -> None:
        """Mark every unresolved conflict for the given source term as handled.

        A term left flagged ``conflict`` once its rows are handled would silently drop out of the
        always-on set and keep reading as contested, so the status returns to ``ok``. An
        operator's lock is never downgraded: only a term still marked contested is restored.
        """
        self.conn.execute("UPDATE term_conflicts SET resolved=1 WHERE source=?", (source,))
        self.conn.execute(
            "UPDATE glossary SET status='ok', updated_at=? WHERE source=? AND status='conflict'",
            (time.time(), source),
        )
        if self._owns_connection:
            self.conn.commit()

    def open_conflicts(self) -> list[dict[str, Any]]:
        """Return conflicts awaiting human resolution in occurrence order.

        A locked term already carries the operator's authoritative target, so the proposals
        recorded against it are moot: nothing re-raises them and they must not gate a run.
        """
        rows = self.conn.execute(
            """SELECT c.* FROM term_conflicts c
               LEFT JOIN glossary g ON g.source = c.source
               WHERE c.resolved = 0 AND COALESCE(g.status, '') <> ?
               ORDER BY c.created_at""",
            (MANUAL_STATUS,),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, int]:
        """Return glossary and unresolved-conflict counts, skipping moot locked conflicts."""
        terms = self.conn.execute("SELECT COUNT(*) FROM glossary").fetchone()[0]
        conflicts = self.conn.execute(
            """SELECT COUNT(*) FROM term_conflicts c
               LEFT JOIN glossary g ON g.source = c.source
               WHERE c.resolved = 0 AND COALESCE(g.status, '') <> ?""",
            (MANUAL_STATUS,),
        ).fetchone()[0]
        return {"terms": terms, "open_conflicts": conflicts}


def merge_always_on(
    selected: list[GlossaryTerm],
    all_terms: list[GlossaryTerm],
    source_corpus: str,
    *,
    always_types: Sequence[str] = (TYPE_PERSON,),
    min_occurrences: int = 3,
    max_always: int = 12,
) -> list[GlossaryTerm]:
    """Append locked high-frequency always-on entities missing from a filtered list.

    Always-on entities are terms of the configured types whose source or aliases occur at
    least ``min_occurrences`` times in the book corpus: either a settled mapping, or one an
    operator set by hand. Keep chapter-filtered terms in place and append extras in insertion
    order so main characters stay visible even when a chapter does not mention them. A term
    under unresolved disagreement is never always-on.
    """
    if max_always <= 0:
        return list(selected)
    type_set = set(always_types)
    selected_keys = {term.source for term in selected}
    candidates = [
        term
        for term in all_terms
        if term.type in type_set
        and (term.status == "ok" or term.status == MANUAL_STATUS)
        and term.source not in selected_keys
    ]
    if not candidates:
        return list(selected)
    recurring = GlossaryStore.recurring_terms(
        candidates,
        source_corpus,
        min_occurrences=min_occurrences,
    )
    extras: list[GlossaryTerm] = []
    for term in recurring:
        if len(extras) >= max_always:
            break
        if term.source in selected_keys:
            continue
        extras.append(term)
        selected_keys.add(term.source)
    return [*selected, *extras]

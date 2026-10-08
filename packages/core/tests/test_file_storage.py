"""The local adapter preserves glossary lifetime and read-only Review semantics."""

import sqlite3
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import pytest
from wenyi_core.glossary.store import GlossaryStore, GlossaryTerm
from wenyi_core.storage.file import FileStorage
from wenyi_core.storage.locks import exclusive_file_lock


@pytest.mark.parametrize("interrupted", [False, True])
def test_run_lock_closes_glossary_and_allows_reuse(tmp_path, interrupted):
    storage = FileStorage(str(tmp_path))
    term = GlossaryTerm(source="Alice", target="爱丽丝")
    with pytest.raises(RuntimeError, match="interrupted") if interrupted else nullcontext():
        with storage.lock():
            storage.upsert_term(term)
            connection = storage._g.conn
            if interrupted:
                raise RuntimeError("interrupted")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with storage.lock():
        assert storage.get_term(term.source) == term


def test_reading_terms_keeps_formal_glossary_and_wal_unchanged(tmp_path):
    storage = FileStorage(str(tmp_path))
    writer = GlossaryStore(storage.glossary_path)
    try:
        writer.conn.execute("PRAGMA wal_autocheckpoint=0")
        term = GlossaryTerm(source="Alice", target="爱丽丝")
        writer.upsert_term(term)

        def snapshot():
            return {
                path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in tmp_path.glob("glossary.db*")
            }

        before = snapshot()
        assert storage.all_terms() == [term]
        assert storage._glossary is None
        assert snapshot() == before
    finally:
        storage.close()
        writer.close()


def test_reading_missing_glossary_does_not_create_database(tmp_path):
    storage = FileStorage(str(tmp_path))
    try:
        assert storage.all_terms() == []
        assert not Path(storage.glossary_path).exists()
    finally:
        storage.close()


def test_exclusive_file_lock_waits_for_a_held_lock(tmp_path):
    """A held lock makes the next acquirer wait rather than give up after ten retries."""
    lock_path = str(tmp_path / "shared.lock")
    holding = threading.Event()
    release = threading.Event()

    def hold():
        with exclusive_file_lock(lock_path):
            holding.set()
            release.wait(10)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    try:
        assert holding.wait(10), "the holder never acquired the lock"
        # Release while the main thread is already blocked on the same lock.
        threading.Timer(0.5, release.set).start()
        started = time.monotonic()
        with exclusive_file_lock(lock_path):
            pass
        assert time.monotonic() - started >= 0.4
    finally:
        release.set()
        holder.join(10)


def test_exclusive_file_lock_does_not_use_the_bounded_windows_lock():
    """LK_LOCK abandons the wait after ten one-second retries and raises EDEADLK, which turns
    "another run holds this book" into a hard failure; the shared lock has to keep waiting."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(exclusive_file_lock))
    modes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "msvcrt"
    }
    assert "LK_NBLCK" in modes
    assert "LK_LOCK" not in modes

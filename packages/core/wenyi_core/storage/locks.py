"""Cross-process advisory file lock shared by the local state stores."""

from __future__ import annotations

import errno
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def exclusive_file_lock(lock_path: str) -> Iterator[None]:
    """Hold an exclusive OS-level lock on ``lock_path`` (created if missing) until exit."""
    with open(lock_path, "a+b") as lock_file:
        if os.name == "nt":  # pragma: no cover - Windows-specific
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            # LK_LOCK abandons the wait after ten one-second retries and raises EDEADLK, which
            # turns "another run holds this book" into a hard failure instead of waiting.
            while True:
                lock_file.seek(0)
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

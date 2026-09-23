"""Unit tests for constrained automatic glossary locking."""

from __future__ import annotations

import unittest

from wenyi_core.glossary.auto_lock import can_auto_lock, should_write_auto_lock
from wenyi_core.glossary.store import GlossaryTerm


class TestAutoLock(unittest.TestCase):
    def test_requires_all_four_gates(self):
        term = GlossaryTerm(source="Ann", target="安", type="person")
        self.assertTrue(
            can_auto_lock(
                term,
                history_aligned=True,
                occurrences=2,
                has_open_conflict=False,
            )
        )
        self.assertFalse(
            can_auto_lock(
                term,
                history_aligned=False,
                occurrences=2,
                has_open_conflict=False,
            )
        )
        self.assertFalse(
            can_auto_lock(
                term,
                history_aligned=True,
                occurrences=1,
                has_open_conflict=False,
            )
        )
        self.assertFalse(
            can_auto_lock(
                term,
                history_aligned=True,
                occurrences=2,
                has_open_conflict=True,
            )
        )
        speech = GlossaryTerm(source="っぽい", target="口癖", type="speech")
        self.assertFalse(
            can_auto_lock(
                speech,
                history_aligned=True,
                occurrences=5,
                has_open_conflict=False,
            )
        )

    def test_never_overwrites_established_target(self):
        established = GlossaryTerm(source="Ann", target="安", status="ok")
        self.assertFalse(
            should_write_auto_lock(established, GlossaryTerm(source="Ann", target="安娜"))
        )
        empty = GlossaryTerm(source="Ann", target="", status="ok")
        self.assertTrue(should_write_auto_lock(empty, GlossaryTerm(source="Ann", target="安")))
        self.assertTrue(should_write_auto_lock(None, GlossaryTerm(source="Ann", target="安")))
        self.assertFalse(should_write_auto_lock(None, GlossaryTerm(source="Ann", target="   ")))


if __name__ == "__main__":
    unittest.main()

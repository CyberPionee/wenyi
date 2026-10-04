"""Tests for source-anchored glossary target writeback."""

from __future__ import annotations

import os
import tempfile
import unittest

from wenyi_core.glossary.store import GlossaryTerm
from wenyi_core.glossary.writeback import apply_term_writeback, rewrite_target_text
from wenyi_core.ingest.models import Chapter, Document, Segment
from wenyi_core.storage.file import FileStorage


def _segment(index: int, source: str, target: str) -> Segment:
    return Segment(index=index, source=source, target=target)


def _store_with_segments(directory: str, segments: list[Segment]) -> FileStorage:
    store = FileStorage(os.path.join(directory, "state"))
    doc = Document(
        title="T",
        fmt="text",
        source_lang="ja",
        target_lang="zh",
        chapters=[Chapter(index=0, title="c", segments=segments)],
    )
    store.begin_initialization("a" * 64)
    manifest = store.stage_document(doc, source_hash="a" * 64)
    manifest["initialized"] = True
    store.save_manifest(manifest)
    return store


class RewriteTextTests(unittest.TestCase):
    def test_replaces_only_when_old_present(self):
        self.assertEqual(
            rewrite_target_text("海豚酒店很大", "海豚酒店", "海豚酒店（Dolphin Hotel）"),
            "海豚酒店（Dolphin Hotel）很大",
        )
        self.assertIsNone(rewrite_target_text("别的译文", "海豚酒店", "海豚酒店（Dolphin Hotel）"))
        self.assertIsNone(rewrite_target_text("海豚酒店", "海豚酒店", "海豚酒店"))
        self.assertIsNone(rewrite_target_text("x", "", "y"))


class WritebackTests(unittest.TestCase):
    def test_rewrites_only_matching_segments(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store_with_segments(
                d,
                [
                    _segment(0, "ドルフィン・ホテルへ行く。", "去了海豚酒店。"),
                    _segment(1, "別の場所へ行く。", "去了海豚酒店。"),
                    _segment(2, "ドルフィン・ホテルに戻る。", "回到旅馆。"),
                ],
            )
            term = GlossaryTerm(
                source="ドルフィン・ホテル",
                target="海豚酒店（Dolphin Hotel）",
                type="place",
            )
            summary = apply_term_writeback(
                store,
                term_source="ドルフィン・ホテル",
                term=term,
                old_target="海豚酒店",
                new_target="海豚酒店（Dolphin Hotel）",
            )

            self.assertEqual(summary["segments_replaced"], 1)
            self.assertEqual(summary["chapters_touched"], 1)
            self.assertEqual(summary["matched_segments"], 1)
            updated = store.load_chapter(0)
            self.assertEqual(updated.segments[0].target, "去了海豚酒店（Dolphin Hotel）。")
            self.assertEqual(updated.segments[1].target, "去了海豚酒店。")
            self.assertEqual(updated.segments[2].target, "回到旅馆。")

    def test_rewrites_by_alias_match(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store_with_segments(
                d,
                [
                    _segment(0, "海豚ホテルは静かだ。", "海豚酒店很安静。"),
                ],
            )
            term = GlossaryTerm(
                source="ドルフィン・ホテル",
                target="海豚酒店（Dolphin Hotel）",
                type="place",
                aliases=["海豚ホテル"],
            )
            summary = apply_term_writeback(
                store,
                term_source="ドルフィン・ホテル",
                term=term,
                old_target="海豚酒店",
                new_target="海豚酒店（Dolphin Hotel）",
            )
            self.assertEqual(summary["segments_replaced"], 1)
            self.assertEqual(
                store.load_chapter(0).segments[0].target, "海豚酒店（Dolphin Hotel）很安静。"
            )

    def test_noop_when_target_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store_with_segments(d, [_segment(0, "A is here.", "B is here.")])
            summary = apply_term_writeback(
                store,
                term_source="A",
                term=GlossaryTerm(source="A", target="B", type="term"),
                old_target="B",
                new_target="B",
            )
            self.assertEqual(summary["segments_replaced"], 0)
            self.assertEqual(summary["matched_segments"], 0)


if __name__ == "__main__":
    unittest.main()

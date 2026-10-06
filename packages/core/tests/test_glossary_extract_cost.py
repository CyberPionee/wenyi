"""Cost and injection selection guards for glossary extraction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wenyi_core.agents.prompts import render_glossary
from wenyi_core.config import Config
from wenyi_core.glossary.extractor import GlossaryExtractor, TranslatedSegmentEvidence
from wenyi_core.glossary.injection import select_extraction_terms
from wenyi_core.glossary.store import (
    GlossaryOccurrenceMatcher,
    GlossaryStore,
    GlossaryTerm,
)


class _CountingClient:
    def __init__(self, response=None):
        self.calls = 0
        self.response = response if response is not None else {}

    def complete(self, **kwargs):  # pragma: no cover - simplified surface
        self.calls += 1
        return self.response


def _terms(n: int) -> list[GlossaryTerm]:
    return [
        GlossaryTerm(source=f"S{i}", target=f"T{i}", type="person", note=f"note-{i}")
        for i in range(n)
    ]


class RenderGlossaryTests(unittest.TestCase):
    def test_extraction_render_omits_notes(self):
        text = render_glossary(_terms(2), include_note=False, max_note_chars=120)
        self.assertNotIn("Note:", text)
        self.assertIn("S0", text)

    def test_translation_render_keeps_notes(self):
        text = render_glossary(_terms(1), include_note=True, max_note_chars=120)
        self.assertIn("Note: note-0", text)


class InjectionSelectionTests(unittest.TestCase):
    def test_batch_hit_wins_over_later_terms(self):
        terms = [
            GlossaryTerm(source="无关词", target="A", type="term"),
            GlossaryTerm(source="田中", target="田中", type="person"),
        ]
        picked = select_extraction_terms(
            terms,
            batch_text="田中说。",
            budget_chars=10_000,
            min_terms=0,
            core_max=0,
            recent_max=0,
        )
        self.assertEqual(picked[0].source, "田中")
        self.assertTrue(any(t.source == "田中" for t in picked))

    def test_budget_caps_long_glossary(self):
        terms = [
            GlossaryTerm(source=f"角色{i:03d}", target=f"T{i}", type="person") for i in range(80)
        ]
        picked = select_extraction_terms(
            terms,
            batch_text="",
            budget_chars=400,
            min_terms=2,
            core_max=2,
            recent_max=2,
        )
        self.assertLess(len(picked), 80)
        self.assertGreaterEqual(len(picked), 2)

    def test_hit_only_mode(self):
        terms = [
            GlossaryTerm(source="田中", target="田中", type="person"),
            GlossaryTerm(source="无关", target="X", type="term"),
        ]
        picked = select_extraction_terms(
            terms,
            batch_text="田中来了",
            mode="hit_only",
            min_terms=0,
        )
        self.assertEqual([t.source for t in picked], ["田中"])

    def test_zero_hit_fallback_min_terms(self):
        terms = [GlossaryTerm(source=f"词{i}", target=f"T{i}", type="term") for i in range(10)]
        picked = select_extraction_terms(
            terms,
            batch_text="完全无关的段落",
            budget_chars=10_000,
            min_terms=5,
            core_max=0,
            recent_max=0,
        )
        self.assertGreaterEqual(len(picked), 5)


class FinalizeChapterTests(unittest.TestCase):
    def test_finalize_is_local_and_fills_from_history(self):
        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(str(Path(d) / "g.db"))
            try:
                store.upsert_term(GlossaryTerm(source="田中", target="", type="person"))
                cfg = Config.from_dict({})
                client = _CountingClient()
                extractor = GlossaryExtractor(client, cfg)  # type: ignore[arg-type]
                before = (1, 10)
                history = [
                    TranslatedSegmentEvidence(
                        chapter=1, segment=2, source="田中说。", target="田中说。"
                    )
                ]
                summary = extractor.finalize_chapter_glossary(
                    store,
                    1,
                    history=history,
                    before=before,
                    source_corpus="田中说。田中又来。",
                )
                self.assertEqual(summary.get("llm_calls"), 0)
                self.assertEqual(summary.get("mode"), "local_finalize")
                self.assertEqual(client.calls, 0)
                term = store.get_term("田中")
                self.assertIsNotNone(term)
                assert term is not None
                self.assertTrue((term.target or "").strip())
            finally:
                store.close()

    def test_finalize_reports_only_mappings_it_established(self):
        """Every chapter close-out used to re-lock every established term in the book."""
        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(str(Path(d) / "g.db"))
            try:
                store.upsert_term(GlossaryTerm(source="Ann", target="安", type="person"))
                store.upsert_term(GlossaryTerm(source="Beth", target="", type="person"))
                extractor = GlossaryExtractor(_CountingClient(), Config.from_dict({}))  # type: ignore[arg-type]
                history = [
                    TranslatedSegmentEvidence(chapter=1, segment=2, source="Ann", target="安"),
                    TranslatedSegmentEvidence(chapter=1, segment=3, source="Beth", target="贝丝"),
                ]
                locked: list[tuple[str, str]] = []
                summary = extractor.finalize_chapter_glossary(
                    store,
                    1,
                    history=history,
                    before=(1, 10),
                    source_corpus="Ann Ann Beth Beth",
                    on_auto_lock=lambda source, target: locked.append((source, target)),
                )
                self.assertEqual(locked, [("Beth", "贝丝")])
                self.assertEqual(summary["auto_locked"], 1)
            finally:
                store.close()

    def test_recurrence_matcher_is_reused_across_calls(self):
        """Rebuilding the matcher per call re-normalized the whole book corpus every time."""
        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(str(Path(d) / "g.db"))
            try:
                store.upsert_term(GlossaryTerm(source="Ann", target="安", type="person"))
                extractor = GlossaryExtractor(_CountingClient(), Config.from_dict({}))  # type: ignore[arg-type]
                corpus = "Ann Ann"
                with mock.patch(
                    "wenyi_core.glossary.extractor.GlossaryOccurrenceMatcher",
                    wraps=GlossaryOccurrenceMatcher,
                ) as matcher:
                    for _ in range(3):
                        extractor.finalize_chapter_glossary(store, 1, source_corpus=corpus)
                    self.assertEqual(matcher.call_count, 1)
            finally:
                store.close()


class ConfigDefaultsTests(unittest.TestCase):
    def test_inject_defaults(self):
        cfg = Config.from_dict({})
        self.assertEqual(cfg.pipeline.glossary_extract_inject, "smart")
        self.assertEqual(cfg.pipeline.glossary_extract_budget_chars, 4000)


if __name__ == "__main__":
    unittest.main()

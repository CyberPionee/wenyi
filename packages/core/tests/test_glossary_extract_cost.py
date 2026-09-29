"""Cost and prompt-size guards for glossary extraction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wenyi_core.agents.prompts import render_glossary
from wenyi_core.config import Config
from wenyi_core.glossary.extractor import (
    GlossaryExtractor,
    TranslatedSegmentEvidence,
    select_extraction_context_terms,
)
from wenyi_core.glossary.store import GlossaryStore, GlossaryTerm


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
    def test_max_terms_caps_prompt(self):
        text = render_glossary(_terms(20), include_note=False, max_terms=5)
        self.assertEqual(len([line for line in text.splitlines() if line.startswith("- ")]), 5)

    def test_extraction_render_omits_notes(self):
        text = render_glossary(_terms(2), include_note=False, max_note_chars=120)
        self.assertNotIn("Note:", text)
        self.assertIn("S0", text)

    def test_translation_render_keeps_notes(self):
        text = render_glossary(_terms(1), include_note=True, max_note_chars=120)
        self.assertIn("Note: note-0", text)


class ExtractionContextSelectionTests(unittest.TestCase):
    def test_selects_only_batch_related_terms(self):
        terms = [
            GlossaryTerm(source="田中", target="田中", type="person"),
            GlossaryTerm(source="大阪", target="大阪", type="place"),
            GlossaryTerm(source="独有招式", target="绝技", type="technique"),
        ]
        picked = select_extraction_context_terms(
            terms, "田中去了东京。", "田中去了东京。", max_terms=80
        )
        self.assertEqual([t.source for t in picked], ["田中"])

    def test_target_only_hits_appended_after_source(self):
        terms = [
            GlossaryTerm(source="田中", target="田中", type="person"),
            GlossaryTerm(source="Tanaka", target="田中", type="appellation"),
        ]
        picked = select_extraction_context_terms(terms, "他说完了。", "田中说完了。", max_terms=80)
        self.assertEqual(sorted(t.source for t in picked), ["Tanaka", "田中"])
        picked2 = select_extraction_context_terms(terms, "田中走了。", "Tanaka left.", max_terms=80)
        self.assertIn("田中", [t.source for t in picked2])

    def test_safety_cap_only_when_over_limit(self):
        terms = [GlossaryTerm(source=f"词{i}", target=f"译{i}", type="term") for i in range(5)]
        text = " ".join(f"词{i}" for i in range(5))
        picked = select_extraction_context_terms(terms, text, max_terms=3)
        self.assertEqual(len(picked), 3)
        picked_all = select_extraction_context_terms(terms, text, max_terms=80)
        self.assertEqual(len(picked_all), 5)


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


class ExtractPromptBudgetTests(unittest.TestCase):
    def test_default_prompt_term_cap(self):
        cfg = Config.from_dict({})
        self.assertEqual(cfg.pipeline.glossary_extract_max_prompt_terms, 80)


if __name__ == "__main__":
    unittest.main()

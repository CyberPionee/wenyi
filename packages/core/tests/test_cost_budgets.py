"""Structural cost-cap checks for standard-tier translation prompts (plan §5)."""

from __future__ import annotations

import unittest

from wenyi_core.config import Config
from wenyi_core.llm.operations import OPERATIONS


class TestCostBudgets(unittest.TestCase):
    def test_standard_quality_budgets_match_plan(self):
        cfg = Config.from_dict({"llm": {"preset": "fake"}})
        pipeline = cfg.pipeline
        # Rolling context: 8 source-target pairs (quality tier may use 12).
        self.assertEqual(pipeline.rolling_context_segments, 8)
        self.assertLessEqual(pipeline.rolling_context_segments, 12)
        # Glossary note / always-on caps.
        self.assertEqual(pipeline.glossary_note_chars, 120)
        self.assertEqual(pipeline.glossary_always_min_occurrences, 3)
        # C-batch stays opt-in so the one-click path is unchanged.
        for flag in (
            "self_revision",
            "editorial_pass",
            "final_polish",
            "chapter_selfcheck",
            "back_translation",
        ):
            self.assertFalse(getattr(pipeline, flag), flag)

    def test_quality_operations_are_registered(self):
        for name in (
            "quality.self_revision",
            "quality.editorial",
            "quality.final_polish",
            "quality.chapter_selfcheck",
            "quality.back_translation",
        ):
            self.assertIn(name, OPERATIONS)


if __name__ == "__main__":
    unittest.main()

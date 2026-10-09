"""Unit tests for deterministic residual sweeps and auto_qa report aggregation."""

from __future__ import annotations

import unittest

from wenyi_core.glossary.store import GlossaryTerm
from wenyi_core.review.sweep import (
    scan_foreign_unbracketed,
    scan_number_residue,
    scan_segment,
    scan_term_drift,
    scan_untranslated_residue,
)


class TestSweep(unittest.TestCase):
    def test_number_residue_detects_missing_digits(self):
        finding = scan_number_residue("Chapter 12 starts in 2024", "Chapter starts")
        self.assertIsNotNone(finding)
        assert finding is not None
        self.assertEqual(finding["kind"], "number_residue")
        self.assertEqual(finding["missing_numbers"], ["12", "2024"])
        self.assertIsNone(scan_number_residue("Chapter 12", "第12章"))
        # Substring lookalikes must not hide a missing number.
        self.assertIsNotNone(scan_number_residue("room 12", "room 112"))
        self.assertIsNotNone(scan_number_residue("room 1", "room 21"))
        self.assertIsNone(scan_number_residue("room 12", "room 12 done"))

    def test_untranslated_residue_flags_cjk_in_latin_target(self):
        finding = scan_untranslated_residue("彼は言った", "He said something 彼は言った here")
        self.assertIsNotNone(finding)
        assert finding is not None
        self.assertEqual(finding["kind"], "untranslated_residue")
        self.assertIsNone(scan_untranslated_residue("彼は言った", "他说"))

    def test_untranslated_residue_skips_urls_and_proper_nouns(self):
        # URLs / domain fragments in a CJK target
        self.assertIsNone(
            scan_untranslated_residue(
                "版权页 permissions hbgusa com",
                "版权声明：permissions / hbgusa / com 请遵守。",
            )
        )
        self.assertIsNone(
            scan_untranslated_residue(
                "Copyright Little Brown Company",
                "版权所有 Little / Brown 出版公司。",
            )
        )
        # Real leftover lowercase prose is still flagged
        finding = scan_untranslated_residue(
            "He said hello world clearly",
            "他认真的说完了 hello world 这一段话之后就离开了这里",
        )
        self.assertIsNotNone(finding)

    def test_untranslated_residue_flags_latin_in_cjk_target(self):
        finding = scan_untranslated_residue(
            "He said Hello world clearly",
            "他认真的说完了 Hello world 这一段话之后就离开了这里",
        )
        self.assertIsNotNone(finding)
        assert finding is not None
        self.assertEqual(finding["kind"], "untranslated_residue")
        # Loanwords that never appear in the source stay unflagged.
        self.assertIsNone(scan_untranslated_residue("他开口说话了", "他说 OK 了"))

    def test_untranslated_residue_flags_cyrillic_in_latin_target(self):
        finding = scan_untranslated_residue("Он сказал привет", "He said привет now")
        self.assertIsNotNone(finding)
        self.assertIsNone(scan_untranslated_residue("Он сказал привет", "Он сказал"))

    def test_term_drift_requires_fixed_mapping_in_target(self):
        terms = [GlossaryTerm(source="Ann", target="安", type="person")]
        self.assertIsNotNone(scan_term_drift("Ann left", "Anne left", terms))
        self.assertIsNone(scan_term_drift("Ann left", "安 left", terms))
        self.assertIsNone(scan_term_drift("Bob left", "Anne left", terms))

    def test_foreign_run_without_brackets_is_flagged(self):
        finding = scan_foreign_unbracketed(
            "He said hello world clearly and left at once",
            "他说完了 hello world 这一段就离开了这里",
        )
        self.assertIsNotNone(finding)
        assert finding is not None
        self.assertEqual(finding["kind"], "foreign_unbracketed")
        self.assertIn("hello world", finding["unbracketed"])

    def test_half_width_brackets_make_the_passage_compliant(self):
        self.assertIsNone(
            scan_foreign_unbracketed(
                "He said hello world clearly and left at once",
                "他说完了 hello world (这一段)就离开了这里",
            )
        )

    def test_full_width_brackets_make_the_passage_compliant(self):
        self.assertIsNone(
            scan_foreign_unbracketed(
                "He said hello world clearly and left at once",
                "他说完了 hello world（这一段）就离开了这里",
            )
        )

    def test_single_space_before_the_bracket_is_allowed(self):
        self.assertIsNone(
            scan_foreign_unbracketed(
                "He said hello world clearly and left at once",
                "他说完了 hello world （这一段）就离开了这里",
            )
        )

    def test_isolated_proper_noun_never_requires_brackets(self):
        self.assertIsNone(
            scan_foreign_unbracketed(
                "They visited Paris today by train",
                "他们今天乘火车访问了 Paris 地区",
            )
        )

    def test_url_context_is_skipped(self):
        self.assertIsNone(
            scan_foreign_unbracketed(
                "See example.com/path for the archive",
                "请在 example.com/path 页面查看存档内容",
            )
        )

    def test_foreign_run_absent_from_source_is_ignored(self):
        # The rule only covers passages the source itself writes in another language;
        # wording the translator introduced is a different concern.
        self.assertIsNone(
            scan_foreign_unbracketed(
                "他说了很多中文内容然后停了下来",
                "他说了很多中文内容 elephant 很大然后停了下来",
            )
        )

    def test_empty_inputs_return_none(self):
        self.assertIsNone(scan_foreign_unbracketed("", "他说中文内容很多"))
        self.assertIsNone(scan_foreign_unbracketed("Source text here", ""))

    def test_scan_segment_aggregates_foreign_unbracketed(self):
        findings = scan_segment(
            "He said hello world clearly and left at once",
            "他说完了 hello world 这一段就离开了这里",
        )
        kinds = {item["kind"] for item in findings}
        self.assertIn("foreign_unbracketed", kinds)

    def test_bracketed_passage_is_not_untranslated_residue(self):
        # Regression: correctly bracketed foreign text must not count as residue,
        # matching reviewer_system.txt's instruction never to report it.
        self.assertIsNone(
            scan_untranslated_residue(
                "He said hello world clearly and left at once",
                "他说完了 hello world (这一段)就离开了这里",
            )
        )
        self.assertIsNone(
            scan_untranslated_residue(
                "He said hello world clearly and left at once",
                "他说完了 hello world（这一段）就离开了这里",
            )
        )

    def test_foreign_unbracketed_maps_to_missing_issue_type(self):
        from wenyi_core.pipeline.autofix_candidates import _SWEEP_ISSUE_TYPE

        self.assertEqual(_SWEEP_ISSUE_TYPE["foreign_unbracketed"], "missing")

    def test_scan_segment_collects_all_residuals(self):
        terms = [GlossaryTerm(source="Ann", target="安", type="person")]
        findings = scan_segment("Ann left in 1999", "Anne left", terms)
        kinds = {item["kind"] for item in findings}
        self.assertEqual(kinds, {"number_residue", "term_drift"})


if __name__ == "__main__":
    unittest.main()

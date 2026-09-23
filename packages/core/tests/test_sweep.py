"""Unit tests for deterministic residual sweeps and auto_qa report aggregation."""

from __future__ import annotations

import unittest

from wenyi_core.glossary.store import GlossaryTerm
from wenyi_core.review.sweep import (
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

    def test_untranslated_residue_flags_cjk_in_latin_target(self):
        finding = scan_untranslated_residue("彼は言った", "He said something 彼は言った here")
        self.assertIsNotNone(finding)
        assert finding is not None
        self.assertEqual(finding["kind"], "untranslated_residue")
        self.assertIsNone(scan_untranslated_residue("彼は言った", "他说"))

    def test_term_drift_requires_fixed_mapping_in_target(self):
        terms = [GlossaryTerm(source="Ann", target="安", type="person")]
        self.assertIsNotNone(scan_term_drift("Ann left", "Anne left", terms))
        self.assertIsNone(scan_term_drift("Ann left", "安 left", terms))
        self.assertIsNone(scan_term_drift("Bob left", "Anne left", terms))

    def test_scan_segment_collects_all_residuals(self):
        terms = [GlossaryTerm(source="Ann", target="安", type="person")]
        findings = scan_segment("Ann left in 1999", "Anne left", terms)
        kinds = {item["kind"] for item in findings}
        self.assertEqual(kinds, {"number_residue", "term_drift"})


if __name__ == "__main__":
    unittest.main()

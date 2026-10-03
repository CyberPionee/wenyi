"""Unit tests for L0-L3 evaluation gate and decision anchors."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from wenyi_core.glossary.store import GlossaryTerm
from wenyi_core.pipeline.decision_anchors import (
    distill_decision_anchors,
    render_decision_anchors,
)
from wenyi_core.pipeline.evaluation import (
    back_translation_similarity,
    build_machine_gate,
    evaluation_low_score_issues,
    quality_notes_to_autofix_issues,
    scan_term_consistency,
    score_back_translations,
    segment_risk_reasons,
    select_risk_segments,
)


def _chapter(index: int, segments: list[tuple[int, str, str]]):
    texts = [SimpleNamespace(index=i, source=s, target=t) for i, s, t in segments]
    return SimpleNamespace(index=index, text_segments=texts)


class RiskSelectionTests(unittest.TestCase):
    def test_dialog_and_empty_are_flagged(self):
        reasons = segment_risk_reasons("「你好。」", "Hello.", findings=[])
        self.assertIn("dialog", reasons)
        self.assertEqual(segment_risk_reasons("你好。", ""), ["empty_target"])

    def test_select_includes_risk_and_sample(self):
        chapters = [
            _chapter(
                1,
                [
                    (0, "普通叙述短句。", "Plain short line."),
                    (1, "「对话来了。」他说。", '"Dialogue," he said.'),
                    (2, "另一句普通。", "Another plain line."),
                    (3, "还是普通。", "Still plain."),
                ],
            )
        ]
        selected = select_risk_segments(chapters, sample_ratio=0.25)
        reasons = {item.index: item.reasons for item in selected}
        self.assertIn(1, reasons)
        self.assertIn("dialog", reasons[1])
        self.assertTrue(any("sample" in item.reasons for item in selected))


class GateTests(unittest.TestCase):
    def test_gate_pass_and_fail(self):
        ok = build_machine_gate(
            l0={
                "empty_target_count": 0,
                "open_conflict_count": 0,
                "residual_finding_count": 0,
                "open_issue_count": 0,
            }
        )
        self.assertTrue(ok["passed"])
        self.assertFalse(ok["blocking"])
        bad = build_machine_gate(
            l0={
                "empty_target_count": 1,
                "open_conflict_count": 0,
                "residual_finding_count": 0,
                "open_issue_count": 0,
            },
            back_translation=[{"score": 0.1}],
            judge_scores=[{"score": 2.0}],
            bt_score_min=0.45,
            judge_score_min=3.5,
        )
        self.assertFalse(bad["passed"])
        self.assertTrue(bad["blocking"])
        self.assertEqual(bad["bt_low_count"], 1)

    def test_similarity_and_score_pairs(self):
        self.assertGreater(back_translation_similarity("hello world", "hello world"), 0.9)
        scores = score_back_translations([(0, 3, "hello world", "你好世界")], ["hello world"])
        self.assertEqual(scores[0]["score"] > 0.5, True)
        self.assertEqual(scores[0]["chapter"], 0)
        self.assertEqual(scores[0]["index"], 3)

    def test_quality_notes_map_to_issues(self):
        issues = quality_notes_to_autofix_issues(
            [{"chapter": 2, "index": 5, "suggested": "改后的句子"}]
        )
        self.assertEqual(issues[0]["index"], 5)
        self.assertEqual(issues[0]["suggestion"], "改后的句子")


class TermConsistencyTests(unittest.TestCase):
    """L2 measures glossary target usage as a rate over covered segments."""

    def test_rate_and_drift_items(self):
        overrides = {
            "田中": "田中",
            "海豚酒店": "海豚酒店（Dolphin Hotel）",
        }
        terms = [
            GlossaryTerm(source=src, target=tgt, type="place") for src, tgt in overrides.items()
        ]
        chapters = [
            _chapter(
                1,
                [
                    (0, "田中说。", "田中说。"),  # consistent
                    (1, "海豚酒店很安静。", "那家旅馆很安静。"),  # drifted
                    (2, "无关段落。", "无关段落。"),  # not covered
                ],
            )
        ]
        result = scan_term_consistency(chapters, terms)
        self.assertEqual(result["checked"], 2)
        self.assertEqual(result["drifted"], 1)
        self.assertEqual(result["consistency_rate"], 0.5)
        self.assertEqual(result["items"][0]["index"], 1)
        self.assertIn("海豚酒店（Dolphin Hotel）", result["items"][0]["missing_targets"])

    def test_empty_targets_are_not_l2(self):
        terms = [GlossaryTerm(source="田中", target="", type="person")]
        chapters = [_chapter(1, [(0, "田中说。", "他说。")])]
        result = scan_term_consistency(chapters, terms)
        self.assertEqual(result["checked"], 0)
        self.assertEqual(result["consistency_rate"], 1.0)

    def test_no_terms_is_neutral(self):
        chapters = [_chapter(1, [(0, "甲。", "甲。")])]
        result = scan_term_consistency(chapters, [])
        self.assertEqual(result["checked"], 0)
        self.assertEqual(result["items"], [])


class GateL2Tests(unittest.TestCase):
    def test_default_threshold_keeps_strict_behavior(self):
        gate = build_machine_gate(
            l0={
                "empty_target_count": 0,
                "open_conflict_count": 0,
                "residual_finding_count": 1,
                "open_issue_count": 0,
            },
            l2={"checked": 10, "drifted": 1, "consistency_rate": 0.9},
        )
        self.assertFalse(gate["l2_passed"])
        self.assertFalse(gate["l0_passed"])
        self.assertTrue(gate["blocking"])

    def test_tolerance_shifts_drift_decision_to_l2(self):
        gate = build_machine_gate(
            l0={
                "empty_target_count": 0,
                "open_conflict_count": 0,
                "residual_finding_count": 1,
                "open_issue_count": 0,
            },
            l2={"checked": 10, "drifted": 1, "consistency_rate": 0.9},
            l2_min_consistency=0.8,
        )
        # The same drift must not fail L0 twice once L2 owns the decision.
        self.assertEqual(gate["l0_residual_finding_count"], 0)
        self.assertTrue(gate["l0_passed"])
        self.assertTrue(gate["l2_passed"])
        self.assertTrue(gate["passed"])

    def test_drift_above_tolerance_still_fails(self):
        gate = build_machine_gate(
            l0={
                "empty_target_count": 0,
                "open_conflict_count": 0,
                "residual_finding_count": 2,
                "open_issue_count": 0,
            },
            l2={"checked": 10, "drifted": 3, "consistency_rate": 0.7},
            l2_min_consistency=0.8,
        )
        self.assertFalse(gate["l2_passed"])
        self.assertTrue(gate["blocking"])


class EvaluationIssueMappingTests(unittest.TestCase):
    def test_low_score_and_drift_become_located_issues(self):
        evaluation = {
            "l2": {
                "items": [
                    {
                        "chapter": 1,
                        "index": 2,
                        "source_term": "A",
                        "expected_target": "甲",
                        "missing_targets": ["甲"],
                    }
                ]
            },
            "back_translation": [{"chapter": 1, "index": 5, "score": 0.1}],
            "judge_scores": [{"chapter": 1, "index": 7, "score": 2.0, "note": "fluent but flat"}],
        }
        issues = evaluation_low_score_issues(evaluation, bt_min=0.45, judge_min=3.5)
        self.assertEqual(len(issues), 3)
        self.assertEqual({issue["index"] for issue in issues}, {2, 5, 7})
        l2_issue = next(issue for issue in issues if issue["issue_key"].startswith("eval_l2"))
        self.assertEqual(l2_issue["suggestion"], "甲")
        self.assertEqual(l2_issue["type"], "terminology")

    def test_passing_findings_are_not_mapped(self):
        evaluation = {
            "l2": {"items": []},
            "back_translation": [{"chapter": 0, "index": 0, "score": 0.9}],
            "judge_scores": [{"chapter": 0, "index": 1, "score": 4.5}],
        }
        issues = evaluation_low_score_issues(evaluation, bt_min=0.45, judge_min=3.5)
        self.assertEqual(issues, [])


class AnchorTests(unittest.TestCase):
    def test_distill_and_render(self):
        pairs = [
            ("田中说：好。", "田中说：好。"),
            ("田中又来了。", "田中又来了。"),
            ("Hello, world.", "你好，世界。"),
            ("Hello, again.", "你好，再次。"),
        ]
        anchors = distill_decision_anchors(pairs)
        self.assertFalse(anchors["empty"])
        text = render_decision_anchors(anchors)
        self.assertIn("MUST", text)
        self.assertIn("Decision anchors", text)
        self.assertEqual(render_decision_anchors({"empty": True}), "")


if __name__ == "__main__":
    unittest.main()

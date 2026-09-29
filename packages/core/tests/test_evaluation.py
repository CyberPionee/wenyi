"""Unit tests for L0-L3 evaluation gate and decision anchors."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from wenyi_core.pipeline.decision_anchors import (
    distill_decision_anchors,
    render_decision_anchors,
)
from wenyi_core.pipeline.evaluation import (
    back_translation_similarity,
    build_machine_gate,
    quality_notes_to_autofix_issues,
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
        scores = score_back_translations([("hello world", "你好世界")], ["hello world"])
        self.assertEqual(scores[0]["score"] > 0.5, True)

    def test_quality_notes_map_to_issues(self):
        issues = quality_notes_to_autofix_issues(
            [{"chapter": 2, "index": 5, "suggested": "改后的句子"}]
        )
        self.assertEqual(issues[0]["index"], 5)
        self.assertEqual(issues[0]["suggestion"], "改后的句子")


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

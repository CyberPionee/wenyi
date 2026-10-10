"""Unit tests for L0-L3 evaluation gate and decision anchors."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from wenyi_core.glossary.store import GlossaryTerm
from wenyi_core.pipeline.decision_anchors import (
    distill_decision_anchors,
    render_decision_anchors,
)
from wenyi_core.pipeline.evaluation import (
    EvaluationService,
    apply_autonomy_tier,
    back_translation_similarity,
    build_machine_gate,
    evaluation_low_score_issues,
    quality_notes_to_autofix_issues,
    quality_pass_notes_to_issues,
    scan_term_consistency,
    score_back_translations,
    segment_risk_reasons,
    select_risk_segments,
)


def _chapter(index: int, segments: list[tuple[int, str, str]]):
    texts = [SimpleNamespace(index=i, source=s, target=t) for i, s, t in segments]
    return SimpleNamespace(index=index, text_segments=texts)


class _StubStore:
    """Only the reads and artifact access EvaluationService performs."""

    def __init__(self, chapters, artifacts=None):
        self._chapters = {chapter.index: chapter for chapter in chapters}
        self.artifacts = artifacts if artifacts is not None else {}

    def load_manifest(self):
        return {"chapters": [{"index": index, "status": "done"} for index in self._chapters]}

    def load_chapter(self, index):
        return self._chapters[index]

    def read_artifact(self, key):
        return self.artifacts.get(key)

    def write_artifact(self, key, value):
        self.artifacts[key] = value


class AcceptanceProgressTests(unittest.TestCase):
    def test_run_reports_each_acceptance_step(self):
        """The acceptance work reports progress; the word-count bar is already saturated."""
        store = _StubStore([_chapter(0, [(0, "原文。", "译文。")])])
        seen: list[tuple[int, int, str]] = []

        EvaluationService(store, risk_back_translation=True, quality_judge=False).run(
            l0={},
            terms=[],
            back_translate=lambda targets: ["" for _ in targets],
            progress=lambda done, total, label: seen.append((done, total, label)),
        )

        labels = [label for _done, _total, label in seen]
        self.assertTrue(any("glossary consistency" in label for label in labels))
        self.assertTrue(any("back-translation sample" in label for label in labels))
        self.assertTrue(any("acceptance gate" in label for label in labels))
        # The judge is off, so it never reports, and the step count matches the work planned.
        self.assertFalse(any("quality judge" in label for label in labels))
        self.assertEqual({total for _done, total, _label in seen}, {3})
        self.assertEqual([done for done, _t, _l in seen], [0, 1, 2, 3])
        self.assertEqual(seen[-1][2], "Machine evaluation · complete")

    def test_sample_results_are_reused_while_the_sampling_and_binding_hold(self):
        """A repeated report step must not pay for the same back-translation and judge calls."""
        store = _StubStore([_chapter(0, [(0, "原文。", "译文。")])])
        calls: list[str] = []

        def evaluate(identity):
            return EvaluationService(store, risk_back_translation=True, quality_judge=True).run(
                l0={},
                terms=[],
                back_translate=lambda targets: calls.append("bt") or ["" for _ in targets],
                judge=lambda pairs: calls.append("judge") or [{"score": 4.0} for _ in pairs],
                sample_identity=identity,
            )

        first = evaluate({"style": "brief", "inference": "a"})
        self.assertEqual(calls, ["bt", "judge"])

        second = evaluate({"style": "brief", "inference": "a"})
        self.assertEqual(calls, ["bt", "judge"], "an unchanged sampling must reuse the results")
        self.assertEqual(second.back_translation, first.back_translation)
        self.assertEqual(second.judge_scores, first.judge_scores)

        # Changing what binds the cache recomputes instead of hiding the change.
        evaluate({"style": "rewritten", "inference": "a"})
        self.assertEqual(calls, ["bt", "judge", "bt", "judge"])


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


class AutonomyTierTests(unittest.TestCase):
    def test_speed_tier_only_blocks_on_l0(self):
        effective = apply_autonomy_tier({"bt_score_min": 0.45, "judge_score_min": 3.5}, "speed")
        self.assertTrue(effective["block_on_l0_only"])
        gate = build_machine_gate(
            l0={
                "empty_target_count": 0,
                "open_conflict_count": 0,
                "residual_finding_count": 0,
                "open_issue_count": 0,
            },
            back_translation=[{"score": 0.1}],
            judge_scores=[{"score": 1.0}],
            block_on_l0_only=effective["block_on_l0_only"],
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["blocking"])

    def test_precise_tier_tightens_thresholds(self):
        effective = apply_autonomy_tier(
            {"bt_score_min": 0.45, "judge_score_min": 3.5, "risk_sample_ratio": 0.1},
            "precise",
        )
        self.assertEqual(effective["bt_score_min"], 0.6)
        self.assertEqual(effective["judge_score_min"], 4.0)
        self.assertEqual(effective["l2_min_consistency"], 1.0)
        self.assertAlmostEqual(effective["risk_sample_ratio"], 0.2)


class QualityPassNoteMappingTests(unittest.TestCase):
    def test_selfcheck_and_low_back_translation_become_issues(self):
        quality = {
            "self_revision_notes": [{"chapter": 0, "index": 1, "suggested": "改好的句子"}],
            "chapter_selfcheck_findings": [{"chapter": 0, "index": 2, "detail": "语气不稳"}],
            "back_translation_notes": [
                {"chapter": 0, "index": 3, "score": 0.1},
                {"chapter": 0, "index": 4, "score": 0.95},
            ],
            "editorial_notes": ["书级笔记不应映射到段落"],
        }
        issues = quality_pass_notes_to_issues(quality, bt_min=0.45)
        self.assertEqual({issue["index"] for issue in issues}, {1, 2, 3})
        selfcheck = next(issue for issue in issues if issue["index"] == 2)
        self.assertEqual(selfcheck["suggestion"], "语气不稳")

    def test_empty_suggestion_without_detail_is_skipped(self):
        quality = {"chapter_selfcheck_findings": [{"chapter": 0, "index": 2}]}
        self.assertEqual(quality_pass_notes_to_issues(quality), [])

    def test_editorial_findings_map_but_book_notes_stay_analysis_only(self):
        quality = {
            "editorial_notes": ["全书体例笔记没有段落定位"],
            "editorial_findings": [
                {"chapter": 5, "index": 11, "detail": "重复的州字", "suggested": "改后的句子"}
            ],
        }
        issues = quality_pass_notes_to_issues(quality, bt_min=0.45)
        self.assertEqual(len(issues), 1)
        self.assertEqual((issues[0]["chapter"], issues[0]["index"]), (5, 11))
        self.assertEqual(issues[0]["suggestion"], "改后的句子")
        self.assertEqual(issues[0]["type"], "fluency")


class EvaluationTrendTests(unittest.TestCase):
    def test_history_is_appended_and_capped(self):
        from wenyi_core.storage.file import FileStorage

        with tempfile.TemporaryDirectory() as d:
            store = FileStorage(str(Path(d) / "state"))
            try:
                for _ in range(25):
                    from wenyi_core.pipeline.finalization import ReportService

                    history = ReportService._record_evaluation_trend(
                        store, {"machine_gate": {"passed": True, "judge_avg": 4.2}}
                    )
                self.assertEqual(len(history), 20)
                self.assertTrue(history[-1]["passed"])
                self.assertEqual(history[-1]["judge_avg"], 4.2)
            finally:
                store.close()


class AnchorTests(unittest.TestCase):
    def test_distill_and_render(self):
        pairs = [
            ("田中先生が来た。", "田中先生来了。"),
            ("佐藤先生も来た。", "佐藤先生也来了。"),
            ("山田さんが笑った。", "山田先生笑了。"),
        ]
        anchors = distill_decision_anchors(pairs)
        self.assertFalse(anchors["empty"])
        text = render_decision_anchors(anchors)
        self.assertIn("MUST", text)
        self.assertIn("Decision anchors", text)
        self.assertNotIn("人名倾向", text)
        self.assertEqual(render_decision_anchors({"empty": True}), "")

    def test_ordinary_cjk_fragments_are_not_name_anchors(self):
        """Any 2-3 CJK characters used to become a "name preference" MUST.

        A real run distilled 人名倾向「建议以」/「因所用」/「三年三」 from a copyright page and
        injected those MUST lines into the style brief of every later translation call.
        """
        pairs = [
            ("本作品は、縦書き表示での閲覧を推奨いたします。", "本作品建议以竖排方式阅读。"),
            (
                "ご利用になるブラウザにより表示が異なります。",
                "因所用的浏览器不同，显示会有所差异。",
            ),
        ]
        anchors = distill_decision_anchors(pairs)
        self.assertEqual(anchors["must"], [])
        self.assertEqual(anchors["examples"], [])
        self.assertTrue(anchors["empty"])


if __name__ == "__main__":
    unittest.main()

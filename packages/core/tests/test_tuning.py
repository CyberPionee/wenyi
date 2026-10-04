"""Machine tuning: tier policy, batch-budget scaling and score calibration."""

from __future__ import annotations

import os
import tempfile
import unittest

from wenyi_core.config import Config
from wenyi_core.llm.providers.fake import FakeClient
from wenyi_core.pipeline.evaluation import _percentile, apply_autonomy_tier
from wenyi_core.pipeline.orchestrator import Orchestrator
from wenyi_core.pipeline.runtime import PipelineRuntime
from wenyi_core.pipeline.tuning import (
    EVALUATION_TUNED_KEYS,
    FIXED_KEYS,
    POLICY_KEYS,
    TUNED_KEYS,
    describe_tuning,
    evaluation_policy,
    install_run_tuning,
    pipeline_defaults,
    run_tuning,
)

from .fake_llm import routing_handler
from .sample_data import write_sample_txt


def _pipeline(**overrides):
    return {**pipeline_defaults(), **overrides}


def _history(count: int, *, bt_p10: float, judge_p10: float):
    return [{"bt_p10": bt_p10, "judge_p10": judge_p10, "bt_sample_count": 12} for _ in range(count)]


class RunTuningTests(unittest.TestCase):
    def test_standard_tier_changes_nothing_on_shipped_defaults(self):
        """The shipped tier and the shipped values must stay in step, so defaults never move."""
        plan = run_tuning(_pipeline(), segment_max_tokens=1800)
        self.assertEqual(plan.mode, "auto")
        self.assertEqual(plan.tier, "standard")
        self.assertEqual(dict(plan.updates), {})
        self.assertEqual({decision.key for decision in plan.decisions}, set(TUNED_KEYS[:10]))

    def test_speed_tier_halves_sampling_and_narrows_review(self):
        plan = run_tuning(_pipeline(autonomy_tier="speed"), segment_max_tokens=1800)
        self.assertEqual(
            dict(plan.updates),
            {
                "review_scope": "risk",
                "max_auto_redo_rounds": 1,
            },
        )

    def test_precise_tier_enables_dual_judging_and_more_redo_rounds(self):
        plan = run_tuning(_pipeline(autonomy_tier="precise"), segment_max_tokens=1800)
        self.assertEqual(plan.updates["quality_judge_dual"], True)
        self.assertEqual(plan.updates["max_auto_redo_rounds"], 3)
        scope = next(item for item in plan.decisions if item.key == "review_scope")
        self.assertEqual(scope.value, "all")

    def test_off_tier_stops_back_translation_and_review_costs(self):
        plan = run_tuning(_pipeline(autonomy_tier="off"), segment_max_tokens=1800)
        self.assertEqual(plan.updates["risk_back_translation"], False)
        self.assertEqual(plan.updates["review_scope"], "risk")
        self.assertEqual(plan.updates["max_auto_redo_rounds"], 0)

    def test_pinned_value_survives_auto_mode(self):
        plan = run_tuning(
            _pipeline(autonomy_tier="precise", review_scope="risk"), segment_max_tokens=1800
        )
        decision = next(item for item in plan.decisions if item.key == "review_scope")
        self.assertEqual((decision.value, decision.source), ("risk", "pinned"))
        self.assertNotIn("review_scope", plan.updates)

    def test_manual_mode_keeps_every_configured_value(self):
        plan = run_tuning(
            _pipeline(tuning="manual", autonomy_tier="precise"), segment_max_tokens=3600
        )
        self.assertEqual(dict(plan.updates), {})
        self.assertTrue(all(item.source == "pinned" for item in plan.decisions))

    def test_glossary_budgets_scale_with_the_batch_budget(self):
        plan = run_tuning(_pipeline(), segment_max_tokens=3600)
        self.assertEqual(plan.updates["glossary_extract_budget_chars"], 8000)
        self.assertEqual(plan.updates["glossary_note_chars"], 240)
        self.assertEqual(plan.updates["glossary_extract_core_max"], 24)
        self.assertEqual(plan.updates["glossary_extract_recent_max"], 40)
        self.assertEqual(plan.updates["glossary_extract_min_terms"], 10)
        scaled = {"glossary_extract_budget_chars", "glossary_note_chars"}
        self.assertTrue(
            all(item.source == "budget" for item in plan.decisions if item.key in scaled)
        )

    def test_install_returns_a_validated_config_without_mutating_the_input(self):
        config = Config.from_dict({"pipeline": {"autonomy_tier": "speed"}})
        plan = run_tuning(config.pipeline.model_dump(), segment_max_tokens=1800)
        tuned = install_run_tuning(config, plan)
        self.assertEqual(tuned.pipeline.review_scope, "risk")
        self.assertEqual(tuned.pipeline.max_auto_redo_rounds, 1)
        self.assertEqual(config.pipeline.review_scope, "all")


class EvaluationPolicyTests(unittest.TestCase):
    def test_standard_tier_keeps_configured_ratios_and_thresholds(self):
        policy = evaluation_policy(_pipeline(), tier="standard")
        self.assertEqual(policy["risk_sample_ratio"], 0.08)
        self.assertEqual(policy["judge_sample_ratio"], 0.05)
        self.assertEqual(policy["bt_score_min"], 0.45)
        self.assertEqual(policy["judge_score_min"], 3.5)
        self.assertFalse(policy["block_on_l0_only"])

    def test_precise_tier_tightens_without_history(self):
        policy = evaluation_policy(_pipeline(), tier="precise")
        self.assertAlmostEqual(policy["risk_sample_ratio"], 0.16)
        self.assertAlmostEqual(policy["judge_sample_ratio"], 0.10)
        self.assertEqual(policy["bt_score_min"], 0.6)
        self.assertEqual(policy["judge_score_min"], 4.0)

    def test_off_tier_samples_nothing_and_only_l0_blocks(self):
        policy = evaluation_policy(_pipeline(), tier="off")
        self.assertEqual(policy["risk_sample_ratio"], 0.0)
        self.assertEqual(policy["judge_sample_ratio"], 0.0)
        self.assertTrue(policy["block_on_l0_only"])

    def test_history_calibrates_between_floor_and_cap(self):
        policy = evaluation_policy(
            _pipeline(), tier="standard", history=_history(4, bt_p10=0.38, judge_p10=3.2)
        )
        self.assertEqual(policy["bt_score_min"], 0.38)
        self.assertEqual(policy["judge_score_min"], 3.2)
        sources = {item.key: item.source for item in policy["decisions"]}
        self.assertEqual(sources["bt_score_min"], "history")
        self.assertEqual(sources["judge_score_min"], "history")
        self.assertEqual(policy["calibration"], ())

    def test_history_below_the_floor_holds_the_bar_and_asks_the_operator(self):
        policy = evaluation_policy(
            _pipeline(), tier="standard", history=_history(3, bt_p10=0.12, judge_p10=1.8)
        )
        self.assertEqual(policy["bt_score_min"], 0.35)
        self.assertEqual(policy["judge_score_min"], 3.0)
        suggested = {item.key: item.suggested_value for item in policy["calibration"]}
        self.assertEqual(suggested, {"bt_score_min": 0.12, "judge_score_min": 1.8})

    def test_history_never_tightens_past_the_configured_cap(self):
        policy = evaluation_policy(
            _pipeline(), tier="standard", history=_history(3, bt_p10=0.9, judge_p10=4.9)
        )
        self.assertEqual(policy["bt_score_min"], 0.45)
        self.assertEqual(policy["judge_score_min"], 3.5)

    def test_two_runs_are_not_enough_to_move_a_threshold(self):
        policy = evaluation_policy(
            _pipeline(), tier="standard", history=_history(2, bt_p10=0.3, judge_p10=3.0)
        )
        self.assertEqual(policy["bt_score_min"], 0.45)
        self.assertEqual(policy["judge_score_min"], 3.5)

    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(_percentile([], 0.1), None)
        self.assertEqual(_percentile([0.5], 0.1), 0.5)
        self.assertEqual(_percentile([0.1, 0.2, 0.3], 0.1), 0.1)
        self.assertEqual(_percentile([0.2, 0.1, 0.9, 0.8], 0.5), 0.2)

    def test_tier_helper_matches_the_shared_policy(self):
        """apply_autonomy_tier stays the compatibility surface over the shared policy."""
        settings = {"bt_score_min": 0.45, "judge_score_min": 3.5, "risk_sample_ratio": 0.1}
        effective = apply_autonomy_tier(settings, "precise")
        policy = evaluation_policy(settings, tier="precise")
        self.assertEqual(effective["bt_score_min"], policy["bt_score_min"])
        self.assertEqual(effective["risk_sample_ratio"], policy["risk_sample_ratio"])
        self.assertTrue(apply_autonomy_tier({}, "speed")["block_on_l0_only"])


class DescribeTuningTests(unittest.TestCase):
    def test_payload_lists_every_knob_after_its_source(self):
        pipeline = _pipeline(autonomy_tier="precise")
        plan = run_tuning(pipeline, segment_max_tokens=1800)
        policy = evaluation_policy(pipeline, tier="precise")
        payload = describe_tuning(pipeline=pipeline, run_plan=plan, evaluation=policy)

        self.assertEqual(payload["mode"], "auto")
        self.assertEqual(payload["tier"], "precise")
        keys = [item["key"] for item in payload["items"]]
        self.assertEqual(len(keys), len(POLICY_KEYS) + len(TUNED_KEYS) + len(FIXED_KEYS))
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), set(POLICY_KEYS + TUNED_KEYS + FIXED_KEYS))
        sources = {item["source"] for item in payload["items"]}
        self.assertTrue(sources <= {"pinned", "tier", "budget", "history", "default"})
        # The operator's dial leads the list; derived knobs follow.
        self.assertEqual(payload["items"][0]["key"], "autonomy_tier")
        self.assertEqual(payload["items"][0]["source"], "pinned")
        self.assertIn("tier", sources)

    def test_pinned_knobs_are_listed_first(self):
        pipeline = _pipeline(bt_score_min=0.8)
        plan = run_tuning(pipeline, segment_max_tokens=1800)
        policy = evaluation_policy(pipeline, tier="standard")
        payload = describe_tuning(pipeline=pipeline, run_plan=plan, evaluation=policy)
        self.assertEqual(payload["items"][0]["key"], "bt_score_min")
        self.assertEqual(payload["items"][0]["source"], "pinned")


class RuntimeInstallTests(unittest.TestCase):
    def test_runtime_installs_derived_values_before_building_agents(self):
        runtime = PipelineRuntime(Config.from_dict({"pipeline": {"autonomy_tier": "precise"}}))
        self.assertEqual(runtime.config.pipeline.quality_judge_dual, True)
        self.assertEqual(runtime.config.pipeline.max_auto_redo_rounds, 3)
        self.assertEqual(runtime.run_tuning.tier, "precise")
        self.assertEqual(runtime.translator.config.pipeline.review_scope, "all")

    def test_evaluation_tuned_keys_stay_out_of_the_installed_section(self):
        """Ratios and thresholds are resolved at report time, so the run config keeps them raw."""
        runtime = PipelineRuntime(Config.from_dict({"pipeline": {"autonomy_tier": "precise"}}))
        self.assertEqual(runtime.config.pipeline.risk_sample_ratio, 0.08)
        self.assertEqual(runtime.config.pipeline.bt_score_min, 0.45)
        self.assertEqual(
            set(EVALUATION_TUNED_KEYS) & set(runtime.run_tuning.updates),
            set(),
        )


class ReportTuningTests(unittest.TestCase):
    """The report must carry the effective values, so the Web can show what the run chose."""

    @staticmethod
    def _config(directory: str, tier: str) -> Config:
        return Config.from_dict(
            {
                "language": {"source": "ja", "target": "zh"},
                "llm": {
                    "preset": "fake",
                    "models": {"default_strong": {"provider": "default", "model": "fake-strong"}},
                },
                "pipeline": {
                    "book_understanding": False,
                    "review": False,
                    "polish": False,
                    "autonomy_tier": tier,
                },
                "paths": {"state_dir": os.path.join(directory, "state")},
            }
        )

    def test_report_payload_explains_every_effective_value(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "novel.txt")
            write_sample_txt(source)
            config = self._config(directory, "speed")
            orchestrator = Orchestrator(config, client=FakeClient(handler=routing_handler))
            store = orchestrator.run_steps(source, {"translate"})["store"]
            orchestrator.run_report(source)

            payload = store.read_artifact("report.json")
            self.assertIsNotNone(payload)
            assert payload is not None
            tuning = payload["evaluation"]["tuning"]
            self.assertEqual(tuning["mode"], "auto")
            self.assertEqual(tuning["tier"], "speed")
            items = {item["key"]: item for item in tuning["items"]}
            self.assertEqual(items["review_scope"]["value"], "risk")
            self.assertEqual(items["review_scope"]["source"], "tier")
            self.assertEqual(items["max_auto_redo_rounds"]["value"], 1)
            self.assertEqual(items["l2_min_consistency"]["source"], "default")
            self.assertEqual(items["autonomy_tier"]["value"], "speed")

    def test_manual_mode_reports_every_knob_as_the_operators(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "novel.txt")
            write_sample_txt(source)
            config = self._config(directory, "precise")
            config = config.model_copy(
                update={
                    "pipeline": config.pipeline.model_copy(
                        update={"tuning": "manual", "review_scope": "risk"}
                    )
                }
            )
            orchestrator = Orchestrator(config, client=FakeClient(handler=routing_handler))
            store = orchestrator.run_steps(source, {"translate"})["store"]
            orchestrator.run_report(source)

            payload = store.read_artifact("report.json")
            assert payload is not None
            tuning = payload["evaluation"]["tuning"]
            self.assertEqual(tuning["mode"], "manual")
            items = {item["key"]: item for item in tuning["items"]}
            self.assertEqual(items["review_scope"]["value"], "risk")
            self.assertEqual(items["review_scope"]["source"], "pinned")
            # Manual mode keeps the run knobs, yet the tier still tightens the accept thresholds.
            self.assertEqual(items["quality_judge_dual"]["value"], False)
            self.assertEqual(items["bt_score_min"]["value"], 0.6)
            self.assertEqual(items["bt_score_min"]["source"], "tier")


if __name__ == "__main__":
    unittest.main()

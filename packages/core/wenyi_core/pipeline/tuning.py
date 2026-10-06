"""Machine tuning of pipeline knobs: tier policy, batch-budget scaling and score calibration.

A run needs very few human numbers. ``autonomy_tier`` is the quality/cost dial, while
``evaluation_enabled`` and ``auto_qa_strict`` are policies. Everything else in the evaluation
and glossary budget families is derived here:

- **run tuning** resolves before the run starts from the tier and the batch token budget, so
  review coverage, redo budget, dual judging and the glossary prompt budgets never need a
  manual value;
- **evaluation policy** resolves at report time, where the configured ratios and thresholds are
  scaled by the tier and, once enough runs are recorded, calibrated against the observed score
  distribution.

Calibration never loosens the bar below the tier floor and never tightens it above the value the
operator configured: the configured number stays the cap, the tier floor is the policy limit, and
the recorded distribution may only move the bar between them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from ..config import Config, PipelineConfig

AUTONOMY_TIERS = ("off", "speed", "standard", "precise")

# The knobs a human owns: the dial plus the policies it does not decide.
POLICY_KEYS: tuple[str, ...] = (
    "autonomy_tier",
    "evaluation_enabled",
    "auto_qa_strict",
    "quality_passes",
)

# Resolved before the run starts, from the tier and the batch token budget.
RUN_TUNED_KEYS: tuple[str, ...] = (
    "review_scope",
    "risk_back_translation",
    "max_auto_redo_rounds",
    "quality_judge_dual",
    "glossary_extract_inject",
    "glossary_extract_budget_chars",
    "glossary_extract_core_max",
    "glossary_extract_recent_max",
    "glossary_extract_min_terms",
    "glossary_note_chars",
)

# Resolved at report time, where recorded score distributions are available.
EVALUATION_TUNED_KEYS: tuple[str, ...] = (
    "risk_sample_ratio",
    "judge_sample_ratio",
    "bt_score_min",
    "judge_score_min",
)

TUNED_KEYS: tuple[str, ...] = RUN_TUNED_KEYS + EVALUATION_TUNED_KEYS

# The post-translation passes pipeline.quality_passes decides from the tier. Their individual
# switches only apply in that knob's "manual" mode, because five independent booleans left the
# dial unable to offset them: each pass costs about one model call per chapter.
QUALITY_PASS_KEYS: tuple[str, ...] = (
    "self_revision",
    "editorial_pass",
    "final_polish",
    "chapter_selfcheck",
    "back_translation",
)

# Definitions and cost gates: documented, reported, but not tuned.
FIXED_KEYS: tuple[str, ...] = (
    "l2_min_consistency",
    "glossary_always_types",
    "glossary_always_min_occurrences",
    "quality_judge",
    "decision_anchors",
)

SOURCE_PINNED = "pinned"
SOURCE_TIER = "tier"
SOURCE_BUDGET = "budget"
SOURCE_HISTORY = "history"
SOURCE_DEFAULT = "default"

# Glossary prompt budgets scale with the batch budget; these divisors reproduce the shipped
# defaults exactly at the shipped batch size, so tuning changes nothing unless the batch does.
_GLOSSARY_BATCH_REFERENCE = 1800
_GLOSSARY_SCALES: Mapping[str, float] = {
    "glossary_extract_budget_chars": 4000 / _GLOSSARY_BATCH_REFERENCE,
    "glossary_extract_core_max": 12 / _GLOSSARY_BATCH_REFERENCE,
    "glossary_extract_recent_max": 20 / _GLOSSARY_BATCH_REFERENCE,
    "glossary_extract_min_terms": 5 / _GLOSSARY_BATCH_REFERENCE,
    "glossary_note_chars": 120 / _GLOSSARY_BATCH_REFERENCE,
}

# Configuration key to the tier policy entry that decides it.
_TIER_KEY_MAP: Mapping[str, str] = {
    "review_scope": "review_scope",
    "risk_back_translation": "back_translation",
    "max_auto_redo_rounds": "redo_rounds",
    "quality_judge_dual": "judge_dual",
}

# Modes whose adaptive behavior lives in the selector rather than in a number.
_MODE_DEFAULTS: Mapping[str, str] = {"glossary_extract_inject": "smart"}

# Policy floors. Calibration may relax a threshold down to its floor, never below it, and the
# floor is set by the tier the operator chose rather than by the book's own scores.
_THRESHOLD_FLOORS: Mapping[str, tuple[float, float]] = {
    "off": (0.30, 2.5),
    "speed": (0.35, 3.0),
    "standard": (0.35, 3.0),
    "precise": (0.45, 3.5),
}
# Tier multipliers and switches applied to the configured values.
_TIER_POLICY: Mapping[str, Mapping[str, Any]] = {
    "off": {
        "risk_multiplier": 0.0,
        "judge_multiplier": 0.0,
        "back_translation": False,
        "redo_rounds": 0,
        "judge_dual": False,
        "review_scope": "risk",
        "block_on_l0_only": True,
        "bt_floor_min": 0.0,
        "judge_floor_min": 0.0,
    },
    "speed": {
        "risk_multiplier": 0.5,
        "judge_multiplier": 0.6,
        "back_translation": True,
        "redo_rounds": 1,
        "judge_dual": False,
        "review_scope": "risk",
        "block_on_l0_only": True,
        "bt_floor_min": 0.0,
        "judge_floor_min": 0.0,
    },
    "standard": {
        "risk_multiplier": 1.0,
        "judge_multiplier": 1.0,
        "back_translation": True,
        "redo_rounds": 2,
        "judge_dual": False,
        "review_scope": "all",
        "block_on_l0_only": False,
        "bt_floor_min": 0.0,
        "judge_floor_min": 0.0,
    },
    "precise": {
        "risk_multiplier": 2.0,
        "judge_multiplier": 2.0,
        "back_translation": True,
        "redo_rounds": 3,
        "judge_dual": True,
        "review_scope": "all",
        "block_on_l0_only": False,
        "bt_floor_min": 0.6,
        "judge_floor_min": 4.0,
    },
}

# Coverage of a tier's quality passes: "risk" runs them only on chapters whose deterministic
# scans found something, "all" runs them on every translated chapter.
QUALITY_PASS_COVERAGE_ALL = "all"
QUALITY_PASS_COVERAGE_RISK = "risk"

# Which passes each tier turns on, and how much of the book they cover. self_revision rewrites
# whole chapters like final polish, so only precise runs both; back_translation re-measures the
# fidelity L1 already samples, so only precise repeats it there; editorial_pass is a single
# whole-book call, so it rides along from standard up.
_AUTO_QUALITY_PASSES: Mapping[str, tuple[frozenset[str], str]] = {
    "off": (frozenset(), QUALITY_PASS_COVERAGE_ALL),
    "speed": (frozenset({"chapter_selfcheck"}), QUALITY_PASS_COVERAGE_RISK),
    "standard": (
        frozenset({"chapter_selfcheck", "final_polish", "editorial_pass"}),
        QUALITY_PASS_COVERAGE_RISK,
    ),
    "precise": (frozenset(QUALITY_PASS_KEYS), QUALITY_PASS_COVERAGE_ALL),
}


def quality_pass_plan(
    mode: str, *, tier: str, configured: Mapping[str, object]
) -> tuple[frozenset[str], str]:
    """Return the passes to run and their coverage for the configured ``quality_passes`` mode.

    ``auto`` derives both from the autonomy tier, ``full`` runs everything on every chapter,
    ``off`` runs nothing, and ``manual`` keeps the individual switches exactly as written.
    """
    if mode == "off":
        return frozenset(), QUALITY_PASS_COVERAGE_ALL
    if mode == "full":
        return frozenset(QUALITY_PASS_KEYS), QUALITY_PASS_COVERAGE_ALL
    if mode == "manual":
        return (
            frozenset(key for key in QUALITY_PASS_KEYS if configured.get(key)),
            QUALITY_PASS_COVERAGE_ALL,
        )
    return _AUTO_QUALITY_PASSES.get(tier, _AUTO_QUALITY_PASSES["standard"])


# Runs needed before recorded scores may move a threshold at all.
_MIN_HISTORY = 3


@dataclass(frozen=True)
class TuningDecision:
    """One knob's effective value and where it came from."""

    key: str
    value: Any
    source: str
    note: str = ""


@dataclass(frozen=True)
class TuningCalibration:
    """A threshold the recorded distribution argues against, for an operator to confirm."""

    key: str
    suggested_value: float
    reason: str


@dataclass(frozen=True)
class TuningPlan:
    """Run-level decisions plus the values to install into the run configuration."""

    mode: str
    tier: str
    decisions: tuple[TuningDecision, ...]
    updates: Mapping[str, Any] = field(default_factory=dict)

    def derived_keys(self) -> tuple[str, ...]:
        return tuple(
            decision.key
            for decision in self.decisions
            if decision.source in {SOURCE_TIER, SOURCE_BUDGET, SOURCE_HISTORY}
        )


def pipeline_defaults() -> dict[str, Any]:
    """Shipped value for every pipeline key, used to detect pinned settings."""
    return PipelineConfig().model_dump()


def normalize_tier(tier: str) -> str:
    """Unknown tier names fall back to standard, matching the evaluation gate."""
    return tier if tier in AUTONOMY_TIERS else "standard"


def _is_pinned(pipeline: Mapping[str, Any], key: str, defaults: Mapping[str, Any]) -> bool:
    """A knob the operator moved off its shipped default is theirs to keep.

    The Web writes every key into the stored document, so presence cannot tell a deliberate
    setting from a written-back default; a value that differs from the shipped default can.
    """
    return key in pipeline and pipeline[key] != defaults[key]


def _glossary_value(key: str, segment_max_tokens: int) -> int:
    scaled = int(round(segment_max_tokens * _GLOSSARY_SCALES[key]))
    return max(0, scaled)


def run_tuning(pipeline: Mapping[str, Any], *, segment_max_tokens: int) -> TuningPlan:
    """Resolve the knobs a run needs before it starts, without touching recorded history."""
    defaults = pipeline_defaults()
    mode = str(pipeline.get("tuning") or defaults.get("tuning") or "auto")
    tier = normalize_tier(str(pipeline.get("autonomy_tier", defaults["autonomy_tier"])))
    policy = _TIER_POLICY[tier]
    decisions: list[TuningDecision] = []
    updates: dict[str, Any] = {}

    for key in RUN_TUNED_KEYS:
        current = pipeline.get(key, defaults[key])
        if mode == "manual":
            decisions.append(TuningDecision(key, current, SOURCE_PINNED, "manual tuning"))
            continue
        if _is_pinned(pipeline, key, defaults):
            decisions.append(
                TuningDecision(key, current, SOURCE_PINNED, "set apart from the shipped default")
            )
            continue
        if key in _MODE_DEFAULTS:
            value = _MODE_DEFAULTS[key]
            decisions.append(
                TuningDecision(key, value, SOURCE_DEFAULT, "adaptive mode; pin to override")
            )
        elif key in _GLOSSARY_SCALES:
            value = _glossary_value(key, segment_max_tokens)
            if value == defaults[key] or segment_max_tokens == _GLOSSARY_BATCH_REFERENCE:
                decisions.append(
                    TuningDecision(key, value, SOURCE_DEFAULT, "shipped default applies")
                )
            else:
                decisions.append(
                    TuningDecision(
                        key,
                        value,
                        SOURCE_BUDGET,
                        f"scaled from a {segment_max_tokens}-token batch budget",
                    )
                )
        else:
            value = policy[_TIER_KEY_MAP[key]]
            decisions.append(TuningDecision(key, value, SOURCE_TIER, f"{tier} tier"))
        if value != current:
            updates[key] = value

    return TuningPlan(mode=mode, tier=tier, decisions=tuple(decisions), updates=updates)


def _percentile_median(entries: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """Median of one recorded per-run percentile, or None when too few runs carry it."""
    values = [
        float(entry[key])
        for entry in entries
        if isinstance(entry.get(key), (int, float)) and not isinstance(entry.get(key), bool)
    ]
    if len(values) < _MIN_HISTORY:
        return None
    return float(median(values))


def evaluation_policy(
    settings: Mapping[str, Any],
    *,
    tier: str,
    history: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Resolve sampling and thresholds, calibrating against recorded scores when available.

    ``settings`` carries the configured values. The tier scales them; the configured threshold
    stays the cap; the recorded p10 may relax a threshold down to the tier floor. Everything is
    reported in ``decisions`` and ``calibration`` so a run can explain its own bar.
    """
    defaults = pipeline_defaults()
    tier = normalize_tier(tier)
    policy = _TIER_POLICY[tier]
    configured_risk = float(settings.get("risk_sample_ratio", defaults["risk_sample_ratio"]))
    configured_judge = float(settings.get("judge_sample_ratio", defaults["judge_sample_ratio"]))
    risk_ratio = _clamp_ratio(configured_risk * float(policy["risk_multiplier"]))
    judge_ratio = _clamp_ratio(configured_judge * float(policy["judge_multiplier"]))
    configured_bt = float(settings.get("bt_score_min", defaults["bt_score_min"]))
    configured_judge_min = float(settings.get("judge_score_min", defaults["judge_score_min"]))
    # The tier may tighten the operator's number, never relax it: the configured value is the cap.
    bt_cap = max(configured_bt, float(policy["bt_floor_min"]))
    judge_cap = max(configured_judge_min, float(policy["judge_floor_min"]))
    bt_floor = min(_THRESHOLD_FLOORS[tier][0], bt_cap)
    judge_floor = min(_THRESHOLD_FLOORS[tier][1], judge_cap)

    decisions: list[TuningDecision] = [
        _ratio_decision("risk_sample_ratio", configured_risk, risk_ratio, policy, tier, defaults),
        _ratio_decision(
            "judge_sample_ratio", configured_judge, judge_ratio, policy, tier, defaults
        ),
    ]
    calibration: list[TuningCalibration] = []

    bt_observed = _percentile_median(history, "bt_p10")
    judge_observed = _percentile_median(history, "judge_p10")
    bt_min, bt_decision = _calibrate(
        "bt_score_min",
        bt_observed,
        floor=bt_floor,
        cap=bt_cap,
        configured=configured_bt,
        default=defaults["bt_score_min"],
        calibration=calibration,
    )
    judge_min, judge_decision = _calibrate(
        "judge_score_min",
        judge_observed,
        floor=judge_floor,
        cap=judge_cap,
        configured=configured_judge_min,
        default=defaults["judge_score_min"],
        calibration=calibration,
    )
    decisions.extend((bt_decision, judge_decision))

    l2_min = 1.0 if tier == "precise" else float(settings.get("l2_min_consistency", 1.0))
    return {
        "tier": tier,
        "risk_sample_ratio": risk_ratio,
        "judge_sample_ratio": judge_ratio,
        "bt_score_min": bt_min,
        "judge_score_min": judge_min,
        "l2_min_consistency": l2_min,
        "block_on_l0_only": bool(policy["block_on_l0_only"]),
        "decisions": tuple(decisions),
        "calibration": tuple(calibration),
    }


def _clamp_ratio(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _ratio_decision(
    key: str,
    configured: float,
    value: float,
    policy: Mapping[str, Any],
    tier: str,
    defaults: Mapping[str, Any],
) -> TuningDecision:
    """Label a sampling ratio: scaled by the tier, the operator's number, or the default."""
    multiplier = float(
        policy["risk_multiplier" if key == "risk_sample_ratio" else "judge_multiplier"]
    )
    if multiplier != 1.0:
        source = SOURCE_TIER
        note = f"{tier} tier scales the configured ratio by {multiplier:g}"
        if value == 0.0:
            note = f"{tier} tier samples nothing here"
    elif configured != defaults[key]:
        source, note = SOURCE_PINNED, "operator value"
    else:
        source, note = SOURCE_DEFAULT, "shipped default applies"
    return TuningDecision(key, round(value, 4), source, note)


def _static_source(
    configured: float, default: float, *, tightened: float | None = None
) -> tuple[str, str]:
    """Label a threshold that recorded scores did not move."""
    if tightened is not None and tightened != configured:
        return SOURCE_TIER, "tier floor applies"
    if configured != default:
        return SOURCE_PINNED, "operator value"
    return SOURCE_DEFAULT, "shipped default applies"


def _calibrate(
    key: str,
    observed: float | None,
    *,
    floor: float,
    cap: float,
    configured: float,
    default: float,
    calibration: list[TuningCalibration],
) -> tuple[float, TuningDecision]:
    """Move one threshold within [floor, cap] using the recorded p10."""
    if observed is None:
        source, label = _static_source(configured, default, tightened=cap)
        return round(cap, 4), TuningDecision(key, round(cap, 4), source, label)
    if observed < floor:
        # The book's own distribution sits below the policy floor: report it instead of
        # silently lowering the bar, because the floor is the operator's quality policy.
        calibration.append(
            TuningCalibration(
                key=key,
                suggested_value=round(observed, 4),
                reason=(
                    f"recorded p10 median {observed:.3f} is below the {floor:.2f} floor for this "
                    f"tier; pin a lower value only if the language pair scores low by nature"
                ),
            )
        )
        return round(floor, 4), TuningDecision(
            key, round(floor, 4), SOURCE_HISTORY, f"held at the tier floor {floor:.2f}"
        )
    value = min(cap, max(floor, observed))
    if value == round(cap, 4):
        source, label = _static_source(configured, default, tightened=cap)
        return round(cap, 4), TuningDecision(
            key,
            round(cap, 4),
            source,
            f"{label}; recorded p10 median {observed:.3f} clears it",
        )
    return round(value, 4), TuningDecision(
        key,
        round(value, 4),
        SOURCE_HISTORY,
        f"calibrated to the recorded p10 median {observed:.3f}",
    )


def describe_tuning(
    *,
    pipeline: Mapping[str, Any],
    run_plan: TuningPlan,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the report payload: every knob's effective value and where it came from."""
    defaults = pipeline_defaults()
    by_source_order = {
        SOURCE_PINNED: 0,
        SOURCE_HISTORY: 1,
        SOURCE_TIER: 2,
        SOURCE_BUDGET: 3,
        SOURCE_DEFAULT: 4,
    }
    items: list[TuningDecision] = list(run_plan.decisions)
    items.extend(evaluation["decisions"])
    for key in POLICY_KEYS:
        value = pipeline.get(key, defaults[key])
        source = SOURCE_PINNED if _is_pinned(pipeline, key, defaults) else SOURCE_DEFAULT
        items.append(TuningDecision(key, value, source, "operator policy"))
    # The post-translation passes follow pipeline.quality_passes, so they are reported from that
    # plan instead of as fixed defaults.
    pass_mode = str(pipeline.get("quality_passes", defaults["quality_passes"]))
    passes, coverage = quality_pass_plan(pass_mode, tier=evaluation["tier"], configured=pipeline)
    for key in QUALITY_PASS_KEYS:
        items.append(
            TuningDecision(
                key,
                key in passes,
                SOURCE_PINNED if pass_mode == "manual" else SOURCE_TIER,
                f"quality_passes={pass_mode} ({coverage})",
            )
        )
    for key in FIXED_KEYS:
        items.append(
            TuningDecision(key, pipeline.get(key, defaults[key]), SOURCE_DEFAULT, "fixed default")
        )
    ordered = sorted(items, key=lambda item: (by_source_order.get(item.source, 5), item.key))
    return {
        "mode": run_plan.mode,
        "tier": evaluation["tier"],
        "items": [
            {
                "key": item.key,
                "value": item.value,
                "source": item.source,
                "note": item.note,
            }
            for item in ordered
        ],
        "calibration": [
            {
                "key": entry.key,
                "suggested_value": entry.suggested_value,
                "reason": entry.reason,
            }
            for entry in evaluation.get("calibration", ())
        ],
    }


def install_run_tuning(config: Config, plan: TuningPlan) -> Config:
    """Return a config whose pipeline section carries the derived run-level values."""
    if not plan.updates:
        return config
    pipeline = PipelineConfig.model_validate({**config.pipeline.model_dump(), **plan.updates})
    return config.model_copy(update={"pipeline": pipeline})


def history_entries(history: Iterable[Any]) -> tuple[Mapping[str, Any], ...]:
    """Keep only well-formed recorded run summaries."""
    return tuple(entry for entry in history if isinstance(entry, Mapping))

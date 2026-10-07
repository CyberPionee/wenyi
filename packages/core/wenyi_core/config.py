"""Load config.yaml with typed defaults using Pydantic v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from .i18n.languages import require_language
from .i18n.policy.models import Phase, PolicyContext, PolicyPlan, content_hash
from .i18n.policy.resolver import resolve_policy
from .llm.configuration import LLMConfig


def parse_config_yaml(text: str) -> Any:
    """Parse CLI and Web configuration with the same safe scalar semantics."""
    return yaml.safe_load(text)


_DEFAULT_CONFIG_YAML = """\
# Wenyi configuration (experimental multilingual fiction translation)
# Configure model providers, workflow stages and output here; no code changes are needed.

language:
  source: auto # auto detects the source language; use an explicit code such as ja / en / ko / ru / de / vi to override
  target: zh # Target: zh / zh-Hant / en / ja / ko / fr / de / es / it / pt / ru / vi; run languages for the full list

# ── LLM ──────────────────────────────────────────────────────────────────
llm:
  preset: deepseek # All tiers: deepseek-flash, thinking enabled, reasoning_effort high
  # Add providers, models and routes to override individual operations.
  # Inspect effective settings with: wenyi models list

# ── Segmentation ─────────────────────────────────────────────────────────────────
segment:
  # Target batch size in tokens (tiktoken cl100k_base).
  max_tokens_per_batch: 1800
  # Split longer paragraphs at sentence boundaries by the same token budget; merge on export.
  max_tokens_per_segment: 1200

# ── Pipeline options (quality and cost)───────────────────────────────────────────
pipeline:
  review: true # Run final review after whole-book translation; disable with --no-review
  align_retry_limit: 2
  polish: true # Polish the full translation with the strong tier; enabled by default and adds substantial cost
  translation_mode: standard # standard | best_of_three; best_of_three requires polish
  rolling_context_segments: 6 # Number of recent source-target pairs supplied as context
  rolling_context_with_source: true # Include source lines with each recent context pair
  book_understanding: true # Prescan the source for a whole-book synopsis and chapter digests used during translation
  prescan_concurrency: 4 # Concurrent chapter-digest workers; chapters are independent, 1 runs serially
  annotation_alignment: true # Align EPUB annotation links per paragraph; if disabled, target links fall back to paragraph ends
  annotation_alignment_concurrency: 4 # Maximum concurrent alignment requests when a paragraph has multiple annotations
  review_concurrency: 4 # Concurrent review blocks over a read-only translation/glossary snapshot; 1 runs serially
  review_output_retries: 2 # Additional retries for malformed single-paragraph review output; 2 allows 3 attempts total
  review_agent_loop: true # Use evidence-based verification after the initial review identifies candidates
  review_agent_max_evidence_rounds: 2 # At most two rounds of selective evidence requests before a final decision
  review_conflict_arbitration: true # Arbitrate contradictory consistency proposals after all review blocks finish
  glossary_conflict_arbitration: true # Settle terminology conflicts from book context once translation finishes, before review
  review_fix_loop: true # Revise an in-memory shadow translation and review it blindly; this loop does not publish changes
  review_fix_max_rounds: 2 # At most two replacement rounds; consecutive clean confirmations also affect total review rounds
  review_clean_confirmations: 2 # Require two consecutive clean rounds to accept the shadow translation
  review_autofix: true # Publish review revisions to formal chapters; use --no-autofix for recommendations only
  review_scope: "all" # all | risk — risk reviews only chapters containing risk segments
  glossary_scope: chapter # chapter=terms relevant to this chapter; full=entire glossary
  glossary_always_types: [person] # Term types kept in chapter-filtered prompts even when absent from the chapter
  glossary_always_min_occurrences: 3 # Minimum book-wide source/alias occurrences for always-on entities
  glossary_note_chars: 120 # Maximum glossary note characters rendered into prompts
  glossary_extract_inject: "smart" # smart | all | hit_only — how existing terms enter extraction prompts
  glossary_extract_budget_chars: 4000 # Character budget for extraction glossary injection
  glossary_extract_core_max: 12
  glossary_extract_recent_max: 20
  glossary_extract_min_terms: 5
  auto_qa_strict: false # When true, block export while auto_qa reports unresolved residuals
  tuning: "auto" # auto: derive the tunable knobs from the tier, batch budget and recorded scores; manual: keep the values below
  quality_passes: manual # Post-translation passes. manual: the switches below (the default, so an existing project keeps exactly the passes it configured); auto: derive them from the tier and risk-gate chapters; full: every pass on every chapter; off: none
  self_revision: false # Optional C-batch draft revision notes (analysis/events only; "manual" mode only)
  editorial_pass: false # Optional whole-book editorial notes (analysis/events only; "manual" mode only)
  final_polish: false # Optional final polish candidates (analysis/events only; "manual" mode only)
  chapter_selfcheck: false # Optional per-chapter LLM self-check notes (analysis/events only; "manual" mode only)
  back_translation: false # Optional back-translation QA notes (analysis/events only; "manual" mode only)
  autonomy_tier: "standard" # off | speed | standard | precise
  evaluation_enabled: true # L0-L3 machine gate for autonomous acceptance
  risk_back_translation: true # L1 selective back-translation on risk/sampled segments
  risk_sample_ratio: 0.08 # Per-chapter sample ratio for risk evaluation (0-1)
  quality_judge: true # L3 LLM fluency/style scoring on sampled segments
  quality_judge_dual: false # Average two independent judge passes to reduce rater noise
  judge_sample_ratio: 0.05
  judge_score_min: 3.5 # L3 pass threshold (1-5)
  l2_min_consistency: 1.0 # L2 glossary consistency threshold; 1.0 means any drift fails
  bt_score_min: 0.45 # L1 back-translation similarity threshold (0-1)
  max_auto_redo_rounds: 2 # Automatic local redo rounds when the machine gate fails
  decision_anchors: "off" # off | auto | risk — optional target-side decision anchors
  # PDF backend: mineru (default, supports scans) | babeldoc (optional, preserves layout via external AGPL HTTP bridge)
  pdf_backend: mineru
  babeldoc_bridge_url: http://127.0.0.1:8765
  # babeldoc_pages: "15"   # Optional page restriction for the bridge (one-based)
  babeldoc_timeout: 600

# ── Honorific strategy (language-specific rules apply where available)────────────────────
honorific:
  # keep_style: preserve relationship and tone; normalize: apply consistent conventions; drop: omit where meaning permits
  strategy: keep_style

# ── Paths ─────────────────────────────────────────────────────────────────
paths:
  state_dir: state # Run state, intermediate chapter files and glossary

# ── Output ───────────────────────────────────────────────────────────────────
output:
  mono: true # Monolingual output (<title>.<target-language>.epub; default target is zh)
  bilingual: false # Bilingual output (<title>.<target-language>-bi.epub)
  bilingual_order: target_first # target_first=translation first; source_first=source first
  bilingual_preserve_source_style: false # true=preserve original source styling; false=render source in muted gray
  about_page: true # Append an About This Translation page
  punctuation_normalize: true # Normalize only exported copies; preserve formal translation state
"""


class SegmentConfig(BaseModel):
    """Source packing budgets measured with tiktoken ``cl100k_base``."""

    model_config = ConfigDict(extra="forbid")

    max_tokens_per_batch: int = 1800
    max_tokens_per_segment: int = 1200


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review: bool = True
    align_retry_limit: int = (
        2  # Retry misaligned batches this many times before falling back to single paragraphs
    )
    polish: bool = (
        True  # Polish the full translation with the strong tier by default; disable to save cost
    )
    translation_mode: Literal["standard", "best_of_three"] = "standard"
    rolling_context_segments: int = 6
    rolling_context_with_source: bool = True
    # Prescan for a synopsis and chapter digests; disable to save prescan cost.
    book_understanding: bool = True
    prescan_concurrency: int = (
        4  # Concurrent chapter-digest workers; chapters are independent, 1 runs serially
    )
    annotation_alignment: bool = (
        True  # Align links after each annotated logical paragraph is finalized
    )
    # For multiple annotations in a logical paragraph, align each with a concurrent request.
    # This limit bounds concurrency and avoids all markers falling back after one bad response.
    annotation_alignment_concurrency: int = 4
    review_concurrency: int = (
        4  # Concurrent review blocks; merge in original order, 1 runs serially
    )
    review_output_retries: int = Field(
        default=2,
        ge=0,
        le=5,
    )  # Additional retries for malformed single-paragraph output
    review_agent_loop: bool = (
        True  # Start the bounded evidence agent loop when initial review finds candidates
    )
    review_agent_max_evidence_rounds: int = Field(
        default=2,
        ge=0,
        le=2,
    )
    review_conflict_arbitration: bool = (
        True  # Arbitrate contradictory consistency proposals after all blocks finish
    )
    glossary_conflict_arbitration: bool = (
        True  # Settle terminology conflicts from book context before review
    )
    review_fix_loop: bool = (
        True  # Revise only the in-memory shadow translation and review it blindly
    )
    review_fix_max_rounds: int = Field(default=2, ge=0, le=4)
    review_clean_confirmations: int = Field(default=2, ge=1, le=2)
    review_autofix: bool = True  # Publish formal translations through a separate stage after review
    review_scope: Literal["all", "risk"] = "all"
    glossary_scope: str = (
        "chapter"  # chapter=terms occurring in this chapter (saves tokens); full=entire glossary
    )
    glossary_always_types: list[str] = Field(
        default_factory=lambda: ["person"],
        description="Term types kept in chapter-filtered prompts even when absent from the chapter",
    )
    glossary_always_min_occurrences: int = Field(default=3, ge=1)
    glossary_note_chars: int = Field(default=120, ge=0)
    glossary_extract_inject: Literal["smart", "all", "hit_only"] = "smart"
    glossary_extract_budget_chars: int = Field(default=4000, ge=0)
    glossary_extract_core_max: int = Field(default=12, ge=0)
    glossary_extract_recent_max: int = Field(default=20, ge=0)
    glossary_extract_min_terms: int = Field(default=5, ge=0)
    auto_qa_strict: bool = False  # Block export while auto_qa residuals remain
    # auto: derive the tunable knobs from the autonomy tier, the batch budget and recorded
    # score distributions. manual: use the configured values as written.
    tuning: Literal["auto", "manual"] = "auto"
    quality_passes: Literal["auto", "full", "manual", "off"] = "manual"
    self_revision: bool = False
    editorial_pass: bool = False
    final_polish: bool = False
    chapter_selfcheck: bool = False
    back_translation: bool = False
    autonomy_tier: Literal["off", "speed", "standard", "precise"] = "standard"
    evaluation_enabled: bool = True
    risk_back_translation: bool = True
    risk_sample_ratio: float = Field(default=0.08, ge=0.0, le=1.0)
    quality_judge: bool = True
    quality_judge_dual: bool = False
    judge_sample_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    judge_score_min: float = Field(default=3.5, ge=1.0, le=5.0)
    l2_min_consistency: float = Field(default=1.0, ge=0.0, le=1.0)
    bt_score_min: float = Field(default=0.45, ge=0.0, le=1.0)
    max_auto_redo_rounds: int = Field(default=2, ge=0, le=5)
    decision_anchors: Literal["off", "auto", "risk"] = "off"
    # PDF: mineru=HTML path for scans (default); babeldoc=external AGPL HTTP bridge (no imports)
    pdf_backend: Literal["mineru", "babeldoc"] = "mineru"
    babeldoc_bridge_url: str = "http://127.0.0.1:8765"
    babeldoc_pages: str | None = None  # For example "15" / "6-8"; None=whole book
    babeldoc_timeout: float = 600.0

    @model_validator(mode="before")
    @classmethod
    def ignore_legacy_precision_concurrency(cls, value: Any) -> Any:
        """Drop the retired initial-draft concurrency key instead of rejecting saved configs."""
        if isinstance(value, dict) and "precision_concurrency" in value:
            return {key: item for key, item in value.items() if key != "precision_concurrency"}
        return value

    @model_validator(mode="after")
    def validate_translation_mode(self) -> PipelineConfig:
        """Three-draft synthesis replaces the standard translate-and-polish path."""
        if self.translation_mode == "best_of_three" and not self.polish:
            raise ValueError("best_of_three translation mode requires pipeline.polish=true")
        return self


class OutputConfig(BaseModel):
    mono: bool = True  # Generate monolingual output
    bilingual: bool = False  # Generate bilingual output
    bilingual_order: Literal["target_first", "source_first"] = (
        "target_first"  # target_first=translation first (default); source_first=source first
    )
    bilingual_preserve_source_style: bool = False
    about_page: bool = True  # Append the project about page
    punctuation_normalize: bool = (
        True  # Normalize export copies only; never write back to chapter target
    )


class Config(BaseModel):
    source_lang: str = "auto"  # auto | ja | en | … (auto uses model detection)
    target_lang: str = "zh"
    _language_plans: dict[str, PolicyPlan] = PrivateAttr(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    honorific_strategy: str = "keep_style"
    state_dir: str = "state"

    @model_validator(mode="after")
    def validate_builtin_language_policy(self) -> Config:
        resolve_policy(
            PolicyContext(
                self.source_lang,
                self.target_lang,
                punctuation_normalize=self.output.punctuation_normalize,
            ),
        )
        return self

    def language_policy(
        self,
        phase: Phase = "translation",
        *,
        path: Literal["book", "srt"] = "book",
        format: str = "",
        backend: str = "native",
        source_identity: str = "",
    ) -> PolicyPlan:
        """Reuse the invocation's frozen semantic plan; exports resolve their snapshot."""
        key = f"{path}:{phase}"
        if phase != "export" and key in self._language_plans:
            return self._language_plans[key]
        context = PolicyContext(
            self.source_lang,
            self.target_lang,
            phase=phase,
            path=path,
            format=format,
            backend=backend,
            punctuation_normalize=self.output.punctuation_normalize,
            honorific_strategy=self.honorific_strategy,
            source_identity=source_identity,
            task_groups=self._policy_tasks(phase, path),
            bilingual=self.output.bilingual,
            order=self.output.bilingual_order,
            preserve_source_style=self.output.bilingual_preserve_source_style,
            about_page=self.output.about_page,
        )
        return resolve_policy(context)

    def _policy_tasks(self, phase: Phase, path: str) -> tuple[str, ...]:
        if path == "srt":
            return ("srt_batch", "srt_single")
        if phase == "analysis":
            return (
                ("analyzer", "chapter_digest", "book_synopsis")
                if self.pipeline.book_understanding
                else ("analyzer",)
            )
        if phase == "translation":
            if self.pipeline.translation_mode == "best_of_three":
                # Precision mode replaces the translator and polisher with three drafts and a
                # synthesis, so those prompts belong to this plan instead.
                groups = ["precision", "title_translator", "glossary_extractor", "glossary_history"]
            else:
                groups = [
                    "translator",
                    "title_translator",
                    "glossary_extractor",
                    "glossary_history",
                ]
                if self.pipeline.polish:
                    groups.append("polisher")
            # The glossary arbiter settles terminology for the translated text and runs in this
            # workflow, so its templates belong to this plan. Hashing them into the review plan
            # instead would change the review phase fingerprint, and a completed review is
            # reused and resumed on that fingerprint: a terminology-prompt edit would silently
            # throw away a finished whole-book review.
            if self.pipeline.glossary_conflict_arbitration:
                groups.append("glossary_arbiter")
            # Optional passes render their prompts from this phase, so their templates and rules
            # belong to its revision: a prompt edit must invalidate results derived from them.
            # The post-translation passes are decided by quality_passes, so their prompt groups
            # follow that plan rather than the individual switches.
            from .pipeline.tuning import quality_pass_plan

            passes, _coverage = quality_pass_plan(
                self.pipeline.quality_passes,
                tier=self.pipeline.autonomy_tier,
                configured=self.pipeline.model_dump(),
            )
            for enabled, group in (
                ("self_revision" in passes, "self_revision"),
                ("editorial_pass" in passes, "editorial_pass"),
                ("final_polish" in passes, "final_polish"),
                ("chapter_selfcheck" in passes, "chapter_selfcheck"),
                (
                    "back_translation" in passes or self.pipeline.risk_back_translation,
                    "back_translation",
                ),
                (self.pipeline.quality_judge, "quality_judge"),
            ):
                if enabled:
                    groups.append(group)
            return tuple(groups)
        if phase == "review":
            groups = ["reviewer"]
            if self.pipeline.review_agent_loop or self.pipeline.review_autofix:
                groups.append("review_agent")
            if self.pipeline.review_agent_loop and self.pipeline.review_conflict_arbitration:
                groups.append("review_arbiter")
            if self.pipeline.review_fix_loop or self.pipeline.review_autofix:
                groups.append("review_fixer")
            return tuple(groups)
        return ()

    def freeze_language_policies(self, source_identity: str = "") -> None:
        """Freeze resources after source detection, before any consuming model call."""
        self._language_plans.clear()
        plans = {
            phase: self.language_policy(phase, source_identity=source_identity)
            for phase in ("analysis", "translation", "review")
        }
        self._language_plans.update({f"book:{phase}": plan for phase, plan in plans.items()})

    def language_document(self) -> dict[str, Any]:
        return {
            "source": self.source_lang,
            "target": self.target_lang,
        }

    def language_policy_revision(self, *, path: Literal["book", "srt"] = "book") -> str:
        """Describe the built-in semantic revision without source-content hashes."""
        if path == "srt":
            return self.language_policy("translation", path=path).fingerprint
        return content_hash(
            {
                phase: self.language_policy(phase).semantic_fingerprint
                for phase in ("analysis", "translation")
            }
        )

    def language_policy_preview(
        self,
        *,
        format: str | None = None,
        backend: str = "native",
        path: Literal["book", "srt"] = "book",
    ) -> dict[str, Any]:
        """Use the execution resolver for diagnostics; auto source remains unresolved."""
        phases = ("translation",) if path == "srt" else ("analysis", "translation", "review")
        result = {phase: self.language_policy(phase, path=path).preview() for phase in phases}
        result["revision"] = {"fingerprint": self.language_policy_revision(path=path)}
        from .llm.operations import configured_operations
        from .llm.routing import resolve_routes

        routes = resolve_routes(self.llm)
        for phase in phases:
            workflow = (
                "srt"
                if path == "srt"
                else "prepare"
                if phase == "analysis"
                else "review"
                if phase == "review"
                else "translate"
            )
            operations = configured_operations(self, workflow)
            if phase == "translation" and path == "book":
                operations = tuple(
                    operation
                    for operation in operations
                    if not operation.startswith(
                        ("language.", "analysis.", "synopsis.", "review.", "autofix.")
                    )
                )
            result[phase]["model_routes"] = [
                routes[operation].describe() for operation in operations
            ]
        if format is not None:
            result["export"] = self.language_policy(
                "export", path=path, format=format, backend=backend
            ).preview()
        else:
            formats = (
                ("srt",) if path == "srt" else ("epub", "docx", "html", "txt", "markdown", "pdf")
            )
            exports = {}
            for selected in formats:
                try:
                    exports[selected] = self.language_policy(
                        "export", path=path, format=selected, backend=backend
                    ).preview()
                except ValueError as error:
                    exports[selected] = {"available": False, "reason": str(error)}
            result["export"] = exports
        return result

    @field_validator("source_lang")
    @classmethod
    def validate_source_lang(cls, value: str) -> str:
        return require_language(value, allow_auto=True)

    @field_validator("target_lang")
    @classmethod
    def validate_target_lang(cls, value: str) -> str:
        return require_language(value)

    @staticmethod
    def create_default_file(path: str) -> bool:
        """Atomically create the default config if absent; return whether it was created."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("x", encoding="utf-8") as f:
                f.write(_DEFAULT_CONFIG_YAML)
            return True
        except FileExistsError:
            return False

    @classmethod
    def load(cls, path: str = "config.yaml") -> Config:
        """Load YAML configuration and apply typed defaults for missing fields."""
        with open(path, "r", encoding="utf-8") as f:
            raw = parse_config_yaml(f.read()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Any) -> Config:
        """Convert a nested YAML dictionary into the runtime configuration model."""
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a mapping of sections.")
        sections = {"language", "llm", "segment", "pipeline", "output", "honorific", "paths"}
        unknown = set(raw) - sections
        if unknown:
            raise ValueError(
                "Unknown configuration sections: " + ", ".join(sorted(map(str, unknown)))
            )
        lang = raw.get("language", {})
        if not isinstance(lang, dict) or set(lang) - {"source", "target"}:
            raise ValueError("Invalid or unknown language configuration fields")
        llm_raw = raw.get("llm", {})
        llm = LLMConfig.model_validate({} if llm_raw is None else llm_raw)
        segment = SegmentConfig.model_validate(raw.get("segment", {}) or {})
        pipeline = PipelineConfig.model_validate(raw.get("pipeline", {}) or {})
        output = OutputConfig.model_validate(raw.get("output", {}) or {})
        return cls(
            source_lang=lang.get("source", "auto"),
            target_lang=lang.get("target", "zh"),
            llm=llm,
            segment=segment,
            pipeline=pipeline,
            output=output,
            honorific_strategy=raw.get("honorific", {}).get("strategy", "keep_style"),
            state_dir=raw.get("paths", {}).get("state_dir", "state"),
        )

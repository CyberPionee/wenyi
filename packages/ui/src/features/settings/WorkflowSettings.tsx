import { useI18n } from "@/i18n";
import { Disclosure } from "@/components/ui/disclosure";
import { Input, Label } from "@/components/ui/form";
import { Select, SelectItem } from "@/components/ui/select";

const section = (config: Record<string, unknown>, key: string) =>
  (config[key] || {}) as Record<string, unknown>;

/** Update workflow settings without changing the project's creation-time translation mode. */
export function workflowField(
  config: Record<string, unknown>,
  group: string,
  key: string,
  value: unknown,
) {
  return {
    ...config,
    [group]: {
      ...section(config, group),
      [key]: value,
    },
  };
}

// Keys the system derives from the autonomy tier, the batch budget and the score history
// when pipeline.tuning is "auto"; they stay pinned to the values below in "manual" mode.
const AUTO_TUNED_KEYS = [
  "review_scope",
  "risk_back_translation",
  "risk_sample_ratio",
  "judge_sample_ratio",
  "max_auto_redo_rounds",
  "quality_judge_dual",
  "bt_score_min",
  "judge_score_min",
  "glossary_extract_inject",
  "glossary_extract_budget_chars",
  "glossary_extract_core_max",
  "glossary_extract_recent_max",
  "glossary_extract_min_terms",
  "glossary_note_chars",
];

// Controls on this form that stand for an auto-tuned key. The remaining auto-tuned keys have
// no control here and can only be pinned through config.yaml.
const AUTO_TUNED_CONTROLS = [
  "risk_back_translation",
  "bt_score_min",
  "judge_score_min",
];

const CONFIG_ONLY_TUNED_KEYS =
  AUTO_TUNED_KEYS.length - AUTO_TUNED_CONTROLS.length;

const fieldValue = (value: unknown) => {
  if (value == null) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
};

export function WorkflowSettings({
  config,
  scope,
  disabled,
  error,
  subtitles = false,
  pdf = false,
  onField,
}: {
  config: Record<string, unknown>;
  scope: "global" | "project";
  disabled: boolean;
  error?: unknown;
  subtitles?: boolean;
  pdf?: boolean;
  onField: (group: string, key: string, value: unknown) => void;
}) {
  const { t: tr } = useI18n();
  const precision =
    scope === "project" &&
    section(config, "pipeline").translation_mode === "best_of_three";
  const PIPELINE: [string, string][] = [
    ["book_understanding", tr("settings.bookUnderstanding")],
    ["polish", tr("settings.polishing")],
    ["review", tr("common.wholeBookReview")],
    ["review_autofix", tr("settings.applyAutofixesToTheSavedTranslation")],
    ["risk_back_translation", tr("settings.riskBackTranslation")],
    ["quality_judge", tr("settings.qualityJudge")],
    ["self_revision", tr("settings.selfRevision")],
    ["editorial_pass", tr("settings.editorialPass")],
    ["final_polish", tr("settings.finalPolish")],
    ["chapter_selfcheck", tr("settings.chapterSelfcheck")],
    ["back_translation", tr("settings.backTranslation")],
  ];
  const HANDS_ON: [string, string][] = [
    ["evaluation_enabled", tr("settings.evaluationEnabled")],
    ["auto_qa_strict", tr("settings.autoQaStrict")],
  ];
  const pipeline = section(config, "pipeline");
  const autoTuned = String(pipeline.tuning || "auto") !== "manual";
  const isManaged = (key: string) =>
    autoTuned && AUTO_TUNED_CONTROLS.includes(key);
  const managedNote = (key: string) => {
    if (!autoTuned) return null;
    const value = pipeline[key];
    return tr("settings.tuningAutoManaged", {
      value:
        typeof value === "boolean"
          ? tr(value ? "settings.tuningEnabled" : "settings.tuningDisabled")
          : fieldValue(value),
    });
  };

  return (
    <fieldset disabled={disabled} className="space-y-4 disabled:opacity-60">
      {!subtitles && (
        <>
          <div className="rounded-lg border border-primary/50 bg-muted/40 p-3 space-y-3">
            <div>
              <p className="text-sm font-medium">
                {tr("settings.tuningHandsOn")}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                {tr("settings.tuningHandsOnHelp")}
              </p>
            </div>
            <div className="grid sm:grid-cols-2 gap-3">
              <div>
                <Label htmlFor="tuning-mode">{tr("settings.tuning")}</Label>
                <Select
                  id="tuning-mode"
                  value={autoTuned ? "auto" : "manual"}
                  onValueChange={(value) => onField("pipeline", "tuning", value)}
                  className="mt-2"
                >
                  <SelectItem value="auto">
                    {tr("settings.tuningAuto")}
                  </SelectItem>
                  <SelectItem value="manual">
                    {tr("settings.tuningManual")}
                  </SelectItem>
                </Select>
                <p className="mt-1 text-xs text-muted-foreground">
                  {tr("settings.tuningHelp")}{" "}
                  {tr("settings.tuningConfigOnly", {
                    count: CONFIG_ONLY_TUNED_KEYS,
                  })}
                </p>
              </div>
              <div>
                <Label htmlFor="autonomy-tier">
                  {tr("settings.autonomyTier")}
                </Label>
                <Select
                  id="autonomy-tier"
                  value={String(pipeline.autonomy_tier || "standard")}
                  onValueChange={(value) =>
                    onField("pipeline", "autonomy_tier", value)
                  }
                  className="mt-2"
                >
                  {["off", "speed", "standard", "precise"].map((s) => (
                    <SelectItem key={s} value={s}>
                      {s === "off"
                        ? tr("settings.autonomyTierOff")
                        : s === "speed"
                          ? tr("settings.autonomyTierSpeed")
                          : s === "precise"
                            ? tr("settings.autonomyTierPrecise")
                            : tr("settings.autonomyTierStandard")}
                    </SelectItem>
                  ))}
                </Select>
                <p className="mt-1 text-xs text-muted-foreground">
                  {tr("settings.autonomyTierHelp")}
                </p>
              </div>
            </div>
            <div className="grid sm:grid-cols-2 gap-3">
              {HANDS_ON.map(([key, label]) => (
                <label key={key} className="flex gap-2 items-center text-sm">
                  <input
                    type="checkbox"
                    checked={Boolean(pipeline[key])}
                    onChange={(e) => onField("pipeline", key, e.target.checked)}
                  />
                  {label}
                </label>
              ))}
            </div>
          </div>
          <div className="grid sm:grid-cols-2 gap-3">
            {PIPELINE.map(([key, label]) => {
              const managed = isManaged(key);
              return (
                <div
                  key={key}
                  className="flex flex-wrap gap-2 items-center text-sm"
                >
                  <label className="flex gap-2 items-center">
                    <input
                      type="checkbox"
                      checked={Boolean(pipeline[key])}
                      disabled={managed || (key === "polish" && precision)}
                      onChange={(e) =>
                        onField("pipeline", key, e.target.checked)
                      }
                    />
                    {label}
                  </label>
                  {managed && (
                    <span className="text-xs text-muted-foreground">
                      {managedNote(key)}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
          <div className="grid sm:grid-cols-2 gap-3">
            <div>
              <Label htmlFor="glossary-scope">
                {tr("settings.glossaryScope")}
              </Label>
              <Select
                id="glossary-scope"
                value={String(
                  section(config, "pipeline").glossary_scope || "chapter",
                )}
                onValueChange={(value) =>
                  onField("pipeline", "glossary_scope", value)
                }
                className="mt-2"
              >
                {["chapter", "full"].map((s) => (
                  <SelectItem key={s} value={s}>
                    {s === "chapter"
                      ? tr("settings.glossaryScopeChapter")
                      : tr("settings.glossaryScopeFull")}
                  </SelectItem>
                ))}
              </Select>
              <p className="mt-1 text-xs text-muted-foreground">
                {tr("settings.glossaryScopeHelp")}
              </p>
            </div>
            <div>
              <Label htmlFor="decision-anchors">
                {tr("settings.decisionAnchors")}
              </Label>
              <Select
                id="decision-anchors"
                value={String(
                  section(config, "pipeline").decision_anchors || "off",
                )}
                onValueChange={(value) =>
                  onField("pipeline", "decision_anchors", value)
                }
                className="mt-2"
              >
                {["off", "auto", "risk"].map((s) => (
                  <SelectItem key={s} value={s}>
                    {s}
                  </SelectItem>
                ))}
              </Select>
            </div>
          </div>
          <div className="grid sm:grid-cols-3 gap-3">
            <div>
              <Label htmlFor="judge-score-min">
                {tr("settings.judgeScoreMin")}
              </Label>
              <Input
                id="judge-score-min"
                type="number"
                min={1}
                max={5}
                step={0.5}
                disabled={isManaged("judge_score_min")}
                value={Number(pipeline.judge_score_min ?? 3.5)}
                onChange={(e) =>
                  onField("pipeline", "judge_score_min", Number(e.target.value))
                }
                className="mt-2"
              />
              {isManaged("judge_score_min") && (
                <p className="mt-1 text-xs text-muted-foreground">
                  {managedNote("judge_score_min")}
                </p>
              )}
            </div>
            <div>
              <Label htmlFor="bt-score-min">{tr("settings.btScoreMin")}</Label>
              <Input
                id="bt-score-min"
                type="number"
                min={0}
                max={1}
                step={0.05}
                disabled={isManaged("bt_score_min")}
                value={Number(pipeline.bt_score_min ?? 0.45)}
                onChange={(e) =>
                  onField("pipeline", "bt_score_min", Number(e.target.value))
                }
                className="mt-2"
              />
              {isManaged("bt_score_min") && (
                <p className="mt-1 text-xs text-muted-foreground">
                  {managedNote("bt_score_min")}
                </p>
              )}
            </div>
            <div>
              <Label htmlFor="l2-min-consistency">
                {tr("settings.l2MinConsistency")}
              </Label>
              <Input
                id="l2-min-consistency"
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={Number(
                  section(config, "pipeline").l2_min_consistency ?? 1,
                )}
                onChange={(e) =>
                  onField(
                    "pipeline",
                    "l2_min_consistency",
                    Number(e.target.value),
                  )
                }
                className="mt-2"
              />
            </div>
          </div>
        </>
      )}
      {subtitles && (
        <p className="text-sm text-muted-foreground">
          {tr("settings.subtitlesUseASeparateWorkflowWithoutBook")}
        </p>
      )}
      <Disclosure
        title={tr("settings.performance")}
        error={error}
        summary={tr("settings.performanceSummary", {
          tokens: Number(
            section(config, "segment").max_tokens_per_batch ?? 1800,
          ),
        })}
      >
        {!subtitles && (
          <label className="flex gap-2 items-center text-sm">
            <input
              type="checkbox"
              checked={Boolean(
                section(config, "pipeline").annotation_alignment,
              )}
              onChange={(e) =>
                onField("pipeline", "annotation_alignment", e.target.checked)
              }
            />
            {tr("settings.paragraphAnnotationAlignment")}
          </label>
        )}
        <div className="grid sm:grid-cols-2 gap-4">
          <div>
            <Label htmlFor="batch-tokens">
              {tr("settings.tokensPerBatch")}
            </Label>
            <Input
              id="batch-tokens"
              type="number"
              min={1}
              value={Number(
                section(config, "segment").max_tokens_per_batch ?? 1800,
              )}
              onChange={(e) =>
                onField(
                  "segment",
                  "max_tokens_per_batch",
                  Number(e.target.value),
                )
              }
              className="mt-2"
            />
          </div>
          <div>
            <Label htmlFor="segment-tokens">
              {tr("settings.tokensPerParagraph")}
            </Label>
            <Input
              id="segment-tokens"
              type="number"
              min={1}
              value={Number(
                section(config, "segment").max_tokens_per_segment ?? 1200,
              )}
              onChange={(e) =>
                onField(
                  "segment",
                  "max_tokens_per_segment",
                  Number(e.target.value),
                )
              }
              className="mt-2"
            />
          </div>
          {!subtitles && (
            <div>
              <Label htmlFor="review-concurrency">
                {tr("settings.reviewConcurrency")}
              </Label>
              <Input
                id="review-concurrency"
                type="number"
                min={1}
                value={Number(
                  section(config, "pipeline").review_concurrency ?? 4,
                )}
                onChange={(e) =>
                  onField(
                    "pipeline",
                    "review_concurrency",
                    Number(e.target.value),
                  )
                }
                className="mt-2"
              />
            </div>
          )}
          {pdf && (
            <div>
              <Label htmlFor="pdf-backend">{tr("settings.pdfParser")}</Label>
              <Select
                id="pdf-backend"
                value={String(
                  section(config, "pipeline").pdf_backend || "mineru",
                )}
                onValueChange={(value) =>
                  onField("pipeline", "pdf_backend", value)
                }
                className="mt-2"
              >
                {["mineru", "babeldoc"].map((s) => (
                  <SelectItem key={s} value={s}>
                    {s}
                  </SelectItem>
                ))}
              </Select>
            </div>
          )}
        </div>
      </Disclosure>
    </fieldset>
  );
}

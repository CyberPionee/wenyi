import { useI18n, type MessageKey } from "@/i18n";
import { Badge } from "@/components/ui/badge";
import { Disclosure } from "@/components/ui/disclosure";
import type { EvaluationData, TuningItem } from "@/lib/api";

const riskLabel = (reason: string) => reason.replace(/_/g, " ");

// Pinned values first so the keys the system decides stand out.
const TUNING_SOURCE_ORDER: TuningItem["source"][] = [
  "pinned",
  "history",
  "tier",
  "budget",
  "default",
];

const TUNING_SOURCE_KEYS: Record<string, MessageKey> = {
  pinned: "eval.tuningSourcePinned",
  history: "eval.tuningSourceHistory",
  tier: "eval.tuningSourceTier",
  budget: "eval.tuningSourceBudget",
  default: "eval.tuningSourceDefault",
};

const TUNING_TIER_KEYS: Record<string, MessageKey> = {
  off: "eval.tuningTierOff",
  speed: "eval.tuningTierSpeed",
  standard: "eval.tuningTierStandard",
  precise: "eval.tuningTierPrecise",
};

const tuningValue = (value: unknown) => {
  if (value == null) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
};

const sourceRank = (source: TuningItem["source"]) => {
  const rank = TUNING_SOURCE_ORDER.indexOf(source);
  return rank < 0 ? TUNING_SOURCE_ORDER.length : rank;
};

export function EvaluationDetails({ evaluation }: { evaluation?: EvaluationData }) {
  const { t: tr } = useI18n();
  if (!evaluation) return null;

  const gate = evaluation.machine_gate;
  const l2 = evaluation.l2;
  const drift = l2?.items || [];
  const btMin = gate?.bt_score_min ?? 0.45;
  const judgeMin = gate?.judge_score_min ?? 3.5;
  const backTranslations = (evaluation.back_translation || []).filter(
    (item) => (item.score ?? 1) < btMin,
  );
  const judgeScores = (evaluation.judge_scores || []).filter(
    (item) => item.score != null && item.score < judgeMin,
  );
  const riskSegments = evaluation.risk_segments || [];
  const l2Checked = l2?.checked ?? gate?.l2_checked_count ?? 0;
  const l2Drifted = l2?.drifted ?? gate?.l2_drift_count ?? 0;
  const l2Rate = l2?.consistency_rate ?? gate?.l2_consistency_rate;

  const tuning = evaluation.tuning;
  const tuningItems = tuning?.items || [];
  const calibration = tuning?.calibration || [];
  const hasTuning =
    Boolean(tuning) && (tuningItems.length > 0 || calibration.length > 0);
  const sortedTuning = [...tuningItems].sort(
    (a, b) => sourceRank(a.source) - sourceRank(b.source),
  );
  const tierLabel = (tier: string) => {
    const key = TUNING_TIER_KEYS[tier];
    return key ? tr(key) : tier;
  };
  const sourceLabel = (item: TuningItem) => {
    const key = TUNING_SOURCE_KEYS[item.source];
    return key ? tr(key) : item.source;
  };

  const hasAny =
    drift.length > 0 ||
    backTranslations.length > 0 ||
    judgeScores.length > 0 ||
    riskSegments.length > 0 ||
    l2Checked > 0 ||
    hasTuning;
  if (!hasAny) return null;

  const location = (chapter: number, index: number) =>
    tr("eval.location", { chapter: chapter + 1, index: index + 1 });

  return (
    <Disclosure title={tr("eval.details")} summary={tr("eval.detailsSummary")}>
      <div className="space-y-4 text-sm">
        <div className="flex flex-wrap gap-2">
          <Badge variant={gate?.l2_passed === false ? "destructive" : "outline"}>
            {tr("eval.l2")}
            {": "}
            {l2Rate != null ? `${Math.round(l2Rate * 100)}%` : "—"}
            {` (${l2Drifted}/${l2Checked})`}
          </Badge>
          <Badge variant={gate?.bt_passed === false ? "destructive" : "outline"}>
            {tr("eval.l1Low")}
            {": "}
            {gate?.bt_low_count ?? backTranslations.length}
            {` / ${gate?.bt_sample_count ?? (evaluation.back_translation || []).length}`}
          </Badge>
          <Badge variant={gate?.judge_passed === false ? "destructive" : "outline"}>
            {tr("eval.l3Avg")}
            {": "}
            {gate?.judge_avg ?? "—"}
          </Badge>
        </div>

        {drift.length > 0 && (
          <section>
            <h4 className="mb-2 font-medium">{tr("eval.l2DriftTitle")}</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead className="text-muted-foreground">
                  <tr>
                    <th className="p-2 font-medium">{tr("eval.locationColumn")}</th>
                    <th className="p-2 font-medium">{tr("eval.sourceTerm")}</th>
                    <th className="p-2 font-medium">{tr("eval.expectedTarget")}</th>
                    <th className="p-2 font-medium">{tr("eval.targetPreview")}</th>
                  </tr>
                </thead>
                <tbody>
                  {drift.map((item, position) => (
                    <tr key={`${item.chapter}-${item.index}-${position}`} className="border-t">
                      <td className="p-2 text-muted-foreground">
                        {location(item.chapter, item.index)}
                      </td>
                      <td className="p-2">{item.source_term || "—"}</td>
                      <td className="p-2">
                        {(item.missing_targets?.length
                          ? item.missing_targets
                          : [item.expected_target || ""]
                        )
                          .filter(Boolean)
                          .join(" / ") || "—"}
                      </td>
                      <td className="p-2 text-muted-foreground">
                        {item.target_preview || "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {backTranslations.length > 0 && (
          <section>
            <h4 className="mb-2 font-medium">{tr("eval.l1Title")}</h4>
            <ul className="space-y-1">
              {backTranslations.map((item, position) => (
                <li key={position} className="border-t pt-1">
                  <span className="text-muted-foreground">
                    {tr("eval.score")}: {item.score?.toFixed(3) ?? "—"}
                  </span>
                  <div className="truncate" title={item.source_preview}>
                    {item.source_preview}
                  </div>
                  <div className="truncate text-muted-foreground" title={item.back_preview}>
                    {item.back_preview}
                  </div>
                </li>
              ))}
            </ul>
          </section>
        )}

        {judgeScores.length > 0 && (
          <section>
            <h4 className="mb-2 font-medium">{tr("eval.l3Title")}</h4>
            <ul className="space-y-1">
              {judgeScores.map((item, position) => (
                <li key={position} className="border-t pt-1">
                  <span className="text-muted-foreground">
                    {item.index != null
                      ? tr("eval.judgeIndex", { index: (item.index ?? 0) + 1 })
                      : tr("eval.score")}
                    {": "}
                    {item.score ?? "—"}
                  </span>
                  {item.note ? <div>{item.note}</div> : null}
                </li>
              ))}
            </ul>
          </section>
        )}

        {riskSegments.length > 0 && (
          <section>
            <h4 className="mb-2 font-medium">{tr("eval.riskTitle")}</h4>
            <ul className="space-y-1">
              {riskSegments.map((item, position) => (
                <li key={position} className="border-t pt-1">
                  <span className="text-muted-foreground">
                    {location(item.chapter, item.index)}
                    {" · "}
                    {(item.reasons || []).map(riskLabel).join(", ")}
                  </span>
                  <div className="truncate" title={item.source_preview}>
                    {item.source_preview}
                  </div>
                </li>
              ))}
            </ul>
          </section>
        )}
        {hasTuning && (
          <Disclosure
            title={tr("eval.tuningTitle")}
            summary={tr("eval.tuningSummary", {
              mode:
                tuning?.mode === "manual"
                  ? tr("eval.tuningModeManual")
                  : tr("eval.tuningModeAuto"),
              tier: tierLabel(tuning?.tier || "standard"),
            })}
          >
            <p className="text-xs text-muted-foreground">
              {tr("eval.tuningHelp")}
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead className="text-muted-foreground">
                  <tr>
                    <th className="p-2 font-medium">{tr("eval.tuningKeyColumn")}</th>
                    <th className="p-2 font-medium">
                      {tr("eval.tuningValueColumn")}
                    </th>
                    <th className="p-2 font-medium">
                      {tr("eval.tuningSourceColumn")}
                    </th>
                    <th className="p-2 font-medium">{tr("eval.tuningNoteColumn")}</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedTuning.map((item, position) => (
                    <tr key={`${item.key}-${position}`} className="border-t align-top">
                      <td className="p-2 font-mono text-xs">{item.key}</td>
                      <td className="p-2">{tuningValue(item.value)}</td>
                      <td className="p-2 text-muted-foreground">
                        {sourceLabel(item)}
                      </td>
                      <td className="p-2 text-muted-foreground">
                        {item.note || "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {calibration.length > 0 && (
              <section className="rounded border border-dashed p-2">
                <h4 className="mb-1 font-medium">
                  {tr("eval.tuningCalibrationTitle")}
                </h4>
                <ul className="space-y-1 text-xs text-muted-foreground">
                  {calibration.map((item, position) => (
                    <li key={`${item.key}-${position}`}>
                      {tr("eval.tuningCalibrationItem", {
                        key: item.key,
                        value: tuningValue(item.suggested_value),
                        reason: item.reason,
                      })}
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </Disclosure>
        )}
        {(evaluation.history || []).length > 1 && (
          <section>
            <h4 className="mb-2 font-medium">{tr("eval.historyTitle")}</h4>
            <ul className="space-y-1">
              {evaluation.history!.slice(-5).reverse().map((item, position) => (
                <li key={position} className="border-t pt-1 text-muted-foreground">
                  <span>{item.ts}</span>
                  {" · "}
                  <span>
                    {item.passed
                      ? tr("data.machineGatePassed")
                      : tr("data.machineGateFailed")}
                  </span>
                  {" · "}
                  <span>
                    {tr("eval.l2")}
                    {": "}
                    {item.l2_consistency_rate != null
                      ? `${Math.round(item.l2_consistency_rate * 100)}%`
                      : "—"}
                  </span>
                  {" · "}
                  <span>
                    {tr("eval.l3Avg")}
                    {": "}
                    {item.judge_avg ?? "—"}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </Disclosure>
  );
}

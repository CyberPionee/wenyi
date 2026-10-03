import { useI18n } from "@/i18n";
import { Badge } from "@/components/ui/badge";
import { Disclosure } from "@/components/ui/disclosure";
import type { EvaluationData } from "@/lib/api";

const riskLabel = (reason: string) => reason.replace(/_/g, " ");

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

  const hasAny =
    drift.length > 0 ||
    backTranslations.length > 0 ||
    judgeScores.length > 0 ||
    riskSegments.length > 0 ||
    l2Checked > 0;
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
      </div>
    </Disclosure>
  );
}

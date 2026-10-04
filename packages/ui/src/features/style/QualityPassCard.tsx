import { useI18n } from "@/i18n";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type Finding = Record<string, unknown>;

function num(v: unknown): string {
  return typeof v === "number" ? String(v) : String(v ?? "");
}

function text(v: unknown): string {
  return typeof v === "string" ? v : v == null ? "" : JSON.stringify(v);
}

function Location({ chapter, index }: { chapter?: unknown; index?: unknown }) {
  const { t } = useI18n();
  if (chapter == null && index == null) return null;
  return (
    <span className="text-xs text-muted-foreground">
      {t("style.qpChapter", { chapter: num(chapter) })}
      {index != null ? " · " + t("style.qpParagraph", { number: num(index) }) : ""}
    </span>
  );
}

export function QualityPassCard({
  qualityPass,
}: {
  qualityPass?: Record<string, unknown> | null;
}) {
  const { t } = useI18n();
  const qp = qualityPass || {};
  const notes = (qp.editorial_notes as string[] | undefined) || [];
  const revisions = (qp.self_revision_notes as Finding[] | undefined) || [];
  const polishes = (qp.final_polish_notes as Finding[] | undefined) || [];
  const checks = (qp.chapter_selfcheck_findings as Finding[] | undefined) || [];
  const backs = (qp.back_translation_notes as Finding[] | undefined) || [];
  const empty =
    !notes.length &&
    !revisions.length &&
    !polishes.length &&
    !checks.length &&
    !backs.length &&
    !backs.length;

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("style.aiSuggestions")}</CardTitle>
      </CardHeader>
      <CardContent className="text-sm space-y-4">
        <p className="text-muted-foreground">{t("style.aiSuggestionsHint")}</p>
        {empty ? <p className="text-muted-foreground">{t("style.qpEmpty")}</p> : null}

        {checks.length > 0 && (
          <section className="space-y-2">
            <h4 className="font-medium">
              {t("style.qpSelfcheck")}
              <Badge variant="secondary" className="ml-2">
                {t("style.qpCount", { count: String(checks.length) })}
              </Badge>
            </h4>
            <ul className="divide-y">
              {checks.map((f, i) => (
                <li key={i} className="py-2">
                  <Location chapter={f.chapter} index={f.index} />
                  <div className="text-xs text-muted-foreground">{text(f.kind)}</div>
                  <div>{text(f.detail)}</div>
                </li>
              ))}
            </ul>
          </section>
        )}

        {(revisions.length > 0 || polishes.length > 0) && (
          <section className="space-y-2">
            <h4 className="font-medium">{t("style.qpCandidates")}</h4>
            <p className="text-xs text-muted-foreground">{t("style.qpCandidatesHint")}</p>
            <ul className="divide-y">
              {([
                ...revisions.map((f) => ({ ...f, _kind: "self" })),
                ...polishes.map((f) => ({ ...f, _kind: "polish" })),
              ] as (Finding & { _kind: string })[]).map((f, i) => (
                <li key={i} className="py-2">
                  <Location chapter={f.chapter} index={f.index} />
                  <div className="text-xs text-muted-foreground">
                    {f._kind === "self" ? t("style.qpSelfRevision") : t("style.qpFinalPolish")}
                  </div>
                  <div className="whitespace-pre-wrap">{text(f.suggested)}</div>
                </li>
              ))}
            </ul>
          </section>
        )}

        {notes.length > 0 && (
          <section className="space-y-2">
            <h4 className="font-medium">{t("style.qpEditorial")}</h4>
            <ul className="list-disc pl-5 space-y-1">
              {notes.map((n, i) => (
                <li key={i}>{text(n)}</li>
              ))}
            </ul>
          </section>
        )}

        {backs.length > 0 && (
          <section className="space-y-2">
            <h4 className="font-medium">{t("style.qpBackTranslation")}</h4>
            <ul className="divide-y">
              {backs.map((f, i) => (
                <li key={i} className="py-2">
                  <Location chapter={f.chapter} />
                  <div className="text-xs text-muted-foreground">{text(f.source_preview)}</div>
                  <div>{text(f.back_preview)}</div>
                </li>
              ))}
            </ul>
          </section>
        )}
      </CardContent>
    </Card>
  );
}

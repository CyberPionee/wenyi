import { useMemo } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { StatusBadge } from "@/components/StatusBadge";
import { PageContainer, PageHeader } from "@/components/layout/AppLayout";
import { Card, CardContent } from "@/components/ui/card";
import { ErrorNotice } from "@/components/ui/data";
import { useI18n } from "@/i18n";
import { workflowStageLabel } from "@/i18n/labels";
import { api, isProjectBusy } from "@/lib/api";
import { progressInterval } from "@/lib/runtime";
import { useProjectProgress } from "@/lib/ws";

type Rec = Record<string, unknown>;

/** Milestone events worth surfacing on the run timeline (raw llm noise stays out). */
const MILESTONES = new Set([
  "translate_run_finished",
  "chapter_done",
  "quality_pass_finished",
  "glossary_disambiguation_finished",
  "evaluation_finished",
  "evaluation_redo_round",
  "evaluation_redo_finished",
  "review_autofix_finished",
  "report_saved",
  "run_resumed",
  "task_paused",
]);

function record(value: unknown): Rec {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Rec)
    : {};
}

function num(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function milestoneSummary(type: string, payload: Rec): string {
  switch (type) {
    case "chapter_done":
      return `#${payload.chapter} ${String(payload.title ?? "")}`;
    case "quality_pass_finished":
      return `chapters=${num(payload.chapters)}${
        num(payload.failed) ? ` failed=${num(payload.failed)}` : ""
      }`;
    case "glossary_disambiguation_finished":
      return `judged=${num(payload.judged)} unresolved=${num(
        payload.unresolved,
      )} deferred=${num(payload.deferred)}`;
    case "evaluation_redo_round":
      return `round ${num(payload.round) || ""}`.trim();
    case "task_paused":
      return String(payload.kind ?? "");
    default:
      return "";
  }
}

type StageState = "disabled" | "done" | "running" | "pending";

interface StageContext {
  busy: boolean;
  initialized: boolean;
  chaptersDone: boolean;
  kind: string;
  label: string;
  hasQuality: boolean;
  hasEvaluation: boolean;
  hasReview: boolean;
  hasReport: boolean;
}

function stageState(id: string, enabled: boolean, ctx: StageContext): StageState {
  if (!enabled) return "disabled";
  const label = ctx.label.toLowerCase();
  switch (id) {
    case "prepare":
    case "book_understanding":
      return ctx.initialized ? "done" : ctx.busy ? "running" : "pending";
    case "translation":
      return ctx.chaptersDone
        ? "done"
        : ctx.busy && ctx.kind === "translation"
          ? "running"
          : "pending";
    case "polish":
    case "annotation_alignment":
      return ctx.chaptersDone ? "done" : "pending";
    case "quality_pass":
      if (ctx.hasQuality) return "done";
      return ctx.busy && label.includes("quality") ? "running" : "pending";
    case "review":
      if (ctx.hasReview) return "done";
      return ctx.busy && ctx.kind === "review" ? "running" : "pending";
    case "review_autofix":
      return ctx.hasReview ? "done" : "pending";
    case "evaluation":
      if (ctx.hasEvaluation) return "done";
      return ctx.busy && (label.includes("autofix") || label.includes("evaluat"))
        ? "running"
        : "pending";
    case "report":
      return ctx.hasReport ? "done" : "pending";
    default:
      return "pending";
  }
}

const STATE_STYLE: Record<StageState, string> = {
  done: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
  running: "bg-blue-500/15 text-blue-700 dark:text-blue-300",
  pending: "bg-muted text-muted-foreground",
  disabled: "bg-muted/50 text-muted-foreground/60",
};

function StatRow({ items }: { items: Array<[string, unknown]> }) {
  const shown = items.filter(([, value]) => value !== undefined && value !== null);
  if (!shown.length) return null;
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
      {shown.map(([label, value]) => (
        <span key={label}>
          <span className="text-muted-foreground">{label}: </span>
          <span className="font-medium tabular-nums">{String(value)}</span>
        </span>
      ))}
    </div>
  );
}

export default function PipelinePage() {
  const { t: tr, locale } = useI18n();
  const { pid = "" } = useParams();
  const { msg, connected } = useProjectProgress(pid);
  const enabled = !!pid;

  const projectQuery = useQuery({
    queryKey: ["project", pid],
    queryFn: () => api.getProject(pid),
    refetchInterval: progressInterval(connected, 2500),
    enabled,
  });
  const chaptersQuery = useQuery({
    queryKey: ["chapters", pid],
    queryFn: () => api.listChapters(pid),
    refetchInterval: progressInterval(connected, 3000),
    enabled,
  });
  const workflowQuery = useQuery({
    queryKey: ["workflow", pid],
    queryFn: () => api.getWorkflow(pid),
    refetchInterval: progressInterval(connected, 2500),
    enabled,
  });
  const reportQuery = useQuery({
    queryKey: ["report", pid],
    queryFn: () => api.getReport(pid),
    enabled,
  });
  const analysisQuery = useQuery({
    queryKey: ["analysis", pid],
    queryFn: () => api.getAnalysis(pid),
    enabled,
  });
  const eventsQuery = useQuery({
    queryKey: ["events", pid],
    queryFn: () => api.listEvents(pid),
    refetchInterval: 5000,
    enabled,
  });

  const project = projectQuery.data;
  const workflow = workflowQuery.data;
  const report = reportQuery.data;
  const analysis = record(analysisQuery.data?.analysis);
  const busy = isProjectBusy(project?.status);

  // Live label merge, same rule as WorkflowPanel: prefer the WS message when it is newer
  // and belongs to the current run, otherwise the workflow snapshot.
  const cached = workflow?.progress;
  const cachedIsNewer =
    typeof cached?.updated_at === "string" &&
    !!msg?.updated_at &&
    Date.parse(cached.updated_at) > Date.parse(msg.updated_at);
  const live =
    msg?.run_id && msg.run_id === workflow?.run_id && msg.label && !cachedIsNewer
      ? msg
      : cached;
  const liveLabel = String(live?.label ?? "");

  const chapters = chaptersQuery.data ?? [];
  const chaptersDone =
    chapters.length > 0 && chapters.every((c) => c.status === "done");

  const events = eventsQuery.data ?? [];
  const latestOf = (type: string) => {
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i].type === type) return events[i];
    }
    return undefined;
  };

  const qualityDone = record(analysis.quality_pass_done);
  const qualityResult = record(analysis.quality_pass);
  const autoQa = record(report?.auto_qa);
  const reviewInfo = record(report?.review);
  const machineGate = record(report?.machine_gate);

  const ctx: StageContext = {
    busy,
    initialized: !!project?.initialized,
    chaptersDone,
    kind: String(workflow?.kind ?? ""),
    label: liveLabel,
    hasQuality: Object.keys(qualityDone).length > 0,
    hasEvaluation: !!report?.evaluation || !!latestOf("evaluation_finished"),
    hasReview: !!report?.review || !!latestOf("review_autofix_finished"),
    hasReport: !!report,
  };

  const timeline = useMemo(
    () =>
      (workflow?.stages ?? []).map((stage: Rec, index: number) => ({
        index: index + 1,
        id: String(stage.id ?? ""),
        enabled: stage.enabled !== false,
        label: workflowStageLabel(
          String(stage.id ?? ""),
          String(stage.label ?? ""),
          tr,
        ) as string,
        state: stageState(
          String(stage.id ?? ""),
          stage.enabled !== false,
          ctx,
        ),
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [workflow, ctx.busy, ctx.chaptersDone, ctx.label, ctx.hasQuality, ctx.hasEvaluation, ctx.hasReview, ctx.hasReport, tr],
  );

  const disambiguation = record(
    latestOf("glossary_disambiguation_finished")?.payload,
  );
  const qualityEvent = record(latestOf("quality_pass_finished")?.payload);
  const milestones = events.filter((e) => MILESTONES.has(e.type)).slice(0, 12);

  const timeOf = (iso: unknown) =>
    typeof iso === "string" ? new Date(iso).toLocaleString(locale) : "";

  return (
    <>
      <PageHeader title={tr("pipeline.title")} subtitle={tr("pipeline.subtitle")} />
      <PageContainer className="space-y-4">
        <ErrorNotice
          error={
            projectQuery.error ??
            workflowQuery.error ??
            reportQuery.error ??
            analysisQuery.error ??
            eventsQuery.error
          }
        />

        {/* Hero: current stage */}
        <Card>
          <CardContent className="p-5 space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="font-medium">{tr("pipeline.currentStage")}</h2>
              {project?.status && <StatusBadge status={project.status} />}
            </div>
            <div role="status" className="rounded border p-3 text-sm">
              {liveLabel || tr("pipeline.idle")}
              {live && num(live.total) > 0 && (
                <span className="ml-2 text-muted-foreground">
                  {tr("workflow.stageCount", {
                    done: num(live.done),
                    total: num(live.total),
                  })}
                </span>
              )}
            </div>
            <p className="text-sm text-muted-foreground">
              {tr("workflow.stageCount", {
                done: chapters.filter((c) => c.status === "done").length,
                total: chapters.length,
              })}
            </p>
          </CardContent>
        </Card>

        {/* Stage timeline */}
        <Card>
          <CardContent className="p-5 space-y-3">
            <h2 className="font-medium">{tr("pipeline.timeline")}</h2>
            {timeline.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                {tr("pipeline.notStarted")}
              </p>
            ) : (
              <ol className="grid gap-2 sm:grid-cols-2">
                {timeline.map((stage) => (
                  <li
                    key={stage.id}
                    className="flex items-center gap-3 rounded border px-3 py-2"
                  >
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {stage.index}
                    </span>
                    <span className="flex-1 text-sm font-medium">
                      {stage.label}
                    </span>
                    <span
                      className={`rounded px-2 py-0.5 text-xs ${STATE_STYLE[stage.state]}`}
                    >
                      {stage.state === "done"
                        ? tr("pipeline.completed")
                        : stage.state === "running"
                          ? tr("common.running")
                          : stage.state === "disabled"
                            ? tr("workflowPanel.disabled")
                            : tr("pipeline.notStarted")}
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </CardContent>
        </Card>

        {/* Stage-specific results (our pipeline's own instrumentation) */}
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardContent className="p-5 space-y-3">
              <h2 className="font-medium">
                {tr("pipeline.terminologyCollisions")}
              </h2>
              {disambiguation.collision_id || "collisions" in disambiguation ? (
                <>
                  <StatRow
                    items={[
                      [tr("pipeline.judged"), disambiguation.judged],
                      [tr("pipeline.unresolved"), disambiguation.unresolved],
                      [tr("pipeline.deferred"), disambiguation.deferred],
                      [
                        tr("pipeline.renderingsApplied"),
                        disambiguation.renderings_applied,
                      ],
                    ]}
                  />
                  <p className="text-xs text-muted-foreground">
                    {tr("pipeline.lastRun", {
                      time: timeOf(latestOf("glossary_disambiguation_finished")?.created_at),
                    })}
                  </p>
                </>
              ) : (
                <p className="text-sm text-muted-foreground">
                  {tr("pipeline.notStarted")}
                </p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-5 space-y-3">
              <h2 className="font-medium">{tr("pipeline.qualityPasses")}</h2>
              <StatRow
                items={[
                  [
                    tr("pipeline.editorialNotes"),
                    (qualityResult.editorial_notes as unknown[] | undefined)?.length,
                  ],
                  [
                    tr("pipeline.polishCandidates"),
                    (qualityResult.final_polish_notes as unknown[] | undefined)?.length,
                  ],
                  [
                    tr("pipeline.selfcheckFindings"),
                    (
                      qualityResult.chapter_selfcheck_findings as unknown[] | undefined
                    )?.length,
                  ],
                  [
                    tr("pipeline.completed"),
                    qualityEvent.chapters
                      ? `${num(qualityEvent.chapters)}`
                      : undefined,
                  ],
                ]}
              />
              {latestOf("quality_pass_finished") && (
                <p className="text-xs text-muted-foreground">
                  {tr("pipeline.lastRun", {
                    time: timeOf(latestOf("quality_pass_finished")?.created_at),
                  })}
                </p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-5 space-y-3">
              <h2 className="font-medium">
                {tr("pipeline.evaluationAutofix")}
              </h2>
              <StatRow
                items={[
                  [
                    autoQa.passed === true
                      ? tr("pipeline.gatePassed")
                      : autoQa.passed === false
                        ? tr("pipeline.gateBlocked")
                        : tr("pipeline.notStarted"),
                    autoQa.residual_finding_count !== undefined
                      ? autoQa.residual_finding_count
                      : undefined,
                  ],
                  [
                    tr("pipeline.residuals"),
                    autoQa.residual_finding_count,
                  ],
                  [
                    tr("pipeline.autofixApplied"),
                    reviewInfo.autofix_applied_segment_count,
                  ],
                  [
                    tr("pipeline.autofixFailed"),
                    reviewInfo.autofix_failed_issue_count,
                  ],
                  [tr("pipeline.gateBlocked"), machineGate.blocking],
                ]}
              />
              {latestOf("evaluation_finished") && (
                <p className="text-xs text-muted-foreground">
                  {tr("pipeline.lastRun", {
                    time: timeOf(latestOf("evaluation_finished")?.created_at),
                  })}
                </p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-5 space-y-3">
              <h2 className="font-medium">{tr("pipeline.reviewResult")}</h2>
              <StatRow
                items={[
                  [tr("pipeline.reviewIssues"), reviewInfo.issue_count],
                  [tr("pipeline.autofixApplied"), reviewInfo.change_count],
                  [
                    tr("pipeline.completed"),
                    reviewInfo.autofix_status ?? undefined,
                  ],
                ]}
              />
              {reviewInfo.review_id ? (
                <p className="text-xs text-muted-foreground">
                  {tr("pipeline.lastRun", { time: String(reviewInfo.review_id) })}
                </p>
              ) : (
                <p className="text-sm text-muted-foreground">
                  {tr("pipeline.notStarted")}
                </p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Milestones */}
        <Card>
          <CardContent className="p-5 space-y-3">
            <h2 className="font-medium">{tr("pipeline.milestones")}</h2>
            {milestones.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                {tr("pipeline.noMilestones")}
              </p>
            ) : (
              <ul className="divide-y text-sm">
                {milestones.map((e) => {
                  const summary = milestoneSummary(e.type, record(e.payload));
                  return (
                    <li key={e.id} className="flex gap-3 py-1.5">
                      <span className="shrink-0 text-xs text-muted-foreground tabular-nums">
                        {timeOf(e.created_at)}
                      </span>
                      <span className="font-medium">{e.type}</span>
                      {summary && (
                        <span className="text-muted-foreground">{summary}</span>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </CardContent>
        </Card>
      </PageContainer>
    </>
  );
}

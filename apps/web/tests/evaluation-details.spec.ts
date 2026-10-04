import { expect, test } from "@playwright/test";
import { fakeApi, pid } from "./fixtures";

const report = {
  summary: { chapters_done: 1, review_issues: 0 },
  auto_qa: {
    passed: false,
    empty_target_count: 0,
    open_conflict_count: 0,
    residual_finding_count: 1,
    open_issue_count: 0,
    blocking: true,
  },
  machine_gate: {
    passed: false,
    blocking: true,
    l0_passed: false,
    l2_passed: false,
    bt_passed: false,
    judge_passed: false,
    l2_checked_count: 12,
    l2_drift_count: 2,
    l2_consistency_rate: 0.83,
    l2_min_consistency: 1.0,
    bt_sample_count: 4,
    bt_low_count: 1,
    bt_score_min: 0.45,
    judge_sample_count: 3,
    judge_avg: 3.1,
    judge_low_count: 1,
    judge_score_min: 3.5,
    residual_finding_count: 1,
    empty_target_count: 0,
    open_conflict_count: 0,
    open_issue_count: 0,
  },
  evaluation: {
    l2: {
      checked: 12,
      drifted: 2,
      consistency_rate: 0.83,
      items: [
        {
          chapter: 0,
          index: 3,
          source_term: "ドルフィン・ホテル",
          expected_target: "海豚酒店（Dolphin Hotel）",
          missing_targets: ["海豚酒店（Dolphin Hotel）"],
          source_preview: "ドルフィン・ホテルへ向かった",
          target_preview: "他去了那家旅馆",
        },
      ],
    },
    back_translation: [
      { source_preview: "彼は立ち止まった", back_preview: "He stopped walking", score: 0.21 },
    ],
    judge_scores: [{ index: 5, score: 2.5, note: "翻译腔明显" }],
    risk_segments: [
      {
        chapter: 0,
        index: 3,
        source_preview: "ドルフィン・ホテルへ向かった",
        target_preview: "他去了那家旅馆",
        reasons: ["dialog", "term_like"],
      },
    ],
    machine_gate: {},
    history: [
      {
        ts: "2026-10-03T10:00:00+08:00",
        passed: true,
        l2_consistency_rate: 1.0,
        judge_avg: 4.2,
      },
      {
        ts: "2026-10-03T11:00:00+08:00",
        passed: false,
        l2_consistency_rate: 0.83,
        judge_avg: 3.1,
      },
    ],
  },
};

test("evaluation details list L2 drift, low scores, and risk segments", async ({ page }) => {
  await fakeApi(page, { [`/projects/${pid}/report`]: report });
  await page.goto(`/projects/${pid}`);

  const details = page.locator("details", { hasText: "Evaluation details" });
  await details.locator("summary").click();

  await expect(details).toContainText("Term consistency");
  await expect(details).toContainText("83%");
  await expect(details).toContainText("Glossary terms not used in the translation");
  await expect(details).toContainText("ドルフィン・ホテル");
  await expect(details).toContainText("海豚酒店（Dolphin Hotel）");
  await expect(details).toContainText("Low back-translation similarity");
  await expect(details).toContainText("0.210");
  await expect(details).toContainText("Low judge scores");
  await expect(details).toContainText("翻译腔明显");
  await expect(details).toContainText("Risk segments");
  await expect(details).toContainText("Ch. 1 ¶4");
  await expect(details).toContainText("Recent evaluation runs");
  await expect(details).toContainText("2026-10-03T11:00:00+08:00");
});

test("evaluation details stay hidden when the report has no evaluation", async ({ page }) => {
  await fakeApi(page, {
    [`/projects/${pid}/report`]: { summary: { chapters_done: 1 } },
  });
  await page.goto(`/projects/${pid}`);

  await expect(page.locator("details", { hasText: "Evaluation details" })).toHaveCount(0);
});

import { expect, test, type Page } from "@playwright/test";
import { fakeApi, pid, configuration, effective } from "./fixtures";

const calibration = [
  {
    key: "judge_score_min",
    suggested_value: 4.2,
    reason: "38% of segments scored low in the last 5 runs; 3.5 is too lenient",
  },
];

const tuning = {
  mode: "auto",
  tier: "standard",
  items: [
    {
      key: "judge_score_min",
      value: 4.2,
      source: "pinned",
      note: "pinned in project settings",
    },
    {
      key: "bt_score_min",
      value: 0.42,
      source: "history",
      note: "p10 of 5 runs, floor 0.35, cap 0.45",
    },
    {
      key: "review_scope",
      value: "risk",
      source: "tier",
      note: "standard tier reviews risk chapters only",
    },
    {
      key: "risk_sample_ratio",
      value: 0.08,
      source: "budget",
      note: "batch budget 1800 tokens",
    },
    { key: "quality_judge_dual", value: false, source: "default" },
  ],
  calibration,
};

const evaluation = {
  l2: { checked: 12, drifted: 2, consistency_rate: 0.83, items: [] },
};

async function openTuningPanel(page: Page) {
  const details = page.locator("details", { hasText: "Evaluation details" });
  await details.locator("summary").first().click();
  const panel = details.locator("details", {
    hasText: "Effective tuning values",
  });
  await panel.locator("summary").click();
  return panel;
}

test("report lists effective tuning values, sources and calibration suggestions", async ({
  page,
}) => {
  await fakeApi(page, {
    [`/projects/${pid}/report`]: {
      summary: { chapters_done: 1 },
      evaluation: { ...evaluation, tuning },
    },
  });
  await page.goto(`/projects/${pid}`);

  const panel = await openTuningPanel(page);
  await expect(panel.locator("summary")).toContainText("Auto · Standard tier");
  await expect(panel).toContainText(
    "The system derived these values from the autonomy tier",
  );
  await expect(panel.locator("thead th")).toHaveText([
    "Key",
    "Effective value",
    "Source",
    "Note",
  ]);
  await expect(panel.locator("tbody tr td:first-child")).toHaveText([
    "judge_score_min",
    "bt_score_min",
    "review_scope",
    "risk_sample_ratio",
    "quality_judge_dual",
  ]);
  await expect(panel.locator("tbody tr").first().locator("td")).toHaveText([
    "judge_score_min",
    "4.2",
    "Pinned by you",
    "pinned in project settings",
  ]);
  await expect(panel).toContainText("0.08");
  await expect(panel).toContainText("false");
  await expect(panel).toContainText("From score history");
  await expect(panel).toContainText("From autonomy tier");
  await expect(panel).toContainText("From batch budget");
  await expect(panel).toContainText("Fixed default");
  await expect(panel).toContainText(
    "Threshold calibration suggested by past runs",
  );
  await expect(panel).toContainText(
    "judge_score_min → 4.2: 38% of segments scored low in the last 5 runs; 3.5 is too lenient",
  );
});

test("manual tuning reports label the mode and drop calibration suggestions", async ({
  page,
}) => {
  await fakeApi(page, {
    [`/projects/${pid}/report`]: {
      summary: { chapters_done: 1 },
      evaluation: {
        ...evaluation,
        tuning: { ...tuning, mode: "manual", calibration: [] },
      },
    },
  });
  await page.goto(`/projects/${pid}`);

  const panel = await openTuningPanel(page);
  await expect(panel.locator("summary")).toContainText("Manual · Standard tier");
  await expect(panel).toContainText("review_scope");
  await expect(panel).not.toContainText(
    "Threshold calibration suggested by past runs",
  );
});

test("report without tuning data renders no tuning block", async ({ page }) => {
  await fakeApi(page, {
    [`/projects/${pid}/report`]: {
      summary: { chapters_done: 1 },
      evaluation,
    },
  });
  await page.goto(`/projects/${pid}`);

  const details = page.locator("details", { hasText: "Evaluation details" });
  await details.locator("summary").first().click();
  await expect(details).toContainText("Term consistency");
  await expect(
    details.locator("summary", { hasText: "Effective tuning values" }),
  ).toHaveCount(0);
});

test("auto tuning disables the managed controls until manual mode is selected", async ({
  page,
}) => {
  const document = {
    ...effective,
    pipeline: {
      ...effective.pipeline,
      tuning: "auto",
      risk_back_translation: true,
      quality_judge: true,
      bt_score_min: 0.42,
      judge_score_min: 4.2,
      l2_min_consistency: 1,
    },
  };
  await fakeApi(page, {
    [`/projects/${pid}/config`]: {
      ...configuration,
      effective: document,
      yaml: JSON.stringify(document),
    },
  });
  await page.goto(`/projects/${pid}/settings`);

  const mode = page.getByLabel("Tuning mode");
  await expect(mode).toHaveValue("auto");
  const riskBackTranslation = page.getByLabel(
    "Risk-gated back-translation (L1)",
  );
  await expect(riskBackTranslation).toBeChecked();
  await expect(riskBackTranslation).toBeDisabled();
  await expect(page.getByText("Managed by the system · enabled")).toHaveCount(
    1,
  );
  // The L3 layer switch is a cost gate the operator keeps, unlike the dual-judge knob.
  await expect(page.getByLabel("Quality judge scores (L3)")).toBeEnabled();
  const btScoreMin = page.getByLabel("Back-translation score min");
  await expect(btScoreMin).toBeDisabled();
  await expect(btScoreMin).toHaveValue("0.42");
  await expect(page.getByText("Managed by the system · 0.42")).toBeVisible();
  await expect(page.getByLabel("Judge score min")).toBeDisabled();
  await expect(page.getByLabel("Term consistency min")).toBeEnabled();
  await expect(
    page.getByLabel("Strict auto QA (block export on residuals)"),
  ).toBeEnabled();
  await expect(page.getByLabel("Autonomy tier")).toBeEnabled();

  await mode.selectOption("manual");
  await expect(riskBackTranslation).toBeEnabled();
  await expect(btScoreMin).toBeEnabled();
  await expect(page.getByText("Managed by the system", { exact: false })).toHaveCount(
    0,
  );
});

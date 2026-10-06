import { expect, test } from "@playwright/test";
import { fakeApi, pid } from "./fixtures";

test("bulk keep current settles every open conflict and refreshes the list", async ({
  page,
}) => {
  await page.addInitScript(() => localStorage.setItem("wenyi.locale", "en"));
  let open = [
    {
      id: 1,
      source: "Zed",
      existing_target: "泽德",
      proposed_target: "泽德二",
      chapter: 1,
    },
    {
      id: 2,
      source: "Amy",
      existing_target: "艾米",
      proposed_target: "艾米二",
      chapter: 2,
    },
  ];
  const methods: string[] = [];
  await fakeApi(page);
  await page.route(`**/api/projects/${pid}/glossary/terms**`, (route) =>
    route.fulfill({ json: [] }),
  );
  await page.route(`**/api/projects/${pid}/glossary/conflicts`, (route) =>
    route.fulfill({ json: open }),
  );
  await page.route(
    `**/api/projects/${pid}/glossary/conflicts/keep-current`,
    async (route) => {
      methods.push(route.request().method());
      const sources = open.map((conflict) => conflict.source);
      open = [];
      await route.fulfill({
        json: { message: "resolved", sources, segments_replaced: 3 },
      });
    },
  );

  await page.goto(`/projects/${pid}/glossary`);
  await expect(page.getByText("Open conflicts (2)")).toBeVisible();
  await expect(page.getByText("Zed").first()).toBeVisible();
  await expect(page.getByText("泽德二")).toBeVisible();

  await page.getByRole("button", { name: "Keep all current" }).click();

  await expect.poll(() => methods).toEqual(["POST"]);
  await expect(
    page.getByText(
      "Kept the current value for 2 term(s) · 3 translation(s) updated",
    ),
  ).toBeVisible();
  await expect(page.getByText("Open conflicts (2)")).toHaveCount(0);
});

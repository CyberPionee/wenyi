import { expect, test } from "@playwright/test";
import { fakeApi, pid } from "./fixtures";

// Native widgets differ per browser, platform and zoom level; the application draws its own
// checkbox and radio geometry, so it must hold at every scale.
for (const theme of ["light", "dark"]) {
  for (const zoom of [1, 1.25]) {
    test(`${theme} controls at ${zoom} zoom keep application geometry and drafts`, async ({
      page,
    }) => {
      await fakeApi(page, {});
      await page.goto(`/projects/${pid}/export`);
      await page.getByRole("radio").first().waitFor();
      await page.evaluate(
        ({ theme, zoom }) => {
          document.documentElement.classList.toggle("dark", theme === "dark");
          document.documentElement.style.zoom = String(zoom);
        },
        { theme, zoom },
      );

      const summary = page.locator("summary").first();
      await expect(summary.locator("svg")).toHaveCount(1);
      await expect(summary).toHaveCSS("list-style-type", "none");
      expect(
        await summary.evaluate(
          (node) => getComputedStyle(node, "::marker").content,
        ),
      ).toBe('""');

      const inputs = page.locator(
        'input[type="radio"], input[type="checkbox"]',
      );
      expect(await inputs.count()).toBeGreaterThan(0);
      for (const input of await inputs.all()) {
        await expect(input).toHaveCSS("appearance", "none");
        await expect(input).toHaveCSS("width", "16px");
        await expect(input).toHaveCSS("height", "16px");
        await expect(input).toHaveCSS("line-height", "16px");
        const geometry = await input.evaluate((node) => {
          const box = node.getBoundingClientRect();
          const label = node.closest("label")!.getBoundingClientRect();
          return {
            size: box.width,
            center: box.y + box.height / 2 - label.y - label.height / 2,
          };
        });
        expect(geometry.size).toBeCloseTo(16 * zoom);
        expect(Math.abs(geometry.center)).toBeLessThan(1);
      }

      // Folding the section must not discard the value a control already holds.
      await summary.click();
      const checkbox = page.locator('input[type="checkbox"]').first();
      await expect(checkbox).toBeVisible();
      await checkbox.check();
      await summary.click();
      await expect(checkbox).toBeAttached();
      await summary.focus();
      await page.keyboard.press("Space");
      await expect(checkbox).toBeVisible();
      await expect(checkbox).toBeChecked();
    });
  }
}

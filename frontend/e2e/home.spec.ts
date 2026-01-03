import { test, expect } from './fixtures';

test.describe('Home Page', () => {
  test('displays the main heading and navigation buttons', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByText('Pussel')).toBeVisible();
    await expect(page.getByText('AI-powered puzzle piece position detection')).toBeVisible();

    await expect(page.getByRole('link', { name: 'New Puzzle' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Test Mode' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'About' })).toBeVisible();
  });

  test('navigates to play page when clicking New Puzzle', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: 'New Puzzle' }).click();

    await expect(page).toHaveURL('/play');
    await expect(page.getByRole('heading', { name: 'Capture Puzzle' })).toBeVisible();
  });

  test('navigates to about page', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('link', { name: 'About' }).click();

    await expect(page).toHaveURL('/about');
  });
});

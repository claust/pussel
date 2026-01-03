import { test, expect } from '@playwright/test';

test.describe('Test Mode - Grid Selection', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/test-mode');
  });

  test('displays test mode page with grid options', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Test Mode' })).toBeVisible();
    await expect(page.getByText('Select the grid size for testing')).toBeVisible();

    // Verify both grid options are displayed
    await expect(page.getByText('2×2 Grid')).toBeVisible();
    await expect(page.getByText('4 pieces total')).toBeVisible();
    await expect(page.getByText('3×3 Grid')).toBeVisible();
    await expect(page.getByText('9 pieces total')).toBeVisible();
  });

  test('back button navigates to home page', async ({ page }) => {
    await page.locator('header button').first().click();
    await expect(page).toHaveURL('/');
  });

  test('selecting 2x2 grid navigates to puzzle selection', async ({ page }) => {
    await page.getByText('2×2 Grid').click();

    await expect(page).toHaveURL('/test-mode/select');
    await expect(page.getByRole('heading', { name: 'Select Puzzle' })).toBeVisible();
    await expect(page.getByText('Grid: 2x2')).toBeVisible();
  });

  test('selecting 3x3 grid navigates to puzzle selection', async ({ page }) => {
    await page.getByText('3×3 Grid').click();

    await expect(page).toHaveURL('/test-mode/select');
    await expect(page.getByRole('heading', { name: 'Select Puzzle' })).toBeVisible();
    await expect(page.getByText('Grid: 3x3')).toBeVisible();
  });
});

test.describe('Test Mode - Puzzle Selection', () => {
  test.beforeEach(async ({ page }) => {
    // Go to test mode and select 2x2 grid first
    await page.goto('/test-mode');
    await page.getByText('2×2 Grid').click();
    await expect(page).toHaveURL('/test-mode/select');
  });

  test('displays all test puzzles', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Select Puzzle' })).toBeVisible();

    // Verify all 10 puzzles are displayed using exact match
    for (let i = 1; i <= 10; i++) {
      await expect(page.getByText(`Puzzle ${i}`, { exact: true })).toBeVisible();
    }
  });

  test('puzzle images are displayed', async ({ page }) => {
    // Check that puzzle images are rendered (each puzzle has an image)
    const puzzleImages = page.locator('img[alt^="Puzzle"]');
    await expect(puzzleImages).toHaveCount(10);
  });

  test('back button navigates to grid selection', async ({ page }) => {
    await page.locator('header button').first().click();
    await expect(page).toHaveURL('/test-mode');
  });

  test('clicking a puzzle navigates to puzzle page', async ({ page }) => {
    await page.getByText('Puzzle 1', { exact: true }).click();
    await expect(page).toHaveURL('/test-mode/puzzle/001');
  });
});

test.describe('Test Mode - Full Flow with Backend', () => {
  // These tests require the backend to be running
  test.beforeAll(async () => {
    // Fail early if backend is not available
    let response: Response;
    try {
      response = await fetch('http://localhost:8000/health');
    } catch {
      throw new Error('Backend is not running. Start it with: make start-backend');
    }
    if (!response.ok) {
      throw new Error(
        `Backend health check failed with status ${response.status}. Ensure backend is running on http://localhost:8000`
      );
    }
  });

  test('loads puzzle and displays UI elements', async ({ page }) => {
    // Navigate through the flow
    await page.goto('/test-mode');
    await page.getByText('2×2 Grid').click();
    await page.getByText('Puzzle 1', { exact: true }).click();

    // Wait for puzzle page to load (URL change + heading visible)
    await expect(page).toHaveURL('/test-mode/puzzle/001');
    await expect(page.getByRole('heading', { name: 'Puzzle 1' })).toBeVisible({
      timeout: 10000,
    });
    await expect(page.getByText('2x2 grid')).toBeVisible();

    // Verify "Add Piece from Grid" button is visible (indicates puzzle is fully loaded)
    await expect(page.getByRole('button', { name: /Add Piece from Grid/i })).toBeVisible({
      timeout: 10000,
    });
  });

  test('shows grid overlay when clicking Add Piece button', async ({ page }) => {
    // Navigate to puzzle
    await page.goto('/test-mode');
    await page.getByText('2×2 Grid').click();
    await page.getByText('Puzzle 1', { exact: true }).click();

    // Wait for puzzle page to load (URL change + Add Piece button visible)
    await expect(page).toHaveURL('/test-mode/puzzle/001');
    await expect(page.getByRole('button', { name: /Add Piece from Grid/i })).toBeVisible({
      timeout: 10000,
    });

    // Click Add Piece button
    await page.getByRole('button', { name: /Add Piece from Grid/i }).click();

    // Verify Cancel Selection button appears
    await expect(page.getByRole('button', { name: /Cancel Selection/i })).toBeVisible();

    // Verify mode toggle buttons are visible (Grid and Realistic)
    await expect(page.getByRole('button', { name: 'Grid' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Realistic' })).toBeVisible();
  });

  test('can cancel piece selection', async ({ page }) => {
    // Navigate to puzzle
    await page.goto('/test-mode');
    await page.getByText('2×2 Grid').click();
    await page.getByText('Puzzle 1', { exact: true }).click();

    // Wait for puzzle page to load (URL change + Add Piece button visible)
    await expect(page).toHaveURL('/test-mode/puzzle/001');
    await expect(page.getByRole('button', { name: /Add Piece from Grid/i })).toBeVisible({
      timeout: 10000,
    });

    // Click Add Piece button
    await page.getByRole('button', { name: /Add Piece from Grid/i }).click();
    await expect(page.getByRole('button', { name: /Cancel Selection/i })).toBeVisible();

    // Click Cancel Selection
    await page.getByRole('button', { name: /Cancel Selection/i }).click();

    // Verify we're back to normal view
    await expect(page.getByRole('button', { name: /Add Piece from Grid/i })).toBeVisible();
  });

  test('back button navigates to puzzle selection', async ({ page }) => {
    // Navigate to puzzle
    await page.goto('/test-mode');
    await page.getByText('2×2 Grid').click();
    await page.getByText('Puzzle 1', { exact: true }).click();

    // Wait for puzzle page to load (URL change + Add Piece button visible)
    await expect(page).toHaveURL('/test-mode/puzzle/001');
    await expect(page.getByRole('button', { name: /Add Piece from Grid/i })).toBeVisible({
      timeout: 10000,
    });

    // Click back button
    await page.locator('header button').first().click();

    // Verify we're back at puzzle selection
    await expect(page).toHaveURL('/test-mode/select');
  });

  test('view mode toggle works', async ({ page }) => {
    // Navigate to puzzle
    await page.goto('/test-mode');
    await page.getByText('2×2 Grid').click();
    await page.getByText('Puzzle 1', { exact: true }).click();

    // Wait for puzzle page to load (URL change + Add Piece button visible)
    await expect(page).toHaveURL('/test-mode/puzzle/001');
    await expect(page.getByRole('button', { name: /Add Piece from Grid/i })).toBeVisible({
      timeout: 10000,
    });

    // Click fullscreen toggle (last button in header)
    const fullscreenButton = page.locator('header button').last();
    await fullscreenButton.click();

    // In fullscreen mode, Add Piece button should not be visible
    await expect(page.getByRole('button', { name: /Add Piece from Grid/i })).not.toBeVisible();

    // Click to exit fullscreen (click on puzzle image - only one img[alt="Puzzle"] should be visible now)
    await page.locator('img[alt="Puzzle"]').click();

    // Add Piece button should be visible again
    await expect(page.getByRole('button', { name: /Add Piece from Grid/i })).toBeVisible();
  });

  test('3x3 grid flow works', async ({ page }) => {
    // Navigate through 3x3 flow
    await page.goto('/test-mode');
    await page.getByText('3×3 Grid').click();
    await page.getByText('Puzzle 2', { exact: true }).click();

    // Wait for puzzle page to load (URL change + heading visible)
    await expect(page).toHaveURL('/test-mode/puzzle/002');
    await expect(page.getByRole('heading', { name: 'Puzzle 2' })).toBeVisible({
      timeout: 10000,
    });
    await expect(page.getByText('3x3 grid')).toBeVisible();

    // Wait for Add Piece button (indicates puzzle is fully loaded)
    await expect(page.getByRole('button', { name: /Add Piece from Grid/i })).toBeVisible({
      timeout: 10000,
    });
  });
});

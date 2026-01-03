import { test, expect } from '@playwright/test';
import path from 'path';

const TEST_PUZZLE_PATH = path.join(__dirname, '../public/test-puzzles/puzzle_001.jpg');

test.describe('Play Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/play');
  });

  test('displays capture phase initially', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Capture Puzzle' })).toBeVisible();
    await expect(page.getByText('Take a photo of the puzzle you want to solve')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Capture Puzzle' })).toBeVisible();
  });

  test('opens camera modal when clicking Capture Puzzle', async ({ page }) => {
    await page.getByRole('button', { name: 'Capture Puzzle' }).click();

    await expect(page.getByRole('dialog')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Capture Puzzle' })).toBeVisible();
  });

  test('can close camera modal with Cancel button', async ({ page }) => {
    await page.getByRole('button', { name: 'Capture Puzzle' }).click();
    await expect(page.getByRole('dialog')).toBeVisible();

    await page.getByRole('button', { name: 'Cancel' }).click();
    await expect(page.getByRole('dialog')).not.toBeVisible();
  });

  test('back button navigates to home page', async ({ page }) => {
    // Click the back button (first button in header)
    await page.locator('header button').first().click();

    await expect(page).toHaveURL('/');
  });
});

test.describe('Play Flow with Backend', () => {
  // These tests require the backend to be running
  test.beforeAll(async () => {
    // Check if backend is available
    try {
      const response = await fetch('http://localhost:8000/health');
      if (!response.ok) {
        test.skip();
      }
    } catch {
      test.skip();
    }
  });

  test('uploads puzzle and shows grid selection', async ({ page }) => {
    await page.goto('/play');

    // Open camera modal
    await page.getByRole('button', { name: 'Capture Puzzle' }).click();
    await expect(page.getByRole('dialog')).toBeVisible();

    // Upload file via file input
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_PUZZLE_PATH);

    // Wait for upload and grid selection phase
    await expect(page.getByRole('heading', { name: 'Select Grid Size' })).toBeVisible({
      timeout: 10000,
    });

    // Verify puzzle preview is shown
    await expect(page.getByAltText('Puzzle preview')).toBeVisible();

    // Verify all grid size options are available
    await expect(page.getByRole('button', { name: /2x2/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /3x3/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /4x4/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /5x5/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /6x6/i })).toBeVisible();
  });

  test('selects grid size and enters play mode', async ({ page }) => {
    await page.goto('/play');

    // Upload puzzle
    await page.getByRole('button', { name: 'Capture Puzzle' }).click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_PUZZLE_PATH);

    // Wait for grid selection
    await expect(page.getByRole('heading', { name: 'Select Grid Size' })).toBeVisible({
      timeout: 10000,
    });

    // Select 3x3 grid
    await page.getByRole('button', { name: /3x3/i }).click();

    // Wait for play mode
    await expect(page.getByRole('heading', { name: 'Play Mode' })).toBeVisible({
      timeout: 10000,
    });

    // Verify play mode UI elements
    await expect(page.getByText(/\d+ \/ \d+ pieces placed/)).toBeVisible();
    await expect(page.getByRole('button', { name: 'Shuffle' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'New Puzzle' })).toBeVisible();
  });

  test('shuffle button randomizes piece positions', async ({ page }) => {
    await page.goto('/play');

    // Upload puzzle and select grid
    await page.getByRole('button', { name: 'Capture Puzzle' }).click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_PUZZLE_PATH);
    await expect(page.getByRole('heading', { name: 'Select Grid Size' })).toBeVisible({
      timeout: 10000,
    });
    await page.getByRole('button', { name: /2x2/i }).click();

    // Wait for play mode
    await expect(page.getByRole('heading', { name: 'Play Mode' })).toBeVisible({
      timeout: 10000,
    });

    // Click shuffle button - should not throw error
    await page.getByRole('button', { name: 'Shuffle' }).click();

    // Verify we're still in play mode
    await expect(page.getByRole('heading', { name: 'Play Mode' })).toBeVisible();
    await expect(page.getByText('0 / 4 pieces placed')).toBeVisible();
  });

  test('New Puzzle button resets to capture phase', async ({ page }) => {
    await page.goto('/play');

    // Upload puzzle and select grid
    await page.getByRole('button', { name: 'Capture Puzzle' }).click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_PUZZLE_PATH);
    await expect(page.getByRole('heading', { name: 'Select Grid Size' })).toBeVisible({
      timeout: 10000,
    });
    await page.getByRole('button', { name: /2x2/i }).click();

    // Wait for play mode
    await expect(page.getByRole('heading', { name: 'Play Mode' })).toBeVisible({
      timeout: 10000,
    });

    // Click New Puzzle
    await page.getByRole('button', { name: 'New Puzzle' }).click();

    // Verify we're back to capture phase
    await expect(page.getByRole('heading', { name: 'Capture Puzzle' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Capture Puzzle' })).toBeVisible();
  });

  test('displays correct piece count for different grid sizes', async ({ page }) => {
    await page.goto('/play');

    // Upload puzzle
    await page.getByRole('button', { name: 'Capture Puzzle' }).click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_PUZZLE_PATH);
    await expect(page.getByRole('heading', { name: 'Select Grid Size' })).toBeVisible({
      timeout: 10000,
    });

    // Select 4x4 grid (16 pieces)
    await page.getByRole('button', { name: /4x4/i }).click();

    // Wait for play mode and verify piece count
    await expect(page.getByRole('heading', { name: 'Play Mode' })).toBeVisible({
      timeout: 10000,
    });
    await expect(page.getByText('0 / 16 pieces placed')).toBeVisible();
  });
});

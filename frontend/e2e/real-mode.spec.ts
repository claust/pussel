import { test, expect } from './fixtures';
import path from 'path';

const TEST_PUZZLE_PATH = path.join(__dirname, '../public/test-puzzles/puzzle_001.jpg');
// Frame detection may fall back to rembg segmentation, which is slow on the
// first request after backend startup (model download + session warm-up in CI)
const API_TIMEOUT = 40000;
// Piece prediction runs background removal + CNN inference and may download models on first run
const PIECE_TIMEOUT = 60000;

test.describe('Real Mode Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/real');
  });

  test('displays capture phase initially', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Solve Real Puzzle' })).toBeVisible();
    await expect(page.getByText('Capture Your Puzzle')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Take Puzzle Photo' })).toBeVisible();
  });

  test('opens camera modal when clicking Take Puzzle Photo', async ({ page }) => {
    await page.getByRole('button', { name: 'Take Puzzle Photo' }).click();

    await expect(page.getByRole('dialog')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Capture Puzzle' })).toBeVisible();
  });

  test('can close camera modal with Cancel button', async ({ page }) => {
    await page.getByRole('button', { name: 'Take Puzzle Photo' }).click();
    await expect(page.getByRole('dialog')).toBeVisible();

    await page.getByRole('button', { name: 'Cancel' }).click();
    await expect(page.getByRole('dialog')).not.toBeVisible();
  });

  test('back button navigates to home page', async ({ page }) => {
    await page.locator('header a').first().click();

    await expect(page).toHaveURL('/');
  });
});

test.describe('Real Mode Flow with Backend', () => {
  // These flows hit rembg segmentation, which downloads its model on the first
  // request in a fresh CI environment. Allow well beyond the default 30s per-test
  // timeout so the assertion timeouts below (which include that download) can elapse.
  test.describe.configure({ timeout: 120000 });

  // These tests require the backend to be running and TEST_AUTH_TOKEN to be set
  test.beforeAll(async () => {
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

    if (!process.env.TEST_AUTH_TOKEN) {
      console.warn(
        'WARNING: TEST_AUTH_TOKEN not set. API requests may fail with 401. ' +
          'Generate token with: python backend/scripts/generate_test_token.py'
      );
    }
  });

  test('detects puzzle frame and shows trim confirmation', async ({ page }) => {
    await page.goto('/real');

    await page.getByRole('button', { name: 'Take Puzzle Photo' }).click();
    await expect(page.getByRole('dialog')).toBeVisible();

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_PUZZLE_PATH);

    // Wait for detection result
    await expect(page.getByRole('heading', { name: 'Is this your puzzle?' })).toBeVisible({
      timeout: API_TIMEOUT,
    });

    await expect(page.getByAltText('Trimmed puzzle')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Use This' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Adjust Corners' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Retake' })).toBeVisible();
  });

  test('adjust corners shows draggable handles and returns to confirmation', async ({ page }) => {
    await page.goto('/real');

    await page.getByRole('button', { name: 'Take Puzzle Photo' }).click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_PUZZLE_PATH);
    await expect(page.getByRole('heading', { name: 'Is this your puzzle?' })).toBeVisible({
      timeout: API_TIMEOUT,
    });

    await page.getByRole('button', { name: 'Adjust Corners' }).click();

    // All four handles are visible
    await expect(page.getByTestId('corner-handle-topLeft')).toBeVisible();
    await expect(page.getByTestId('corner-handle-topRight')).toBeVisible();
    await expect(page.getByTestId('corner-handle-bottomRight')).toBeVisible();
    await expect(page.getByTestId('corner-handle-bottomLeft')).toBeVisible();

    // Apply re-runs the trim with manual corners and returns to confirmation
    await page.getByRole('button', { name: 'Apply' }).click();
    await expect(page.getByRole('heading', { name: 'Is this your puzzle?' })).toBeVisible({
      timeout: API_TIMEOUT,
    });
  });

  test('accepts trim and runs a piece through the capture queue', async ({ page }) => {
    await page.goto('/real');

    // Photograph the puzzle
    await page.getByRole('button', { name: 'Take Puzzle Photo' }).click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_PUZZLE_PATH);
    await expect(page.getByRole('heading', { name: 'Is this your puzzle?' })).toBeVisible({
      timeout: API_TIMEOUT,
    });

    // Accept the trimmed image (uploads it as the puzzle); the pipeline view
    // appears: live capture area (or its no-camera fallback) plus empty queue
    await page.getByRole('button', { name: 'Use This' }).click();
    await expect(
      page.getByTestId('live-capture').or(page.getByTestId('live-capture-fallback'))
    ).toBeVisible({ timeout: API_TIMEOUT });
    await expect(page.getByText('0 pieces captured')).toBeVisible();
    await expect(page.getByTestId('piece-queue-empty')).toBeVisible();

    // Add a piece via upload (any image works; the backend predicts regardless).
    // With no camera in the test environment this exercises the fallback path;
    // either way the upload is committed straight into the capture queue.
    await page.locator('input[type="file"]').setInputFiles(TEST_PUZZLE_PATH);

    // The capture is committed to the queue immediately...
    await expect(page.getByText('1 piece captured')).toBeVisible();
    await expect(page.getByTestId('piece-queue')).toBeVisible();

    // ...and the worker drains it through prediction to done
    await expect(page.getByTestId('queue-entry-done')).toBeVisible({ timeout: PIECE_TIMEOUT });

    // Deleting the piece removes it from the queue entirely
    await page.getByTestId('queue-entry-delete').click();
    await expect(page.getByTestId('piece-queue-empty')).toBeVisible();
    await expect(page.getByText('0 pieces captured')).toBeVisible();
  });
});

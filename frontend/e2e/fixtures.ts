import { test as base } from '@playwright/test';

/**
 * Extended test fixture that injects auth token into API requests.
 *
 * When TEST_AUTH_TOKEN env var is set, this fixture intercepts all
 * requests to the backend API and adds the Authorization header.
 */
export const test = base.extend({
  page: async ({ page }, use) => {
    const authToken = process.env.TEST_AUTH_TOKEN;

    if (authToken) {
      // Intercept all requests to the backend API and add auth header
      await page.route('**/api/v1/**', async (route) => {
        const headers = {
          ...route.request().headers(),
          Authorization: `Bearer ${authToken}`,
        };
        await route.continue({ headers });
      });
    }

    await use(page);
  },
});

export { expect } from '@playwright/test';

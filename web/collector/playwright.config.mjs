import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests/browser', timeout: 150000, workers: 1, retries: 0,
  use: { baseURL: 'http://127.0.0.1:4173', headless: true, viewport: { width: 1440, height: 1000 },
    launchOptions: { args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'] },
    permissions: ['microphone'], trace: 'retain-on-failure'
  },
  webServer: { command: 'npm run dev', url: 'http://127.0.0.1:4173', reuseExistingServer: true }
});

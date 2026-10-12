/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The UI is served by the API under /app. In development, Vite proxies API
// paths to a local API so the browser sees a single origin and no CORS setup
// is needed.
const api = process.env.INCIDENTIQ_API ?? "http://localhost:8000";

export default defineConfig({
  base: "/app/",
  plugins: [react()],
  server: {
    proxy: {
      "/investigations": api,
      "/approvals": api,
      "/me": api,
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
  },
});

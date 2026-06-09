import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:9000",
      "/reports": "http://127.0.0.1:9000",
      "/eval-output": "http://127.0.0.1:9000"
    }
  },
  test: {
    environment: "jsdom",
    setupFiles: ["src/test/setup.ts"],
    globals: true
  }
});

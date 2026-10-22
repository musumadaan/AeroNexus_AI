import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const api = process.env.VITE_API_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: { port: 3000, proxy: { "/graphql": api, "/metrics": api, "/upload": api } },
  test: { environment: "node" },
});

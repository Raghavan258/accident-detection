import { fileURLToPath, URL } from "node:url"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vitest/config"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/mlops": "http://127.0.0.1:8000",
      "/predict": {
        target: "http://127.0.0.1:8000",
        bypass(req) {
          const path = req.url?.split("?")[0] ?? ""
          if (path === "/predict" || path.startsWith("/predict/")) return undefined
          return "/index.html"
        },
      },
      "/health": "http://127.0.0.1:8000",
      "/synthetic": "http://127.0.0.1:8000",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
  },
})

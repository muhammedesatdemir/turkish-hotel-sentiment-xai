import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
  },
  build: {
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks: {
          "react-vendor":     ["react", "react-dom", "react-router-dom"],
          "query-vendor":     ["@tanstack/react-query"],
          "recharts-vendor":  ["recharts"],
          "ui-vendor":        ["lucide-react", "react-hot-toast", "react-dropzone"],
        },
      },
    },
  },
});

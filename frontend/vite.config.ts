import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The API is proxied in development so the frontend can use relative /api paths
// in both dev and production, and so no CORS preflight is involved locally.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})

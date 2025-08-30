import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/redact': 'http://localhost:8000',
      '/proxy': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/ner_status': 'http://localhost:8000'
    }
  }
})

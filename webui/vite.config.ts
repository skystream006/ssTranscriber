import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '..', '')
  const apiPort = env.WEB_API_PORT || '8000'

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': `http://127.0.0.1:${apiPort}`,
      },
    },
  }
})
// @ts-check
import { defineConfig } from 'astro/config';

import mdx from '@astrojs/mdx';

// https://astro.build/config
export default defineConfig({
  site: 'https://herodthegreater.github.io',
  base: '/pigeon',
  integrations: [mdx()],
  vite: {
    optimizeDeps: { exclude: ['onnxruntime-web'] },
    worker: { format: 'es' },
  }
});
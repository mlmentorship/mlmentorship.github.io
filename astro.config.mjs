import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import mermaid from 'astro-mermaid';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const prepToolsEnabled = process.env.PUBLIC_PREP_TOOLS === undefined || process.env.PUBLIC_PREP_TOOLS === 'true';

// https://astro.build/config
export default defineConfig({
  site: 'https://mlmentorship.com',
  redirects: {
    '/readiness': prepToolsEnabled ? '/prep/readiness' : '/questions',
    '/practice': prepToolsEnabled ? '/prep/practice' : '/questions',
    '/story-bank': prepToolsEnabled ? '/prep/story-bank' : '/questions',
    '/plans': prepToolsEnabled ? '/prep/plans' : '/questions',
    '/final-week': prepToolsEnabled ? '/prep/final-week' : '/questions',
  },
  integrations: [
    mermaid({ theme: 'neutral', autoTheme: true, enableLog: false }),
    mdx(),
    sitemap(),
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, { output: 'html' }]],
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
      wrap: true,
    },
  },
  build: {
    format: 'directory',
  },
  prefetch: {
    prefetchAll: true,
    defaultStrategy: 'viewport',
  },
});

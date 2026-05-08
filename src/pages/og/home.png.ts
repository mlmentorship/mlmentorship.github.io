import type { APIRoute } from 'astro';
import satori from 'satori';
import { Resvg } from '@resvg/resvg-js';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const fontBold = readFileSync(resolve(process.cwd(), 'src/og-fonts/InterTight-Bold.ttf'));
const fontRegular = readFileSync(resolve(process.cwd(), 'src/og-fonts/Inter-Regular.ttf'));

export const GET: APIRoute = async () => {
  const svg = await satori(
    {
      type: 'div',
      props: {
        style: {
          width: '1200px',
          height: '630px',
          background: '#ffffff',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '60px 70px',
          fontFamily: 'Inter',
          color: '#0f172a',
        },
        children: [
          {
            type: 'div',
            props: {
              style: { display: 'flex' },
              children: [
                {
                  type: 'div',
                  props: {
                    style: {
                      display: 'flex',
                      alignItems: 'center',
                      background: '#fed7aa',
                      color: '#c2410c',
                      fontSize: '20px',
                      fontWeight: 600,
                      letterSpacing: '0.12em',
                      textTransform: 'uppercase',
                      padding: '6px 14px',
                      borderRadius: '999px',
                    },
                    children: 'mlmentorship.com',
                  },
                },
              ],
            },
          },
          {
            type: 'div',
            props: {
              style: {
                display: 'flex',
                flexDirection: 'column',
                gap: '18px',
              },
              children: [
                {
                  type: 'div',
                  props: {
                    style: {
                      fontFamily: 'Inter Tight',
                      fontWeight: 700,
                      fontSize: '64px',
                      lineHeight: 1.1,
                      letterSpacing: '-0.025em',
                      color: '#0f172a',
                      maxWidth: '1060px',
                    },
                    children: 'Senior ML interviews, calibrated.',
                  },
                },
                {
                  type: 'div',
                  props: {
                    style: {
                      fontSize: '28px',
                      lineHeight: 1.35,
                      color: '#475569',
                      maxWidth: '1000px',
                    },
                    children:
                      'Questions, guides, and concept notes for L5+ Applied Scientist, MLE, and Research Engineer loops.',
                  },
                },
              ],
            },
          },
          {
            type: 'div',
            props: {
              style: {
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                borderTop: '1px solid #e2e8f0',
                paddingTop: '20px',
              },
              children: [
                {
                  type: 'div',
                  props: {
                    style: { display: 'flex', alignItems: 'baseline' },
                    children: [
                      { type: 'span', props: { style: { fontSize: '26px', fontWeight: 700, color: '#0f172a' }, children: 'ml' } },
                      { type: 'span', props: { style: { fontSize: '26px', fontWeight: 400, color: '#64748b' }, children: 'mentorship' } },
                    ],
                  },
                },
                {
                  type: 'div',
                  props: { style: { fontSize: '20px', color: '#64748b' }, children: 'by Hamid Saghir' },
                },
              ],
            },
          },
        ],
      },
    },
    {
      width: 1200,
      height: 630,
      fonts: [
        { name: 'Inter Tight', data: fontBold, weight: 700, style: 'normal' },
        { name: 'Inter', data: fontRegular, weight: 400, style: 'normal' },
        { name: 'Inter', data: fontBold, weight: 700, style: 'normal' },
      ],
    }
  );

  const png = new Resvg(svg).render().asPng();
  return new Response(png, {
    headers: {
      'Content-Type': 'image/png',
      'Cache-Control': 'public, max-age=31536000, immutable',
    },
  });
};

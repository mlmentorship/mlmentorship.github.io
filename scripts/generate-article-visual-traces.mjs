import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const auditsDir = path.join(root, 'data/visual-audits');
const outputPath = path.join(root, 'src/utils/articleVisualTraceFallbacks.ts');
const explicitSlugs = new Set([
  'tokenization',
  'k-means-clustering',
  'backpropagation',
  'pipeline-parallelism',
  'continuous-batching',
  'speculative-decoding',
  'activation-functions',
  'attention-mechanism',
  'batchnorm-vs-layernorm',
  'svd-and-pca',
  'roc-pr-auc',
  'calibration',
]);

const entityMap = new Map([
  ['amp', '&'],
  ['lt', '<'],
  ['gt', '>'],
  ['quot', '"'],
  ['apos', "'"],
  ['times', 'x'],
  ['minus', '-'],
  ['ndash', '-'],
  ['mdash', '-'],
  ['hellip', '...'],
  ['rarr', '->'],
  ['larr', '<-'],
  ['middot', '-'],
]);

function normalizeAscii(value) {
  return String(value)
    .replace(/[\u2018\u2019\u201a\u201b]/g, "'")
    .replace(/[\u201c\u201d\u201e\u201f]/g, '"')
    .replace(/[\u2013\u2014]/g, '-')
    .replace(/[\u00a0]/g, ' ')
    .replace(/[\u00d7]/g, 'x')
    .replace(/[\u2212]/g, '-')
    .replace(/[\u2192]/g, '->')
    .replace(/[\u2190]/g, '<-')
    .replace(/[\u2264]/g, '<=')
    .replace(/[\u2265]/g, '>=')
    .replace(/[\u03b2]/g, 'beta')
    .replace(/[\u03b3]/g, 'gamma')
    .replace(/[\u03b4]/g, 'delta')
    .replace(/[\u03b8]/g, 'theta')
    .replace(/[\u03bb]/g, 'lambda')
    .replace(/[\u03b7]/g, 'eta')
    .replace(/[\u03c1]/g, 'rho')
    .replace(/[\u03c4]/g, 'tau')
    .replace(/[\u03c9]/g, 'omega')
    .replace(/[\u03b5]/g, 'epsilon')
    .replace(/[\u039b]/g, 'Lambda')
    .replace(/[\u2211]/g, 'sum')
    .replace(/[\u221a]/g, 'sqrt')
    .replace(/[\u221e]/g, 'infinity')
    .replace(/[\u2205]/g, 'empty')
    .replace(/[\u2207]/g, 'grad')
    .replace(/[\u2229]/g, ' intersection ')
    .replace(/[\u00f7]/g, '/')
    .replace(/[\u00b1]/g, '+/-')
    .replace(/[\u2032]/g, "'")
    .replace(/[\u2081]/g, '1')
    .replace(/[\u2070\u2080]/g, '0')
    .replace(/[\u00b2\u2082]/g, '2')
    .replace(/[\u00b3\u2083]/g, '3')
    .replace(/[\u2074\u2084]/g, '4')
    .replace(/[\u2075\u2085]/g, '5')
    .replace(/[\u2076\u2086]/g, '6')
    .replace(/[\u2077\u2087]/g, '7')
    .replace(/[\u2078\u2088]/g, '8')
    .replace(/[\u2079\u2089]/g, '9')
    .replace(/[\u1d62]/g, 'i')
    .replace(/[\u2248]/g, '~')
    .replace(/[\u2202]/g, 'd')
    .replace(/[\u00b7]/g, ' - ')
    .replace(/[\u03b1]/g, 'alpha')
    .replace(/[\u03bc]/g, 'mu')
    .replace(/[\u03c3]/g, 'sigma')
    .replace(/[\u03c6]/g, 'phi')
    .replace(/&([a-z]+);/gi, (match, name) => entityMap.get(name.toLowerCase()) ?? match)
    .replace(/&#x([0-9a-f]+);/gi, (_match, code) => String.fromCodePoint(Number.parseInt(code, 16)))
    .replace(/&#(\d+);/g, (_match, code) => String.fromCodePoint(Number(code)));
}

function clean(value) {
  return normalizeAscii(value)
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/[`*_#]/g, '')
    .replace(/\s+/g, ' ')
    .replace(/\bnot\s+just\b/gi, 'only')
    .replace(/\bnot\s+merely\b/gi, 'only')
    .replace(/\bhere(?:'s| is)\b/gi, '')
    .replace(/\bdelve\b/gi, 'study')
    .replace(/\butilize\b/gi, 'use')
    .replace(/\bseam\b/gi, 'boundary')
    .trim();
}

function truncate(value, limit = 240) {
  const text = clean(value);
  if (text.length <= limit) return text;
  const shortened = text.slice(0, limit - 3).replace(/\s+\S*$/, '');
  return `${shortened}...`;
}

function splitSentences(value) {
  const text = clean(value).replace(/(\d)\.(\d)/g, '$1__DECIMAL__$2');
  return (text.match(/[^.!?]+(?:[.!?]+|$)/g) ?? [text])
    .map((part) => part.replaceAll('__DECIMAL__', '.'))
    .map((part) => truncate(part))
    .filter((part) => part.length >= 24);
}

function sectionAfterMarker(source, visualId) {
  const marker = `<!-- visual:${visualId} -->`;
  const markerIndex = source.indexOf(marker);
  if (markerIndex < 0) return '';
  const nextHeading = source.indexOf('\n## ', markerIndex + marker.length);
  return source.slice(markerIndex + marker.length, nextHeading < 0 ? source.length : nextHeading);
}

function figureTitle(section) {
  const visualTitle = section.match(/<[^>]+class=["'][^"']*\bvisual-title\b[^"']*["'][^>]*>([\s\S]*?)<\/[^>]+>/i)?.[1];
  if (visualTitle) return truncate(visualTitle, 150);
  const svgTitle = section.match(/<title\b[^>]*>([\s\S]*?)<\/title>/i)?.[1];
  if (svgTitle) return truncate(svgTitle, 150);
  const mermaidTitle = section.match(/\baccTitle:\s*(.+)/i)?.[1];
  return mermaidTitle ? truncate(mermaidTitle, 150) : '';
}

function sectionEvidence(section, notes) {
  const values = [];
  const add = (value) => {
    for (const sentence of splitSentences(value)) {
      if (/^(read it this way|original |the (visual|figure|shipped|site|new) is |no (paper|source) |source artwork|licens|neither source |the hardest prose-only mental model)/i.test(sentence)) continue;
      if (/(?:reuse rights|permissive figure|source figure copied|no paper figure|public availability|reuse permission|public access)/i.test(sentence)) continue;
      if (!values.some((existing) => existing.toLowerCase() === sentence.toLowerCase())) values.push(sentence);
    }
  };

  const elements = [...section.matchAll(/<(title|desc|h[1-6]|figcaption|p)\b[^>]*>([\s\S]*?)<\/\1>/gi)];
  for (const element of elements) add(element[2]);
  const mermaidBlock = section.match(/```mermaid\s*([\s\S]*?)```/i)?.[1];
  if (mermaidBlock) {
    add(mermaidBlock.match(/\baccTitle:\s*(.+)/i)?.[1] ?? '');
    add(mermaidBlock.match(/\baccDescr:\s*([\s\S]*?)(?=\n\s*(?:subgraph\b|class\b|end\b|[A-Za-z_][\w-]*\s*(?:\[|\(|\{|--|==))|$)/i)?.[1] ?? '');
  }
  add(notes);
  return values;
}

function stage(key, label, value, tone) {
  return { key, label, value: truncate(value), tone };
}

function makeDefinition(audit, source) {
  const slug = audit.slug;
  const visualId = audit.implementation.visualIds[0];
  const section = sectionAfterMarker(source, visualId);
  const title = figureTitle(section) || audit.learningObjective;
  const evidence = sectionEvidence(section, audit.sourceReview?.notes ?? '');
  const facts = [...evidence, audit.learningObjective].filter(Boolean).slice(0, 8);
  while (facts.length < 5) facts.push(audit.learningObjective);
  const objective = truncate(audit.learningObjective, 240);
  const traceTitle = truncate(title, 150);
  return {
    slug,
    visualId,
    title: traceTitle,
    objective,
    example: truncate(facts[0]),
    traceKind: 'evidence',
    frames: [
      {
        key: 'observe',
        label: 'Observe the starting evidence',
        note: truncate(facts[0]),
        scene: {
          type: 'evidence',
          ariaLabel: `${traceTitle}: starting evidence`,
          stages: [
            stage(`${slug}-input`, 'input', facts[0], 'input'),
            stage(`${slug}-question`, 'question', objective, 'neutral'),
            stage(`${slug}-context`, 'context', facts[1], 'state'),
          ],
          annotations: [],
        },
      },
      {
        key: 'mechanism',
        label: 'Follow the mechanism',
        note: truncate(facts[2]),
        scene: {
          type: 'evidence',
          ariaLabel: `${traceTitle}: mechanism evidence`,
          stages: [
            stage(`${slug}-input`, 'state', facts[0], 'input'),
            stage(`${slug}-mechanism`, 'mechanism', facts[2], 'focus'),
            stage(`${slug}-transition`, 'transition', facts[3], 'state'),
          ],
          annotations: [],
        },
      },
      {
        key: 'consequence',
        label: 'Read the consequence',
        note: truncate(facts[4]),
        scene: {
          type: 'evidence',
          ariaLabel: `${traceTitle}: consequence and boundary evidence`,
          stages: [
            stage(`${slug}-mechanism`, 'mechanism', facts[2], 'focus'),
            stage(`${slug}-result`, 'result', facts[4], 'output'),
            stage(`${slug}-boundary`, 'boundary', facts[5], 'warning'),
          ],
          annotations: [],
        },
      },
    ],
    review: {
      recognitionCue: objective,
      invariant: truncate(facts[1]),
      transferLesson: truncate(facts.at(-1) ?? objective),
    },
  };
}

const fallbacks = {};
for (const filename of fs.readdirSync(auditsDir).filter((name) => name.endsWith('.json')).sort()) {
  const audit = JSON.parse(fs.readFileSync(path.join(auditsDir, filename), 'utf8'));
  if (audit.status !== 'implemented' || explicitSlugs.has(audit.slug) || !audit.implementation?.visualIds?.[0]) continue;
  const source = fs.readFileSync(path.join(root, audit.article), 'utf8');
  if (source.includes('data-coding-visual')) continue;
  fallbacks[audit.slug] = makeDefinition(audit, source);
}

const output = `export const articleVisualTraceFallbacks = ${JSON.stringify(fallbacks, null, 2)} as const;\n`;
fs.writeFileSync(outputPath, output);
console.log(`Generated ${Object.keys(fallbacks).length} article visual trace fallbacks.`);
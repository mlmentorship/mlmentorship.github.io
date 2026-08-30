import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { basename, join, relative, resolve } from 'node:path';
import process from 'node:process';

const root = resolve(import.meta.dirname, '..');
const postsDir = join(root, 'src/content/posts');
const auditsDir = join(root, 'data/visual-audits');
const allowedStatuses = new Set(['planned', 'implemented', 'no-visual']);
const allowedMedia = new Set(['mermaid', 'svg', 'semantic-html', 'interaction', 'paper-figure', 'mixed', 'none']);
const urlPattern = /^https:\/\//;
const prioritySlugs = [
  'causal-inference-for-ml-decisions',
  'confusion-matrix-and-classification-metrics',
  'roc-pr-auc',
  'backpropagation',
  'batchnorm-vs-layernorm',
  'svd-and-pca',
  'decision-thresholds-asymmetric-costs-abstention',
  'data-leakage-point-in-time-correctness',
  'calibration',
  'activation-functions',
  'multi-head-attention',
  'flashattention',
];

function fail(message) {
  console.error(`Visual coverage check failed: ${message}`);
  process.exitCode = 1;
}

function frontmatter(source, file) {
  const match = source.match(/^---\n([\s\S]*?)\n---/);
  if (!match) throw new Error(`Missing frontmatter in ${file}`);
  const data = {};
  for (const line of match[1].split('\n')) {
    const separator = line.indexOf(':');
    if (separator < 0) continue;
    const key = line.slice(0, separator).trim();
    const raw = line.slice(separator + 1).trim();
    if (key === 'title' || key === 'category' || key === 'draft') {
      data[key] = raw.replace(/^['"]|['"]$/g, '');
    }
  }
  return data;
}

function publishedEntries() {
  return readdirSync(postsDir)
    .filter((name) => /\.mdx?$/.test(name))
    .map((name) => {
      const path = join(postsDir, name);
      const source = readFileSync(path, 'utf8');
      const data = frontmatter(source, name);
      return {
        slug: name.replace(/\.mdx?$/, '').replace(/^\d{4}-\d{2}-\d{2}-/, ''),
        title: data.title,
        category: data.category,
        draft: data.draft === 'true',
        file: relative(root, path).replaceAll('\\', '/'),
        source,
      };
    })
    .filter((entry) => !entry.draft)
    .sort((a, b) => {
      const aRank = prioritySlugs.indexOf(a.slug);
      const bRank = prioritySlugs.indexOf(b.slug);
      if (aRank !== -1 || bRank !== -1) {
        if (aRank === -1) return 1;
        if (bRank === -1) return -1;
        return aRank - bRank;
      }
      return a.slug.localeCompare(b.slug);
    });
}

function auditFiles() {
  if (!existsSync(auditsDir)) return [];
  return readdirSync(auditsDir).filter((name) => name.endsWith('.json')).sort();
}

function nonEmpty(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function validateAudit(audit, file, entry) {
  const label = `${file} (${entry.slug})`;
  if (audit.schemaVersion !== 1) fail(`${label} must use schemaVersion 1`);
  if (audit.slug !== entry.slug) fail(`${label} slug does not match its filename and article`);
  if (audit.article !== entry.file) fail(`${label} article path must be ${entry.file}`);
  if (!allowedStatuses.has(audit.status)) fail(`${label} has unknown status ${JSON.stringify(audit.status)}`);
  if (!allowedMedia.has(audit.medium)) fail(`${label} has unknown medium ${JSON.stringify(audit.medium)}`);
  if (!nonEmpty(audit.learningObjective)) fail(`${label} needs one learningObjective`);
  if (!nonEmpty(audit.mediumRationale)) fail(`${label} needs a mediumRationale`);
  if (!nonEmpty(audit.deckReview?.notes)) fail(`${label} needs deckReview.notes, even when no relevant slide exists`);
  if (!Array.isArray(audit.deckReview?.pages)) fail(`${label} deckReview.pages must be an array`);
  for (const page of audit.deckReview?.pages ?? []) {
    if (!Number.isInteger(page.page) || page.page < 1 || page.page > 214 || !nonEmpty(page.insight)) {
      fail(`${label} has an invalid deck page reference`);
    }
  }
  if (!nonEmpty(audit.sourceReview?.notes)) fail(`${label} needs sourceReview.notes`);
  if (!Array.isArray(audit.sourceReview?.sources)) fail(`${label} sourceReview.sources must be an array`);
  for (const source of audit.sourceReview?.sources ?? []) {
    if (!nonEmpty(source.title) || !urlPattern.test(source.url ?? '') || !nonEmpty(source.usage) || !nonEmpty(source.license)) {
      fail(`${label} has an incomplete source reference`);
    }
  }
  if (!nonEmpty(audit.agentReview?.reviewer) || !/^\d{4}-\d{2}-\d{2}$/.test(audit.agentReview?.reviewedAt ?? '')) {
    fail(`${label} needs an agent reviewer and ISO review date`);
  }
  if (!nonEmpty(audit.agentReview?.summary)) fail(`${label} needs an agentReview.summary`);

  if (audit.status === 'no-visual' && audit.medium !== 'none') fail(`${label} no-visual status requires medium none`);
  if (audit.status !== 'no-visual' && audit.medium === 'none') fail(`${label} medium none requires no-visual status`);

  const visualIds = audit.implementation?.visualIds;
  if (!Array.isArray(visualIds)) fail(`${label} implementation.visualIds must be an array`);
  if (audit.status === 'implemented' && visualIds?.length === 0) fail(`${label} implemented status needs a visual ID`);
  if (audit.status !== 'implemented' && visualIds?.length > 0) fail(`${label} non-implemented status cannot claim visual IDs`);
  for (const id of visualIds ?? []) {
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id)) fail(`${label} has invalid visual ID ${JSON.stringify(id)}`);
    const marker = `<!-- visual:${id} -->`;
    const markerCount = entry.source.split(marker).length - 1;
    if (markerCount !== 1) fail(`${label} needs exactly one article marker for ${id}, found ${markerCount}`);
  }

  if (audit.status === 'implemented' && !nonEmpty(audit.implementation?.accessibility)) {
    fail(`${label} needs an implementation.accessibility note`);
  }
  if (audit.status === 'implemented' && audit.medium === 'mermaid') {
    for (const id of visualIds ?? []) {
      const marker = entry.source.indexOf(`<!-- visual:${id} -->`);
      const blockStart = entry.source.indexOf('```mermaid', marker);
      const blockEnd = entry.source.indexOf('```', blockStart + 10);
      const block = entry.source.slice(blockStart, blockEnd);
      if (blockStart < 0 || !block.includes('accTitle:') || !block.includes('accDescr:')) {
        fail(`${label} Mermaid visual ${id} needs accTitle and accDescr`);
      }
      const after = entry.source.slice(blockEnd, blockEnd + 700);
      if (!after.includes('<p class="diagram-caption"><strong>Read it this way:</strong>')) {
        fail(`${label} Mermaid visual ${id} needs a direct "Read it this way" caption`);
      }
    }
  }
  if (audit.status === 'implemented' && audit.medium === 'svg') {
    for (const id of visualIds ?? []) {
      const marker = entry.source.indexOf(`<!-- visual:${id} -->`);
      const figureEnd = entry.source.indexOf('</figure>', marker);
      const figure = entry.source.slice(marker, figureEnd + 9);
      if (/\n\s*\n/.test(figure)) {
        fail(`${label} SVG visual ${id} must be one uninterrupted Markdown HTML block`);
      }
      if (figureEnd < 0 || !figure.includes('<svg') || !figure.includes('role="img"') || !figure.includes('<title') || !figure.includes('<desc') || !figure.includes('<figcaption>')) {
        fail(`${label} SVG visual ${id} needs a figure, role, title, description, and caption`);
      }
      const svgTags = figure.match(/<svg\b[^>]*>/g) ?? [];
      if (svgTags.length === 0 || svgTags.some((tag) => !tag.includes('viewBox='))) {
        fail(`${label} SVG visual ${id} needs a viewBox on every SVG for responsive rendering`);
      }
      if (!figure.includes('<figcaption><strong>Read it this way:</strong>')) {
        fail(`${label} SVG visual ${id} needs a direct "Read it this way" caption`);
      }
    }
  }
  if (audit.status === 'implemented' && audit.medium === 'semantic-html') {
    for (const id of visualIds ?? []) {
      const marker = entry.source.indexOf(`<!-- visual:${id} -->`);
      const figureEnd = entry.source.indexOf('</figure>', marker);
      const figure = entry.source.slice(marker, figureEnd + 9);
      if (/\n\s*\n/.test(figure)) {
        fail(`${label} semantic visual ${id} must be one uninterrupted Markdown HTML block`);
      }
      if (figureEnd < 0 || !figure.includes('<figure') || !figure.includes('aria-labelledby=') || !figure.includes('aria-label=') || !figure.includes('<figcaption>')) {
        fail(`${label} semantic visual ${id} needs a labelled figure, accessible child, and caption`);
      }
      if (!figure.includes('<figcaption><strong>Read it this way:</strong>')) {
        fail(`${label} semantic visual ${id} needs a direct "Read it this way" caption`);
      }
    }
  }
}

const entries = publishedEntries();
const bySlug = new Map(entries.map((entry) => [entry.slug, entry]));
const audits = new Map();
const claimedVisualIds = new Map();

for (const file of auditFiles()) {
  const slug = basename(file, '.json');
  const entry = bySlug.get(slug);
  if (!entry) {
    fail(`${file} does not match a published article`);
    continue;
  }
  let audit;
  try {
    audit = JSON.parse(readFileSync(join(auditsDir, file), 'utf8'));
  } catch (error) {
    fail(`${file} is not valid JSON: ${error.message}`);
    continue;
  }
  audits.set(slug, audit);
  validateAudit(audit, file, entry);
  for (const id of audit.implementation?.visualIds ?? []) {
    const owner = claimedVisualIds.get(id);
    if (owner) fail(`${file} duplicates visual ID ${id} already claimed by ${owner}`);
    claimedVisualIds.set(id, file);
  }
}

const counts = { implemented: 0, planned: 0, 'no-visual': 0, unreviewed: 0 };
for (const entry of entries) {
  const status = audits.get(entry.slug)?.status ?? 'unreviewed';
  counts[status] += 1;
}

const incomplete = entries
  .filter((entry) => {
    const status = audits.get(entry.slug)?.status;
    return status === undefined || status === 'planned';
  })
  .sort((a, b) => {
    const aPlanned = audits.get(a.slug)?.status === 'planned' ? 0 : 1;
    const bPlanned = audits.get(b.slug)?.status === 'planned' ? 0 : 1;
    return aPlanned - bPlanned;
  });
console.log(`Visual coverage: ${entries.length} published entries`);
console.log(`  implemented: ${counts.implemented}`);
console.log(`  planned:     ${counts.planned}`);
console.log(`  no visual:   ${counts['no-visual']}`);
console.log(`  unreviewed:  ${counts.unreviewed}`);

const nextArg = process.argv.find((arg) => arg.startsWith('--next='));
if (nextArg) {
  const limit = Math.max(1, Number(nextArg.split('=')[1]) || 1);
  console.log('\nNext incomplete entries:');
  for (const entry of incomplete.slice(0, limit)) {
    const status = audits.get(entry.slug)?.status ?? 'unreviewed';
    console.log(`  ${entry.slug}\t${status}\t${entry.category}\t${entry.file}\t${entry.title}`);
  }
}

const expectedResolvedArg = process.argv.find((arg) => arg.startsWith('--expect-resolved='));
if (expectedResolvedArg) {
  const expectedSlugs = expectedResolvedArg
    .slice('--expect-resolved='.length)
    .split(',')
    .map((slug) => slug.trim())
    .filter(Boolean);
  if (expectedSlugs.length === 0 || new Set(expectedSlugs).size !== expectedSlugs.length) {
    fail('--expect-resolved needs a non-empty, duplicate-free comma-separated slug list');
  }
  for (const slug of expectedSlugs) {
    const status = audits.get(slug)?.status ?? 'unreviewed';
    if (!['implemented', 'no-visual'].includes(status)) {
      fail(`expected batch article ${slug} to be resolved, found ${status}`);
    }
  }
  const baselineArg = process.argv.find((arg) => arg.startsWith('--baseline-ref='));
  if (baselineArg) {
    const baselineRef = baselineArg.slice('--baseline-ref='.length).trim();
    if (!baselineRef) fail('--baseline-ref needs a git revision');
    const baselineStatuses = new Map();
    for (const file of auditFiles()) {
      try {
        const source = execFileSync('git', ['show', `${baselineRef}:data/visual-audits/${file}`], {
          cwd: root,
          encoding: 'utf8',
          stdio: ['ignore', 'pipe', 'pipe'],
        });
        baselineStatuses.set(basename(file, '.json'), JSON.parse(source).status);
      } catch {
        baselineStatuses.set(basename(file, '.json'), 'unreviewed');
      }
    }
    const newlyResolved = [...audits]
      .filter(([slug, audit]) => {
        const before = baselineStatuses.get(slug) ?? 'unreviewed';
        return !['implemented', 'no-visual'].includes(before) && ['implemented', 'no-visual'].includes(audit.status);
      })
      .map(([slug]) => slug)
      .sort();
    const expected = [...expectedSlugs].sort();
    if (JSON.stringify(newlyResolved) !== JSON.stringify(expected)) {
      fail(`resolved articles since ${baselineRef} are ${newlyResolved.join(', ') || 'none'}; expected ${expected.join(', ')}`);
    }
  }
}

if (process.argv.includes('--require-complete') && incomplete.length > 0) {
  fail(`${incomplete.length} published entries still need review or implementation`);
}

if (process.argv.includes('--dist')) {
  for (const [slug, audit] of audits) {
    if (audit.status !== 'implemented') continue;
    const entry = bySlug.get(slug);
    const output = join(root, 'dist', entry.category, slug, 'index.html');
    if (!existsSync(output)) {
      fail(`missing generated page for ${slug}`);
      continue;
    }
    const html = readFileSync(output, 'utf8');
    for (const id of audit.implementation.visualIds) {
      const marker = html.indexOf(`<!-- visual:${id} -->`);
      if (audit.medium === 'mermaid') {
        const blockStart = html.indexOf('<pre class="mermaid">', marker);
        const blockEnd = html.indexOf('</pre>', blockStart);
        const block = html.slice(blockStart, blockEnd + 6);
        const after = html.slice(blockEnd, blockEnd + 700);
        if (marker < 0 || blockStart < 0 || blockEnd < 0 || !block.includes('accTitle:') || !block.includes('accDescr:')) {
          fail(`generated Mermaid visual ${id} on ${slug} lost its accessible source`);
        }
        if (!after.includes('<p class="diagram-caption"><strong>Read it this way:</strong>')) {
          fail(`generated Mermaid visual ${id} on ${slug} lost its direct caption`);
        }
        continue;
      }
      const figureEnd = html.indexOf('</figure>', marker);
      const figure = html.slice(marker, figureEnd + 9);
      if (marker < 0 || figureEnd < 0 || !figure.includes('<figcaption>') || figure.includes('&lt;figcaption')) {
        fail(`generated visual ${id} on ${slug} is malformed or escaped`);
      }
      if (audit.medium === 'svg' && (!figure.includes('<svg') || !figure.includes('<title') || !figure.includes('<desc'))) {
        fail(`generated SVG visual ${id} on ${slug} lost accessible markup`);
      }
      if (!figure.includes('<figcaption><strong>Read it this way:</strong>')) {
        fail(`generated visual ${id} on ${slug} lost its direct caption`);
      }
    }
  }
}

if (process.exitCode) process.exit(process.exitCode);

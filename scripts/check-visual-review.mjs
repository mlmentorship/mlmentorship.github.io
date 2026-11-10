import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import process from 'node:process';
import { fromMarkdown } from 'mdast-util-from-markdown';
import { parse, parseFragment, serializeOuter } from 'parse5';

const root = resolve(import.meta.dirname, '..');
const dist = join(root, 'dist');
const auditsDir = join(root, 'data', 'visual-audits');
const reviewCardSource = readFileSync(join(root, 'src/components/VisualReviewCard.astro'), 'utf8');
const sourceMode = process.argv.includes('--source');
const catalog = JSON.parse(readFileSync(join(root, 'data/visual-lessons.json'), 'utf8'));
const postSlugs = readdirSync(join(root, 'src/content/posts')).filter(filename => filename.endsWith('.md')).map(filename => filename.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, '')).sort();
if (JSON.stringify(Object.keys(catalog.lessons).sort()) !== JSON.stringify(postSlugs)) throw new Error('Every source post needs exactly one authored visual lesson');

if (!sourceMode && !process.argv.includes('--dist')) {
  throw new Error('check-visual-review.mjs validates generated output; pass --dist after astro build');
}
if (!sourceMode && !existsSync(dist)) throw new Error('dist does not exist; run astro build first');
if (/visual-review-card--(?:article|deck)[\s\S]{0,500}overflow-y:\s*auto/.test(reviewCardSource)) {
  throw new Error('visual review cards must use the document scrollbar, not nested vertical scrolling');
}

const failures = [];
const implemented = new Map();
const implementedVisuals = new Map();

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function elements(root) {
  const result = [];
  const pending = [root];
  while (pending.length) {
    const node = pending.pop();
    if (node.tagName) result.push(node);
    pending.push(...(node.childNodes ?? []).toReversed());
  }
  return result;
}

const attribute = (node, name) => node.attrs?.find((item) => item.name === name)?.value;
const textContent = (node) => node.value ?? (node.childNodes ?? []).map(textContent).join('');

function sourceVisualHtml(source) {
  const fragments = [];
  function visit(node) {
    if (node.type === 'html') fragments.push(node.value);
    for (const child of node.children ?? []) visit(child);
  }
  visit(fromMarkdown(source));
  return fragments.join('\n');
}

if (sourceMode) {
  const artwork = JSON.parse(readFileSync(join(root, 'data/visual-lesson-artwork.json'), 'utf8'));
  const requested = process.argv.find((argument) => argument.startsWith('--slugs='))?.slice('--slugs='.length).split(',');
  const slugs = requested ?? Object.keys(catalog.lessons);
  for (const slug of slugs) {
    const lesson = catalog.lessons[slug];
    const audit = JSON.parse(readFileSync(join(auditsDir, `${slug}.json`), 'utf8'));
    const source = readFileSync(join(root, audit.article), 'utf8');
    const figures = [];
    function visitMarkdown(node) {
      if (node.type === 'html') {
        figures.push(...elements(parseFragment(node.value)).filter((element) => element.tagName === 'figure' && /(?:^|\s)learning-figure(?:\s|$)/.test(attribute(element, 'class') ?? '')));
      }
      for (const child of node.children ?? []) visitMarkdown(child);
    }
    visitMarkdown(fromMarkdown(source));
    if (!lesson?.idea || !lesson.steps?.length) {
      failures.push(`${slug}: a visual lesson needs an idea and ordered teaching steps`);
      continue;
    }
    const identities = new Set();
    for (const step of lesson.steps) {
      const identity = `${step.visualId}:${step.view}`;
      if (identities.has(identity)) failures.push(`${slug}: repeated teaching step ${identity}`);
      identities.add(identity);
      const scene = artwork.scenes[step.visualId];
      const figure = scene
        ? elements(parseFragment(scene.html)).find((element) => attribute(element, 'data-lesson-visual') === step.visualId)
        : figures.find((element) => elements(element).some((child) => attribute(child, 'data-visual-argument') === step.visualId));
      if (!figure) {
        failures.push(`${slug}: missing visual asset ${step.visualId}`);
        continue;
      }
      const children = elements(figure);
      if (!['mechanism', 'detail', 'trace'].includes(step.view)) failures.push(`${slug}: unsupported teaching view ${step.view}`);
      if (step.view === 'detail' && !children.some((element) => /(?:^|\s)visual-derivation-content(?:\s|$)/.test(attribute(element, 'class') ?? ''))) failures.push(`${slug}: ${identity} has no detailed drawing`);
      if (step.view === 'trace' && !children.some((element) => attribute(element, 'data-frame-key') !== undefined)) failures.push(`${slug}: ${identity} has no executable visual checkpoints`);
    }
  }
  if (failures.length) {
    console.error(failures.join('\n'));
    process.exit(1);
  }
  console.log(`Visual lesson source validation passed: ${slugs.length} posts and their ordered asset views.`);
  process.exit(0);
}

const { renderVisualLessonSteps } = await import('../src/utils/visualLessonHtml.ts');

for (const filename of readdirSync(auditsDir).filter((name) => name.endsWith('.json'))) {
  const audit = JSON.parse(readFileSync(join(auditsDir, filename), 'utf8'));
  if (audit.status !== 'implemented') continue;
  const article = readFileSync(join(root, audit.article), 'utf8');
  for (const visualId of audit.implementation?.visualIds ?? []) {
    const marker = article.match(new RegExp(`<!--\\s*visual:${escapeRegExp(visualId)}\\s*-->\\s*([\\s\\S]{0,240})`));
    const supported = marker?.[1]?.match(/^(?:\s|<!--[^]*?-->)*(<figure\b[^>]*class=["'][^"']*learning-figure|```mermaid\b|<pre\b[^>]*class=["'][^"']*mermaid)/);
    if (!supported) failures.push(`${audit.slug}: ${visualId} does not use a reviewable figure or Mermaid form`);
  }
  const category = article.match(/^category:\s*["']?([^"'\n]+)["']?\s*$/m)?.[1]?.trim();
  if (!category) {
    failures.push(`${audit.slug}: category could not be read from ${audit.article}`);
    continue;
  }
  const output = join(dist, category, audit.slug, 'index.html');
  if (!existsSync(output)) {
    failures.push(`${audit.slug}: generated article is missing`);
    continue;
  }
  const html = readFileSync(output, 'utf8');
  if (!html.includes('data-article-quick-review')) failures.push(`${audit.slug}: quick-review lead is missing`);
  if (!html.includes('data-reading-mode="visual"') || !html.includes('data-reading-mode="full"')) failures.push(`${audit.slug}: global reading-mode selector is missing`);
  if (!html.includes(`/review/#${audit.slug}`)) failures.push(`${audit.slug}: library-wide visual-review link is missing`);
  if (!html.includes('href="#full-explanation"')) failures.push(`${audit.slug}: full-content jump is missing`);
  if (!html.includes('data-full-explanation')) failures.push(`${audit.slug}: full-content boundary is missing`);
  const nodes = elements(parse(html));
  const lead = nodes.find((node) => attribute(node, 'data-article-quick-review') !== undefined);
  const answer = nodes.find((node) => attribute(node, 'id') === 'answer');
  if (!lead || attribute(lead, 'hidden') !== undefined || attribute(lead, 'data-ready') !== 'true' || attribute(lead, 'data-inline-visual') !== undefined) {
    failures.push(`${audit.slug}: the main illustration must be server-rendered and visible without JavaScript`);
  }
  if (!answer || nodes.indexOf(lead) >= nodes.indexOf(answer)) {
    failures.push(`${audit.slug}: the main illustration slot must precede the article body`);
  }
  const lesson = catalog.lessons[audit.slug];
  const selected = nodes.filter(node => attribute(node, 'data-visual-lesson-step') !== undefined);
  const expected = renderVisualLessonSteps(sourceVisualHtml(article), lesson, 'article-lesson');
  if (selected.length !== expected.length) failures.push(`${audit.slug}: incomplete authored article lesson`);
  for (const [index, expectedHtml] of expected.entries()) {
    const expectedFigure = elements(parseFragment(expectedHtml))[0];
    if (!selected[index] || serializeOuter(selected[index]) !== serializeOuter(expectedFigure)) failures.push(`${audit.slug}: built teaching step ${index + 1} differs from its source-selected figure`);
    let parent = selected[index]?.parentNode;
    while (parent) {
      if (parent.tagName === 'details') failures.push(`${audit.slug}: essential teaching step ${index + 1} is inside a fold`);
      parent = parent.parentNode;
    }
  }
  if (category === 'questions') {
    const practice = nodes.find((node) => attribute(node, 'data-practice-start') !== undefined);
    if (practice && textContent(practice).trim() !== 'Practice') {
      failures.push(`${audit.slug}: the practice command must use the topic-neutral label Practice`);
    }
    const body = answer ? elements(answer) : [];
    const firstSection = body.find((node) => node.tagName === 'h2');
    const lastField = body.findIndex((node) => node.tagName === 'strong' && ['Example.', 'Constraints.'].includes(node.childNodes?.map((child) => child.value ?? '').join('')));
    const visual = body.findIndex((node) => /(?:^|\s)(?:learning-figure|mermaid)(?:\s|$)/.test(attribute(node, 'class') ?? ''));
    if (attribute(firstSection ?? {}, 'id') !== 'problem' || lastField < 0 || visual <= lastField) {
      failures.push(`${audit.slug}: the complete Problem statement must precede the solution visual`);
    }
  }
  implemented.set(audit.slug, 0);
  implementedVisuals.set(audit.slug, audit.implementation?.visualIds ?? []);
}

const reviewOutput = join(dist, 'review', 'index.html');
if (!existsSync(reviewOutput)) failures.push('library-wide visual review route is missing');
else {
  const html = readFileSync(reviewOutput, 'utf8');
  if (!html.includes('data-visual-review-mode')) failures.push('library-wide review shell is missing');
  if (!html.includes('name="robots" content="noindex,follow"')) failures.push('library-wide review route must be noindex');
  if (!/<body[^>]*data-pagefind-ignore="all"/.test(html)) failures.push('library-wide review route must be excluded from Pagefind');
  if (!html.includes('data-review-fallback')) failures.push('library-wide review no-JavaScript fallback is missing');
  if (!html.includes('data-review-explanation')) failures.push('library-wide review explanation panel is missing');
  if (!html.includes('data-review-controls')) failures.push('library-wide review bottom controls are missing');
  const dataMatch = html.match(/<script id="visual-review-data" type="application\/json">([\s\S]*?)<\/script>/);
  if (!dataMatch) failures.push('library-wide review entry data is missing');
  else {
    let entries;
    try { entries = JSON.parse(dataMatch[1]); } catch { failures.push('library-wide review entry data is invalid JSON'); }
    if (Array.isArray(entries)) {
      const slugs = new Set();
      for (const entry of entries) {
        if (!entry?.slug || !entry?.href || !entry?.objective || !entry?.visualId || !entry?.volumeId || !entry?.chapterId) {
          failures.push('library-wide review contains an incomplete entry');
          continue;
        }
        if (slugs.has(entry.slug)) failures.push(`library-wide review duplicates ${entry.slug}`);
        slugs.add(entry.slug);
        if (!implemented.has(entry.slug)) failures.push(`library-wide review includes unresolved ${entry.slug}`);
        else implemented.set(entry.slug, implemented.get(entry.slug) + 1);
        if (JSON.stringify(entry.visualIds) !== JSON.stringify(implementedVisuals.get(entry.slug))) {
          failures.push(`library-wide review must include every declared visual for ${entry.slug}`);
        }
        const lesson = catalog.lessons[entry.slug];
        if (JSON.stringify(entry.lesson) !== JSON.stringify({ idea: lesson.idea, steps: lesson.steps })) failures.push(`${entry.slug}: article and review selections differ`);
      }
      if (entries.length !== implemented.size) failures.push(`library-wide review has ${entries.length} entries; expected ${implemented.size}`);
    }
  }
}

const libraryRoot = join(dist, 'library');
let reviewLinkCount = 0;
for (const volume of readdirSync(libraryRoot, { withFileTypes: true }).filter((entry) => entry.isDirectory())) {
  const volumePath = join(libraryRoot, volume.name);
  for (const chapter of readdirSync(volumePath, { withFileTypes: true }).filter((entry) => entry.isDirectory())) {
    const chapterPath = join(volumePath, chapter.name);
    const chapterOutput = join(chapterPath, 'index.html');
    if (!existsSync(chapterOutput)) continue;
    const chapterHtml = readFileSync(chapterOutput, 'utf8');
    const reviewLink = chapterHtml.match(/href="\/review\/#([^"]+)"[^>]*data-visual-review-count="(\d+)"/);
    if (!reviewLink) continue;
    reviewLinkCount += 1;
    if (!implemented.has(reviewLink[1])) failures.push(`${volume.name}/${chapter.name}: review starts at unresolved ${reviewLink[1]}`);
    if (Number(reviewLink[2]) < 1) failures.push(`${volume.name}/${chapter.name}: review count must be positive`);
  }
}

for (const [slug, count] of implemented) {
  if (count !== 1) failures.push(`${slug}: expected in exactly one chapter review, found ${count}`);
}

for (const sitemap of readdirSync(dist).filter((name) => /^sitemap.*\.xml$/.test(name))) {
  if (readFileSync(join(dist, sitemap), 'utf8').includes('/review/')) failures.push(`${sitemap}: review routes must be excluded`);
}

if (failures.length > 0) {
  console.error(`Visual review validation failed (${failures.length}):`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}

console.log(`Visual review validation passed: ${implemented.size} article leads, one library-wide sequence, and ${reviewLinkCount} chapter entry points.`);
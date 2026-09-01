import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import process from 'node:process';

const root = resolve(import.meta.dirname, '..');
const dist = join(root, 'dist');
const auditsDir = join(root, 'data', 'visual-audits');

if (!process.argv.includes('--dist')) {
  throw new Error('check-visual-review.mjs validates generated output; pass --dist after astro build');
}
if (!existsSync(dist)) throw new Error('dist does not exist; run astro build first');

const failures = [];
const implemented = new Map();

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

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
  if (!html.includes('href="#full-explanation"')) failures.push(`${audit.slug}: full-content jump is missing`);
  if (!html.includes('data-full-explanation')) failures.push(`${audit.slug}: full-content boundary is missing`);
  implemented.set(audit.slug, 0);
}

const libraryRoot = join(dist, 'library');
let reviewPageCount = 0;
for (const volume of readdirSync(libraryRoot, { withFileTypes: true }).filter((entry) => entry.isDirectory())) {
  const volumePath = join(libraryRoot, volume.name);
  for (const chapter of readdirSync(volumePath, { withFileTypes: true }).filter((entry) => entry.isDirectory())) {
    const chapterPath = join(volumePath, chapter.name);
    const chapterOutput = join(chapterPath, 'index.html');
    if (!existsSync(chapterOutput)) continue;
    const chapterHtml = readFileSync(chapterOutput, 'utf8');
    const reviewLink = chapterHtml.match(/href="([^"]+\/review\/)"[^>]*data-visual-review-count="(\d+)"/);
    if (!reviewLink) continue;

    const reviewOutput = join(chapterPath, 'review', 'index.html');
    if (!existsSync(reviewOutput)) {
      failures.push(`${volume.name}/${chapter.name}: linked review route is missing`);
      continue;
    }
    const html = readFileSync(reviewOutput, 'utf8');
    reviewPageCount += 1;
    if (!html.includes('data-visual-review-mode')) failures.push(`${volume.name}/${chapter.name}: review shell is missing`);
    if (!html.includes('name="robots" content="noindex,follow"')) failures.push(`${volume.name}/${chapter.name}: review route must be noindex`);
    if (!/<body[^>]*data-pagefind-ignore="all"/.test(html)) failures.push(`${volume.name}/${chapter.name}: review route must be excluded from Pagefind`);
    if (!html.includes('data-review-fallback')) failures.push(`${volume.name}/${chapter.name}: no-JavaScript fallback is missing`);
    const dataMatch = html.match(/<script id="visual-review-data" type="application\/json">([\s\S]*?)<\/script>/);
    if (!dataMatch) {
      failures.push(`${volume.name}/${chapter.name}: review entry data is missing`);
      continue;
    }
    let entries;
    try { entries = JSON.parse(dataMatch[1]); } catch { failures.push(`${volume.name}/${chapter.name}: review entry data is invalid JSON`); continue; }
    if (!Array.isArray(entries) || entries.length === 0) failures.push(`${volume.name}/${chapter.name}: review entry data is empty`);
    if (entries.length !== Number(reviewLink[2])) failures.push(`${volume.name}/${chapter.name}: review count does not match its chapter link`);
    const localSlugs = new Set();
    for (const entry of entries) {
      if (!entry?.slug || !entry?.href || !entry?.objective || !entry?.visualId) {
        failures.push(`${volume.name}/${chapter.name}: incomplete review entry`);
        continue;
      }
      if (localSlugs.has(entry.slug)) failures.push(`${volume.name}/${chapter.name}: duplicate ${entry.slug}`);
      localSlugs.add(entry.slug);
      if (!implemented.has(entry.slug)) failures.push(`${volume.name}/${chapter.name}: ${entry.slug} is not implemented`);
      else implemented.set(entry.slug, implemented.get(entry.slug) + 1);
    }
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

console.log(`Visual review validation passed: ${implemented.size} article leads across ${reviewPageCount} chapter decks.`);
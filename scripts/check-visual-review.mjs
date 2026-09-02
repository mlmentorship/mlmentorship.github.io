import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import process from 'node:process';

const root = resolve(import.meta.dirname, '..');
const dist = join(root, 'dist');
const auditsDir = join(root, 'data', 'visual-audits');
const reviewCardSource = readFileSync(join(root, 'src/components/VisualReviewCard.astro'), 'utf8');

if (!process.argv.includes('--dist')) {
  throw new Error('check-visual-review.mjs validates generated output; pass --dist after astro build');
}
if (!existsSync(dist)) throw new Error('dist does not exist; run astro build first');
if (/visual-review-card--(?:article|deck)[\s\S]{0,500}overflow-y:\s*auto/.test(reviewCardSource)) {
  throw new Error('visual review cards must use the document scrollbar, not nested vertical scrolling');
}

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
  if (!html.includes('data-reading-mode="visual"') || !html.includes('data-reading-mode="full"')) failures.push(`${audit.slug}: global reading-mode selector is missing`);
  if (!html.includes(`/review/#${audit.slug}`)) failures.push(`${audit.slug}: library-wide visual-review link is missing`);
  if (!html.includes('href="#full-explanation"')) failures.push(`${audit.slug}: full-content jump is missing`);
  if (!html.includes('data-full-explanation')) failures.push(`${audit.slug}: full-content boundary is missing`);
  implemented.set(audit.slug, 0);
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
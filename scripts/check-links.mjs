import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve('dist');
const htmlFiles = [];

function walk(directory) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) walk(fullPath);
    else if (entry.name.endsWith('.html')) htmlFiles.push(fullPath);
  }
}

function builtTarget(href) {
  const cleanHref = decodeURIComponent(href.split('#')[0].split('?')[0]);
  if (!cleanHref) return null;

  const target = path.join(root, cleanHref);
  if (cleanHref.endsWith('/')) return path.join(target, 'index.html');
  if (!path.extname(target)) {
    const fileTarget = `${target}.html`;
    return fs.existsSync(fileTarget) ? fileTarget : path.join(target, 'index.html');
  }
  return target;
}

walk(root);

const broken = [];
for (const file of htmlFiles) {
  const html = fs.readFileSync(file, 'utf8');
  for (const match of html.matchAll(/href=["']([^"']+)["']/g)) {
    const href = match[1];
    if (!href.startsWith('/') || href.startsWith('//')) continue;
    const target = builtTarget(href);
    if (target && !fs.existsSync(target)) {
      broken.push(`${path.relative(root, file)} -> ${href}`);
    }
  }
}

const rss = fs.readFileSync(path.join(root, 'rss.xml'), 'utf8');
for (const chunk of rss.split('<link>').slice(1)) {
  const url = chunk.split('</link>')[0];
  if (!url.startsWith('https://mlmentorship.com/')) continue;
  const href = url.slice('https://mlmentorship.com'.length);
  if (href === '/') continue;
  const target = builtTarget(href);
  if (target && !fs.existsSync(target)) broken.push(`rss.xml -> ${href}`);
}

if (broken.length > 0) {
  console.error(`Found ${broken.length} broken internal link(s):`);
  console.error(broken.join('\n'));
  process.exit(1);
}

console.log(`Link check passed: ${htmlFiles.length} HTML pages and RSS items resolve.`);

import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const scanRoots = ['.github', 'docs', 'public', 'scripts', 'src'];
const rootFiles = ['CONTENT_ROADMAP.md', '.env.example', 'astro.config.mjs', 'package.json'];
const textExtensions = new Set([
  '.astro', '.css', '.html', '.js', '.json', '.md', '.mjs', '.py', '.toml', '.ts', '.txt', '.yaml', '.yml',
]);
const forbidden = String.fromCodePoint(0x2014);
const violations = [];

function scanFile(file) {
  if (!fs.existsSync(file) || !textExtensions.has(path.extname(file))) return;
  const text = fs.readFileSync(file, 'utf8');
  text.split('\n').forEach((line, index) => {
    if (line.includes(forbidden)) violations.push(`${path.relative(root, file)}:${index + 1}`);
  });
}

function walk(directory) {
  if (!fs.existsSync(directory)) return;
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) walk(fullPath);
    else scanFile(fullPath);
  }
}

scanRoots.forEach((directory) => walk(path.join(root, directory)));
rootFiles.forEach((file) => scanFile(path.join(root, file)));

if (violations.length > 0) {
  console.error(`Editorial style check failed: forbidden punctuation found in ${violations.length} line(s):`);
  console.error(violations.join('\n'));
  process.exit(1);
}

console.log('Editorial style check passed: no em dashes found.');

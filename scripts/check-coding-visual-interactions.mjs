import assert from 'node:assert/strict';
import fs from 'node:fs';

const generator = fs.readFileSync('scripts/generate-coding-question-book.mjs', 'utf8');
const client = fs.readFileSync('src/utils/codingVisuals.ts', 'utf8');
const styles = fs.readFileSync('src/styles/global.css', 'utf8');
const sample = fs.readFileSync('src/content/posts/2026-09-01-two-sum.md', 'utf8');

assert.match(generator, /data-coding-previous/, 'Previous control is required');
assert.match(generator, /data-coding-next/, 'Next control is required');
assert.match(generator, /data-coding-play/, 'Play control is required');
assert.match(client, /ArrowLeft|ArrowRight/, 'keyboard arrow semantics are required');
assert.match(client, /prefers-reduced-motion: reduce/, 'reduced-motion stepping is required');
assert.match(client, /data-motion-key/, 'stable motion-key interpolation is required');
assert.match(styles, /@media print[\s\S]*coding-trace-frame/, 'print must reveal authored frames');
assert.match(styles, /@media \(max-width: 640px\)/, 'exact-mobile behavior is required');
assert.match(sample, /data-coding-frame="0"(?![^>]*hidden)/, 'no-JS first frame must be visible');
assert.match(sample, /data-coding-controls hidden/, 'no-JS controls must stay hidden');

console.log('Coding visual no-JS, keyboard, reduced-motion, mobile, print, and motion checks passed.');

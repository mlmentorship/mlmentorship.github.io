import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { parseFragment } from 'parse5';

const root = process.cwd();
const scanRoots = ['.github', 'docs', 'public', 'scripts', 'src'];
const rootFiles = ['CONTENT_ROADMAP.md', '.env.example', 'astro.config.mjs', 'package.json'];
const textExtensions = new Set([
  '.astro', '.css', '.html', '.js', '.json', '.md', '.mjs', '.py', '.toml', '.ts', '.txt', '.yaml', '.yml',
]);
const emDash = String.fromCodePoint(0x2014);
const contentRoot = path.join(root, 'src', 'content', 'posts');
const contentRules = [
  ['why-it-matters filler', /\bwhy (?:it|this|the distinction|the difference) matters\b/i],
  ['distinction-matters filler', /\b(?:the|this|that) distinction matters\b/i],
  ['difference-matters filler', /\b(?:the|this|that) difference matters\b/i],
  ['key-insight filler', /\bthe key (?:idea|insight|point|difference)\b/i],
  ['inflated superlative', /\b(?:the )?single (?:most important|highest-leverage)\b/i],
  ['conversational framing', /\bhere['’]s\b/i],
  ['not-just-but construction', /\bnot just[^.!?\n]{0,120}\bbut(?: also)?\b/i],
  ['not-merely-but construction', /\bnot merely[^.!?\n]{0,120}\bbut(?: also)?\b/i],
  ['seam metaphor', /\b(?:seam|seams|seamless|seamlessly)\b/i],
  ['delve filler', /\bdelve(?:s|d|ing)?\b/i],
  ['vague landscape metaphor', /\b(?:modern|optimizer|technique|architecture) landscape\b/i],
  ['inflated adjective', /\b(?:crucial(?:ly)?|pivotal|game[- ]changer|revolutionary|groundbreaking|transformative)\b/i],
  ['indirect use verb', /\butilize(?:s|d|ing)?\b/i],
  ['unlock metaphor', /\bunlock(?:s|ed|ing)?\b/i],
  ['note filler', /\bit(?: is|['’]s) (?:important to note|worth noting)\b/i],
  ['core filler', /\bat its core\b/i],
  ['serves-as filler', /\bserves as\b/i],
  ['importance filler', /\b(?:a testament to|underscores|highlights the importance)\b/i],
  ['stale coming-soon marker', /\(coming soon\)/i],
  ['related-reference inconsistency', /^\*Related reference:/i],
];
const violations = [];
const parser = unified().use(remarkParse).use(remarkGfm).use(remarkMath);
const sentenceSegmenter = new Intl.Segmenter('en', { granularity: 'sentence' });
const qualified = [];
const learningPieces = new Map(JSON.parse(fs.readFileSync(path.join(root, 'src/data/learningMechanisms.json'), 'utf8')).pieces.map((piece) => [piece.slug, piece]));

function wordCount(value) {
  return (value.match(/[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*/g) ?? []).length;
}

function preserveNewlines(value) {
  return '\n'.repeat((value.match(/\n/g) ?? []).length);
}

export function qualifyProse(body) {
  const result = { core: 0, extension: 0, sentences: [] };
  let layer = 'core';
  function markdownText(node) {
    if (node.type === 'text') return node.value;
    if (node.type === 'inlineCode') return 'identifier';
    if (node.type === 'inlineMath') return 'formula';
    if (node.type === 'break') return ' ';
    return (node.children ?? []).map(markdownText).join('');
  }
  function record(text, line, quoted = false) {
    const readable = text.replace(/\s+/g, ' ').trim();
    result[layer] += wordCount(readable);
    if (quoted) return;
    const ordinary = readable.replace(/"[^"\n]*"|“[^”\n]*”/g, 'quotation');
    for (const { segment } of sentenceSegmenter.segment(ordinary)) {
      const words = wordCount(segment);
      if (words > 20) result.sentences.push({ line, words, text: segment.trim() });
    }
  }
  function htmlText(node) {
    if (['code', 'pre', 'svg', 'style', 'script'].includes(node.nodeName)) return '';
    return node.nodeName === '#text' ? node.value : (node.childNodes ?? []).map(htmlText).join('');
  }
  function htmlWalk(node, startLine) {
    const attributes = Object.fromEntries((node.attrs ?? []).map((attribute) => [attribute.name, attribute.value]));
    const classes = new Set((attributes.class ?? '').split(/\s+/));
    if (['svg', 'pre', 'style', 'script'].includes(node.nodeName) || ['diagram-source', 'visual-title', 'visual-kicker'].some((name) => classes.has(name))) return;
    if (node.nodeName === 'p' || node.nodeName === 'figcaption') {
      record(htmlText(node), startLine + (node.sourceCodeLocation?.startLine ?? 1) - 1);
      return;
    }
    for (const child of node.childNodes ?? []) htmlWalk(child, startLine);
  }
  function walk(node, quoted = false) {
    if (node.type === 'html') {
      if (node.value.includes('<!-- learning:core-end -->')) layer = 'extension';
      htmlWalk(parseFragment(node.value, { sourceCodeLocationInfo: true }), node.position.start.line);
      return;
    }
    if (['heading', 'code', 'math', 'table', 'definition'].includes(node.type)) return;
    if (node.type === 'paragraph') {
      const text = markdownText(node);
      if (!/^Related:/.test(text)) record(text, node.position.start.line, quoted);
      return;
    }
    for (const child of node.children ?? []) walk(child, quoted || node.type === 'blockquote');
  }
  walk(parser.parse(body));
  return result;
}

function scanContentFile(file, text) {
  const relative = path.relative(root, file);
  const lines = text.split('\n');

  for (const [label, pattern] of contentRules) {
    lines.forEach((line, index) => {
      if (pattern.test(line)) violations.push(`${relative}:${index + 1}: ${label}`);
    });
  }

  const category = text.match(/^category:\s*"([^"]+)"\s*$/m)?.[1];
  const description = text.match(/^description:\s*"(.*)"\s*$/m)?.[1];
  if (!description) {
    violations.push(`${relative}: missing description`);
  } else if (wordCount(description) > 32) {
    violations.push(`${relative}: description exceeds 32 words`);
  }

  const frontmatter = text.match(/^---[\s\S]*?---\s*/)?.[0] ?? '';
  const body = text.slice(frontmatter.length);
  const firstHeading = body.match(/^## (.+)$/m)?.[1];
  if (category === 'concepts' && firstHeading !== 'Summary') {
    violations.push(`${relative}: first concept section must be "## Summary"`);
  } else if (category === 'concepts') {
    const summaryStart = body.indexOf('## Summary');
    const contentStart = body.indexOf('\n', summaryStart) + 1;
    const nextSection = body.indexOf('\n## ', contentStart);
    const summary = body
      .slice(contentStart, nextSection < 0 ? body.length : nextSection)
      .replace(/```[\s\S]*?```/g, '')
      .replace(/\$\$[\s\S]*?\$\$/g, '')
      .replace(/[#*_`>|\[\]()]/g, ' ');
    if (wordCount(summary) < 15) violations.push(`${relative}: Summary must contain a substantive answer`);
  }
  if (category === 'questions' || category === 'guides') {
    const firstHeadingIndex = body.search(/^## /m);
    const opening = (firstHeadingIndex >= 0 ? body.slice(0, firstHeadingIndex) : body)
      .replace(/^>.*$/gm, '')
      .trim();
    if (wordCount(opening) < 10) violations.push(`${relative}: opening must state the answer or thesis before the first section`);
  }

  const bodyLineOffset = frontmatter.split('\n').length - 1;
  if (body.includes('<!-- learning:core-end -->')) {
    const slug = path.basename(file).replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, '');
    const piece = learningPieces.get(slug);
    const prose = qualifyProse(body);
    for (const sentence of prose.sentences) {
      violations.push(`${relative}:${bodyLineOffset + sentence.line}: sentence exceeds 20 words (${sentence.words}): ${sentence.text}`);
    }
    for (const layer of ['core', 'extension']) {
      if (piece && prose[layer] > piece.budget[layer]) violations.push(`${relative}: ${layer} prose exceeds ${piece.budget[layer]} words (${prose[layer]})`);
    }
    qualified.push({ slug, core: prose.core, extension: prose.extension });
    return;
  }
  const prose = body
    .replace(/```[\s\S]*?```/g, preserveNewlines)
    .replace(/\$\$[\s\S]*?\$\$/g, preserveNewlines);
  prose.split('\n').forEach((rawLine, index) => {
    if (!rawLine.trim() || /^\s*(?:#|\||<)/.test(rawLine)) return;
    const line = rawLine
      .replace(/^\s*(?:>|[-*+] |\d+\. )+/, '')
      .replace(/`[^`]*`/g, 'token')
      .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
      .replace(/\$[^$]+\$/g, 'formula')
      .replace(/[*_~]/g, '');
    const sentences = line.match(/[^.!?]+(?:[.!?]+["'’”]?|$)/g) ?? [];
    for (const sentence of sentences) {
      if (wordCount(sentence) > 48) {
        violations.push(`${relative}:${bodyLineOffset + index + 1}: sentence exceeds 48 words`);
      }
    }
  });
}

function scanFile(file) {
  if (!fs.existsSync(file) || !textExtensions.has(path.extname(file))) return;
  const text = fs.readFileSync(file, 'utf8');
  text.split('\n').forEach((line, index) => {
    if (line.includes(emDash)) violations.push(`${path.relative(root, file)}:${index + 1}: em dash`);
  });
  if (path.dirname(file) === contentRoot && path.extname(file) === '.md') scanContentFile(file, text);
}

function walk(directory) {
  if (!fs.existsSync(directory)) return;
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) walk(fullPath);
    else scanFile(fullPath);
  }
}

if (process.argv.includes('--self-test')) {
  const long = Array.from({ length: 21 }, (_, index) => `word${index}`).join(' ');
  assert.equal(qualifyProse(`${long}.`).sentences.length, 1);
  assert.equal(qualifyProse(`${long.replace('word10 ', 'word10\n')}.`).sentences.length, 1);
  for (const source of [`\`\`\`python\n${long}\n\`\`\``, `$$\n${long}\n$$`, `> ${long}.`, `"${long}."`, `| Header |\n| --- |\n| ${long} |`]) {
    assert.equal(qualifyProse(source).sentences.length, 0);
  }
  assert.equal(qualifyProse(`<figure><figcaption>${long}.</figcaption></figure>`).sentences.length, 1);
  const layers = qualifyProse('One.\n\n<!-- learning:core-end -->\n\n<div data-learning-extension>\n\nTwo.\n\n</div>');
  assert.deepEqual([layers.core, layers.extension], [1, 1]);
  console.log('Editorial selftest passed: multiline sentences, literal exceptions, captions and reading layers.');
  process.exit(0);
}

scanRoots.forEach((directory) => walk(path.join(root, directory)));
rootFiles.forEach((file) => scanFile(path.join(root, file)));

if (violations.length > 0) {
  console.error(`Editorial style check failed with ${violations.length} violation(s):`);
  console.error(violations.join('\n'));
  process.exit(1);
}

console.log('Editorial style check passed: pyramid openings, concise prose, and banned-language rules satisfied.');
console.log(`Pattern-first prose qualified: ${qualified.length}/${learningPieces.size} articles; 20-word sentences and approved prose budgets.`);
if (process.argv.includes('--report')) console.log(JSON.stringify(qualified, null, 2));

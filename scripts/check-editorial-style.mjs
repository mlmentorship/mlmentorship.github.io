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
const requestedSlugs = new Set((process.argv.find((argument) => argument.startsWith('--slugs='))?.slice('--slugs='.length) ?? '').split(',').filter(Boolean));

function wordCount(value) {
  return (value.match(/[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*/g) ?? []).length;
}

function preserveNewlines(value) {
  return '\n'.repeat((value.match(/\n/g) ?? []).length);
}

export function qualifyProse(body, { separateProblem = false } = {}) {
  const result = { problem: 0, core: 0, extension: 0, sentences: [] };
  let layer = 'core';
  let inProblem = false;
  function markdownText(node) {
    if (node.type === 'text') return node.value;
    if (node.type === 'inlineCode') return 'identifier';
    if (node.type === 'inlineMath') return 'formula';
    if (node.type === 'break') return ' ';
    return (node.children ?? []).map(markdownText).join('');
  }
  function record(text, line, quoted = false) {
    const readable = text.replace(/\s+/g, ' ').trim();
    result[inProblem ? 'problem' : layer] += wordCount(readable);
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
      if (node.value.includes('<!-- learning:core-end -->')) {
        layer = 'extension';
        inProblem = false;
      }
      htmlWalk(parseFragment(node.value, { sourceCodeLocationInfo: true }), node.position.start.line);
      return;
    }
    if (node.type === 'heading' && node.depth === 2) {
      inProblem = separateProblem && layer === 'core' && markdownText(node) === 'Problem';
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

export function problemStatementIssues(body, { coding = true } = {}) {
  const nodes = parser.parse(body).children;
  const headingText = (node) => node.value ?? (node.children ?? []).map(headingText).join('');
  const start = nodes.findIndex((node) => node.type === 'heading' && node.depth === 2);
  if (start < 0 || headingText(nodes[start]) !== 'Problem') return [`first ${coding ? 'coding' : 'question'} section must be "## Problem"`];
  const next = nodes.findIndex((node, index) => index > start && node.type === 'heading' && node.depth === 2);
  const statement = nodes.slice(start + 1, next < 0 ? nodes.length : next);
  const labels = statement.filter((node) => node.type === 'paragraph' && node.children[0]?.type === 'strong')
    .map((node) => ({ label: headingText(node.children[0]).replace(/[.:]$/, ''), text: headingText(node).slice(headingText(node.children[0]).length).trim() }));
  const issues = [];
  const required = coding ? ['Input', 'Output', 'Rules', 'Example'] : ['Context', 'Task', 'Constraints'];
  let previous = -1;
  for (const label of required) {
    const index = labels.findIndex((field) => field.label === label && field.text.length > 0);
    if (index < 0) issues.push(`Problem must state **${label}.** before the solution`);
    else if (index <= previous) issues.push(`Problem fields must follow ${required.join(', ')} order`);
    previous = Math.max(previous, index);
  }
  if (nodes.slice(0, start).some((node) => node.type === 'code')) issues.push('solution code must follow the problem statement');
  return issues;
}

function scanContentFile(file, text) {
  const relative = path.relative(root, file);
  const slug = path.basename(file).replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, '');
  if (requestedSlugs.size && !requestedSlugs.has(slug)) return;
  const coding = fs.existsSync(path.join(root, 'scripts/coding-visuals/problems', `${slug}.mjs`));
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
  if (coding || category === 'questions') {
    for (const issue of problemStatementIssues(body, { coding })) violations.push(`${relative}: ${issue}`);
  }
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
    if (wordCount(opening) < 10) violations.push(`${relative}: opening must frame the task or topic before the first section`);
  }

  const bodyLineOffset = frontmatter.split('\n').length - 1;
  if (body.includes('<!-- learning:core-end -->')) {
    const piece = learningPieces.get(slug);
    const prose = qualifyProse(body, { separateProblem: coding || category === 'questions' });
    for (const sentence of prose.sentences) {
      violations.push(`${relative}:${bodyLineOffset + sentence.line}: sentence exceeds 20 words (${sentence.words}): ${sentence.text}`);
    }
    for (const layer of ['core', 'extension']) {
      if (piece && prose[layer] > piece.budget[layer]) violations.push(`${relative}: ${layer} prose exceeds ${piece.budget[layer]} words (${prose[layer]})`);
    }
    qualified.push({ slug, problem: prose.problem, core: prose.core, extension: prose.extension });
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
  const statement = 'Find distinct indices for a target sum without reusing a position.\n\n## Problem\n\n**Input.** Integer values and a target.\n\n**Output.** Return two original indices.\n\n**Rules.** Positions must differ.\n\n**Example.** Two threes make six.\n\n## Picture\n\nExplain the solution.';
  assert.deepEqual(problemStatementIssues(statement), []);
  for (const label of ['Input', 'Output', 'Rules', 'Example']) {
    assert(problemStatementIssues(statement.replace(`**${label}.**`, '**Missing.**')).length > 0);
  }
  assert(problemStatementIssues(statement.replace('## Problem', '## Solution')).length > 0);
  assert(problemStatementIssues(statement.replace('**Input.**', '**Output.**').replace('**Output.** Return', '**Input.** Return')).length > 0);
  const question = 'Frame a model decision before comparing possible interventions and their evidence.\n\n## Problem\n\n**Context.** A support model cites an old policy.\n\n**Task.** Specify evidence needed before changing the system.\n\n**Constraints.** Preserve authorization and distinguish missing facts from formatting errors.\n\n## Answer\n\nCompare the evidence.';
  assert.deepEqual(problemStatementIssues(question, { coding: false }), []);
  for (const label of ['Context', 'Task', 'Constraints']) {
    assert(problemStatementIssues(question.replace(`**${label}.**`, '**Missing.**'), { coding: false }).length > 0);
  }
  assert(problemStatementIssues(question.replace('## Problem', '## Answer'), { coding: false }).length > 0);
  assert(problemStatementIssues(question.replace('**Context.**', '**Task.**').replace('**Task.** Specify', '**Context.** Specify'), { coding: false }).length > 0);
  const separated = qualifyProse(statement, { separateProblem: true });
  assert(separated.problem > 0);
  assert.equal(qualifyProse(statement).core, separated.core + separated.problem);
  assert.equal(qualifyProse(`## Problem\n\n${long}.\n\n## Picture\n\nShort.`, { separateProblem: true }).sentences.length, 1);
  const boundary = qualifyProse('## Problem\n\nTask.\n\n<!-- learning:core-end -->\n\nExtension.', { separateProblem: true });
  assert.deepEqual([boundary.problem, boundary.extension], [1, 1]);
  console.log('Editorial selftest passed: sentences, reading layers, complete problem fields and separate statement budgets.');
  process.exit(0);
}

scanRoots.forEach((directory) => walk(path.join(root, directory)));
rootFiles.forEach((file) => scanFile(path.join(root, file)));

if (violations.length > 0) {
  console.error(`Editorial style check failed with ${violations.length} violation(s):`);
  console.error(violations.join('\n'));
  process.exit(1);
}

console.log('Editorial style check passed: complete question statements, task context, concise prose, and banned-language rules satisfied.');
console.log(`Pattern-first prose qualified: ${qualified.length}/${learningPieces.size} articles; 20-word sentences and approved prose budgets.`);
if (process.argv.includes('--report')) console.log(JSON.stringify(qualified, null, 2));

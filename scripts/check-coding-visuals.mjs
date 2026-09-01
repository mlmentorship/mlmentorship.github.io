import fs from 'node:fs';
import path from 'node:path';
import { codingQuestionVisuals } from './coding-question-visuals.mjs';

const root = process.cwd();
const postsDir = path.join(root, 'src/content/posts');
const auditsDir = path.join(root, 'data/visual-audits');
const allowedTypes = new Set(['array', 'array-map', 'table', 'grid', 'stack', 'queue-grid', 'graph', 'tree', 'intervals', 'linked', 'trie', 'bits', 'shapes', 'attention', 'buckets', 'prefix', 'dual-window', 'choices', 'lru', 'heap']);
const allowedTones = new Set(['focus', 'state', 'output', 'warning', 'neutral']);
const failures = [];

function fail(message) {
  failures.push(message);
}

function nonEmpty(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function checkMarks(marks, rows, columns, label) {
  if (!Array.isArray(marks)) return;
  for (const mark of marks) {
    if (!Number.isInteger(mark.index) && (!Number.isInteger(mark.row) || !Number.isInteger(mark.col))) fail(`${label}: invalid mark coordinate`);
    if (mark.index !== undefined && (!Number.isInteger(mark.index) || mark.index < 0 || mark.index >= rows * columns)) fail(`${label}: mark index outside scene`);
    if (mark.row !== undefined && (mark.row < 0 || mark.row >= rows || mark.col < 0 || mark.col >= columns)) fail(`${label}: grid mark outside scene`);
    if (!nonEmpty(mark.label)) fail(`${label}: mark needs a visible label`);
    if (!allowedTones.has(mark.tone ?? 'neutral')) fail(`${label}: unknown mark tone ${mark.tone}`);
  }
}

function checkScene(scene, label) {
  if (!scene || typeof scene !== 'object' || !allowedTypes.has(scene.type)) {
    fail(`${label}: unknown scene type ${scene?.type}`);
    return;
  }
  if (scene.type === 'array' || scene.type === 'array-map' || scene.type === 'bits') {
    if (!Array.isArray(scene.items) && !Array.isArray(scene.values)) fail(`${label}: array scene needs items`);
    const items = scene.items ?? scene.values;
    if (items.length < 1 || items.some((item) => typeof item !== 'string')) fail(`${label}: array scene has a non-string item`);
    checkMarks(scene.marks, 1, items.length, label);
  }
  if (scene.type === 'array-map') {
    if (!Array.isArray(scene.map)) fail(`${label}: array-map scene needs a map`);
    for (const entry of scene.map ?? []) if (!Array.isArray(entry) || entry.length !== 2 || !nonEmpty(String(entry[0])) || !nonEmpty(String(entry[1]))) fail(`${label}: invalid map entry`);
  }
  if (scene.type === 'table') {
    if (!Array.isArray(scene.columns) || scene.columns.length < 1 || !Array.isArray(scene.rows) || scene.rows.length < 1) fail(`${label}: table needs columns and rows`);
    if ((scene.rows ?? []).some((row) => !Array.isArray(row) || row.length !== scene.columns.length)) fail(`${label}: table row width mismatch`);
    for (const index of scene.active ?? []) if (!Number.isInteger(index) || index < 0 || index >= scene.rows.length * scene.columns.length) fail(`${label}: active table cell outside table`);
  }
  if (scene.type === 'grid' || scene.type === 'attention') {
    if (!Array.isArray(scene.rows) || scene.rows.length < 1 || scene.rows.some((row) => !Array.isArray(row) || row.length !== scene.rows[0].length || row.length < 1)) fail(`${label}: grid rows must be rectangular and nonempty`);
    checkMarks(scene.marks, scene.rows?.length ?? 0, scene.rows?.[0]?.length ?? 0, label);
  }
  if (scene.type === 'queue-grid') {
    if (!Array.isArray(scene.rows) || scene.rows.length < 1 || scene.rows.some((row) => !Array.isArray(row) || row.length < 1)) fail(`${label}: queue grid rows must be nonempty`);
  }
  if (scene.type === 'stack') {
    if (!nonEmpty(scene.input) || !Array.isArray(scene.values)) fail(`${label}: stack needs input and values`);
  }
  if (scene.type === 'graph') {
    if (!Array.isArray(scene.nodes) || scene.nodes.length < 1 || !Array.isArray(scene.edges)) fail(`${label}: graph needs nodes and edges`);
  }
  if (scene.type === 'tree') {
    if (!Array.isArray(scene.levels) || scene.levels.length < 1 || scene.levels.some((level) => !Array.isArray(level) || level.length < 1)) fail(`${label}: tree needs nonempty levels`);
    checkMarks(scene.marks, scene.levels.flat().length, 1, label);
  }
  if (scene.type === 'intervals') {
    if (!Array.isArray(scene.items) || scene.items.length < 1) fail(`${label}: intervals need items`);
    for (const item of scene.items ?? []) if (!nonEmpty(item.label) || !Number.isFinite(item.start) || !Number.isFinite(item.end) || item.end < item.start || !allowedTones.has(item.tone ?? 'neutral')) fail(`${label}: invalid interval item`);
  }
  if (scene.type === 'linked') {
    if (!Array.isArray(scene.nodes) || scene.nodes.length < 1 || scene.nodes.some((node) => !nonEmpty(node.value))) fail(`${label}: linked scene needs labelled nodes`);
  }
  if (scene.type === 'trie') {
    if (!Array.isArray(scene.paths) || scene.paths.length < 1 || scene.paths.some((item) => !nonEmpty(item.word) || !nonEmpty(item.prefix))) fail(`${label}: trie scene needs word paths`);
  }
  if (scene.type === 'shapes') {
    if (!Array.isArray(scene.items) || scene.items.length < 2 || scene.items.some((item) => !nonEmpty(item))) fail(`${label}: shape scene needs at least two labelled shapes`);
  }
  if (scene.type === 'buckets') {
    if (!Array.isArray(scene.items) || scene.items.length < 1 || scene.items.some((item) => !nonEmpty(item.count) || !Array.isArray(item.items))) fail(`${label}: bucket scene needs count and items`);
  }
  if (scene.type === 'prefix') {
    if (!Array.isArray(scene.items) || !Array.isArray(scene.left) || !Array.isArray(scene.right) || !Array.isArray(scene.answer)) fail(`${label}: prefix scene needs all rows`);
  }
  if (scene.type === 'dual-window') {
    if (!Array.isArray(scene.items) || !Array.isArray(scene.windows) || scene.windows.length < 2) fail(`${label}: dual-window scene needs two windows`);
    for (const window of scene.windows ?? []) if (!nonEmpty(window.label) || !Array.isArray(window.range) || window.range.length !== 2 || window.range[0] < 0 || window.range[1] >= scene.items.length || window.range[0] > window.range[1]) fail(`${label}: invalid window range`);
  }
  if (scene.type === 'choices') {
    if (!Array.isArray(scene.path) || !Array.isArray(scene.branches) || scene.branches.length < 1) fail(`${label}: choice scene needs path and branches`);
  }
  if (scene.type === 'lru') {
    if (!Array.isArray(scene.map) || !Array.isArray(scene.order) || scene.order.length < 1) fail(`${label}: LRU scene needs map and order`);
  }
  if (scene.type === 'heap') {
    if (!Array.isArray(scene.values) || scene.values.length < 1 || scene.values.some((value) => !nonEmpty(value))) fail(`${label}: heap scene needs values`);
  }
}

const slugs = Object.keys(codingQuestionVisuals);
if (slugs.length !== 106) fail(`expected 106 visual definitions, found ${slugs.length}`);
for (const slug of slugs) {
  const definition = codingQuestionVisuals[slug];
  const label = `visuals/${slug}`;
  if (!nonEmpty(definition.objective)) fail(`${label}: missing objective`);
  if (!Array.isArray(definition.frames) || definition.frames.length < 3) fail(`${label}: needs at least three frames`);
  if (!nonEmpty(definition.frames.at(-1)?.scene?.result)) fail(`${label}: final frame needs an explicit result`);
  for (const [index, item] of (definition.frames ?? []).entries()) {
    if (!nonEmpty(item.label) || !nonEmpty(item.note)) fail(`${label}/frame-${index}: missing label or note`);
    checkScene(item.scene, `${label}/frame-${index}`);
  }
  const articlePath = path.join(postsDir, `2026-09-01-${slug}.md`);
  const auditPath = path.join(auditsDir, `${slug}.json`);
  if (!fs.existsSync(articlePath)) fail(`${label}: missing generated article`);
  if (!fs.existsSync(auditPath)) fail(`${label}: missing audit sidecar`);
  if (fs.existsSync(articlePath)) {
    const article = fs.readFileSync(articlePath, 'utf8');
    const frameCount = (article.match(/data-coding-frame="\d+"/g) ?? []).length;
    if (frameCount !== definition.frames.length) fail(`${label}: generated frame count ${frameCount} differs from ${definition.frames.length}`);
    if (!article.includes(`<!-- visual:${slug}-state -->`)) fail(`${label}: marker missing`);
  }
  if (fs.existsSync(auditPath)) {
    const audit = JSON.parse(fs.readFileSync(auditPath, 'utf8'));
    if (audit.learningObjective !== definition.objective) fail(`${label}: audit objective drift`);
  }
}

if (failures.length > 0) {
  console.error(`Coding visual check failed with ${failures.length} issue(s):`);
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`Coding visual check passed: ${slugs.length} problem-specific definitions, frames, generated pages, and audits are synchronized.`);

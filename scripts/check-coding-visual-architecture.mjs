import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const problemsDir = path.join(root, 'scripts/coding-visuals/problems');

assert.ok(fs.existsSync(problemsDir), 'per-problem visual module directory must exist');
const modules = fs.readdirSync(problemsDir).filter((name) => name.endsWith('.mjs')).sort();
assert.equal(modules.length, 106, 'exactly 106 per-problem modules are required');

const definitions = [];
for (const moduleName of modules) {
  const module = await import(path.join(problemsDir, moduleName));
  assert.equal(typeof module.default, 'object', `${moduleName} must default-export one definition`);
  assert.equal(module.default.slug, moduleName.replace(/\.mjs$/, ''), `${moduleName} slug must match its filename`);
  definitions.push(module.default);
}

const { codingQuestionVisuals } = await import('./coding-visuals/index.mjs');
const compatibility = await import('./coding-question-visuals.mjs');
assert.equal(Object.keys(codingQuestionVisuals).length, 106);
assert.deepEqual(compatibility.codingQuestionVisuals, codingQuestionVisuals);

const metadataFields = [
  'pattern',
  'recognitionCue',
  'invariant',
  'stateModel',
  'visualRationale',
  'rejectedAlternatives',
  'transferLesson',
  'reviewStatus',
];
for (const definition of definitions) {
  for (const field of metadataFields) assert.ok(definition.review?.[field], `${definition.slug} needs review.${field}`);
  assert.ok(['pending', 'reviewed'].includes(definition.review.reviewStatus), `${definition.slug} has an invalid review status`);
  for (const frame of definition.frames) {
    assert.ok(frame.key, `${definition.slug} frame needs a stable key`);
    assert.ok(
      frame.scene.motion?.some((item) => item.key),
      `${definition.slug}/${frame.key} needs at least one stable motion key`,
    );
  }
}

const generator = fs.readFileSync(path.join(root, 'scripts/generate-coding-question-book.mjs'), 'utf8');
const checker = fs.readFileSync(path.join(root, 'scripts/check-coding-visuals.mjs'), 'utf8');
const interactions = fs.readFileSync(path.join(root, 'scripts/check-coding-visual-interactions.mjs'), 'utf8');
const client = fs.readFileSync(path.join(root, 'src/utils/codingVisuals.ts'), 'utf8');

assert.match(generator, /--slugs/);
assert.match(checker, /--slugs/);
assert.match(checker, /--require-reviewed/);
assert.match(generator, /data-motion-key/);
assert.match(generator, /<svg/);
assert.match(generator, /coding-trace-edge-line/);
assert.match(generator, /coding-trace-heap-edge/);
assert.match(client, /keydown/);
assert.match(client, /prefers-reduced-motion/);
assert.match(client, /data-motion-key/);
assert.match(interactions, /no-JS/);
assert.match(interactions, /print/);
assert.ok(fs.existsSync(path.join(root, 'docs/CODING_VISUAL_STANDARD.md')));

console.log('Coding visual architecture check passed.');

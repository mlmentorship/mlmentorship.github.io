import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = process.cwd();
const registryPath = path.join(root, 'src/utils/articleVisualTraces.ts');
const componentPath = path.join(root, 'src/components/ArticleVisualTrace.astro');
const scenePath = path.join(root, 'src/components/ArticleTraceScene.astro');
const layoutPath = path.join(root, 'src/layouts/DocsLayout.astro');
const expectedSlugs = [
  'tokenization',
  'k-means-clustering',
  'backpropagation',
  'pipeline-parallelism',
  'continuous-batching',
  'speculative-decoding',
  'activation-functions',
  'attention-mechanism',
  'batchnorm-vs-layernorm',
  'svd-and-pca',
  'roc-pr-auc',
  'calibration',
];
const expectedVisualIds = {
  tokenization: 'bpe-corpus-merge-trace',
  'k-means-clustering': 'kmeans-assign-then-update',
  backpropagation: 'backprop-forward-reverse-trace',
  'pipeline-parallelism': 'pipeline-fill-drain-bubble',
  'continuous-batching': 'continuous-batching-reuses-finished-slot',
  'speculative-decoding': 'speculative-decoding-first-rejection-boundary',
  'activation-functions': 'activation-gradient-regions',
  'attention-mechanism': 'attention-keys-choose-values-contribute',
  'batchnorm-vs-layernorm': 'normalization-shared-statistics',
  'svd-and-pca': 'pca-rank-one-projection',
  'roc-pr-auc': 'roc-pr-same-operating-point',
  calibration: 'calibration-reliability-gap',
};

assert.ok(fs.existsSync(registryPath), 'article trace registry must exist');
assert.ok(fs.existsSync(componentPath), 'article trace figure component must exist');
assert.ok(fs.existsSync(scenePath), 'article trace scene component must exist');

const { articleVisualTraces } = await import(pathToFileURL(registryPath).href);
assert.deepEqual(Object.keys(articleVisualTraces).sort(), [...expectedSlugs].sort(), 'the first article trace batch must stay explicit');
const fallbackPath = path.join(root, 'src/utils/articleVisualTraceFallbacks.ts');
assert.ok(fs.existsSync(fallbackPath), 'generated article trace fallback catalog must exist');
const { articleVisualTraceFallbacks } = await import(pathToFileURL(fallbackPath).href);

const collectKeys = (value, keys = []) => {
  if (Array.isArray(value)) {
    for (const item of value) collectKeys(item, keys);
  } else if (value && typeof value === 'object') {
    if (typeof value.key === 'string') keys.push(value.key);
    for (const child of Object.values(value)) collectKeys(child, keys);
  }
  return keys;
};

for (const slug of expectedSlugs) {
  const definition = articleVisualTraces[slug];
  assert.equal(definition.slug, slug, `${slug} slug must match its registry key`);
  assert.equal(definition.visualId, expectedVisualIds[slug], `${slug} visual ID must match its audit`);
  assert.ok(definition.objective && definition.example, `${slug} needs an objective and worked example`);
  assert.equal(definition.traceKind, 'mechanism', `${slug} explicit traces must be mechanism traces`);
  assert.ok(definition.frames.length >= 3, `${slug} needs at least three authored states`);
  assert.equal(new Set(definition.frames.map((frame) => frame.key)).size, definition.frames.length, `${slug} frame keys must be unique`);
  for (const frame of definition.frames) {
    assert.ok(frame.label && frame.note, `${slug}/${frame.key} needs a label and transition note`);
    assert.ok(['lanes', 'table', 'evidence', 'plot', 'flow', 'grid', 'schedule', 'speculative'].includes(frame.scene.type), `${slug}/${frame.key} has an unsupported scene type`);
    assert.ok(collectKeys(frame.scene).length > 0, `${slug}/${frame.key} needs stable entity keys`);
  }
  for (const field of ['recognitionCue', 'invariant', 'transferLesson']) {
    assert.ok(definition.review[field], `${slug} needs review.${field}`);
  }

  const auditPath = path.join(root, 'data/visual-audits', `${slug}.json`);
  const audit = JSON.parse(fs.readFileSync(auditPath, 'utf8'));
  assert.ok(audit.implementation?.visualIds?.includes(definition.visualId), `${slug} trace must use the audited visual ID`);
}

const expectedFallbackSlugs = [];
for (const filename of fs.readdirSync(path.join(root, 'data/visual-audits')).filter((name) => name.endsWith('.json'))) {
  const audit = JSON.parse(fs.readFileSync(path.join(root, 'data/visual-audits', filename), 'utf8'));
  const source = fs.readFileSync(path.join(root, audit.article), 'utf8');
  if (audit.status === 'implemented' && !expectedSlugs.includes(audit.slug) && !source.includes('data-coding-visual')) expectedFallbackSlugs.push(audit.slug);
}
assert.deepEqual(Object.keys(articleVisualTraceFallbacks).sort(), expectedFallbackSlugs.sort(), 'fallback catalog must cover every remaining static article exactly once');
for (const [slug, definition] of Object.entries(articleVisualTraceFallbacks)) {
  assert.equal(definition.slug, slug, `${slug} fallback slug must match its registry key`);
  const audit = JSON.parse(fs.readFileSync(path.join(root, 'data/visual-audits', `${slug}.json`), 'utf8'));
  assert.equal(definition.visualId, audit.implementation.visualIds[0], `${slug} fallback visual ID must match its audit`);
  assert.equal(definition.traceKind, 'evidence', `${slug} fallback traces must identify as evidence traces`);
  assert.ok(definition.frames.length >= 3, `${slug} fallback needs at least three authored evidence states`);
  for (const frame of definition.frames) {
    assert.equal(frame.scene.type, 'evidence', `${slug}/${frame.key} fallback must preserve evidence stages`);
    assert.equal(frame.scene.stages.length, 3, `${slug}/${frame.key} fallback needs three evidence stages`);
    assert.ok(frame.scene.stages.every((stage) => stage.key && stage.label && stage.value), `${slug}/${frame.key} fallback stages need stable keys and values`);
  }
  for (const field of ['recognitionCue', 'invariant', 'transferLesson']) assert.ok(definition.review[field], `${slug} fallback needs review.${field}`);
}

const component = fs.readFileSync(componentPath, 'utf8');
const scene = fs.readFileSync(scenePath, 'utf8');
const layout = fs.readFileSync(layoutPath, 'utf8');
assert.match(component, /data-coding-visual/);
assert.match(component, /data-coding-frame/);
assert.match(component, /data-coding-frame-button/);
assert.match(component, /data-article-trace/);
assert.match(scene, /data-motion-key/);
assert.match(scene, /role="img"/);
assert.match(scene, /article-trace-evidence/);
assert.match(layout, /getArticleVisualTrace/);
assert.match(layout, /<ArticleVisualTrace definition=/);

console.log(`Article visual trace check passed: ${expectedSlugs.length} authored article traces.`);
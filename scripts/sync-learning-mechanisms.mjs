import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import yaml from 'js-yaml';
import { fromMarkdown } from 'mdast-util-from-markdown';
import GithubSlugger from 'github-slugger';
import { parseFragment } from 'parse5';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const registryPath = path.join(root, 'src/data/learningMechanisms.json');
const baselinePath = path.join(root, 'data/pattern-first-baseline.json');
const defaultPlan = path.resolve(root, '../ml_interview_book/docs/mlmentorship-transformation-20261023/source-map.json');
const hash = (value) => crypto.createHash('sha256').update(value).digest('hex');
const serialize = (value) => `${JSON.stringify(value, null, 2)}\n`;
const git = (...args) => execFileSync('git', ['-C', root, ...args], {
  encoding: 'utf8', env: { ...process.env, GIT_OPTIONAL_LOCKS: '0' },
}).trim();

export function validatePlan(plan, expectedCount = 391) {
  assert.equal(plan.pieces.length, expectedCount, 'Article count differs');
  const pieces = new Map(plan.pieces.map((piece) => [piece.slug, piece]));
  assert.equal(pieces.size, expectedCount, 'Duplicate article');
  const families = new Map(plan.families.map((family) => [family.id, family]));
  assert.equal(families.size, plan.families.length, 'Duplicate family');
  for (const family of families.values()) {
    assert(family.members.includes(family.owner), `Foreign owner: ${family.id}`);
    assert.equal(new Set(family.members).size, family.members.length, `Duplicate member: ${family.id}`);
    for (const slug of family.members) {
      const piece = pieces.get(slug);
      assert(piece && piece.family === family.id && piece.canonical_owner === family.owner, `Invalid membership: ${slug}`);
    }
  }
  const visiting = new Set();
  const visited = new Set();
  function visit(slug) {
    assert(!visiting.has(slug), `Prerequisite cycle: ${slug}`);
    if (visited.has(slug)) return;
    const piece = pieces.get(slug);
    assert(piece, `Missing article: ${slug}`);
    assert.equal(piece.original_route, piece.destination_route, `Route change: ${slug}`);
    assert.equal(piece.source_path, piece.destination_source_path, `Source move: ${slug}`);
    const family = families.get(piece.family);
    assert(family?.members.includes(slug), `Unassigned article: ${slug}`);
    visiting.add(slug);
    for (const dependency of piece.required_sources) visit(dependency);
    for (const related of piece.related_sources) assert(pieces.has(related), `Unknown contrast: ${related}`);
    visiting.delete(slug);
    visited.add(slug);
  }
  for (const slug of pieces.keys()) visit(slug);
  assert.equal(new Set(plan.reading_order).size, expectedCount, 'Incomplete reading order');
  const positions = new Map(plan.reading_order.map((slug, index) => [slug, index]));
  for (const piece of pieces.values()) {
    assert(positions.has(piece.slug), `Missing reading position: ${piece.slug}`);
    for (const dependency of piece.required_sources) {
      assert(positions.get(dependency) < positions.get(piece.slug), `Invalid reading order: ${piece.slug}`);
    }
  }
}

export function identities(source) {
  const match = source.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/);
  assert(match, 'Missing frontmatter');
  const metadata = yaml.load(match[1]);
  const body = source.slice(match[0].length);
  const tree = fromMarkdown(body);
  const slugger = new GithubSlugger();
  const headingIds = [];
  const explicitIds = [];
  const frames = [];
  const motionKeys = [];
  const visuals = [];
  const python = [];
  function plain(node) {
    return node.type === 'inlineCode' || node.type === 'text' ? node.value : (node.children ?? []).map(plain).join('');
  }
  function htmlWalk(node) {
    const attributes = Object.fromEntries((node.attrs ?? []).map((attribute) => [attribute.name, attribute.value]));
    if (attributes.id) explicitIds.push(attributes.id);
    if ('data-frame-key' in attributes) frames.push(attributes['data-frame-key']);
    if ('data-motion-key' in attributes) motionKeys.push(attributes['data-motion-key']);
    for (const child of node.childNodes ?? []) htmlWalk(child);
  }
  function walk(node) {
    if (node.type === 'heading') headingIds.push(slugger.slug(plain(node)));
    if (node.type === 'html') {
      htmlWalk(parseFragment(node.value));
      for (const match of node.value.matchAll(/<!--\s*visual:([\w-]+)\s*-->/g)) visuals.push(match[1]);
    }
    if (node.type === 'code' && node.lang === 'python') python.push(hash(node.value));
    for (const child of node.children ?? []) walk(child);
  }
  walk(tree);
  return { metadata, heading_ids: headingIds, explicit_ids: explicitIds, visual_ids: visuals, frame_keys: frames, motion_keys: motionKeys, python_sha256: python };
}

export function makeRegistry(plan) {
  return {
    schemaVersion: 1,
    families: plan.families.map((family) => ({
      id: family.id, title: family.title, owner: family.owner, members: family.members,
    })),
    pieces: plan.pieces.map((piece) => ({
      slug: piece.slug, title: piece.destination_title, href: piece.destination_route,
      family: piece.family, owner: piece.canonical_owner, treatment: piece.treatment,
      required: piece.required_sources, related: piece.related_sources,
      budget: piece.target_words,
    })),
    readingOrder: plan.reading_order,
  };
}

export function assertCodeRetained(original, current, slug) {
  const remaining = [...current];
  for (const digest of original) {
    const index = remaining.indexOf(digest);
    assert(index >= 0, `Lost original Python fence: ${slug}/${digest}`);
    remaining.splice(index, 1);
  }
}

export function assertMotionRetained(original, current, slug) {
  const expected = [...original];
  if (slug === 'merge-two-sorted-lists') {
    assert.deepEqual(expected.slice(-3), ['A7', 'B8', 'pointer-tail'], 'Unexpected merge baseline');
    expected.splice(-3, 3, 'A7', 'pointer-tail', 'B8', 'pointer-second');
  }
  assert.deepEqual(current, expected, `Lost motion key: ${slug}`);
}

export function checkPreservation(baseline) {
  for (const piece of baseline.pieces) {
    const current = identities(fs.readFileSync(path.join(root, piece.source_path), 'utf8'));
    for (const [key, value] of Object.entries(piece.identities.metadata)) {
      if (['title', 'description', 'updated', 'reviewed'].includes(key)) continue;
      assert.deepEqual(current.metadata[key], value, `Metadata changed: ${piece.slug}/${key}`);
    }
    const ids = new Set([...current.heading_ids, ...current.explicit_ids]);
    for (const id of [...piece.identities.heading_ids, ...piece.identities.explicit_ids]) {
      assert(ids.has(id), `Lost fragment: ${piece.slug}#${id}`);
    }
    assert.deepEqual(current.visual_ids, piece.identities.visual_ids, `Lost visual: ${piece.slug}`);
    assert.deepEqual(current.frame_keys, piece.identities.frame_keys, `Lost frame: ${piece.slug}`);
    assertMotionRetained(piece.identities.motion_keys, current.motion_keys, piece.slug);
    assertCodeRetained(piece.identities.python_sha256, current.python_sha256, piece.slug);
  }
}

function selftest() {
  const source = '---\ntitle: "Fixture"\ncategory: concepts\naliases: ["old"]\n---\n## Repeated\n\n## Repeated\n\n<!-- visual:fixture -->\n<figure id="geometry"><div data-frame-key="first" data-motion-key="step"></div></figure>\n\n```python\nprint(2)\n```\n';
  const captured = identities(source);
  assert.deepEqual(captured.heading_ids, ['repeated', 'repeated-1']);
  assert.deepEqual(captured.explicit_ids, ['geometry']);
  assert.deepEqual(captured.frame_keys, ['first']);
  assert.deepEqual(captured.motion_keys, ['step']);
  assert.deepEqual(captured.visual_ids, ['fixture']);
  assert.equal(captured.python_sha256.length, 1);
  assertCodeRetained(['original'], ['new', 'original'], 'fixture');
  assert.throws(() => assertCodeRetained(['original'], ['changed'], 'fixture'), /Lost original Python fence/);
  assert.throws(() => assertCodeRetained(['original', 'original'], ['original'], 'fixture'), /Lost original Python fence/);
  assertMotionRetained(['unchanged'], ['unchanged'], 'fixture');
  assert.throws(() => assertMotionRetained(['unchanged'], ['different'], 'fixture'), /Lost motion key/);
  const mergeBefore = ['prefix', 'A7', 'B8', 'pointer-tail'];
  const mergeAfter = ['prefix', 'A7', 'pointer-tail', 'B8', 'pointer-second'];
  assertMotionRetained(mergeBefore, mergeAfter, 'merge-two-sorted-lists');
  assert.throws(() => assertMotionRetained(mergeBefore, mergeBefore, 'merge-two-sorted-lists'), /Lost motion key/);
  assert.throws(() => assertMotionRetained(mergeBefore, mergeAfter.slice(1), 'merge-two-sorted-lists'), /Lost motion key/);
  const piece = { slug: 'first', source_path: 'first.md', destination_source_path: 'first.md', original_route: '/first/', destination_route: '/first/', family: 'only', canonical_owner: 'first', required_sources: [], related_sources: [] };
  const plan = { pieces: [piece], families: [{ id: 'only', owner: 'first', members: ['first'] }], reading_order: ['first'] };
  validatePlan(plan, 1);
  assert.throws(() => validatePlan({ ...plan, pieces: [{ ...piece, required_sources: ['first'] }] }, 1), /cycle/);
  assert.throws(() => validatePlan({ ...plan, families: [{ ...plan.families[0], owner: 'foreign' }] }, 1), /Foreign owner/);
  assert.throws(() => validatePlan({ ...plan, pieces: [{ ...piece, destination_route: '/changed/' }] }, 1), /Route change/);
  console.log('Learning contracts selftest passed: structured identities, original code, foreign owners, cycles and route changes.');
}

function main() {
  if (process.argv.includes('--self-test')) return selftest();
  const planPath = process.argv.find((value) => value.startsWith('--plan='))?.slice(7) ?? defaultPlan;
  if (process.argv.includes('--check')) {
    const baseline = JSON.parse(fs.readFileSync(baselinePath, 'utf8'));
    const registrySource = fs.readFileSync(registryPath, 'utf8');
    const registry = JSON.parse(registrySource);
    const originals = new Map(baseline.pieces.map((piece) => [piece.slug, piece]));
    const plan = {
      families: registry.families,
      reading_order: registry.readingOrder,
      pieces: registry.pieces.map((piece) => {
        const original = originals.get(piece.slug);
        assert(original, `Unexpected article: ${piece.slug}`);
        return { slug: piece.slug, family: piece.family, canonical_owner: piece.owner,
          source_path: original.source_path, destination_source_path: original.source_path,
          original_route: `/${original.identities.metadata.category}/${piece.slug}/`, destination_route: piece.href,
          required_sources: piece.required, related_sources: piece.related };
      }),
    };
    validatePlan(plan);
    checkPreservation(baseline);
    if (fs.existsSync(planPath)) {
      const approvedSource = fs.readFileSync(planPath, 'utf8');
      assert.equal(hash(approvedSource), baseline.planSha256, 'Approved plan changed');
      assert.equal(registrySource, serialize(makeRegistry(JSON.parse(approvedSource))), 'Stale learning registry');
    }
    console.log(`Learning contracts passed: ${plan.pieces.length} articles, ${plan.families.length} owners; current identities preserved.`);
    return;
  }
  const planSource = fs.readFileSync(planPath, 'utf8');
  const plan = JSON.parse(planSource);
  validatePlan(plan);
  const registry = serialize(makeRegistry(plan));
  if (process.argv.includes('--capture')) {
    assert(!fs.existsSync(baselinePath), 'Baseline exists; refusing to overwrite it');
    const head = git('rev-parse', 'HEAD');
    const pieces = plan.pieces.map((piece) => {
      const current = fs.readFileSync(path.join(root, piece.source_path), 'utf8');
      const committed = execFileSync('git', ['-C', root, 'show', `${head}:${piece.source_path}`], { encoding: 'utf8' });
      return { slug: piece.slug, source_path: piece.source_path, frozen_sha256: piece.source_sha256,
        committed_sha256: hash(committed), authoring_sha256: hash(current), identities: identities(current) };
    });
    fs.writeFileSync(baselinePath, serialize({ schemaVersion: 1, capturedUtc: new Date().toISOString(), head,
      planSha256: hash(planSource), reconciliation: 'Retain current committed corrections and the existing determinant pilot; never restore frozen defects.', pieces }), { flag: 'wx' });
  }
  assert(fs.existsSync(baselinePath), 'Capture the approved current baseline before generation');
  const baseline = JSON.parse(fs.readFileSync(baselinePath, 'utf8'));
  assert.equal(baseline.planSha256, hash(planSource), 'Approved plan changed');
  assert.deepEqual(baseline.pieces.map((piece) => piece.slug), plan.pieces.map((piece) => piece.slug), 'Baseline coverage changed');
  checkPreservation(baseline);
  fs.writeFileSync(registryPath, registry);
  console.log(`Learning contracts passed: ${plan.pieces.length} articles, ${plan.families.length} owners; current identities preserved.`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
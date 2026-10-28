import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fromMarkdown } from 'mdast-util-from-markdown';
import { parseFragment } from 'parse5';
import { codingQuestionVisuals } from './coding-question-visuals.mjs';
import { renderStateTransition } from './coding-visuals/state-transition.mjs';
import { redraws, supplementalRedraws, renderRedraw } from './illustration-redraws.mjs';
import { checkPreservation } from './sync-learning-mechanisms.mjs';

const root = path.resolve(import.meta.dirname, '..');
const evidence = path.resolve(process.argv.find((argument) => argument.startsWith('--evidence-dir='))?.slice('--evidence-dir='.length) ?? path.join(root, 'scratch/illustration-edition'));
const baselinePath = path.join(evidence, 'preservation-baseline.json');
const review = JSON.parse(fs.readFileSync(path.join(root, 'data/visual-comprehension-review.json'), 'utf8'));
const audits = fs.readdirSync(path.join(root, 'data/visual-audits')).filter((name) => name.endsWith('.json'))
  .map((name) => JSON.parse(fs.readFileSync(path.join(root, 'data/visual-audits', name), 'utf8')));
const owners = new Map(audits.flatMap((audit) => (audit.implementation?.visualIds ?? []).map((identity) => [identity, audit])));
const preservedFragments = new Map(JSON.parse(fs.readFileSync(path.join(root, 'data/pattern-first-baseline.json'), 'utf8')).pieces.map((piece) => [piece.source_path, piece.identities.explicit_ids]));
const hash = (value) => createHash('sha256').update(value).digest('hex');
const readerQuestions = {
  'quantization-scale-granularity': 'Why does the shared scale collapse -0.5 and +0.5? Why does the channel-specific scale preserve their signs?',
  'ols-orthogonal-projection': 'Why is the residual orthogonal to X even when the coefficient vector is not unique?',
  'importance-weights-effective-sample-size': 'Why are there five actual draws but only 2.5 effective samples?',
  'vae-training-and-generation-paths': 'Why does generation need neither an observed x nor q(z|x), unlike training?',
  'residual-add-gradient-split': 'Why does the add node copy 2.2 into both branches? Why does the residual branch return only 0.22?',
  'cold-start-signal-handoff': 'Why do two successes and one failure yield Beta(3,2), not certainty about the save rate?',
  'weight-decay-adaptive-paths': 'Why does Adam scale equal L2 penalties differently, while AdamW applies equal direct decay fractions?',
  'loss-estimation-vs-decision': 'Why can probability 0.8 favor a negative action under these costs?',
  'logit-odds-probability-map': 'Why does z increasing from 0 to 1 multiply odds by 2.72 rather than add fixed probability?',
  'gpu-roofline-batching-ridge': 'Why does halving HBM traffic double this ceiling without proving a measured speedup?',
  'bm25-saturation-length-penalty': 'Why does term-frequency gain diminish? Why does a longer document reduce the factor at fixed frequency?',
  'bellman-stochastic-backup': 'Why average branch returns as 0.75 times 2 plus 0.25 times 1, instead of taking their maximum?',
  'distribution-observation-unit': 'Why use Poisson for event counts but Exponential for waiting until the next event?',
  'minimum-window-substring-state': 'Why can the first A and B be removed? Why does removing A at index 2 invalidate the window?',
  'pad-variable-length-sequences-state': 'How does the mask distinguish a real zero-valued token from zero-valued padding?',
  'partition-equal-subset-sum-state': 'Why can the second 5 create ten, but the first 5 cannot reuse itself?',
  'permutation-in-string-state': 'Why does ba qualify regardless of order? Why does a longer window with these letters fail this test?',
  'rotting-oranges-state': 'Why can an orange scheduled for time four not spread at time three?',
  'set-matrix-zeroes-state': 'Why does a marker zero not imply that the original first column was entirely zero?',
  'word-search-ii-state': 'Why can the oat terminal be removed while its prefix must remain for oath?',
  'moe-top-two-routing-accounting': 'Why execute E1 and E3 while retaining stored weights for E2 and E4?',
  'potential-shaping-cancels-backtracking': 'Why does revisiting yield zero shaping return here, but positive return under an entry bonus?',
  'robotics-policy-dagger-recovery': 'Why must the expert label the learner-visited state, not just the original demonstration state?',
  'two-tower-cross-encoder-fanout': 'Why jointly encode only K pairs? Can reranking recover an item outside those candidates?',
  'speculative-decoding-weight-read-ledger': 'Why can expensive drafting or a short accepted prefix erase verification savings?',
};

function htmlNodes(source) {
  const found = [];
  function walkHtml(node, offset) {
    const attributes = Object.fromEntries((node.attrs ?? []).map((item) => [item.name, item.value]));
    found.push({ node, attributes, offset });
    for (const child of node.childNodes ?? []) walkHtml(child, offset);
  }
  function walkMarkdown(node) {
    if (node.type === 'html') walkHtml(parseFragment(node.value, { sourceCodeLocationInfo: true }), node.position.start.offset);
    for (const child of node.children ?? []) walkMarkdown(child);
  }
  walkMarkdown(fromMarkdown(source));
  return found;
}

function identities(source) {
  const markdown = fromMarkdown(source);
  const python = [];
  const headings = [];
  function walk(node) {
    if (node.type === 'code' && ['python', 'py'].includes(node.lang)) python.push(node.value);
    if (node.type === 'heading') headings.push(source.slice(node.position.start.offset, node.position.end.offset));
    for (const child of node.children ?? []) walk(child);
  }
  walk(markdown);
  const html = htmlNodes(source);
  return {
    python,
    headings,
    visuals: [...source.matchAll(/<!-- visual:([^ ]+) -->/g)].map((match) => match[1]),
    supplemental: html.flatMap(({ attributes }) => attributes['data-supplemental-visual'] ? [attributes['data-supplemental-visual']] : []),
    frames: html.flatMap(({ attributes }) => attributes['data-frame-key'] ? [attributes['data-frame-key']] : []),
    motion: html.flatMap(({ attributes }) => attributes['data-motion-key'] ? [attributes['data-motion-key']] : []),
  };
}

function replaceRanges(source, replacements) {
  const ordered = [...replacements].sort((left, right) => right.start - left.start);
  let boundary = source.length;
  for (const replacement of ordered) {
    assert(replacement.end <= boundary, 'Illustration edits must not overlap');
    source = source.slice(0, replacement.start) + replacement.value + source.slice(replacement.end);
    boundary = replacement.start;
  }
  return source;
}

function capture() {
  assert(!fs.existsSync(baselinePath), 'Never replace the original preservation receipt');
  const files = Object.fromEntries(audits.map((audit) => {
    const source = fs.readFileSync(path.join(root, audit.article), 'utf8');
    return [audit.article, { sha256: hash(source), identities: identities(source) }];
  }));
  const protectedPaths = ['data/pattern-first-baseline.json', '../ml_interview_book/docs/dsa/Coding_Questions_Phone_Guide.md'];
  const protectedFiles = Object.fromEntries(protectedPaths.filter((file) => fs.existsSync(path.resolve(root, file)))
    .map((file) => [file, hash(fs.readFileSync(path.resolve(root, file)))]));
  fs.mkdirSync(evidence, { recursive: true });
  fs.writeFileSync(baselinePath, `${JSON.stringify({ note: 'Captured after the isolated all-reduce pilot; includes preexisting user edits.', files, protectedFiles }, null, 2)}\n`, { flag: 'wx' });
  console.log(`Preservation baseline: ${Object.keys(files).length} articles; protected source hashes recorded.`);
}

function verify() {
  const baseline = JSON.parse(fs.readFileSync(baselinePath, 'utf8'));
  let frames = 0;
  let motion = 0;
  for (const [file, before] of Object.entries(baseline.files)) {
    const after = identities(fs.readFileSync(path.join(root, file), 'utf8'));
    assert.deepEqual(after, before.identities, `${file}: lesson/code/visual/frame/motion identities changed`);
    frames += after.frames.length;
    motion += after.motion.length;
  }
  for (const [file, before] of Object.entries(baseline.protectedFiles)) {
    assert.equal(hash(fs.readFileSync(path.resolve(root, file))), before, `${file}: protected source changed`);
  }
  checkPreservation(JSON.parse(fs.readFileSync(path.join(root, 'data/pattern-first-baseline.json'), 'utf8')));
  console.log(`Preservation passed: ${Object.keys(baseline.files).length} articles, ${frames} frames, ${motion} ordered motion identities, all headings/Python/visual identities unchanged.`);
}

function applyCoding() {
  assert(fs.existsSync(baselinePath), 'Capture preservation evidence before generation');
  let count = 0;
  let frames = 0;
  for (const definition of Object.values(codingQuestionVisuals)) {
    const audit = owners.get(`${definition.slug}-state`);
    assert(audit, `No owner for ${definition.slug}`);
    const file = path.join(root, audit.article);
    const source = fs.readFileSync(file, 'utf8');
    const replacements = [];
    const checkpoints = htmlNodes(source).filter(({ attributes }) => attributes['data-coding-frame'] !== undefined);
    assert.equal(checkpoints.length, definition.frames.length, `${definition.slug}: checkpoint count`);
    for (const { node, attributes, offset } of checkpoints) {
      const index = Number(attributes['data-coding-frame']);
      const item = definition.frames[index];
      assert.equal(attributes['data-frame-key'], item.key, `${definition.slug}: checkpoint identity`);
      const location = node.sourceCodeLocation;
      const generated = renderStateTransition(item.scene, definition.frames[index - 1]?.scene);
      const oldMechanism = (node.childNodes ?? []).find((child) => child.attrs?.some((attribute) => attribute.name === 'class' && attribute.value === 'coding-mechanism'));
      const existing = (node.childNodes ?? []).find((child) => child.attrs?.some((attribute) => attribute.name === 'class' && attribute.value === 'coding-state-comparison'));
      if (existing) {
        const previous = existing.sourceCodeLocation;
        const start = offset + (oldMechanism?.sourceCodeLocation.startOffset ?? previous.startOffset);
        const end = offset + previous.endOffset;
        if (source.slice(start, end) !== generated) replacements.push({ start, end, value: generated });
      } else if (generated) {
        replacements.push({ start: offset + location.endTag.startOffset, end: offset + location.endTag.startOffset, value: generated });
      }
      frames++;
    }
    if (replacements.length) {
      fs.writeFileSync(file, replaceRanges(source, replacements));
      count++;
    }
  }
  console.log(`Retained-state comparisons applied to ${count} illustrations; ${frames} checkpoint identities checked. Existing scene drawings and lesson bodies retained.`);
  verify();
}

function reconcile() {
  const definitions = Object.values(codingQuestionVisuals);
  review.implementationStatus = 'Editorial edition implemented; see editionReconciliation for scope and qualification. Original assessments and previous qualification remain historical.';
  review.editionReconciliation = {
    status: 'implemented; final qualification pending',
    registeredVisuals: owners.size,
    supplementalVisuals: Object.keys(supplementalRedraws).length,
    articleCount: audits.length,
    commonImprovements: ['Editorial figure typography and responsive composition', 'A figure-specific inference question on every registered caption'],
    registeredRedraws: Object.entries(redraws).map(([visualId, model]) => ({ visualId, slug: owners.get(visualId).slug, change: model.title, readerInference: model.caption })),
    supplementalRedraws: Object.entries(supplementalRedraws).map(([visualId, model]) => ({ visualId, slug: model.slug, change: model.title, readerInference: model.caption })),
    manualRevisions: [
      { visualId: 'hierarchical-all-reduce-path', change: 'Rank values 1,3 and 2,4 form local partials 4 and 6; the slow-boundary crossing produces 10 and all four ranks receive 10. This is a logical reduction, not a measured physical packet schedule.' },
      { visualId: 'bayes-positive-test-denominator', change: 'The exact positive-test denominator has count-proportional 99 and 999 segments over 1098 positives; the posterior is about 9%, not the test sensitivity.' },
    ],
    coding: {
      definitions: definitions.length,
      checkpoints: definitions.reduce((sum, definition) => sum + definition.frames.length, 0),
      comparisonPolicy: 'Selected values shown at consecutive authored checkpoints. The first is a snapshot; an absent field means not shown, not deletion. No intervening execution is invented.',
      mechanismSpecificRevisions: definitions.filter((definition) => definition.frames.some((item, index) => renderStateTransition(item.scene, definition.frames[index - 1]?.scene).includes('class="coding-mechanism"'))).map((definition) => definition.slug),
      histogramClarification: 'The final view distinguishes unresolved positive heights from the actual stack, which still contains the zero sentinel.',
    },
    preservation: { headings: true, pythonFences: true, orderedVisualIds: true, authoredFrameIds: true, orderedMotionIds: true, protectedSources: true },
    successfulGeometryPolicy: 'Retain useful existing plots and explicit counterexamples rather than replacing them merely to increase the redraw count.',
    qualificationPolicy: 'Source review, native rendering, contact-sheet review and individual native-image inspection are distinct. See docs/VISUAL_COMPREHENSION_REVIEW.md and the final evidence manifest for accepted coverage.',
    measuredTenfoldImprovement: false,
    learnerStudy: false,
    pdfPaginationVerified: false,
    publicationPerformed: false,
  };
  fs.writeFileSync(path.join(root, 'data/visual-comprehension-review.json'), `${JSON.stringify(review, null, 2)}\n`);
  console.log(`Edition reconciliation recorded: ${owners.size} registered, ${Object.keys(redraws).length} registered redraws, ${Object.keys(supplementalRedraws).length} supplemental redraws, ${definitions.length} coding traces.`);
}

function reviewBatch(start, count) {
  for (const item of review.reviews.slice(start, start + count)) {
    const audit = owners.get(item.visualId);
    assert(audit, `Unowned visual: ${item.visualId}`);
    const source = fs.readFileSync(path.join(root, audit.article), 'utf8');
    const marker = source.indexOf(`<!-- visual:${item.visualId} -->`);
    const next = source.indexOf('<!-- visual:', marker + 1);
    const excerpt = source.slice(marker, next < 0 ? undefined : next);
    const svgLabels = htmlNodes(excerpt).filter(({ node }) => ['text', 'title', 'desc'].includes(node.nodeName)).map(({ node }) => {
      const text = (element) => element.nodeName === '#text' ? element.value : (element.childNodes ?? []).map(text).join('');
      return text(node);
    });
    console.log(JSON.stringify({ index: start++, id: item.visualId, medium: audit.medium, keep: item.keep, improve: item.improvement, check: item.readerCheck, labels: svgLabels.join(' | ').slice(0, 1700) }));
  }
}

function applyReaderChecks() {
  let changed = 0;
  for (const audit of audits) {
    const file = path.join(root, audit.article);
    const source = fs.readFileSync(file, 'utf8');
    const html = htmlNodes(source);
    const replacements = [];
    for (const identity of audit.implementation.visualIds) {
      const item = review.reviews.find((entry) => entry.visualId === identity);
      const marker = source.indexOf(`<!-- visual:${identity} -->`);
      const figure = html.find(({ node, offset }) => node.nodeName === 'figure' && offset + node.sourceCodeLocation.startOffset > marker);
      assert(figure && item?.readerCheck, `${identity}: figure and specific reader check required`);
      const caption = figure.node.childNodes.find((node) => node.nodeName === 'figcaption');
      assert(caption, `${identity}: direct caption required`);
      const question = (readerQuestions[identity] ?? item.readerCheck).replace(/,? and (why|what|how|which|when|where)\b/g, (_, word) => `? ${word[0].toUpperCase()}${word.slice(1)}`)
        .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
      const existing = (caption.childNodes ?? []).find((node) => node.attrs?.some((attribute) => attribute.name === 'data-reader-check' && attribute.value === identity));
      for (const child of caption.childNodes ?? []) {
        if (child.attrs?.some((attribute) => attribute.name === 'data-reader-check' && attribute.value !== identity)) {
          replacements.push({ start: figure.offset + child.sourceCodeLocation.startOffset, end: figure.offset + child.sourceCodeLocation.endOffset, value: '' });
        }
      }
      const start = figure.offset + (existing?.sourceCodeLocation.startOffset ?? caption.sourceCodeLocation.endTag.startOffset);
      const end = figure.offset + (existing?.sourceCodeLocation.endOffset ?? caption.sourceCodeLocation.endTag.startOffset);
      const prefix = /\s$/.test(source.slice(0, start)) ? '' : ' ';
      const value = `${prefix}<span class="visual-reader-check" data-reader-check="${identity}"><strong>Check:</strong> ${question}</span>`;
      if (source.slice(start, end) !== value) replacements.push({ start, end, value });
    }
    if (replacements.length) {
      fs.writeFileSync(file, replaceRanges(source, replacements));
      changed += replacements.length;
    }
  }
  console.log(`Figure-specific comprehension questions updated: ${changed}/${owners.size}.`);
  verify();
}

function preserveFragments(source, start, end, generated, required = []) {
  const html = htmlNodes(source);
  const currentIds = new Set(html.flatMap(({ attributes }) => attributes.id ? [attributes.id] : []));
  const generatedIds = new Set(htmlNodes(generated).flatMap(({ attributes }) => attributes.id ? [attributes.id] : []));
  const removedIds = html.flatMap(({ node, attributes, offset }) => attributes.id && offset + node.sourceCodeLocation.startOffset >= start && offset + node.sourceCodeLocation.endOffset <= end ? [attributes.id] : []);
  const aliases = [...new Set([...removedIds, ...required.filter((id) => !currentIds.has(id))])].filter((id) => !generatedIds.has(id));
  const anchors = aliases.map((id) => `<span class="visual-fragment-anchor" id="${id.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;')}" aria-hidden="true"></span>`).join('');
  return generated.replace(/(<figure\b[^>]*>)/, `$1${anchors}`);
}

function applyRedraws() {
  assert(fs.existsSync(baselinePath), 'Capture preservation evidence before generation');
  let changed = 0;
  for (const [identity, model] of Object.entries({ ...redraws, ...supplementalRedraws })) {
    const isSupplemental = Object.hasOwn(supplementalRedraws, identity);
    const audit = isSupplemental ? audits.find((item) => item.slug === model.slug) : owners.get(identity);
    assert(audit, `No owner for ${identity}`);
    const file = path.join(root, audit.article);
    const source = fs.readFileSync(file, 'utf8');
    if (isSupplemental) {
      const html = htmlNodes(source);
      const anchor = html.find(({ attributes }) => attributes['data-supplemental-visual'] === identity);
      assert(anchor, `Missing supplemental identity ${identity}`);
      const start = anchor.offset + anchor.node.sourceCodeLocation.startOffset;
      let end = anchor.offset + anchor.node.sourceCodeLocation.endOffset;
      if (anchor.node.nodeName !== 'figure') {
        const diagrams = [];
        function walk(node) {
          if (node.type === 'code' && node.lang === 'mermaid' && node.position.start.offset > start) diagrams.push(node);
          for (const child of node.children ?? []) walk(child);
        }
        walk(fromMarkdown(source));
        const diagram = diagrams[0];
        assert(diagram && diagram.position.start.offset - end < 200, `${identity}: no adjacent diagram`);
        const caption = html.find(({ node, attributes, offset }) => (attributes.class ?? '').split(' ').includes('diagram-caption') && offset + node.sourceCodeLocation.startOffset > diagram.position.end.offset);
        assert(caption && caption.offset + caption.node.sourceCodeLocation.startOffset - diagram.position.end.offset < 500, `${identity}: no direct caption`);
        end = caption.offset + caption.node.sourceCodeLocation.endOffset;
      }
      const value = preserveFragments(source, start, end, renderRedraw(identity, model), preservedFragments.get(audit.article));
      if (source.slice(start, end) !== value) {
        fs.writeFileSync(file, replaceRanges(source, [{ start, end, value }]));
        changed++;
      }
      continue;
    }
    const marker = `<!-- visual:${identity} -->`;
    const markerOffset = source.indexOf(marker);
    assert(markerOffset >= 0, `Missing marker ${identity}`);
    const html = htmlNodes(source);
    const existing = html.find(({ node, offset }) => node.nodeName === 'figure' && offset + node.sourceCodeLocation.startOffset > markerOffset);
    let start;
    let end;
    if (existing && source.slice(markerOffset, existing.offset + existing.node.sourceCodeLocation.startOffset).trim() === marker) {
      start = existing.offset + existing.node.sourceCodeLocation.startOffset;
      end = existing.offset + existing.node.sourceCodeLocation.endOffset;
    } else {
      const mermaids = [];
      function walk(node) {
        if (node.type === 'code' && node.lang === 'mermaid' && node.position.start.offset > markerOffset) mermaids.push(node);
        for (const child of node.children ?? []) walk(child);
      }
      walk(fromMarkdown(source));
      const diagram = mermaids[0];
      assert(diagram && diagram.position.start.offset - markerOffset < 100, `${identity}: no adjacent Mermaid diagram`);
      const caption = html.find(({ node, attributes, offset }) => (attributes.class ?? '').split(' ').includes('diagram-caption') && offset + node.sourceCodeLocation.startOffset > diagram.position.end.offset);
      assert(caption && caption.offset + caption.node.sourceCodeLocation.startOffset - diagram.position.end.offset < 500, `${identity}: no direct caption`);
      const title = html.filter(({ node, attributes, offset }) => (attributes.class ?? '').split(' ').includes('visual-title') && offset + node.sourceCodeLocation.endOffset < markerOffset).at(-1);
      assert(title && markerOffset - title.offset - title.node.sourceCodeLocation.endOffset < 200, `${identity}: no direct title`);
      start = title.offset + title.node.sourceCodeLocation.startOffset;
      end = caption.offset + caption.node.sourceCodeLocation.endOffset;
    }
    const readerCheck = html.find(({ attributes }) => attributes['data-reader-check'] === identity);
    const question = readerCheck ? source.slice(readerCheck.offset + readerCheck.node.sourceCodeLocation.startOffset, readerCheck.offset + readerCheck.node.sourceCodeLocation.endOffset) : '';
    const generated = preserveFragments(source, start, end, renderRedraw(identity, model), preservedFragments.get(audit.article)).replace('</figcaption>', `${question ? ` ${question}` : ''}</figcaption>`);
    const replacingMarker = start < markerOffset;
    const value = `${replacingMarker ? `${marker}\n` : ''}${generated}`;
    if (source.slice(start, end) !== value) {
      fs.writeFileSync(file, replaceRanges(source, [{ start, end, value }]));
      changed++;
    }
    const auditFile = path.join(root, 'data/visual-audits', `${audit.slug}.json`);
    audit.medium = 'svg';
    audit.mediumRationale = 'An original, phone-width explanatory drawing keeps the complete comparison visible without panning. A topic-specific timeline, quantity, evidence gate, or state relation replaces the component inventory. Geometry, direct labels, accessible descriptions, theme tokens, and static output carry the explanation without interaction.';
    audit.implementation.accessibility = `${model.description} All labels use at least 12 viewBox units in a 360-unit-wide composition. Meaning is also encoded by positions, words, line styles, and numbers, not color alone.`;
    fs.writeFileSync(auditFile, `${JSON.stringify(audit, null, 2)}\n`);
  }
  console.log(`Purpose-drawn illustrations updated: ${changed}/${Object.keys(redraws).length + Object.keys(supplementalRedraws).length}.`);
  verify();
}

if (process.argv.includes('--capture')) capture();
else if (process.argv.includes('--apply-coding')) applyCoding();
else if (process.argv.includes('--apply-redraws')) applyRedraws();
else if (process.argv.includes('--apply-reader-checks')) applyReaderChecks();
else if (process.argv.includes('--verify')) verify();
else if (process.argv.includes('--reconcile')) reconcile();
else if (process.argv.some((argument) => argument.startsWith('--review='))) {
  const [start, count = 10] = process.argv.find((argument) => argument.startsWith('--review=')).slice(9).split(':').map(Number);
  reviewBatch(start, count);
} else if (process.argv.includes('--self-test')) {
  const source = '<!-- visual:example -->\n<figure><div data-coding-frame="0" data-frame-key="first"><span data-motion-key="value-0">3</span></div></figure>\n\n## Heading\n\n```python\nprint(3)\n```\n';
  assert.deepEqual(identities(source), { python: ['print(3)'], headings: ['## Heading'], visuals: ['example'], supplemental: [], frames: ['first'], motion: ['value-0'] });
  assert.equal(replaceRanges('abcdef', [{ start: 1, end: 3, value: 'X' }, { start: 4, end: 5, value: 'Y' }]), 'aXdYf');
  assert.throws(() => replaceRanges('abcdef', [{ start: 1, end: 4, value: '' }, { start: 3, end: 5, value: '' }]), /overlap/);
  const fragmentSource = '<figure id="old-title"><svg><title id="old-svg">Old</title></svg></figure><p id="outside">Keep</p>';
  const fragmentEnd = fragmentSource.indexOf('</figure>') + '</figure>'.length;
  const retained = preserveFragments(fragmentSource, 0, fragmentEnd, '<figure><p id="new-title">New</p></figure>', ['missing-legacy', 'outside']);
  assert.deepEqual(htmlNodes(retained).flatMap(({ attributes }) => attributes.id ? [attributes.id] : []), ['old-title', 'old-svg', 'missing-legacy', 'new-title']);
  assert.equal(preserveFragments(retained, 0, retained.length, '<figure><p id="new-title">New</p></figure>'), retained);
  assert.equal(owners.size, review.expectedVisuals);
  assert.equal(review.reviews.length, owners.size);
  for (const audit of audits) {
    const source = fs.readFileSync(path.join(root, audit.article), 'utf8');
    const questions = htmlNodes(source).filter(({ attributes }) => attributes['data-reader-check']);
    const questionIds = questions.map(({ attributes }) => attributes['data-reader-check']);
    assert.deepEqual(questionIds, identities(source).visuals, `${audit.slug}: reader questions must follow their figures`);
    assert.deepEqual(new Set(questionIds), new Set(audit.implementation.visualIds), `${audit.slug}: each registered figure needs its own reader question`);
  }
  for (const [identity, model] of Object.entries({ ...redraws, ...supplementalRedraws })) {
    const audit = owners.get(identity) ?? audits.find((entry) => entry.slug === model.slug);
    assert(audit, `${identity}: redraw needs an article owner`);
    const source = fs.readFileSync(path.join(root, audit.article), 'utf8');
    const current = htmlNodes(source).find(({ node, attributes }) => node.nodeName === 'svg' && attributes['aria-labelledby'] === `${identity}-edition-svg-title ${identity}-edition-svg-desc`);
    assert(current, `${identity}: generated illustration is missing`);
    const generated = renderRedraw(identity, model);
    const expected = htmlNodes(generated).find(({ node }) => node.nodeName === 'svg');
    assert.equal(source.slice(current.offset + current.node.sourceCodeLocation.startOffset, current.offset + current.node.sourceCodeLocation.endOffset), generated.slice(expected.offset + expected.node.sourceCodeLocation.startOffset, expected.offset + expected.node.sourceCodeLocation.endOffset), `${identity}: generated drawing is out of sync`);
  }
  const bayes = fs.readFileSync(path.join(root, owners.get('bayes-positive-test-denominator').article), 'utf8');
  const positiveSegments = htmlNodes(bayes).filter(({ attributes }) => attributes['data-count']);
  assert.equal(positiveSegments.length, 2);
  const population = positiveSegments.reduce((sum, { attributes }) => sum + Number(attributes['data-count']), 0);
  const stripWidth = positiveSegments.reduce((sum, { attributes }) => sum + Number(attributes.width), 0);
  assert.equal(population, 1098);
  assert.equal(stripWidth, 312);
  for (const { attributes } of positiveSegments) {
    assert(Math.abs(Number(attributes.width) / stripWidth - Number(attributes['data-count']) / population) < 1e-10, 'Bayes strip must draw the stated proportions');
  }
  console.log(`Edition tooling passed: structured HTML edits, preservation identities, ${owners.size} owned reader questions, and ${Object.keys(redraws).length + Object.keys(supplementalRedraws).length} synchronized drawings.`);
} else {
  console.log('Use --self-test, --capture, --apply-coding, --apply-redraws, --apply-reader-checks, --verify, --reconcile, or --review=START:COUNT. Generation and preservation use --evidence-dir=PATH.');
}
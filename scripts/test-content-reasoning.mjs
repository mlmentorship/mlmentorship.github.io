import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { test } from 'node:test';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkMath from 'remark-math';

const directory = new URL('../src/content/posts/', import.meta.url);
const files = new Map(readdirSync(directory).map((file) => [file.replace(/^\d{4}-\d{2}-\d{2}-/, '').replace(/\.md$/, ''), file]));
const parser = unified().use(remarkParse).use(remarkMath);
const read = (slug) => readFileSync(new URL(files.get(slug), directory), 'utf8');

function nodes(source, type) {
  const found = [];
  function visit(node) {
    if (node.type === type) found.push(node.value);
    for (const child of node.children ?? []) visit(child);
  }
  visit(parser.parse(source));
  return found;
}

function arrays(source) {
  return nodes(source, 'inlineCode').filter((value) => /^\[[\d.,\s]+\]$/.test(value)).map((value) => JSON.parse(value));
}

function close(actual, expected) {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);
}

test('shared temperature, class-specific scaling, and operating decisions are distinct', () => {
  const source = read('expected-calibration-error');
  const [logits, temperatures, sharedExpected, vectorExpected] = arrays(source);
  assert.deepEqual(logits, [2, 1]);
  const shared = logits.map((value) => value / temperatures[0]);
  const vector = logits.map((value, index) => value / temperatures[index]);
  assert.deepEqual(shared, sharedExpected);
  assert.deepEqual(vector, vectorExpected);
  assert.ok(shared[0] > shared[1] && vector[0] < vector[1]);
  const originalConfidence = 1 / (1 + Math.exp(logits[1] - logits[0]));
  const scaledConfidence = 1 / (1 + Math.exp(shared[1] - shared[0]));
  assert.ok(originalConfidence > 0.7 && scaledConfidence < 0.7);
  assert.match(source, /\$T>0\$/);
  assert.doesNotMatch(source, /Calibration is \*\*lossless on accuracy\*\*|usually halves ECE|often halves ECE/);
});

test('an unbiased value baseline need not minimize score-function variance', () => {
  const source = read('policy-gradient');
  const probability = Number(source.match(/`action - ([0-9.]+)`/)[1]);
  function moments(baseline) {
    const outcomes = [0, 1].map((action) => ({
      probability: action ? probability : 1 - probability,
      estimate: (action - probability) * (action - baseline),
    }));
    const mean = outcomes.reduce((sum, item) => sum + item.probability * item.estimate, 0);
    const variance = outcomes.reduce((sum, item) => sum + item.probability * (item.estimate - mean) ** 2, 0);
    return { mean, variance };
  }
  const value = moments(probability);
  const optimum = moments(1 - probability);
  close(value.mean, 0.16);
  close(optimum.mean, value.mean);
  close(value.variance, 0.0576);
  close(optimum.variance, 0);
  assert.match(source, /score-norm weighting/);
  assert.doesNotMatch(source, /Optimal baseline: a value function/);
});

test('PPO surrogate clipping does not impose a probability-ratio constraint', () => {
  const source = read('policy-gradient');
  const previous = Number(source.match(/old action probability is ([0-9]+(?:\.[0-9]+)?)/i)[1]);
  const epsilon = Number(source.match(/clipping parameter is ([0-9]+(?:\.[0-9]+)?)/)[1]);
  const current = [...source.matchAll(/new probability ([0-9]+(?:\.[0-9]+)?)/g)].map((match) => Number(match[1]));
  assert.deepEqual(current, [0.9, 0.12]);
  const ratios = current.map((value) => value / previous);
  const objectives = ratios.map((ratio) => Math.min(ratio, Math.max(1 - epsilon, Math.min(1 + epsilon, ratio))));
  close(objectives[0], objectives[1]);
  assert.ok(ratios[0] > 1 + epsilon);
  assert.match(source, /not a hard constraint/);
  assert.doesNotMatch(source, /clipping caps the staleness/);
});

test('speculative output counts include one replacement-or-bonus and can lose on cost', () => {
  const source = read('speculative-decoding');
  const length = Number(source.match(/\$K=(\d+)\$/)[1]);
  const draftCost = Number(source.match(/\$d=([0-9.]+)\$/)[1]);
  const verifyCost = Number(source.match(/\$v_K=([0-9.]+)\$/)[1]);
  for (const acceptance of [0, 0.2, 0.8, 1]) {
    let enumerated = acceptance ** length * (length + 1);
    let geometric = 0;
    for (let accepted = 0; accepted < length; accepted += 1) {
      enumerated += acceptance ** accepted * (1 - acceptance) * (accepted + 1);
    }
    for (let exponent = 0; exponent <= length; exponent += 1) geometric += acceptance ** exponent;
    close(enumerated, geometric);
  }
  close(1 / (length * draftCost + verifyCost), 5 / 7);
  assert.match(source, /Do not add a second bonus/);
  assert.doesNotMatch(source, /pure free speedup|are SOTA|speedup typical/);
});

test('a block mask isolates examples while fixed-frequency RoPE permits common shifts', () => {
  const source = read('sequence-packing');
  const fixtures = arrays(source);
  const lengths = fixtures.find((value) => value.length === 3);
  const [reset, continued] = fixtures.filter((value) => value.length === 6);
  const examples = lengths.flatMap((length, index) => Array(length).fill(index));
  function rotate(vector, position) {
    const angle = position * 0.4;
    return [vector[0] * Math.cos(angle) - vector[1] * Math.sin(angle), vector[0] * Math.sin(angle) + vector[1] * Math.cos(angle)];
  }
  function attention(positions, isolate) {
    return positions.map((position, queryIndex) => {
      const query = rotate([Math.cos(queryIndex + 0.2), 0.5], position);
      const scores = positions.map((keyPosition, keyIndex) => {
        if (keyIndex > queryIndex || (isolate && examples[queryIndex] !== examples[keyIndex])) return -Infinity;
        const key = rotate([1, Math.sin(keyIndex + 0.3)], keyPosition);
        return (query[0] * key[0] + query[1] * key[1]) / Math.sqrt(2);
      });
      const maximum = Math.max(...scores);
      const weights = scores.map((value) => Math.exp(value - maximum));
      const total = weights.reduce((sum, value) => sum + value, 0);
      return weights.map((value) => value / total);
    });
  }
  const original = attention(reset, true);
  const shifted = attention(continued, true);
  const unmasked = attention(reset, false);
  let leaked = 0;
  for (let queryIndex = 0; queryIndex < examples.length; queryIndex += 1) {
    for (let keyIndex = 0; keyIndex < examples.length; keyIndex += 1) {
      close(original[queryIndex][keyIndex], shifted[queryIndex][keyIndex]);
      if (examples[queryIndex] !== examples[keyIndex]) {
        assert.equal(shifted[queryIndex][keyIndex], 0);
        leaked += unmasked[queryIndex][keyIndex];
      }
    }
  }
  assert.ok(leaked > 0);
  assert.doesNotMatch(source, /attention learns to treat packed boundaries as long-range dependencies/);
});

test('recall-weighted AP differs from linearly interpolated PR area on the same scores', () => {
  const source = read('roc-pr-auc');
  const [scores, labels] = arrays(source);
  assert.equal(scores.length, labels.length);
  assert.ok(scores.every((score, index) => index === 0 || score < scores[index - 1]));
  const positives = labels.reduce((sum, value) => sum + value, 0);
  let truePositives = 0;
  let previousRecall = 0;
  let previousPrecision = 1;
  let averagePrecision = 0;
  let area = 0;
  labels.forEach((label, index) => {
    truePositives += label;
    const recall = truePositives / positives;
    const precision = truePositives / (index + 1);
    averagePrecision += (recall - previousRecall) * precision;
    area += (recall - previousRecall) * (precision + previousPrecision) / 2;
    previousRecall = recall;
    previousPrecision = precision;
  });
  close(averagePrecision, 5 / 6);
  close(area, 19 / 24);
  assert.match(source, /half the probability of a tie/);
  assert.doesNotMatch(source, /PR-AUC is the average precision over recall thresholds/);
});

test('float32 HMM underflow depends on probabilities rather than a universal length', () => {
  const source = read('forward-backward-and-viterbi');
  const cases = nodes(source, 'inlineMath').map((value) => value.match(/^(0\.[0-9]+)\^\{([0-9]+)\}/)).filter(Boolean);
  const results = cases.map((match) => {
    let likelihood = Math.fround(1);
    for (let step = 0; step < Number(match[2]); step += 1) likelihood = Math.fround(likelihood * Math.fround(Number(match[1])));
    return { length: Number(match[2]), likelihood };
  });
  assert.equal(results.find((item) => item.length === 16).likelihood, 0);
  assert.ok(results.find((item) => item.length === 100).likelihood > 0);
  assert.match(source, /accumulated log-normalizers/);
  assert.doesNotMatch(source, /underflow guaranteed beyond/);
});

test('long design references have one contiguous extension after a worked decision and transfer check', () => {
  for (const slug of ['design-foundation-model-data-platform', 'design-ai-coding-product', 'design-agent-safety-control-plane', 'design-reasoning-model-fixed-budget']) {
    const source = read(slug);
    const opening = source.indexOf('<section data-learning-extension="true"');
    const core = source.slice(0, opening);
    assert.ok(opening > source.indexOf('## The prompt'), slug);
    assert.equal(source.match(/data-learning-extension=/g)?.length, 1, slug);
    for (const label of ['**Worked case:', '**Objective.**', '**Invariant.**', '**State.**', '**Decision.**', '**Failure and recovery.**', '**Transfer check.**']) {
      assert.ok(core.includes(label), `${slug}: ${label}`);
    }
    assert.match(core, /<details>[\s\S]*<summary>[\s\S]*<\/details>/, slug);
    assert.match(source.slice(opening), /<\/section>\s+\*Related:/, slug);
    assert.doesNotMatch(source, /<!-- learning:core-end -->/, slug);
  }
});

test('fixed-budget first path accounts for the complete allocation and an eight-accelerator loss', () => {
  const source = read('design-reasoning-model-fixed-budget');
  const allocation = arrays(source).find((value) => value.length === 5);
  assert.deepEqual(allocation, [64, 24, 8, 16, 16]);
  const total = allocation.reduce((sum, value) => sum + value, 0);
  const committed = allocation.slice(0, -1).reduce((sum, value) => sum + value, 0);
  assert.equal(total, 128);
  assert.equal(total - 8 - committed, 8);
  assert.ok(total + committed > total);
  assert.match(source, /not proof that the resulting allocation meets the traffic SLOs/);
});

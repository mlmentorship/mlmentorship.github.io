import { defineVisual, frame, table, visual } from '../primitives.mjs';

const example = 'labels = [1,0,1,1,0], predictions = [1,1,0,1,0]';
const rows = [
  ['0', '1', '1', 'TP'],
  ['1', '0', '1', 'FP'],
  ['2', '1', '0', 'FN'],
  ['3', '1', '1', 'TP'],
  ['4', '0', '0', 'TN'],
];
const motion = (step, label) => [
  { key: 'metric-cursor', kind: 'pointer', x: step, y: 0, label },
  ...rows.map((row, index) => ({ key: `example-${index}`, kind: 'example', x: index, y: 1, label: row[3] })),
];
const state = (active, step, label, extra = {}) => table(
  ['i', 'label', 'prediction', 'cell'],
  rows,
  active,
  { example, ...extra, motion: motion(step, label) },
);

const draft = visual('Route each pair with Boolean masks, then divide TP by predicted-positive and actual-positive totals safely.', [
  frame('Align labels and predictions', 'Five index-aligned pairs route to TP, FP, FN, TP, and TN respectively.', state([], 0, 'aligned input', { pairs: '(1,1),(0,1),(1,0),(1,1),(0,0)' }), 'align-inputs'),
  frame('Build the true-positive mask', '(labels==1) AND (predictions==1) is [T,F,F,T,F], whose sum is TP=2.', state([1,13], 1, 'TP mask', { mask: '[T,F,F,T,F]', arithmetic: '1 + 0 + 0 + 1 + 0 = 2', truePositive: '2' }), 'count-tp'),
  frame('Build the false-positive mask', '(labels==0) AND (predictions==1) is [F,T,F,F,F], so FP=1.', state([5], 2, 'FP mask', { mask: '[F,T,F,F,F]', arithmetic: '0 + 1 + 0 + 0 + 0 = 1', falsePositive: '1' }), 'count-fp'),
  frame('Build the false-negative mask', '(labels==1) AND (predictions==0) is [F,F,T,F,F], so FN=1.', state([9], 3, 'FN mask', { mask: '[F,F,T,F,F]', arithmetic: '0 + 0 + 1 + 0 + 0 = 1', falseNegative: '1' }), 'count-fn'),
  frame('Compute the precision denominator', 'precision_total=TP+FP=2+1=3, the number of predicted positives.', state([1,5,13], 4, 'predicted positives', { precisionTotal: '2 + 1 = 3' }), 'precision-denominator'),
  frame('Compute precision', 'Because precision_total is nonzero, precision=TP/precision_total=2/3.', state([1,5,13], 5, 'precision', { safeBranch: '3 is nonzero', arithmetic: '2 / 3 = 0.6667', precision: '0.6667' }), 'precision-value'),
  frame('Compute the recall denominator', 'recall_total=TP+FN=2+1=3, the number of actual positives.', state([1,9,13], 6, 'actual positives', { recallTotal: '2 + 1 = 3' }), 'recall-denominator'),
  frame('Compute recall and return', 'Because recall_total is nonzero, recall=TP/recall_total=2/3; return both metrics.', state([1,9,13], 7, 'recall', { safeBranch: '3 is nonzero', arithmetic: '2 / 3 = 0.6667', result: '{"precision":0.6667,"recall":0.6667}' }), 'return-metrics'),
]);

const review = {
  pattern: 'Vectorized Boolean masks for confusion counts followed by denominator-specific safe division.',
  recognitionCue: 'Binary labels and predictions must be reduced to precision and recall, whose numerators overlap but whose predicted-positive and actual-positive populations differ.',
  invariant: 'Each aligned example contributes to exactly one confusion cell; TP, FP, and FN masks count disjoint conditions, and each metric uses TP over its named population.',
  stateModel: 'Retain aligned label/prediction arrays, three Boolean masks or their sums, precision_total, and recall_total. True negatives do not enter either denominator.',
  visualRationale: 'A per-example routing table keeps every pair and confusion category visible while a stable metric cursor moves through masks, sums, denominators, and guarded divisions.',
  rejectedAlternatives: [
    'A bare formula triangle hides how Boolean array conditions produce TP, FP, and FN.',
    'A 2x2 matrix alone loses per-example verification and the safe-division branches.',
    'Two unrelated gauges obscure that precision and recall share TP but select different populations.',
  ],
  transferLesson: 'Define the population before dividing: precision conditions on predicted positives, recall on actual positives; the same discipline prevents denominator mistakes in rates and conditional metrics.',
  reviewStatus: 'reviewed',
};

export default defineVisual('binary-precision-and-recall', draft, review);

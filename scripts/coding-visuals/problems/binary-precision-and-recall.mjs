import { defineVisual, frame, table, visual } from '../primitives.mjs';

const observations = [
  ['0', '1', '1', 'TP'],
  ['1', '0', '1', 'FP'],
  ['2', '1', '0', 'FN'],
  ['3', '0', '0', 'TN'],
];
const cursor = (row, extra = {}) => table(
  ['index', 'label', 'prediction', 'cell'],
  observations,
  [row * 4, row * 4 + 1, row * 4 + 2, row * 4 + 3],
  {
    labels: '[1,0,1,0]',
    predictions: '[1,1,0,0]',
    motion: [{ key: 'observation-cursor', kind: 'row', x: 0, y: row, label: `observation ${row}` }],
    ...extra,
  },
);

const draft = visual('Build TP, FP, and FN with elementwise Boolean masks, then divide TP by predicted-positive and actual-positive totals with zero-denominator guards.', [
  frame('Classify index 0 as true positive', 'label=1 and prediction=1 makes both equality masks true, so TP increases from 0 to 1.', cursor(0, {
    masks: '(label==1)&(prediction==1) = true',
    counts: 'TP=1 FP=0 FN=0',
  }), 'index-zero'),
  frame('Classify index 1 as false positive', 'label=0 and prediction=1 satisfies the false-positive mask, so FP becomes 1.', cursor(1, {
    masks: '(label==0)&(prediction==1) = true',
    counts: 'TP=1 FP=1 FN=0',
  }), 'index-one'),
  frame('Classify index 2 as false negative', 'label=1 and prediction=0 satisfies the false-negative mask, so FN becomes 1.', cursor(2, {
    masks: '(label==1)&(prediction==0) = true',
    counts: 'TP=1 FP=1 FN=1',
  }), 'index-two'),
  frame('Ignore index 3 for both metrics', 'label=0 and prediction=0 is TN. None of the three counted masks matches, so TP, FP, and FN stay unchanged.', cursor(3, {
    masks: 'TP=false, FP=false, FN=false',
    counts: 'TP=1 FP=1 FN=1',
  }), 'index-three'),
  frame('Form the two different denominators', 'Precision counts predicted positives: TP+FP=1+1=2. Recall counts actual positives: TP+FN=1+1=2.', table(
    ['metric', 'numerator', 'denominator meaning', 'total'],
    [
      ['precision', 'TP=1', 'TP+FP', '2'],
      ['recall', 'TP=1', 'TP+FN', '2'],
    ],
    [3, 7],
    {
      arithmetic: 'precision_total=2; recall_total=2',
      motion: [{ key: 'observation-cursor', kind: 'row', x: 3, y: 1, label: 'denominator totals' }],
    },
  ), 'denominators'),
  frame('Take the guarded division branches', 'Both totals are nonzero, so precision=1/2=0.5 and recall=1/2=0.5; neither zero fallback runs.', table(
    ['metric', 'guard', 'division', 'value'],
    [
      ['precision', '2 != 0', '1 / 2', '0.5'],
      ['recall', '2 != 0', '1 / 2', '0.5'],
    ],
    [3, 7],
    {
      safeDivision: 'use 0.0 only when its denominator is zero',
      motion: [{ key: 'observation-cursor', kind: 'row', x: 3, y: 0, label: 'metric result' }],
      result: '{"precision":0.5,"recall":0.5}',
    },
  ), 'return'),
]);

const review = {
  pattern: 'Elementwise Boolean confusion masks followed by metric-specific safe division.',
  recognitionCue: 'Use these masks when binary labels and predictions must be reduced into precision and recall, whose denominators answer different conditional questions.',
  invariant: 'After each index, TP, FP, and FN equal the counts of their Boolean conjunctions over the processed prefix; TN affects neither precision nor recall numerator or denominator.',
  stateModel: 'Retain aligned label and prediction arrays, three integer mask sums, predicted-positive total TP+FP, actual-positive total TP+FN, and guarded divisions.',
  visualRationale: 'A concrete row per observation keeps label/prediction alignment visible while a stable cursor advances through every mask outcome, then a compact metric table contrasts denominator meanings and guards.',
  rejectedAlternatives: [
    'A confusion-matrix template without input arrays cannot verify how observations produce counts.',
    'Reporting only final ratios hides the critical difference between predicted-positive and actual-positive denominators.',
    'Using one unconditional division can produce undefined metrics when a denominator is zero.',
  ],
  transferLesson: 'Define classification metrics as Boolean event counts first, then name each denominator population and guard empty populations before division.',
  reviewStatus: 'reviewed',
};

export default defineVisual('binary-precision-and-recall', draft, review);

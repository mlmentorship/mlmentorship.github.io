import { defineVisual, frame, table, visual } from '../primitives.mjs';

const motion = (label, y = 0) => [{ key: 'example-row', kind: 'tensor-row', x: 0, y, label }];
const columns = ['example', 'class 0', 'class 1', 'class 2', 'label'];

const draft = visual('Compute stable log-normalizers per example, subtract the indexed correct logits, then average example losses.', [
  frame('Read logits and integer labels', 'Use logits [[2,1,0],[0,1,2]] with labels [0,1]. Each label selects one class in its own row.', table(columns, [
    ['0', '2', '1', '0', '0'],
    ['1', '0', '1', '2', '1'],
  ], [1, 9], {
    motion: motion('example 0'),
  }), 'inputs'),
  frame('Shift each row by its maximum', 'Both row maxima are 2. Shifted rows become [0,-1,-2] and [-2,-1,0], keeping the selected entries 0 and -1.', table(columns, [
    ['0', '0', '-1', '-2', '0'],
    ['1', '-2', '-1', '0', '1'],
  ], [1, 8], {
    maxima: '[2,2]',
    motion: motion('shifted example 0'),
  }), 'shift'),
  frame('Compute row log-normalizers', 'Each shifted row has exponential sum 1 + 0.3679 + 0.1353 = 1.5032, so log_normalizer = log(1.5032) = 0.4076.', table(['example', 'exp sum', 'log normalizer'], [
    ['0', '1.5032', '0.4076'],
    ['1', '1.5032', '0.4076'],
  ], [2, 5], {
    motion: motion('normalizer example 0'),
  }), 'normalizers'),
  frame('Subtract each selected logit', 'Example 0 loss is 0.4076 - 0 = 0.4076. Example 1 label 1 selects shifted logit -1, so loss is 0.4076 - (-1) = 1.4076.', table(['example', 'normalizer', 'correct shifted logit', 'loss'], [
    ['0', '0.4076', '0', '0.4076'],
    ['1', '0.4076', '-1', '1.4076'],
  ], [3, 7], {
    motion: motion('loss example 0'),
  }), 'losses'),
  frame('Average the batch', 'Mean cross-entropy = (0.4076 + 1.4076) / 2 = 0.9076.', table(['example', 'loss'], [
    ['0', '0.4076'],
    ['1', '1.4076'],
    ['mean', '0.9076'],
  ], [5], {
    arithmetic: '1.8152 / 2 = 0.9076',
    motion: motion('mean loss', 2),
    result: 'mean = 0.9076',
  }), 'mean'),
]);

const review = {
  pattern: 'Stable row-wise log-softmax combined with indexed correct-class selection and batch reduction.',
  recognitionCue: 'Use it when multiclass logits and integer labels require negative log likelihood without explicitly forming potentially unstable probabilities.',
  invariant: 'For each row, log_normalizer minus the shifted selected logit equals negative log probability of the label; max shifting cancels from both terms.',
  stateModel: 'The minimal state is shifted logits, one log-normalizer per example, labels for advanced indexing, selected shifted logits, per-example losses, and their mean.',
  visualRationale: 'A batch-by-class table keeps row and class axes visible through shifting and indexed selection, then narrows to per-example loss. Arithmetic remains explicit in static monochrome form.',
  rejectedAlternatives: [
    'A single-example trace was rejected because the source returns a batch mean and uses row-wise indexing.',
    'Building softmax probabilities first was rejected because it obscures the stable log-sum-exp formulation.',
    'A loss curve was rejected because it does not depict class selection or batch reduction.',
  ],
  transferLesson: 'For stable classification losses, compute log-normalization in score space, gather the target score by aligned row indices, and reduce only after producing per-example losses.',
  reviewStatus: 'reviewed',
};

export default defineVisual('cross-entropy-from-logits', draft, review);

import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const motion = (label, x = 0) => [{ key: 'logit-cell', kind: 'tensor-cell', x, y: 0, label }];

const draft = visual('Subtract the row maximum, exponentiate nonpositive shifts, and divide by their shared sum.', [
  frame('Find the maximum logit', 'For logits [1000,1001,999], the row maximum is 1001. Direct exp(1001) can overflow.', array(
    ['1000', '1001', '999'],
    [mark(1, 'max = 1001', 'focus', 'row-maximum')],
    { motion: motion('logit 1001', 1) },
  ), 'maximum'),
  frame('Shift without changing ratios', 'Subtract 1001 from every entry to get [-1,0,-2]. Pairwise gaps are unchanged, and every shifted logit is at most zero.', array(
    ['-1', '0', '-2'],
    [mark(1, 'largest = 0', 'state', 'row-maximum')],
    { arithmetic: '[1000,1001,999] - 1001', motion: motion('shifted logit 0', 1) },
  ), 'shift'),
  frame('Exponentiate safely', 'Exponentials are [exp(-1),exp(0),exp(-2)] = [0.3679,1.0000,0.1353], whose sum is 1.5032.', array(
    ['0.3679', '1.0000', '0.1353'],
    [mark(1, 'exp(0)', 'focus', 'largest-exponential')],
    { denominator: '0.3679 + 1.0000 + 0.1353 = 1.5032', motion: motion('exponential 1', 1) },
  ), 'exponentiate'),
  frame('Normalize the row', 'Divide each exponential by 1.5032 to get [0.2447,0.6652,0.0900]. The rounded probabilities sum to 0.9999 (approximately 1).', array(
    ['0.2447', '0.6652', '0.0900'],
    [mark(1, 'largest probability', 'output', 'largest-probability')],
    { normalization: 'exponentials / 1.5032', motion: motion('probability 0.6652', 1), result: '[0.2447,0.6652,0.0900]' },
  ), 'normalize'),
]);

const review = {
  pattern: 'Numerically stable row-wise normalization by max shifting before exponentiation.',
  recognitionCue: 'Use it when exponentials normalize scores into probabilities and large positive logits may overflow even though only relative differences should affect the result.',
  invariant: 'Subtracting one constant from every row element leaves every exponential ratio unchanged; after max shifting, the largest exponent is exp(0)=1 and none can overflow.',
  stateModel: 'The minimal state is the logits vector, row maximum retained along the normalized axis, shifted logits, exponentials, and their keep-dimension sum.',
  visualRationale: 'A stable three-cell row keeps class identity fixed while values move through max, shift, exponent, and division stages. Equations make the trace independent of color or JavaScript.',
  rejectedAlternatives: [
    'A probability bar chart alone was rejected because it hides the overflow-prevention step.',
    'A raw exp(1000) calculation was rejected because it demonstrates failure rather than the supplied stable implementation.',
    'Shape-only tensor boxes were rejected because numerical stability depends on actual value magnitudes.',
  ],
  transferLesson: 'Before exponentiating normalized scores, exploit shift invariance to center at the maximum; preserve the reduced dimension so broadcasting back across the row is explicit.',
  reviewStatus: 'reviewed',
};

export default defineVisual('stable-softmax', draft, review);

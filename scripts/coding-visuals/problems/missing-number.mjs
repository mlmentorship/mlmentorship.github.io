import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const nums = ['3', '0', '1'];
const xorSteps = [
  { index: 0, before: 3, num: 3, after: 0 },
  { index: 1, before: 0, num: 0, after: 1 },
  { index: 2, before: 1, num: 1, after: 2 },
];

const draft = visual('Initialize with n, then XOR every expected index and actual value so matched numbers cancel and the missing value survives.', [
  frame(
    'Seed the unmatched endpoint',
    'For nums = [3, 0, 1], n = 3. Initialize missing = 3 because enumerate supplies expected indices 0, 1, and 2 but not endpoint 3.',
    array(nums, [mark(0, 'next i=0', 'focus', 'scan')], {
      expectedDomain: '[0, 1, 2, 3]',
      accumulator: 'missing = len(nums) = 3 = 0011₂',
    }),
    'initialize',
  ),
  ...xorSteps.map(({ index, before, num, after }) => frame(
    `XOR index ${index} and value ${num}`,
    `missing = ${before} XOR ${index} XOR ${num} = ${after}; XOR order does not change the eventual cancellations.`,
    array(nums, [mark(index, `i=${index}, num=${num}`, index === 2 ? 'output' : 'focus', 'scan')], {
      before: `${before} = ${before.toString(2).padStart(4, '0')}₂`,
      operation: `${before} xor ${index} xor ${num} = ${after}`,
      binaryResult: `${after.toString(2).padStart(4, '0')}₂`,
      cancellationsSeen: index === 0 ? '3 xor 3 = 0' : index === 1 ? '0 xor 0 = 0' : '1 xor 1 = 0',
      ...(index === 2 ? { result: '2' } : {}),
    }),
    `index-${index}`,
  )),
]);

export default defineVisual('missing-number', draft, {
  pattern: 'XOR cancellation across the expected domain and actual values.',
  recognitionCue: 'Values should contain every integer from 0 through n exactly once except one, making the actual multiset differ from the expected multiset by a single value.',
  invariant: 'After processing indices before i, missing equals n XOR every processed expected index XOR every processed actual value. Equal values may be regrouped and cancel because XOR is associative, commutative, and self-inverse.',
  stateModel: 'Keep one XOR accumulator initialized to n plus the current index and array value. No set, sorting, or arithmetic sum is required.',
  visualRationale: 'The actual indexed array keeps expected indices aligned with supplied values, a stable scan key moves through each pair, and every accumulator equation shows the exact intermediate binary result.',
  rejectedAlternatives: [
    'A set-difference diagram uses linear auxiliary space and does not trace the supplied code.',
    'Sorting and scanning changes the input order and costs O(n log n).',
    'The arithmetic-sum formula can overflow in fixed-width languages and hides XOR cancellation.',
  ],
  transferLesson: 'When all values occur in canceling pairs except one, XOR removes order and pair placement from the problem; the same invariant finds a unique element or separates parity-based membership differences.',
  reviewStatus: 'reviewed',
});

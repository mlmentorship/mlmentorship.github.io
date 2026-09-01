import { arrayMap, defineVisual, frame, mark, visual } from '../primitives.mjs';

const nums = ['1', '-1', '1', '-1', '1'];
const current = (index, label = `i=${index}`) => [mark(index, label, 'focus', 'scan')];

const draft = visual('Count earlier prefix sums equal to current prefix minus target.', [
  frame(
    'Seed the empty prefix',
    'For nums = [1, -1, 1, -1, 1] and target 0, record prefix 0 once before scanning so subarrays starting at index 0 can match.',
    arrayMap(nums, [['0', 'count 1']], current(0, 'next i=0'), {
      mapLabel: 'earlier prefix counts',
      prefix: '0',
      target: '0',
      answer: '0',
    }),
    'empty-prefix',
  ),
  frame(
    'Read index 0',
    'Prefix becomes 1. Need 1 - 0 = 1, which has appeared 0 times; answer stays 0, then prefix 1 is stored once.',
    arrayMap(nums, [['0', 'count 1'], ['1', 'count 1']], current(0), {
      query: '1 - 0 = 1; prior count 0',
      prefix: '1',
      answer: '0 + 0 = 0',
    }),
    'scan-index-0',
  ),
  frame(
    'Read index 1',
    'Prefix returns to 0. Need prefix 0, seen once, so [0..1] is counted; then the stored count for 0 becomes 2.',
    arrayMap(nums, [['0', 'count 2'], ['1', 'count 1']], current(1), {
      query: '0 - 0 = 0; prior count 1',
      matched: '[0..1]',
      answer: '0 + 1 = 1',
    }),
    'scan-index-1',
  ),
  frame(
    'Read index 2',
    'Prefix becomes 1. Need prefix 1, seen once, so [1..2] is counted; then the stored count for 1 becomes 2.',
    arrayMap(nums, [['0', 'count 2'], ['1', 'count 2']], current(2), {
      query: '1 - 0 = 1; prior count 1',
      matched: '[1..2]',
      answer: '1 + 1 = 2',
    }),
    'scan-index-2',
  ),
  frame(
    'Read index 3',
    'Prefix returns to 0. Two earlier zero prefixes produce [0..3] and [2..3]; add both before raising count(0) to 3.',
    arrayMap(nums, [['0', 'count 3'], ['1', 'count 2']], current(3), {
      query: '0 - 0 = 0; prior count 2',
      matched: '[0..3], [2..3]',
      answer: '2 + 2 = 4',
    }),
    'scan-index-3',
  ),
  frame(
    'Read index 4',
    'Prefix becomes 1. Two earlier prefix-1 positions produce [1..4] and [3..4]; the final count is 6.',
    arrayMap(nums, [['0', 'count 3'], ['1', 'count 3']], current(4), {
      query: '1 - 0 = 1; prior count 2',
      matched: '[1..4], [3..4]',
      answer: '4 + 2 = 6',
      result: '6 subarrays',
    }),
    'scan-index-4',
  ),
]);

export default defineVisual('subarray-sum-equals-k', draft, {
  pattern: 'Running prefix sum plus a frequency map of earlier prefix sums.',
  recognitionCue: 'The task counts contiguous ranges with an exact sum, values may be negative, and many ranges can end at the same index, so a monotone sliding window is unsafe.',
  invariant: 'Before processing nums[i], the map counts every prefix ending before i. After prefix is updated, count(prefix - target) is exactly the number of valid subarrays ending at i; only then is the current prefix inserted.',
  stateModel: 'Keep the current prefix sum, the accumulated answer, and a map from each earlier prefix sum to its frequency. Frequencies, rather than a set, preserve multiple valid starts.',
  visualRationale: 'The actual signed array, a stable scan pointer, and the prefix-frequency map expose the subtraction identity and show every answer increment without relying on color or animation.',
  rejectedAlternatives: [
    'A sliding-window diagram suggests monotone growth and shrinkage, which negative values invalidate.',
    'A prefix-sum line without frequencies cannot explain why one endpoint can add two matches.',
    'A quadratic subarray grid displays candidates but obscures the constant-time lookup used by the solution.',
  ],
  transferLesson: 'Rewrite a range equation as current prefix minus earlier prefix. Store counts when duplicate earlier states represent distinct starts; this also transfers to subarray sums divisible by k and equal-count prefix signatures.',
  reviewStatus: 'reviewed',
});

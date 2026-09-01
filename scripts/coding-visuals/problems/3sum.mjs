import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const sorted = ['-4', '-1', '-1', '0', '1', '2'];
const pointers = (fixed, left, right, fixedLabel = `fixed i=${fixed}`) => [
  mark(fixed, fixedLabel, 'state', 'fixed'),
  ...(left === undefined ? [] : [mark(left, `L=${left}`, 'focus', 'left')]),
  ...(right === undefined ? [] : [mark(right, `R=${right}`, 'focus', 'right')]),
];

const draft = visual('After sorting and fixing one value, pointer moves discard pairs that cannot sum to its complement.', [
  frame(
    'Sort and test first -4',
    'Input [-1, 0, 1, 2, -1, -4] sorts to [-4, -1, -1, 0, 1, 2]. With fixed -4, -4 + -1 + 2 = -3, so move L right.',
    array(sorted, pointers(0, 1, 5), {
      coveredRange: 'pair search [1..5]',
      total: '-4 + -1 + 2 = -3 < 0',
      move: 'L: 1 -> 2',
    }),
    'fixed-minus-4-left-1',
  ),
  frame(
    'Advance past the second -1',
    'With fixed -4 and L at index 2, the total is still -3, so this pair is also too small and L moves right.',
    array(sorted, pointers(0, 2, 5), {
      coveredRange: 'pair search [2..5]',
      total: '-4 + -1 + 2 = -3 < 0',
      move: 'L: 2 -> 3',
    }),
    'fixed-minus-4-left-2',
  ),
  frame(
    'Test zero with two',
    'Now -4 + 0 + 2 = -2. Sorted order proves every pair using this left value and a smaller right value is also too small, so move L.',
    array(sorted, pointers(0, 3, 5), {
      coveredRange: 'pair search [3..5]',
      total: '-4 + 0 + 2 = -2 < 0',
      move: 'L: 3 -> 4',
    }),
    'fixed-minus-4-left-3',
  ),
  frame(
    'Exhaust fixed -4',
    'The last pair gives -4 + 1 + 2 = -1. Move L to R; no triple beginning with -4 can total zero.',
    array(sorted, pointers(0, 4, 5), {
      coveredRange: 'pair search [4..5]',
      total: '-4 + 1 + 2 = -1 < 0',
      move: 'L: 4 -> 5; pair search ends',
    }),
    'fixed-minus-4-exhausted',
  ),
  frame(
    'Record the first triple',
    'Fix the first -1 at index 1. The initial pair gives -1 + -1 + 2 = 0, so record [-1, -1, 2] and move both pointers.',
    array(sorted, pointers(1, 2, 5), {
      coveredRange: 'pair search [2..5]',
      total: '-1 + -1 + 2 = 0',
      move: 'L: 2 -> 3; R: 5 -> 4',
      answer: '[[-1, -1, 2]]',
    }),
    'first-triple',
  ),
  frame(
    'Record the second triple',
    'The next pair gives -1 + 0 + 1 = 0, so record [-1, 0, 1]. Moving both pointers makes them cross.',
    array(sorted, pointers(1, 3, 4), {
      coveredRange: 'pair search [3..4]',
      total: '-1 + 0 + 1 = 0',
      move: 'L: 3 -> 4; R: 4 -> 3',
      answer: '[[-1, -1, 2], [-1, 0, 1]]',
    }),
    'second-triple',
  ),
  frame(
    'Skip the duplicate fixed value',
    'Index 2 is another -1, equal to nums[1]. Skip it so the same two triples are not emitted again.',
    array(sorted, pointers(2, undefined, undefined, 'duplicate: skip'), {
      branch: 'index > 0 and nums[2] == nums[1]',
      action: 'continue to index 3',
      answer: '[[-1, -1, 2], [-1, 0, 1]]',
    }),
    'skip-duplicate-fixed',
  ),
  frame(
    'Discard a total above zero',
    'Fix 0 at index 3. The only pair gives 0 + 1 + 2 = 3, so move R left; the pair search ends.',
    array(sorted, pointers(3, 4, 5), {
      coveredRange: 'pair search [4..5]',
      total: '0 + 1 + 2 = 3 > 0',
      move: 'R: 5 -> 4; pair search ends',
    }),
    'fixed-zero-right-move',
  ),
  frame(
    'Finish the outer scan',
    'Indices 4 and 5 have fewer than two values to their right, so their while loops never run. The two recorded triples are the complete unique answer.',
    array(sorted, pointers(4, undefined, undefined, 'remaining fixed values'), {
      remaining: 'i=4 and i=5: L is not less than R',
      result: '[[-1, -1, 2], [-1, 0, 1]]',
    }),
    'complete',
  ),
]);

export default defineVisual('3sum', draft, {
  pattern: 'Sort, fix one value, then run a two-pointer complement search.',
  recognitionCue: 'The result needs unique triples satisfying a sum equation; sorting is allowed, and fixing one member reduces the remaining choice to a pair with an ordered sum.',
  invariant: 'For a fixed index, every pair outside [L, R] is already emitted or safely discarded. If the total is low, no pair using the current L can work; if high, no pair using the current R can work.',
  stateModel: 'Retain the sorted array, fixed index, left and right pair boundaries, and accumulated unique triples. Duplicate fixed values are skipped, and duplicate left values are skipped after a hit.',
  visualRationale: 'An indexed sorted array with independently keyed fixed, L, and R markers shows the active pair range, direction of every safe move, equality hits, and duplicate branch in reading order.',
  rejectedAlternatives: [
    'A hash-set view can find complements but hides the ordered discard proof and uniqueness handling.',
    'A triangle of all O(n^3) combinations emphasizes enumeration rather than why two pointers are safe.',
    'A result-only table omits the low, high, equality, and duplicate branches executed by the code.',
  ],
  transferLesson: 'After sorting, fix enough values to reduce k-sum to 2-sum, then use order to discard an entire boundary at once; carry duplicate-skipping rules at every fixed level.',
  reviewStatus: 'reviewed',
});

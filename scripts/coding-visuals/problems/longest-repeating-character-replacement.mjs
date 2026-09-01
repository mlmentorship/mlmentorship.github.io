import { arrayMap, defineVisual, frame, mark, visual } from '../primitives.mjs';

const items = ['A', 'A', 'B', 'A', 'B', 'B', 'A'];
const pointers = (left, right) => [
  mark(left, 'L', 'state', 'left'),
  mark(right, 'R', 'focus', 'right'),
];
const counts = (a, b) => [['A count', String(a)], ['B count', String(b)]];

const draft = visual('Keep a candidate length while window length - historical max frequency stays within k = 1.', [
  frame('Initialize the scan', 'For "AABABBA" with k = 1, start L = 0, largest_count = 0, best = 0, and inspect R = 0.', arrayMap(items, [], pointers(0, 0), {
    range: 'empty before reading index 0',
    budget: 'k = 1',
    best: '0',
  }), 'initialize'),
  frame('Read the first A', 'A count becomes 1 and largest_count becomes 1; 1 - 1 = 0 replacements, so best becomes 1.', arrayMap(items, counts(1, 0), pointers(0, 0), {
    range: '[0..0] = "A"',
    formula: '1 - 1 = 0 <= 1',
    best: '1',
  }), 'read-a-0'),
  frame('Extend to AA', 'At R = 1, A count and largest_count become 2; 2 - 2 = 0, so best becomes 2.', arrayMap(items, counts(2, 0), pointers(0, 1), {
    range: '[0..1] = "AA"',
    formula: '2 - 2 = 0 <= 1',
    best: '2',
  }), 'read-a-1'),
  frame('Spend one replacement', 'At R = 2, counts are A:2 and B:1; 3 - largest_count 2 = 1, so "AAB" fits and best becomes 3.', arrayMap(items, counts(2, 1), pointers(0, 2), {
    range: '[0..2] = "AAB"',
    formula: '3 - 2 = 1 <= 1',
    best: '3',
  }), 'read-b-2'),
  frame('Reach the best length', 'At R = 3, A count and largest_count become 3; 4 - 3 = 1, so "AABA" fits and best becomes 4.', arrayMap(items, counts(3, 1), pointers(0, 3), {
    range: '[0..3] = "AABA"',
    formula: '4 - 3 = 1 <= 1',
    best: '4',
  }), 'read-a-3'),
  frame('Shrink after the budget breaks', 'Adding B at R = 4 makes 5 - 3 = 2. Remove items[0] = A and move L right to 1; the retained length is 4.', arrayMap(items, counts(2, 2), pointers(1, 4), {
    range: '[1..4] = "ABAB"',
    direction: 'L: 0 -> 1 because 5 - 3 > 1',
    formula: '4 - historical max 3 = 1',
    best: '4',
  }), 'shrink-at-4'),
  frame('Shrink again at R = 5', 'Adding B makes counts A:2, B:3 over length 5, so remove items[1] = A and move L right to 2.', arrayMap(items, counts(1, 3), pointers(2, 5), {
    range: '[2..5] = "BABB"',
    direction: 'L: 1 -> 2 because 5 - 3 > 1',
    formula: '4 - historical max 3 = 1',
    best: '4',
  }), 'shrink-at-5'),
  frame('Finish without exceeding four', 'Adding A at R = 6 makes length 5 with historical max 3, so remove items[2] = B and move L to 3. best remains 4.', arrayMap(items, counts(2, 2), pointers(3, 6), {
    range: '[3..6] = "ABBA"',
    direction: 'L: 2 -> 3 because 5 - 3 > 1',
    formula: 'best = max(4, 4) = 4',
    result: '4',
  }), 'finish-at-6'),
]);

const review = {
  pattern: 'Variable-size sliding window with per-letter counts and a monotone historical maximum frequency.',
  recognitionCue: 'Use it for a longest contiguous span that may change at most k items into one repeated value; window length minus its dominant frequency is the required edit count.',
  invariant: 'After each shrink, window length is at most largest_count + k, where largest_count is the greatest frequency seen while expanding. best is the greatest retained candidate length seen so far.',
  stateModel: 'The minimal state is L, R, counts, monotone largest_count, best, and k. The indexed array fixes positions while stable pointers show every expansion and forced left move.',
  visualRationale: 'An indexed array plus the live frequency map makes both terms of length - largest_count visible. Named boundaries and written inequalities preserve the argument in print and without color.',
  rejectedAlternatives: [
    'A frequency bar chart was rejected because it hides which indices leave when L advances.',
    'A table of window lengths was rejected because it does not depict the covered substring or boundary movement.',
    'Animating replaced letters was rejected because the implementation never chooses actual replacement positions.',
  ],
  transferLesson: 'When edits can homogenize a window, rewrite validity as window size minus the best keepable group. A monotone summary can be enough for maximizing length even when it overestimates a later window frequency.',
  reviewStatus: 'reviewed',
};

export default defineVisual('longest-repeating-character-replacement', draft, review);

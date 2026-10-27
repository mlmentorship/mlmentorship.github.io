import { arrayMap, defineVisual, frame, mark, visual } from '../primitives.mjs';

const text = ['A', 'B', 'A', 'A', 'C'];
const pointers = (left, right) => [
  mark(left, 'L', 'state', 'left-pointer'),
  mark(right, 'R', 'focus', 'right-pointer'),
];
const windowScene = (left, right, need, extra = {}) => arrayMap(text, need, pointers(left, right), {
  fitArray: true,
  arrayColumns: 5,
  mapLabel: 'need = required count minus window count',
  ...extra,
  motion: [
    { key: 'left-pointer', kind: 'pointer', x: left, y: 0, label: 'L' },
    { key: 'right-pointer', kind: 'pointer', x: right, y: 0, label: 'R' },
  ],
});

const draft = visual('Save valid covers before each left removal. Removing the first essential copy makes the window invalid again.', [
  frame('Initialize required counts', 'For ABAAC and required AAC, start L at zero. Need two A copies and one C, so missing is three.', windowScene(0, 0, [['A', '2'], ['C', '1']], {
    range: 'empty before reading index 0',
    required: 'need A:2, C:1',
    missing: '3',
    best: 'none',
  }), 'initialize'),
  frame('Expand through A at index 0', 'need[A] was positive, so this A fills one requirement: missing becomes 2 and need[A] becomes 1.', windowScene(0, 0, [['A', '1'], ['C', '1']], {
    range: '[0..0] = "A"',
    direction: 'R reads index 0',
    missing: 'from 3 to 2',
  }), 'expand-a-0'),
  frame('Expand through irrelevant B', 'B has no positive deficit, so missing stays two. Decrementing need[B] to negative one records surplus.', windowScene(0, 1, [['A', '1'], ['B', '-1'], ['C', '1']], {
    range: '[0..1] = "AB"',
    direction: 'R reads index 1',
    missing: '2',
  }), 'expand-b-1'),
  frame('Expand through the second A', 'need[A] was 1, so index 2 fills the second required A: missing becomes 1 and need[A] becomes 0.', windowScene(0, 2, [['A', '0'], ['B', '-1'], ['C', '1']], {
    range: '[0..2] = "ABA"',
    direction: 'R reads index 2',
    missing: 'from 2 to 1',
  }), 'expand-a-2'),
  frame('Record a surplus A', 'need[A] is already 0, so the A at index 3 is surplus: missing stays 1 and need[A] becomes -1.', windowScene(0, 3, [['A', '-1'], ['B', '-1'], ['C', '1']], {
    range: '[0..3] = "ABAA"',
    direction: 'R reads index 3',
    missing: '1',
  }), 'expand-a-3'),
  frame('Complete the first valid window', 'C fills the final deficit, making missing = 0. Save "ABAAC" as the first valid window of length 5.', windowScene(0, 4, [['A', '-1'], ['B', '-1'], ['C', '0']], {
    range: '[0..4] = "ABAAC"',
    direction: 'R reads index 4',
    missing: 'from 1 to 0',
    best: '"ABAAC" (length 5)',
    savedRange: { start: 0, end: 5 },
  }), 'expand-c-4'),
  frame('Remove surplus A, then measure again', 'Restore need[A] from negative one to zero and move L to one. Coverage remains complete. Next, save BAAC with length four.', windowScene(1, 4, [['A', '0'], ['B', '-1'], ['C', '0']], {
    range: '[1..4] = "BAAC"',
    direction: 'L moves from 0 to 1',
    reason: 'removed surplus A; still valid',
    best: '"BAAC" (length 4)',
    savedRange: { start: 1, end: 5 },
  }), 'shrink-a-0'),
  frame('Remove surplus B, then measure again', 'Restore need[B] from negative one to zero and move L to two. Next, save AAC with length three.', windowScene(2, 4, [['A', '0'], ['B', '0'], ['C', '0']], {
    range: '[2..4] = "AAC"',
    direction: 'L moves from 1 to 2',
    reason: 'removed surplus B; still valid',
    best: '"AAC" (length 3)',
    update: 'best = "AAC" (length 3)',
    savedRange: { start: 2, end: 5 },
  }), 'shrink-b-1'),
  frame('Stop after removing a required A', 'Increment need[A] from 0 to 1, so missing becomes 1. L moves to 3 and shrinking stops because [3..4] lacks one A.', windowScene(3, 4, [['A', '1'], ['B', '0'], ['C', '0']], {
    range: '[3..4] = "AC"',
    direction: 'L moves from 2 to 3',
    reason: 'required A removed; validity breaks',
    missing: 'from 0 to 1',
    savedRange: { start: 2, end: 5 },
  }), 'break-a-2'),
  frame('Return the shortest saved window', 'The scan is complete and the saved slice starts at 2 with length 3, so the result is "AAC".', arrayMap(text, [['best_start', '2'], ['best_length', '3']], [
    mark(2, 'bL', 'output', 'best-left'),
    mark(4, 'bR', 'output', 'best-right'),
  ], {
    fitArray: true,
    arrayColumns: 5,
    savedRange: { start: 2, end: 5 },
    mapLabel: 'saved best bounds',
    range: '[2..4] = "AAC"',
    result: '"AAC"',
  }), 'result'),
]);

const review = {
  pattern: 'Variable-size sliding window that grows to feasibility and shrinks to minimality.',
  recognitionCue: 'Use it when a contiguous result must cover a multiset of required values and validity can be updated by adding or removing one boundary value.',
  invariant: 'need[x] is the remaining deficit for x (negative means surplus), missing is the total number of required copies absent from [L..R], and best is the shortest valid window seen before the current transition.',
  stateModel: 'The minimal state is L, R, need, missing, best_start, and best_length. The fixed indexed text, moving authored pointers, covered range, and signed deficits expose every gain and loss of validity.',
  visualRationale: 'The fitted indexed row preserves all original pointer and map identities. Signed deficits distinguish duplicates and surplus, while saved-best underlines persist when the final essential removal makes the active AC window invalid. Explicit from/to wording keeps movement labels intact on mobile.',
  rejectedAlternatives: [
    'A substring-only animation was rejected because it hides stable indices and why each boundary moves.',
    'A counts-only table was rejected because it hides which concrete contiguous range the counts describe.',
    'A final-window highlight was rejected because it omits the grow-until-valid and shrink-until-invalid mechanism.',
  ],
  transferLesson: 'When validity is monotone under expansion, grow until feasible and record before each left removal. Surplus removals preserve validity; removing the first essential copy deliberately breaks it. Exact deficits support constant-time boundary updates without comparing full multisets.',
  reviewStatus: 'reviewed',
};

export default defineVisual('minimum-window-substring', draft, review);

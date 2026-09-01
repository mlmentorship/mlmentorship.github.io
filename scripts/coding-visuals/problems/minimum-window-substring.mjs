import { arrayMap, defineVisual, frame, mark, visual } from '../primitives.mjs';

const text = ['A', 'B', 'A', 'A', 'C'];
const pointers = (left, right) => [
  mark(left, 'L', 'state', 'left-pointer'),
  mark(right, 'R', 'focus', 'right-pointer'),
];
const windowScene = (left, right, need, extra = {}) => arrayMap(text, need, pointers(left, right), {
  ...extra,
  motion: [
    { key: 'left-pointer', kind: 'pointer', x: left, y: 0, label: 'L' },
    { key: 'right-pointer', kind: 'pointer', x: right, y: 0, label: 'R' },
  ],
});

const draft = visual('Maintain the exact deficit count while the right boundary gains characters and the left boundary removes only proven surplus.', [
  frame('Initialize required counts', 'For text "ABAAC" and required "AAC", L = 0, the next R is 0, need has A:2 and C:1, and missing = 3.', windowScene(0, 0, [['A', '2'], ['C', '1']], {
    range: 'empty before reading index 0',
    required: 'need A:2, C:1',
    missing: '3',
    best: 'none',
  }), 'initialize'),
  frame('Expand through A at index 0', 'need[A] was positive, so this A fills one requirement: missing becomes 2 and need[A] becomes 1.', windowScene(0, 0, [['A', '1'], ['C', '1']], {
    range: '[0..0] = "A"',
    direction: 'R reads index 0',
    missing: '3 -> 2',
  }), 'expand-a-0'),
  frame('Expand through irrelevant B', 'need[B] is 0, so B does not change missing; decrementing need[B] to -1 records one removable surplus.', windowScene(0, 1, [['A', '1'], ['B', '-1'], ['C', '1']], {
    range: '[0..1] = "AB"',
    direction: 'R: 0 -> 1',
    missing: '2',
  }), 'expand-b-1'),
  frame('Expand through the second A', 'need[A] was 1, so index 2 fills the second required A: missing becomes 1 and need[A] becomes 0.', windowScene(0, 2, [['A', '0'], ['B', '-1'], ['C', '1']], {
    range: '[0..2] = "ABA"',
    direction: 'R: 1 -> 2',
    missing: '2 -> 1',
  }), 'expand-a-2'),
  frame('Record a surplus A', 'need[A] is already 0, so the A at index 3 is surplus: missing stays 1 and need[A] becomes -1.', windowScene(0, 3, [['A', '-1'], ['B', '-1'], ['C', '1']], {
    range: '[0..3] = "ABAA"',
    direction: 'R: 2 -> 3',
    missing: '1',
  }), 'expand-a-3'),
  frame('Complete the first valid window', 'C fills the final deficit, making missing = 0. Save "ABAAC" as the first valid window of length 5.', windowScene(0, 4, [['A', '-1'], ['B', '-1'], ['C', '0']], {
    range: '[0..4] = "ABAAC"',
    direction: 'R: 3 -> 4',
    missing: '1 -> 0',
    best: '"ABAAC" (length 5)',
  }), 'expand-c-4'),
  frame('Remove the surplus A', 'Increment need[A] from -1 to 0. No required copy is lost, so missing remains 0 and L moves right.', windowScene(1, 4, [['A', '0'], ['B', '-1'], ['C', '0']], {
    range: '[1..4] = "BAAC"',
    direction: 'L: 0 -> 1',
    reason: 'removed surplus A; still valid',
    best: '"BAAC" (length 4)',
  }), 'shrink-a-0'),
  frame('Remove the irrelevant B', 'Increment need[B] from -1 to 0. The window remains valid, so L advances and the best becomes "AAC".', windowScene(2, 4, [['A', '0'], ['B', '0'], ['C', '0']], {
    range: '[2..4] = "AAC"',
    direction: 'L: 1 -> 2',
    reason: 'removed surplus B; still valid',
    best: '"AAC" (length 3)',
    update: 'best = "AAC" (length 3)',
  }), 'shrink-b-1'),
  frame('Stop after removing a required A', 'Increment need[A] from 0 to 1, so missing becomes 1. L moves to 3 and shrinking stops because [3..4] lacks one A.', windowScene(3, 4, [['A', '1'], ['B', '0'], ['C', '0']], {
    range: '[3..4] = "AC"',
    direction: 'L: 2 -> 3',
    reason: 'required A removed; validity breaks',
    missing: '0 -> 1',
  }), 'break-a-2'),
  frame('Return the shortest saved window', 'The scan is complete and the saved slice starts at 2 with length 3, so the result is "AAC".', arrayMap(text, [['best_start', '2'], ['best_length', '3']], [
    mark(2, 'best L', 'output', 'best-left'),
    mark(4, 'best R', 'output', 'best-right'),
  ], {
    range: '[2..4] = "AAC"',
    result: '"AAC"',
  }), 'result'),
]);

const review = {
  pattern: 'Variable-size sliding window that grows to feasibility and shrinks to minimality.',
  recognitionCue: 'Use it when a contiguous result must cover a multiset of required values and validity can be updated by adding or removing one boundary value.',
  invariant: 'need[x] is the remaining deficit for x (negative means surplus), missing is the total number of required copies absent from [L..R], and best is the shortest valid window seen before the current transition.',
  stateModel: 'The minimal state is L, R, need, missing, best_start, and best_length. The fixed indexed text, moving authored pointers, covered range, and signed deficits expose every gain and loss of validity.',
  visualRationale: 'An indexed array paired with the signed need map shows both boundary geometry and duplicate counts; explicit directions and reasons remain understandable without color, JavaScript, or source-code recall.',
  rejectedAlternatives: [
    'A substring-only animation was rejected because it hides stable indices and why each boundary moves.',
    'A counts-only table was rejected because it hides which concrete contiguous range the counts describe.',
    'A final-window highlight was rejected because it omits the grow-until-valid and shrink-until-invalid mechanism.',
  ],
  transferLesson: 'When validity is monotone under expansion, grow until feasible, then remove leftmost surplus while feasible; a deficit counter can collapse a full multiset comparison into constant-time boundary updates.',
  reviewStatus: 'reviewed',
};

export default defineVisual('minimum-window-substring', draft, review);

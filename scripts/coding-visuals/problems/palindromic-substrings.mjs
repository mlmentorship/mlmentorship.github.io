import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const chars = ['a', 'a', 'a'];
const center = (middle, left, right, odd, even, total, result) => array(chars, [
  mark(left, `L=${left}`, 'state', 'left'),
  mark(right, `R=${right}`, result ? 'output' : 'focus', 'right'),
  mark(middle, `middle=${middle}`, 'focus', 'middle'),
], {
  input: 'aaa',
  oddExpansion: odd,
  evenExpansion: even,
  contribution: `${odd.match(/\d+$/)?.[0] ?? '?'} odd + ${even.match(/\d+$/)?.[0] ?? '?'} even`,
  runningTotal: String(total),
  ...(result ? { result } : {}),
});

const draft = visual('Each successful radius around each odd or even center identifies exactly one distinct palindromic substring occurrence.', [
  frame('Initialize center sum', 'For aaa, the generator starts at middle 0 and the accumulated sum is 0.', array(chars, [mark(0, 'next middle=0', 'focus', 'middle')], { input: 'aaa', runningTotal: '0' }), 'initialize'),
  frame('Count middle 0', 'Odd expansion counts a at [0,0]. Even expansion counts aa at [0,1]. The center contributes 2.', center(0, 0, 1, '[0,0] a -> 1', '[0,1] aa -> 1', 2), 'middle-0'),
  frame('Count middle 1', 'Odd expansion counts a and aaa for 2. Even expansion counts aa at [1,2] for 1. Running total becomes 5.', center(1, 0, 2, '[1,1] a; [0,2] aaa -> 2', '[1,2] aa -> 1', 5), 'middle-1'),
  frame('Count middle 2 and return', 'Odd expansion counts a at [2,2]. Even expansion starts with right=3 out of bounds. Total is 6.', center(2, 2, 2, '[2,2] a -> 1', 'right=3 out of bounds -> 0', 6, '6'), 'middle-2'),
]);

export default defineVisual('palindromic-substrings', draft, {
  pattern: 'Count successful expansions from every odd and even center.',
  recognitionCue: 'The task counts contiguous palindrome occurrences, including equal text at different positions, so each center-radius pair should contribute separately.',
  invariant: 'Each successful while iteration adds one new palindrome for that center and then expands one cell outward. Completed centers contribute disjoint occurrences because every palindrome has one unique center.',
  stateModel: 'Keep middle, left, right, a local expansion count, and the outer sum. Characters remain unchanged and no set is needed because occurrences, not distinct values, are counted.',
  visualRationale: 'An indexed character row with stable middle/L/R keys exposes odd and even center geometry and lists every accepted range, contribution, mismatch or boundary stop, and running sum.',
  rejectedAlternatives: [
    'A set of palindrome strings incorrectly merges duplicate occurrences.',
    'A final count of six does not show the three singles, two pairs, and one triple.',
    'A DP matrix is valid but heavier than the constant-space implementation.',
  ],
  transferLesson: 'Map each combinatorial object to a unique center and radius, then count every successful expansion once; the same enumeration underlies longest-palindrome and palindrome-radius algorithms.',
  reviewStatus: 'reviewed',
});

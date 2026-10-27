import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const chars = ['a', 'a', 'a'];
const center = (middle, left, right, odd, even, total, result) => array(chars, [
  mark(left, 'L', 'state', 'left'),
  mark(right, 'R', result ? 'output' : 'focus', 'right'),
  mark(middle, 'm', 'focus', 'middle'),
], {
  input: 'aaa',
  fitArray: true,
  arrayColumns: 3,
  oddExpansion: odd.map(([start, end]) => `[${start},${end}) = ${chars.slice(start, end).join('')}`).join(', '),
  evenExpansion: even.map(([start, end]) => `[${start},${end}) = ${chars.slice(start, end).join('')}`).join(', ') || 'none: right=3 is outside',
  contribution: `${odd.length} odd + ${even.length} even`,
  runningTotal: String(total),
  ...(result ? { result } : {}),
});

const draft = visual('Each accepted radius contributes one range. Equal text at different positions counts separately.', [
  frame('Initialize center sum', 'For aaa, the lazy sum has not consumed a center yet. Its accumulated total is zero.', array(chars, [mark(0, 'm', 'focus', 'middle')], { input: 'aaa', fitArray: true, arrayColumns: 3, runningTotal: '0' }), 'initialize'),
  frame('Count middle 0', 'Middle zero accepts [0,1) and [0,2). Two new occurrences increase the total from zero to two.', center(0, 0, 1, [[0,1]], [[0,2]], 2), 'middle-0'),
  frame('Count middle 1', 'Middle one accepts [1,2), [0,3) and [1,3). Three more occurrences raise the total to five.', center(1, 0, 2, [[1,2],[0,3]], [[1,3]], 5), 'middle-1'),
  frame('Count middle 2 and return', 'Middle two accepts [2,3). The even call starts outside the string. Total six includes three separate singleton positions.', center(2, 2, 2, [[2,3]], [], 6, '6'), 'middle-2'),
]);

export default defineVisual('palindromic-substrings', draft, {
  pattern: 'Count successful expansions from every odd and even center.',
  recognitionCue: 'The task counts contiguous palindrome occurrences, including equal text at different positions, so each center-radius pair should contribute separately.',
  invariant: 'Each successful while iteration adds one new palindrome for that center and then expands one cell outward. Completed centers contribute disjoint occurrences because every palindrome has one unique center.',
  stateModel: 'Keep middle, left, right, a local expansion count, and the outer sum. Characters remain unchanged and no set is needed because occurrences, not distinct values, are counted.',
  visualRationale: 'A fitted indexed character row retains the original middle/L/R motion identities. Structured half-open ranges distinguish all six equal-text occurrences, expose odd/even contributions and preserve cumulative totals zero, two, five and six.',
  rejectedAlternatives: [
    'A set of palindrome strings incorrectly merges duplicate occurrences.',
    'A final count of six does not show the three singles, two pairs, and one triple.',
    'A DP matrix is valid but heavier than the constant-space implementation.',
  ],
  transferLesson: 'Map each combinatorial object to a unique center and radius, then count every successful expansion once; the same enumeration underlies longest-palindrome and palindrome-radius algorithms.',
  reviewStatus: 'reviewed',
});

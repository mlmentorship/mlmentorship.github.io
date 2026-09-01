import { arrayMap, defineVisual, frame, mark, visual } from '../primitives.mjs';

const characters = ['a', 'a', 'b', '|', 'a', 'b', 'a'];
const example = 'first = "aab", second = "aba"';
const scan = (index, label) => mark(index, label, 'focus', 'counter-cursor');

const draft = visual('Two strings are anagrams exactly when their character-count maps are equal.', [
  frame(
    'Initialize both counters',
    'Counter starts with no entries for either string; the divider separates first="aab" from second="aba".',
    arrayMap(characters, [], [scan(0, 'count first')], { example, mapLabel: 'Counter(first) and Counter(second)' }),
    'initialize-counters',
  ),
  frame(
    'Count the first a',
    'Reading first[0]="a" changes the left count for a from 0 to 1.',
    arrayMap(characters, [['a', 'left 1 | right 0']], [scan(0, 'left a: 0 -> 1')], { example }),
    'count-first-a',
  ),
  frame(
    'Count the repeated a',
    'Reading first[1]="a" changes the same left entry from 1 to 2.',
    arrayMap(characters, [['a', 'left 2 | right 0']], [scan(1, 'left a: 1 -> 2')], { example }),
    'count-second-a',
  ),
  frame(
    'Finish the first counter',
    'Reading first[2]="b" adds b:1, so Counter(first) is {a:2, b:1}.',
    arrayMap(characters, [['a', 'left 2 | right 0'], ['b', 'left 1 | right 0']], [scan(2, 'left b: 0 -> 1')], { example, leftCounter: '{a:2, b:1}' }),
    'finish-first-counter',
  ),
  frame(
    'Start the second counter',
    'Counter(second) is independent. Reading second[0]="a" sets its a count to 1.',
    arrayMap(characters, [['a', 'left 2 | right 1'], ['b', 'left 1 | right 0']], [scan(4, 'right a: 0 -> 1')], { example }),
    'count-second-word-a',
  ),
  frame(
    'Count b in the second string',
    'Reading second[1]="b" sets its b count to 1.',
    arrayMap(characters, [['a', 'left 2 | right 1'], ['b', 'left 1 | right 1']], [scan(5, 'right b: 0 -> 1')], { example }),
    'count-second-word-b',
  ),
  frame(
    'Finish the second counter',
    'Reading second[2]="a" changes its a count from 1 to 2, producing {a:2, b:1}.',
    arrayMap(characters, [['a', 'left 2 | right 2'], ['b', 'left 1 | right 1']], [scan(6, 'right a: 1 -> 2')], { example, rightCounter: '{a:2, b:1}' }),
    'finish-second-counter',
  ),
  frame(
    'Compare the complete maps',
    'Both keys and counts agree: a has 2 on each side and b has 1 on each side, so Counter(first) == Counter(second).',
    arrayMap(characters, [['a', 'left 2 = right 2'], ['b', 'left 1 = right 1']], [mark(2, 'left complete', 'output', 'left-result'), scan(6, 'right complete')], { example, comparison: '{a:2,b:1} = {a:2,b:1}', result: 'true' }),
    'compare-counters',
  ),
]);

const review = {
  pattern: 'Frequency-map equality using one character count per distinct symbol.',
  recognitionCue: 'Order may differ, but the question asks whether two strings contain exactly the same characters with the same multiplicities.',
  invariant: 'After each Counter scan step, the visible count for a character equals its occurrences in the processed prefix; complete equal maps are necessary and sufficient for anagrams.',
  stateModel: 'Retain two maps from character to count and a cursor for the string currently counted. Positions and permutations do not need to be retained.',
  visualRationale: 'A stable character strip plus the two evolving counts exposes repeated letters, order independence, and the final map equality without requiring code recall.',
  rejectedAlternatives: [
    'Drawing lines between matching letters becomes ambiguous with duplicates and suggests positional pairing.',
    'Sorting both strings is valid but teaches a different O(n log n) mechanism than the supplied Counter solution.',
    'Showing only the two final maps hides how repeated characters change the state.',
  ],
  transferLesson: 'Replace irrelevant order with a canonical frequency summary; the same idea supports ransom-note checks, multiset equality, permutation detection, and histogram comparison.',
  reviewStatus: 'reviewed',
};

export default defineVisual('valid-anagram', draft, review);

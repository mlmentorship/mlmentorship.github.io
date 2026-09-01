import { arrayMap, defineVisual, frame, mark, visual } from '../primitives.mjs';

const items = ['e', 'i', 'd', 'b', 'a', 'o', 'o', 'o'];
const pointers = (left, right) => [
  mark(left, 'L', 'state', 'left'),
  mark(right, 'R', 'focus', 'right'),
];
const state = (window) => [
  ['need', 'a:1, b:1'],
  ['window', window],
];

const draft = visual('Slide one width-2 window over "eidbaooo"; add R, remove the outgoing L character, then compare counts.', [
  frame('Count the first width-2 window', 'pattern = "ab" needs a:1,b:1. text[0..1] = "ei" has e:1,i:1, so the counters differ.', arrayMap(items, state('e:1, i:1'), pointers(0, 1), {
    range: '[0..1] = "ei"',
    comparison: 'e:1,i:1 != a:1,b:1',
  }), 'initial-window'),
  frame('Slide from ei to id', 'At R = 2 add d, remove outgoing text[0] = e, delete e after its count reaches zero, and move L right to 1.', arrayMap(items, state('d:1, i:1'), pointers(1, 2), {
    range: '[1..2] = "id"',
    direction: 'L: 0 -> 1; R: 1 -> 2',
    comparison: 'd:1,i:1 != a:1,b:1',
  }), 'window-id'),
  frame('Slide from id to db', 'At R = 3 add b, remove outgoing text[1] = i, delete zero-count i, and move L right to 2.', arrayMap(items, state('b:1, d:1'), pointers(2, 3), {
    range: '[2..3] = "db"',
    direction: 'L: 1 -> 2; R: 2 -> 3',
    comparison: 'b:1,d:1 != a:1,b:1',
  }), 'window-db'),
  frame('Match the permutation ba', 'At R = 4 add a, remove outgoing text[2] = d, and delete zero-count d. The window counter is now exactly a:1,b:1.', arrayMap(items, state('a:1, b:1'), pointers(3, 4), {
    range: '[3..4] = "ba"',
    direction: 'L: 2 -> 3; R: 3 -> 4',
    comparison: 'a:1,b:1 = a:1,b:1',
    result: 'true',
  }), 'window-ba-match'),
]);

const review = {
  pattern: 'Fixed-size sliding window with a pattern counter and an incrementally maintained window counter.',
  recognitionCue: 'Use it when a contiguous match may appear in any order: equal multisets require equal length, so only windows whose width equals the pattern length can qualify.',
  invariant: 'Before each comparison, window contains exactly text[R - pattern_length + 1..R], because the new right character was added and the one character that fell off the left was removed.',
  stateModel: 'The minimal state is the immutable need counter, current window counter, and width-sized L and R boundaries. Zero-count keys are deleted so Counter equality matches the source implementation.',
  visualRationale: 'The indexed text keeps every candidate position visible while the adjacent counters explain order independence. Stable L and R labels show the fixed-width movement without relying on color or JavaScript.',
  rejectedAlternatives: [
    'A counters-only table was rejected because it hides which text character enters and which one leaves.',
    'A sorted-substring animation was rejected because sorting every candidate is not the supplied algorithm.',
    'A permutation tree was rejected because enumerating pattern orders adds factorial clutter and misses the count invariant.',
  ],
  transferLesson: 'When every candidate has a known width, maintain its summary by adding the entering item and removing the outgoing item; compare summaries instead of rebuilding or enumerating arrangements.',
  reviewStatus: 'reviewed',
};

export default defineVisual('permutation-in-string', draft, review);

import { arrayMap, defineVisual, frame, mark, visual } from '../primitives.mjs';

const items = ['a', 'b', 'b', 'a'];
const pointers = (left, right) => [
  mark(left, 'L', 'state', 'left'),
  mark(right, 'R', 'focus', 'right'),
];

const draft = visual('Keep [L..R] duplicate-free by jumping L past a repeated character that is still inside the window.', [
  frame('Initialize the empty history', 'For text "abba", start L = 0, best = 0, and inspect R = 0.', arrayMap(items, [], pointers(0, 0), {
    range: 'empty before reading index 0',
    best: '0',
  }), 'initialize'),
  frame('Read a at index 0', 'No earlier a exists, so keep L = 0, save a -> 0, and set best = max(0, 1) = 1.', arrayMap(items, [['a', '0']], pointers(0, 0), {
    range: '[0..0] = "a"',
    length: '0 - 0 + 1 = 1',
    best: '1',
  }), 'read-a-0'),
  frame('Extend through b at index 1', 'b is new, so [0..1] stays distinct; save b -> 1 and raise best to 2.', arrayMap(items, [['a', '0'], ['b', '1']], pointers(0, 1), {
    range: '[0..1] = "ab"',
    length: '1 - 0 + 1 = 2',
    best: '2',
  }), 'read-b-1'),
  frame('Jump past the repeated b', 'At R = 2, the old b is at 1 inside [0..1], so move L right to max(0, 1 + 1) = 2.', arrayMap(items, [['a', '0'], ['b', '2']], pointers(2, 2), {
    range: '[2..2] = "b"',
    direction: 'L: 0 -> 2',
    best: 'max(2, 1) = 2',
  }), 'repair-b-2'),
  frame('Ignore an occurrence left of L', 'At R = 3, the old a is at 0, outside [2..2]; max(2, 0 + 1) keeps L = 2, so "ba" has length 2.', arrayMap(items, [['a', '3'], ['b', '2']], pointers(2, 3), {
    range: '[2..3] = "ba"',
    direction: 'L stays 2; R: 2 -> 3',
    best: 'max(2, 2) = 2',
    result: '2',
  }), 'finish-a-3'),
]);

const review = {
  pattern: 'Variable-size sliding window with a map from each character to its latest index.',
  recognitionCue: 'Use it when the answer is a longest contiguous span constrained by uniqueness and a repeated value tells exactly how far the left boundary can safely jump.',
  invariant: 'After processing index R, [L..R] has no repeated character; L never moves backward, and best is the maximum length of every valid window ending at or before R.',
  stateModel: 'The minimal state is L, R, best, and last_seen[character]. The trace keeps the indexed text fixed while the authored left and right pointers move and the map records exact indices.',
  visualRationale: 'An indexed array beside the last-seen map directly exposes whether an old occurrence lies inside [L..R]. Pointer labels and arithmetic remain meaningful without color or playback.',
  rejectedAlternatives: [
    'A counts-only table was rejected because it hides character positions and cannot explain the direct left-boundary jump.',
    'A substring-only animation was rejected because removing indices obscures stale occurrences outside the current window.',
    'A prose timeline was rejected because readers would have to reconstruct the covered range and map in working memory.',
  ],
  transferLesson: 'For longest windows repaired by the newest violation, store enough history to jump over the violating occurrence, but clamp the new left boundary so stale history can never move it backward.',
  reviewStatus: 'reviewed',
};

export default defineVisual('longest-substring-without-repeating-characters', draft, review);

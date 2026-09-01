import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Compare letter counts, not letter positions.', [
    frame('Count the first word', 'eat contributes one e, one a, and one t.', table(['letter', 'eat', 'tea'], [['a', '1', '-'], ['e', '1', '-'], ['t', '1', '-']], [1, 4, 7])),
    frame('Consume the second word', 'tea removes the same three counts in a different order.', table(['letter', 'eat', 'tea'], [['a', '1', '1'], ['e', '1', '1'], ['t', '1', '1']], [1, 2, 4, 5, 7, 8], { status: 'all counts match' })),
    frame('Accept', 'Every count is equal, so the strings are anagrams.', table(['letter', 'left', 'right'], [['a', '1', '1'], ['e', '1', '1'], ['t', '1', '1']], [], { status: 'true', result: 'true' })),
  ]);

export default defineVisual('valid-anagram', draft, pendingReview(draft.objective));

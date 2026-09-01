import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A literal follows one trie child; a dot branches over every child.', [
    frame('Store words', 'bad, dad, and mad share the suffix ad after different first letters.', trie([{ word: 'bad', prefix: 'b-a-d' }, { word: 'dad', prefix: 'd-a-d' }, { word: 'mad', prefix: 'm-a-d' }], { action: 'insert three words' })),
    frame('Match a wildcard', 'For .ad, the dot tries b, d, and m, then follows a-d.', trie([{ word: '.ad', prefix: 'b/d/m -> a -> d', tone: 'focus' }], { query: '.ad', action: 'branch at dot' })),
    frame('Return true', 'One wildcard branch reaches a terminal word marker.', trie([{ word: 'bad', prefix: 'b-a-d', tone: 'output' }, { word: 'dad', prefix: 'd-a-d' }, { word: 'mad', prefix: 'm-a-d' }], { result: 'true' })),
  ]);

export default defineVisual('design-add-and-search-words', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A shared character path stores prefixes once, with a terminal marker for complete words.', [
    frame('Insert cat', 'The path root-c-a-t is created and t gets an end marker.', trie([{ word: 'cat', prefix: 'c-a-t', tone: 'focus' }], { action: 'insert cat' })),
    frame('Share c-a', 'Inserting car reuses c-a and branches only at the final character.', trie([{ word: 'cat', prefix: 'c-a-t', tone: 'state' }, { word: 'car', prefix: 'c-a-r', tone: 'focus' }], { action: 'share prefix c-a' })),
    frame('Search a prefix', 'starts_with("ca") succeeds even before choosing t or r.', trie([{ word: 'cat', prefix: 'c-a-t', tone: 'output' }, { word: 'car', prefix: 'c-a-r', tone: 'output' }], { query: 'ca', result: 'true' })),
  ]);

export default defineVisual('implement-trie', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A map finds a node; a doubly linked list keeps least-recent to most-recent order.', [
    frame('Insert and read', 'After put(1), put(2), get(1), the order is 2 -> 1.', lru([['1', 'node'], ['2', 'node']], ['least: 2', 'most: 1'], { action: 'get(1) moves it right' })),
    frame('Evict the left edge', 'put(3) appends 3 and removes least-recent key 2.', lru([['1', 'node'], ['3', 'node']], ['least: 1', 'most: 3'], { evicted: '2' })),
    frame('Lookup misses', 'get(2) returns -1 because the map and list no longer contain it.', lru([['1', 'node'], ['3', 'node']], ['least: 1', 'most: 3'], { result: 'get(2) = -1' })),
  ]);

export default defineVisual('lru-cache', draft, pendingReview(draft.objective));

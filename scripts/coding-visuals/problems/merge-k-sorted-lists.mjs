import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('The heap holds one current head per list; pop the smallest and replace it with that list next.', [
    frame('Seed one head per list', 'The heap contains 1 from list A, 1 from B, and 2 from C.', heap(['A:1', 'B:1', 'C:2'], { root: 'A:1', detail: 'one head per list' })),
    frame('Pop and replace', 'After taking A:1, insert A:4 while B:1 remains the root.', heap(['B:1', 'C:2', 'A:4'], { root: 'B:1', detail: 'replace from same list' })),
    frame('Finish from the remaining heads', 'After emitting 1,1,2,3,4,4, the remaining heads are 5 and 6.', heap(['A:5', 'C:6'], { root: 'A:5', detail: 'emit 5, then 6', result: '1,1,2,3,4,4,5,6' })),
  ]);

export default defineVisual('merge-k-sorted-lists', draft, pendingReview(draft.objective));

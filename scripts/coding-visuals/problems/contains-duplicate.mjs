import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('The first repeated value is visible when it is already in the seen set.', [
    frame('Save new values', '1, 2, and 3 have not appeared before.', arrayMap(['1', '2', '3', '1'], [['1', 'seen'], ['2', 'seen'], ['3', 'seen']], [mark(2, 'current', 'focus')])),
    frame('Detect the repeat', 'The final 1 is already in the set.', arrayMap(['1', '2', '3', '1'], [['1', 'seen'], ['2', 'seen'], ['3', 'seen']], [mark(0, 'same value', 'output'), mark(3, 'repeat', 'output')])),
    frame('Return true', 'A set membership hit proves a duplicate exists.', array(['1', '2', '3', '1'], [mark(3, 'duplicate', 'output')], { result: 'true' })),
  ]);

export default defineVisual('contains-duplicate', draft, pendingReview(draft.objective));

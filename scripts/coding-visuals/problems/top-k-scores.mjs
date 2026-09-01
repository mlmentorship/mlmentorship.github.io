import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Partition finds membership in the top group; sort only that group for output order.', [
    frame('Partition the scores', 'Scores 0.9 and 0.8 belong to the top-2 group.', array(['0.1', '0.9', '0.4', '0.8'], [mark(1, 'candidate', 'state'), mark(3, 'candidate', 'state')], { action: 'argpartition' })),
    frame('Sort selected candidates', 'Only selected indices 1 and 3 need final ordering.', array(['index 1: 0.9', 'index 3: 0.8'], [mark(0, 'first', 'focus'), mark(1, 'second', 'state')], { action: 'sort k' })),
    frame('Return indices', 'The descending top-k indices are [1,3].', array(['1', '3'], [mark(0, 'top', 'output'), mark(1, 'top', 'output')], { result: '[1,3]' })),
  ]);

export default defineVisual('top-k-scores', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('At each house, choose between skipping it and taking it after the previous house.', [
    frame('Before any house', 'The best totals two houses back and one house back are both zero.', array(['2', '7', '9', '3', '1'], [mark(0, 'current', 'focus')], { states: 'two_back=0, one_back=0' })),
    frame('Compare at 9', 'Skip 9 gives 7; take 9 gives 0 + 9. Keep 11 after the first three houses.', array(['2', '7', '9', '3', '1'], [mark(2, 'take', 'focus')], { states: 'skip=7, take=11, best=11' })),
    frame('Finish the line', 'The best non-adjacent selection is 2 + 9 + 1 = 12.', array(['2', '7', '9', '3', '1'], [mark(0, 'take', 'output'), mark(2, 'take', 'output'), mark(4, 'take', 'output')], { result: '12' })),
  ]);

export default defineVisual('house-robber', draft, pendingReview(draft.objective));

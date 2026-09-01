import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Discard a negative running prefix before extending a future subarray.', [
    frame('Carry a running sum', 'At 1, the negative prefix -2 is worse than starting a new subarray.', array(['-2', '1', '-3', '4', '-1', '2', '1'], [mark(0, 'drop', 'warning'), mark(1, 'start', 'focus')], { current: '1' })),
    frame('Extend the best ending here', 'Starting at 4, the running sum grows through -1, 2, and 1.', array(['4', '-1', '2', '1'], [mark(0, 'start', 'state'), mark(3, 'best ending', 'focus')], { current: '6' })),
    frame('Keep the global best', 'The maximum subarray is [4,-1,2,1] with sum 6.', array(['-2', '1', '-3', '4', '-1', '2', '1'], [mark(3, 'best', 'output'), mark(4, 'best', 'output'), mark(5, 'best', 'output'), mark(6, 'best', 'output')], { result: '6' })),
  ]);

export default defineVisual('maximum-subarray', draft, pendingReview(draft.objective));

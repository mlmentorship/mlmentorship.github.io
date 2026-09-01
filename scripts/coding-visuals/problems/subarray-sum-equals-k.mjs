import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Turn a target subarray into a lookup between two prefix sums.', [
    frame('Record the empty prefix', 'Before reading values, prefix sum 0 has appeared once.', arrayMap(['0', '1', '2', '3'], [['0', 'count 1']], [mark(0, 'prefix 0', 'state')])),
    frame('Reach prefix 2', 'After the second 1, current prefix is 2. It needs an earlier prefix 0.', arrayMap(['0', '1', '2', '3'], [['0', 'count 1'], ['1', 'count 1']], [mark(2, 'prefix 2', 'focus')], { query: '2 - k = 0' })),
    frame('Count every match', 'The prefix-2 query finds prefix 0; prefix 3 later finds prefix 1.', arrayMap(['0', '1', '2', '3'], [['0', 'count 1'], ['1', 'count 1'], ['2', 'count 1'], ['3', 'count 1']], [mark(2, 'match', 'output'), mark(3, 'match', 'output')], { result: '2 subarrays' })),
  ]);

export default defineVisual('subarray-sum-equals-k', draft, pendingReview(draft.objective));

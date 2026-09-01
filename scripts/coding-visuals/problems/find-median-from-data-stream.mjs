import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Keep the lower half in a max-heap and the upper half in a min-heap.', [
    frame('Add 1', 'The lower half contains 1; the upper half is empty.', heap(['lower max:1', 'upper min:-'], { root: 'lower 1' })),
    frame('Add 2', 'Balance the halves: lower has 1 and upper has 2.', heap(['lower max:1', 'upper min:2'], { detail: 'two roots bracket the median' })),
    frame('Read the middle', 'With two values, median is (1+2)/2 = 1.5.', heap(['lower max:1', 'upper min:2'], { result: '1.5' })),
  ]);

export default defineVisual('find-median-from-data-stream', draft, pendingReview(draft.objective));

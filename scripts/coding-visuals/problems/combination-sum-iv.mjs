import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Count ordered sequences by choosing the final number of each target.', [
    frame('Base count', 'There is one way to make total 0: choose nothing.', array(['1', '0', '0', '0', '0'], [mark(0, 'base', 'state')])),
    frame('Build totals', 'ways[3] includes sequences ending in 1, 2, or 3.', array(['1', '1', '2', '4', '7'], [mark(3, '4 ways', 'state'), mark(4, '7 ways', 'focus')])),
    frame('Return target count', 'The seven sequences for target 4 include 1+3 and 3+1 separately.', array(['1', '1', '2', '4', '7'], [mark(4, 'answer', 'output')], { result: '7' })),
  ]);

export default defineVisual('combination-sum-iv', draft, pendingReview(draft.objective));

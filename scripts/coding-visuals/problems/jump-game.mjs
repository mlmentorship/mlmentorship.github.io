import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Carry the farthest index reachable from everything scanned so far.', [
    frame('Reach from index 0', 'At index 0 with jump 2, the reachable boundary is 2.', array(['2', '3', '1', '1', '4'], [mark(0, 'scan', 'focus'), mark(2, 'reach', 'state')], { reach: '2' })),
    frame('Extend the boundary', 'Index 1 can reach 4, so the boundary moves to the last index.', array(['2', '3', '1', '1', '4'], [mark(1, 'scan', 'focus'), mark(4, 'reach', 'output')], { reach: '4' })),
    frame('Reach the end', 'The last index is at or before the farthest boundary.', array(['2', '3', '1', '1', '4'], [mark(4, 'goal', 'output')], { result: 'true' })),
  ]);

export default defineVisual('jump-game', draft, pendingReview(draft.objective));

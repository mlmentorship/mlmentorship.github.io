import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A single cursor yields non-overlapping slices and keeps the final short slice.', [
    frame('Take the first slice', 'Start 0 with size 3 yields items 0,1,2.', array(['0', '1', '2', '3', '4', '5', '6'], [mark(0, 'start', 'focus'), mark(2, 'end', 'focus')], { batch: '[0:3]' })),
    frame('Advance the cursor', 'Start 3 yields the next three items.', array(['0', '1', '2', '3', '4', '5', '6'], [mark(3, 'start', 'focus'), mark(5, 'end', 'focus')], { batch: '[3:6]' })),
    frame('Keep the remainder', 'Start 6 yields [6] instead of dropping it.', array(['0', '1', '2', '3', '4', '5', '6'], [mark(6, 'remainder', 'output')], { result: '[[0,1,2],[3,4,5],[6]]' })),
  ]);

export default defineVisual('mini-batches', draft, pendingReview(draft.objective));

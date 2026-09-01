import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Use the sorted half to decide which side of the rotation can contain the target.', [
    frame('Identify a sorted half', 'For [4,5,6,7,0,1,2], the left half 4..7 is sorted.', array(['4', '5', '6', '7', '0', '1', '2'], [mark(0, 'L'), mark(3, 'mid', 'focus'), mark(6, 'R')], { detail: 'left half sorted' })),
    frame('Choose the other half', 'Target 0 is not inside 4..7, so discard the sorted left half.', array(['4', '5', '6', '7', '0', '1', '2'], [mark(4, 'L', 'state'), mark(5, 'mid', 'focus'), mark(6, 'R', 'state')], { detail: 'search right half' })),
    frame('Find the target', 'The right half reaches 0 at index 4.', array(['4', '5', '6', '7', '0', '1', '2'], [mark(4, 'found', 'output')], { result: 'index 4' })),
  ]);

export default defineVisual('search-in-rotated-sorted-array', draft, pendingReview(draft.objective));

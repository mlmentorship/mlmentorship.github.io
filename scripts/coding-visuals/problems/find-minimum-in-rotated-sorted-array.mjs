import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('The drop lies on the side where the middle value exceeds the right boundary.', [
    frame('Compare middle and right', 'At mid 7 and right 2, the minimum is to the right of mid.', array(['4', '5', '6', '7', '0', '1', '2'], [mark(3, 'mid', 'focus'), mark(6, 'right', 'state')])),
    frame('Keep the rotation', 'The interval becomes [0,1,2].', array(['4', '5', '6', '7', '0', '1', '2'], [mark(4, 'lo', 'state'), mark(5, 'mid', 'focus'), mark(6, 'hi', 'state')])),
    frame('Minimum is at lo', 'When lo meets hi, that element is the minimum.', array(['4', '5', '6', '7', '0', '1', '2'], [mark(4, 'minimum', 'output')], { result: '0' })),
  ]);

export default defineVisual('find-minimum-in-rotated-sorted-array', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Binary-search the smallest eating speed that finishes within the hour limit.', [
    frame('Test a speed', 'At speed 4, piles [3,6,7,11] take 1+2+2+3 = 8 hours.', array(['speed 1', 'speed 2', 'speed 3', 'speed 4', 'speed 5', '...'], [mark(3, 'test', 'focus')], { detail: 'hours = 8; feasible' })),
    frame('Discard an infeasible speed', 'Speed 3 takes 10 hours, so the smallest feasible speed is at least 4. Keep [4,5].', array(['1', '2', '3', '4', '5', '...','11'], [mark(2, 'test', 'warning'), mark(3, 'lo', 'state'), mark(4, 'hi', 'state')], { detail: '10 hours > 8' })),
    frame('Return the first feasible speed', 'Speed 4 is the smallest speed whose hours fit.', array(['1', '2', '3', '4', '5', '...','11'], [mark(3, 'answer', 'output')], { result: '4 bananas/hour' })),
  ]);

export default defineVisual('koko-eating-bananas', draft, pendingReview(draft.objective));

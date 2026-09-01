import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Each step can be reached from one step back or two steps back.', [
    frame('Base cases', 'The rolling state starts with a sentinel before step 1 and one way to reach step 1.', array(['before 1', 'step 1', 'step 2', 'step 3', 'step 4', 'step 5'], [mark(0, 'sentinel', 'state'), mark(1, '1 way', 'state')], { states: 'previous=0, current=1' })),
    frame('Build forward', 'ways(3) = ways(2) + ways(1) = 2 + 1 = 3.', array(['0', '1', '2', '3', '?', '?'], [mark(2, '2', 'state'), mark(3, '3', 'focus')])),
    frame('Keep only two totals', 'ways(5) = 8, and earlier totals are no longer needed.', array(['0', '1', '2', '3', '5', '8'], [mark(5, '8', 'output')], { result: '8' })),
  ]);

export default defineVisual('climbing-stairs', draft, pendingReview(draft.objective));

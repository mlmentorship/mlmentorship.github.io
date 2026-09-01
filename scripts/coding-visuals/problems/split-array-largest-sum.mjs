import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Guess a maximum part sum, greedily count required parts, and binary-search the smallest feasible guess.', [
    frame('Set answer bounds', 'The largest part must be at least max(nums)=10 and at most total 32.', array(['10', '11', '12', '...', '31', '32'], [mark(0, 'lo', 'state'), mark(5, 'hi', 'state')])),
    frame('Test a limit', 'With limit 18, greedy cuts [7,2,5] and [10,8], using two parts.', array(['7+2+5=14', '10+8=18'], [mark(0, 'part 1', 'focus'), mark(1, 'part 2', 'state')], { parts: '2 <= k' })),
    frame('Return the smallest feasible limit', '18 works, while 17 would require three parts.', array(['17', '18', '19'], [mark(1, 'answer', 'output')], { result: '18' })),
  ]);

export default defineVisual('split-array-largest-sum', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('The topological queue is also the feasible course order.', [
    frame('Seed zero-indegree courses', 'Only course 0 has no unmet prerequisite.', graph(['0', '1', '2'], ['0 -> 1', '1 -> 2'], { indegree: ['0:0', '1:1', '2:1'], ready: ['0'] })),
    frame('Append and decrement', 'Taking 0 makes 1 ready; taking 1 makes 2 ready.', graph(['0', '1', '2'], ['0 -> 1', '1 -> 2'], { order: ['0', '1'], ready: ['2'] })),
    frame('Return the order', 'The queue emitted a valid prerequisite-respecting sequence.', graph(['0', '1', '2'], ['0 -> 1', '1 -> 2'], { order: ['0', '1', '2'], result: '[0,1,2]' })),
  ]);

export default defineVisual('course-schedule-ii', draft, pendingReview(draft.objective));

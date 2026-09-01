import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A course becomes ready when its remaining prerequisite count reaches zero.', [
    frame('Count prerequisites', 'Course 0 is ready; course 1 has one incoming edge.', graph(['course 0', 'course 1'], ['0 -> 1'], { indegree: ['0:0', '1:1'], ready: ['0'] })),
    frame('Complete a ready course', 'Removing course 0 decrements course 1 from 1 to 0.', graph(['course 0', 'course 1'], ['0 -> 1'], { indegree: ['0:done', '1:0'], ready: ['1'] })),
    frame('Finish all nodes', 'Every course entered the ready queue, so no cycle remains.', graph(['course 0', 'course 1'], ['0 -> 1'], { order: ['0', '1'], result: 'true' })),
  ]);

export default defineVisual('course-schedule', draft, pendingReview(draft.objective));

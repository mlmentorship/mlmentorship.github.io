import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Start a run only at a value with no predecessor.', [
    frame('Find a run start', '4 is skipped because 3 exists. 1 has no predecessor, so it starts a run.', arrayMap(['100', '4', '200', '1', '3', '2'], [['1', 'start']], [mark(3, 'start', 'focus')])),
    frame('Walk forward', 'The set answers 2, 3, and 4 in constant-time average lookups.', arrayMap(['1', '2', '3', '4'], [['1', 'run length 4']], [mark(0, 'start', 'state'), mark(3, 'end', 'output')])),
    frame('Keep the longest', 'Every other value either starts a shorter run or belongs to this one.', arrayMap(['1', '2', '3', '4'], [['1', 'best = 4']], [mark(0, 'best', 'output'), mark(1, 'best', 'output'), mark(2, 'best', 'output'), mark(3, 'best', 'output')], { result: '4' })),
  ]);

export default defineVisual('longest-consecutive-sequence', draft, pendingReview(draft.objective));

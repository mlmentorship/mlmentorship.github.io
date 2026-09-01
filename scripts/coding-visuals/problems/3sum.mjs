import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Fix one value, then solve the remaining pair with two sorted pointers.', [
    frame('Sort and fix', 'After sorting, fix -1 at index 1. The pair search starts to its right.', array(['-4', '-1', '-1', '0', '1', '2'], [mark(1, 'fixed', 'state'), mark(2, 'L'), mark(5, 'R')])),
    frame('Move toward zero', 'The sum -1 + -1 + 2 is 0, so record it and move both pointers.', array(['-4', '-1', '-1', '0', '1', '2'], [mark(1, 'fixed', 'state'), mark(2, 'pair', 'output'), mark(5, 'pair', 'output')], { result: '[-1,-1,2]' })),
    frame('Find the second triple', 'With fixed -1, pointers reach 0 and 1 and record [-1,0,1].', array(['-4', '-1', '-1', '0', '1', '2'], [mark(1, 'fixed', 'state'), mark(3, 'pair', 'output'), mark(4, 'pair', 'output')], { result: '[-1,0,1]' })),
  ]);

export default defineVisual('3sum', draft, pendingReview(draft.objective));

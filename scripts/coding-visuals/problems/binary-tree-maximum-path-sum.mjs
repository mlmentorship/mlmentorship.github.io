import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A node returns one child branch upward but can score both child branches locally.', [
    frame('Return one branch', 'At node 20, the larger child contribution is 15, while the full path can use 15 and 7.', tree([['-10'], ['9', '20'], ['-', '-', '15', '7']], [mark(5, 'return 15', 'state'), mark(2, 'score 42', 'focus')])),
    frame('Reject negative branches', 'A negative child contribution is replaced by zero before combining.', tree([['-10'], ['9', '20'], ['-', '-', '15', '7']], [mark(0, 'left 0', 'state'), mark(2, 'both children', 'focus')], { formula: '20 + 15 + 7 = 42' })),
    frame('Update global best', 'The path 15 -> 20 -> 7 has the maximum sum 42.', tree([['-10'], ['9', '20'], ['-', '-', '15', '7']], [mark(2, 'best path', 'output'), mark(5, 'best path', 'output'), mark(6, 'best path', 'output')], { result: '42' })),
  ]);

export default defineVisual('binary-tree-maximum-path-sum', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Pass the full inherited lower and upper bounds down each tree branch.', [
    frame('Set the root bounds', 'Root 5 must lie between negative and positive infinity.', tree([['5'], ['1', '7'], ['-', '-', '4', '-']], [mark(0, 'bounds (-inf,inf)', 'focus')])),
    frame('Carry an ancestor bound', 'Node 4 is in the right subtree of 5, so its lower bound is 5.', tree([['5'], ['1', '7'], ['-', '-', '4', '-']], [mark(5, '4 not > 5', 'warning')], { bounds: '4 must be > 5' })),
    frame('Reject the tree', 'A parent-only check would miss this violation; inherited bounds catch it.', tree([['5'], ['1', '7'], ['-', '-', '4', '-']], [mark(5, 'invalid', 'output')], { result: 'false' })),
  ]);

export default defineVisual('validate-binary-search-tree', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Try the full-tree equality test at each candidate node.', [
    frame('Scan candidate roots', 'The root 3 does not match subroot root 4, so search its children.', tree([['3'], ['4', '5'], ['1', '2', '-', '-']], [mark(0, 'try', 'focus')])),
    frame('Match at node 4', 'The subtree rooted at 4 has the same value and child shape.', tree([['3'], ['4', '5'], ['1', '2', '-', '-']], [mark(1, 'match', 'output'), mark(3, 'match', 'output'), mark(4, 'match', 'output')])),
    frame('Return true', 'One complete matching subtree is enough.', tree([['3'], ['4', '5'], ['1', '2', '-', '-']], [mark(1, 'subtree', 'output')], { result: 'true' })),
  ]);

export default defineVisual('subtree-of-another-tree', draft, pendingReview(draft.objective));

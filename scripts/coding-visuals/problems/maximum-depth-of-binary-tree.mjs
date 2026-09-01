import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A node returns one plus the larger depth returned by its children.', [
    frame('Solve the leaves', 'Every leaf returns depth 1.', tree([['3'], ['9', '20'], ['-', '-', '15', '7']], [mark(1, '1', 'state'), mark(5, '1', 'state'), mark(6, '1', 'state')])),
    frame('Combine child depths', 'Node 20 receives 1 and 1, so its depth is 2.', tree([['3'], ['9', '20'], ['-', '-', '15', '7']], [mark(2, 'depth 2', 'focus')])),
    frame('Return to the root', 'Root 3 returns 1 + max(1,2) = 3.', tree([['3'], ['9', '20'], ['-', '-', '15', '7']], [mark(0, 'depth 3', 'output')], { result: '3' })),
  ]);

export default defineVisual('maximum-depth-of-binary-tree', draft, pendingReview(draft.objective));

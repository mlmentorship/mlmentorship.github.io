import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Preorder gives the next root; inorder splits the left and right ranges.', [
    frame('Choose the root', 'Preorder starts with 3. Inorder places 3 between 9 and 15,20,7.', table(['preorder next', 'inorder left', 'root', 'inorder right'], [['3', '[9]', '3', '[15,20,7]']], [2])),
    frame('Recurse on ranges', 'The next preorder values become roots of the left and right ranges.', tree([['3'], ['9', '20'], ['-', '-', '15', '7']], [mark(1, 'left range', 'state'), mark(2, 'right range', 'focus')])),
    frame('Return the tree', 'Every inorder range is reconstructed with one preorder root.', tree([['3'], ['9', '20'], ['-', '-', '15', '7']], [mark(0, 'root', 'output')], { result: 'tree rebuilt' })),
  ]);

export default defineVisual('construct-tree-from-preorder-and-inorder-traversal', draft, pendingReview(draft.objective));

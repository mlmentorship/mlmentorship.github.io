import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('BST ordering tells whether both targets lie left, right, or split at the current node.', [
    frame('Start at 6', 'Targets 2 and 8 lie on opposite sides of 6.', tree([['6'], ['2', '8']], [mark(0, 'split', 'focus')])),
    frame('Stop at the split', 'If both targets were left or right, descend; here 6 is the first split.', tree([['6'], ['2', '8']], [mark(0, 'ancestor', 'output')], { path: '2 < 6 < 8' })),
    frame('Return the ancestor', 'Node 6 is the lowest node whose subtree contains both targets.', tree([['6'], ['2', '8']], [mark(0, 'LCA', 'output')], { result: '6' })),
  ]);

export default defineVisual('lowest-common-ancestor-in-a-bst', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Swap the two child links at every node.', [
    frame('Original children', 'Node 2 points left to 1 and right to 3.', tree([['2'], ['1', '3']], [mark(0, 'current', 'focus')])),
    frame('Swap at the root', 'The root now points left to 3 and right to 1.', tree([['2'], ['3', '1']], [mark(0, 'swapped', 'focus')])),
    frame('Return the inverted tree', 'The same swap happens recursively below every node.', tree([['2'], ['3', '1']], [mark(0, 'done', 'output')], { result: 'inverted' })),
  ]);

export default defineVisual('invert-binary-tree', draft, pendingReview(draft.objective));

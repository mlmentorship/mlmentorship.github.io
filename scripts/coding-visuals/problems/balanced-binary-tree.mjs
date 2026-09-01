import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Return a failure sentinel as soon as child heights differ by more than one.', [
    frame('Compute child heights', 'A chain gives the left subtree height 2 and the right height 0.', tree([['1'], ['2', '-'], ['3', '-']], [mark(0, 'check', 'focus')], { detail: 'left=2, right=0' })),
    frame('Propagate failure', 'The difference is 2, so this subtree returns -1.', tree([['1'], ['2', '-'], ['3', '-']], [mark(0, 'unbalanced', 'focus')], { detail: 'return -1' })),
    frame('Answer false', 'The root sees the sentinel and stops.', tree([['1'], ['2', '-'], ['3', '-']], [mark(0, 'false', 'output')], { result: 'false' })),
  ]);

export default defineVisual('balanced-binary-tree', draft, pendingReview(draft.objective));

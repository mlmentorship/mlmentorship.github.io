import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Each attention row can read its own position and every earlier position, never a future one.', [
    frame('Build all pair scores', 'Query-key scores start as a full square matrix.', attention([['.', '.', '.'], ['.', '.', '.'], ['.', '.', '.']], { action: 'QK^T' })),
    frame('Apply the causal mask', 'Future positions become forbidden before softmax.', attention([['read', 'mask', 'mask'], ['read', 'read', 'mask'], ['read', 'read', 'read']], { action: 'mask future scores' })),
    frame('Mix allowed values', 'Each row can assign weights to its prefix, while every future weight is zero.', attention([['w0', 'mask', 'mask'], ['w0', 'w1', 'mask'], ['w0', 'w1', 'w2']], { result: 'prefix-only reads; each row sums to 1' })),
  ]);

export default defineVisual('causal-attention', draft, pendingReview(draft.objective));

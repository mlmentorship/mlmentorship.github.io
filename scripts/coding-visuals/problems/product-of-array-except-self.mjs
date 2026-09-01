import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Build each answer from the product to its left and the product to its right.', [
    frame('Write left products', 'At each index, store the product strictly before it.', prefix(['1', '2', '3', '4'], { left: ['1', '1', '2', '6'], right: ['-', '-', '-', '-'], answer: ['1', '1', '2', '6'], active: 0 })),
    frame('Walk back from the right', 'A suffix product is multiplied into each saved prefix product.', prefix(['1', '2', '3', '4'], { left: ['1', '1', '2', '6'], right: ['24', '12', '4', '1'], answer: ['24', '12', '8', '6'], active: 2 })),
    frame('Exclude the current value', 'Each answer combines everything on both sides and never divides.', prefix(['1', '2', '3', '4'], { left: ['1', '1', '2', '6'], right: ['24', '12', '4', '1'], answer: ['24', '12', '8', '6'], active: 3, status: 'complete', result: '[24,12,8,6]' })),
  ]);

export default defineVisual('product-of-array-except-self', draft, pendingReview(draft.objective));

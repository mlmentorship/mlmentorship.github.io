import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('The operation x & (x-1) removes exactly the lowest set bit.', [
    frame('Read the bits', '11 is binary 1011 and has three set bits.', bits(['1', '0', '1', '1'], [mark(3, 'lowest 1', 'focus')])),
    frame('Clear one bit', '1011 becomes 1010; two more applications produce 1000 and then 0000.', bits(['1', '0', '1', '0'], [mark(3, 'cleared', 'state')], { action: 'count = 1; next 1000 -> 0000' })),
    frame('Stop at zero', 'Three bit removals means Hamming weight 3.', bits(['0', '0', '0', '0'], [], { result: '3' })),
  ]);

export default defineVisual('number-of-1-bits', draft, pendingReview(draft.objective));

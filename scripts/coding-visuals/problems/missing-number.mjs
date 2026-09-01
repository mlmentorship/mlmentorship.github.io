import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('XOR cancels every value that appears in both the expected and actual sets.', [
    frame('Pair expected and actual', 'Expected values are 0,1,2,3; actual values are 3,0,1.', table(['expected', 'actual', 'xor'], [['0', '3', '0 xor 3'], ['1', '0', '1 xor 0'], ['2', '-', '2 remains'], ['3', '1', '3 xor 1']], [6])),
    frame('Cancel matches', '0, 1, and 3 cancel in pairs; only 2 remains.', bits(['0', '0', '1', '0'], [mark(2, 'uncancelled 2', 'focus')], { detail: 'XOR result = 2' })),
    frame('Return the survivor', 'The missing value is 2.', bits(['0', '0', '1', '0'], [mark(2, 'missing', 'output')], { result: '2' })),
  ]);

export default defineVisual('missing-number', draft, pendingReview(draft.objective));

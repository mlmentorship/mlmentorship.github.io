import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('XOR supplies sum bits without carry; AND shifted left supplies the carry.', [
    frame('Separate sum and carry', 'For 3 (0011) and 1 (0001), XOR gives 0010 and the carry is 0010.', bits(['0', '0', '1', '0'], [mark(2, 'xor', 'state'), mark(3, 'xor', 'state')], { sum: '0010', carry: '0010' })),
    frame('Move the carry left', 'The next pass combines 0010 with 0010, producing no sum bits and carry 0100.', bits(['0', '0', '0', '0'], [mark(2, 'carry', 'focus')], { sum: '0000', carry: '0100' })),
    frame('Stop when carry is zero', 'A final pass produces 0100, the sum of 3 and 1.', bits(['0', '1', '0', '0'], [mark(1, '4', 'output')], { result: '4' })),
  ]);

export default defineVisual('sum-of-two-integers', draft, pendingReview(draft.objective));

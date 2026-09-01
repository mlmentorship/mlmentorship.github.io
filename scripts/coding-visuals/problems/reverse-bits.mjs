import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Read one input bit from the right and append it to the answer on the left.', [
    frame('Read the low bit', 'The input cursor starts at the least-significant bit. The drawing shows an 8-bit slice; the implementation repeats the same move 32 times.', bits(['1', '0', '1', '1', '0', '0', '1', '0'], [mark(0, 'read', 'focus')], { input: 'right -> left', output: 'empty', width: '8-bit illustration' })),
    frame('Append to output', 'Shift the output left and place the read bit at its low end.', bits(['1', '0', '1', '1', '0', '0', '1', '0'], [mark(0, 'read', 'state'), mark(7, 'write', 'focus')], { output: '1' })),
    frame('Repeat 32 times', 'After fixed-width processing, the bit order is reversed.', bits(['0', '1', '0', '0', '1', '1', '0', '1'], [], { result: 'reversed 32-bit word' })),
  ]);

export default defineVisual('reverse-bits', draft, pendingReview(draft.objective));

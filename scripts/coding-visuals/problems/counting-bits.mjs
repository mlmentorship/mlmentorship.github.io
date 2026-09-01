import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Remove the lowest bit and reuse the answer for the shifted number.', [
    frame('Use a smaller number', 'For 6, shift right to 3 and inspect the low bit 0.', table(['value', 'value >> 1', 'value & 1', 'count'], [['6', '3', '0', '?'], ['3', '1', '1', '2']], [0])),
    frame('Apply the recurrence', 'count[6] = count[3] + 0 = 2.', table(['value', 'shifted count', 'last bit', 'answer'], [['6', '2', '0', '2'], ['5', '2', '1', '2']], [0], { action: 'reuse DP' })),
    frame('Fill the line', 'Every value reuses a previously solved value.', array(['0', '1', '1', '2', '1', '2'], [mark(5, 'count(5)=2', 'output')], { result: 'counts 0..5' })),
  ]);

export default defineVisual('counting-bits', draft, pendingReview(draft.objective));

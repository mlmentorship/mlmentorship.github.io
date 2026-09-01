import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Compare each fixed-width window with the pattern counts.', [
    frame('Build the pattern count', 'The pattern ab needs one a and one b. The first text window ei has neither.', table(['window', 'a', 'b'], [['ab', '1', '1'], ['ei', '0', '0']], [0, 1, 2])),
    frame('Slide to a candidate', 'The window ba has the same counts as ab, even though the order differs.', table(['window', 'a', 'b'], [['ab', '1', '1'], ['ba', '1', '1']], [3, 4, 5], { status: 'match' })),
    frame('Return true', 'A matching count window means a permutation appears in the text.', array(['e', 'i', 'd', 'b', 'a', 'o', 'o', 'o'], [mark(3, 'window', 'output'), mark(4, 'window', 'output')], { result: 'true' })),
  ]);

export default defineVisual('permutation-in-string', draft, pendingReview(draft.objective));

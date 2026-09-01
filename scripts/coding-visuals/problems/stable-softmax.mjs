import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Subtract the row maximum before exponentiating; relative gaps do not change.', [
    frame('See the large logits', 'Exponentiating 1000 and 1001 directly can overflow.', array(['1000', '1001'], [mark(1, 'row max', 'focus')], { detail: 'raw logits' })),
    frame('Shift the row', 'Subtract 1001 to get [-1,0].', array(['-1', '0'], [mark(0, 'shifted', 'state'), mark(1, 'shifted', 'state')], { action: 'logits - max' })),
    frame('Normalize safely', 'exp(-1) and exp(0) divide by their sum to form probabilities.', array(['0.2689', '0.7311'], [mark(0, 'p', 'output'), mark(1, 'p', 'output')], { result: 'sum = 1' })),
  ]);

export default defineVisual('stable-softmax', draft, pendingReview(draft.objective));

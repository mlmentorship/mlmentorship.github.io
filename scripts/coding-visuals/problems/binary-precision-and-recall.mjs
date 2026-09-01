import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Each example enters one confusion cell; the metric chooses its denominator afterward.', [
    frame('Classify observations', 'Truth and prediction route examples to TN, FP, FN, or TP.', table(['', 'pred 0', 'pred 1'], [['true 0', 'TN', 'FP'], ['true 1', 'FN', 'TP']], [1, 2, 4, 5])),
    frame('Count the cells', 'For the example, TP=1, FP=1, FN=1, TN=1.', table(['', 'pred 0', 'pred 1'], [['true 0', '1 TN', '1 FP'], ['true 1', '1 FN', '1 TP']], [1, 2, 4, 5], { counts: 'TP=1 FP=1 TN=1 FN=1' })),
    frame('Choose the denominator', 'Precision uses predicted positives; recall uses actual positives.', table(['metric', 'numerator', 'denominator'], [['precision', 'TP=1', 'TP+FP=2'], ['recall', 'TP=1', 'TP+FN=2']], [1, 2, 4, 5], { result: 'precision=.5, recall=.5' })),
  ]);

export default defineVisual('binary-precision-and-recall', draft, pendingReview(draft.objective));

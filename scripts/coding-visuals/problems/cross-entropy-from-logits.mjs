import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Cross-entropy from logits is a stable log-sum-exp minus the selected logit.', [
    frame('Choose the correct class', 'For logits [2,1,0], label 0 selects logit 2.', array(['class 0: 2', 'class 1: 1', 'class 2: 0'], [mark(0, 'correct', 'focus')])),
    frame('Compute the normalizer', 'logsumexp summarizes all class logits without building probabilities first.', array(['2', '1', '0'], [mark(0, 'selected', 'state')], { formula: 'log(exp(2)+exp(1)+exp(0))' })),
    frame('Subtract the correct logit', 'Loss = logsumexp(row) - 2 = 0.4076.', array(['logsumexp(row)', '-', 'correct logit 2'], [mark(0, 'normalizer', 'state'), mark(2, 'subtract', 'focus')], { result: '0.4076' })),
  ]);

export default defineVisual('cross-entropy-from-logits', draft, pendingReview(draft.objective));

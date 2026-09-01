import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Use frequency as a bucket coordinate, then scan from the highest bucket.', [
    frame('Count values', 'The counts are 1 -> 3, 2 -> 2, and 3 -> 1.', buckets([{ count: '3', items: ['1'], tone: 'focus' }, { count: '2', items: ['2'] }, { count: '1', items: ['3'] }])),
    frame('Scan high to low', 'Take 1 from bucket 3 and 2 from bucket 2.', buckets([{ count: '3', items: ['1'], tone: 'output' }, { count: '2', items: ['2'], tone: 'output' }, { count: '1', items: ['3'] }], { result: 'two values collected' })),
    frame('Return top k', 'The answer is [1,2]; no global sort is needed.', buckets([{ count: '3', items: ['1'], tone: 'output' }, { count: '2', items: ['2'], tone: 'output' }, { count: '1', items: ['3'] }], { result: '[1, 2]' })),
  ]);

export default defineVisual('top-k-frequent-elements', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Copy intervals before the new range, merge overlaps, then copy the suffix.', [
    frame('Copy the prefix', 'With new interval [4,8], [1,2] ends before it and stays untouched.', intervals([{ label: '[1,2]', start: 1, end: 2, tone: 'state' }, { label: '[3,5]', start: 3, end: 5, tone: 'state' }, { label: 'new [4,8]', start: 4, end: 8, tone: 'focus' }, { label: '[6,9]', start: 6, end: 9 }], { max: 10 })),
    frame('Merge the overlap', '[4,8] overlaps [3,5] and [6,9], producing [3,9].', intervals([{ label: '[1,2]', start: 1, end: 2, tone: 'state' }, { label: '[3,9]', start: 3, end: 9, tone: 'output' }], { max: 10 })),
    frame('Return the ordered result', 'The final answer keeps the prefix and the merged range.', intervals([{ label: '[1,2]', start: 1, end: 2, tone: 'output' }, { label: '[3,9]', start: 3, end: 9, tone: 'output' }], { max: 10, result: '[[1,2],[3,9]]' })),
  ]);

export default defineVisual('insert-interval', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Sort by start and extend the last merged interval whenever ranges overlap.', [
    frame('Start with the first range', 'The merged output begins with [1,3].', intervals([{ label: '[1,3]', start: 1, end: 3, tone: 'focus' }, { label: '[2,6]', start: 2, end: 6 }, { label: '[8,10]', start: 8, end: 10 }], { max: 10 })),
    frame('Extend on overlap', 'Since 2 <= 3, merge [1,3] and [2,6] into [1,6].', intervals([{ label: '[1,6]', start: 1, end: 6, tone: 'output' }, { label: '[8,10]', start: 8, end: 10 }], { max: 10 })),
    frame('Start a new range', 'The next interval starts after 6, so it stays separate.', intervals([{ label: '[1,6]', start: 1, end: 6, tone: 'output' }, { label: '[8,10]', start: 8, end: 10, tone: 'output' }], { max: 10, result: '[[1,6],[8,10]]' })),
  ]);

export default defineVisual('merge-intervals', draft, pendingReview(draft.objective));

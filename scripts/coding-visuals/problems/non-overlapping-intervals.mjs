import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('When intervals overlap, keep the one with the earlier end.', [
    frame('Sort by end', 'The candidate ending at 2 leaves the most room for later intervals.', intervals([{ label: '[1,2]', start: 1, end: 2, tone: 'focus' }, { label: '[1,3]', start: 1, end: 3 }, { label: '[2,3]', start: 2, end: 3 }, { label: '[3,4]', start: 3, end: 4 }], { max: 4 })),
    frame('Reject the late-ending overlap', '[1,3] overlaps the kept [1,2], so remove it and keep checking the remaining ranges.', intervals([{ label: '[1,2]', start: 1, end: 2, tone: 'state' }, { label: '[1,3]', start: 1, end: 3, tone: 'warning' }, { label: '[2,3]', start: 2, end: 3, tone: 'focus' }, { label: '[3,4]', start: 3, end: 4 }], { max: 4, detail: 'remove 1' })),
    frame('Keep room for the future', '[2,3] or [3,4] can follow the earliest-ending choice.', intervals([{ label: '[1,2]', start: 1, end: 2, tone: 'output' }, { label: '[2,3]', start: 2, end: 3, tone: 'output' }, { label: '[3,4]', start: 3, end: 4, tone: 'output' }], { max: 4, result: 'remove 1 interval' })),
  ]);

export default defineVisual('non-overlapping-intervals', draft, pendingReview(draft.objective));

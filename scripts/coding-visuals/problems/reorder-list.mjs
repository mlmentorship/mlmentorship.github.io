import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Find the middle, reverse the second half, then interleave the two lists.', [
    frame('Split at the middle', 'Slow and fast leave first half 1,2,3 and second half 4,5.', linked([{ value: '1' }, { value: '2' }, { value: '3', pointer: 'split' }, { value: '4' }, { value: '5' }], { detail: 'first: 1->2->3; second: 4->5' })),
    frame('Reverse the second half', 'The second list becomes 5->4.', linked([{ value: '1' }, { value: '2' }, { value: '3' }, { value: '5', tone: 'focus' }, { value: '4' }], { detail: 'second: 5->4' })),
    frame('Interleave', 'Take one node from each half: 1,5,2,4,3.', linked([{ value: '1', tone: 'output' }, { value: '5', tone: 'output' }, { value: '2', tone: 'output' }, { value: '4', tone: 'output' }, { value: '3', tone: 'output' }], { result: '1->5->2->4->3' })),
  ]);

export default defineVisual('reorder-list', draft, pendingReview(draft.objective));

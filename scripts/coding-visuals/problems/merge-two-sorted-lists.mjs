import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Attach the smaller current head and advance only that list.', [
    frame('Compare two heads', 'Heads 1 and 1 tie; attach one and advance its list.', linked([{ value: 'A:1', pointer: 'head A' }, { value: 'A:2' }, { value: 'A:4' }], { second: ['B:1', 'B:3', 'B:4'], detail: 'take 1' })),
    frame('Continue the merge', 'Compare the next heads and attach 2, then 3.', linked([{ value: '1' }, { value: '1' }, { value: '2', tone: 'focus' }, { value: '3', tone: 'focus' }], { detail: 'tail always points at last result node' })),
    frame('Append the remainder', 'When one list ends, attach the other suffix unchanged.', linked([{ value: '1' }, { value: '1' }, { value: '2' }, { value: '3' }, { value: '4' }, { value: '4' }], { result: 'sorted merged list' })),
  ]);

export default defineVisual('merge-two-sorted-lists', draft, pendingReview(draft.objective));

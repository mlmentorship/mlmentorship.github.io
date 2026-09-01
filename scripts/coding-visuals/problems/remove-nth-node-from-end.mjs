import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A fixed pointer gap makes the left pointer stop just before the node to remove.', [
    frame('Create a gap', 'Move right two nodes ahead of left for n=2.', linked([{ value: 'dummy', pointer: 'left' }, { value: '1' }, { value: '2', pointer: 'right' }, { value: '3' }, { value: '4' }, { value: '5' }], { arrows: ['dummy -> 1 -> 2 -> 3 -> 4 -> 5'], detail: 'gap = 2' })),
    frame('Walk together', 'When right reaches 5, left is at node 3.', linked([{ value: '3', pointer: 'left' }, { value: '4' }, { value: '5', pointer: 'right' }], { arrows: ['3 -> 4 -> 5'], detail: 'left.next is node 4' })),
    frame('Skip the target', 'Redirect 3.next around node 4.', linked([{ value: '1' }, { value: '2' }, { value: '3', pointer: 'link changed', tone: 'focus' }, { value: '5', tone: 'output' }], { arrows: ['1 -> 2 -> 3 -> 5'], result: '[1,2,3,5]' })),
  ]);

export default defineVisual('remove-nth-node-from-end', draft, pendingReview(draft.objective));

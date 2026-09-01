import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Save the outgoing link, reverse the current link, then advance.', [
    frame('Save next', 'Before changing 1.next, save the route to node 2.', linked([{ value: '1', pointer: 'current' }, { value: '2', pointer: 'next' }, { value: '3' }], { arrows: ['1 -> 2', '2 -> 3'] })),
    frame('Reverse one link', 'Point 1 back to previous, then advance current to saved node 2.', linked([{ value: '1', pointer: 'previous' }, { value: '2', pointer: 'current' }, { value: '3', pointer: 'next' }], { arrows: ['2 -> 3', '1 -> null'] })),
    frame('Return new head', 'After all links reverse, previous points at 3.', linked([{ value: '3', pointer: 'head', tone: 'output' }, { value: '2' }, { value: '1' }], { arrows: ['3 -> 2', '2 -> 1'], result: '3 -> 2 -> 1' })),
  ]);

export default defineVisual('reverse-linked-list', draft, pendingReview(draft.objective));

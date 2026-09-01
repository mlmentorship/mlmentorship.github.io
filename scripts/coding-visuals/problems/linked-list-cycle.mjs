import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A one-step pointer and a two-step pointer must meet inside a cycle.', [
    frame('Move at different speeds', 'After one move, slow is at 2 and fast is at 3.', linked([{ value: '1' }, { value: '2', pointer: 'slow' }, { value: '3', pointer: 'fast' }, { value: '4' }], { arrows: ['1 -> 2', '2 -> 3', '3 -> 4', '4 -> 2'] })),
    frame('Enter the loop', 'After the next move, slow is at 3 and fast has wrapped to 2.', linked([{ value: '2', pointer: 'fast' }, { value: '3', pointer: 'slow' }, { value: '4' }], { arrows: ['2 -> 3', '3 -> 4', '4 -> 2'] })),
    frame('Meet', 'On the next move both pointers reach 4, proving a cycle exists.', linked([{ value: '4', pointer: 'slow + fast', tone: 'output' }, { value: '2' }, { value: '3' }], { arrows: ['4 -> 2', '2 -> 3', '3 -> 4'], result: 'true' })),
  ]);

export default defineVisual('linked-list-cycle', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('At each start, remove rooms whose meetings have already ended.', [
    frame('First meeting', 'Meeting [0,30] occupies one room.', intervals([{ label: '[0,30]', start: 0, end: 30, tone: 'focus' }], { max: 30, rooms: '1 active room' })),
    frame('Overlap needs another room', 'At start 5, [0,30] is still active, so [5,10] uses room 2.', intervals([{ label: '[0,30]', start: 0, end: 30, tone: 'state' }, { label: '[5,10]', start: 5, end: 10, tone: 'focus' }], { max: 30, rooms: '2 active rooms' })),
    frame('Reuse after an end', 'At start 15, [5,10] is gone; the maximum active count was 2.', intervals([{ label: '[0,30]', start: 0, end: 30, tone: 'output' }, { label: '[15,20]', start: 15, end: 20, tone: 'output' }], { max: 30, result: '2 rooms' })),
  ]);

export default defineVisual('meeting-rooms-ii', draft, pendingReview(draft.objective));

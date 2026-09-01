import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('After sorting by start time, only the previous end can overlap the next meeting.', [
    frame('Sort meetings', 'The starts are 0, 5, and 15.', intervals([{ label: '[0,30]', start: 0, end: 30, tone: 'state' }, { label: '[5,10]', start: 5, end: 10, tone: 'focus' }, { label: '[15,20]', start: 15, end: 20 }], { max: 30 })),
    frame('Find the overlap', 'The next start 5 is before previous end 30.', intervals([{ label: '[0,30]', start: 0, end: 30, tone: 'warning' }, { label: '[5,10]', start: 5, end: 10, tone: 'focus' }], { max: 30, detail: '5 < 30' })),
    frame('Return false', 'One person cannot attend overlapping meetings.', intervals([{ label: '[0,30]', start: 0, end: 30, tone: 'warning' }, { label: '[5,10]', start: 5, end: 10, tone: 'warning' }], { max: 30, result: 'false' })),
  ]);

export default defineVisual('meeting-rooms', draft, pendingReview(draft.objective));

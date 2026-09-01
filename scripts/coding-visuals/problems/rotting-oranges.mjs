import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Multi-source BFS makes each queue layer one minute of spread.', [
    frame('Seed all sources', 'Every rotten orange starts in minute 0.', queueGrid([['2', '1', '1'], ['1', '1', '0'], ['0', '1', '1']], ['(0,0)'], { minute: '0' })),
    frame('Spread one layer', 'The minute-1 frontier reaches its fresh neighbors.', queueGrid([['2', '2', '1'], ['2', '1', '0'], ['0', '1', '1']], ['(0,1)', '(1,0)'], { minute: '1' })),
    frame('Finish at the last layer', 'The final reachable orange rots at minute 4.', queueGrid([['2', '2', '2'], ['2', '2', '0'], ['0', '2', '2']], [], { minute: '4', result: '4' })),
  ]);

export default defineVisual('rotting-oranges', draft, pendingReview(draft.objective));

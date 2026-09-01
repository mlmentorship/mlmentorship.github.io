import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Memoize the best increasing path starting at each cell; larger-only moves cannot cycle.', [
    frame('Find increasing neighbors', 'From 1, move to 2, then 6, then 9.', grid([['9', '9', '4'], ['6', '6', '8'], ['2', '1', '1']], [{ row: 2, col: 1, label: '1', tone: 'focus' }, { row: 2, col: 0, label: '2', tone: 'state' }])),
    frame('Cache a cell answer', 'The memo table stores the best path length from every cell; the path from 1 has length 4.', grid([['1', '1', '3'], ['2', '2', '2'], ['3', '4', '3']], [{ row: 2, col: 1, label: 'path length 4', tone: 'output' }], { action: 'memoize' })),
    frame('Take the maximum cached value', 'Every cell is solved once; the largest cached path is 4.', grid([['1', '1', '3'], ['2', '2', '2'], ['3', '4', '3']], [{ row: 2, col: 1, label: 'max 4', tone: 'output' }], { result: '4' })),
  ]);

export default defineVisual('longest-increasing-path-in-a-matrix', draft, pendingReview(draft.objective));

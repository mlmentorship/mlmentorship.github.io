import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Each cell receives paths from the cell above and the cell to its left.', [
    frame('Initialize the top edge', 'Only one path reaches every cell along the top row.', grid([['1', '1', '1', '1', '1', '1', '1'], ['1', '.', '.', '.', '.', '.', '.'], ['1', '.', '.', '.', '.', '.', '.']], [], { action: 'base paths' })),
    frame('Add from two directions', 'The center cell gets paths from above plus paths from the left.', grid([['1', '1', '1', '1', '1', '1', '1'], ['1', '2', '3', '4', '5', '6', '7'], ['1', '3', '6', '10', '15', '21', '28']], [{ row: 2, col: 2, label: '6', tone: 'focus' }], { formula: '3 + 3 = 6' })),
    frame('Read the destination', 'The bottom-right cell contains 28 paths for a 3 by 7 grid.', grid([['1', '1', '1', '1', '1', '1', '1'], ['1', '2', '3', '4', '5', '6', '7'], ['1', '3', '6', '10', '15', '21', '28']], [{ row: 2, col: 6, label: '28', tone: 'output' }], { result: '28' })),
  ]);

export default defineVisual('unique-paths', draft, pendingReview(draft.objective));

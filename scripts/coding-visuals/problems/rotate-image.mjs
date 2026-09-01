import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Reverse the row order, then transpose across the main diagonal.', [
    frame('Reverse rows', 'The bottom row moves to the top.', grid([['7', '8', '9'], ['4', '5', '6'], ['1', '2', '3']], [], { action: 'reverse rows' })),
    frame('Transpose', 'Swap cells across the diagonal: (row,col) becomes (col,row).', grid([['7', '4', '1'], ['8', '5', '2'], ['9', '6', '3']], [{ row: 0, col: 0, label: 'fixed', tone: 'state' }, { row: 0, col: 2, label: 'moved', tone: 'focus' }])),
    frame('Read clockwise result', 'The matrix is rotated in place without a second matrix.', grid([['7', '4', '1'], ['8', '5', '2'], ['9', '6', '3']], [], { result: '90 degrees clockwise' })),
  ]);

export default defineVisual('rotate-image', draft, pendingReview(draft.objective));

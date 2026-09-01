import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Padding creates a rectangle; the boolean mask preserves which cells were real.', [
    frame('Start with ragged rows', 'The sequences have lengths 2 and 1.', shapes(['[3,4]', '[9]'], { action: 'ragged input' })),
    frame('Fill the rectangle', 'Use the longest width and a pad value for unused cells.', grid([['3', '4'], ['9', '0']], [{ row: 1, col: 1, label: 'pad', tone: 'state' }], { tensor: 'tokens [2,2]' })),
    frame('Carry the mask', 'The same padded position is false in the validity mask.', grid([['1', '1'], ['1', '0']], [{ row: 1, col: 1, label: 'false', tone: 'output' }], { tensor: 'mask [2,2]', result: 'tokens + mask' })),
  ]);

export default defineVisual('pad-variable-length-sequences', draft, pendingReview(draft.objective));

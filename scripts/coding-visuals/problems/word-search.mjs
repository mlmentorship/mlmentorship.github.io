import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Mark a board cell while it belongs to the current path, then restore it on return.', [
    frame('Start at A', 'The first matching cell starts the path A.', grid([['A', 'B', 'C', 'E'], ['S', 'F', 'C', 'S'], ['A', 'D', 'E', 'E']], [{ row: 0, col: 0, label: 'A', tone: 'focus' }], { word: 'A B C C E D' })),
    frame('Extend the path', 'Move through adjacent B, C, and C cells while marking them used.', grid([['#', '#', '#', 'E'], ['S', 'F', '#', 'S'], ['A', 'D', 'E', 'E']], [{ row: 0, col: 0, label: 'A', tone: 'state' }, { row: 0, col: 1, label: 'B', tone: 'state' }, { row: 0, col: 2, label: 'C', tone: 'state' }, { row: 1, col: 2, label: 'C', tone: 'focus' }], { word: 'A -> B -> C -> C' })),
    frame('Reach the final D', 'Continue to E and D; restore cells when a branch fails.', grid([['#', '#', '#', 'E'], ['S', 'F', '#', 'S'], ['A', 'D', '#', 'E']], [{ row: 2, col: 1, label: 'D', tone: 'output' }], { result: 'ABCCED found' })),
  ]);

export default defineVisual('word-search', draft, pendingReview(draft.objective));

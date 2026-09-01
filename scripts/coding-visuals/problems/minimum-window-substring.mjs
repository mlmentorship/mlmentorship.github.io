import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Grow until all required characters are present, then shrink while the window remains valid.', [
    frame('Gather ABC', 'ADOBEC contains A, B, and C, so the first valid window ends at C.', array(['A', 'D', 'O', 'B', 'E', 'C', 'O', 'D', 'E', 'B', 'A', 'N', 'C'], [mark(0, 'L'), mark(5, 'R', 'focus')], { range: 'ADOBEC', need: 'A,B,C' })),
    frame('Shrink from the left', 'Dropping A breaks validity, so grow again until the new valid window is BANC.', array(['A', 'D', 'O', 'B', 'E', 'C', 'O', 'D', 'E', 'B', 'A', 'N', 'C'], [mark(5, 'old valid', 'state'), mark(9, 'L', 'focus'), mark(12, 'R', 'focus')], { range: 'BANC', action: 'shrink then regrow' })),
    frame('Keep the shortest', 'BANC is the shortest window containing A, B, and C.', array(['A', 'D', 'O', 'B', 'E', 'C', 'O', 'D', 'E', 'B', 'A', 'N', 'C'], [mark(9, 'B', 'output'), mark(10, 'A', 'output'), mark(11, 'N', 'output'), mark(12, 'C', 'output')], { result: 'BANC' })),
  ]);

export default defineVisual('minimum-window-substring', draft, pendingReview(draft.objective));

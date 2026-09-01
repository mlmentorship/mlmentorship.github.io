import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A window is valid when every non-majority character fits inside the replacement budget.', [
    frame('Count the window', 'AAB has majority count 2. One B needs one replacement.', array(['A', 'A', 'B', 'A', 'B', 'B', 'A'], [mark(0, 'L'), mark(2, 'R', 'focus')], { range: 'AAB', formula: '3 - max_count 2 = 1' })),
    frame('Keep a valid length-4 window', 'AABA uses one replacement: length 4 minus majority count 3 equals 1.', array(['A', 'A', 'B', 'A', 'B', 'B', 'A'], [mark(0, 'L', 'focus'), mark(3, 'R', 'focus')], { range: 'AABA', formula: '4 - max_count 3 = 1' })),
    frame('Return the longest length', 'The scan may later see ABAB, but the valid window AABA already proves length 4.', array(['A', 'A', 'B', 'A', 'B', 'B', 'A'], [mark(0, 'best', 'output'), mark(3, 'best', 'output')], { range: 'AABA', result: '4' })),
  ]);

export default defineVisual('longest-repeating-character-replacement', draft, pendingReview(draft.objective));

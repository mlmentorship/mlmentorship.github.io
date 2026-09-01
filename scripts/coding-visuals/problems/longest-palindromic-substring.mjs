import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Every palindrome grows from one character center or one gap center.', [
    frame('Try an odd center', 'Expand around b in babad to get bab.', array(['b', 'a', 'b', 'a', 'd'], [mark(1, 'center', 'focus'), mark(0, 'edge', 'state'), mark(2, 'edge', 'state')], { candidate: 'bab' })),
    frame('Try an even center', 'A gap between two equal characters handles even-length palindromes.', array(['c', 'b', 'b', 'd'], [mark(1, 'gap', 'focus'), mark(2, 'gap', 'focus')], { candidate: 'bb' })),
    frame('Keep the longest', 'The widest expansion wins.', array(['b', 'a', 'b', 'a', 'd'], [mark(0, 'best', 'output'), mark(1, 'best', 'output'), mark(2, 'best', 'output')], { result: 'bab or aba' })),
  ]);

export default defineVisual('longest-palindromic-substring', draft, pendingReview(draft.objective));

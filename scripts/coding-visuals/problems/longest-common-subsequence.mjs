import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A matching pair advances both prefixes; a mismatch keeps the better skipped prefix.', [
    frame('Compare prefixes', 'The grid state answers LCS for prefixes of abcde and ace.', table(['', '0', 'a', 'c', 'e'], [['0', '0', '0', '0', '0'], ['a', '0', '1', '1', '1'], ['b', '0', '1', '1', '1'], ['c', '0', '1', '2', '2'], ['d', '0', '1', '2', '2'], ['e', '0', '1', '2', '3']], [1, 1])),
    frame('Match c', 'The c/c cell takes the diagonal answer and adds one.', table(['', '0', 'a', 'c', 'e'], [['0', '0', '0', '0', '0'], ['a', '0', '1', '1', '1'], ['b', '0', '1', '1', '1'], ['c', '0', '1', '2', '2'], ['d', '0', '1', '2', '2'], ['e', '0', '1', '2', '3']], [18], { action: 'diagonal + 1' })),
    frame('Read the bottom-right', 'The complete prefixes share subsequence ace of length 3.', table(['', '0', 'a', 'c', 'e'], [['0', '0', '0', '0', '0'], ['a', '0', '1', '1', '1'], ['b', '0', '1', '1', '1'], ['c', '0', '1', '2', '2'], ['d', '0', '1', '2', '2'], ['e', '0', '1', '2', '3']], [29], { result: '3' })),
  ]);

export default defineVisual('longest-common-subsequence', draft, pendingReview(draft.objective));

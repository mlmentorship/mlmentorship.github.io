import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Each mismatch chooses the cheapest of insert, delete, and replace.', [
    frame('Initialize empty prefixes', 'The first row and column count edits against an empty string; the interior is not solved yet.', table(['', '0', 'r', 'o', 's'], [['0', '0', '1', '2', '3'], ['h', '1', '?', '?', '?'], ['o', '2', '?', '?', '?'], ['r', '3', '?', '?', '?'], ['s', '4', '?', '?', '?'], ['e', '5', '?', '?', '?']], [0])),
    frame('Choose a local operation', 'At the final e/s mismatch, the cell is 1 plus the smallest neighbor.', table(['', '0', 'r', 'o', 's'], [['0', '0', '1', '2', '3'], ['h', '1', '1', '2', '3'], ['o', '2', '2', '1', '2'], ['r', '3', '2', '2', '2'], ['s', '4', '3', '3', '2'], ['e', '5', '4', '4', '3']], [29], { action: 'min(insert, delete, replace) + 1' })),
    frame('Read the final cost', 'The bottom-right cell gives the distance from horse to ros.', table(['', '0', 'r', 'o', 's'], [['0', '0', '1', '2', '3'], ['h', '1', '1', '2', '3'], ['o', '2', '2', '1', '2'], ['r', '3', '2', '2', '2'], ['s', '4', '3', '3', '2'], ['e', '5', '4', '4', '3']], [29], { result: '3' })),
  ]);

export default defineVisual('edit-distance', draft, pendingReview(draft.objective));

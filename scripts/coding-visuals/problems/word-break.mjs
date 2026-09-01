import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A reachable string position can start any dictionary word that matches there.', [
    frame('Start at position 0', 'The empty prefix is reachable before reading any word.', array(['0', '1', '2', '3', '4', '5', '6', '7', '8'], [mark(0, 'start', 'state')], { text: 'leetcode' })),
    frame('Reach position 4', 'The word leet matches positions 0..3, so position 4 becomes reachable.', array(['l', 'e', 'e', 't', '|', 'c', 'o', 'd', 'e'], [mark(4, 'reachable', 'focus')], { word: 'leet' })),
    frame('Reach the end', 'The word code starts at 4 and reaches position 8.', array(['l', 'e', 'e', 't', '|', 'c', 'o', 'd', 'e'], [mark(8, 'reachable', 'output')], { result: 'true' })),
  ]);

export default defineVisual('word-break', draft, pendingReview(draft.objective));

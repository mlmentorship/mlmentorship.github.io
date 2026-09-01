import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('For each subsequence length, keep the smallest possible ending value.', [
    frame('Read 10, 9, 2', 'Each new smaller value replaces the tail for length 1.', array(['10', '9', '2', '5', '3', '7', '101'], [mark(2, 'tails=[2]', 'state')], { tails: '[2]' })),
    frame('Extend and replace tails', '5, 3, and 7 produce tails [2,3,7].', array(['2', '3', '7'], [mark(2, 'length 3', 'focus')], { tails: '[2,3,7]' })),
    frame('Append 101', '101 extends the tail list, giving length 4.', array(['2', '3', '7', '101'], [mark(3, 'append', 'output')], { result: '4' })),
  ]);

export default defineVisual('longest-increasing-subsequence', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Choose in nondecreasing index order and carry the remaining target.', [
    frame('Start with target 7', 'The first choices are 2, 3, 6, or 7.', choices([], ['2 (remain 5)', '3 (remain 4)', '6 (remain 1)', '7 (remain 0)'], { target: '7' })),
    frame('Reuse a choice', 'From remainder 5, choosing 2 again leaves 3; [2,2,3] reaches zero.', choices(['2', '2'], ['choose 3 -> remain 0', 'choose 6 -> too large'], { target: '3' })),
    frame('Collect complete paths', 'The valid combinations are [2,2,3] and [7].', choices([], ['[2,2,3] = 7', '[7] = 7'], { result: '2 combinations' })),
  ]);

export default defineVisual('combination-sum', draft, pendingReview(draft.objective));

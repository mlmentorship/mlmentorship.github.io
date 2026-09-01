import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Every partial path is already one valid subset; branch by choosing the next index.', [
    frame('Start with the empty path', 'The empty subset is a result before any choice.', choices([], ['take 1', 'skip 1'], { input: '[1,2]' })),
    frame('Choose 1, then 2', 'The path [1] branches to [1,2] or stops at [1].', choices(['1'], ['take 2 -> [1,2]', 'skip 2 -> [1]'], { input: '[1,2]' })),
    frame('Collect every path', 'The four paths are [], [1], [2], and [1,2].', choices([], ['[]', '[1]', '[2]', '[1,2]'], { result: '4 subsets' })),
  ]);

export default defineVisual('subsets', draft, pendingReview(draft.objective));

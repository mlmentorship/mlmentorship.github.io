import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Fill one position with each unused value, then undo it for the next branch.', [
    frame('Choose the first position', 'For [1,2,3], any of the three values can start the path.', choices([], ['1__', '2__', '3__'], { used: 'none' })),
    frame('Choose below 1', 'After choosing 1, only 2 and 3 remain for the next position.', choices(['1'], ['12_', '13_'], { used: '1' })),
    frame('Reach complete leaves', 'The tree ends at all six orderings.', choices([], ['123', '132', '213', '231', '312', '321'], { result: '6 permutations' })),
  ]);

export default defineVisual('permutations', draft, pendingReview(draft.objective));

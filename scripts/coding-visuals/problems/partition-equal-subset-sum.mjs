import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Reach half the total; every reachable sum is a state.', [
    frame('Set the target', 'Total is 22, so the wanted subset sum is 11.', array(['0', '1', '5', '11', '16'], [mark(3, 'target 11', 'focus')], { target: '11' })),
    frame('Add reachable sums', 'After processing 1, 5, and 11, the set contains 11.', array(['0', '1', '5', '6', '11'], [mark(4, 'reachable', 'output')])),
    frame('Accept the partition', 'A subset totals 11, so the remaining values also total 11.', array(['[1,5,5]', '[11]'], [mark(0, 'sum 11', 'output'), mark(1, 'sum 11', 'output')], { result: 'true' })),
  ]);

export default defineVisual('partition-equal-subset-sum', draft, pendingReview(draft.objective));

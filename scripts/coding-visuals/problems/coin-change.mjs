import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('The best way to make amount t is one coin plus the best way to make t-coin.', [
    frame('Initialize amount zero', 'Zero coins make amount 0; other amounts are unreachable.', array(['0', 'inf', 'inf', 'inf', 'inf', 'inf', 'inf'], [mark(0, 'base', 'state')])),
    frame('Build amount 6', 'For coin 5, look at amount 1 and add one coin.', array(['0', '1', '1', '2', '2', '1', '2'], [mark(1, 'fewest[1]=1', 'state'), mark(6, 'fewest[6]=2', 'focus')])),
    frame('Return the minimum', 'With coins 1, 2, and 5, 6 is made by 1+5 in two coins.', array(['0', '1', '1', '2', '2', '1', '2'], [mark(6, 'answer 2', 'output')], { result: '2 coins' })),
  ]);

export default defineVisual('coin-change', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('For each selling day, pair the price with the lowest earlier buy price.', [
    frame('Track the cheapest buy', 'Prices 7 then 1 leave lowest buy price 1.', array(['7', '1', '5', '3', '6', '4'], [mark(1, 'lowest buy', 'state')], { low: '1', profit: '0' })),
    frame('Sell at 6', 'Selling at 6 after buying at 1 produces profit 5.', array(['7', '1', '5', '3', '6', '4'], [mark(1, 'buy', 'state'), mark(4, 'sell', 'focus')], { low: '1', profit: '5' })),
    frame('Return the best profit', 'No later sale beats 5.', array(['7', '1', '5', '3', '6', '4'], [mark(1, 'buy', 'output'), mark(4, 'sell', 'output')], { result: '5' })),
  ]);

export default defineVisual('best-time-to-buy-and-sell-stock', draft, pendingReview(draft.objective));

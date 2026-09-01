import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Keep both product extremes because a negative number can swap their roles.', [
    frame('Start both extremes', 'At 2 and 3, max and min ending products are both positive.', array(['2', '3', '-2', '4'], [mark(1, 'max=6,min=3', 'state')], { max: '6', min: '3' })),
    frame('A negative flips them', 'At -2, restarting at -2 is the maximum while the carried products become negative.', array(['2', '3', '-2', '4'], [mark(2, 'flip', 'focus')], { max: '-2', min: '-12', detail: 'candidates: -2, 6*-2, 3*-2' })),
    frame('Recover with another negative', 'The best product 6 comes from [2,3], while later values are checked the same way.', array(['2', '3', '-2', '4'], [mark(0, 'best', 'output'), mark(1, 'best', 'output')], { result: '6' })),
  ]);

export default defineVisual('maximum-product-subarray', draft, pendingReview(draft.objective));

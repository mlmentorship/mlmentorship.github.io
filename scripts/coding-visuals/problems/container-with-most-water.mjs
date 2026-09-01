import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('The shorter wall limits area, so move that wall inward.', [
    frame('Start at both ends', 'Width is largest, but the left wall of height 1 limits the water.', array(['1', '8', '6', '2', '5', '4', '8', '3', '7'], [mark(0, 'L', 'focus'), mark(8, 'R', 'focus')], { measure: 'area = 8' })),
    frame('Move the shorter wall', 'Moving the height-7 wall cannot improve the height-1 limit. Move L.', array(['1', '8', '6', '2', '5', '4', '8', '3', '7'], [mark(1, 'L', 'focus'), mark(8, 'R', 'focus')], { measure: 'height limit = 7' })),
    frame('Keep the best area', 'At heights 8 and 7, width 7 gives the best area 49.', array(['1', '8', '6', '2', '5', '4', '8', '3', '7'], [mark(1, 'best', 'output'), mark(8, 'best', 'output')], { measure: 'best = 49', result: '49' })),
  ]);

export default defineVisual('container-with-most-water', draft, pendingReview(draft.objective));

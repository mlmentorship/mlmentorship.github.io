import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A shorter bar ends every taller increasing-stack bar and reveals its maximal width.', [
    frame('Push increasing bars', 'Height 1 closes the earlier height 2; heights 5 and 6 then wait in the increasing stack.', array(['2', '1', '5', '6', '2', '3'], [mark(2, 'push', 'state'), mark(3, 'push', 'focus')], { stack: '[1,5,6]' })),
    frame('Short bar closes rectangles', 'Height 2 pops 6 and 5; their widths are measured to the current index.', array(['2', '1', '5', '6', '2', '3'], [mark(2, 'height 5', 'focus'), mark(3, 'height 6', 'warning'), mark(4, 'boundary', 'state')], { areas: '6*1 and 5*2' })),
    frame('Keep the largest area', 'The bars 5 and 6 form the best rectangle of area 10.', array(['2', '1', '5', '6', '2', '3'], [mark(2, 'width 2', 'output'), mark(3, 'width 2', 'output')], { result: '10' })),
  ]);

export default defineVisual('largest-rectangle-in-histogram', draft, pendingReview(draft.objective));

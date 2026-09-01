import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Keep the sorted half that can still contain the target.', [
    frame('Probe the middle', 'The middle value 5 is below target 7, so the lower half is finished.', array(['1', '3', '5', '7', '9'], [mark(2, 'mid', 'focus'), mark(0, 'discard'), mark(1, 'discard')])),
    frame('Narrow the interval', 'The remaining interval is [7, 9].', array(['1', '3', '5', '7', '9'], [mark(3, 'lo', 'state'), mark(4, 'hi', 'state')])),
    frame('Hit the target', 'The next middle is 7, at index 3.', array(['1', '3', '5', '7', '9'], [mark(3, 'found', 'output')], { result: 'index 3' })),
  ]);

export default defineVisual('binary-search', draft, pendingReview(draft.objective));

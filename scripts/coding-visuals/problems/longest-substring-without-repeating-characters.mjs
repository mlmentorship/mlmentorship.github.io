import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Keep the longest window whose characters are all distinct.', [
    frame('Grow the window', 'The first window abc contains no duplicate.', array(['a', 'b', 'c', 'a', 'b', 'c', 'b', 'b'], [mark(0, 'L'), mark(2, 'R', 'focus')], { range: 'abc', state: 'a,b,c' })),
    frame('Repair the duplicate', 'The next a repeats, so move L past the old a before continuing.', array(['a', 'b', 'c', 'a', 'b', 'c', 'b', 'b'], [mark(1, 'L', 'focus'), mark(3, 'R', 'focus')], { range: 'bca', state: 'b,c,a' })),
    frame('Save the best window', 'The longest distinct window seen has length 3.', array(['a', 'b', 'c', 'a', 'b', 'c', 'b', 'b'], [mark(1, 'best', 'output'), mark(3, 'best', 'output')], { range: 'bca', result: '3' })),
  ]);

export default defineVisual('longest-substring-without-repeating-characters', draft, pendingReview(draft.objective));

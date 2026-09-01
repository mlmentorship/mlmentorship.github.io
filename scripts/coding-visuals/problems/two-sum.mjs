import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Find the complement in the values already scanned.', [
    frame('Read the first value', '2 is current. Nothing is saved yet.', arrayMap(['2', '7', '11', '15'], [], [mark(0, 'current')])),
    frame('Ask for the complement', 'At 7, the target 9 needs 2. The map already stores 2 at index 0.', arrayMap(['2', '7', '11', '15'], [['2', 'index 0']], [mark(0, 'saved', 'state'), mark(1, 'current'), mark(1, 'need 2', 'focus')])),
    frame('Return the pair', 'The complement is present, so indices 0 and 1 finish the search.', arrayMap(['2', '7', '11', '15'], [['2', 'index 0']], [mark(0, 'pair', 'output'), mark(1, 'pair', 'output')], { result: '[0, 1]' })),
  ]);

export default defineVisual('two-sum', draft, pendingReview(draft.objective));

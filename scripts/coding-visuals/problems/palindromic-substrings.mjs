import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Each successful center expansion contributes exactly one palindrome.', [
    frame('Count an odd center', 'Center a gives a, then expand to aba.', array(['a', 'b', 'a'], [mark(1, 'center', 'focus'), mark(0, 'palindrome', 'state'), mark(2, 'palindrome', 'state')], { count: '2' })),
    frame('Count every center', 'For aaa, three single letters, two pairs, and aaa all count.', array(['a', 'a', 'a'], [mark(0, '1', 'state'), mark(1, '4', 'focus'), mark(2, '1', 'state')], { count: '6 total' })),
    frame('Return the total', 'The six palindromic substrings of aaa are all center expansions.', array(['a', 'a', 'a'], [mark(0, 'a', 'output'), mark(1, 'a/aa', 'output'), mark(2, 'a', 'output')], { result: '6' })),
  ]);

export default defineVisual('palindromic-substrings', draft, pendingReview(draft.objective));

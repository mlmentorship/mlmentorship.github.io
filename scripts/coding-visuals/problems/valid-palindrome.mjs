import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Move inward while comparing the next alphanumeric character from each end.', [
    frame('Skip punctuation', 'Ignore spaces and commas; the meaningful endpoints are A and a.', array(['A', 'm', 'a', 'n', 'a', 'm', 'a'], [mark(0, 'L', 'focus'), mark(6, 'R', 'focus')], { normalize: 'lowercase, alphanumeric' })),
    frame('Compare inward', 'Matching pairs move both pointers toward the center.', array(['A', 'm', 'a', 'n', 'a', 'm', 'a'], [mark(1, 'match', 'state'), mark(5, 'match', 'state')], { detail: 'm == m' })),
    frame('Meet in the middle', 'Every pair matches, so the normalized string is a palindrome.', array(['A', 'm', 'a', 'n', 'a', 'm', 'a'], [mark(3, 'center', 'output')], { result: 'true' })),
  ]);

export default defineVisual('valid-palindrome', draft, pendingReview(draft.objective));

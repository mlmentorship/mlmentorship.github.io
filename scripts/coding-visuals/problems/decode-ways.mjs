import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A digit can extend one prior decoding; a valid two-digit number can extend two prior decodings.', [
    frame('Read 2', 'The first digit gives one decoding.', array(['2', '2', '6'], [mark(0, '1 way', 'state')])),
    frame('At 22', '22 is valid, so one-digit and two-digit choices contribute.', array(['2', '2', '6'], [mark(1, '2 ways', 'focus')], { choices: '2|2 and 22' })),
    frame('At 226', '6 can follow 22 or stand after 2, giving three total decodings.', array(['2', '2', '6'], [mark(2, '3 ways', 'output')], { result: '3' })),
  ]);

export default defineVisual('decode-ways', draft, pendingReview(draft.objective));

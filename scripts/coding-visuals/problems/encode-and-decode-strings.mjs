import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A length prefix tells the decoder exactly how many characters belong to each string.', [
    frame('Encode with lengths', 'lint becomes 4#lint and # becomes 1##.', array(['4#lint', '1##', '0#'], [mark(0, 'length 4', 'focus')])),
    frame('Read one length', 'The decoder reads 4, skips #, and consumes exactly four characters.', array(['4', '#', 'l', 'i', 'n', 't'], [mark(0, 'read length', 'state'), mark(2, 'start', 'focus'), mark(5, 'end', 'focus')])),
    frame('Recover the list', 'Lengths make delimiters inside the original strings harmless.', array(['lint', '#', ''], [mark(0, 'decoded', 'output'), mark(1, 'decoded', 'output')], { result: '["lint","#",""]' })),
  ]);

export default defineVisual('encode-and-decode-strings', draft, pendingReview(draft.objective));

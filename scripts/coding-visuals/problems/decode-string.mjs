import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Save the outer text and repeat count whenever a nested bracket opens.', [
    frame('Enter the outer repeat', '3[ starts a new inner string while saving repeat 3.', stack('3[a2[c]]', ['outer="", count=3'], { current: '[' })),
    frame('Nest again', 'At 2[, save the current a and repeat count 2.', stack('3[a2[c]]', ['outer="", count=3', 'outer="a", count=2'], { current: '[' })),
    frame('Close from the inside', 'c becomes cc, then acc, then accaccacc.', stack('3[a2[c]]', ['outer="", count=3'], { current: ']', action: 'restore outer', result: 'accaccacc' })),
  ]);

export default defineVisual('decode-string', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('The newest unmatched opening bracket must match the next closing bracket.', [
    frame('Push openings', 'Read ( and [; both remain unfinished in the stack.', stack('([', ['(', '['], { current: '[' })),
    frame('Match the top', 'The next ] matches the stack top [, then } must match {.', stack('([{}])', ['(', '[', '{'], { current: '}', action: 'pop {' })),
    frame('Empty means valid', 'All openings were closed in reverse order.', stack('([{}])', [], { result: 'true' })),
  ]);

export default defineVisual('valid-parentheses', draft, pendingReview(draft.objective));

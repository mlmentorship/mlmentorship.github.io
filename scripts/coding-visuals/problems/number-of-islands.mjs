import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Start a flood only at unseen land, then mark the whole component.', [
    frame('Find the first land', 'The top-left 1 starts island 1.', queueGrid([['1', '1', '1', '1'], ['1', '1', '0', '1'], ['1', '1', '0', '0'], ['0', '0', '0', '0']], ['(0,0)'], { action: 'start island 1' })),
    frame('Flood the component', 'Every connected 1 becomes visited water 0.', queueGrid([['0', '0', '0', '0'], ['0', '0', '0', '0'], ['0', '0', '0', '0'], ['0', '0', '0', '0']], [], { action: 'component visited' })),
    frame('Count starts, not cells', 'Only the first unseen land cell increments the island count.', queueGrid([['0', '0', '0'], ['0', '0', '0']], [], { result: '1 island' })),
  ]);

export default defineVisual('number-of-islands', draft, pendingReview(draft.objective));

import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('An edge is redundant when both endpoints already have the same representative root.', [
    frame('Join separate components', 'Edges 1-2 and 1-3 create one component rooted at 1.', graph(['1', '2', '3'], ['1 - 2', '1 - 3'], { components: ['root 1: {1,2,3}'] })),
    frame('Test the closing edge', 'For edge 2-3, find(2) and find(3) both return root 1.', graph(['1', '2', '3'], ['1 - 2', '1 - 3', '2 - 3'], { roots: ['2 -> 1', '3 -> 1'], current: '2 - 3' })),
    frame('Reject the cycle edge', 'Adding 2-3 would close a cycle, so return it.', graph(['1', '2', '3'], ['1 - 2', '1 - 3', '2 - 3'], { current: '2 - 3', result: '[2,3]' })),
  ]);

export default defineVisual('redundant-connection', draft, pendingReview(draft.objective));

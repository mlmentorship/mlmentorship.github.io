import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Every unseen node starts one DFS component and marks its whole group.', [
    frame('Start component 1', 'Node 0 reaches 1 and 2.', graph(['0', '1', '2', '3', '4'], ['0-1', '1-2', '3-4'], { visited: ['0', '1', '2'], components: '1' })),
    frame('Find the next unseen node', 'Node 3 starts a second flood and reaches 4.', graph(['0', '1', '2', '3', '4'], ['0-1', '1-2', '3-4'], { visited: ['0', '1', '2', '3', '4'], components: '2' })),
    frame('Return the count', 'Two starting floods mean two connected components.', graph(['0', '1', '2', '3', '4'], ['0-1', '1-2', '3-4'], { result: '2' })),
  ]);

export default defineVisual('number-of-connected-components', draft, pendingReview(draft.objective));

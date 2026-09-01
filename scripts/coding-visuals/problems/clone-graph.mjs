import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Map each original node to one copy, then connect copies using the map.', [
    frame('Copy the start', 'Original node 1 gets exactly one copy before neighbors are explored.', graph(['original 1', 'original 2', 'copy 1'], ['1 <-> 2'], { visited: ['original 1'], copies: ['1 -> copy 1'] })),
    frame('Copy neighbors once', 'When node 2 appears, create copy 2 and reuse copy 1 for the reverse edge.', graph(['original 1', 'original 2', 'copy 1', 'copy 2'], ['copy 1 <-> copy 2'], { copies: ['1 -> copy 1', '2 -> copy 2'] })),
    frame('Return the copied component', 'Every original edge has a matching copied edge.', graph(['copy 1', 'copy 2'], ['copy 1 <-> copy 2'], { result: 'deep copy' })),
  ]);

export default defineVisual('clone-graph', draft, pendingReview(draft.objective));

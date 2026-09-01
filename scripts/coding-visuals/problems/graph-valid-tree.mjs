import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A valid tree needs exactly n-1 edges and one connected component.', [
    frame('Check the edge count', 'Five nodes require exactly four edges.', graph(['0', '1', '2', '3', '4'], ['0-1', '0-2', '0-3', '1-4'], { detail: 'edges = 4 = n-1' })),
    frame('Traverse once', 'DFS from 0 reaches every node without finding a cycle.', graph(['0', '1', '2', '3', '4'], ['0-1', '0-2', '0-3', '1-4'], { visited: ['0', '1', '2', '3', '4'] })),
    frame('Accept the tree', 'Correct edge count plus full reachability proves a tree.', graph(['0', '1', '2', '3', '4'], ['0-1', '0-2', '0-3', '1-4'], { result: 'true' })),
  ]);

export default defineVisual('graph-valid-tree', draft, pendingReview(draft.objective));

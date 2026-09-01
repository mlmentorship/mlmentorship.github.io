import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('Dijkstra finalizes the node whose total path cost is smallest.', [
    frame('Start at node 2', 'Known distance is 0. Its outgoing paths cost 1 to nodes 1 and 3.', graph(['1', '2', '3'], ['2 -1-> 1', '2 -1-> 3'], { start: '2', frontier: ['1:1', '3:1'], visited: ['2:0'] })),
    frame('Finalize the cheapest path', 'Pop node 1 at distance 1; node 3 is already reachable at distance 1.', graph(['1', '2', '3'], ['2 -1-> 1', '2 -1-> 3', '1 -1-> 3'], { frontier: ['3:1'], visited: ['2:0', '1:1'] })),
    frame('Take the farthest finalized distance', 'Every node is reached; the delay is max(0,1,1) = 1.', graph(['1', '2', '3'], ['2 -1-> 1', '2 -1-> 3'], { visited: ['2:0', '1:1', '3:1'], result: '1' })),
  ]);

export default defineVisual('network-delay-time', draft, pendingReview(draft.objective));

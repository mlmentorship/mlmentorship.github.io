import { defineVisual, frame, graph, visual } from '../primitives.mjs';

const nodes = ['1', '2', '3', '4'];
const edges = ['1 -1-> 2', '1 -4-> 3', '2 -2-> 3', '2 -6-> 4', '3 -1-> 4'];
const positions = { 1: { x: 58, y: 115 }, 2: { x: 190, y: 52 }, 3: { x: 190, y: 178 }, 4: { x: 410, y: 115 } };
const state = (extra) => graph(nodes, edges, { positions, edgeLabelMode: 'weight', ...extra });

const draft = visual('Finalize shortest signal times by always popping the cheapest pending path.', [
  frame(
    'Initialize the source path',
    'For start=1, the min-heap contains only (distance 0, node 1); no node has a finalized distance yet.',
    state({ start: '1', frontier: ['(0,1)'], visited: [], input: 'n=4, start=1' }),
    'initialize-source',
  ),
  frame(
    'Finalize node 1 and relax its edges',
    'Pop (0,1), finalize distance[1]=0, then push (0+1,2)=(1,2) and (0+4,3)=(4,3).',
    state({ start: '1', frontier: ['(1,2)', '(4,3)'], visited: ['1:0'], relaxation: '1->2 gives 1; 1->3 gives 4' }),
    'finalize-one',
  ),
  frame(
    'Finalize node 2',
    'Pop (1,2), finalize distance[2]=1, and push (1+2,3)=(3,3) plus (1+6,4)=(7,4).',
    state({ start: '2', frontier: ['(3,3)', '(4,3)', '(7,4)'], visited: ['1:0', '2:1'], relaxation: '2->3 gives 3; 2->4 gives 7' }),
    'finalize-two',
  ),
  frame(
    'Improve the route to node 4',
    'Pop (3,3), finalize distance[3]=3, then edge 3->4 pushes (3+1,4)=(4,4), cheaper than the pending (7,4).',
    state({ start: '3', frontier: ['(4,3)', '(4,4)', '(7,4)'], visited: ['1:0', '2:1', '3:3'], relaxation: '3->4 gives 4' }),
    'finalize-three',
  ),
  frame(
    'Skip the stale path',
    'Pop (4,3). Node 3 is already finalized at distance 3, so the implementation takes the continue branch and adds no edges.',
    state({ start: '3', frontier: ['(4,4)', '(7,4)'], visited: ['1:0', '2:1', '3:3'], decision: 'skip duplicate node 3' }),
    'skip-stale-three',
  ),
  frame(
    'Finalize the last node',
    'Pop (4,4) and finalize distance[4]=4. The later (7,4) is stale and skipped; all four nodes are reached.',
    state({ start: '4', frontier: [], visited: ['1:0', '2:1', '3:3', '4:4'], decision: 'skip later (7,4)' }),
    'finalize-four',
  ),
  frame(
    'Return the network delay',
    'The signal arrival times are {1:0, 2:1, 3:3, 4:4}; the last arrival is max(0,1,3,4)=4.',
    state({ visited: ['1:0', '2:1', '3:3', '4:4'], frontier: [], result: '4' }),
    'return-delay',
  ),
]);

const review = {
  pattern: 'Dijkstra traversal with a min-heap of total path cost and a finalized-distance map.',
  recognitionCue: 'The input is a directed graph with nonnegative edge costs, and the task asks when a source reaches every node, so shortest paths from one source determine the answer.',
  invariant: 'Whenever an unfinalized node is popped with the globally smallest pending cost, that cost is its shortest distance; duplicate heap entries for finalized nodes are safely ignored.',
  stateModel: 'Keep the weighted adjacency list, a min-heap of (total cost, node) paths, and one finalized distance per reached node. The answer is the maximum finalized distance only if all nodes were reached.',
  visualRationale: 'A connected weighted graph preserves edge direction and cost while the visible heap frontier and finalized labels expose each pop, relaxation, stale-entry skip, and final maximum.',
  rejectedAlternatives: [
    'A distance table alone hides the directed graph paths that create each candidate cost.',
    'An unweighted BFS wave incorrectly suggests that edge count, rather than total cost, determines processing order.',
    'A heap-only view hides which outgoing edges generate the pending entries.',
  ],
  transferLesson: 'Reuse Dijkstra for nonnegative weighted routing: enqueue improved path candidates, trust only the first pop of each node, and derive the requested aggregate from finalized shortest distances.',
  reviewStatus: 'reviewed',
};

export default defineVisual('network-delay-time', draft, review);

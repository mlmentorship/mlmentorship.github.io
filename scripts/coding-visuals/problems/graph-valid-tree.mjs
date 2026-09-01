import { defineVisual, frame, queueGrid, visual } from '../primitives.mjs';

const edges = ['0-1', '0-2', '1-3', '1-4'];
const adjacency = [['1', '2'], ['0', '3', '4'], ['0'], ['1'], ['1']];
const scene = (extra = {}) => {
  const { frontier = [], ...metadata } = extra;
  return queueGrid(adjacency, frontier, {
    edgeTopology: edges.join(', '),
    rowMeaning: 'row index = node; cells = adjacent nodes',
    ...metadata,
    motion: [
      { key: 'graph-topology', kind: 'state', x: 0, y: 0, label: edges.join(', ') },
      ...frontier.map((item, index) => ({ key: `frontier-${item}`, kind: 'frontier', x: index, y: 1, label: item })),
    ],
  });
};

const draft = visual('With exactly n-1 undirected edges, reaching all n nodes proves the graph is one connected acyclic tree.', [
  frame(
    'Pass the edge-count gate',
    'For node_count = 5 and edges [[0,1],[0,2],[1,3],[1,4]], len(edges) = 4 = 5 - 1, so build the undirected adjacency lists.',
    scene({ adjacency: '0:[1,2]; 1:[0,3,4]; 2:[0]; 3:[1]; 4:[1]', edgeCheck: '4 = n-1' }),
    'check-edge-count',
  ),
  frame(
    'Seed DFS at node 0',
    'Initialize seen = {0} and stack = [0]. The stack contains reached nodes whose neighbors still need inspection.',
    scene({ start: '0', visited: ['0'], frontier: ['0'], stackState: '[0]' }),
    'seed-dfs',
  ),
  frame(
    'Pop 0 and discover 1, then 2',
    'Adjacency[0] is [1,2]. Add each unseen neighbor to seen and append it, producing seen {0,1,2} and stack [1,2].',
    scene({ start: '0', visited: ['0', '1', '2'], frontier: ['1', '2'], stackState: '[1,2]', transition: 'pop 0; push 1; push 2' }),
    'expand-0',
  ),
  frame(
    'Pop 2',
    'LIFO order pops 2. Its only neighbor 0 is already seen, so no state is added and stack becomes [1].',
    scene({ start: '2', visited: ['0', '1', '2'], frontier: ['1'], stackState: '[1]', transition: '2->0 skipped: seen' }),
    'expand-2',
  ),
  frame(
    'Pop 1 and discover 3, then 4',
    'Adjacency[1] is [0,3,4]. Skip seen node 0; add 3 and 4, producing seen {0,1,2,3,4} and stack [3,4].',
    scene({ start: '1', visited: ['0', '1', '2', '3', '4'], frontier: ['3', '4'], stackState: '[3,4]', transition: 'skip 0; push 3; push 4' }),
    'expand-1',
  ),
  frame(
    'Pop leaf 4',
    'Node 4 only links to seen node 1. Add nothing and leave stack [3].',
    scene({ start: '4', visited: ['0', '1', '2', '3', '4'], frontier: ['3'], stackState: '[3]', transition: '4->1 skipped: seen' }),
    'expand-4',
  ),
  frame(
    'Pop leaf 3',
    'Node 3 only links to seen node 1. Add nothing; the stack is empty and traversal ends.',
    scene({ start: '3', visited: ['0', '1', '2', '3', '4'], stackState: '[]', transition: '3->1 skipped: seen' }),
    'expand-3',
  ),
  frame(
    'Accept full reachability',
    'len(seen) = 5 = node_count, so return true. A connected graph with exactly n-1 edges cannot contain a cycle.',
    scene({ visited: ['0', '1', '2', '3', '4'], finalCheck: 'len(seen)=5=node_count', result: 'true' }),
    'return-true',
  ),
]);

const review = {
  pattern: 'Undirected tree validation by the n-1 edge-count gate followed by one iterative DFS reachability test.',
  recognitionCue: 'Use this proof when an undirected graph must be exactly one tree: reject any edge count other than n-1, then test whether one traversal reaches every vertex.',
  invariant: 'seen contains every discovered vertex and stack contains discovered vertices not yet expanded. After the n-1 gate passes, full reachability is equivalent to being a tree.',
  stateModel: 'The minimal state is an adjacency list, a seen set, and a DFS stack. The trace follows the supplied neighbor order and shows every pop, seen-neighbor skip, discovery, push, and final cardinality comparison.',
  visualRationale: 'Persistent adjacency rows expose every undirected edge in both directions while the adjacent frontier and seen labels make iterative DFS readable without color. A stable topology key preserves graph identity across all expansions and remains legible at 390px.',
  rejectedAlternatives: [
    'Union-find was rejected because it would depict a different implementation and omit the supplied reachability proof.',
    'An edge-count formula alone was rejected because n-1 edges can still be disconnected.',
    'An edge-count and visited-count summary was rejected because it hides which adjacency causes each discovery or seen-neighbor skip.',
  ],
  transferLesson: 'Combine a cheap global structural count with one local traversal invariant. Similar count-plus-connectivity proofs validate arborescences, spanning trees, and network skeletons.',
  reviewStatus: 'reviewed',
};

export default defineVisual('graph-valid-tree', draft, review);

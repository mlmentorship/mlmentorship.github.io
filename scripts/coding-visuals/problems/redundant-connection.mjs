import { defineVisual, frame, graph, visual } from '../primitives.mjs';

const nodes = ['1', '2', '3', '4'];

const draft = visual('Union-find accepts edges between different representative roots and identifies the first edge whose endpoints already share one root.', [
  frame(
    'Initialize singleton components',
    'For edges [[1,2],[3,4],[2,3],[4,2]], allocate indices 0..4. Nodes 1 through 4 begin as separate roots with size 1.',
    graph(nodes, [], { parent: '[0,1,2,3,4]', componentSize: '[1,1,1,1,1]', current: '[1,2]' }),
    'initialize-sets',
  ),
  frame(
    'Union edge 1-2',
    'find(1)=1 and find(2)=2. Equal-size roots need no swap; set parent[2]=1 and size[1]=1+1=2.',
    graph(nodes, ['1 - 2'], { roots: ['1->1', '2->2'], parent: '[0,1,1,3,4]', componentSize: 'size(1)=2', accepted: '[1,2]' }),
    'union-1-2',
  ),
  frame(
    'Union edge 3-4',
    'find(3)=3 and find(4)=4. Set parent[4]=3 and size[3]=2, creating a second two-node component.',
    graph(nodes, ['1 - 2', '3 - 4'], { roots: ['3->3', '4->4'], parent: '[0,1,1,3,3]', componentSize: 'size(1)=2; size(3)=2', accepted: '[3,4]' }),
    'union-3-4',
  ),
  frame(
    'Join the two components',
    'For edge 2-3, find(2) follows 2->1 while find(3)=3. Both roots have size 2, so attach root 3 under root 1 and set size[1]=4.',
    graph(nodes, ['1 - 2', '3 - 4', '2 - 3'], { roots: ['2->1', '3->3'], parent: '[0,1,1,1,3]', componentSize: 'size(1)=2+2=4', accepted: '[2,3]' }),
    'union-2-3',
  ),
  frame(
    'Compress the path from node 4',
    'For candidate edge 4-2, find(4) sees 4->3->1. Path halving writes parent[4]=parent[3]=1; find(2) also returns 1.',
    graph(nodes, ['1 - 2', '3 - 4', '2 - 3'], { roots: ['4->3->1', '2->1'], parentBefore: '[0,1,1,1,3]', parentAfter: '[0,1,1,1,1]', current: '[4,2]' }),
    'compress-4',
  ),
  frame(
    'Reject the cycle-closing edge',
    'Both endpoints of 4-2 have representative root 1, so union returns false. Adding 4-2 would close the path 4-3-2-4.',
    graph(nodes, ['1 - 2', '3 - 4', '2 - 3', '4 - 2'], { roots: ['4->1', '2->1'], union: 'false: roots equal', result: '[4,2]' }),
    'return-redundant-edge',
  ),
]);

const review = {
  pattern: 'Disjoint-set union with path halving and union by component size for incremental cycle detection.',
  recognitionCue: 'Use union-find when undirected edges arrive over time and each edge needs a fast “already connected?” decision without repeatedly traversing the whole graph.',
  invariant: 'Following parent links from any node reaches the unique representative of its connected component; component_size is authoritative at roots. Union changes connectivity only when roots differ.',
  stateModel: 'The minimal state is parent and component_size arrays. Each edge performs two finds, optional path compression, optional size-based root swap, and one parent/size update or immediate false return.',
  visualRationale: 'The accepted undirected edges show the actual component topology while root paths and parent arrays expose union-find’s hidden forest. Stable graph keys preserve vertices and accepted links as components merge.',
  rejectedAlternatives: [
    'Repeated DFS was rejected because it depicts a different O(EV) style connectivity check.',
    'Only parent arrays were rejected because they hide the graph cycle that the rejected edge would close.',
    'Only the final triangle was rejected because it skips union-by-size and path compression.',
  ],
  transferLesson: 'Ask whether roots differ before adding connectivity; attach the smaller tree under the larger and shorten paths during find. The same state supports component counts, Kruskal MST, and dynamic grouping.',
  reviewStatus: 'reviewed',
};

export default defineVisual('redundant-connection', draft, review);

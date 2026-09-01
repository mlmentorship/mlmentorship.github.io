import { defineVisual, frame, graph, visual } from '../primitives.mjs';

const originalEdges = ['O1 <-> O2', 'O1 <-> O3', 'O2 <-> O3'];

const draft = visual('Create one mapped copy per original node, then reproduce every adjacency entry between copies.', [
  frame(
    'Create the first copy',
    'For triangle O1-O2-O3, create C1 before traversal and enqueue O1. The map is {O1:C1}.',
    graph(['O1', 'O2', 'O3', 'C1'], originalEdges, { start: 'O1', current: 'O1', frontier: ['O1'], copies: 'O1->C1' }),
    'seed-copy-map',
  ),
  frame(
    'Process original node O1',
    'Pop O1. Discover O2, create C2, enqueue O2, and append C2 to C1.neighbors.',
    graph(['O1', 'O2', 'O3', 'C1', 'C2'], [...originalEdges, 'C1 -> C2'], { start: 'O1', current: 'O1', frontier: ['O2'], copies: 'O1->C1, O2->C2' }),
    'copy-o2',
  ),
  frame(
    'Finish O1 adjacency',
    'O1 next reaches unseen O3: create C3, enqueue O3, and append C3 to C1.neighbors. The queue is [O2,O3].',
    graph(['O1', 'O2', 'O3', 'C1', 'C2', 'C3'], [...originalEdges, 'C1 -> C2', 'C1 -> C3'], { start: 'O1', current: 'O1', frontier: ['O2', 'O3'], copies: 'O1->C1, O2->C2, O3->C3' }),
    'copy-o3',
  ),
  frame(
    'Reuse copies while processing O2',
    'Pop O2. Both neighbors are already mapped, so append existing C1 and C3 to C2.neighbors without creating or enqueueing another node.',
    graph(['O1', 'O2', 'O3', 'C1', 'C2', 'C3'], [...originalEdges, 'C1 -> C2', 'C1 -> C3', 'C2 -> C1', 'C2 -> C3'], { start: 'O2', current: 'O2', frontier: ['O3'], copies: '3 copies; no duplicates' }),
    'connect-c2',
  ),
  frame(
    'Finish the copied adjacency',
    'Pop O3 and append existing C1 and C2 to C3.neighbors. The queue becomes empty after every original node is processed once.',
    graph(['O1', 'O2', 'O3', 'C1', 'C2', 'C3'], [...originalEdges, 'C1 -> C2', 'C1 -> C3', 'C2 -> C1', 'C2 -> C3', 'C3 -> C1', 'C3 -> C2'], { start: 'O3', current: 'O3', frontier: [], copies: 'O1->C1, O2->C2, O3->C3' }),
    'connect-c3',
  ),
  frame(
    'Return the independent component',
    'Return C1. Copies C1,C2,C3 have the triangle adjacency of O1,O2,O3, but every copied edge points only to copied nodes.',
    graph(['O1', 'O2', 'O3', 'C1', 'C2', 'C3'], [...originalEdges, 'C1 -> C2', 'C1 -> C3', 'C2 -> C1', 'C2 -> C3', 'C3 -> C1', 'C3 -> C2'], { current: 'return C1', copies: 'all clone edges stay inside C1,C2,C3', result: 'deep copy rooted at C1' }),
    'return-deep-copy',
  ),
]);

const review = {
  pattern: 'Breadth-first graph traversal plus an original-node to copied-node identity map.',
  recognitionCue: 'A cyclic connected object must be deep-copied while preserving adjacency, so each original identity needs exactly one reusable copied identity before edges are wired.',
  invariant: 'Every key in copies maps to exactly one clone, and after an original node is processed, its clone has one copied neighbor entry for every original neighbor processed in order.',
  stateModel: 'Keep a queue of discovered originals and a map from each original object to its clone. The map doubles as the visited set and as the lookup used to build clone-to-clone edges.',
  visualRationale: 'Showing original and copied nodes together with real edges, the queue, and the identity map makes cycles, reuse, and the prohibition on edges back to originals directly visible.',
  rejectedAlternatives: [
    'A node-value table cannot show whether copied adjacency points to originals or clones.',
    'A recursive call tree duplicates shared graph nodes and misrepresents cyclic topology.',
    'A final before-and-after snapshot hides when the map prevents duplicate copies.',
  ],
  transferLesson: 'When copying any cyclic object graph, allocate and memoize an object before traversing its references, then resolve every copied reference through that memo.',
  reviewStatus: 'reviewed',
};

export default defineVisual('clone-graph', draft, review);

import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const topology = [
  ['from\\to', '0', '1', '2', '3'],
  ['0', '-', '0->1', '0->2', '-'],
  ['1', '-', '-', '-', '1->3'],
  ['2', '-', '-', '-', '2->3'],
  ['3', '-', '-', '-', '-'],
];
const state = (edge, extra = {}) => grid(topology, edge ? [{
  row: edge[0], col: edge[1], label: 'remove edge', tone: 'focus', key: 'edge-cursor',
}] : [], {
  example: 'course_count=4, prerequisites=[[1,0],[2,0],[3,1],[3,2]]',
  edgeMeaning: 'matrix cell is prerequisite -> course',
  motion: [
    ...topology.flatMap((row, y) => row.map((value, x) => ({ key: `cell-${y}-${x}`, kind: 'cell', x, y, label: value }))),
    ...(edge ? [{ key: 'edge-cursor', kind: 'pointer', x: edge[1], y: edge[0], label: topology[edge[0]][edge[1]] }] : []),
  ],
  ...extra,
});

const draft = visual('Emit only zero-indegree courses; remove each outgoing edge and enqueue a neighbor exactly when its count reaches zero.', [
  frame('Build the prerequisite DAG', 'Edges are 0->1, 0->2, 1->3, and 2->3; indegrees are [0,1,1,2], so ready=[0].', state(null, { indegree: '[0,1,1,2]', ready: '[0]', order: '[]' }), 'seed-ready'),
  frame('Emit course 0', 'Pop 0, append it to order, and inspect its adjacency list [1,2] in insertion order.', state(null, { current: '0', indegree: '[0,1,1,2]', ready: '[]', order: '[0]' }), 'emit-0'),
  frame('Remove edge 0->1', 'Decrement indegree[1] from 1 to 0, then enqueue 1.', state([1,2], { indegree: '[0,0,1,2]', ready: '[1]', order: '[0]', transition: '1 -> 0; enqueue 1' }), 'release-one'),
  frame('Remove edge 0->2', 'Decrement indegree[2] from 1 to 0, then enqueue 2 after 1.', state([1,3], { indegree: '[0,0,0,2]', ready: '[1,2]', order: '[0]', transition: '1 -> 0; enqueue 2' }), 'release-two'),
  frame('Emit course 1', 'Pop 1 and append it; course 2 remains at the front of ready.', state(null, { current: '1', indegree: '[0,0,0,2]', ready: '[2]', order: '[0,1]' }), 'emit-1'),
  frame('Remove edge 1->3', 'Decrement indegree[3] from 2 to 1; because it is not zero, do not enqueue 3.', state([2,4], { indegree: '[0,0,0,1]', ready: '[2]', order: '[0,1]', transition: '2 -> 1; do not enqueue' }), 'decrement-three'),
  frame('Emit course 2', 'Pop 2 and append it, temporarily emptying ready.', state(null, { current: '2', indegree: '[0,0,0,1]', ready: '[]', order: '[0,1,2]' }), 'emit-2'),
  frame('Remove edge 2->3', 'Decrement indegree[3] from 1 to 0 and enqueue 3.', state([3,4], { indegree: '[0,0,0,0]', ready: '[3]', order: '[0,1,2]', transition: '1 -> 0; enqueue 3' }), 'release-three'),
  frame('Emit course 3', 'Pop 3 and append it; it has no outgoing edges and ready becomes empty.', state(null, { current: '3', indegree: '[0,0,0,0]', ready: '[]', order: '[0,1,2,3]' }), 'emit-3'),
  frame('Return the feasible order', 'Order length 4 equals course_count, so return [0,1,2,3].', state(null, { check: '4 == 4', edgeCheck: '0<1, 0<2, 1<3, 2<3', result: '[0,1,2,3]' }), 'return-order'),
]);

export default defineVisual('course-schedule-ii', draft, {
  pattern: 'Kahn topological sort with an indegree array, FIFO zero-indegree frontier, and emitted order.',
  recognitionCue: 'Directed prerequisites require one valid linear order; a cycle is present when some vertex never reaches zero indegree.',
  invariant: 'Every ready vertex has zero incoming edges from un-emitted vertices, and order is a valid topological prefix; each outgoing edge is removed exactly once.',
  stateModel: 'Retain adjacency lists, indegrees, FIFO ready queue, and emitted order.',
  visualRationale: 'A compact adjacency matrix preserves every directed edge while each decrement, zero test, enqueue, dequeue, and append appears separately.',
  rejectedAlternatives: ['A final order hides readiness.', 'A prose queue omits topology.', 'DFS finishing times depict a different implementation.'],
  transferLesson: 'Remove resolved requirements one edge at a time and enqueue exactly on transition to zero; retain the emitted prefix when an order, not just feasibility, is required.',
  independentReview: '3.4 source-to-frame replay',
  reviewStatus: 'reviewed',
});

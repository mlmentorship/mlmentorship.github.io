import { defineVisual, frame, graph, visual } from '../primitives.mjs';

const nodes = ['0', '1', '2', '3'];
const edges = ['0 -> 1', '0 -> 2', '1 -> 3', '2 -> 3'];

const draft = visual('Kahn’s queue emits a valid order because a course enters it only after every incoming prerequisite edge has been removed.', [
  frame(
    'Build the prerequisite DAG',
    'For 4 courses and prerequisites [[1,0],[2,0],[3,1],[3,2]], add edges 0->1, 0->2, 1->3, and 2->3. Indegrees are [0,1,1,2].',
    graph(nodes, edges, { indegree: ['0:0', '1:1', '2:1', '3:2'], order: [], ready: ['0'] }),
    'seed-ready',
  ),
  frame(
    'Emit course 0',
    'Pop 0 and append it. Removing 0->1 changes indegree[1] 1->0 and removing 0->2 changes indegree[2] 1->0, so enqueue 1 then 2.',
    graph(nodes, edges, { indegree: ['0:0', '1:0', '2:0', '3:2'], order: ['0'], ready: ['1', '2'], removed: '0->1, 0->2' }),
    'emit-0',
  ),
  frame(
    'Emit course 1',
    'Pop 1 and append it. Removing 1->3 changes indegree[3] 2->1, which is not zero, so course 3 stays out of the queue.',
    graph(nodes, edges, { indegree: ['0:0', '1:0', '2:0', '3:1'], order: ['0', '1'], ready: ['2'], removed: '1->3' }),
    'emit-1',
  ),
  frame(
    'Emit course 2',
    'Pop 2 and append it. Removing 2->3 changes indegree[3] 1->0; now every prerequisite of 3 is emitted, so enqueue 3.',
    graph(nodes, edges, { indegree: ['0:0', '1:0', '2:0', '3:0'], order: ['0', '1', '2'], ready: ['3'], removed: '2->3' }),
    'emit-2',
  ),
  frame(
    'Emit the last course',
    'Pop 3 and append it. Course 3 has no outgoing edges, so the ready queue becomes empty and order has length 4.',
    graph(nodes, edges, { indegree: ['0:0', '1:0', '2:0', '3:0'], order: ['0', '1', '2', '3'], ready: [], check: 'len(order)=4=course_count' }),
    'emit-3',
  ),
  frame(
    'Return the feasible order',
    'Because every course was emitted, return [0,1,2,3]. Each edge points from an earlier course to a later course in this order.',
    graph(nodes, edges, { order: ['0', '1', '2', '3'], edgeCheck: '0<1, 0<2, 1<3, 2<3', result: '[0,1,2,3]' }),
    'return-order',
  ),
]);

const review = {
  pattern: 'Kahn topological sort with an indegree array, FIFO zero-indegree frontier, and emitted order.',
  recognitionCue: 'Use it when directed prerequisites or dependencies require one valid linear order, and cycles must be detected because they prevent some vertices from ever becoming ready.',
  invariant: 'Every vertex in ready has zero incoming edges from un-emitted vertices, and order is a valid topological prefix. Decrementing an outgoing neighbor once per removed edge preserves both facts.',
  stateModel: 'The minimal state is the directed adjacency list, one indegree per course, a FIFO ready queue, and the emitted order. The trace shows every pop, edge removal, decrement, zero test, and enqueue.',
  visualRationale: 'A connected directed graph exposes prerequisite topology and the converging edges into course 3. Persistent node/edge keys accompany explicit ready, indegree, removed-edge, and order labels, so meaning does not depend on color.',
  rejectedAlternatives: [
    'An indegree table alone was rejected because it hides which outgoing edge causes each decrement.',
    'A final ordered list was rejected because it does not explain readiness or cycle detection.',
    'A DFS finishing-time stack was rejected because it depicts a different topological-sort implementation.',
  ],
  transferLesson: 'Maintain the number of unresolved incoming dependencies, enqueue exactly on the transition to zero, and compare emitted count with vertex count; this transfers to build systems and workflow scheduling.',
  reviewStatus: 'reviewed',
};

export default defineVisual('course-schedule-ii', draft, review);

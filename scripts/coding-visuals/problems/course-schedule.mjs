import { defineVisual, frame, graph, visual } from '../primitives.mjs';

const nodes = ['0', '1', '2', '3'];
const edges = ['0 -> 1', '0 -> 2', '1 -> 3', '2 -> 3'];
const example = 'course_count = 4, prerequisites = [[1,0], [2,0], [3,1], [3,2]]';
const state = (extra = {}) => graph(nodes, edges, { example, edgeMeaning: 'prerequisite -> course', ...extra });

const draft = visual('Only dequeue zero-indegree courses; removing their outgoing edges may make new courses ready.', [
  frame(
    'Build the dependency graph',
    'The four prerequisite pairs create edges 0->1, 0->2, 1->3, and 2->3 with indegrees [0,1,1,2].',
    state({ indegree: ['0:0', '1:1', '2:1', '3:2'], ready: ['0'], completed: '0' }),
    'initialize-graph-and-queue',
  ),
  frame(
    'Dequeue course 0',
    'Course 0 has no unmet prerequisite. Remove it from ready and increment completed from 0 to 1.',
    state({ start: '0', visited: ['0'], indegree: ['0:0', '1:1', '2:1', '3:2'], ready: [], completed: '1' }),
    'dequeue-zero',
  ),
  frame(
    'Release course 1',
    'Following edge 0->1 decrements indegree[1] from 1 to 0, so append course 1 to ready.',
    state({ visited: ['0'], indegree: ['0:0', '1:0', '2:1', '3:2'], ready: ['1'], transition: 'indegree[1]: 1 -> 0; enqueue 1', completed: '1' }),
    'release-one',
  ),
  frame(
    'Release course 2',
    'Following edge 0->2 decrements indegree[2] from 1 to 0, so append course 2 after course 1.',
    state({ visited: ['0'], indegree: ['0:0', '1:0', '2:0', '3:2'], ready: ['1', '2'], transition: 'indegree[2]: 1 -> 0; enqueue 2', completed: '1' }),
    'release-two',
  ),
  frame(
    'Dequeue course 1',
    'Pop course 1 from the left of ready and increment completed to 2; course 2 remains queued.',
    state({ start: '1', visited: ['0', '1'], indegree: ['0:0', '1:0', '2:0', '3:2'], ready: ['2'], completed: '2' }),
    'dequeue-one',
  ),
  frame(
    'Remove the first requirement of course 3',
    'Following edge 1->3 changes indegree[3] from 2 to 1. Course 3 is not ready, so it is not enqueued.',
    state({ visited: ['0', '1'], indegree: ['0:0', '1:0', '2:0', '3:1'], ready: ['2'], transition: 'indegree[3]: 2 -> 1; do not enqueue', completed: '2' }),
    'decrement-three-once',
  ),
  frame(
    'Dequeue course 2',
    'Pop course 2 and increment completed to 3. The ready queue is temporarily empty.',
    state({ start: '2', visited: ['0', '1', '2'], indegree: ['0:0', '1:0', '2:0', '3:1'], ready: [], completed: '3' }),
    'dequeue-two',
  ),
  frame(
    'Release course 3',
    'Following edge 2->3 changes indegree[3] from 1 to 0, so append course 3 to ready.',
    state({ visited: ['0', '1', '2'], indegree: ['0:0', '1:0', '2:0', '3:0'], ready: ['3'], transition: 'indegree[3]: 1 -> 0; enqueue 3', completed: '3' }),
    'release-three',
  ),
  frame(
    'Dequeue course 3',
    'Pop the final ready course and increment completed from 3 to 4. It has no outgoing edges.',
    state({ start: '3', visited: ['0', '1', '2', '3'], indegree: ['0:0', '1:0', '2:0', '3:0'], ready: [], completed: '4' }),
    'dequeue-three',
  ),
  frame(
    'Compare completed with course count',
    'The queue is empty after all four nodes were removed. completed == course_count is 4 == 4, so return true.',
    state({ visited: ['0', '1', '2', '3'], indegree: ['0:0', '1:0', '2:0', '3:0'], ready: [], comparison: '4 == 4', result: 'true' }),
    'return-true',
  ),
]);

const review = {
  pattern: 'Kahn topological sort using adjacency lists, indegree counts, and a zero-indegree queue.',
  recognitionCue: 'Directed prerequisite pairs ask whether every node can be placed after all requirements; failure to process every node means a directed cycle blocks the remainder.',
  invariant: 'Every queued course has indegree zero in the remaining graph, completed counts dequeued courses, and each outgoing edge is removed exactly once when its prerequisite completes.',
  stateModel: 'Retain the directed adjacency graph, one indegree per course, FIFO ready queue, and completed count. A full ordering is unnecessary for this boolean variant.',
  visualRationale: 'A real four-node directed topology stays fixed while ready, completed nodes, and indegrees change, exposing both converging prerequisites and the exact zero transition.',
  rejectedAlternatives: [
    'An indegree table without edges hides which completed prerequisite decrements which course.',
    'A generic queue flowchart omits the dependency topology and cannot explain a blocked cycle.',
    'DFS recursion can detect cycles but would depict a different state model from the supplied Kahn implementation.',
  ],
  transferLesson: 'Repeatedly remove entities with no unmet requirements and decrement their dependents; the same mechanism orders builds, recipes, jobs, and dependency migrations, while leftovers certify a cycle.',
  reviewStatus: 'reviewed',
};

export default defineVisual('course-schedule', draft, review);

import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const steps = ['step 0', 'step 1', 'step 2', 'step 3', 'step 4', 'step 5'];

function stairState(iteration, values, previous, current, operation, result) {
  const currentIndex = Math.min(iteration, 5);
  return array(values, [
    mark(Math.max(0, currentIndex - 1), 'previous', 'state', 'previous'),
    mark(currentIndex, 'current', result ? 'output' : 'focus', 'current'),
  ], {
    input: 'n = 5',
    rollingState: `previous=${previous}, current=${current}`,
    dependency: operation,
    ...(result ? { result } : {}),
  });
}

const draft = visual('The count for the next step is the sum of the counts for the two preceding positions.', [
  frame(
    'Initialize two rolling totals',
    'For n = 5, previous = 0 and current = 1 represent the two values needed before the first loop update.',
    stairState(0, ['0', '1', '?', '?', '?', '?'], '0', '1', 'base state before iteration 1'),
    'initialize',
  ),
  frame(
    'Update for step 1',
    'Parallel assignment computes (previous, current) = (1, 0 + 1), so there is 1 way to reach step 1.',
    stairState(1, ['0', '1', '?', '?', '?', '?'], '1', '1', 'ways(1) = 0 + 1 = 1'),
    'step-1',
  ),
  frame(
    'Update for step 2',
    'The next pair is (1, 1 + 1) = (1, 2): reach step 2 from step 1 or step 0.',
    stairState(2, ['0', '1', '2', '?', '?', '?'], '1', '2', 'ways(2) = ways(1) + ways(0) = 1 + 1 = 2'),
    'step-2',
  ),
  frame(
    'Update for step 3',
    'The next pair is (2, 1 + 2) = (2, 3), preserving only the two dependencies needed next.',
    stairState(3, ['0', '1', '2', '3', '?', '?'], '2', '3', 'ways(3) = ways(2) + ways(1) = 2 + 1 = 3'),
    'step-3',
  ),
  frame(
    'Update for step 4',
    'The next pair is (3, 2 + 3) = (3, 5). Earlier counts cannot affect a later state directly.',
    stairState(4, ['0', '1', '2', '3', '5', '?'], '3', '5', 'ways(4) = ways(3) + ways(2) = 3 + 2 = 5'),
    'step-4',
  ),
  frame(
    'Update for step 5',
    'The fifth loop update produces (5, 3 + 5) = (5, 8), and the function returns current = 8.',
    stairState(5, ['0', '1', '2', '3', '5', '8'], '5', '8', 'ways(5) = ways(4) + ways(3) = 5 + 3 = 8', '8'),
    'step-5',
  ),
]);

export default defineVisual('climbing-stairs', draft, {
  pattern: 'One-dimensional dynamic programming compressed to the last two answers.',
  recognitionCue: 'The task counts paths to position n, every final move has one of two fixed lengths, and paths ending with different final moves are disjoint.',
  invariant: 'Before each iteration, previous and current are consecutive recurrence values. Parallel assignment shifts current into previous and stores previous + current as the complete count for the next step.',
  stateModel: 'Only two integers are needed because ways(i) depends solely on ways(i - 1) and ways(i - 2). The loop index identifies which stair count current represents.',
  visualRationale: 'An indexed stair sequence shows fill order and both predecessor dependencies, while stable previous/current keys visibly advance one position per update and printed equations preserve meaning without animation.',
  rejectedAlternatives: [
    'A full binary recursion tree repeats the same subproblems and obscures the linear recurrence.',
    'A complete DP table stores more history than the implementation needs.',
    'A Fibonacci label alone requires memorizing the recurrence rather than seeing why the two predecessor counts add.',
  ],
  transferLesson: 'Classify solutions by their final decision: if every state can only arrive from a fixed small set of predecessor states, add those disjoint counts and retain only the dependency horizon.',
  reviewStatus: 'reviewed',
});

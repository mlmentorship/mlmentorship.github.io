import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const matrix = [['3', '4', '5'], ['2', '1', '6']];
const state = (row, col, extra = {}) => grid(matrix, [
  { row, col, label: 'DFS cursor', tone: 'focus', key: 'dfs-cursor' },
], extra);

const draft = visual('Treat larger-neighbor moves as a DAG, recursively solve each cell once, and memoize the longest suffix length returned to its predecessors.', [
  frame('Start DFS at value 3', 'The outer max begins at (0,0)=3. Only larger neighbor 4 can extend this path.', state(0, 0, {
    memo: '[.,.,.; .,.,.]',
    dependency: '3 -> 4',
    best: 'path_from(3) starts at 1',
  }), 'start-3'),
  frame('Descend through 4 and 5', 'At 4, only 5 is larger; at 5, the down neighbor 6 is larger.', state(0, 2, {
    memo: '[.,.,.; .,.,.]',
    callStack: '3 -> 4 -> 5',
    dependency: '5 -> 6',
  }), 'descend-5'),
  frame('Cache the peak 6', '6 has no larger in-bounds neighbor, so path_from(6)=1.', state(1, 2, {
    memo: '[.,.,.; .,.,1]',
    arithmetic: 'best(6) = 1',
  }), 'cache-6'),
  frame('Return and cache 5', 'The recursive result from 6 gives best(5)=max(1,1+1)=2.', state(0, 2, {
    memo: '[.,.,2; .,.,1]',
    arithmetic: 'best(5) = max(1, 1 + memo[6]) = 2',
  }), 'cache-5'),
  frame('Return through 4 and 3', 'Cache best(4)=1+2=3, then best(3)=1+3=4.', state(0, 0, {
    memo: '[4,3,2; .,.,1]',
    arithmetic: 'best(4)=3; best(3)=4',
  }), 'cache-3'),
  frame('Solve value 2 with a cache hit', 'The outer scan reaches (1,0)=2. Its larger neighbor 3 is cached at 4, so best(2)=1+4=5.', state(1, 0, {
    memo: '[4,3,2; 5,.,1]',
    cacheHit: 'memo[3] = 4',
    arithmetic: 'best(2) = max(1, 1 + 4) = 5',
  }), 'cache-2'),
  frame('Solve value 1 from three larger choices', 'From (1,1)=1, candidates are 1+memo[4]=4, 1+memo[6]=2, and 1+memo[2]=6; choose 6.', state(1, 1, {
    memo: '[4,3,2; 5,6,1]',
    dependencies: '1 -> 4, 1 -> 6, 1 -> 2',
    arithmetic: 'max(1, 1+3, 1+1, 1+5) = 6',
  }), 'cache-1'),
  frame('Take the maximum over all starts', 'All six cells are memoized once. max(4,3,2,5,6,1)=6 for path 1->2->3->4->5->6.', state(1, 1, {
    memo: '[4,3,2; 5,6,1]',
    path: '(1,1)->(1,0)->(0,0)->(0,1)->(0,2)->(1,2)',
    result: '6',
  }), 'finish'),
]);

const review = {
  pattern: 'DFS dynamic programming on the acyclic graph induced by strictly increasing grid edges.',
  recognitionCue: 'Use memoized DFS when every cell asks for an optimal path through neighboring states and a strict monotone move rule prevents cycles.',
  invariant: 'Once path_from(r,c) returns, its memo value is the longest increasing path starting at that cell; every candidate is 1 plus an already correct recursively computed larger-neighbor suffix.',
  stateModel: 'Retain the immutable matrix, a cache keyed by coordinates, the current DFS call stack, and a local best initialized to one.',
  visualRationale: 'The fixed grid preserves adjacency while a stable DFS cursor moves along real coordinate dependencies; visible cache states and recurrence arithmetic expose both recursion and reuse.',
  rejectedAlternatives: [
    'Enumerating every increasing path repeats the same suffixes exponentially.',
    'Sorting cells and filling iteratively is valid but does not match the supplied DFS call order.',
    'A final heatmap hides the dependency returns and cache-hit mechanism.',
  ],
  transferLesson: 'A strict ranking function can turn implicit neighbor moves into a DAG; memoize the optimal suffix at each state and combine it from higher-ranked neighbors.',
  reviewStatus: 'reviewed',
};

export default defineVisual('longest-increasing-path-in-a-matrix', draft, review);

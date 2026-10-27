import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const matrix = [['3', '4', '5'], ['2', '1', '6']];
const completedOrder = ['r1c2', 'r0c2', 'r0c1', 'r0c0', 'r1c0', 'r1c1'];
const state = (row, col, extra = {}, completed = 0) => grid(matrix, [
  { row, col, label: 'DFS cursor', tone: 'focus', key: 'dfs-cursor' },
], {
  ...extra,
  graphCompanion: {
    nodes: ['r0c0', 'r0c1', 'r0c2', 'r1c0', 'r1c1', 'r1c2'],
    nodeLabels: { r0c0: '3', r0c1: '4', r0c2: '5', r1c0: '2', r1c1: '1', r1c2: '6' },
    edges: ['r0c0->r0c1', 'r0c1->r0c2', 'r0c2->r1c2', 'r1c0->r0c0', 'r1c1->r0c1', 'r1c1->r1c2', 'r1c1->r1c0'],
    positions: { r0c0: { x: 45, y: 45 }, r0c1: { x: 175, y: 45 }, r0c2: { x: 305, y: 45 }, r1c0: { x: 45, y: 185 }, r1c1: { x: 175, y: 185 }, r1c2: { x: 305, y: 185 } },
    width: 350,
    height: 235,
    treeLayout: true,
    start: `r${row}c${col}`,
    stateNodes: completedOrder.slice(0, completed),
    markerId: `increasing-path-${row}-${col}-${completed}`,
    edgeLabels: false,
    completed: String(completed),
    ariaLabel: 'Six matrix cells with arrows to strictly larger orthogonal neighbors and completed suffix marks',
  },
  gridSummary: 'Original matrix and DFS cursor',
});

const draft = visual('Treat larger-neighbor moves as a DAG, recursively solve each cell once, and memoize the longest suffix length returned to its predecessors.', [
  frame('Start DFS at value 3', 'The outer max begins at (0,0)=3. Only larger neighbor 4 can extend this path.', state(0, 0, {
    memo: '[.,.,.; .,.,.]',
    dependency: '3 -> 4',
    best: 'path_from(0,0) starts at 1',
  }), 'start-3'),
  frame('Descend through 4 and 5', 'At 4, only 5 is larger. At 5, the down neighbor 6 is larger.', state(0, 2, {
    memo: '[.,.,.; .,.,.]',
    callStack: '3 -> 4 -> 5',
    dependency: '5 -> 6',
  }), 'descend-5'),
  frame('Cache the peak 6', 'Value 6 has no larger neighbor, so path_from(1,2)=1.', state(1, 2, {
    memo: '[.,.,.; .,.,1]',
    arithmetic: 'best(6) = 1',
  }, 1), 'cache-6'),
  frame('Return and cache 5', 'The recursive result from 6 gives best(5)=max(1,1+1)=2.', state(0, 2, {
    memo: '[.,.,2; .,.,1]',
    arithmetic: 'L(0,2) = max(1, 1 + memo[1,2]) = 2',
  }, 2), 'cache-5'),
  frame('Return through 4 and 3', 'Cache best(4)=1+2=3, then best(3)=1+3=4.', state(0, 0, {
    memo: '[4,3,2; .,.,1]',
    arithmetic: 'best(4)=3; best(3)=4',
  }, 4), 'cache-3'),
  frame('Solve value 2 with a cache hit', 'The outer scan reaches (1,0)=2. Its larger neighbor 3 is cached at 4, so best(2)=1+4=5.', state(1, 0, {
    memo: '[4,3,2; 5,.,1]',
    cacheHit: 'memo[0,0] = 4',
    arithmetic: 'best(2) = max(1, 1 + 4) = 5',
  }, 5), 'cache-2'),
  frame('Solve the three choices at 1', 'Value 1 can extend through 4, 6 or 2. Their candidate lengths are four, two and six. Choose six.', state(1, 1, {
    memo: '[4,3,2; 5,6,1]',
    dependencies: '1 -> 4, 1 -> 6, 1 -> 2',
    arithmetic: 'max(1, 1+3, 1+1, 1+5) = 6',
  }, 6), 'cache-1'),
  frame('Take the maximum over all starts', 'All six cells are memoized once. The maximum length is six, following values 1,2,3,4,5,6.', state(1, 1, {
    memo: '[4,3,2; 5,6,1]',
    path: '(1,1)->(1,0)->(0,0)->(0,1)->(0,2)->(1,2)',
    result: '6',
  }, 6), 'finish'),
]);

const review = {
  pattern: 'DFS dynamic programming on the acyclic graph induced by strictly increasing grid edges.',
  recognitionCue: 'Use memoized DFS when every cell asks for an optimal path through neighboring states and a strict monotone move rule prevents cycles.',
  invariant: 'Once path_from(r,c) returns, its memo value is the longest increasing path starting at that cell; every candidate is 1 plus an already correct recursively computed larger-neighbor suffix.',
  stateModel: 'Retain the immutable matrix, a cache keyed by coordinates, the current DFS call stack, and a local best initialized to one.',
  visualRationale: 'Actual coordinate-identified nodes carry the matrix values, and all seven strictly larger-neighbor arrows remain visible. Completed-return marks and the unchanged memo states separate cached suffixes from mere discovery. The original matrix/cursor inventory remains in a disclosure with exact motion identities.',
  rejectedAlternatives: [
    'Enumerating every increasing path repeats the same suffixes exponentially.',
    'Sorting cells and filling iteratively is valid but does not match the supplied DFS call order.',
    'A final heatmap hides the dependency returns and cache-hit mechanism.',
  ],
  transferLesson: 'A strict ranking function can turn implicit neighbor moves into a DAG; memoize the optimal suffix at each state and combine it from higher-ranked neighbors.',
  reviewStatus: 'reviewed',
};

export default defineVisual('longest-increasing-path-in-a-matrix', draft, review);

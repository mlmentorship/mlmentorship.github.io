import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const board = [
  ['A', 'B', 'C', 'E'],
  ['S', 'F', 'C', 'S'],
  ['A', 'D', 'E', 'E'],
];

const path = [
  [0, 0, 'A'],
  [0, 1, 'B'],
  [0, 2, 'C'],
  [1, 2, 'C'],
  [2, 2, 'E'],
  [2, 1, 'D'],
];

function searchGrid(depth, currentDepth, extra = {}) {
  const rows = board.map((row) => [...row]);
  for (const [row, col] of path.slice(0, depth)) rows[row][col] = '#';
  const marks = path.slice(0, depth).map(([row, col, char], index) => ({
    row,
    col,
    label: index === currentDepth ? `${char}: current` : `${char}: path`,
    tone: index === currentDepth ? 'focus' : 'state',
    key: index === currentDepth ? 'current-cell' : `path-${index}`,
  }));
  return grid(rows, marks, {
    word: 'ABCCED',
    matchedPrefix: path.slice(0, depth).map(([, , char]) => char).join(' -> ') || 'empty',
    callStack: path.slice(0, depth).map(([row, col, char], index) => `${char}@(${row},${col}),i=${index}`).join(' | ') || 'empty',
    ...extra,
  });
}

const draft = visual('A recursive path owns its marked cells; failed directions return false, and every accepted cell is restored before that call returns.', [
  frame(
    'Start the outer scan',
    'For the shown board and word ABCCED, the outer generator first calls search(0, 0, 0); A matches word[0].',
    grid(board, [{ row: 0, col: 0, label: 'start A', tone: 'focus', key: 'current-cell' }], {
      word: 'ABCCED',
      call: 'search(0, 0, 0)',
      check: 'board[0][0] = A = word[0]',
    }),
    'outer-start',
  ),
  frame(
    'Mark A and reject down',
    'A becomes # for this path. The first recursive direction, down to (1,0), sees S instead of word[1] = B and returns false.',
    searchGrid(1, 0, {
      attempted: 'down -> (1,0)',
      branch: 'S != B, return false',
    }),
    'mark-a-reject-down',
  ),
  frame(
    'Reject up, then accept B',
    'Up from A is out of bounds. Right reaches B at (0,1), so B is marked and index advances to 2.',
    searchGrid(2, 1, {
      rejected: 'up -> (-1,0), out of bounds',
      accepted: 'right -> B@(0,1)',
    }),
    'accept-b',
  ),
  frame(
    'Reject B branches, then accept C',
    'From B, down sees F and up is out of bounds. Right reaches C at (0,2), matching word[2].',
    searchGrid(3, 2, {
      rejected: 'down F != C; up is out of bounds',
      accepted: 'right -> C@(0,2)',
    }),
    'accept-first-c',
  ),
  frame(
    'Follow down to the second C',
    'The first direction from C at (0,2) is down. C at (1,2) matches word[3], so it joins the marked path.',
    searchGrid(4, 3, {
      attempted: 'down -> (1,2)',
      accepted: 'C = word[3]',
    }),
    'accept-second-c',
  ),
  frame(
    'Follow down to E',
    'The first direction from C at (1,2) is down. E at (2,2) matches word[4] and is marked.',
    searchGrid(5, 4, {
      attempted: 'down -> (2,2)',
      accepted: 'E = word[4]',
    }),
    'accept-e',
  ),
  frame(
    'Reject three directions from E',
    'Looking for D from E: down is out of bounds, up revisits marked #, and right sees E. All three calls return false.',
    searchGrid(5, 4, {
      rejected: 'down out of bounds; up # != D; right E != D',
      next: 'try left -> (2,1)',
    }),
    'reject-e-branches',
  ),
  frame(
    'Accept D and finish the word',
    'Left reaches D at (2,1). After D is marked, search(2,0,6) hits index == len(word) and returns true before reading a cell.',
    searchGrid(6, 5, {
      accepted: 'left -> D@(2,1)',
      baseCase: 'index 6 == len(ABCCED), return true',
    }),
    'accept-d',
  ),
  frame(
    'Restore the successful path',
    'True short-circuits the remaining directions. Each active call restores its saved character while returning true, leaving the input board unchanged.',
    grid(board, path.map(([row, col, char], index) => ({
      row,
      col,
      label: `${index + 1}:${char} restored`,
      tone: 'output',
      key: `path-${index}`,
    })), {
      restoredOrder: 'D, E, C, C, B, A',
      callStack: 'empty after unwinding',
      result: 'true',
    }),
    'restore-and-return',
  ),
]);

export default defineVisual('word-search', draft, {
  pattern: 'Depth-first backtracking over orthogonally adjacent grid cells.',
  recognitionCue: 'A word must be assembled as one neighboring-cell path, choices branch in four directions, and a cell may be used once per candidate path but may be reused by later candidates.',
  invariant: 'At search(row, col, index), every # cell is exactly one character of the current prefix word[0:index]. Before returning, the call restores its own cell, so sibling branches and later starts see the original board.',
  stateModel: 'The minimal state is row, column, word index, and the recursion stack. In-place # marks encode the path-local visited set; each frame shows the board topology and active call stack.',
  visualRationale: 'A labelled 3x4 grid directly preserves adjacency, marked ownership, the moving current cell, failed neighbor checks, call depth, and restoration. Text labels keep every state legible without color or playback.',
  rejectedAlternatives: [
    'A recursion tree shows branching but loses the physical neighbor and cell-reuse constraints.',
    'A prose call log hides which board cells are adjacent and currently unavailable.',
    'A final highlighted word path omits failed directions, short-circuiting, and restoration.',
  ],
  transferLesson: 'For path-constrained search, choose one option, mark only the state owned by that choice, recurse, then undo before returning. The same choose-explore-unchoose discipline applies to mazes, permutations, and constraint search.',
  reviewStatus: 'reviewed',
});

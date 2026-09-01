import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const initial = [
  ['1', '1', '0', '0'],
  ['0', '1', '0', '1'],
  ['1', '0', '0', '1'],
  ['1', '0', '1', '1'],
];

const visited = (cells, active, extra = {}) => {
  const rows = initial.map((row) => [...row]);
  for (const [row, col] of cells) rows[row][col] = '0';
  const marks = cells.map(([row, col]) => ({
    row,
    col,
    label: row === active?.[0] && col === active?.[1] ? 'DFS top' : 'visited',
    tone: row === active?.[0] && col === active?.[1] ? 'focus' : 'state',
    key: row === active?.[0] && col === active?.[1] ? 'pointer-dfs' : `visited-${row}-${col}`,
  }));
  return grid(rows, marks, { input: '4x4 grid; 4-neighbor land only', ...extra });
};

const draft = visual('Count each unseen land seed once, then erase its entire four-neighbor component with DFS.', [
  frame(
    'Start island 1',
    'Row-major scan reaches unseen land (0,0): increment islands to 1, change it to 0, and initialize stack=[(0,0)].',
    visited([[0, 0]], [0, 0], { stack: '[(0,0)]', islands: '1' }),
    'start-island-one',
  ),
  frame(
    'Flood island 1',
    'Popping (0,0) discovers (0,1); popping (0,1) discovers (1,1). After (1,1), no connected land remains and the stack empties.',
    visited([[0, 0], [0, 1], [1, 1]], null, { stack: 'empty', islands: '1', path: '(0,0)->(0,1)->(1,1)' }),
    'finish-island-one',
  ),
  frame(
    'Start island 2',
    'The scan skips water and reaches unseen (1,3): increment islands to 2, change it to 0, and push it.',
    visited([[0, 0], [0, 1], [1, 1], [1, 3]], [1, 3], { stack: '[(1,3)]', islands: '2' }),
    'start-island-two',
  ),
  frame(
    'Move down the second component',
    'Pop (1,3), mark neighboring (2,3) as 0 before pushing it, then pop (2,3) and push (3,3).',
    visited([[0, 0], [0, 1], [1, 1], [1, 3], [2, 3], [3, 3]], [3, 3], { stack: '[(3,3)]', islands: '2', path: '(1,3)->(2,3)->(3,3)' }),
    'flood-island-two-down',
  ),
  frame(
    'Finish island 2',
    'Pop (3,3), discover left neighbor (3,2), and process it. The stack empties with all four cells of island 2 changed to 0.',
    visited([[0, 0], [0, 1], [1, 1], [1, 3], [2, 3], [3, 3], [3, 2]], null, { stack: 'empty', islands: '2', path: '(3,3)->(3,2)' }),
    'finish-island-two',
  ),
  frame(
    'Start and flood island 3',
    'The scan next finds unseen (2,0), increments islands to 3, and DFS reaches its only land neighbor (3,0).',
    visited([[0, 0], [0, 1], [1, 1], [1, 3], [2, 3], [3, 3], [3, 2], [2, 0], [3, 0]], [3, 0], { stack: '[(3,0)]', islands: '3', path: '(2,0)->(3,0)' }),
    'flood-island-three',
  ),
  frame(
    'Return the component count',
    'After (3,0) is processed, the stack and remaining scan contain no unseen land. Exactly three DFS starts occurred.',
    visited([[0, 0], [0, 1], [1, 1], [1, 3], [2, 3], [3, 3], [3, 2], [2, 0], [3, 0]], null, { stack: 'empty', result: '3 islands' }),
    'return-three',
  ),
]);

const review = {
  pattern: 'Outer grid scan with iterative depth-first flood fill for each unseen component.',
  recognitionCue: 'The task asks for connected groups in a binary grid under four-direction adjacency, so every unseen land cell identifies exactly one not-yet-counted component.',
  invariant: 'Before the scan advances, every visited land cell has been changed to 0; each stack contains only cells from the current island, and each completed island can never be counted again.',
  stateModel: 'Keep the mutable grid, island count, row-major scan coordinates, and a DFS stack. Mark neighbors when pushed so no land coordinate enters the stack twice.',
  visualRationale: 'A fixed-coordinate grid preserves adjacency while visible 1-to-0 mutations, a moving DFS-top key, stack contents, and count show both component topology and traversal state.',
  rejectedAlternatives: [
    'A count-only table hides which cells are connected and why a seed is new.',
    'A generic graph conversion adds labels and edges that the grid already expresses spatially.',
    'A single before-and-after grid skips every DFS frontier transition and component boundary.',
  ],
  transferLesson: 'For regions, blobs, and connected-component problems, scan for unseen seeds, count seeds rather than cells, and exhaustively mark each seed’s reachable component before continuing.',
  reviewStatus: 'reviewed',
};

export default defineVisual('number-of-islands', draft, review);

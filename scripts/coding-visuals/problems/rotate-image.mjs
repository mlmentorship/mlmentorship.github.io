import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const keyedGrid = (rows, active, extra = {}) => {
  const marks = rows.flatMap((row, rowIndex) => row.map((value, colIndex) => ({
    row: rowIndex,
    col: colIndex,
    label: active.includes(value) ? `move ${value}` : `value ${value}`,
    tone: active.includes(value) ? 'focus' : 'neutral',
    key: `matrix-value-${value}`,
  })));
  return grid(rows, marks, extra);
};

const draft = visual('Reverse row order, then swap every above-diagonal cell with its reflected below-diagonal partner.', [
  frame('Start with the original coordinates', 'The 3x3 input is [[1,2,3],[4,5,6],[7,8,9]] before any in-place mutation.', keyedGrid(
    [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']],
    [],
    { action: 'initialize' },
  ), 'initialize'),
  frame('Reverse the row order', 'matrix.reverse() moves row [7,8,9] to the top and [1,2,3] to the bottom.', keyedGrid(
    [['7', '8', '9'], ['4', '5', '6'], ['1', '2', '3']],
    ['1', '2', '3', '7', '8', '9'],
    { action: 'row 2 <-> row 0' },
  ), 'reverse-rows'),
  frame('Swap coordinates (0,1) and (1,0)', 'The transpose loop first swaps 8 with 4, fixing output positions (0,1) and (1,0).', keyedGrid(
    [['7', '4', '9'], ['8', '5', '6'], ['1', '2', '3']],
    ['8', '4'],
    { action: '(0,1)=8 <-> (1,0)=4' },
  ), 'transpose-zero-one'),
  frame('Swap coordinates (0,2) and (2,0)', 'Next, 9 and 1 cross the main diagonal.', keyedGrid(
    [['7', '4', '1'], ['8', '5', '6'], ['9', '2', '3']],
    ['9', '1'],
    { action: '(0,2)=9 <-> (2,0)=1' },
  ), 'transpose-zero-two'),
  frame('Swap coordinates (1,2) and (2,1)', 'The final above-diagonal pair swaps 6 with 2; diagonal values 7,5,3 never move.', keyedGrid(
    [['7', '4', '1'], ['8', '5', '2'], ['9', '6', '3']],
    ['6', '2'],
    { action: '(1,2)=6 <-> (2,1)=2' },
  ), 'transpose-one-two'),
  frame('Read the clockwise rotation', 'Every original (r,c) now occupies (c,2-r), producing the 90-degree clockwise image in place.', keyedGrid(
    [['7', '4', '1'], ['8', '5', '2'], ['9', '6', '3']],
    [],
    { mapping: '(r,c) -> (c,2-r)', result: '[[7,4,1],[8,5,2],[9,6,3]]' },
  ), 'finish'),
]);

const review = {
  pattern: 'In-place geometric composition: vertical reflection followed by main-diagonal transpose.',
  recognitionCue: 'Use this transformation for a square matrix that must rotate 90 degrees clockwise without allocating a second matrix.',
  invariant: 'After row reversal, values have moved from (r,c) to (n-1-r,c); after transposition, each reaches (c,n-1-r), exactly its clockwise coordinate.',
  stateModel: 'The matrix itself is the only storage; reverse its row list, then visit each pair strictly above the diagonal once and swap it with (col,row).',
  visualRationale: 'A coordinate-stable grid assigns each numeric value a persistent motion key, so rows visibly reflect and each transpose pair crosses the diagonal in source loop order.',
  rejectedAlternatives: [
    'Writing into a second matrix makes coordinate mapping simple but violates O(1) space.',
    'Four-way layer cycles are valid but more complex than the supplied reverse-then-transpose proof.',
    'Showing only before and after matrices hides the three decisive in-place swaps.',
  ],
  transferLesson: 'Derive a target coordinate mapping, then factor it into simple reversible transforms whose in-place loops visit each swap pair exactly once.',
  reviewStatus: 'reviewed',
};

export default defineVisual('rotate-image', draft, review);

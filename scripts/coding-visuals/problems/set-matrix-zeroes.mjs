import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const cursor = (row, col, label = 'scan') => [{ row, col, label, tone: 'focus', key: 'matrix-cursor' }];
const matrix = (rows, row, col, extra = {}) => grid(rows, cursor(row, col), {
  coordinates: 'row,column',
  ...extra,
});

const draft = visual('Preserve first-row and first-column facts, store interior zero instructions in those borders, then apply markers before restoring the borders.', [
  frame('Inspect the original borders', 'For [[1,0,3],[4,5,6],[7,8,0]], the first row contains 0 but the first column does not.', matrix(
    [['1', '0', '3'], ['4', '5', '6'], ['7', '8', '0']],
    0,
    1,
    { firstRowZero: 'true', firstColZero: 'false', action: 'save border facts before overwriting markers' },
  ), 'inspect-borders'),
  frame('Scan the first interior row', 'Cells (1,1)=5 and (1,2)=6 are nonzero, so row marker (1,0)=4 is unchanged.', matrix(
    [['1', '0', '3'], ['4', '5', '6'], ['7', '8', '0']],
    1,
    2,
    { firstRowZero: 'true', firstColZero: 'false', action: 'no interior marker written in row 1' },
  ), 'scan-row-one'),
  frame('Mark the interior zero', 'Cell (2,2)=0 writes row marker matrix[2][0]=0 and column marker matrix[0][2]=0.', matrix(
    [['1', '0', '0'], ['4', '5', '6'], ['0', '8', '0']],
    2,
    2,
    { action: '(2,2)=0 -> write markers (2,0) and (0,2)' },
  ), 'mark-row-two-col-two'),
  frame('Fill row 1 from column markers', 'At (1,1), matrix[0][1]=0; at (1,2), matrix[0][2]=0. Both interior cells become 0.', matrix(
    [['1', '0', '0'], ['4', '0', '0'], ['0', '8', '0']],
    1,
    2,
    { action: 'column markers zero (1,1) and (1,2)' },
  ), 'fill-row-one'),
  frame('Fill row 2 from its row marker', 'matrix[2][0]=0, so (2,1) and (2,2) are zeroed regardless of their column markers.', matrix(
    [['1', '0', '0'], ['4', '0', '0'], ['0', '0', '0']],
    2,
    1,
    { action: 'row marker zeroes row 2 interior' },
  ), 'fill-row-two'),
  frame('Restore the first row', 'first_row_zero is true, so replace row 0 with [0,0,0]. first_col_zero is false, so no separate column pass runs.', matrix(
    [['0', '0', '0'], ['4', '0', '0'], ['0', '0', '0']],
    0,
    0,
    { action: 'zero first row; skip first-column pass', result: '[[0,0,0],[4,0,0],[0,0,0]]' },
  ), 'restore-borders'),
]);

const review = {
  pattern: 'In-place matrix marking with the first row and first column as storage.',
  recognitionCue: 'Use border markers when entire rows and columns must change from discovered cells, but the required extra space must remain constant.',
  invariant: 'After the marker pass, matrix[r][0]=0 means interior row r must be zeroed and matrix[0][c]=0 means interior column c must be zeroed; saved booleans independently preserve the original first-border obligations.',
  stateModel: 'Retain two booleans plus marker cells already inside the matrix; process interior cells first, apply interior markers second, and mutate the first row and column only at the end.',
  visualRationale: 'A coordinate-preserving grid shows real cells serving as both data and marker storage; the stable matrix-cursor moves through each phase while explicit labels distinguish saved border facts from marker writes.',
  rejectedAlternatives: [
    'Separate row and column sets are simpler but violate the supplied O(1) extra-space mechanism.',
    'Immediately zeroing a row while scanning destroys information needed to discover later original zeroes.',
    'A prose table cannot show the dual role of first-border cells as matrix data and marker memory.',
  ],
  transferLesson: 'When output overwrites input, identify safe in-place metadata cells and separately save any original facts those cells represented before reusing them.',
  reviewStatus: 'reviewed',
};

export default defineVisual('set-matrix-zeroes', draft, review);

import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const marks = (row, col) => [
  { row, col, label: 'current', tone: 'focus', key: 'current-cell' },
  { row: row - 1, col, label: 'above', tone: 'state', key: 'above-dependency' },
  { row, col: col - 1, label: 'left', tone: 'state', key: 'left-dependency' },
];

const draft = visual('Sweep a 3 by 3 grid row by row; each interior count is old ways[col] from above plus new ways[col-1] from the left.', [
  frame('Initialize the first row', 'For a 3 by 3 grid, ways = [1,1,1] because each top-row cell has exactly one all-right path from the start.', grid([
    ['1', '1', '1'],
    ['?', '?', '?'],
    ['?', '?', '?'],
  ], [{ row: 0, col: 0, label: 'start/current', tone: 'state', key: 'current-cell' }], {
    rollingRow: '[1,1,1]',
  }), 'initialize'),
  frame('Update row 1, column 1', 'The old ways[1] = 1 is from above and new ways[0] = 1 is from the left, so ways[1] becomes 1 + 1 = 2.', grid([
    ['1', '1', '1'],
    ['1', '2', '?'],
    ['?', '?', '?'],
  ], marks(1, 1), {
    recurrence: 'above 1 + left 1 = 2',
    rollingRow: '[1,2,1]',
  }), 'row-one-col-one'),
  frame('Update row 1, column 2', 'The old ways[2] = 1 is from above and ways[1] = 2 is from the left, so ways[2] becomes 1 + 2 = 3.', grid([
    ['1', '1', '1'],
    ['1', '2', '3'],
    ['?', '?', '?'],
  ], marks(1, 2), {
    recurrence: 'above 1 + left 2 = 3',
    rollingRow: '[1,2,3]',
  }), 'row-one-col-two'),
  frame('Update row 2, column 1', 'The old ways[1] = 2 is from above and the first-column value is 1, so ways[1] becomes 2 + 1 = 3.', grid([
    ['1', '1', '1'],
    ['1', '2', '3'],
    ['1', '3', '?'],
  ], marks(2, 1), {
    recurrence: 'above 2 + left 1 = 3',
    rollingRow: '[1,3,3]',
  }), 'row-two-col-one'),
  frame('Reach the destination', 'At row 2, column 2, old ways[2] = 3 comes from above and new ways[1] = 3 from the left, so 3 + 3 = 6.', grid([
    ['1', '1', '1'],
    ['1', '2', '3'],
    ['1', '3', '6'],
  ], marks(2, 2), {
    recurrence: 'above 3 + left 3 = 6',
    rollingRow: '[1,3,6]',
    result: '6',
  }), 'destination'),
]);

const review = {
  pattern: 'Grid path-counting dynamic programming compressed to one row and updated left to right.',
  recognitionCue: 'Use it when movement forms an acyclic grid and every path into a cell must arrive through a small fixed set of predecessor directions such as above and left.',
  invariant: 'During a row sweep, ways[col] before update is the count from above and ways[col-1] after update is the count from the left; their sum is the current cell count.',
  stateModel: 'The implementation needs only a cols-length ways array and row/column indices. The conceptual grid retains spatial dependencies while each frame also shows the exact rolling row.',
  visualRationale: 'A real grid is simpler than a prose table because it locates above and left predecessors geometrically. Labels and equations keep the mechanism clear in print and monochrome.',
  rejectedAlternatives: [
    'A paths enumeration tree was rejected because it grows exponentially and repeats grid positions.',
    'A one-row array alone was rejected because it hides which value means above and which means left.',
    'A combinatorial binomial formula was rejected because it does not explain the supplied DP implementation.',
  ],
  transferLesson: 'When dependencies come from the previous row and current row prefix, sweep in the direction that preserves both and compress the grid to one mutable row.',
  reviewStatus: 'reviewed',
};

export default defineVisual('unique-paths', draft, review);

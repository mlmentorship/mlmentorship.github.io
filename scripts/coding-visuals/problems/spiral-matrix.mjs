import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const values = [['1', '2', '3', '4'], ['5', '6', '7', '8'], ['9', '10', '11', '12']];
const matrix = (row, col, extra = {}) => grid(values, [
  { row, col, label: 'read cursor', tone: 'focus', key: 'spiral-cursor' },
], { ...extra });

const draft = visual('Read one side of the current rectangle at a time, shrink that boundary, and guard sides that may have disappeared.', [
  frame('Initialize the outer rectangle', 'For the 3x4 matrix, start top=0, bottom=2, left=0, right=3 with an empty answer.', matrix(0, 0, {
    boundaries: 'top=0 bottom=2 left=0 right=3',
    answer: '[]',
  }), 'initialize'),
  frame('Read the top edge left to right', 'Append row 0 from columns 0 through 3: 1,2,3,4; then increment top to 1.', matrix(0, 3, {
    direction: '(0,0) -> (0,3)',
    boundaries: 'top=1 bottom=2 left=0 right=3',
    answer: '[1,2,3,4]',
  }), 'read-top'),
  frame('Read the right edge downward', 'Append column 3 from rows 1 through 2: 8,12; then decrement right to 2.', matrix(2, 3, {
    direction: '(1,3) -> (2,3)',
    boundaries: 'top=1 bottom=2 left=0 right=2',
    answer: '[1,2,3,4,8,12]',
  }), 'read-right'),
  frame('Read the bottom edge right to left', 'top<=bottom, so append row 2 from columns 2 through 0 in reverse: 11,10,9; then bottom becomes 1.', matrix(2, 0, {
    direction: '(2,2) -> (2,0)',
    boundaries: 'top=1 bottom=1 left=0 right=2',
    answer: '[1,2,3,4,8,12,11,10,9]',
  }), 'read-bottom'),
  frame('Read the left edge upward', 'left<=right, so append column 0 from row 1 up to row 1: 5; then increment left to 1.', matrix(1, 0, {
    direction: '(1,0) -> (1,0)',
    boundaries: 'top=1 bottom=1 left=1 right=2',
    answer: '[1,2,3,4,8,12,11,10,9,5]',
  }), 'read-left'),
  frame('Read the inner top edge', 'The remaining rectangle is row 1, columns 1..2. Append 6,7 and increment top to 2.', matrix(1, 2, {
    direction: '(1,1) -> (1,2)',
    boundaries: 'top=2 bottom=1 left=1 right=2',
    answer: '[1,2,3,4,8,12,11,10,9,5,6,7]',
  }), 'read-inner-top'),
  frame('Skip vanished sides and stop', 'The right-edge loop is empty and makes right=1; top>bottom skips the bottom, then the while condition fails.', matrix(1, 1, {
    boundaries: 'top=2 bottom=1 left=2 right=1',
    guard: 'top>bottom; no row remains',
    result: '[1,2,3,4,8,12,11,10,9,5,6,7]',
  }), 'finish'),
]);

const review = {
  pattern: 'Four shrinking boundaries around an unvisited matrix rectangle.',
  recognitionCue: 'Use boundary peeling when a rectangular grid must be traversed layer by layer around its perimeter in directional order.',
  invariant: 'At each while-loop entry, every cell outside top..bottom and left..right has been emitted exactly once, and every unvisited cell lies inside that closed rectangle.',
  stateModel: 'Maintain answer plus top, bottom, left, and right; each side traversal consumes one boundary and moves it inward, with guards before bottom and left prevent rereading a collapsed row or column.',
  visualRationale: 'The unchanged 3x4 grid preserves geometry while a stable cursor follows each directed edge; visible bounds and answer prefixes explain both movement and collapse guards without color.',
  rejectedAlternatives: [
    'A flattened list shows output order but erases the rectangular boundary geometry.',
    'A visited-cell simulation adds O(rows*cols) memory and uses a different turning rule.',
    'Showing only one square layer misses the asymmetric collapse guards required by rectangular matrices.',
  ],
  transferLesson: 'For perimeter traversals, represent remaining work as explicit inclusive bounds, consume one edge, shrink it immediately, and guard later edges because earlier moves may collapse the region.',
  reviewStatus: 'reviewed',
};

export default defineVisual('spiral-matrix', draft, review);

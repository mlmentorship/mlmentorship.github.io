import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const values = ['-9', '-3', '0', '4', '7', '11', '18', '25'];

const draft = visual('Preserve the inclusive sorted interval that can still contain target 11.', [
  frame(
    'Initialize the candidate interval',
    'For target 11, left=0 and right=7 keep every index; middle=floor((0+7)/2)=3, whose value is 4.',
    array(values, [
      mark(0, 'left', 'state', 'pointer-left'),
      mark(3, 'middle', 'focus', 'pointer-middle'),
      mark(7, 'right', 'state', 'pointer-right'),
    ], { target: '11', interval: '[0, 7]', comparison: '4 < 11' }),
    'initialize-interval',
  ),
  frame(
    'Discard indices 0 through 3',
    'Because the array is sorted and nums[3]=4 < 11, none of indices 0..3 can match. Set left=middle+1=4; the new middle is floor((4+7)/2)=5.',
    array(values, [
      mark(4, 'left', 'state', 'pointer-left'),
      mark(5, 'middle', 'focus', 'pointer-middle'),
      mark(7, 'right', 'state', 'pointer-right'),
    ], { target: '11', interval: '[4, 7]', comparison: '11 = 11', discarded: 'indices 0..3' }),
    'discard-lower-half',
  ),
  frame(
    'Return the matching index',
    'nums[5]=11 equals the target, so the loop returns index 5 without probing any discarded index.',
    array(values, [
      mark(4, 'left', 'state', 'pointer-left'),
      mark(5, 'middle = target', 'output', 'pointer-middle'),
      mark(7, 'right', 'state', 'pointer-right'),
    ], { target: '11', result: 'index 5' }),
    'return-index',
  ),
]);

const review = {
  pattern: 'Binary search over a sorted index interval.',
  recognitionCue: 'The input is sorted and the task asks for an exact target position, so comparing one middle value can order the target relative to an entire half.',
  invariant: 'Before every probe, if target 11 exists, its index is inside the inclusive interval [left, right]; every removed index is provably too small or too large.',
  stateModel: 'Keep only left, right, and the derived middle index. The array and target never change; left or right moves strictly past middle after a miss.',
  visualRationale: 'An indexed array with stable left, middle, and right pointers directly shows the surviving interval and the coordinates skipped by sorted order.',
  rejectedAlternatives: [
    'A decision tree adds topology that the iterative implementation does not store.',
    'A prose table hides the physical interval removed by each comparison.',
    'An unindexed number line obscures that the function returns an index rather than a value.',
  ],
  transferLesson: 'For lower-bound, upper-bound, and insertion-position variants, first define what the interval promises, then choose whether middle is excluded or retained after each comparison.',
  reviewStatus: 'reviewed',
};

export default defineVisual('binary-search', draft, review);

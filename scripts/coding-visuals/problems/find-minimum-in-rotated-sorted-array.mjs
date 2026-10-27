import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const values = ['6', '7', '8', '1', '2', '3', '4', '5'];
const fitted = (items, marks, extra) => array(items, marks, { fitArray: true, arrayColumns: 8, ...extra });

const draft = visual('Retain the minimum by comparing middle with the current right boundary.', [
  frame(
    'Keep the midpoint',
    'left=0, right=7, middle=3. The comparison 1 < 5 permits a midpoint minimum, so set right=3.',
    fitted(values, [
      mark(0, 'L', 'state', 'pointer-left'),
      mark(3, 'm', 'focus', 'pointer-middle'),
      mark(7, 'R', 'state', 'pointer-right'),
    ], { interval: '[0, 7]', comparison: '1 <= 5', decision: 'right = 3' }),
    'retain-middle',
  ),
  frame(
    'Discard the left part',
    'Now left=0, right=3, middle=1. Since 7 > 1, the drop follows middle, so set left=2.',
    fitted(values, [
      mark(0, 'L', 'state', 'pointer-left'),
      mark(1, 'm', 'focus', 'pointer-middle'),
      mark(3, 'R', 'state', 'pointer-right'),
    ], { interval: '[0, 3]', comparison: '7 > 1', decision: 'left = 2' }),
    'discard-left-run',
  ),
  frame(
    'Discard midpoint 2',
    'Now left=2, right=3, middle=2. Since 8 > 1, index 2 cannot be minimum, so set left=3.',
    fitted(values, [
      mark(2, 'L=m', 'focus', 'pointer-middle'),
      mark(3, 'R', 'state', 'pointer-right'),
    ], { interval: '[2, 3]', comparison: '8 > 1', decision: 'left = 3' }),
    'discard-eight',
  ),
  frame(
    'Return the converged value',
    'left and right meet at index 3. The invariant leaves nums[3]=1 as the only possible minimum.',
    fitted(values, [
      mark(3, 'min', 'output', 'pointer-middle'),
    ], { interval: '[3, 3]', result: '1' }),
    'return-minimum',
  ),
]);

const review = {
  pattern: 'Binary search for a rotation pivot by comparing middle with the current right endpoint.',
  recognitionCue: 'Distinct values form two increasing runs after one rotation, and the task asks for the smallest value rather than a particular target.',
  invariant: 'The minimum is always inside inclusive [left, right]. If nums[middle] exceeds nums[right], middle is on the high run; otherwise middle may be the minimum and must be retained.',
  stateModel: 'Track only left and right indices and derive middle. The right endpoint supplies the reference run; each comparison either advances left past middle or moves right onto middle.',
  visualRationale: 'The indexed rotated array displays both increasing runs and the drop, while stable left, middle, and right pointers show why middle is excluded in one branch but retained in the other.',
  rejectedAlternatives: [
    'A circular rotation diagram makes the contiguous search interval harder to follow.',
    'A line chart overstates numeric magnitude when only ordering and the single drop matter.',
    'A comparison table hides the exact indices removed and the crucial right=middle asymmetry.',
  ],
  transferLesson: 'Pivot searches depend on choosing an endpoint reference and preserving the candidate when equality or ordering does not prove it impossible; this same boundary discipline applies to first-occurrence searches.',
  reviewStatus: 'reviewed',
};

export default defineVisual('find-minimum-in-rotated-sorted-array', draft, review);

import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const speeds = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'];

const bounds = (left, middle, right, extra = {}) => array(speeds, [
  mark(left - 1, 'left', 'state', 'pointer-left'),
  mark(middle - 1, 'speed', 'focus', 'pointer-speed'),
  mark(right - 1, 'right', 'state', 'pointer-right'),
], { input: 'piles [3,6,7,11], hours 8', ...extra });

const draft = visual('Binary-search the first speed whose ceiling-division total is at most 8 hours.', [
  frame(
    'Test the initial midpoint',
    'Initialize left=1 and right=max(piles)=11, then test speed floor((1+11)/2)=6. Hours are 1+1+2+2=6, so 6 <= 8 and right becomes 6.',
    bounds(1, 6, 11, { tested: 'k=6: ceil(3/6)+ceil(6/6)+ceil(7/6)+ceil(11/6)=1+1+2+2=6', decision: '6 <= 8: right = 6' }),
    'keep-feasible-six',
  ),
  frame(
    'Speed 3 is too slow',
    'At speed 3, hours are 1+2+3+4=10. Since 10 > 8, speeds 1..3 are infeasible; set left=3+1=4 and next test speed 5.',
    bounds(1, 3, 6, { tested: 'k=3: ceil(3/3)+ceil(6/3)+ceil(7/3)+ceil(11/3)=1+2+3+4=10', decision: '10 > 8: left = 4' }),
    'discard-through-three',
  ),
  frame(
    'Speed 5 is feasible',
    'At speed 5, hours are 1+2+2+3=8. Since 8 <= 8, speed 5 may be the first feasible speed, so set right=5 and test speed 4.',
    bounds(4, 5, 6, { tested: 'k=5: ceil(3/5)+ceil(6/5)+ceil(7/5)+ceil(11/5)=1+2+2+3=8', decision: '8 <= 8: right = 5' }),
    'keep-feasible-five',
  ),
  frame(
    'Speed 4 is feasible',
    'At speed 4, hours are 1+2+2+3=8. Keep this feasible candidate by setting right=4; now left and right meet.',
    bounds(4, 4, 5, { tested: 'k=4: ceil(3/4)+ceil(6/4)+ceil(7/4)+ceil(11/4)=1+2+2+3=8', decision: '8 <= 8: right = 4' }),
    'keep-feasible-four',
  ),
  frame(
    'Return the first feasible speed',
    'The interval has converged at speed 4. Speed 3 needed 10 hours, so 4 is not merely feasible; it is the minimum feasible speed.',
    array(speeds, [
      mark(3, 'left = right = answer', 'output', 'pointer-speed'),
    ], { boundary: 'speed 3 false; speed 4 true', result: '4 bananas/hour' }),
    'return-minimum-speed',
  ),
]);

const review = {
  pattern: 'Binary search on a monotone answer predicate, seeking the first feasible speed.',
  recognitionCue: 'The answer is numeric, bounded from 1 to max(piles), and feasibility changes only once: if speed k finishes in time, every faster speed also finishes.',
  invariant: 'The minimum feasible speed always remains in inclusive [left, right]. An infeasible midpoint and every slower speed are removed; a feasible midpoint is retained as right.',
  stateModel: 'Keep left and right speed bounds, derive the tested speed, and evaluate total hours with ceiling division for each pile; no per-pile state persists between tests.',
  visualRationale: 'An ordered speed line with stable bound and probe pointers exposes the false-to-true boundary, while each frame prints the real piles and complete ceiling-hour arithmetic.',
  rejectedAlternatives: [
    'Animating bananas being eaten emphasizes simulation rather than the monotone feasibility boundary.',
    'A pile-height bar chart shows the input but not the ordered answer space being searched.',
    'A feasibility table without moving bounds lists calculations but hides why half the speeds are safely discarded.',
  ],
  transferLesson: 'For capacity, rate, and threshold problems, define a monotone yes/no test, choose bounds that contain the answer, and retain the midpoint on the side that may hold the first true value.',
  reviewStatus: 'reviewed',
};

export default defineVisual('koko-eating-bananas', draft, review);

import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const nums = ['-2', '1', '-3', '4', '-1', '2', '1', '-5', '4'];
const state = (index, start, extra = {}) => array(nums, [
  mark(start, 'current start', 'state', 'current-start'),
  mark(index, 'scan', 'focus', 'scan-cursor'),
], { ...extra });

const draft = visual('At each index, keep the best sum ending exactly there by either extending the prior range or restarting at the current value.', [
  frame('Initialize at index 0', 'The only nonempty subarray ending at -2 is [-2], so current=-2 and best=-2.', state(0, 0, {
    currentRange: '[0..0]',
    arithmetic: 'current = best = -2',
  }), 'initialize'),
  frame('Restart at 1', 'At index 1, max(1, -2 + 1 = -1) chooses 1, so the current range restarts at index 1 and best becomes 1.', state(1, 1, {
    currentRange: '[1..1]',
    arithmetic: 'max(1, -1) = 1; best = 1',
  }), 'restart-one'),
  frame('Extend through -3', 'At index 2, max(-3, 1 + -3 = -2) chooses -2, so current range [1..2] is retained; best stays 1.', state(2, 1, {
    currentRange: '[1..2]',
    arithmetic: 'max(-3, -2) = -2; best = 1',
  }), 'extend-minus-three'),
  frame('Restart at 4', 'At index 3, max(4, -2 + 4 = 2) chooses 4; carrying the negative prefix would be worse.', state(3, 3, {
    currentRange: '[3..3]',
    arithmetic: 'max(4, 2) = 4; best = 4',
  }), 'restart-four'),
  frame('Extend through -1', 'At index 4, max(-1, 4 + -1 = 3) chooses 3, so [4,-1] remains worth carrying.', state(4, 3, {
    currentRange: '[3..4]',
    arithmetic: 'max(-1, 3) = 3; best = 4',
  }), 'extend-minus-one'),
  frame('Extend through 2', 'At index 5, max(2, 3 + 2 = 5) chooses 5, producing a new global best over [3..5].', state(5, 3, {
    currentRange: '[3..5]',
    arithmetic: 'max(2, 5) = 5; best = 5',
  }), 'extend-two'),
  frame('Extend through 1', 'At index 6, max(1, 5 + 1 = 6) chooses 6; [4,-1,2,1] becomes the best range.', state(6, 3, {
    currentRange: '[3..6]',
    arithmetic: 'max(1, 6) = 6; best = 6',
  }), 'extend-one'),
  frame('Absorb -5 without losing best', 'At index 7, max(-5, 6 + -5 = 1) chooses 1; current falls, but global best remains 6.', state(7, 3, {
    currentRange: '[3..7]',
    arithmetic: 'max(-5, 1) = 1; best = 6',
  }), 'extend-minus-five'),
  frame('Finish at 4', 'At index 8, max(4, 1 + 4 = 5) chooses 5, which does not beat the saved best 6.', state(8, 3, {
    currentRange: '[3..8]',
    arithmetic: 'max(4, 5) = 5; best = 6',
    result: '6 from indices [3..6] = [4,-1,2,1]',
  }), 'finish'),
]);

const review = {
  pattern: 'One-dimensional dynamic programming with a best-ending-here state (Kadane algorithm).',
  recognitionCue: 'Use this recurrence when optimizing a nonempty contiguous subarray and extending a negative accumulated prefix can only hurt every future range.',
  invariant: 'After processing index i, current is the maximum sum of any nonempty subarray ending exactly at i, while best is the maximum sum of any subarray contained in indices 0 through i.',
  stateModel: 'Only current and best are required for the value; the visual additionally tracks current start and best endpoints to expose contiguous geometry without changing the supplied recurrence.',
  visualRationale: 'The full indexed array remains fixed while stable scan and current-start pointers show the covered range; each frame displays both recurrence candidates and the chosen sum.',
  rejectedAlternatives: [
    'A DP table lists numbers but hides which contiguous input range each state represents.',
    'A prefix-sum chart can solve the problem through minimum prefixes but does not match the supplied ending-here recurrence.',
    'Highlighting only the final [4,-1,2,1] skips the restart decisions that explain correctness.',
  ],
  transferLesson: 'For contiguous optimization, define the best solution forced to end at the current position, decide extend versus restart, and separately preserve the best endpoint seen anywhere.',
  independentReview: '3.4 source-to-frame replay',
  reviewStatus: 'reviewed',
};

export default defineVisual('maximum-subarray', draft, review);

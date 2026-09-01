import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const nums = ['2', '3', '-2', '4', '-1'];
const state = (index, extra = {}) => array(nums, [
  mark(index, 'scan', 'focus', 'scan-cursor'),
], { processedPrefix: `[0..${index}]`, ...extra });

const draft = visual('Track both extreme products ending at each index because multiplying by a negative swaps which extreme can become largest.', [
  frame('Initialize both extremes', 'At index 0, the only ending product is 2, so current_max=current_min=best=2.', state(0, {
    endingRange: '[2]',
    arithmetic: 'max=2; min=2; best=2',
  }), 'initialize'),
  frame('Extend through 3', 'At index 1, 3 is positive: current_max=max(3,2*3)=6 and current_min=min(3,2*3)=3.', state(1, {
    endingRange: 'max [2,3]; min [3]',
    arithmetic: 'max(3,6)=6; min(3,6)=3; best=6',
  }), 'multiply-three'),
  frame('Swap before multiplying by -2', 'Because -2 is negative, swap the prior extremes: max source becomes 3 and min source becomes 6.', state(2, {
    action: 'swap current_max=6 and current_min=3',
    sourcesAfterSwap: 'max source 3; min source 6',
  }), 'swap-minus-two'),
  frame('Update both products at -2', 'After the swap, max(-2,3*-2=-6)=-2 and min(-2,6*-2=-12)=-12; best remains 6.', state(2, {
    endingRange: 'max [-2]; min [2,3,-2]',
    arithmetic: 'max(-2,-6)=-2; min(-2,-12)=-12; best=6',
  }), 'multiply-minus-two'),
  frame('Process positive 4', 'At index 3, max(4,-2*4=-8)=4 restarts, while min(4,-12*4=-48)=-48 extends the negative product.', state(3, {
    endingRange: 'max [4]; min [2,3,-2,4]',
    arithmetic: 'max(4,-8)=4; min(4,-48)=-48; best=6',
  }), 'multiply-four'),
  frame('Swap before multiplying by -1', 'Because -1 is negative, the prior minimum -48 becomes the source for current_max and prior maximum 4 becomes the source for current_min.', state(4, {
    action: 'swap current_max=4 and current_min=-48',
    sourcesAfterSwap: 'max source -48; min source 4',
  }), 'swap-minus-one'),
  frame('Turn the minimum into the maximum', 'max(-1,-48*-1=48)=48 and min(-1,4*-1=-4)=-4; best becomes 48.', state(4, {
    endingRange: 'max [2,3,-2,4,-1]; min [4,-1]',
    arithmetic: 'max(-1,48)=48; min(-1,-4)=-4; best=48',
    result: '48 from [2,3,-2,4,-1]',
  }), 'multiply-minus-one'),
]);

const review = {
  pattern: 'One-dimensional DP with maximum and minimum products ending at the current index.',
  recognitionCue: 'Use paired extremes for contiguous products when negative values can reverse ordering and zero or a single value may force a restart.',
  invariant: 'After each index, current_max and current_min are respectively the largest and smallest products of nonempty subarrays ending exactly there, and best is the largest current_max seen in the processed prefix.',
  stateModel: 'Keep current_max, current_min, and best; before a negative multiplier, swap the two ending extremes, then independently choose restart-at-num versus extend-by-multiplication.',
  visualRationale: 'The fixed indexed array and stable scan cursor preserve contiguous geometry, while separate swap and multiply frames make the sign reversal and every candidate product explicit.',
  rejectedAlternatives: [
    'Tracking only the current maximum loses a large negative product that a later negative can turn positive.',
    'Prefix and suffix product passes are another approach but do not match the supplied constant-state recurrence.',
    'A final highlighted range cannot explain the essential extreme swap at negative values.',
  ],
  transferLesson: 'When a transition is not monotone, retain every extreme that can become optimal after the next operation; sign-changing multiplication is the canonical max/min pair example.',
  independentReview: '3.4 source-to-frame replay',
  reviewStatus: 'reviewed',
};

export default defineVisual('maximum-product-subarray', draft, review);

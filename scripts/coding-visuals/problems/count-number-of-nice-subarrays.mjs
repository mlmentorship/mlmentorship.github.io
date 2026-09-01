import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const items = ['1', '1', '2', '1', '1'];
const pointers = (leftThree, leftTwo, right) => [
  mark(leftThree, 'L<=3', 'state', 'left-at-most-3'),
  mark(leftTwo, 'L<=2', 'state', 'left-at-most-2'),
  mark(right, 'R', 'focus', 'right'),
];

const draft = visual('Count endings with at most 3 odds and at most 2 odds at every R, then subtract 14 - 12.', [
  frame('Read the first odd at R = 0', 'Both windows start at 0. Each has one valid ending at R, so totals become at_most(3) = 1 and at_most(2) = 1.', array(items, pointers(0, 0, 0), {
    ranges: '<=3 [0..0]; <=2 [0..0]',
    additions: '1 and 1',
    totals: '1 and 1',
  }), 'right-0'),
  frame('Read the second odd at R = 1', 'Both budgets still fit. Each left boundary stays 0, adding R - L + 1 = 2 endings; totals become 3 and 3.', array(items, pointers(0, 0, 1), {
    ranges: '<=3 [0..1]; <=2 [0..1]',
    additions: '2 and 2',
    totals: '3 and 3',
  }), 'right-1'),
  frame('Read the even value at R = 2', 'The even value spends no odd budget. Both L pointers stay 0 and each window adds 2 - 0 + 1 = 3 endings; totals become 6 and 6.', array(items, pointers(0, 0, 2), {
    ranges: '<=3 [0..2]; <=2 [0..2]',
    additions: '3 and 3',
    totals: '6 and 6',
  }), 'right-2'),
  frame('Third odd forces only L<=2 right', 'At R = 3 the <=3 window still starts at 0 and adds 4. The <=2 budget goes negative, so remove nums[0] = 1 and move L<=2 to 1, adding 3.', array(items, pointers(0, 1, 3), {
    ranges: '<=3 [0..3]; <=2 [1..3]',
    direction: 'L<=3 stays 0; L<=2: 0 -> 1',
    additions: '4 and 3',
    totals: '10 and 9',
  }), 'right-3'),
  frame('Fourth odd advances both boundaries', 'At R = 4, remove nums[0] = 1 for <=3 and nums[1] = 1 for <=2. The windows add 4 and 3 endings, producing 14 - 12 = 2 exact matches.', array(items, pointers(1, 2, 4), {
    ranges: '<=3 [1..4]; <=2 [2..4]',
    direction: 'L<=3: 0 -> 1; L<=2: 1 -> 2',
    additions: '4 and 3',
    totals: '14 and 12',
    result: '14 - 12 = 2',
  }), 'right-4-result'),
]);

const review = {
  pattern: 'Exact-count subarrays obtained by subtracting two variable-size at-most sliding-window totals.',
  recognitionCue: 'Use it when subarrays must contain exactly k nonnegative events, while an at-most-k window can be repaired monotonically by moving its left boundary right.',
  invariant: 'For each R, every start from L through R satisfies the relevant at-most limit, so that pass adds exactly R - L + 1 valid endings. The running total includes all right endpoints processed so far.',
  stateModel: 'Each pass needs only L, R, remaining odd budget, and total. The combined trace shows stable L<=3, L<=2, and R pointers over nums = [1,1,2,1,1] with both per-step additions.',
  visualRationale: 'One indexed array with two explicitly named left boundaries reveals the nested windows and their cumulative arithmetic. Labels, ranges, and equations remain complete in monochrome static output.',
  rejectedAlternatives: [
    'A final 14-minus-12 equation alone was rejected because it hides how either cumulative total is formed.',
    'Two disconnected animations were rejected because comparing the left boundaries at the same R exposes the exact-count difference more directly.',
    'A table-only trace was rejected because it does not show the covered ranges or why a boundary moves after an odd value.',
  ],
  transferLesson: 'Convert exact monotone counts into at_most(k) - at_most(k - 1); at each right endpoint, the difference between the two valid-start ranges counts exactly-k subarrays ending there.',
  reviewStatus: 'reviewed',
};

export default defineVisual('count-number-of-nice-subarrays', draft, review);

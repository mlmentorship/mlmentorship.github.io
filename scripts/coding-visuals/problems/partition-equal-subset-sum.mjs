import { arrayMap, defineVisual, frame, mark, visual } from '../primitives.mjs';

const nums = ['1', '5', '11', '5'];
const possibleMap = (values) => values.map((value) => [String(value), 'reachable']);

function subsetState(index, possible, additions, ignored, result) {
  return arrayMap(nums, possibleMap(possible), [mark(index, `current=${nums[index]}`, result ? 'output' : 'focus', 'current-number')], {
    mapLabel: 'possible sums at or below target 11',
    oldSetSnapshot: additions,
    ignoredAboveTarget: ignored,
    ...(result ? { result } : {}),
  });
}

const draft = visual('Each number extends a snapshot of previously reachable sums; reaching half the total proves an equal partition.', [
  frame(
    'Check the total and seed zero',
    'For nums = [1, 5, 11, 5], total = 22 is even, so target = 11. The empty subset makes only sum 0 reachable.',
    arrayMap(nums, possibleMap([0]), [mark(0, 'next=1', 'focus', 'current-number')], {
      mapLabel: 'possible sums at or below target 11',
      arithmetic: 'total=1+5+11+5=22; target=22/2=11',
      branch: '22 % 2 = 0, continue',
    }),
    'initialize',
  ),
  frame(
    'Process 1',
    'The set-comprehension snapshot is {0}. It adds 0 + 1 = 1, then union keeps both 0 and 1.',
    subsetState(0, [0, 1], 'from old {0}: 0+1=1', 'none'),
    'process-1',
  ),
  frame(
    'Process the first 5',
    'From old {0,1}, add 0 + 5 = 5 and 1 + 5 = 6. The reachable set becomes {0,1,5,6}.',
    subsetState(1, [0, 1, 5, 6], 'from old {0,1}: 0+5=5; 1+5=6', 'none'),
    'process-first-5',
  ),
  frame(
    'Process 11',
    'Only 0 + 11 stays within target. Sums 1, 5, and 6 would exceed 11, so union adds 11.',
    subsetState(2, [0, 1, 5, 6, 11], 'from old {0,1,5,6}: 0+11=11', '12, 16, 17'),
    'process-11',
  ),
  frame(
    'Process the final 5',
    'The old-set snapshot adds 5, 6, 10, and 11. Values 16 are ignored; union preserves existing sums and adds 10.',
    subsetState(3, [0, 1, 5, 6, 10, 11], '0+5=5; 1+5=6; 5+5=10; 6+5=11', '11+5=16'),
    'process-final-5',
  ),
  frame(
    'Test target membership',
    'After every input value is processed, target 11 is in possible. Subset [11] has sum 11 and the remaining [1,5,5] also sums to 11.',
    arrayMap(nums, possibleMap([0, 1, 5, 6, 10, 11]), [mark(2, 'subset sum 11', 'output', 'current-number')], {
      mapLabel: 'final possible sums',
      membership: '11 in possible -> true',
      partition: '[11] | [1,5,5]',
      arithmetic: '11 = 11',
      result: 'true',
    }),
    'return-true',
  ),
]);

export default defineVisual('partition-equal-subset-sum', draft, {
  pattern: 'Subset-sum dynamic programming represented as a set of reachable totals.',
  recognitionCue: 'The input must split into equal-sum groups, so one group must realize exactly half the total; each positive number is either included once or excluded.',
  invariant: 'After processing a prefix of nums, possible contains exactly the sums at most target realizable from that prefix. New sums are computed from an old-set snapshot, so the current number cannot be reused within its own iteration.',
  stateModel: 'Keep target and the set of reachable sums. Each transition unions old sums with old sum + current number when it does not exceed target; the original elements need no other history.',
  visualRationale: 'An input array paired with the explicit reachable-sum set shows which number moves, which old states generate new states, which sums are clipped, and the final membership proof in static text.',
  rejectedAlternatives: [
    'A final two-group picture proves one example but not how the algorithm discovers it.',
    'A recursion tree duplicates equivalent remaining-sum states and obscures the set union.',
    'A boolean matrix is correct but heavier than the one-row reachable-set state used by the supplied implementation.',
  ],
  transferLesson: 'Convert a partition condition into a target-reachability problem, then update from a snapshot when each item may be used once. The same state model solves 0/1 knapsack feasibility and constrained subset sums.',
  independentReview: '3.3 source-to-frame replay',
  reviewStatus: 'reviewed',
});

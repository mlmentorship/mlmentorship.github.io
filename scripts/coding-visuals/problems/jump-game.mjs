import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const nums = ['3', '2', '1', '0', '4'];
const example = 'nums = [3, 2, 1, 0, 4]';
const scan = (index, label) => mark(index, label, 'focus', 'scan-index');
const reach = (index) => mark(index, `farthest=${index}`, 'state', 'farthest-boundary');
const state = (index, marks, extra = {}) => array(nums, [scan(index, `index=${index}`), ...marks], { example, ...extra });

const draft = visual('Scan only reachable positions and preserve the farthest index any processed jump can reach.', [
  frame(
    'Initialize the reachable frontier',
    'Before scanning, farthest=0, so only index 0 is known reachable.',
    state(0, [reach(0)], { reachableRange: '0..0', farthest: '0' }),
    'initialize-farthest',
  ),
  frame(
    'Extend from index 0',
    'Index 0 <= farthest 0, so it is reachable. Jump 3 changes farthest to max(0, 0+3)=3.',
    state(0, [reach(3)], { check: '0 <= 0: reachable', update: 'max(0, 0 + 3) = 3', reachableRange: '0..3', farthest: '3' }),
    'extend-from-zero',
  ),
  frame(
    'Scan index 1',
    'Index 1 <= 3, so inspect jump 2. It reaches 1+2=3, leaving farthest unchanged.',
    state(1, [reach(3)], { check: '1 <= 3: reachable', update: 'max(3, 1 + 2) = 3', reachableRange: '0..3', farthest: '3' }),
    'scan-one',
  ),
  frame(
    'Scan index 2',
    'Index 2 <= 3. Its jump 1 also reaches only 3, so the frontier does not move.',
    state(2, [reach(3)], { check: '2 <= 3: reachable', update: 'max(3, 2 + 1) = 3', reachableRange: '0..3', farthest: '3' }),
    'scan-two',
  ),
  frame(
    'Scan the dead end',
    'Index 3 <= 3, but jump 0 reaches only 3. The last index 4 remains outside the reachable range.',
    state(3, [reach(3)], { check: '3 <= 3: reachable', update: 'max(3, 3 + 0) = 3', reachableRange: '0..3', farthest: '3' }),
    'scan-three',
  ),
  frame(
    'Detect the unreachable index',
    'At index 4, 4 > farthest 3. No processed position reaches index 4, so the function returns false before using nums[4].',
    state(4, [mark(3, 'farthest=3', 'warning', 'farthest-boundary')], { check: '4 > 3: unreachable', branch: 'return before update', result: 'false' }),
    'return-false',
  ),
]);

const review = {
  pattern: 'Greedy scan with a farthest-reachable frontier.',
  recognitionCue: 'Each array value is a maximum forward jump and the question asks only whether the final position is reachable, not for a path or minimum jump count.',
  invariant: 'Before index i is processed, every position through farthest is reachable using processed jumps; if i exceeds farthest, no earlier choice can reach i or anything beyond it.',
  stateModel: 'Retain the immutable jump array, moving scan index, and one scalar farthest. Individual paths and predecessor choices are irrelevant to the yes/no result.',
  visualRationale: 'Indexed cells with independently keyed scan and farthest markers show the covered range, frontier movement, and the exact failing comparison without relying on color.',
  rejectedAlternatives: [
    'A graph of every possible jump creates quadratic-looking edges and obscures the single frontier summary.',
    'Dynamic programming reachability stores one boolean per index although the reachable prefix is represented by one boundary.',
    'Showing only a successful leap path suggests committing to jumps, which the supplied greedy scan never does.',
  ],
  transferLesson: 'When all feasible positions form a prefix, summarize every prior choice by its farthest boundary; this transfers to interval coverage, refueling reach, and minimum-jump layer scans.',
  independentReview: '3.3 source-to-frame replay',
  reviewStatus: 'reviewed',
};

export default defineVisual('jump-game', draft, review);

import { arrayMap, defineVisual, frame, mark, visual } from '../primitives.mjs';

const nums = ['3', '2', '4', '7'];
const example = 'nums = [3, 2, 4, 7], target = 6';
const cursor = (index, label) => mark(index, label, 'focus', 'scan-cursor');

const draft = visual('Check each complement before saving the current value.', [
  frame(
    'Initialize the lookup',
    'Before index 0, seen is empty; the only retained state will map each scanned value to its index.',
    arrayMap(nums, [], [cursor(0, 'current i=0')], { example, target: '6', mapLabel: 'seen: value -> index' }),
    'initialize-seen',
  ),
  frame(
    'Miss on the first complement',
    'At i=0, num=3 and needed=6-3=3. The empty map has no 3, so the algorithm does not return.',
    arrayMap(nums, [], [cursor(0, 'need 3: absent')], { example, arithmetic: '6 - 3 = 3', branch: 'miss' }),
    'query-first-complement',
  ),
  frame(
    'Save the first value',
    'After the failed lookup, save 3 -> 0. Checking first is what prevents index 0 from pairing with itself.',
    arrayMap(nums, [['3', 'index 0']], [cursor(0, 'saved 3 -> 0')], { example, invariant: 'seen contains exactly indices before the next cursor' }),
    'save-first-value',
  ),
  frame(
    'Miss and save at index 1',
    'At i=1, num=2 and needed=6-2=4. Four is absent, so save 2 -> 1 and continue.',
    arrayMap(nums, [['3', 'index 0'], ['2', 'index 1']], [cursor(1, 'need 4: absent; save')], { example, arithmetic: '6 - 2 = 4', branch: 'miss, then save' }),
    'query-and-save-second-value',
  ),
  frame(
    'Find the saved complement',
    'At i=2, num=4 and needed=6-4=2. The map contains 2 -> 1, so the two distinct indices are known.',
    arrayMap(nums, [['3', 'index 0'], ['2', 'index 1']], [mark(1, 'saved complement', 'state', 'saved-complement'), cursor(2, 'current i=2')], { example, arithmetic: '6 - 4 = 2', branch: 'hit' }),
    'find-complement',
  ),
  frame(
    'Return the indices',
    'Return [seen[2], i] = [1, 2]; nums[1] + nums[2] = 2 + 4 = 6.',
    arrayMap(nums, [['3', 'index 0'], ['2', 'index 1']], [mark(1, 'answer index 1', 'output', 'saved-complement'), cursor(2, 'answer index 2')], { example, arithmetic: '2 + 4 = 6', result: '[1, 2]' }),
    'return-indices',
  ),
]);

const review = {
  pattern: 'One-pass hash map from a previously seen value to its index.',
  recognitionCue: 'The prompt asks for two positions whose values satisfy a target sum, so each current value determines one exact complement that can be looked up.',
  invariant: 'Before processing index i, seen contains exactly the useful values from indices below i; therefore a hit is an earlier, distinct element.',
  stateModel: 'Keep the input array, target, moving index i, computed needed value, and seen[value] = earlier index. No pair list or nested search is needed.',
  visualRationale: 'An indexed array beside the evolving value-to-index map makes the check-before-save order and the moving current index directly visible.',
  rejectedAlternatives: [
    'A pair-sum matrix shows quadratic combinations and obscures why one lookup replaces the inner loop.',
    'A sorted two-pointer line loses the original indices and depicts a different algorithm.',
    'A prose table can list iterations but does not visibly move the same scan cursor across stable array cells.',
  ],
  transferLesson: 'When a current item determines one exact missing partner, store only prior facts needed to recognize that partner; this transfers to complement, difference, and prefix-sum lookup problems.',
  reviewStatus: 'reviewed',
};

export default defineVisual('two-sum', draft, review);

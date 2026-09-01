import { array, arrayMap, defineVisual, frame, mark, visual } from '../primitives.mjs';

const bucketAxis = ['f=0 []', 'f=1 [3]', 'f=2 [2]', 'f=3 [1]', 'f=4 []', 'f=5 []', 'f=6 []'];

const draft = visual('Frequency is a bounded coordinate: place each value at its count, then scan counts from n down until k values are collected.', [
  frame(
    'Count the concrete input',
    'For nums = [1,1,1,2,2,3] and k = 2, Counter produces 1 -> 3, 2 -> 2, and 3 -> 1.',
    arrayMap(['1', '1', '1', '2', '2', '3'], [['1', '3'], ['2', '2'], ['3', '1']], [], { k: '2', mapLabel: 'frequency' }),
    'count-values',
  ),
  frame(
    'Place values at frequency coordinates',
    'Allocate n+1 = 7 buckets. Put value 3 at index 1, value 2 at index 2, and value 1 at index 3.',
    array(bucketAxis, [mark(6, 'scan starts', 'focus', 'frequency-scan')], { answer: '[]', direction: '6 -> 0' }),
    'build-buckets',
  ),
  frame(
    'Skip empty high frequencies',
    'Buckets 6, 5, and 4 are empty, so extending the answer changes nothing and the scan moves down to frequency 3.',
    array(bucketAxis, [mark(3, 'scan f=3', 'focus', 'frequency-scan')], { answer: '[]', skipped: 'f=6,5,4' }),
    'skip-empty-buckets',
  ),
  frame(
    'Collect the most frequent value',
    'Extend with bucket 3 = [1]. The answer is [1], whose length 1 is still below k = 2.',
    array(bucketAxis, [mark(3, 'take [1]', 'output', 'frequency-scan')], { answer: '[1]', check: '1 < 2; continue' }),
    'take-frequency-3',
  ),
  frame(
    'Collect until k is reached',
    'Move to bucket 2 = [2] and extend: [1] + [2] = [1,2]. Its length is now k, so return the first two values.',
    array(bucketAxis, [mark(2, 'take [2]', 'output', 'frequency-scan')], { answer: '[1,2]', check: '2 >= k', result: '[1, 2]' }),
    'take-frequency-2',
  ),
]);

const review = {
  pattern: 'Frequency counting followed by reverse traversal of an n-bounded bucket array.',
  recognitionCue: 'Use frequency buckets when the answer ranks values by occurrence count, counts cannot exceed the input length n, and linear time matters more than ordering values within equal-frequency buckets.',
  invariant: 'Before scanning bucket f, the answer contains every value from frequencies greater than f and no value from a lower frequency. Therefore the first k collected values are among the k most frequent.',
  stateModel: 'The minimal state is a Counter, n+1 buckets indexed by frequency, a descending frequency cursor, and the collected answer. The trace shows counting, placement, empty-bucket skips, extension, and the len(answer) >= k return branch.',
  visualRationale: 'An indexed array is the bucket geometry: position is frequency, cell content is the values with that count, and one named scan marker moves from n toward zero. Labels make frequency and direction readable without color.',
  rejectedAlternatives: [
    'A bar chart was rejected because it emphasizes magnitudes but obscures the actual n+1 bucket array used by the implementation.',
    'A min-heap was rejected because it would depict a different O(n log k) algorithm.',
    'A frequency table alone was rejected because it hides the reverse bucket scan and early-stop condition.',
  ],
  transferLesson: 'Whenever a ranking key is a small bounded integer, use that key as an array coordinate and scan coordinates in answer order; this transfers to counting sort, histogram selection, and bounded-score ranking.',
  reviewStatus: 'reviewed',
};

export default defineVisual('top-k-frequent-elements', draft, review);

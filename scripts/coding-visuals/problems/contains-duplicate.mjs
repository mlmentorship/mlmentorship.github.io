import { arrayMap, defineVisual, frame, mark, visual } from '../primitives.mjs';

const nums = ['4', '1', '4', '2'];
const scan = (index, unique, extra = {}) => arrayMap(
  nums,
  unique.map((value) => [value, 'one set entry']),
  [mark(index, 'set reads', 'focus', 'set-cursor')],
  { inputLength: '4', ...extra },
);

const draft = visual('A set keeps one entry per distinct value, so any duplicate makes its final size smaller than the input size.', [
  frame('Start set construction', 'For nums=[4,1,4,2], set(nums) begins empty before reading index 0.', scan(0, [], {
    setSize: '0',
    action: 'initialize unique set',
  }), 'initialize'),
  frame('Insert the first 4', 'Reading index 0 inserts 4; one input item has produced one distinct set entry.', scan(0, ['4'], {
    setSize: '1',
    action: 'insert 4',
  }), 'insert-four'),
  frame('Insert 1', 'Reading index 1 inserts a new entry for 1, so the unique set becomes {4,1}.', scan(1, ['4', '1'], {
    setSize: '2',
    action: 'insert 1',
  }), 'insert-one'),
  frame('Collapse the repeated 4', 'Reading index 2 finds that 4 already has a set entry; the set remains {4,1} with size 2.', scan(2, ['4', '1'], {
    setSize: '2',
    action: '4 already present; size unchanged',
  }), 'repeat-four'),
  frame('Insert the final 2', 'Reading index 3 inserts 2, producing the complete distinct set {4,1,2}.', scan(3, ['4', '1', '2'], {
    setSize: '3',
    action: 'insert 2',
  }), 'insert-two'),
  frame('Compare the two lengths', 'len(nums)=4 and len(set(nums))=3; because 4 != 3, the function returns true.', scan(3, ['4', '1', '2'], {
    comparison: '4 != 3',
    result: 'true',
  }), 'compare-lengths'),
]);

const review = {
  pattern: 'Distinct-value set and cardinality comparison.',
  recognitionCue: 'Use a set when the question asks whether any equality collision exists and the values themselves, not their positions or counts, determine duplication.',
  invariant: 'After set construction has consumed a prefix, the set contains exactly one entry for every distinct value in that prefix, so its size grows only on first occurrences.',
  stateModel: 'The supplied expression materializes one set of unique values, then compares its cardinality with the original list length; no index map or frequency counts are required.',
  visualRationale: 'An indexed input beside explicit set entries shows the many-to-one collapse of the second 4; a stable set-cursor moves through every value and visible cardinalities prove the final inequality.',
  rejectedAlternatives: [
    'A sorting timeline would expose adjacent duplicates but changes order and uses a different O(n log n) algorithm.',
    'A frequency table stores counts the boolean question never needs.',
    'An early-return seen-set scan is valid but does not match the supplied whole-set cardinality implementation.',
  ],
  transferLesson: 'When only uniqueness matters, canonicalize values into a set and compare cardinality; retain counts or positions only when the follow-up question actually needs them.',
  reviewStatus: 'reviewed',
};

export default defineVisual('contains-duplicate', draft, review);

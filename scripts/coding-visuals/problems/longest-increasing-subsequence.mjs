import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const input = [10, 9, 2, 5, 3, 7, 101, 18];
const example = 'nums = [10, 9, 2, 5, 3, 7, 101, 18]';
const slots = (values) => [...values.map(String), ...Array(8 - values.length).fill('-')];
const state = (tails, inputIndex, slot, extra = {}) => grid(
  [input.map(String), slots(tails)],
  [
    { row: 0, col: inputIndex, label: `input i=${inputIndex}`, tone: 'state', key: 'input-cursor' },
    { row: 1, col: slot, label: `bisect slot=${slot}`, tone: 'focus', key: 'bisect-slot' },
  ],
  {
    example,
    rowAxis: 'top = nums; bottom = smallest_end slots for lengths 1..8',
    nextValue: String(input[inputIndex] ?? 'done'),
    tails: `[${tails.join(',')}]`,
    ...extra,
  },
);

const draft = visual('For every achievable length, retain the smallest ending value found so far.', [
  frame('Initialize empty tails', 'Before reading nums, smallest_end is empty; the first bisect position is slot 0.', state([], 0, 0, { operation: 'start' }), 'initialize-tails'),
  frame('Append 10', 'bisect_left([],10)=0 equals the list length, so append 10: tails=[10].', state([10], 0, 0, { search: 'bisect_left([], 10) = 0', operation: 'append' }), 'append-ten'),
  frame('Replace 10 with 9', 'bisect_left([10],9)=0; replace slot 0 so the length-1 tail becomes smaller.', state([9], 1, 0, { search: 'bisect_left([10], 9) = 0', operation: 'replace 10 -> 9' }), 'replace-with-nine'),
  frame('Replace 9 with 2', 'bisect_left([9],2)=0; tails becomes [2], preserving length 1 with more future room.', state([2], 2, 0, { search: 'bisect_left([9], 2) = 0', operation: 'replace 9 -> 2' }), 'replace-with-two'),
  frame('Append 5', 'bisect_left([2],5)=1 equals the list length, so append and represent an increasing subsequence of length 2.', state([2,5], 3, 1, { search: 'bisect_left([2], 5) = 1', operation: 'append' }), 'append-five'),
  frame('Replace 5 with 3', 'bisect_left([2,5],3)=1; replace 5 with 3 without changing the best length.', state([2,3], 4, 1, { search: 'bisect_left([2,5], 3) = 1', operation: 'replace 5 -> 3' }), 'replace-with-three'),
  frame('Append 7', 'bisect_left([2,3],7)=2, so append 7 and raise the represented length to 3.', state([2,3,7], 5, 2, { search: 'bisect_left([2,3], 7) = 2', operation: 'append' }), 'append-seven'),
  frame('Append 101', 'bisect_left([2,3,7],101)=3, so append 101 and raise the length to 4.', state([2,3,7,101], 6, 3, { search: 'bisect_left([2,3,7], 101) = 3', operation: 'append' }), 'append-one-oh-one'),
  frame('Replace 101 with 18', 'bisect_left([2,3,7,101],18)=3; replacing the length-4 tail keeps length 4 but makes extension easier.', state([2,3,7,18], 7, 3, { search: 'bisect_left([2,3,7,101], 18) = 3', operation: 'replace 101 -> 18' }), 'replace-with-eighteen'),
  frame('Return the number of tail slots', 'smallest_end has four entries, so the longest strictly increasing subsequence length is 4.', state([2,3,7,18], 7, 3, { witness: '2 < 3 < 7 < 18', result: '4' }), 'return-length-four'),
]);

const review = {
  pattern: 'Patience-sorting tails with binary search for the first tail not smaller than each value.',
  recognitionCue: 'The problem asks only for the length of a strictly increasing subsequence in a long array, suggesting O(n log n) tail compression rather than quadratic pair DP.',
  invariant: 'After each input prefix, smallest_end[k] is the minimum possible tail of any increasing subsequence of length k+1 in that prefix, and the tail array is increasing.',
  stateModel: 'Retain only the sorted smallest_end array, current input value, and bisect_left slot. These tails need not form one actual subsequence, but their count equals the optimum length.',
  visualRationale: 'Two aligned rows keep every input index fixed above the length-indexed tail slots; independent stable input and bisect markers show append versus replacement and why lowering a tail preserves length while increasing future options.',
  rejectedAlternatives: [
    'Highlighting one final subsequence hides the alternative tails that justify replacements.',
    'An O(n^2) predecessor graph depicts a different algorithm and overwhelms the invariant.',
    'A binary-search tree obscures that bisect searches a compact sorted tail array.',
  ],
  transferLesson: 'Keep the most permissive representative for every achieved progress level; the same dominance idea prunes states in scheduling, envelopes, and frontier DP.',
  reviewStatus: 'reviewed',
};

export default defineVisual('longest-increasing-subsequence', draft, review);

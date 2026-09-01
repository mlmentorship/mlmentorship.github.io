import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const scores = ['0.20', '0.90', '0.40', '0.80', '0.70'];
const example = 'scores = [0.20,0.90,0.40,0.80,0.70], k = 3';
const state = (items, marks, extra = {}) => array(items, marks, { example, ...extra });

const draft = visual('Partition establishes top-k membership; only those k candidates are sorted for descending output.', [
  frame('Validate k', 'k=3 satisfies 1 <= 3 <= len(scores)=5, so partial selection is defined.', state(scores, [mark(0, 'validate', 'focus', 'phase-cursor')], { check: '1 <= 3 <= 5: true' }), 'validate-k'),
  frame('Partition around the top group', 'argpartition(scores,-3) guarantees that the final three positions contain indices {1,3,4}; their scores 0.90, 0.80, 0.70 exceed the other group.', state(scores, [mark(1, 'candidate', 'state', 'candidate-1'), mark(3, 'candidate', 'state', 'candidate-3'), mark(4, 'candidate', 'state', 'candidate-4')], { operation: 'np.argpartition(scores, -3)[-3:]', candidates: '{1,3,4}; internal order unspecified' }), 'partition-membership'),
  frame('Gather candidate scores', 'Indexing scores[candidates] gathers exactly 0.90, 0.80, and 0.70; no noncandidate enters the ordering step.', state(['1:0.90', '3:0.80', '4:0.70'], [mark(0, 'selected', 'state', 'candidate-1'), mark(1, 'selected', 'state', 'candidate-3'), mark(2, 'selected', 'state', 'candidate-4')], { selectedCount: '3' }), 'gather-candidates'),
  frame('Sort the selected values ascending', 'argsort over the three selected scores orders them by 0.70 < 0.80 < 0.90, corresponding to indices [4,3,1].', state(['4:0.70', '3:0.80', '1:0.90'], [mark(0, 'lowest selected', 'focus', 'phase-cursor')], { operation: 'np.argsort(scores[candidates])', ascendingIndices: '[4,3,1]' }), 'sort-selected'),
  frame('Reverse to descending order', 'The [::-1] slice reverses only the selected ordering, producing indices [1,3,4].', state(['1:0.90', '3:0.80', '4:0.70'], [mark(0, 'rank 1', 'output', 'rank-cursor')], { operation: '[::-1]', descendingScores: '0.90 >= 0.80 >= 0.70' }), 'reverse-ranking'),
  frame('Return original indices', 'Return [1,3,4], which point back to the three largest values in the original score array.', state(scores, [mark(1, 'rank 1', 'output', 'candidate-1'), mark(3, 'rank 2', 'output', 'candidate-3'), mark(4, 'rank 3', 'output', 'candidate-4')], { verification: 'scores[[1,3,4]] = [0.90,0.80,0.70]', result: '[1,3,4]' }), 'return-top-k'),
]);

const review = {
  pattern: 'Linear-average partial selection followed by sorting only the selected k entries.',
  recognitionCue: 'The output needs a small ranked subset of a much larger score vector, so fully sorting all n scores performs unnecessary ordering.',
  invariant: 'After argpartition, candidate membership contains the k largest values although internal order is unspecified; after argsort and reversal, those same candidates are descending.',
  stateModel: 'Retain the original scores and indices, k candidate indices, their gathered values, and selected-only order. Noncandidate relative order is irrelevant.',
  visualRationale: 'Indexed score cells preserve original identities while candidate keys move into a compact selected array and then back as ranked output.',
  rejectedAlternatives: [
    'A fully sorted score array teaches O(n log n) work rather than partial selection.',
    'A size-k heap is valid but depicts a different implementation and update mechanism.',
    'Highlighting final winners alone conflates unordered membership with required output order.',
  ],
  transferLesson: 'Separate selecting which items qualify from ordering the small qualified set; this transfers to retrieval reranking, beam pruning, and percentile candidate generation.',
  reviewStatus: 'reviewed',
};

export default defineVisual('top-k-scores', draft, review);

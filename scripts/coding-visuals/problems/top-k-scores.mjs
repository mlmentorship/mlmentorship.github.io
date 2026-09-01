import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const scores = ['0.1', '0.9', '0.4', '0.8'];
const selectedMotion = (firstX, secondX, firstLabel, secondLabel) => [
  { key: 'selected-index-1', kind: 'score', x: firstX, y: 0, label: firstLabel },
  { key: 'selected-index-3', kind: 'score', x: secondX, y: 0, label: secondLabel },
];

const draft = visual('Use partial partition only to select top-k membership, then sort those candidate indices by their scores for the required descending order.', [
  frame('Validate k against score count', 'For scores [0.1,0.9,0.4,0.8] and k=2, the guard 1<=2<=4 passes.', array(scores, [
    mark(0, 'input start', 'focus', 'selection-cursor'),
  ], {
    guard: '1 <= k=2 <= len(scores)=4',
  }), 'validate'),
  frame('Partition around the top-two boundary', 'np.argpartition(scores,-2)[-2:] selects indices 1 and 3 as an unordered candidate set; it does not promise their order.', array(scores, [
    mark(1, 'candidate index 1', 'state', 'candidate-one'),
    mark(3, 'candidate index 3', 'state', 'candidate-three'),
  ], {
    candidates: '{1,3}',
    scoresAtCandidates: '{0.9,0.8}',
    motion: selectedMotion(1, 3, 'index 1 score 0.9', 'index 3 score 0.8'),
  }), 'partition'),
  frame('Read only the selected scores', 'Advanced indexing scores[candidates] gives the two values 0.9 and 0.8; all other scores leave the ordering work.', array(['index 1: 0.9', 'index 3: 0.8'], [
    mark(0, 'selected index 1', 'focus', 'candidate-one'),
    mark(1, 'selected index 3', 'state', 'candidate-three'),
  ], {
    subsetSize: 'k=2',
    motion: selectedMotion(0, 1, 'index 1 score 0.9', 'index 3 score 0.8'),
  }), 'gather'),
  frame('Sort candidate positions by score', 'argsort over [0.9,0.8] returns ascending positions [1,0]; reversing gives [0,1].', array(['position 0 -> index 1 -> 0.9', 'position 1 -> index 3 -> 0.8'], [
    mark(0, 'descending first', 'focus', 'candidate-one'),
    mark(1, 'descending second', 'state', 'candidate-three'),
  ], {
    arithmetic: 'argsort=[1,0]; reverse=[0,1]',
    motion: selectedMotion(0, 1, 'index 1 first', 'index 3 second'),
  }), 'sort-selected'),
  frame('Index candidates in descending order', 'candidates[[0,1]] returns original score indices [1,3], whose values satisfy 0.9>=0.8.', array(['1', '3'], [
    mark(0, 'score 0.9', 'output', 'candidate-one'),
    mark(1, 'score 0.8', 'output', 'candidate-three'),
  ], {
    verification: 'scores[1]=0.9 >= scores[3]=0.8',
    motion: selectedMotion(0, 1, 'output index 1', 'output index 3'),
    result: '[1,3]',
  }), 'return'),
]);

const review = {
  pattern: 'Partial top-k membership selection followed by sorting only the selected subset.',
  recognitionCue: 'Use argpartition when k ranked elements are needed from a much larger score vector and fully sorting all n scores is unnecessary.',
  invariant: 'After partition, every selected candidate belongs to the top-k score group although candidate order is unspecified; after subset argsort reversal, those same indices are descending by score.',
  stateModel: 'Retain the score vector, valid k, k candidate indices, their gathered scores, and the permutation that orders only those candidates.',
  visualRationale: 'Stable candidate identities move from their original score positions into the gathered subset and final index output, while labels explicitly separate unordered membership from descending order.',
  rejectedAlternatives: [
    'Sorting all scores is simpler but changes the supplied O(n + k log k) mechanism to O(n log n).',
    'A heap can maintain top k but does not match the vectorized argpartition implementation.',
    'Calling partition output already sorted is incorrect because argpartition guarantees membership, not candidate order.',
  ],
  transferLesson: 'Separate selection from ordering: first isolate the small set that can contain the answer, then pay sorting cost only on that set.',
  reviewStatus: 'reviewed',
};

export default defineVisual('top-k-scores', draft, review);

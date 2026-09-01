import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const values = ['4', '5', '6', '7', '0', '1', '2'];

const draft = visual('Keep the interval containing target 0 by identifying one sorted half at each probe.', [
  frame(
    'Use the sorted left half',
    'left=0, right=6, and middle=3. Since nums[0]=4 <= nums[3]=7, indices 0..3 are sorted; target 0 is not in [4,7), so set left=4.',
    array(values, [
      mark(0, 'left', 'state', 'pointer-left'),
      mark(3, 'middle', 'focus', 'pointer-middle'),
      mark(6, 'right', 'state', 'pointer-right'),
    ], { target: '0', sorted: 'indices 0..3', decision: '0 not in [4, 7): left = 4' }),
    'discard-sorted-left',
  ),
  frame(
    'Keep the sorted left half',
    'Now left=4, right=6, and middle=5. Values 0..1 are sorted and 0 is in [nums[4], nums[5])=[0,1), so set right=middle-1=4.',
    array(values, [
      mark(4, 'left', 'state', 'pointer-left'),
      mark(5, 'middle', 'focus', 'pointer-middle'),
      mark(6, 'right', 'state', 'pointer-right'),
    ], { target: '0', sorted: 'indices 4..5', decision: '0 in [0, 1): right = 4' }),
    'keep-sorted-left',
  ),
  frame(
    'Return the target',
    'left=right=middle=4 and nums[4]=0, so the search returns index 4.',
    array(values, [
      mark(4, 'left = middle = right', 'output', 'pointer-middle'),
    ], { target: '0', comparison: '0 = 0', result: 'index 4' }),
    'return-target',
  ),
]);

const review = {
  pattern: 'Modified binary search that first identifies the sorted half around middle.',
  recognitionCue: 'The array was sorted and rotated once with distinct values, so although the whole interval may cross the pivot, at least one side of middle remains sorted.',
  invariant: 'If target 0 exists, it remains in inclusive [left, right]. A half is discarded only after its sorted value range proves whether target can occur there.',
  stateModel: 'Track left, middle, and right indices plus the immutable target. The endpoint-to-middle comparison identifies a sorted half; a target-range comparison chooses the next interval.',
  visualRationale: 'The indexed rotated array keeps the pivot geometry visible while stable boundary pointers move across it, making both sorted-half tests readable without color.',
  rejectedAlternatives: [
    'A circular diagram emphasizes rotation but makes index intervals and return coordinates harder to read.',
    'A branch flowchart shows conditions without showing which concrete values justify them.',
    'A table of left, middle, and right values hides the contiguous half removed after each branch.',
  ],
  transferLesson: 'When a monotone ordering is disrupted once, find a locally ordered region and use its endpoint values to make the same safe-elimination argument as ordinary binary search.',
  reviewStatus: 'reviewed',
};

export default defineVisual('search-in-rotated-sorted-array', draft, review);

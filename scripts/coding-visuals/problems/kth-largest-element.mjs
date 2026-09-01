import { defineVisual, frame, heap, visual } from '../primitives.mjs';

const draft = visual('A size-k min-heap retains the k largest values seen, so its root is the kth largest.', [
  frame(
    'Seed the candidate heap',
    'For nums = [3,2,1,5,6,4] and k = 2, push 3. The heap has room, so nothing is removed.',
    heap(['3'], { input: '[3,2,1,5,6,4]', current: '3', processed: '1 of 6', size: '1 <= k' }),
    'push-3',
  ),
  frame(
    'Fill the second slot',
    'Push 2. The min-heap orders the two protected candidates as [2,3], with the weaker candidate 2 at the root.',
    heap(['2', '3'], { current: '2', processed: '2 of 6', size: '2 = k', action: 'keep both' }),
    'push-2',
  ),
  frame(
    'Discard an undersized candidate',
    'Push 1 to make [1,3,2]. Size 3 exceeds k, so pop root 1; [2,3] still holds the largest two of [3,2,1].',
    heap(['2', '3'], { current: '1', processed: '3 of 6', overflow: '[1,3,2]', action: 'pop 1' }),
    'reject-1',
  ),
  frame(
    'Raise the protected floor',
    'Push 5 to make [2,3,5], then pop 2. The retained candidates become [3,5], so the root threshold rises to 3.',
    heap(['3', '5'], { current: '5', processed: '4 of 6', overflow: '[2,3,5]', action: 'pop 2' }),
    'accept-5',
  ),
  frame(
    'Keep the next large value',
    'Push 6 to make [3,5,6], then pop 3. The heap [5,6] is exactly the largest two values seen so far.',
    heap(['5', '6'], { current: '6', processed: '5 of 6', overflow: '[3,5,6]', action: 'pop 3' }),
    'accept-6',
  ),
  frame(
    'Reject the last weaker value',
    'Push 4 to make [4,6,5], then pop 4. The protected set remains [5,6] after all six inputs.',
    heap(['5', '6'], { current: '4', processed: '6 of 6', overflow: '[4,6,5]', action: 'pop 4' }),
    'reject-4',
  ),
  frame(
    'Read the kth largest',
    'The root is the smallest of the two retained values: min([5,6]) = 5, so the second largest is 5.',
    heap(['5', '6'], { k: '2', arithmetic: 'min([5,6]) = 5', result: '5' }),
    'return-root',
  ),
]);

const review = {
  pattern: 'Streaming top-k selection with a min-heap capped at k entries.',
  recognitionCue: 'Use it when an unsorted or streaming input asks for the kth largest item or the largest k items, while sorting every value would retain more order than the answer needs.',
  invariant: 'After each input is pushed and any size-(k+1) overflow root is popped, the heap contains exactly the largest min(k, processed) values seen; its root is the weakest retained candidate.',
  stateModel: 'The minimal state is the input cursor, k, and a min-heap of at most k values. The trace shows every push branch, each overflow heap, the removed root, and the new threshold.',
  visualRationale: 'An actual complete binary min-heap makes the parent-child ordering and changing root threshold visible. Stable value keys carry 3, 5, and 6 as they move to new heap positions.',
  rejectedAlternatives: [
    'A fully sorted array was rejected because it hides the bounded-memory invariant and suggests O(n log n) work.',
    'A table of heap contents was rejected because it does not expose the root or complete-tree geometry.',
    'Quickselect partitions were rejected because they do not match the supplied heap implementation.',
  ],
  transferLesson: 'For a bounded best-k set, choose the opposite heap polarity: a min-heap protects the largest k and a max-heap protects the smallest k; the root is always the next item to evict.',
  reviewStatus: 'reviewed',
};

export default defineVisual('kth-largest-element', draft, review);

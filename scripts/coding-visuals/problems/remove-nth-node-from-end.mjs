import { defineVisual, frame, linked, visual } from '../primitives.mjs';

const list = (left, right, extra = {}) => linked(
  ['dummy', '1', '2', '3', '4', '5'].map((value) => ({
    value,
    key: `node-${value}`,
    ...(value === left ? { pointer: 'left' } : {}),
    ...(value === right && value !== left ? { pointer: 'right' } : {}),
  })),
  { input: 'head=[1,2,3,4,5], n=2', ...extra },
);

const draft = visual('Maintain a two-node gap so left stops immediately before the second node from the end.', [
  frame(
    'Start both pointers at the dummy',
    'A dummy points to 1, so removing the head would use the same link update. Both left and right start at dummy.',
    list('dummy', 'dummy', { right: 'dummy', gap: '0 nodes' }),
    'initialize-dummy',
  ),
  frame(
    'Advance right once',
    'The first of n=2 gap steps moves right from dummy to node 1; left stays at dummy.',
    list('dummy', '1', { gap: '1 node' }),
    'advance-right-one',
  ),
  frame(
    'Finish the two-node gap',
    'The second gap step moves right to node 2. Right is now exactly two links ahead of left.',
    list('dummy', '2', { gap: '2 nodes' }),
    'advance-right-two',
  ),
  frame(
    'Move both pointers: step 1',
    'Because right.next is node 3, move left to 1 and right to 3 while preserving the two-link gap.',
    list('1', '3', { gap: '2 nodes' }),
    'walk-together-one',
  ),
  frame(
    'Move both pointers: step 2',
    'Because right.next is node 4, move left to 2 and right to 4; the gap remains two links.',
    list('2', '4', { gap: '2 nodes' }),
    'walk-together-two',
  ),
  frame(
    'Move both pointers: step 3',
    'Because right.next is node 5, move left to 3 and right to 5. Now right.next is null, so the loop stops.',
    list('3', '5', { gap: '2 nodes', target: 'left.next is node 4' }),
    'walk-together-three',
  ),
  frame(
    'Rewire around node 4',
    'Set left.next = left.next.next, changing the link 3->4 into 3->5. Return dummy.next at node 1.',
    linked([
      { value: 'dummy', key: 'node-dummy' },
      { value: '1', key: 'node-1' },
      { value: '2', key: 'node-2' },
      { value: '3', key: 'node-3', pointer: 'left', tone: 'focus' },
      { value: '5', key: 'node-5', tone: 'output' },
    ], { changedLink: '3.next: node 4 -> node 5', removed: 'node 4', result: '[1,2,3,5]' }),
    'remove-four',
  ),
]);

const review = {
  pattern: 'Two linked-list pointers separated by a fixed n-node gap, anchored by a dummy node.',
  recognitionCue: 'The node is identified relative to the list end, but only one forward pass is allowed, so a leading pointer can convert distance-from-end into a simultaneous stopping position.',
  invariant: 'After the initial advance, right remains exactly n links ahead of left. Therefore, when right is the tail, left.next is the nth node from the end.',
  stateModel: 'Keep a dummy-to-head link plus left and right pointers. Advance right n times, move both while right.next exists, then replace left.next with left.next.next.',
  visualRationale: 'A full linked chain with stable node and pointer identities directly shows the fixed gap, each synchronized move, the stopping condition, and the changed 3-to-5 link.',
  rejectedAlternatives: [
    'An indexed array suggests random access that the linked-list solution does not have.',
    'A pointer-position table hides the actual next-link that is rewired.',
    'Counting length in a first pass misses the one-pass fixed-gap mechanism.',
  ],
  transferLesson: 'Use a fixed lead whenever a linked-list target is defined by distance from the end; add a dummy when the operation may change the head.',
  independentReview: '3.4 source-to-frame replay',
  reviewStatus: 'reviewed',
};

export default defineVisual('remove-nth-node-from-end', draft, review);

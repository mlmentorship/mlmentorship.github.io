import { defineVisual, frame, linked, visual } from '../primitives.mjs';

const cycleLinks = ['1->2', '2->3', '3->4', '4->2'];

const state = (slowNode, fastNode, extra = {}) => linked([
  { value: '1', pointer: slowNode === '1' ? 'slow' : fastNode === '1' ? 'fast' : undefined },
  { value: '2', pointer: slowNode === '2' ? 'slow' : fastNode === '2' ? 'fast' : undefined },
  { value: '3', pointer: slowNode === '3' ? 'slow' : fastNode === '3' ? 'fast' : undefined },
  { value: '4', pointer: slowNode === '4' ? 'slow' : fastNode === '4' ? 'fast' : undefined },
], {
  arrows: cycleLinks,
  slowAt: slowNode,
  fastAt: fastNode,
  motion: [
    { key: 'pointer-slow', kind: 'pointer', x: Number(slowNode), y: 0, label: `slow at ${slowNode}` },
    { key: 'pointer-fast', kind: 'pointer', x: Number(fastNode), y: 0, label: `fast at ${fastNode}` },
    ...['1', '2', '3', '4'].map((value, index) => ({ key: `node-${value}`, kind: 'node', x: index, y: 0, label: value })),
  ],
  ...extra,
});

const draft = visual('With a cycle, a pointer moving two links per round gains one cycle position on a pointer moving one link, so they must meet.', [
  frame(
    'Initialize both pointers at head',
    'For 1->2->3->4 with 4.next = 2, set slow = fast = node 1. The loop guard passes because fast and fast.next both exist.',
    state('1', '1', { guard: 'fast=1 and fast.next=2' }),
    'initialize-at-head',
  ),
  frame(
    'Advance round 1',
    'Move slow one link 1->2. Move fast two links 1->2->3. They are different nodes, so continue.',
    state('2', '3', { movement: 'slow: 1->2; fast: 1->2->3', identityCheck: '2 is not 3' }),
    'advance-round-1',
  ),
  frame(
    'Advance round 2',
    'Move slow 2->3. Move fast 3->4->2, wrapping through the cycle. They still differ.',
    state('3', '2', { movement: 'slow: 2->3; fast: 3->4->2', identityCheck: '3 is not 2' }),
    'advance-round-2',
  ),
  frame(
    'Advance round 3',
    'Move slow 3->4. Move fast 2->3->4. Both object references now identify node 4.',
    state('4', '4', { movement: 'slow: 3->4; fast: 2->3->4', identityCheck: 'slow is fast at node 4' }),
    'advance-round-3',
  ),
  frame(
    'Return true on identity',
    'The in-loop identity check succeeds, so return true before another guard evaluation.',
    state('4', '4', { check: 'slow is fast', result: 'true' }),
    'return-true',
  ),
]);

const review = {
  pattern: 'Floyd tortoise-and-hare cycle detection with one-step and two-step pointers.',
  recognitionCue: 'Use it for cycle detection in a deterministic next-pointer structure when constant extra space is required and storing every visited node would be unnecessary.',
  invariant: 'After r loop iterations, slow has followed r next links and fast has followed 2r. If no cycle exists fast reaches null; inside a cycle their relative offset changes by one per round.',
  stateModel: 'The minimal state is two node references. The loop guard validates both fast hops, then slow advances once, fast twice, and object identity rather than equal values determines a meeting.',
  visualRationale: 'The full list and explicit back edge 4->2 remain visible in every frame. Authored slow/fast motion keys carry the two pointers around the fixed node topology, with movement paths written for color-independent reading.',
  rejectedAlternatives: [
    'A visited set was rejected because it depicts an O(n)-space algorithm instead of the supplied constant-space method.',
    'A pointer-position table was rejected because it hides the back edge and why fast wraps to node 2.',
    'A circular ring alone was rejected because it omits the non-cyclic stem from node 1.',
  ],
  transferLesson: 'Different traversal speeds convert a hidden cycle into an inevitable identity collision; guard every fast hop. The technique transfers to repeated-state sequences and cycle-entry algorithms.',
  independentReview: '3.4 source-to-frame replay',
  reviewStatus: 'reviewed',
};

export default defineVisual('linked-list-cycle', draft, review);

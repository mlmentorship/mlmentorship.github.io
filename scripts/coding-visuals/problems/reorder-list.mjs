import { defineVisual, frame, linked, visual } from '../primitives.mjs';

const node = (value, pointer) => ({ value: String(value), key: `list-node-${value}`, ...(pointer ? { pointer } : {}) });
const pointerMotion = (positions) => Object.entries(positions).map(([key, x]) => ({
  key: `pointer-${key}`,
  kind: 'pointer',
  x,
  y: 0,
  label: `${key} at node ${x}`,
})).concat([1, 2, 3, 4, 5].map((value) => ({
  key: `list-node-${value}`,
  kind: 'node',
  x: value,
  y: 0,
  label: String(value),
})));

const draft = visual('Find the first-half tail, reverse the detached second half, then splice one reversed node after each first-half node.', [
  frame(
    'Initialize middle pointers',
    'For 1->2->3->4->5, set slow = 1 and fast = 1. Both fast.next and fast.next.next exist.',
    linked([node(1, 'slow / fast'), node(2), node(3), node(4), node(5)], {
      guard: 'fast.next=2 and fast.next.next=3',
      motion: pointerMotion({ slow: 1, fast: 1 }),
    }),
    'initialize-middle-scan',
  ),
  frame(
    'Advance middle pointers once',
    'Move slow one link 1->2 and fast two links 1->2->3. The loop guard still passes at fast = 3.',
    linked([node(1), node(2, 'slow'), node(3, 'fast'), node(4), node(5)], {
      movement: 'slow: 1->2; fast: 1->2->3',
      motion: pointerMotion({ slow: 2, fast: 3 }),
    }),
    'middle-round-1',
  ),
  frame(
    'Advance middle pointers twice',
    'Move slow 2->3 and fast 3->4->5. At fast = 5, fast.next is null, so the middle loop stops.',
    linked([node(1), node(2), node(3, 'slow'), node(4), node(5, 'fast')], {
      movement: 'slow: 2->3; fast: 3->4->5',
      guard: 'fast.next is null',
      motion: pointerMotion({ slow: 3, fast: 5 }),
    }),
    'middle-round-2',
  ),
  frame(
    'Detach the second half',
    'Save second = slow.next = 4, then write slow.next = null. The independent halves are 1->2->3 and 4->5.',
    linked([node(4, 'second'), node(5)], { firstHalf: '1->2->3->null', secondHalf: '4->5->null' }),
    'split-halves',
  ),
  frame(
    'Reverse second-half node 4',
    'With previous = null and second = 4, save next_node = 5, write 4.next = null, then advance previous = 4 and second = 5.',
    linked([node(4, 'previous')], { second: ['5'], firstHalf: '1->2->3->null', reversed: '4->null', savedNext: '5' }),
    'reverse-node-4',
  ),
  frame(
    'Reverse second-half node 5',
    'Save next_node = null, write 5.next = 4, then advance previous = 5 and second = null. The reversed half is 5->4.',
    linked([node(5, 'previous'), node(4)], { firstHalf: '1->2->3->null', reversed: '5->4->null', secondPointer: 'null' }),
    'reverse-node-5',
  ),
  frame(
    'Initialize the merge',
    'Set first = head = 1 and second = previous = 5. The two independent chains are 1->2->3 and 5->4.',
    linked([node(5, 'second'), node(4)], {
      firstHalf: '1->2->3',
      secondPointer: '5',
      motion: pointerMotion({ first: 1, second: 5 }),
    }),
    'initialize-merge',
  ),
  frame(
    'Splice node 5 after node 1',
    'Save first_next = 2 and second_next = 4. Write 1.next = 5 and 5.next = 2, then advance first = 2 and second = 4.',
    linked([node(1), node(5), node(2, 'first'), node(3)], {
      second: ['4'],
      saved: 'first_next=2; second_next=4',
      secondPointer: '4',
      motion: pointerMotion({ first: 2, second: 4 }),
    }),
    'merge-node-5',
  ),
  frame(
    'Splice node 4 after node 2',
    'Save first_next = 3 and second_next = null. Write 2.next = 4 and 4.next = 3, then second becomes null.',
    linked([node(1), node(5), node(2), node(4), node(3, 'first')], {
      saved: 'first_next=3; second_next=null',
      secondPointer: 'null',
      motion: pointerMotion({ first: 3, second: 0 }),
    }),
    'merge-node-4',
  ),
  frame(
    'Finish when the second half is empty',
    'The merge guard fails at second = null. The in-place list is 1->5->2->4->3->null.',
    linked([node(1), node(5), node(2), node(4), node(3)], { result: '1->5->2->4->3' }),
    'finish-reordered-list',
  ),
]);

const review = {
  pattern: 'Three-phase in-place list transform: tortoise-hare midpoint, second-half reversal, then alternating pointer splices.',
  recognitionCue: 'Use it when nodes must alternate from the front and back of a singly linked list without an auxiliary array or stack.',
  invariant: 'The midpoint scan keeps fast twice as far along; reversal preserves a reversed prefix plus untouched suffix; merging preserves the final alternating prefix and two unconsumed chains.',
  stateModel: 'The minimal state is slow/fast for splitting, previous/second/next_node for reversal, and first/second plus their saved next pointers for merging. Every destructive link write follows its required save.',
  visualRationale: 'Persistent linked nodes show actual chain topology, detachment, reversal, and both splices. Named pointer labels and written links make each mutation understandable without color, and stable node keys visibly carry nodes into final positions.',
  rejectedAlternatives: [
    'An output value array was rejected because the algorithm rewires nodes rather than copying values.',
    'A deque of nodes was rejected because it uses O(n) extra space and skips the supplied reversal mechanism.',
    'Only split/reversed/final snapshots were rejected because they hide middle-pointer rounds and saved-next safety.',
  ],
  transferLesson: 'Complex list transforms become safe compositions of small invariants: locate a boundary, detach, reverse with a saved route, then splice only after saving both continuations.',
  reviewStatus: 'reviewed',
};

export default defineVisual('reorder-list', draft, review);

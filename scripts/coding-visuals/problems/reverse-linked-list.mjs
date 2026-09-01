import { defineVisual, frame, linked, visual } from '../primitives.mjs';

const draft = visual('Preserve the unreversed suffix before redirecting each current.next, then advance the previous/current boundary one node.', [
  frame(
    'Initialize the two list regions',
    'For 1->2->3->4->null, previous = null and head/current = 1. The whole list is still the unreversed suffix.',
    linked([{ value: '1', pointer: 'current' }, { value: '2' }, { value: '3' }, { value: '4' }], {
      previous: 'null',
      actualLinks: '1->2, 2->3, 3->4, 4->null',
    }),
    'initialize-pointers',
  ),
  frame(
    'Reverse the link at node 1',
    'Save next_node = 2, write 1.next = null, then set previous = 1 and head = 2. The saved pointer prevents losing nodes 2->3->4.',
    linked([{ value: '2', pointer: 'current' }, { value: '3' }, { value: '4' }], {
      second: ['1'],
      reversedPrefix: '1->null',
      unreversedSuffix: '2->3->4->null',
      savedNext: '2',
    }),
    'reverse-node-1',
  ),
  frame(
    'Reverse the link at node 2',
    'Save next_node = 3, write 2.next = 1, then advance previous = 2 and head = 3.',
    linked([{ value: '3', pointer: 'current' }, { value: '4' }], {
      second: ['2', '1'],
      reversedPrefix: '2->1->null',
      unreversedSuffix: '3->4->null',
      savedNext: '3',
    }),
    'reverse-node-2',
  ),
  frame(
    'Reverse the link at node 3',
    'Save next_node = 4, write 3.next = 2, then advance previous = 3 and head = 4.',
    linked([{ value: '4', pointer: 'current' }], {
      second: ['3', '2', '1'],
      reversedPrefix: '3->2->1->null',
      unreversedSuffix: '4->null',
      savedNext: '4',
    }),
    'reverse-node-3',
  ),
  frame(
    'Reverse the final link',
    'Save next_node = null, write 4.next = 3, then set previous = 4 and head = null. The while condition now fails.',
    linked([{ value: '4', pointer: 'previous', tone: 'output' }, { value: '3' }, { value: '2' }, { value: '1' }], {
      current: 'null',
      actualLinks: '4->3, 3->2, 2->1, 1->null',
    }),
    'reverse-node-4',
  ),
  frame(
    'Return the new head',
    'previous points to node 4, the head of the fully reversed chain 4->3->2->1->null.',
    linked([{ value: '4', pointer: 'new head', tone: 'output' }, { value: '3' }, { value: '2' }, { value: '1' }], {
      result: '4 -> 3 -> 2 -> 1',
    }),
    'return-previous',
  ),
]);

const review = {
  pattern: 'In-place singly linked-list reversal with previous, current, and saved-next pointers.',
  recognitionCue: 'Use it when a forward-only chain must reverse direction in constant extra space and changing current.next would otherwise destroy access to the unprocessed suffix.',
  invariant: 'Before each iteration, previous heads a fully reversed prefix, head heads the untouched forward suffix, and together those disjoint regions contain every original node exactly once.',
  stateModel: 'The minimal state is previous, current/head, and next_node. Each iteration must save next, redirect current.next, move previous, and move current in that exact order.',
  visualRationale: 'Two visible linked rows separate the reversed prefix from the unreversed suffix while text states the real outgoing links. Stable node identities and current/previous labels show nodes crossing the boundary without color.',
  rejectedAlternatives: [
    'A value array was rejected because reversing values is not reversing node links.',
    'A recursive call stack was rejected because it depicts a different O(n) stack-space implementation.',
    'A before/after list was rejected because it omits the saved-next safety requirement and intermediate disconnected regions.',
  ],
  transferLesson: 'Before mutating the only route to remaining work, save that route; then rewire and advance the boundary. This applies to list splicing, segment reversal, and in-place pointer transformations.',
  independentReview: '3.4 source-to-frame replay',
  reviewStatus: 'reviewed',
};

export default defineVisual('reverse-linked-list', draft, review);

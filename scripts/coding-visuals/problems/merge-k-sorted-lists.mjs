import { array, defineVisual, frame, linked, mark, visual } from '../primitives.mjs';

const state = (values, extra) => array(values, values.length ? [
  mark(0, 'min root', 'focus', 'heap-root'),
] : [], {
  inputLists: 'A:1->4->5; B:1->3->4; C:2->6',
  heapLayout: values.length > 1 ? 'level order: slot 0 parent of slots 1 and 2' : 'level order: slot 0 is root',
  ...extra,
});

const draft = visual('A min-heap containing one current head per nonempty list always exposes the globally smallest unmerged node.', [
  frame(
    'Seed one head from each list',
    'Push tuples (1,0,A1), (1,1,B1), and (2,2,C2). The list index breaks the value-1 tie, so A1 is the root.',
    state(['(1,0,A1)', '(1,1,B1)', '(2,2,C2)'], { heapArray: '[(1,0,A1),(1,1,B1),(2,2,C2)]', tail: 'dummy', outputOrder: '[]' }),
    'seed-current-heads',
  ),
  frame(
    'Emit A1 and replace it with A4',
    'Pop (1,0,A1), write dummy.next = A1, and set tail = A1. Because A1.next is A4, push (4,0,A4).',
    state(['(1,1,B1)', '(2,2,C2)', '(4,0,A4)'], { popped: '(1,0,A1)', pushed: '(4,0,A4)', tail: 'A1', outputOrder: '[A1]' }),
    'emit-a1',
  ),
  frame(
    'Emit B1 and replace it with B3',
    'Pop (1,1,B1), overwrite A1.next = B1, and advance tail. Push B1.next as (3,1,B3).',
    state(['(2,2,C2)', '(4,0,A4)', '(3,1,B3)'], { popped: '(1,1,B1)', pushed: '(3,1,B3)', tail: 'B1', outputOrder: '[A1,B1]' }),
    'emit-b1',
  ),
  frame(
    'Emit C2 and replace it with C6',
    'Pop (2,2,C2), link B1.next = C2, then push C2.next as (6,2,C6).',
    state(['(3,1,B3)', '(4,0,A4)', '(6,2,C6)'], { popped: '(2,2,C2)', pushed: '(6,2,C6)', tail: 'C2', outputOrder: '[A1,B1,C2]' }),
    'emit-c2',
  ),
  frame(
    'Emit B3 and replace it with B4',
    'Pop (3,1,B3), link C2.next = B3, then push (4,1,B4). A4 wins the value-4 tie because list index 0 is smaller than 1.',
    state(['(4,0,A4)', '(6,2,C6)', '(4,1,B4)'], { popped: '(3,1,B3)', pushed: '(4,1,B4)', tail: 'B3', outputOrder: '[A1,B1,C2,B3]' }),
    'emit-b3',
  ),
  frame(
    'Emit A4 and replace it with A5',
    'Pop (4,0,A4), link B3.next = A4, and push A4.next as (5,0,A5).',
    state(['(4,1,B4)', '(6,2,C6)', '(5,0,A5)'], { popped: '(4,0,A4)', pushed: '(5,0,A5)', tail: 'A4', outputOrder: '[A1,B1,C2,B3,A4]' }),
    'emit-a4',
  ),
  frame(
    'Emit B4 without replacement',
    'Pop (4,1,B4) and link A4.next = B4. B4.next is null, so the if branch skips heappush.',
    state(['(5,0,A5)', '(6,2,C6)'], { popped: '(4,1,B4)', pushed: 'none: node.next is null', tail: 'B4', outputOrder: '[A1,B1,C2,B3,A4,B4]' }),
    'emit-b4',
  ),
  frame(
    'Emit A5 without replacement',
    'Pop (5,0,A5), link B4.next = A5, and skip the push because A5.next is null.',
    state(['(6,2,C6)'], { popped: '(5,0,A5)', pushed: 'none: node.next is null', tail: 'A5', outputOrder: '[A1,B1,C2,B3,A4,B4,A5]' }),
    'emit-a5',
  ),
  frame(
    'Emit C6 and empty the heap',
    'Pop (6,2,C6), link A5.next = C6, and skip the push because C6.next is null. The heap is now empty.',
    linked([
      { value: 'A1' }, { value: 'B1' }, { value: 'C2' }, { value: 'B3' },
      { value: 'A4' }, { value: 'B4' }, { value: 'A5' }, { value: 'C6', pointer: 'tail' },
    ], { heapState: '[]', popped: '(6,2,C6)', pushed: 'none: node.next is null', values: '1,1,2,3,4,4,5,6' }),
    'emit-c6',
  ),
  frame(
    'Return the merged head',
    'The while loop stops when the heap empties. Return dummy.next, which is A1 and heads values 1->1->2->3->4->4->5->6.',
    linked([
      { value: 'A1', pointer: 'dummy.next' }, { value: 'B1' }, { value: 'C2' }, { value: 'B3' },
      { value: 'A4' }, { value: 'B4' }, { value: 'A5' }, { value: 'C6' },
    ], { heapState: '[]', nodeOrder: 'A1->B1->C2->B3->A4->B4->A5->C6', result: '1->1->2->3->4->4->5->6' }),
    'return-merged-list',
  ),
]);

const review = {
  pattern: 'K-way linked-list merge using a min-heap with one current node tuple per nonempty source list.',
  recognitionCue: 'Use it when k individually sorted streams must be merged and only their current heads can be candidates for the next global minimum.',
  invariant: 'The heap contains exactly the first unmerged node of each nonexhausted list, and the tail follows all nodes already popped in nondecreasing tuple order.',
  stateModel: 'The minimal state is a heap of (value, list index, node) tuples plus dummy and tail pointers. Each loop pops one node, links it after tail, and pushes only that node next.',
  visualRationale: 'A narrow-width level-order heap row exposes the live frontier with a stable min-root pointer and explicit parent-slot relation, while tuple labels preserve value, tie-break index, and node identity. Popped, pushed, tail, and emitted states remain explicit without color.',
  rejectedAlternatives: [
    'Scanning all k current heads was rejected because it hides the supplied logarithmic heap selection.',
    'Pairwise divide-and-conquer merging was rejected because it depicts a different algorithm.',
    'A final merged list alone was rejected because it omits heap replacements, null-next branches, and tuple tie-breaking.',
  ],
  transferLesson: 'For sorted sources, keep only one frontier item per source and replace it from the same source after extraction. The pattern transfers to external merge, log aggregation, and k-way iterators.',
  reviewStatus: 'reviewed',
};

export default defineVisual('merge-k-sorted-lists', draft, review);

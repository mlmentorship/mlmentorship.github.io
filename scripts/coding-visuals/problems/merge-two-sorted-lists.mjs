import { defineVisual, frame, graph, visual } from '../primitives.mjs';

const nodes = ['dummy', 'A1', 'A4', 'A7', 'B2', 'B3', 'B8'];
const sourceEdges = ['A1 -> A4', 'A4 -> A7', 'B2 -> B3', 'B3 -> B8'];

const draft = visual('Rewire the smaller current node after tail, then advance only the list that supplied it.', [
  frame(
    'Initialize two sorted chains',
    'A1->A4->A7 and B2->B3->B8 are separate. dummy.next is null; tail=dummy, first=A1, and second=B2.',
    graph(nodes, sourceEdges, { start: 'dummy', tail: 'dummy', first: 'A1=1', second: 'B2=2', dummyNext: 'null' }),
    'initialize-heads',
  ),
  frame(
    'Attach A1',
    'Since 1 <= 2, set dummy.next=A1, advance first to A4, then advance tail to A1.',
    graph(nodes, ['dummy -> A1', ...sourceEdges], { start: 'A1', tail: 'A1', first: 'A4=4', second: 'B2=2', comparison: '1 <= 2' }),
    'attach-a1',
  ),
  frame(
    'Attach B2',
    'Compare A4=4 with B2=2. Replace A1.next=A4 with A1.next=B2, advance second to B3, and move tail to B2.',
    graph(nodes, ['dummy -> A1', 'A1 -> B2', 'A4 -> A7', 'B2 -> B3', 'B3 -> B8'], { start: 'B2', tail: 'B2', first: 'A4=4', second: 'B3=3', comparison: '4 > 2' }),
    'attach-b2',
  ),
  frame(
    'Attach B3',
    'Compare A4=4 with B3=3. B2 already points to B3, so advance second to B8 and tail to B3.',
    graph(nodes, ['dummy -> A1', 'A1 -> B2', 'B2 -> B3', 'B3 -> B8', 'A4 -> A7'], { start: 'B3', tail: 'B3', first: 'A4=4', second: 'B8=8', comparison: '4 > 3' }),
    'attach-b3',
  ),
  frame(
    'Attach A4',
    'Compare A4=4 with B8=8. Replace B3.next=B8 with B3.next=A4, advance first to A7, and move tail to A4.',
    graph(nodes, ['dummy -> A1', 'A1 -> B2', 'B2 -> B3', 'B3 -> A4', 'A4 -> A7'], { start: 'A4', tail: 'A4', first: 'A7=7', second: 'B8=8', comparison: '4 <= 8' }),
    'attach-a4',
  ),
  frame(
    'Attach A7',
    'Compare A7=7 with B8=8. A4 already points to A7, so advance first to null and tail to A7.',
    graph(nodes, ['dummy -> A1', 'A1 -> B2', 'B2 -> B3', 'B3 -> A4', 'A4 -> A7'], { start: 'A7', tail: 'A7', first: 'null', second: 'B8=8', comparison: '7 <= 8' }),
    'attach-a7',
  ),
  frame(
    'Append the remaining suffix',
    'The loop ends because first is null. Set A7.next to second=B8 and return dummy.next=A1.',
    graph(nodes, ['dummy -> A1', 'A1 -> B2', 'B2 -> B3', 'B3 -> A4', 'A4 -> A7', 'A7 -> B8'], { start: 'B8', tailNext: 'B8 remainder', result: '[1,2,3,4,7,8]' }),
    'append-b8',
  ),
]);

const review = {
  pattern: 'Two sorted linked-list cursors with a dummy output head and moving tail.',
  recognitionCue: 'Two inputs are already sorted and must be merged by relinking nodes, so only the two current heads can be the next smallest output node.',
  invariant: 'The chain after dummy is sorted and contains exactly the consumed nodes; tail is its final node, while first and second begin the two untouched sorted suffixes.',
  stateModel: 'Keep first, second, and tail pointers plus the dummy head. Compare current values, attach one node, advance only its source cursor, then append the one remaining suffix.',
  visualRationale: 'Stable node identities and explicit directed links show the initially separate chains, every changed next pointer, each cursor decision, and the constant-time suffix append.',
  rejectedAlternatives: [
    'Copying values into an array hides that the implementation reuses and rewires existing nodes.',
    'A comparison table omits the evolving next-pointer chain and tail invariant.',
    'A final merged snapshot skips which source pointer advances after each decision.',
  ],
  transferLesson: 'For sorted stream merges, emit the smaller frontier item, advance only its source, and preserve an output tail; the same invariant scales to k-way merging with a heap.',
  reviewStatus: 'reviewed',
};

export default defineVisual('merge-two-sorted-lists', draft, review);

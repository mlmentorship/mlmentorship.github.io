import { defineVisual, frame, linked, visual } from '../primitives.mjs';

const sourceA = ['A1', 'A4', 'A7'];
const sourceB = ['B2', 'B3', 'B8'];
const sourceX = { A1: 100, A4: 180, A7: 260, B2: 100, B3: 180, B8: 260 };

function state(output, tail, first, second, extra = {}) {
  const consumed = new Set(output);
  const pointerLabels = (key) => [
    key === tail ? 'tail' : '',
    key === first ? 'first' : '',
    key === second ? 'second' : '',
  ].filter(Boolean);
  const nodes = [
    { key: 'dummy', value: 'dummy', x: 62, y: 190, pointer: pointerLabels('dummy'), tone: tail === 'dummy' ? 'focus' : 'neutral' },
    ...output.map((key, index) => ({
      key,
      value: key,
      x: 122 + index * 60,
      y: 190,
      pointer: pointerLabels(key),
      tone: key === tail ? 'focus' : 'output',
    })),
    ...sourceA.filter((key) => !consumed.has(key)).map((key) => ({
      key,
      value: key,
      x: sourceX[key],
      y: 54,
      pointer: pointerLabels(key),
      tone: key === first ? 'state' : 'neutral',
    })),
    ...sourceB.filter((key) => !consumed.has(key)).map((key) => ({
      key,
      value: key,
      x: sourceX[key],
      y: 119,
      pointer: pointerLabels(key),
      tone: key === second ? 'state' : 'neutral',
    })),
  ];
  const edges = [];
  const connect = (keys, row) => keys.slice(1).map((key, index) => ({
    key: `edge-${row}-${keys[index]}-${key}`,
    from: keys[index],
    to: key,
  }));
  edges.push(...connect(['dummy', ...output], 'output'));
  edges.push(...connect(sourceA.filter((key) => !consumed.has(key)), 'a'));
  edges.push(...connect(sourceB.filter((key) => !consumed.has(key)), 'b'));
  return linked(nodes, {
    edges,
    rowLabels: [{ label: 'A', y: 58 }, { label: 'B', y: 123 }, { label: 'out', y: 194 }],
    width: 480,
    height: 230,
    input: 'A:1->4->7; B:2->3->8',
    fixedPrefix: extra.fixedPrefix ?? 'empty',
    ...extra,
  });
}

const draft = visual('Compare the two frontier nodes, link the smaller after tail, and advance only its source cursor.', [
  frame('Initialize two sorted chains', 'tail=dummy, first=A1, second=B2, and dummy.next=null.', state([], 'dummy', 'A1', 'B2'), 'initialize-heads'),
  frame('Attach A1', '1 <= 2, so dummy.next=A1; advance first to A4 and tail to A1.', state(['A1'], 'A1', 'A4', 'B2', { comparison: '1 <= 2', fixedPrefix: '1' }), 'attach-a1'),
  frame('Attach B2', '4 > 2, so A1.next=B2; advance second to B3 and tail to B2.', state(['A1', 'B2'], 'B2', 'A4', 'B3', { comparison: '4 > 2', fixedPrefix: '1->2' }), 'attach-b2'),
  frame('Attach B3', '4 > 3, so B2.next=B3; advance second to B8 and tail to B3.', state(['A1', 'B2', 'B3'], 'B3', 'A4', 'B8', { comparison: '4 > 3', fixedPrefix: '1->2->3' }), 'attach-b3'),
  frame('Attach A4', '4 <= 8, so B3.next=A4; advance first to A7 and tail to A4.', state(['A1', 'B2', 'B3', 'A4'], 'A4', 'A7', 'B8', { comparison: '4 <= 8', fixedPrefix: '1->2->3->4' }), 'attach-a4'),
  frame('Attach A7', '7 <= 8, so A4.next=A7; advance first to null and tail to A7.', state(['A1', 'B2', 'B3', 'A4', 'A7'], 'A7', null, 'B8', { comparison: '7 <= 8', fixedPrefix: '1->2->3->4->7' }), 'attach-a7'),
  frame('Append the remaining suffix', 'first is null, so set A7.next=B8 and return dummy.next.', state(['A1', 'B2', 'B3', 'A4', 'A7', 'B8'], 'B8', null, null, { fixedPrefix: '1->2->3->4->7->8', result: '[1,2,3,4,7,8]' }), 'append-b8'),
]);

export default defineVisual('merge-two-sorted-lists', draft, {
  pattern: 'Two sorted linked-list cursors with a dummy output head and moving tail.',
  recognitionCue: 'Two sorted node streams must be merged by relinking existing nodes.',
  invariant: 'The path from dummy through tail is the consumed sorted prefix; first and second begin untouched suffixes. The old link after tail may remain until the next attachment overwrites it.',
  stateModel: 'Dummy, tail, first, second, and each node next pointer.',
  visualRationale: 'Two source chains keep their unconsumed suffixes visible while stable node identities move into the growing output chain. Drawn next edges and pointer labels expose each rewire directly.',
  rejectedAlternatives: ['Copying values hides rewiring.', 'A final chain skips decisions.', 'A pointer table omits the physical source and output chains.'],
  transferLesson: 'Emit the smaller frontier and advance only its source; append the remaining sorted suffix in constant time.',
  independentReview: '3.4 source-to-frame replay',
  reviewStatus: 'reviewed',
});

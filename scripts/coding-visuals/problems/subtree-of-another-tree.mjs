import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const topology = [
  ['node', 'value', 'left child', 'right child'],
  ['main3', '3', '-> main4', '-> main5'],
  ['main4', '4', '-> main1', '-> main2'],
  ['main1', '1', '-> null', '-> null'],
  ['main2', '2', '-> null', '-> null'],
  ['main5', '5', '-> null', '-> null'],
  ['sub4', '4', '-> sub1', '-> sub2'],
  ['sub1', '1', '-> null', '-> null'],
  ['sub2', '2', '-> null', '-> null'],
];
const rowByNode = new Map(topology.map((row, index) => [row[0], index]));
const trees = (activeNodes, extra = {}) => grid(topology, activeNodes.map((node, index) => ({
  row: rowByNode.get(node),
  col: 1,
  label: index === 0 ? 'first' : 'second',
  tone: index === 0 ? 'focus' : 'state',
  key: index === 0 ? 'first-cursor' : 'second-cursor',
})), {
  mainTree: 'main3(main4(main1,main2),main5)',
  subTree: 'sub4(sub1,sub2)',
  motion: [
    ...topology.flatMap((row, rowIndex) => row.map((value, colIndex) => ({
      key: `topology-${rowIndex}-${colIndex}`,
      kind: 'cell',
      x: colIndex,
      y: rowIndex,
      label: value,
    }))),
    ...activeNodes.map((node, index) => ({
      key: index === 0 ? 'first-cursor' : 'second-cursor',
      kind: 'pointer',
      x: 1,
      y: rowByNode.get(node),
      label: node,
    })),
  ],
  ...extra,
});

const draft = visual('Search each main-tree node as a candidate root, then require equal values and equal left and right topology.', [
  frame('Start at the main root', 'Call is_subtree(main3, sub4); same compares roots 3 and 4 before any child search.', trees(['main3', 'sub4'], {
    callStack: 'is_subtree(main3, sub4) -> same(main3, sub4)',
    comparison: '3 != 4',
  }), 'compare-main-root'),
  frame('Search the left candidate', 'same(main3, sub4) is false, so short-circuit OR continues with is_subtree(main4, sub4).', trees(['main4'], {
    callStack: 'is_subtree(main3, sub4) -> is_subtree(main4, sub4)',
    comparison: 'candidate roots 4 == 4',
  }), 'search-left'),
  frame('Compare matching roots', 'same(main4, sub4) sees 4 == 4, so its AND must next prove both left subtrees equal.', trees(['main4', 'sub4'], {
    visited: ['main4', 'sub4'],
    frontier: ['main1', 'sub1'],
    callStack: 'same(main4, sub4) -> same(main1, sub1)',
  }), 'match-candidate-root'),
  frame('Prove the left branches', 'same(main1, sub1) sees 1 == 1; both pairs of missing children return true by identity.', trees(['main1', 'sub1'], {
    visited: ['main4', 'sub4', 'main1', 'sub1'],
    frontier: ['null == null twice'],
    comparison: '1 == 1 and both child shapes match',
  }), 'match-left-branch'),
  frame('Prove the right branches', 'After the left call returns true, same(main2, sub2) sees 2 == 2 and matching missing children.', trees(['main2', 'sub2'], {
    visited: ['main4', 'sub4', 'main1', 'sub1', 'main2', 'sub2'],
    frontier: ['null == null twice'],
    comparison: '2 == 2 and both child shapes match',
  }), 'match-right-branch'),
  frame('Return the successful candidate', 'Both child comparisons return true, so same(main4, sub4) and then the outer OR return true.', trees(['main4', 'sub4'], {
    visited: ['main4', 'sub4', 'main1', 'sub1', 'main2', 'sub2'],
    callStack: 'same(main4, sub4) -> true',
    result: 'true',
  }), 'return-true'),
]);

const review = {
  pattern: 'Depth-first candidate search plus recursive full-tree equality.',
  recognitionCue: 'Use this composition when one complete rooted tree must appear inside another: candidate roots may occur anywhere, but a candidate succeeds only if every value and missing-child position matches.',
  invariant: 'is_subtree searches every reachable candidate root until one same call succeeds; same(first, second) returns true exactly when the two rooted trees have equal values and recursively identical left and right topology.',
  stateModel: 'The recursion stack holds either a candidate-search node paired with the fixed subroot or a same-tree node pair; no visited set is needed because trees have unique parent paths.',
  visualRationale: 'A compact child-slot topology keeps every real edge and null position visible while paired cursors move through candidate roots and corresponding children.',
  rejectedAlternatives: [
    'Array serialization can support another algorithm, but it hides the recursive parent-child checks used by this implementation.',
    'A value-only node list cannot distinguish equal values with different missing-child topology.',
    'A prose call table records recursion but does not show which actual tree edges each call follows.',
  ],
  transferLesson: 'Separate locate from verify: DFS locates candidate roots, then a stricter recursive predicate proves complete structure; the same composition applies to tree patterns, AST fragments, and directory subtrees.',
  independentReview: '3.3 source-to-frame replay',
  reviewStatus: 'reviewed',
};

export default defineVisual('subtree-of-another-tree', draft, review);

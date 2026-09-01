import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const topology = [
  ['', '', '', '3', '', '', ''],
  ['', '', '/', '', '\\', '', ''],
  ['', '9', '', '', '', '20', ''],
  ['', '', '', '', '/', '', '\\'],
  ['', '', '', '', '15', '', '7'],
];
const positions = { '3': [0, 3], '9': [2, 1], '20': [2, 5], '15': [4, 4], '7': [4, 6] };
const scene = (active, extra = {}) => grid(topology, [{
  row: positions[active][0],
  col: positions[active][1],
  label: `current root ${active}`,
  tone: extra.result ? 'output' : 'focus',
  key: 'active-call',
}], {
  activeRoot: active,
  preorder: '[3, 9, 20, 15, 7]',
  inorder: '[9, 3, 15, 20, 7]',
  positions: '9:0, 3:1, 15:2, 20:3, 7:4',
  ...extra,
});

const draft = visual('The next preorder value becomes a root, and its inorder position partitions the exact ranges passed to its left and right recursive calls.', [
  frame('Initialize traversal state', 'Build the inorder position map, set preorder_index=0, and call build(0,4).',   scene('3', { callStack: 'build(0,4)', preorderIndex: '0' }), 'initialize'),
  frame('Create root 3', 'Read preorder[0]=3, advance index to 1, and split inorder range [0,4] at middle=1 into left [0,0] and right [2,4].', scene('3', { callStack: '3 build(0,4)', split: '[0,0] | 3 | [2,4]', preorderIndex: '1' }), 'root-3'),
  frame('Create left child 9', 'build(0,0) reads preorder[1]=9 and advances to 2. middle=0 gives empty calls build(0,-1) and build(1,0), both returning None.', scene('9', { callStack: '3 build(0,4) -> 9 build(0,0)', split: 'empty | 9 | empty', baseCases: 'build(0,-1)=None; build(1,0)=None', preorderIndex: '2' }), 'node-9'),
  frame('Create right child 20', 'After 9 returns, build(2,4) reads preorder[2]=20 and advances to 3. middle=3 splits [2,2] and [4,4].', scene('20', { callStack: '3 build(0,4) -> 20 build(2,4)', split: '[2,2] | 20 | [4,4]', preorderIndex: '3' }), 'node-20'),
  frame('Create 20 left child 15', 'build(2,2) reads preorder[3]=15 and advances to 4. Its ranges build(2,1) and build(3,2) are empty and return None.', scene('15', { callStack: '3 -> 20 -> 15 build(2,2)', split: 'empty | 15 | empty', baseCases: 'build(2,1)=None; build(3,2)=None', preorderIndex: '4' }), 'node-15'),
  frame('Create 20 right child 7', 'After 15 returns, build(4,4) reads preorder[4]=7 and advances to 5. Both child ranges are empty.', scene('7', { callStack: '3 -> 20 -> 7 build(4,4)', split: 'empty | 7 | empty', baseCases: 'build(4,3)=None; build(5,4)=None', preorderIndex: '5' }), 'node-7'),
  frame('Return the reconstructed tree', 'Return 7, then 20, then 3. Preorder_index=5 has consumed every value, and the real parent-child edges match both traversals.', scene('3', { callStack: 'empty after returns 7 -> 20 -> 3', verification: 'preorder 3,9,20,15,7; inorder 9,3,15,20,7', result: '3(9, 20(15, 7))' }), 'return-tree'),
]);

export default defineVisual('construct-tree-from-preorder-and-inorder-traversal', draft, {
  pattern: 'Recursive reconstruction: preorder chooses roots and inorder partitions subtree ranges.',
  recognitionCue: 'Unique node values are given in preorder and inorder, so root order and left/right membership complement each other to determine one binary tree.',
  invariant: 'build(left,right) consumes exactly the next preorder values belonging to inorder[left:right+1] and returns precisely that subtree. Empty ranges consume nothing and return None.',
  stateModel: 'Keep the inorder value-to-index map, one shared preorder_index, recursive left/right bounds, and constructed nodes. The call stack preserves unfinished parent assignments.',
  visualRationale: 'A compact coordinate grid draws every binary-tree node and parent-child edge with ASCII geometry, while a stable active-root key moves through inorder ranges, splits, base cases, and returns at readable mobile size.',
  rejectedAlternatives: [
    'A table of traversal lists does not depict parent-child topology or recursive returns.',
    'Repeated list slicing hides the shared preorder cursor and adds avoidable allocation.',
    'A final tree alone does not prove how inorder ranges assign left versus right children.',
  ],
  transferLesson: 'Combine one traversal that selects the next root with another that partitions membership, and recurse on index ranges; the same decomposition works with inorder plus postorder by choosing roots from the opposite end.',
  reviewStatus: 'reviewed',
});

import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const topology = [
  ['', '', '', '5', '', '', ''],
  ['', '', '/', '', '\\', '', ''],
  ['', '3', '', '', '', '7', ''],
  ['/', '', '\\', '', '/', '', '\\'],
  ['2', '', '4', '', '6', '', '8'],
  ['|', '', '', '', '', '', ''],
  ['1', '', '', '', '', '', ''],
];
const positions = { '5': [0, 3], '3': [2, 1], '7': [2, 5], '2': [4, 0], '4': [4, 2], '1': [6, 0] };
const state = (node, label, extra) => grid(topology, [{
  row: positions[node][0],
  col: positions[node][1],
  label,
  tone: label.includes('return') ? 'output' : 'focus',
  key: 'inorder-cursor',
}], {
  input: 'tree=[5,3,7,2,4,6,8,1], k=4',
  cursor: label,
  ...extra,
});

const draft = visual('Iterative inorder exposes BST values in ascending order, so the fourth pop is the answer.', [
  frame('Initialize traversal', 'Set root=5, stack=[], and remaining k=4.', state('5', 'root=5', { stack: '[]', remaining: '4' }), 'initialize'),
  frame('Push 5', 'Push root 5 and move root to its left child 3.', state('3', 'root=3', { stack: '[5]', remaining: '4' }), 'push-five'),
  frame('Push 3', 'Push 3 and move root left to 2.', state('2', 'root=2', { stack: '[5,3]', remaining: '4' }), 'push-three'),
  frame('Push 2', 'Push 2 and move root left to 1.', state('1', 'root=1', { stack: '[5,3,2]', remaining: '4' }), 'push-two'),
  frame('Push 1 and reach null', 'Push 1; its left child is null, so the inner loop stops with 1 on top.', state('1', 'root=null after 1', { stack: '[5,3,2,1] top=1', remaining: '4' }), 'push-one'),
  frame('Pop the first value', 'Pop 1, decrement remaining from 4 to 3, then set root=1.right=null.', state('1', 'visit 1', { stack: '[5,3,2]', arithmetic: '4 - 1 = 3' }), 'visit-one'),
  frame('Pop the second value', 'With root null, pop 2, decrement remaining from 3 to 2, then set root=2.right=null.', state('2', 'visit 2', { stack: '[5,3]', arithmetic: '3 - 1 = 2' }), 'visit-two'),
  frame('Pop the third value', 'Pop 3, decrement remaining from 2 to 1, then move root to 3.right at node 4.', state('4', 'root=4', { stack: '[5]', arithmetic: '2 - 1 = 1', visitedValues: '[1,2,3]' }), 'visit-three'),
  frame('Push the next left spine', 'Push node 4 and move root to 4.left=null.', state('4', 'root=null after 4', { stack: '[5,4] top=4', remaining: '1' }), 'push-four'),
  frame('Pop the fourth value and stop', 'Pop 4 and decrement remaining from 1 to 0. Because k is now zero, return 4 before exploring larger nodes.', state('4', 'return 4', { stack: '[5]', arithmetic: '1 - 1 = 0', result: '4' }), 'return-four'),
]);

const review = {
  pattern: 'Iterative inorder traversal with an explicit ancestor stack and early rank stopping.',
  recognitionCue: 'The input is a BST and the question asks for an order statistic, so inorder traversal produces the required sorted rank without sorting all values.',
  invariant: 'Before each pop, the stack top is the smallest unvisited node reachable from the processed search frontier; completed pops are in strictly ascending BST order.',
  stateModel: 'Keep only the current root pointer, the stack of deferred ancestors, and the remaining rank k; push a full left spine, pop once, decrement k, then explore the popped node’s right child.',
  visualRationale: 'A compact coordinate grid keeps every BST node and ASCII parent-child edge fixed while a stable inorder cursor moves; stack and k arithmetic change at every push and pop.',
  rejectedAlternatives: [
    'Flattening and sorting every value hides the BST ordering and costs extra O(n log n) time.',
    'A sorted output row omits the deferred-ancestor stack that makes iterative inorder work.',
    'A recursive call tree obscures the explicit stack used by the supplied implementation.',
  ],
  transferLesson: 'Use inorder as a sorted stream for BST rank, range, predecessor, and successor tasks, and stop as soon as the requested portion of that stream is consumed.',
  reviewStatus: 'reviewed',
};

export default defineVisual('kth-smallest-element-in-a-bst', draft, review);

import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const topology = [
  ['', '', '', '6', '', '', ''],
  ['', '', '/', '', '\\', '', ''],
  ['', '2', '', '', '', '8', ''],
  ['/', '', '\\', '', '/', '', '\\'],
  ['0', '', '4', '', '7', '', '9'],
  ['', '/', '', '\\', '', '', ''],
  ['', '3', '', '5', '', '', ''],
];
const positions = { '6': [0, 3], '2': [2, 1], '4': [4, 2] };
const nodePositions = {
  '6': { x: 230, y: 45 },
  '2': { x: 100, y: 145 },
  '8': { x: 350, y: 145 },
  '0': { x: 40, y: 245 },
  '4': { x: 175, y: 245 },
  '7': { x: 290, y: 245 },
  '9': { x: 410, y: 245 },
  '3': { x: 135, y: 345 },
  '5': { x: 225, y: 345 },
};
const at = (node, label, extra = {}) => grid(topology, [
  {
    row: positions[node][0],
    col: positions[node][1],
    label,
    tone: label === 'split: return 4' ? 'output' : 'focus',
    key: 'current-root',
  },
  { row: 6, col: 1, label: 'first=3', tone: 'state', key: 'first-target' },
  { row: 6, col: 3, label: 'second=5', tone: 'state', key: 'second-target' },
], {
  input: 'first=3, second=5',
  currentRoot: label,
  targets: 'first=3, second=5',
  graphCompanion: {
    nodes: ['6','2','8','0','4','7','9','3','5'],
    edges: ['6->2','6->8','2->0','2->4','8->7','8->9','4->3','4->5'],
    positions: nodePositions,
    width: 460,
    height: 420,
    treeLayout: true,
    start: node,
    visited: ['3','5'],
    markerId: `bst-lca-${node}-${label.replace(/[^a-z0-9]+/gi, '-')}`,
    edgeLabels: false,
    ariaLabel: 'BST rooted at 6: route 6 to 2 to 4 reaches the split between targets 3 and 5',
    regions: [
      { x: nodePositions[node].x, y: nodePositions[node].y + 40, label },
      { x: 135, y: 385, label: 'first=3' },
      { x: 225, y: 385, label: 'second=5' },
    ],
  },
  gridSummary: 'Original coordinate inventory',
  ...extra,
});

const draft = visual('BST ordering discards one whole side until the targets split at their lowest shared ancestor.', [
  frame(
    'Start at root 6',
    'Both targets are smaller: 3 < 6 and 5 < 6. The first branch moves root to 6.left at node 2.',
    at('6', 'root=6', { decision: 'both smaller -> move left to 2' }),
    'move-left',
  ),
  frame(
    'Compare at node 2',
    'Both targets are larger: 3 > 2 and 5 > 2. The second branch moves root to 2.right at node 4.',
    at('2', 'root=2', { decision: 'both larger -> move right to 4' }),
    'move-right',
  ),
  frame(
    'Find the first split at node 4',
    'Target 3 is smaller than 4 while target 5 is larger, so neither same-side branch applies.',
    at('4', 'root=4', { comparison: '3 < 4 < 5', decision: 'targets split -> stop' }),
    'find-split',
  ),
  frame(
    'Return the lowest common ancestor',
    'Earlier nodes 6 and 2 contain both targets but are higher. Node 4 is the first node on the search path whose subtree separates them.',
    at('4', 'split: return 4', { path: '6 -> 2 -> 4', result: '4' }),
    'return-four',
  ),
]);

const review = {
  pattern: 'BST-guided single-path search for the first target split.',
  recognitionCue: 'Both target nodes are in a BST, so comparing both values with the current root reveals whether their lowest common ancestor must be entirely left, entirely right, or current.',
  invariant: 'At the start of every loop iteration, the current root’s subtree contains both targets; moving only when both values are on the same side preserves that fact.',
  stateModel: 'Retain the current root and the two fixed target values. Move left if both are smaller, right if both are larger, otherwise return the current root.',
  visualRationale: 'An actual node-and-edge BST keeps both target labels fixed while the current root moves. The original coordinate inventory retains its cursor and target motion keys in a disclosure.',
  rejectedAlternatives: [
    'Two root-to-target path lists find a common prefix but ignore the BST ordering shortcut.',
    'A generic binary-tree DFS explores unnecessary branches and needs recursion state.',
    'A three-row comparison table hides where targets and discarded subtrees lie in the tree.',
  ],
  transferLesson: 'In an ordered search tree, track where multiple target values fall relative to the current separator; the first separator they do not share is their lowest common routing point.',
  reviewStatus: 'reviewed',
};

export default defineVisual('lowest-common-ancestor-in-a-bst', draft, review);

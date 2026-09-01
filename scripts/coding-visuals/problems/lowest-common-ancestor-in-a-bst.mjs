import { defineVisual, frame, mark, tree, visual } from '../primitives.mjs';

const levels = [['6'], ['2', '8'], ['0', '4', '7', '9'], ['-', '-', '3', '5', '-', '-', '-', '-']];
const at = (index, label, extra = {}) => tree(
  levels,
  [
    mark(index, label, label === 'split: return 4' ? 'output' : 'focus', 'current-root'),
    mark(9, 'first=3', 'state', 'first-target'),
    mark(10, 'second=5', 'state', 'second-target'),
  ],
  { input: 'first=3, second=5', ...extra },
);

const draft = visual('BST ordering discards one whole side until the targets split at their lowest shared ancestor.', [
  frame(
    'Start at root 6',
    'Both targets are smaller: 3 < 6 and 5 < 6. The first branch moves root to 6.left at node 2.',
    at(0, 'root=6', { decision: 'both smaller -> move left to 2' }),
    'move-left',
  ),
  frame(
    'Compare at node 2',
    'Both targets are larger: 3 > 2 and 5 > 2. The second branch moves root to 2.right at node 4.',
    at(1, 'root=2', { decision: 'both larger -> move right to 4' }),
    'move-right',
  ),
  frame(
    'Find the first split at node 4',
    'Target 3 is smaller than 4 while target 5 is larger, so neither same-side branch applies.',
    at(4, 'root=4', { comparison: '3 < 4 < 5', decision: 'targets split -> stop' }),
    'find-split',
  ),
  frame(
    'Return the lowest common ancestor',
    'Earlier nodes 6 and 2 contain both targets but are higher. Node 4 is the first node on the search path whose subtree separates them.',
    at(4, 'split: return 4', { path: '6 -> 2 -> 4', result: '4' }),
    'return-four',
  ),
]);

const review = {
  pattern: 'BST-guided single-path search for the first target split.',
  recognitionCue: 'Both target nodes are in a BST, so comparing both values with the current root reveals whether their lowest common ancestor must be entirely left, entirely right, or current.',
  invariant: 'At the start of every loop iteration, the current root’s subtree contains both targets; moving only when both values are on the same side preserves that fact.',
  stateModel: 'Retain the current root and the two fixed target values. Move left if both are smaller, right if both are larger, otherwise return the current root.',
  visualRationale: 'A fixed BST with independently keyed current-root and target markers shows the real search path, the discarded subtrees, and the exact split geometry.',
  rejectedAlternatives: [
    'Two root-to-target path lists find a common prefix but ignore the BST ordering shortcut.',
    'A generic binary-tree DFS explores unnecessary branches and needs recursion state.',
    'A three-row comparison table hides where targets and discarded subtrees lie in the tree.',
  ],
  transferLesson: 'In an ordered search tree, track where multiple target values fall relative to the current separator; the first separator they do not share is their lowest common routing point.',
  reviewStatus: 'reviewed',
};

export default defineVisual('lowest-common-ancestor-in-a-bst', draft, review);

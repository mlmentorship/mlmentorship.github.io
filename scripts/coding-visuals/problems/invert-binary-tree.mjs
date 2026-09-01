import { defineVisual, frame, mark, tree, visual } from '../primitives.mjs';

const original = [['4'], ['2', '7'], ['1', '3', '6', '9']];

const draft = visual('Postorder recursion returns inverted child subtrees, then the parent assigns old right to left and old left to right.', [
  frame(
    'Start with the original links',
    'For tree [4,2,7,1,3,6,9], call invert_tree(4). The root currently links left to 2 and right to 7.',
    tree(original, [mark(0, 'call invert(4)', 'focus')], { callStack: '[4]', links: '4.left=2; 4.right=7' }),
    'call-root',
  ),
  frame(
    'Evaluate the old right subtree first',
    'The tuple right-hand side evaluates invert_tree(root.right) first. At node 7, invert 9 and 6; both leaves keep their values and null children.',
    tree(original, [mark(2, 'invert old right', 'focus'), mark(5, 'returns 6', 'state'), mark(6, 'returns 9', 'state')], {
      callStack: '[4,7]',
      evaluationOrder: 'invert(9), then invert(6)',
    }),
    'invert-right-subtree',
  ),
  frame(
    'Rewire node 7',
    'Assign node 7 left = inverted old right (9) and right = inverted old left (6). Stable nodes 9 and 6 visibly exchange child positions.',
    tree([['4'], ['2', '7'], ['1', '3', '9', '6']], [mark(2, '7: left=9, right=6', 'output')], {
      returnValue: 'inverted subtree rooted at 7',
      callStack: '[4]',
    }),
    'rewire-7',
  ),
  frame(
    'Invert and rewire the old left subtree',
    'Next evaluate invert_tree(2): invert old right 3, then old left 1, and assign node 2 left = 3 and right = 1.',
    tree([['4'], ['2', '7'], ['3', '1', '9', '6']], [mark(1, '2: left=3, right=1', 'output')], {
      evaluationOrder: 'invert(3), then invert(1)',
      callStack: '[4]',
    }),
    'rewire-2',
  ),
  frame(
    'Swap the root links',
    'Both recursive results are ready. Assign root 4 left = inverted old right subtree 7 and right = inverted old left subtree 2.',
    tree([['4'], ['7', '2'], ['9', '6', '3', '1']], [
      mark(0, 'left=7, right=2', 'output'),
      mark(1, 'moved from right', 'state'),
      mark(2, 'moved from left', 'state'),
    ], { callStack: '[]', result: '[4,7,2,9,6,3,1]' }),
    'rewire-root',
  ),
]);

const review = {
  pattern: 'Postorder tree transformation that recursively rewrites child subtrees before assigning their links at the parent.',
  recognitionCue: 'Use this pattern when every node applies the same local child-pointer transformation and the parent needs transformed child roots returned from recursion.',
  invariant: 'invert_tree(node) returns the same node object after every edge in its subtree is mirrored: its new left is the fully inverted old right and its new right is the fully inverted old left.',
  stateModel: 'The minimal state is the recursion stack and the two returned child roots held while tuple assignment rewires a node. No copied tree or auxiliary map is needed.',
  visualRationale: 'A real edged tree makes each pointer rewire visible. Semantic node keys persist while 9/6, 3/1, and whole subtrees 7/2 move to mirrored positions, so motion communicates identity preservation.',
  rejectedAlternatives: [
    'A before/after pair was rejected because it skips recursive evaluation order and all intermediate rewires.',
    'An array representation was rejected because heap-like indices distract from mutable parent-child links.',
    'A breadth-first queue animation was rejected because it does not match the supplied recursive postorder implementation.',
  ],
  transferLesson: 'For recursive structural edits, first obtain transformed child roots, then reconnect them locally and return the current root; this transfers to pruning, flattening, and persistent tree rewrites.',
  reviewStatus: 'reviewed',
};

export default defineVisual('invert-binary-tree', draft, review);

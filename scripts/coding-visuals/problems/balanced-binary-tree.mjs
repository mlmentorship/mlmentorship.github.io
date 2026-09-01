import { defineVisual, frame, mark, tree, visual } from '../primitives.mjs';

const levels = [['1'], ['2', '5'], ['3', '-', '6', '7'], ['4', '-', '-', '-', '-', '-', '-', '-']];

const draft = visual('A postorder height function returns a nonnegative height for a balanced subtree and -1 as a failure sentinel that ancestors propagate immediately.', [
  frame(
    'Start bottom-up height evaluation',
    'For tree [1,2,5,3,null,6,7,4], call height(1), then descend its left chain 1 -> 2 -> 3 -> 4 before evaluating any right sibling.',
    tree(levels, [mark(0, 'stack start', 'state'), mark(1, 'stack', 'state'), mark(3, 'stack', 'state'), mark(7, 'current', 'focus')], {
      callStack: '[1,2,3,4]',
      untouchedRightSubtree: '5 with children 6,7',
    }),
    'descend-left-chain',
  ),
  frame(
    'Return height 1 from leaf 4',
    'Node 4 gets left = 0 and right = 0 from its None children. The difference is 0, so return 1 + max(0,0) = 1.',
    tree(levels, [mark(7, 'return height 1', 'output')], {
      arithmetic: '|0-0| = 0 <= 1; 1 + max(0,0) = 1',
      callStack: '[1,2,3]',
    }),
    'return-leaf-4',
  ),
  frame(
    'Return height 2 from node 3',
    'Node 3 receives left = 1 and right = 0. Since |1-0| = 1, it is balanced and returns 1 + max(1,0) = 2.',
    tree(levels, [mark(3, 'return height 2', 'output'), mark(7, 'left height 1', 'state')], {
      arithmetic: '|1-0| = 1; 1 + max(1,0) = 2',
      callStack: '[1,2]',
    }),
    'return-node-3',
  ),
  frame(
    'Emit the failure sentinel at node 2',
    'Node 2 receives left = 2 and right = 0. Because |2-0| = 2 > 1, it returns -1 instead of a height.',
    tree(levels, [mark(1, 'return -1', 'warning'), mark(3, 'left height 2', 'state')], {
      arithmetic: '|2-0| = 2 > 1',
      sentinel: '-1 means unbalanced',
      callStack: '[1]',
    }),
    'emit-sentinel',
  ),
  frame(
    'Propagate -1 and skip the right subtree',
    'Root 1 receives left = -1. The `if left < 0` branch returns -1 immediately, so nodes 5, 6, and 7 are never visited.',
    tree(levels, [
      mark(0, 'propagate -1', 'warning'),
      mark(1, 'left returned -1', 'warning'),
      mark(2, 'skipped', 'state'),
      mark(5, 'skipped', 'state'),
      mark(6, 'skipped', 'state'),
    ], { branch: 'left < 0 -> return -1', visited: '1,2,3,4 only' }),
    'propagate-sentinel',
  ),
  frame(
    'Convert the sentinel to false',
    'The outer function checks height(root) >= 0. Here -1 >= 0 is false, so the tree is not height-balanced.',
    tree(levels, [mark(0, 'height(root) = -1', 'output')], {
      arithmetic: '-1 >= 0 -> false',
      result: 'false',
    }),
    'return-false',
  ),
]);

const review = {
  pattern: 'Bottom-up tree DFS that fuses height computation and validation by reserving -1 as an error sentinel.',
  recognitionCue: 'Use a sentinel summary when a parent needs a normal child aggregate but any descendant failure should abort remaining work, as in balanced-height checks or invalid-subtree detection.',
  invariant: 'height(node) returns the exact nonnegative subtree height if every node below is balanced; otherwise it returns -1. Thus ancestors can distinguish valid data from failure without a second traversal.',
  stateModel: 'The minimal state is the recursion stack and each child integer return. The left result is checked before the right call, enabling the supplied implementation to skip the entire right subtree after failure.',
  visualRationale: 'An edged, asymmetric tree shows the height-producing left chain and the untouched right subtree. Return labels and explicit arithmetic distinguish heights, the -1 sentinel, and skipped calls without color.',
  rejectedAlternatives: [
    'A height table was rejected because it hides postorder call flow and early sentinel propagation.',
    'Two separate passes for height and balance were rejected because they depict a different, potentially repeated-work algorithm.',
    'A simple balanced/unbalanced icon was rejected because it does not reveal where the first violation occurs or why work stops.',
  ],
  transferLesson: 'Fuse a subtree summary with validation by reserving an impossible summary value for failure, then check it before doing more recursion; this generalizes to BST validation and parse-tree error propagation.',
  reviewStatus: 'reviewed',
};

export default defineVisual('balanced-binary-tree', draft, review);

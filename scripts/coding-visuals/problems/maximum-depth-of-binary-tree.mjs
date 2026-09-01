import { defineVisual, frame, mark, tree, visual } from '../primitives.mjs';

const levels = [['3'], ['9', '20'], ['-', '-', '15', '7']];

const draft = visual('Each recursive call returns the height of its subtree, so a parent needs only 1 + max(left height, right height).', [
  frame(
    'Start the root call',
    'For tree [3,9,20,null,null,15,7], call max_depth(3). The stack contains [3], and neither child height is known yet.',
    tree(levels, [mark(0, 'call max_depth(3)', 'focus')], { callStack: '[3]', knownReturns: 'none' }),
    'call-root',
  ),
  frame(
    'Resolve node 9 from null children',
    'The left call reaches node 9. Both child calls receive None and return 0, so node 9 returns 1 + max(0,0) = 1.',
    tree(levels, [mark(1, 'return 1', 'output')], { callStack: '[3,9]', baseCases: 'None -> 0 twice', arithmetic: '1 + max(0,0) = 1' }),
    'return-9',
  ),
  frame(
    'Resolve leaves 15 and 7',
    'Under node 20, each leaf also receives two zero returns from None and returns 1.',
    tree(levels, [mark(5, 'return 1', 'state'), mark(6, 'return 1', 'state')], {
      callStack: '[3,20]',
      returns: 'depth(15)=1; depth(7)=1',
    }),
    'return-leaves',
  ),
  frame(
    'Combine at node 20',
    'Node 20 receives left = 1 and right = 1, then returns 1 + max(1,1) = 2 to root 3.',
    tree(levels, [mark(2, 'return 2', 'output'), mark(5, 'left 1', 'state'), mark(6, 'right 1', 'state')], {
      callStack: '[3,20]',
      arithmetic: '1 + max(1,1) = 2',
    }),
    'return-20',
  ),
  frame(
    'Combine at the root',
    'Root 3 receives left = 1 from node 9 and right = 2 from node 20, so it returns 1 + max(1,2) = 3.',
    tree(levels, [mark(0, 'return 3', 'output'), mark(1, 'left 1', 'state'), mark(2, 'right 2', 'state')], {
      callStack: '[]',
      arithmetic: '1 + max(1,2) = 3',
      result: '3',
    }),
    'return-root',
  ),
]);

const review = {
  pattern: 'Bottom-up depth-first recursion that folds two child heights into one subtree height.',
  recognitionCue: 'Use this pattern when a tree property for a node can be computed only after both child subtrees return summaries, especially height, diameter contributions, or root-to-leaf aggregates.',
  invariant: 'Whenever max_depth(node) returns d, d is exactly the number of real nodes on the longest path from node to a leaf; max_depth(None) returns 0, making a leaf return 1.',
  stateModel: 'The minimal state is the recursion stack plus one integer return per completed child. No global visited set or root-to-node path is required because tree edges define unique subproblems.',
  visualRationale: 'An edged binary tree keeps parent-child geometry visible while call/return labels move from leaves toward the root. Arithmetic annotations expose every base case and combination without relying on color.',
  rejectedAlternatives: [
    'A level-order queue was rejected because it computes depth by a different BFS implementation and hides recursive returns.',
    'A flat call table was rejected because it removes the child topology that explains which two heights combine.',
    'A highlighted longest path alone was rejected because it shows the answer but not how all recursive calls establish it.',
  ],
  transferLesson: 'Define the null identity first, ask each child for the smallest sufficient summary, and combine those summaries once; the same postorder fold supports subtree size, height, and path calculations.',
  reviewStatus: 'reviewed',
};

export default defineVisual('maximum-depth-of-binary-tree', draft, review);

import { defineVisual, frame, mark, tree, visual } from '../primitives.mjs';

const levels = [['-10'], ['9', '20'], ['-4', '-', '15', '7']];
const nodeIndexes = { '-10': 0, '9': 1, '20': 2, '-4': 3, '15': 5, '7': 6 };
const treeScene = (active, label, extra = {}, output = []) => tree(levels, [
  mark(nodeIndexes[active], label, output.includes(active) ? 'output' : 'focus', 'active-node'),
  ...output.filter((value) => value !== active).map((value) => mark(nodeIndexes[value], 'best path', 'output', `path-${value}`)),
], {
  ...extra,
  motion: [
    { key: 'active-node', kind: 'pointer', x: nodeIndexes[active], y: 0, label: active },
  ],
});

const draft = visual('Postorder returns one extendable branch to a parent while scoring a complete path through each node with both nonnegative child gains.', [
  frame('Initialize at the root value', 'For the shown tree, set global best = -10. Postorder must finish child gains before scoring each parent.', treeScene('-10', 'call root', {
    callStack: 'one_branch(-10)',
    best: '-10',
    next: 'descend left',
  }), 'initialize'),
  frame('Return the negative leaf', 'Node -4 has two null gains of 0. Its local path and returned branch are both -4, while best remains -4 after max(-10,-4).', treeScene('-4', 'return -4', {
    callStack: '-10 -> 9 -> -4',
    localScore: '-4 + 0 + 0 = -4',
    returnedGain: 'return -4',
    best: '-10 -> -4',
  }), 'visit-negative-4'),
  frame('Clamp the actual negative child', 'At node 9, the left child returned -4, so max(0, -4) = 0; the local score and one-branch return are both 9.', treeScene('9', 'return 9', {
    callStack: '-10 -> 9',
    childGains: 'left = max(0, -4) = 0, right = 0',
    localScore: '9 + 0 + 0 = 9',
    returnedGain: 'return 9',
    best: '-4 -> 9',
  }), 'visit-9'),
  frame('Return leaf 15', 'Node 15 receives two null gains, scores 15, returns 15, and raises the global best from 9 to 15.', treeScene('15', 'return 15', {
    callStack: '-10 -> 20 -> 15',
    localScore: '15 + 0 + 0 = 15',
    returnedGain: 'return 15',
    best: '9 -> 15',
  }), 'visit-15'),
  frame('Return leaf 7', 'Node 7 scores and returns 7. Its local score does not exceed the current global best 15.', treeScene('7', 'return 7', {
    callStack: '-10 -> 20 -> 7',
    localScore: '7 + 0 + 0 = 7',
    returnedGain: 'return 7',
    best: '15',
  }), 'visit-7'),
  frame('Score both branches at node 20', 'Both child gains are positive, so the complete path through 20 scores 20 + 15 + 7 = 42; only the larger branch can continue upward, so return 35.', treeScene('20', 'score 42; return 35', {
    callStack: '-10 -> 20',
    localScore: '20 + 15 + 7 = 42',
    returnedGain: '20 + max(15, 7) = 35; return 35',
    best: '15 -> 42',
  }, ['15', '20', '7']), 'visit-20'),
  frame('Finish at the root', 'The root receives gains 9 and 35. Its through-path is -10 + 9 + 35 = 34, which cannot beat 42; it returns -10 + 35 = 25.', treeScene('-10', 'score 34; return 25', {
    callStack: 'one_branch(-10)',
    localScore: '-10 + 9 + 35 = 34',
    returnedGain: '-10 + max(9, 35) = 25; return 25',
    best: '42',
  }), 'visit-negative-10'),
  frame('Return the global maximum', 'The completed path 15 -> 20 -> 7 has sum 42. It can use both branches because it ends locally rather than extending to a parent.', treeScene('20', 'maximum 42', {
    path: '15 -> 20 -> 7',
    arithmetic: '15 + 20 + 7 = 42',
    result: '42',
  }, ['15', '20', '7']), 'result'),
]);

const review = {
  pattern: 'Postorder tree DP: return one branch upward, but score two branches locally.',
  recognitionCue: 'Use it when a connected tree path may bend once at a highest node, while any value returned to a parent must remain a single non-branching chain.',
  invariant: 'After a node finishes, its return is the maximum sum of one downward branch starting there; global best is the maximum complete path in every finished subtree, and negative child returns contribute zero.',
  stateModel: 'The minimal recursive state is each node’s left and right clamped gains plus one global best. The trace retains real parent-child edges, active call position, postorder return, local two-branch score, and best update.',
  visualRationale: 'A stable tree topology with an authored active-node key exposes the distinction between returned branch and locally completed path; formulas and call-stack labels remain complete without color, JavaScript, or memorized code.',
  rejectedAlternatives: [
    'A values-only postorder table was rejected because it hides which child gains feed each parent.',
    'A recursion tree of function calls was rejected because it duplicates the actual binary-tree topology.',
    'A final highlighted path was rejected because it does not explain why only one branch may return upward.',
  ],
  transferLesson: 'In tree DP, separate what a parent is allowed to extend from what may finish at the current node; clamp harmful optional branches, return the extendable shape, and update a global answer with the richer local shape.',
  reviewStatus: 'reviewed',
};

export default defineVisual('binary-tree-maximum-path-sum', draft, review);

import { defineVisual, frame, mark, tree, visual } from '../primitives.mjs';

const pairedLevels = [['A:1 = B:1'], ['A:2 = B:2', 'A:3 / B:null']];

const draft = visual('Two trees match only if the same-position node pair passes the null, value, left-pair, and right-pair checks.', [
  frame(
    'Pair the two roots',
    'Compare A = [1,2,3] with B = [1,2,null]. Both roots exist and 1 == 1, so Python continues across the and-chain to the left pair.',
    tree(pairedLevels, [mark(0, 'compare pair', 'focus')], { callStack: '[(A:1,B:1)]', check: 'both real; 1 == 1' }),
    'compare-roots',
  ),
  frame(
    'Compare the left pair',
    'Nodes A:2 and B:2 both exist and 2 == 2, so recurse into their left children.',
    tree(pairedLevels, [mark(1, '2 == 2', 'focus')], { callStack: '[(A:1,B:1),(A:2,B:2)]' }),
    'compare-left-values',
  ),
  frame(
    'Match paired missing children',
    'For (A:2.left, B:2.left) = (None,None), the first branch returns first is second, which is true. The right null pair also returns true.',
    tree(pairedLevels, [mark(1, 'null/null -> true', 'output')], {
      baseCases: '(None,None) -> true twice',
      returnTo: '(A:2,B:2)',
    }),
    'match-null-pairs',
  ),
  frame(
    'Return true for the left subtrees',
    'At pair (A:2,B:2), value equality and both child-pair results are true, so this call returns true and root evaluation reaches the right pair.',
    tree(pairedLevels, [mark(1, 'return true', 'output'), mark(2, 'next pair', 'focus')], {
      expression: 'true and true and true',
      callStack: '[(A:1,B:1)]',
    }),
    'return-left-pair',
  ),
  frame(
    'Reject one missing node',
    'For the root right pair, first is A:3 and second is None. The null branch returns A:3 is None, which is false.',
    tree(pairedLevels, [mark(2, 'real/null -> false', 'warning')], {
      branch: 'first is None or second is None',
      returnValue: 'first is second -> false',
    }),
    'reject-shape-mismatch',
  ),
  frame(
    'Short-circuit the root result',
    'The root expression is true value match and true left match and false right match, so same_tree returns false.',
    tree(pairedLevels, [mark(0, 'return false', 'output'), mark(2, 'shape differs', 'warning')], {
      expression: 'true and true and false',
      result: 'false',
    }),
    'return-false',
  ),
]);

const review = {
  pattern: 'Lockstep depth-first recursion over corresponding node pairs in two trees.',
  recognitionCue: 'Use paired DFS when equality, symmetry, or structural correspondence requires comparing both value and shape at the same tree positions rather than comparing traversed value sequences.',
  invariant: 'same_tree(first, second) returns true exactly when the two subtrees rooted at that pair have identical shape and equal values at every corresponding node.',
  stateModel: 'The minimal state is the recursion stack of node pairs. Each call resolves one of three cases: both missing, exactly one missing, or two real nodes whose value and child pairs must all match.',
  visualRationale: 'A paired-node tree preserves the shared positions and edges while each label shows both values at that position. Explicit null and boolean annotations make shape mismatches readable without color.',
  rejectedAlternatives: [
    'Two preorder value arrays were rejected because equal values can still hide different null positions and therefore different shapes.',
    'A comparison table was rejected because it removes parent-child topology and recursive pairing.',
    'Serialized strings were rejected because they depict a different encoding-based implementation.',
  ],
  transferLesson: 'When comparing recursive structures, recurse on aligned pairs and make missing-object cases explicit before reading values; this transfers to mirror symmetry, subtree equality, and AST comparison.',
  reviewStatus: 'reviewed',
};

export default defineVisual('same-tree', draft, review);

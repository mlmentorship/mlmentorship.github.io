import { defineVisual, frame, mark, tree, visual } from '../primitives.mjs';

const levels = [['1'], ['2', '3'], ['-', '-', '4', '-']];
const treeState = (index, label, extra = {}) => tree(
  levels,
  [mark(index, label, 'focus', 'preorder-cursor')],
  extra,
);

const draft = visual('Preorder records each node and each missing child, so the decoder can consume one token per recursive slot and recover the exact topology.', [
  frame('Initialize preorder at the root', 'For root 1 with left leaf 2 and right child 3 whose left child is 4, visit(1) appends 1.', treeState(0, 'visit 1', {
    stream: '[1]',
    callStack: 'visit(1)',
  }), 'serialize-1'),
  frame('Serialize the left node', 'Preorder enters 2 and appends it before either child.', treeState(1, 'visit 2', {
    stream: '[1,2]',
    callStack: 'visit(1) > visit(2)',
  }), 'serialize-2'),
  frame('Record both null children of 2', 'visit(None) appends # twice, closing the complete subtree rooted at 2.', treeState(1, 'return 2', {
    stream: '[1,2,#,#]',
    calls: '2.left=None; 2.right=None',
  }), 'serialize-2-nulls'),
  frame('Serialize the right node', 'The recursion returns to 1, enters 3, and appends 3.', treeState(2, 'visit 3', {
    stream: '[1,2,#,#,3]',
    callStack: 'visit(1) > visit(3)',
  }), 'serialize-3'),
  frame('Serialize node 4 and its nulls', 'visit(4) appends 4,#,#, completing the left subtree of 3.', treeState(3, 'return 4', {
    stream: '[1,2,#,#,3,4,#,#]',
    calls: '4.left=None; 4.right=None',
  }), 'serialize-4'),
  frame('Close the final missing child', '3.right is None, so the ninth token is # and serialize returns "1,2,#,#,3,4,#,#,#".', treeState(2, 'return 3', {
    stream: '[1,2,#,#,3,4,#,#,#]',
    serialized: '1,2,#,#,3,4,#,#,#',
  }), 'serialize-finish'),
  frame('Deserialize token 1', 'build() consumes 1, creates the root, and recursively reserves its left slot first.', treeState(0, 'build 1', {
    consumed: '1 / 9',
    remaining: '2,#,#,3,4,#,#,#',
  }), 'deserialize-1'),
  frame('Build and close node 2', 'The next token creates 2; the following two # tokens return None for its left and right slots.', treeState(1, 'build 2; #; #', {
    consumed: '4 / 9',
    built: '1.left = 2 with two null children',
  }), 'deserialize-2'),
  frame('Build node 3, then node 4', 'Token 3 fills 1.right; token 4 fills 3.left before its two # child markers.', treeState(3, 'build 4; #; #', {
    consumed: '8 / 9',
    built: '1.right = 3; 3.left = 4',
  }), 'deserialize-4'),
  frame('Consume the final null and return', 'The last # sets 3.right=None. Every recursive slot consumed exactly one token, recreating all original edges.', treeState(2, 'final #; return root', {
    consumed: '9 / 9',
    result: 'tree 1 -> (left 2, right 3 -> left 4)',
  }), 'deserialize-finish'),
]);

const review = {
  pattern: 'Preorder DFS serialization with explicit null-child markers.',
  recognitionCue: 'Use structural sentinels when a tree must round-trip through a linear stream and node values alone do not uniquely identify its shape.',
  invariant: 'Each recursive serialize call emits exactly one leading token for its slot, and each deserialize call consumes exactly one leading token before recursively rebuilding the same left and right slots.',
  stateModel: 'Serialization retains an output token list and recursion stack; deserialization retains only the token iterator and the recursive parent slot currently being filled.',
  visualRationale: 'A real binary-tree topology remains visible while one stable preorder cursor moves through calls; the growing and consumed token streams expose why each null marker is required without relying on color.',
  rejectedAlternatives: [
    'Preorder values without null markers cannot distinguish different tree shapes.',
    'Level-order serialization is valid but does not match the supplied recursive implementation.',
    'A token array alone hides which parent-child slot each # closes.',
  ],
  transferLesson: 'To encode recursive structure linearly, emit a token for every recursive slot, including empty ones, and make the decoder consume tokens in the identical traversal order.',
  reviewStatus: 'reviewed',
};

export default defineVisual('serialize-and-deserialize-binary-tree', draft, review);

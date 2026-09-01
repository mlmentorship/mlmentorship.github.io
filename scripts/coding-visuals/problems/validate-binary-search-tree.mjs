import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const topology = [
  ['', '', '', '5', '', '', ''],
  ['', '', '/', '', '\\', '', ''],
  ['', '1', '', '', '', '7', ''],
  ['', '', '', '', '/', '', '\\'],
  ['', '', '', '', '4', '', '8'],
];
const positions = { '5': [0, 3], '1': [2, 1], '7': [2, 5], '4': [4, 4] };
const call = (node, label, extra = {}) => grid(topology, [{
  row: positions[node][0],
  col: positions[node][1],
  label,
  tone: node === '4' ? 'warning' : 'focus',
  key: 'active-call',
}], {
  input: 'tree=[5,1,7,null,null,4,8]',
  activeCheck: label,
  ...extra,
});

const draft = visual('Validate each node against the complete interval inherited from every ancestor.', [
  frame(
    'Call valid(5, -inf, inf)',
    'The root 5 satisfies -inf < 5 < inf, so recursively validate its left subtree before its right subtree.',
    call('5', 'check -inf < 5 < inf', { stack: 'valid(5,-inf,inf)' }),
    'check-root',
  ),
  frame(
    'Descend left with a tighter high bound',
    'For node 1, the root value becomes the upper bound: -inf < 1 < 5 is true.',
    call('1', 'check -inf < 1 < 5', { stack: 'valid(5,-inf,inf) -> valid(1,-inf,5)' }),
    'check-left-child',
  ),
  frame(
    'Return true for the left subtree',
    'Both children of 1 are null, so both base cases return true and valid(1,-inf,5) returns true.',
    call('1', 'left subtree valid', { returns: 'null=true, null=true, node 1=true' }),
    'return-left-subtree',
  ),
  frame(
    'Descend right with a tighter low bound',
    'For node 7, the root value becomes the lower bound: 5 < 7 < inf is true.',
    call('7', 'check 5 < 7 < inf', { stack: 'valid(5,-inf,inf) -> valid(7,5,inf)' }),
    'check-right-child',
  ),
  frame(
    'Carry both ancestor bounds to node 4',
    'Node 4 is left of 7 but still right of 5, so its legal interval is (5,7). The check 5 < 4 < 7 fails.',
    call('4', '5 < 4 < 7 is false', { stack: 'valid(5,-inf,inf) -> valid(7,5,inf) -> valid(4,5,7)' }),
    'reject-descendant',
  ),
  frame(
    'Propagate false to the root',
    'valid(4,5,7) returns false, so the short-circuited conjunction rejects node 7 and then the entire tree.',
    grid(topology, [{
      row: 4,
      col: 4,
      label: '4 violates ancestor bound 5',
      tone: 'output',
      key: 'active-call',
    }], {
      activeCheck: '4 violates ancestor bound 5',
      returns: 'node 4=false -> node 7=false -> node 5=false',
      result: 'false',
    }),
    'return-invalid',
  ),
]);

const review = {
  pattern: 'Depth-first traversal with inherited exclusive lower and upper bounds.',
  recognitionCue: 'BST validity applies against every ancestor, so a local parent-child comparison is insufficient and each recursive call must carry the legal value interval.',
  invariant: 'At valid(node, low, high), every ancestor constraint is summarized by low < node.val < high; the left call tightens high and the right call tightens low.',
  stateModel: 'Retain the current node, its exclusive low/high bounds, and the recursion stack; null returns true and any out-of-range node returns false immediately.',
  visualRationale: 'A compact coordinate grid draws every node and parent-child edge with ASCII geometry; a stable active-call key moves through readable inherited bounds and the false return chain.',
  rejectedAlternatives: [
    'An inorder value list can prove sortedness but hides how the supplied bound-carrying recursion works.',
    'A parent-versus-child comparison diagram misses violations such as node 4 below ancestor 5.',
    'A prose call table hides which ancestor edges created the interval (5,7).',
  ],
  transferLesson: 'When a recursive descendant must satisfy all ancestor decisions, summarize those decisions as constraints in the call state rather than rechecking only the parent.',
  reviewStatus: 'reviewed',
};

export default defineVisual('validate-binary-search-tree', draft, review);

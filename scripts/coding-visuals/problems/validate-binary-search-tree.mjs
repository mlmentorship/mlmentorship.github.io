import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const topology = [
  ['', '', '', '5', '', '', ''],
  ['', '', '/', '', '\\', '', ''],
  ['', '1', '', '', '', '7', ''],
  ['', '', '', '', '/', '', '\\'],
  ['', '', '', '', '4', '', '8'],
];
const positions = { '5': [0, 3], '1': [2, 1], '7': [2, 5], '4': [4, 4] };
const nodePositions = {
  '5': { x: 210, y: 50 },
  '1': { x: 75, y: 160 },
  '7': { x: 315, y: 160 },
  '4': { x: 255, y: 270 },
  '8': { x: 390, y: 270 },
};
const companion = (node, label) => ({
  nodes: ['5','1','7','4','8'],
  edges: ['5->1','5->7','7->4','7->8'],
  positions: nodePositions,
  width: 440,
  height: 340,
  treeLayout: true,
  start: node,
  markerId: `bst-bound-${node}-${label.replace(/[^a-z0-9]+/gi, '-')}`,
  edgeLabels: false,
  ariaLabel: 'Binary tree: 5 has children 1 and 7; 7 has children 4 and 8',
  regions: [{
    x: nodePositions[node].x,
    y: nodePositions[node].y + 43,
    label: label.replace(/^check /, '').replace('left subtree valid', 'valid subtree'),
  }],
});
const call = (node, label, extra = {}) => grid(topology, [{
  row: positions[node][0],
  col: positions[node][1],
  label,
  tone: node === '4' ? 'warning' : 'focus',
  key: 'active-call',
}], {
  input: 'tree=[5,1,7,null,null,4,8]',
  activeCheck: label,
  graphCompanion: companion(node, label),
  gridSummary: 'Original coordinate inventory',
  ...extra,
});

const draft = visual('Validate each node against the complete interval inherited from every ancestor.', [
  frame(
    'Call valid(5, -inf, inf)',
    'Root 5 passes its open interval (-inf, inf). Validate left before right.',
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
      graphCompanion: companion('4', 'reject: 4 is not above 5'),
      gridSummary: 'Original coordinate inventory',
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
  visualRationale: 'Actual node-and-edge topology shows the ancestor route to 4 and its inherited interval. A disclosed coordinate inventory retains every original motion key without making ASCII geometry the primary picture.',
  rejectedAlternatives: [
    'An inorder value list can prove sortedness but hides how the supplied bound-carrying recursion works.',
    'A parent-versus-child comparison diagram misses violations such as node 4 below ancestor 5.',
    'A prose call table hides which ancestor edges created the interval (5,7).',
  ],
  transferLesson: 'When a recursive descendant must satisfy all ancestor decisions, summarize those decisions as constraints in the call state rather than rechecking only the parent.',
  reviewStatus: 'reviewed',
};

export default defineVisual('validate-binary-search-tree', draft, review);

import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const heights = [
  ['1', '2', '2'],
  ['3', '2', '3'],
  ['2', '4', '1'],
];

const cell = (row, col, label, tone = 'state') => ({ row, col, label, tone, key: `cell-${row}-${col}` });
const scene = (marks, extra = {}) => grid(heights, marks, { input: '3x3 heights; top/left=P, bottom/right=A', ...extra });

const pacificSeeds = [
  cell(0, 0, 'P seed'),
  cell(0, 1, 'P seed'),
  cell(0, 2, 'P seed'),
  cell(1, 0, 'P seed'),
  cell(2, 0, 'P seed'),
];
const pacificAll = [
  ...pacificSeeds,
  cell(1, 1, 'P via 2<=2', 'focus'),
  cell(1, 2, 'P via 2<=3', 'focus'),
  cell(2, 1, 'P via 2<=4', 'focus'),
];
const atlanticSeeds = [
  cell(0, 2, 'A seed'),
  cell(1, 2, 'A seed'),
  cell(2, 0, 'A seed'),
  cell(2, 1, 'A seed'),
  cell(2, 2, 'A seed'),
];
const atlanticWave = [
  ...atlanticSeeds,
  cell(0, 1, 'A: 2<=2', 'focus'),
  cell(1, 0, 'A: 2<=3', 'focus'),
];
const atlanticAll = [
  ...atlanticSeeds,
  cell(0, 1, 'A reached'),
  cell(1, 0, 'A reached'),
  cell(1, 1, 'A: 2<=2', 'focus'),
];

const draft = visual('Reverse both flow searches from their ocean borders, move uphill, then intersect the reached cells.', [
  frame(
    'Seed the Pacific reverse search',
    'Pacific starts are every top- or left-border cell: (0,0),(0,1),(0,2),(1,0),(2,0). These five cells initialize seen and the stack.',
    scene(pacificSeeds, { search: 'Pacific', stack: '5 border seeds', reached: '5 cells' }),
    'seed-pacific',
  ),
  frame(
    'Expand Pacific uphill',
    'Legal reverse moves add (1,1) because 2>=2, (1,2) because 3>=2, and (2,1) because 4>=2. Height 1 at (2,2) is lower than its reached neighbors, so it is blocked.',
    scene(pacificAll, { search: 'Pacific', frontier: '(1,1),(1,2),(2,1)', reached: '8 cells', blocked: '(2,2): 1 is downhill' }),
    'expand-pacific',
  ),
  frame(
    'Seed the Atlantic reverse search',
    'Atlantic starts are every bottom- or right-border cell: (0,2),(1,2),(2,0),(2,1),(2,2). The second search owns a separate seen set.',
    scene(atlanticSeeds, { search: 'Atlantic', stack: '5 border seeds', reached: '5 cells' }),
    'seed-atlantic',
  ),
  frame(
    'Expand the Atlantic frontier',
    'From (0,2) height 2, move left to equal-height (0,1). From (2,0) height 2, move up to height 3 at (1,0).',
    scene(atlanticWave, { search: 'Atlantic', frontier: '(0,1),(1,0)', reached: '7 cells' }),
    'expand-atlantic-wave',
  ),
  frame(
    'Finish Atlantic reachability',
    'From reached (0,1) height 2, move down to equal-height (1,1). Cell (0,0) height 1 is lower than 2, so reverse search cannot enter it.',
    scene(atlanticAll, { search: 'Atlantic', frontier: '(1,1)', reached: '8 cells', blocked: '(0,0): 1 is downhill' }),
    'finish-atlantic',
  ),
  frame(
    'Intersect the two reached sets',
    'Both searches reached every cell except Pacific missed (2,2) and Atlantic missed (0,0). Their intersection therefore contains the other seven coordinates.',
    scene([
      cell(0, 1, 'both', 'output'),
      cell(0, 2, 'both', 'output'),
      cell(1, 0, 'both', 'output'),
      cell(1, 1, 'both', 'output'),
      cell(1, 2, 'both', 'output'),
      cell(2, 0, 'both', 'output'),
      cell(2, 1, 'both', 'output'),
    ], { result: '[[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1]]' }),
    'intersect-reached',
  ),
]);

const review = {
  pattern: 'Two reverse graph searches from destination borders followed by set intersection.',
  recognitionCue: 'Many grid cells ask whether they can reach either of two fixed boundary goals; reversing the edges lets each goal share one traversal across all possible starts.',
  invariant: 'Each ocean’s seen set contains exactly cells with a nonincreasing forward path to that ocean. Reverse traversal may move only to an equal-or-higher neighbor, preserving that path witness.',
  stateModel: 'Keep immutable heights plus one seen set and DFS stack per ocean. Seed the appropriate borders, grow each set under the reversed height rule, then intersect the two sets.',
  visualRationale: 'A fixed height grid directly shows adjacency, border seeds, legal uphill expansions, blocked lower neighbors, independent frontiers, and the coordinate intersection without relying on color.',
  rejectedAlternatives: [
    'Running a path animation from every cell repeats work and obscures the shared destination structure.',
    'A reachability table loses the spatial neighbor relation and ocean-border geometry.',
    'A final highlighted grid alone does not explain the reversed inequality or how either reached set was formed.',
  ],
  transferLesson: 'When many sources query reachability to a small set of goals, reverse the graph, traverse once from each goal class, and combine the resulting reachable sets.',
  reviewStatus: 'reviewed',
};

export default defineVisual('pacific-atlantic-water-flow', draft, review);

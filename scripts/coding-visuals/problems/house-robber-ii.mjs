import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const houses = ['1', '2', '3', '1'];
const scan = (index, excluded, label) => [
  mark(index, label, 'focus', 'current-house'),
  mark(excluded, 'excluded', 'state', 'excluded-house'),
];
const circleEdges = ['h0-h1', 'h1-h2', 'h2-h3', 'h3-h0'];
const circle = (items, marks, extra = {}) => {
  const excluded = marks.find((item) => item.key === 'excluded-house')?.index;
  const current = marks.find((item) => item.key === 'current-house')?.index;
  return array(items, marks, {
    ...extra,
    fitArray: true,
    arrayColumns: 4,
    graphCompanion: {
      nodes: ['h0', 'h1', 'h2', 'h3'],
      nodeLabels: Object.fromEntries(items.map((value, index) => [`h${index}`, index === excluded ? 'X' : `${index}:${value}`])),
      edges: circleEdges.filter((edge) => excluded === undefined || !edge.split('-').includes(`h${excluded}`)),
      positions: { h0: { x: 90, y: 45 }, h1: { x: 270, y: 45 }, h2: { x: 270, y: 185 }, h3: { x: 90, y: 185 } },
      width: 360,
      height: 255,
      treeLayout: true,
      start: current === undefined ? undefined : `h${current}`,
      stateNodes: marks.filter((item) => item.tone === 'output').map((item) => `h${item.index}`),
      markerId: `circular-houses-${excluded ?? 'both'}-${current ?? 'split'}`,
      edgeLabels: false,
      ariaLabel: excluded === undefined ? 'Four indexed house values with the first-last conflict edge' : `Remaining line after excluding house ${excluded}`,
      regions: excluded === undefined ? [{ x: 40, y: 120, label: 'wrap' }] : [{ x: 180, y: 240, label: 'X = excluded' }],
    },
    arraySummary: 'Original house values and cursor',
  });
};

const draft = visual('Break the circle into two endpoint-excluding lines, run take-or-skip DP on each, and keep the larger result.', [
  frame('Split the circular constraint', 'A valid plan excludes index three or index zero. Solve case A [1,2,3] and case B [2,3,1].', circle(
    houses,
    [mark(0, 'first', 'state', 'first-endpoint'), mark(3, 'last', 'state', 'last-endpoint')],
    { cases: 'A excludes index 3; B excludes index 0' },
  ), 'split-circle'),
  frame('Case A reads house 0', 'Start two_back = 0 and one_back = 0. With money 1: (two_back, one_back) becomes (0, max(0,0+1)) = (0,1).', circle(
    houses,
    scan(0, 3, 'A current'),
    { case: 'A = indices 0..2', transition: '(0,0) -> (0,1)' },
  ), 'case-a-house-zero'),
  frame('Case A reads house 1', 'With money 2: (0,1) becomes (1, max(skip 1, take 0+2)) = (1,2).', circle(
    houses,
    scan(1, 3, 'A current'),
    { case: 'A', transition: '(0,1) -> (1,2)', choice: 'take house 1' },
  ), 'case-a-house-one'),
  frame('Case A reads house 2', 'With money 3: (1,2) becomes (2, max(skip 2, take 1+3)) = (2,4). Case A returns 4.', circle(
    houses,
    scan(2, 3, 'A current'),
    { case: 'A', transition: '(1,2) -> (2,4)', choice: 'take houses 0 and 2' },
  ), 'case-a-house-two'),
  frame('Case B reads house 1', 'Reset two_back = one_back = 0. With money 2: state becomes (0,2).', circle(
    houses,
    scan(1, 0, 'B current'),
    { case: 'B = indices 1..3', transition: '(0,0) -> (0,2)' },
  ), 'case-b-house-one'),
  frame('Case B reads house 2', 'With money 3: (0,2) becomes (2, max(skip 2, take 0+3)) = (2,3).', circle(
    houses,
    scan(2, 0, 'B current'),
    { case: 'B', transition: '(0,2) -> (2,3)', choice: 'take house 2' },
  ), 'case-b-house-two'),
  frame('Case B reads house 3', 'With money 1: (2,3) becomes (3, max(skip 3, take 2+1)) = (3,3). Case B returns 3.', circle(
    houses,
    scan(3, 0, 'B current'),
    { case: 'B', transition: '(2,3) -> (3,3)', choice: 'skip house 3' },
  ), 'case-b-house-three'),
  frame('Choose the better line', 'The circular answer is max(case A 4, case B 3) = 4. Taking indices 0 and 2 avoids every conflict edge.', circle(
    houses,
    [mark(0, 'take', 'output', 'current-house'), mark(2, 'take', 'output', 'chosen-house-two')],
    { comparison: 'max(4,3) = 4', result: '4' },
  ), 'choose-best'),
]);

const review = {
  pattern: 'Reduce circular adjacency to two linear House Robber dynamic programs, each with rolling take-or-skip state.',
  recognitionCue: 'Use it when a linear adjacency DP gains one wraparound conflict between the first and last items; every feasible solution must omit at least one endpoint.',
  invariant: 'Within each line, one_back is the best value through the current processed prefix and two_back is the prior prefix optimum; the two cases jointly cover every circularly valid plan.',
  stateModel: 'The minimal state is two endpoint ranges plus two_back and one_back for each linear pass. The fixed indexed circle shows the excluded endpoint and current house throughout both scans.',
  visualRationale: 'An actual four-node cycle shows the wraparound conflict. Each case crosses out its excluded endpoint and removes both incident edges, leaving exactly the processed line. Original indexed values/cursors remain in a disclosure with every prior motion identity, while numeric states expose each source transition.',
  rejectedAlternatives: [
    'A circle drawing alone was rejected because it shows adjacency but not either linear DP transition.',
    'A final two-case table was rejected because it hides how each case value is computed.',
    'A full subset tree was rejected because it obscures the constant-state recurrence with exponential branches.',
  ],
  transferLesson: 'When one wraparound edge breaks a linear DP, condition on excluding either endpoint, solve the resulting linear instances, and combine their optima.',
  reviewStatus: 'reviewed',
};

export default defineVisual('house-robber-ii', draft, review);

import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const current = (total, tone = 'focus') => [mark(total, `total ${total}`, tone, 'current-total')];
const countsRow = (items, marks, extra = {}) => {
  const total = marks[0].index;
  const finalValues = [1, 2, 3].filter((value) => value <= total);
  const nodes = [`total${total}`, ...finalValues.map((value) => `total${total - value}`)];
  return array(items, marks, {
    ...extra,
    fitArray: true,
    arrayColumns: 5,
    graphCompanion: {
      nodes,
      nodeLabels: Object.fromEntries(nodes.map((node) => {
        const subtotal = Number(node.slice(5));
        return [node, `${subtotal}:${items[subtotal]}`];
      })),
      edges: finalValues.map((value) => `total${total - value}->total${total}`),
      positions: {
        [`total${total}`]: { x: 200, y: 45 },
        ...Object.fromEntries(finalValues.map((value, index) => [`total${total - value}`, { x: 80 + 120 * index, y: 160 }])),
      },
      width: 400,
      height: 225,
      treeLayout: true,
      start: `total${total}`,
      stateNodes: finalValues.map((value) => `total${total - value}`),
      markerId: `ordered-count-${total}`,
      edgeLabels: false,
      regions: finalValues.map((value, index) => ({ x: 80 + 120 * index, y: 205, label: `last ${value}` })),
      ariaLabel: `Ordered-sequence count at total ${total}, with disjoint final-value predecessor contributions`,
    },
  });
};

const draft = visual('Build ways[total] by choosing each eligible final number and adding ways[total - number].', [
  frame('Initialize the empty sequence', 'For values [1,2,3], ways[0]=1 counts the empty sequence. All positive totals start at zero.', countsRow(
    ['1', '0', '0', '0', '0'],
    [mark(0, 'base', 'state', 'base-total')],
    { indices: 'total 0..4' },
  ), 'initialize'),
  frame('Build total 1', 'Only final number 1 fits: ways[1] = ways[0] = 1. Numbers 2 and 3 are larger than the total.', countsRow(
    ['1', '1', '0', '0', '0'],
    current(1),
    { recurrence: 'end 1 -> ways[0] = 1', sequences: '[1]' },
  ), 'total-one'),
  frame('Build total 2', 'Ending in 1 contributes ways[1]=1. Ending in 2 contributes ways[0]=1. Together, ways[2]=1+1=2.', countsRow(
    ['1', '1', '2', '0', '0'],
    current(2),
    { recurrence: 'ways[1] + ways[0] = 1 + 1 = 2', sequences: '[1,1], [2]' },
  ), 'total-two'),
  frame('Build total 3', 'Final values 1, 2 and 3 contribute ways[2], ways[1] and ways[0]. Thus ways[3]=2+1+1=4.', countsRow(
    ['1', '1', '2', '4', '0'],
    current(3),
    { recurrence: 'ways[2] + ways[1] + ways[0] = 2 + 1 + 1 = 4' },
  ), 'total-three'),
  frame('Build target 4', 'Final values 1, 2 and 3 contribute four, two and one earlier sequences. Their sum is seven. Orders [1,3] and [3,1] count separately.', countsRow(
    ['1', '1', '2', '4', '7'],
    current(4, 'output'),
    { recurrence: 'ways[3] + ways[2] + ways[1] = 4 + 2 + 1 = 7', result: '7' },
  ), 'target-four'),
]);

const review = {
  pattern: 'One-dimensional counting dynamic programming over increasing totals, partitioned by the final chosen number.',
  recognitionCue: 'Use it when order matters, values may be reused, and every sequence reaching a total has one unambiguous final value that reduces it to a smaller solved total.',
  invariant: 'Before processing total t, ways[0..t-1] contain exact ordered-sequence counts; after trying every num <= t, ways[t] counts each sequence exactly once by its final number.',
  stateModel: 'The minimal state is the ways array and nested loop variables total and num. The indexed array exposes every dependency ways[total-num] and the current target cell.',
  visualRationale: 'Directed edges carry completed predecessor counts into the current total, with each numeric final value labeled under its predecessor. Nodes show total:ways, and the original indexed DP row remains visible with every motion identity. Recurrences and ordered examples distinguish sequence counts from unordered combinations.',
  rejectedAlternatives: [
    'A recursion tree was rejected because repeated totals create duplicate subtrees and hide memoized reuse.',
    'A list of seven final sequences was rejected because it does not generalize the recurrence or explain the base case.',
    'A combinations-style coin table was rejected because iterating numbers outside totals would count unordered combinations instead.',
  ],
  transferLesson: 'To count ordered constructions, partition answers by the last decision and iterate states before choices; changing loop order can change the combinatorial object being counted.',
  reviewStatus: 'reviewed',
};

export default defineVisual('combination-sum-iv', draft, review);

import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const current = (row, col, dependencies = []) => [
  { row, col, label: 'Q', tone: 'focus', key: 'current-cell' },
  ...dependencies,
];
const dependency = (row, col, label, key) => ({ row, col, label, tone: 'state', key });
const diag = (row, col) => dependency(row, col, 'R', 'diagonal-dependency');
const left = (row, col) => dependency(row, col, 'I', 'left-dependency');
const up = (row, col) => dependency(row, col, 'D', 'up-dependency');
const prefixGrid = (rows, marks, extra = {}) => {
  const focus = marks.find((mark) => mark.key === 'current-cell');
  const row = focus?.row ?? 0;
  const col = focus?.col ?? 0;
  const sources = marks.filter((mark) => mark.key !== 'current-cell' && mark.key !== 'base-cell').map((mark) => ({
    node: mark.key === 'left-dependency' ? 'I' : mark.key === 'up-dependency' ? 'D' : 'R',
    value: Number(rows[mark.row][mark.col]),
  }));
  const minimum = Math.min(...sources.map((source) => source.value));
  return grid(rows, marks, {
    ...extra,
    columns: 'second prefixes: empty, c, cu, cut',
    rowAxis: 'first prefixes: empty, c, ca, cat',
    previousRow: `[${(row ? rows[row - 1] : rows[0]).join(',')}]`,
    currentRow: row ? `[${rows[row].slice(0, col + 1).join(',')}]` : 'not started',
    operationCosts: sources.length === 3 ? 'I/D/R each add one' : sources.length ? 'R matches with zero added cost' : 'empty-prefix border prices',
    graphCompanion: {
      nodes: [...sources.map((source) => source.node), 'Q'],
      nodeLabels: { ...Object.fromEntries(sources.map((source) => [source.node, `${source.node}:${source.value}`])), Q: `Q:${rows[row][col]}` },
      edges: sources.map((source) => `${source.node}->Q`),
      positions: { I: { x: 70, y: 185 }, D: { x: 250, y: 45 }, R: { x: 70, y: 45 }, Q: { x: 250, y: 185 } },
      start: 'Q',
      stateNodes: sources.filter((source) => source.value === minimum).map((source) => source.node),
      width: 330,
      height: 240,
      treeLayout: true,
      edgeLabels: false,
      markerId: `edit-predecessors-${row}-${col}`,
      ariaLabel: 'Edit-cost predecessor arrows from left insertion, above deletion and diagonal replacement or match into the current cell',
    },
    gridVisible: true,
  });
};

const states = [
  [
    ['0', '1', '2', '3'],
    ['1', '?', '?', '?'],
    ['2', '?', '?', '?'],
    ['3', '?', '?', '?'],
  ],
  [
    ['0', '1', '2', '3'],
    ['1', '0', '?', '?'],
    ['2', '?', '?', '?'],
    ['3', '?', '?', '?'],
  ],
  [
    ['0', '1', '2', '3'],
    ['1', '0', '1', '?'],
    ['2', '?', '?', '?'],
    ['3', '?', '?', '?'],
  ],
  [
    ['0', '1', '2', '3'],
    ['1', '0', '1', '2'],
    ['2', '?', '?', '?'],
    ['3', '?', '?', '?'],
  ],
  [
    ['0', '1', '2', '3'],
    ['1', '0', '1', '2'],
    ['2', '1', '?', '?'],
    ['3', '?', '?', '?'],
  ],
  [
    ['0', '1', '2', '3'],
    ['1', '0', '1', '2'],
    ['2', '1', '1', '?'],
    ['3', '?', '?', '?'],
  ],
  [
    ['0', '1', '2', '3'],
    ['1', '0', '1', '2'],
    ['2', '1', '1', '2'],
    ['3', '?', '?', '?'],
  ],
  [
    ['0', '1', '2', '3'],
    ['1', '0', '1', '2'],
    ['2', '1', '1', '2'],
    ['3', '2', '?', '?'],
  ],
  [
    ['0', '1', '2', '3'],
    ['1', '0', '1', '2'],
    ['2', '1', '1', '2'],
    ['3', '2', '2', '?'],
  ],
  [
    ['0', '1', '2', '3'],
    ['1', '0', '1', '2'],
    ['2', '1', '1', '2'],
    ['3', '2', '2', '1'],
  ],
];

const mismatchMarks = (row, col) => current(row, col, [
  left(row, col - 1),
  up(row - 1, col),
  diag(row - 1, col - 1),
]);

const draft = visual('Fill prefix edit costs: matches copy the diagonal; mismatches add one to min(insert, delete, replace).', [
  frame('Initialize empty-prefix costs', 'For cat to cut, empty-to-prefix costs are zero through three insertions. Prefix-to-empty costs are zero through three deletions.', prefixGrid(states[0], [
    { row: 0, col: 0, label: 'base', tone: 'state', key: 'base-cell' },
  ], {
    columns: 'empty, c, u, t',
    rowAxis: 'empty, c, a, t',
  }), 'initialize'),
  frame('Match c with c', 'Equal endpoints add no edit: current[1] = previous[0] = 0.', prefixGrid(states[1], current(1, 1, [diag(0, 0)]), {
    recurrence: 'match -> diagonal 0',
  }), 'match-c-c'),
  frame('Transform c to cu', 'c differs from u: 1 + min(insert 0, delete 2, replace 1) = 1.', prefixGrid(states[2], mismatchMarks(1, 2), {
    recurrence: '1 + min(0,2,1) = 1',
    chosen: 'insert u',
  }), 'mismatch-c-u'),
  frame('Transform c to cut', 'c differs from t: 1 + min(insert 1, delete 3, replace 2) = 2.', prefixGrid(states[3], mismatchMarks(1, 3), {
    recurrence: '1 + min(1,3,2) = 2',
    chosen: 'insert t',
  }), 'mismatch-c-t'),
  frame('Transform ca to c', 'a differs from c: 1 + min(insert 2, delete 0, replace 1) = 1.', prefixGrid(states[4], mismatchMarks(2, 1), {
    recurrence: '1 + min(2,0,1) = 1',
    chosen: 'delete a',
  }), 'mismatch-a-c'),
  frame('Transform ca to cu', 'a differs from u: 1 + min(insert 1, delete 1, replace 0) = 1.', prefixGrid(states[5], mismatchMarks(2, 2), {
    recurrence: '1 + min(1,1,0) = 1',
    chosen: 'replace a with u',
  }), 'mismatch-a-u'),
  frame('Transform ca to cut', 'a differs from t: 1 + min(insert 1, delete 2, replace 1) = 2.', prefixGrid(states[6], mismatchMarks(2, 3), {
    recurrence: '1 + min(1,2,1) = 2',
  }), 'mismatch-a-t'),
  frame('Transform cat to c', 't differs from c: 1 + min(insert 3, delete 1, replace 2) = 2.', prefixGrid(states[7], mismatchMarks(3, 1), {
    recurrence: '1 + min(3,1,2) = 2',
    chosen: 'delete from first prefix',
  }), 'mismatch-t-c'),
  frame('Transform cat to cu', 't differs from u: 1 + min(insert 2, delete 1, replace 1) = 2.', prefixGrid(states[8], mismatchMarks(3, 2), {
    recurrence: '1 + min(2,1,1) = 2',
  }), 'mismatch-t-u'),
  frame('Match t with t', 'Equal endpoints copy the diagonal: current[3] = previous[2] = 1. Replace a with u for the one required edit.', prefixGrid(states[9], current(3, 3, [diag(2, 2)]), {
    recurrence: 'match -> diagonal 1',
    result: '1',
  }), 'match-t-t'),
]);

const review = {
  pattern: 'Two-dimensional prefix edit-distance dynamic programming, compressed to previous and current rows.',
  recognitionCue: 'Use it when transforming one sequence prefix into another permits local insert, delete, and replace operations and asks for the minimum total operation count.',
  invariant: 'After current[j] is appended, it is the minimum edits from the processed first prefix to second[:j]; its left, up, and diagonal dependencies are already final.',
  stateModel: 'Only previous, the growing current row, and the two loop indices are required. The conceptual grid shows the three predecessor states represented by insert, delete, and replace.',
  visualRationale: 'Each frame keeps the complete conceptual prefix grid visible beside the stored previous/current row values. Actual I/D/R predecessor arrows point into Q with operation costs stated separately; the ca/cu frame marks its uniquely cheapest replacement source. All original grid motion keys remain exact.',
  rejectedAlternatives: [
    'An edit-script animation was rejected because one optimal script does not prove the minimum recurrence.',
    'A recursion tree was rejected because repeated prefix pairs obscure memoized state reuse.',
    'Two unlabeled rolling arrays were rejected because they hide the insert, delete, and replace coordinates.',
  ],
  transferLesson: 'For minimum transformation problems, define prefix states and map each allowed operation to the predecessor state it leaves behind; then add operation cost to the best predecessor.',
  independentReview: '3.3 source-to-frame replay',
  reviewStatus: 'reviewed',
};

export default defineVisual('edit-distance', draft, review);

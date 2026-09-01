import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const current = (row, col, dependencies = []) => [
  { row, col, label: 'current', tone: 'focus', key: 'current-cell' },
  ...dependencies,
];
const diag = (row, col) => ({ row, col, label: 'diagonal', tone: 'state', key: 'diagonal-dependency' });
const left = (row, col) => ({ row, col, label: 'left', tone: 'state', key: 'left-dependency' });
const up = (row, col) => ({ row, col, label: 'up', tone: 'state', key: 'up-dependency' });

const draft = visual('Fill prefix states left to right: a match takes diagonal + 1; a mismatch takes max(left, up).', [
  frame('Initialize empty-prefix answers', 'For first = "abc" and second = "ac", an empty prefix has LCS length 0 with every other prefix.', grid([
    ['0', '0', '0'],
    ['0', '?', '?'],
    ['0', '?', '?'],
    ['0', '?', '?'],
  ], [{ row: 0, col: 0, label: 'base', tone: 'state', key: 'base-cell' }], {
    columns: 'empty, a, c',
    rowAxis: 'empty, a, b, c',
  }), 'initialize'),
  frame('Match a with a', 'Characters match at (a,a), so current[1] = 1 + previous[0] = 1.', grid([
    ['0', '0', '0'],
    ['0', '1', '?'],
    ['0', '?', '?'],
    ['0', '?', '?'],
  ], current(1, 1, [diag(0, 0)]), {
    recurrence: '1 + diagonal 0 = 1',
  }), 'match-a-a'),
  frame('Skip one prefix at a versus c', 'a differs from c, so current[2] = max(current[1], previous[2]) = max(1, 0) = 1.', grid([
    ['0', '0', '0'],
    ['0', '1', '1'],
    ['0', '?', '?'],
    ['0', '?', '?'],
  ], current(1, 2, [left(1, 1), up(0, 2)]), {
    recurrence: 'max(left 1, up 0) = 1',
    completedRow: 'previous = [0,1,1]',
  }), 'mismatch-a-c'),
  frame('Skip b against a', 'b differs from a, so current[1] = max(current[0], previous[1]) = max(0, 1) = 1.', grid([
    ['0', '0', '0'],
    ['0', '1', '1'],
    ['0', '1', '?'],
    ['0', '?', '?'],
  ], current(2, 1, [left(2, 0), up(1, 1)]), {
    recurrence: 'max(left 0, up 1) = 1',
  }), 'mismatch-b-a'),
  frame('Skip b against c', 'b differs from c, so current[2] = max(current[1], previous[2]) = max(1, 1) = 1.', grid([
    ['0', '0', '0'],
    ['0', '1', '1'],
    ['0', '1', '1'],
    ['0', '?', '?'],
  ], current(2, 2, [left(2, 1), up(1, 2)]), {
    recurrence: 'max(left 1, up 1) = 1',
    completedRow: 'previous = [0,1,1]',
  }), 'mismatch-b-c'),
  frame('Skip c against a', 'c differs from a, so current[1] = max(current[0], previous[1]) = max(0, 1) = 1.', grid([
    ['0', '0', '0'],
    ['0', '1', '1'],
    ['0', '1', '1'],
    ['0', '1', '?'],
  ], current(3, 1, [left(3, 0), up(2, 1)]), {
    recurrence: 'max(left 0, up 1) = 1',
  }), 'mismatch-c-a'),
  frame('Match c with c', 'Characters match at (c,c), so current[2] = 1 + previous[1] = 2. The final row is [0,1,2].', grid([
    ['0', '0', '0'],
    ['0', '1', '1'],
    ['0', '1', '1'],
    ['0', '1', '2'],
  ], current(3, 2, [diag(2, 1)]), {
    recurrence: '1 + diagonal 1 = 2',
    subsequence: '"ac"',
    result: '2',
  }), 'match-c-c'),
]);

const review = {
  pattern: 'Two-dimensional prefix dynamic programming, stored as a previous row and a current row.',
  recognitionCue: 'Use it when two sequences must retain relative order while allowing skips, and the answer for two prefixes depends only on shorter prefixes of one or both sequences.',
  invariant: 'After writing current[index], it is the LCS length for the processed first-string prefix and second[:index]; previous holds the complete answers for the prior first-string prefix.',
  stateModel: 'The implementation stores previous and current rows, while the visual expands those rows into a prefix grid so diagonal, left, and up dependencies remain explicit for every cell.',
  visualRationale: 'A labelled grid is the simplest form that exposes both prefix axes and the recurrence geometry. Values, dependency labels, and equations remain complete without color or JavaScript.',
  rejectedAlternatives: [
    'A two-row values-only table was rejected because it hides which string prefixes each index represents.',
    'A recursion tree was rejected because it duplicates overlapping subproblems and obscures the bottom-up fill order.',
    'A highlighted final subsequence alone was rejected because it does not explain mismatch skips or optimality.',
  ],
  transferLesson: 'For ordered matching problems, define a state on two prefixes; matching endpoints consume both, while mismatching endpoints branch by skipping one side and combining optimal smaller states.',
  independentReview: '3.3 source-to-frame replay',
  reviewStatus: 'reviewed',
};

export default defineVisual('longest-common-subsequence', draft, review);

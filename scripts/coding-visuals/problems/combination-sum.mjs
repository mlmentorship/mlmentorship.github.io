import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const callStack = (rows, extra = {}) => grid(
  rows,
  [{ row: rows.length - 1, col: 0, label: 'active call', tone: 'focus', key: 'call-top' }],
  {
    input: 'candidates = [2,3,6,7], target = 7',
    columns: 'path | start | remaining',
    ...extra,
    graphCompanion: {
      nodes: rows.map((_row, index) => `call${index}`),
      nodeLabels: Object.fromEntries(rows.map((row, index) => [`call${index}`, row[2]])),
      edges: rows.slice(1).map((_row, index) => `call${index}->call${index + 1}`),
      edgeLabels: Object.fromEntries(rows.slice(1).map((row, index) => [`call${index}->call${index + 1}`, `-${Number(rows[index][2]) - Number(row[2])}`])),
      positions: Object.fromEntries(rows.map((_row, index) => [`call${index}`, { x: 45 + 90 * index, y: 60 }])),
      width: 360,
      height: 150,
      treeLayout: true,
      start: `call${rows.length - 1}`,
      markerId: `combination-${rows.map((row) => row[2]).join('-')}-${rows.at(-1)[1]}`,
      regions: rows.map((row, index) => ({ x: 45 + 90 * index, y: 115, label: `start ${row[1]}` })),
      ariaLabel: 'Active recursive calls labeled by remaining target, with start indices and subtract-choice edges',
    },
    gridSummary: 'Original path, start and remaining stack',
  },
);

const draft = visual('Carry the remaining target, reuse the current sorted index, and break when the next choice is too large.', [
  frame('Initialize sorted choices', 'Positive unique choices are [2,3,6,7]. Call choose(0,7) with an empty path.', callStack([['[]', '0', '7']], {
    action: 'choices = [2,3,6,7]',
    answer: '[]',
  }), 'initialize'),
  frame('Choose 2 and keep its index', 'Append 2 and call choose(0,5). Passing index zero permits another 2.', callStack([['[]', '0', '7'], ['[2]', '0', '5']], {
    arithmetic: '7 - 2 = 5',
  }), 'choose-first-two'),
  frame('Reuse 2', 'Append 2 again and call choose(0,3), still allowing choice 2 or any later choice.', callStack([['[]', '0', '7'], ['[2]', '0', '5'], ['[2,2]', '0', '3']], {
    arithmetic: '5 - 2 = 3',
  }), 'choose-second-two'),
  frame('Prune a third 2', 'A third 2 leaves remaining one. The next 2 is too large. Break, return, then pop that third 2.', callStack([['[]', '0', '7'], ['[2]', '0', '5'], ['[2,2]', '0', '3'], ['[2,2,2]', '0', '1']], {
    action: '2 > 1; break; pop 2',
  }), 'prune-third-two'),
  frame('Complete 2 + 2 + 3', 'The [2,2] loop advances to choice 3: remaining 3 - 3 = 0, so copy [2,2,3].', callStack([['[]', '0', '7'], ['[2]', '0', '5'], ['[2,2]', '0', '3'], ['[2,2,3]', '1', '0']], {
    arithmetic: '3 - 3 = 0',
    answer: '[[2,2,3]]',
  }), 'record-two-two-three'),
  frame('Prune the 2 + 3 branch', 'Pop back to [2], then choose 3, leaving two. Start one forbids returning to choice 2. The next 3 is too large.', callStack([['[]', '0', '7'], ['[2]', '0', '5'], ['[2,3]', '1', '2']], {
    action: '3 > 2; break; pop 3',
  }), 'prune-two-three'),
  frame('Prune choices above remainder 5', 'Back at [2], the next choice 6 exceeds remaining five. Sorted order makes every later choice too large. Break the loop.', callStack([['[]', '0', '7'], ['[2]', '0', '5']], {
    action: '6 > 5; break; pop outer 2',
  }), 'prune-after-two'),
  frame('Explore the 3 branch', 'At the root choose 3, leaving four. Reusing start one allows another 3.', callStack([['[]', '0', '7'], ['[3]', '1', '4']], {
    arithmetic: '7 - 3 = 4',
  }), 'choose-three'),
  frame('Prune 3 + 3', 'A second 3 leaves one. At start one, the next choice 3 is too large. Return and pop it.', callStack([['[]', '0', '7'], ['[3]', '1', '4'], ['[3,3]', '1', '1']], {
    action: '3 > 1; break; pop 3; then 6 > 4',
  }), 'prune-three-three'),
  frame('Prune the 6 branch', 'Choose 6 at the root, leaving one. Start two points to 6, which is too large. End that branch.', callStack([['[]', '0', '7'], ['[6]', '2', '1']], {
    arithmetic: '7 - 6 = 1',
    action: '6 > 1; break; pop 6',
  }), 'prune-six'),
  frame('Complete the 7 branch', 'Choose 7 at the root. Remaining reaches zero, so copy [7] as the second combination.', callStack([['[]', '0', '7'], ['[7]', '3', '0']], {
    arithmetic: '7 - 7 = 0',
    result: '[[2,2,3],[7]]',
  }), 'record-seven'),
]);

const review = {
  pattern: 'Backtracking with a sorted start index and remaining target.',
  recognitionCue: 'Use this pattern when combinations must sum to a target, order should not duplicate answers, and the same positive candidate may be selected repeatedly.',
  invariant: 'Every path is nondecreasing by choice index, sum(path) + remaining equals the original target, and choose(start, remaining) may use only start or later indices; remaining zero is exactly a solution.',
  stateModel: 'Keep sorted positive unique choices, mutable path, start index, remaining target, result copies, and recursion stack; recurse with the same index for reuse and pop after return.',
  visualRationale: 'Actual directed call edges subtract one chosen value between remaining-target nodes, with each start boundary retained below its node. The original path/start/remainder stack remains in a disclosure with exact call-top and grid motion identities. Every rejected remainder branch stays in its original frame.',
  rejectedAlternatives: [
    'A dynamic-programming table can count or decide sums but does not enumerate the supplied combination paths.',
    'An unrestricted choice tree generates order duplicates such as [2,2,3] and [3,2,2].',
    'A result-only diagram hides same-index reuse and why one oversized sorted choice prunes all later choices.',
  ],
  transferLesson: 'For reusable combination search, carry a decreasing feasibility measure and a nondecreasing choice boundary; sorted positive choices turn the first oversized option into a safe loop break.',
  independentReview: '3.3 source-to-frame replay',
  reviewStatus: 'reviewed',
};

export default defineVisual('combination-sum', draft, review);

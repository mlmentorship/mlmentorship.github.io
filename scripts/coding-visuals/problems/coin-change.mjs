import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const example = 'coins = [2, 3], amount = 7; unreachable = 8';
const current = (total) => mark(total, `total=${total}`, 'focus', 'current-total');
const source = (index) => mark(index, `predecessor=${index}`, 'state', 'predecessor');
const state = (values, total, predecessor, extra = {}) => array(
  values.map(String),
  [current(total), ...(predecessor === null ? [] : [source(predecessor)])],
  { example, indexMeaning: 'amount', ...extra },
);

const draft = visual('For each amount, minimize one plus every already-solved predecessor amount.', [
  frame('Initialize the DP row', 'fewest[0]=0 and amounts 1 through 7 use sentinel 8, meaning unreachable.', state([0,8,8,8,8,8,8,8], 0, null, { fewest: '[0,8,8,8,8,8,8,8]' }), 'initialize-fewest'),
  frame('Reject both coins for total 1', 'Neither 2 nor 3 satisfies coin <= total 1, so fewest[1] remains 8.', state([0,8,8,8,8,8,8,8], 1, null, { checks: '2 <= 1 false; 3 <= 1 false', decision: 'no predecessor' }), 'total-one'),
  frame('Use coin 2 for total 2', 'Candidate 1+fewest[0]=1 beats 8, so fewest[2]=1; coin 3 is too large.', state([0,8,1,8,8,8,8,8], 2, 0, { transition: 'min(8, 1 + fewest[0]) = 1', coin: '2' }), 'total-two'),
  frame('Try coin 2 for total 3', 'fewest[1] is unreachable, so candidate 1+8=9 does not improve sentinel 8.', state([0,8,1,8,8,8,8,8], 3, 1, { transition: 'min(8, 1 + fewest[1]) = min(8,9) = 8', coin: '2' }), 'total-three-coin-two'),
  frame('Use coin 3 for total 3', 'Candidate 1+fewest[0]=1 improves fewest[3] from 8 to 1.', state([0,8,1,1,8,8,8,8], 3, 0, { transition: 'min(8, 1 + fewest[0]) = 1', coin: '3' }), 'total-three-coin-three'),
  frame('Use coin 2 for total 4', 'Candidate 1+fewest[2]=2 sets fewest[4]=2.', state([0,8,1,1,2,8,8,8], 4, 2, { transition: 'min(8, 1 + fewest[2]) = 2', coin: '2' }), 'total-four-coin-two'),
  frame('Try coin 3 for total 4', 'fewest[1] is unreachable, so candidate 9 leaves fewest[4]=2.', state([0,8,1,1,2,8,8,8], 4, 1, { transition: 'min(2, 1 + fewest[1]) = min(2,9) = 2', coin: '3' }), 'total-four-coin-three'),
  frame('Use coin 2 for total 5', 'Candidate 1+fewest[3]=2 sets fewest[5]=2.', state([0,8,1,1,2,2,8,8], 5, 3, { transition: 'min(8, 1 + fewest[3]) = 2', coin: '2' }), 'total-five-coin-two'),
  frame('Confirm total 5 with coin 3', 'Candidate 1+fewest[2]=2 ties the current value, so fewest[5] stays 2.', state([0,8,1,1,2,2,8,8], 5, 2, { transition: 'min(2, 1 + fewest[2]) = 2', coin: '3' }), 'total-five-coin-three'),
  frame('Use coin 2 for total 6', 'Candidate 1+fewest[4]=3 initially sets fewest[6]=3.', state([0,8,1,1,2,2,3,8], 6, 4, { transition: 'min(8, 1 + fewest[4]) = 3', coin: '2' }), 'total-six-coin-two'),
  frame('Improve total 6 with coin 3', 'Candidate 1+fewest[3]=2 improves fewest[6] from 3 to 2.', state([0,8,1,1,2,2,2,8], 6, 3, { transition: 'min(3, 1 + fewest[3]) = 2', coin: '3' }), 'total-six-coin-three'),
  frame('Use coin 2 for total 7', 'Candidate 1+fewest[5]=3 sets fewest[7]=3.', state([0,8,1,1,2,2,2,3], 7, 5, { transition: 'min(8, 1 + fewest[5]) = 3', coin: '2' }), 'total-seven-coin-two'),
  frame('Confirm total 7 with coin 3', 'Candidate 1+fewest[4]=3 ties the current value, leaving fewest[7]=3.', state([0,8,1,1,2,2,2,3], 7, 4, { transition: 'min(3, 1 + fewest[4]) = 3', coin: '3' }), 'total-seven-coin-three'),
  frame('Return the solved target', 'fewest[7]=3 is not sentinel 8; one optimal construction is 2+2+3.', state([0,8,1,1,2,2,2,3], 7, 4, { construction: '2 + 2 + 3 = 7', result: '3' }), 'return-three-coins'),
]);

const review = {
  pattern: 'Bottom-up one-dimensional minimization DP over amounts.',
  recognitionCue: 'The target can be composed repeatedly from reusable coin values and asks for a minimum count, so each total depends on smaller totals.',
  invariant: 'When processing total t, every fewest[x] for x<t is final; after trying each valid coin, fewest[t] is the minimum candidate examined so far.',
  stateModel: 'Retain an amount-indexed array with base fewest[0]=0, sentinel amount+1 for unreachable states, current total, current coin, and predecessor total-coin.',
  visualRationale: 'An indexed DP row with moving current and predecessor keys directly exposes dependency direction and every improving, tying, or unreachable candidate.',
  rejectedAlternatives: [
    'A coin combination tree repeats the same remaining amounts and hides memoized overlap.',
    'A greedy largest-coin picture is incorrect for general denominations.',
    'A final DP row alone omits the predecessor arithmetic that proves each cell.',
  ],
  transferLesson: 'Define the answer for one total from already-solved smaller totals and aggregate with min; this transfers to minimum steps, perfect squares, and shortest composition problems.',
  reviewStatus: 'reviewed',
};

export default defineVisual('coin-change', draft, review);

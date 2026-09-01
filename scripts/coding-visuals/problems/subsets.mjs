import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const callStack = (rows, extra = {}) => grid(
  rows,
  [{ row: rows.length - 1, col: 0, label: 'active call', tone: 'focus', key: 'call-top' }],
  { input: 'nums = [1, 2, 3]', columns: 'path | start', ...extra },
);

const draft = visual('Save every current path, then recurse only to larger indices and pop the chosen value on return.', [
  frame('Save the empty path', 'choose(0) first copies [] into answer; indices 0, 1, and 2 remain available.', callStack([['[]', 'start 0']], {
    action: 'save []',
    answer: '[[]]',
  }), 'save-empty'),
  frame('Choose index 0', 'Append nums[0]=1 and call choose(1); [1] is copied as the second subset.', callStack([['[]', 'start 0'], ['[1]', 'start 1']], {
    action: 'append 1; save [1]',
    answer: '[[], [1]]',
  }), 'choose-one'),
  frame('Choose index 1 after 1', 'Append 2 and call choose(2); increasing start prevents producing [2,1].', callStack([['[]', 'start 0'], ['[1]', 'start 1'], ['[1,2]', 'start 2']], {
    action: 'append 2; save [1,2]',
    answer: '3 subsets saved',
  }), 'choose-one-two'),
  frame('Choose index 2 after 2', 'Append 3 and call choose(3); save [1,2,3], then the empty loop returns and pops 3.', callStack([['[]', 'start 0'], ['[1]', 'start 1'], ['[1,2]', 'start 2'], ['[1,2,3]', 'start 3']], {
    action: 'save [1,2,3]; return; pop 3',
    answer: '4 subsets saved',
  }), 'choose-one-two-three'),
  frame('Backtrack and choose 3 after 1', 'After choose(2) returns, pop 2; its loop advances to index 2, appends 3, and saves [1,3].', callStack([['[]', 'start 0'], ['[1]', 'start 1'], ['[1,3]', 'start 3']], {
    action: 'pop 2; append 3; save [1,3]',
    answer: '5 subsets saved',
  }), 'choose-one-three'),
  frame('Backtrack to choose 2', 'Return twice, popping 3 then 1; choose(0) advances to index 1 and saves [2].', callStack([['[]', 'start 0'], ['[2]', 'start 2']], {
    action: 'pop 3; pop 1; append 2; save [2]',
    answer: '6 subsets saved',
  }), 'choose-two'),
  frame('Extend 2 with 3', 'From start 2, append nums[2]=3, call choose(3), and save [2,3].', callStack([['[]', 'start 0'], ['[2]', 'start 2'], ['[2,3]', 'start 3']], {
    action: 'append 3; save [2,3]',
    answer: '7 subsets saved',
  }), 'choose-two-three'),
  frame('Backtrack to choose 3 alone', 'Pop 3 and 2; the root loop advances to index 2, appends 3, and saves the final subset [3].', callStack([['[]', 'start 0'], ['[3]', 'start 3']], {
    action: 'pop 3; pop 2; append 3; save [3]',
    result: '[[],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]]',
  }), 'choose-three'),
]);

const review = {
  pattern: 'Backtracking with a monotonically increasing start index.',
  recognitionCue: 'Use this pattern when every combination of distinct input positions is valid and order does not matter, so each partial selection must be emitted once.',
  invariant: 'On entry to choose(start), path contains indices in strictly increasing order, answer already contains every path visited earlier in DFS order, and only indices start or greater may extend this path.',
  stateModel: 'Keep the mutable path, the next allowed index, the recursion call stack, and copied result paths; append before recursion and pop exactly once after return.',
  visualRationale: 'A labelled recursion-stack grid shows each active path and start boundary together; the stable call-top key moves with depth, while textual append, save, return, and pop actions remain color-independent.',
  rejectedAlternatives: [
    'A bitmask table lists subsets compactly but does not explain this start-index backtracking implementation.',
    'A complete choice tree becomes wide and obscures the mutable call stack and DFS output order.',
    'An answer-only list hides the append/recurse/pop restoration invariant.',
  ],
  transferLesson: 'When order should not create duplicates, advance a start boundary after each choice; this same skeleton generates combinations, k-subsets, and increasing-index selections.',
  reviewStatus: 'reviewed',
};

export default defineVisual('subsets', draft, review);

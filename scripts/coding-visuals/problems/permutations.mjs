import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const callStack = (rows, extra = {}) => grid(
  rows,
  [{ row: rows.length - 1, col: 0, label: 'active call', tone: 'focus', key: 'call-top' }],
  { input: 'nums = [1, 2, 3]', columns: 'path | used indices', ...extra },
);

const draft = visual('Fill each path position with every unused index, then pop and clear that same index before the next branch.', [
  frame('Initialize all indices unused', 'choose() starts with path=[] and used=[F,F,F]; no permutation is complete yet.', callStack([['[]', 'F,F,F']], {
    action: 'initialize',
    answer: '[]',
  }), 'initialize'),
  frame('Choose 1 for position 0', 'Mark index 0 used and append 1; the next call skips index 0.', callStack([['[]', 'F,F,F'], ['[1]', 'T,F,F']], {
    action: 'used[0]=T; append 1',
  }), 'choose-one'),
  frame('Choose 2 for position 1', 'Mark index 1 used and append 2, leaving only index 2 available.', callStack([['[]', 'F,F,F'], ['[1]', 'T,F,F'], ['[1,2]', 'T,T,F']], {
    action: 'used[1]=T; append 2',
  }), 'choose-one-two'),
  frame('Record 123', 'Append 3; path length is 3, so copy [1,2,3], return, pop 3, and clear used[2].', callStack([['[]', 'F,F,F'], ['[1]', 'T,F,F'], ['[1,2]', 'T,T,F'], ['[1,2,3]', 'T,T,T']], {
    action: 'save 123; pop 3; used[2]=F',
    answer: '[123]',
  }), 'record-123'),
  frame('Backtrack to the 13 branch', 'The [1,2] loop ends; pop 2 and clear index 1, then choose unused index 2 to form [1,3].', callStack([['[]', 'F,F,F'], ['[1]', 'T,F,F'], ['[1,3]', 'T,F,T']], {
    action: 'pop 2; used[1]=F; append 3',
  }), 'choose-one-three'),
  frame('Record 132', 'Only index 1 is unused; append 2, copy [1,3,2], then restore index 1 on return.', callStack([['[]', 'F,F,F'], ['[1]', 'T,F,F'], ['[1,3]', 'T,F,T'], ['[1,3,2]', 'T,T,T']], {
    action: 'save 132; pop 2; used[1]=F',
    answer: '[123,132]',
  }), 'record-132'),
  frame('Backtrack and choose 2 first', 'Return to the root, clearing indices 2 then 0; root advances to index 1 and starts [2].', callStack([['[]', 'F,F,F'], ['[2]', 'F,T,F']], {
    action: 'restore branch 1; used[1]=T; append 2',
  }), 'choose-two'),
  frame('Record 213', 'Choose unused 1 then 3; copy [2,1,3] and restore index 2 after the leaf returns.', callStack([['[]', 'F,F,F'], ['[2]', 'F,T,F'], ['[2,1]', 'T,T,F'], ['[2,1,3]', 'T,T,T']], {
    action: 'save 213; pop 3; used[2]=F',
    answer: '[123,132,213]',
  }), 'record-213'),
  frame('Record 231', 'Backtrack from [2,1], choose 3 for position 1, then choose the remaining 1 and copy [2,3,1].', callStack([['[]', 'F,F,F'], ['[2]', 'F,T,F'], ['[2,3]', 'F,T,T'], ['[2,3,1]', 'T,T,T']], {
    action: 'restore 1; choose 3 then 1; save 231',
    answer: '[123,132,213,231]',
  }), 'record-231'),
  frame('Backtrack and choose 3 first', 'Restore the entire 2 branch; root advances to index 2, marks it used, and starts [3].', callStack([['[]', 'F,F,F'], ['[3]', 'F,F,T']], {
    action: 'restore branch 2; used[2]=T; append 3',
  }), 'choose-three'),
  frame('Record 312', 'Choose unused 1 then 2; copy [3,1,2] and restore index 1 after returning.', callStack([['[]', 'F,F,F'], ['[3]', 'F,F,T'], ['[3,1]', 'T,F,T'], ['[3,1,2]', 'T,T,T']], {
    action: 'save 312; pop 2; used[1]=F',
    answer: '[123,132,213,231,312]',
  }), 'record-312'),
  frame('Record 321 and finish', 'Backtrack to [3], choose 2 then 1, copy [3,2,1], and restore every choice as recursion unwinds.', callStack([['[]', 'F,F,F'], ['[3]', 'F,F,T'], ['[3,2]', 'F,T,T'], ['[3,2,1]', 'T,T,T']], {
    action: 'save 321; unwind and restore used=[F,F,F]',
    result: '[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]',
  }), 'record-321'),
]);

const review = {
  pattern: 'Backtracking with one used flag per input index.',
  recognitionCue: 'Use this pattern when every ordering is required and each input position may appear exactly once in each complete path.',
  invariant: 'At every choose call, used[index] is true exactly when nums[index] appears in path, path length is the next output position, and returning from a branch restores both structures to their entry state.',
  stateModel: 'Maintain a mutable path, a boolean used array keyed by index, copied complete answers, and the recursion call stack; each branch performs mark, append, recurse, pop, unmark.',
  visualRationale: 'The recursion-stack grid binds every partial ordering to its exact used flags and visibly deepens to each leaf; the stable call-top key tracks depth without relying on color.',
  rejectedAlternatives: [
    'A six-leaf result tree shows outputs but can hide the used-index restoration that makes sibling branches legal.',
    'In-place swapping is a valid alternative algorithm but would not match the supplied used-list implementation.',
    'A factorial output table omits DFS order and the append/pop state transitions.',
  ],
  transferLesson: 'When choices are reusable across sibling branches but forbidden within one path, mark before recursion and unmark after return; index-based flags also handle equal values more safely than value membership.',
  reviewStatus: 'reviewed',
};

export default defineVisual('permutations', draft, review);

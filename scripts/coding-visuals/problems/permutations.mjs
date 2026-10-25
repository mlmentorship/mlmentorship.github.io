import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const prefixes = [[], [1], [1,2], [1,2,3], [1,3], [1,3,2], [2], [2,1], [2,1,3], [2,3], [2,3,1], [3], [3,1], [3,1,2], [3,2], [3,2,1]];
const nodeId = (path) => `p${path.join('')}`;
const positions = {
  p: { x: 240, y: 35 }, p1: { x: 90, y: 125 }, p2: { x: 240, y: 125 }, p3: { x: 390, y: 125 },
  p12: { x: 45, y: 220 }, p13: { x: 125, y: 220 }, p21: { x: 205, y: 220 }, p23: { x: 285, y: 220 }, p31: { x: 365, y: 220 }, p32: { x: 445, y: 220 },
  p123: { x: 45, y: 315 }, p132: { x: 125, y: 315 }, p213: { x: 205, y: 315 }, p231: { x: 285, y: 315 }, p312: { x: 365, y: 315 }, p321: { x: 445, y: 315 },
};
const callStack = (rows, extra = {}) => {
  const path = JSON.parse(rows.at(-1)[0]);
  const activeId = nodeId(path);
  const explored = prefixes.slice(0, prefixes.findIndex((prefix) => nodeId(prefix) === activeId) + 1);
  return grid(
    rows,
    [{ row: rows.length - 1, col: 0, label: 'active call', tone: 'focus', key: 'call-top' }],
    {
      input: 'nums = [1, 2, 3]',
      columns: 'path | used indices',
      ...extra,
      usedByIndex: rows.at(-1)[1].split(',').map((flag, index) => `${index}:${flag}`).join(' '),
      graphCompanion: {
        nodes: explored.map(nodeId),
        nodeLabels: Object.fromEntries(explored.map((prefix) => [nodeId(prefix), prefix.join('') || '[]'])),
        edges: explored.slice(1).map((prefix) => `${nodeId(prefix.slice(0, -1))}->${nodeId(prefix)}`),
        positions,
        width: 480,
        height: 370,
        treeLayout: true,
        start: activeId,
        stateNodes: explored.filter((prefix) => prefix.length === 3).map(nodeId),
        markerId: `permutation-${activeId}`,
        edgeLabels: false,
        ariaLabel: 'Explored permutation choice tree with current prefix and completed output leaves',
      },
      gridSummary: 'Original path and used-index call stack',
    },
  );
};

const draft = visual('Fill each path position with every unused index, then pop and clear that same index before the next branch.', [
  frame('Initialize all indices unused', 'Start choose() with path=[] and used=[F,F,F]. No ordering is complete yet.', callStack([['[]', 'F,F,F']], {
    action: 'initialize',
    answer: '[]',
  }), 'initialize'),
  frame('Choose 1 for position 0', 'Mark index zero used and append 1. The next call skips index zero.', callStack([['[]', 'F,F,F'], ['[1]', 'T,F,F']], {
    action: 'used[0]=T; append 1',
  }), 'choose-one'),
  frame('Choose 2 for position 1', 'Mark index 1 used and append 2, leaving only index 2 available.', callStack([['[]', 'F,F,F'], ['[1]', 'T,F,F'], ['[1,2]', 'T,T,F']], {
    action: 'used[1]=T; append 2',
  }), 'choose-one-two'),
  frame('Record 123', 'Append 3. Length three completes [1,2,3], so copy it. Return, pop 3 and clear used[2].', callStack([['[]', 'F,F,F'], ['[1]', 'T,F,F'], ['[1,2]', 'T,T,F'], ['[1,2,3]', 'T,T,T']], {
    action: 'save 123; pop 3; used[2]=F',
    answer: '[123]',
  }), 'record-123'),
  frame('Backtrack to the 13 branch', 'The [1,2] loop ends. Pop 2 and clear index one. Choose unused index two, forming [1,3].', callStack([['[]', 'F,F,F'], ['[1]', 'T,F,F'], ['[1,3]', 'T,F,T']], {
    action: 'pop 2; used[1]=F; append 3',
  }), 'choose-one-three'),
  frame('Record 132', 'Only index one is unused. Append 2 and copy [1,3,2]. Restore index one on return.', callStack([['[]', 'F,F,F'], ['[1]', 'T,F,F'], ['[1,3]', 'T,F,T'], ['[1,3,2]', 'T,T,T']], {
    action: 'save 132; pop 2; used[1]=F',
    answer: '[123,132]',
  }), 'record-132'),
  frame('Backtrack and choose 2 first', 'Return to the root, clearing indices two then zero. The root chooses index one and starts [2].', callStack([['[]', 'F,F,F'], ['[2]', 'F,T,F']], {
    action: 'restore branch 1; used[1]=T; append 2',
  }), 'choose-two'),
  frame('Choose 1 after 2', 'At [2], index zero is unused. Mark it and append 1, forming [2,1].', callStack([['[]', 'F,F,F'], ['[2]', 'F,T,F'], ['[2,1]', 'T,T,F']], {
    action: 'used[0]=T; append 1',
  }), 'choose-two-one'),
  frame('Record 213', 'Only index two is unused. Append 3 and copy [2,1,3]. Restore index two after return.', callStack([['[]', 'F,F,F'], ['[2]', 'F,T,F'], ['[2,1]', 'T,T,F'], ['[2,1,3]', 'T,T,T']], {
    action: 'save 213; pop 3; used[2]=F',
    answer: '[123,132,213]',
  }), 'record-213'),
  frame('Backtrack and choose 3 after 2', 'Pop 1 and clear index zero. The [2] call chooses index two, marks it and appends 3.', callStack([['[]', 'F,F,F'], ['[2]', 'F,T,F'], ['[2,3]', 'F,T,T']], {
    action: 'pop 1; used[0]=F; used[2]=T; append 3',
  }), 'choose-two-three'),
  frame('Record 231', 'Only index zero is unused. Append 1 and copy [2,3,1].', callStack([['[]', 'F,F,F'], ['[2]', 'F,T,F'], ['[2,3]', 'F,T,T'], ['[2,3,1]', 'T,T,T']], {
    action: 'used[0]=T; append 1; save 231',
    answer: '[123,132,213,231]',
  }), 'record-231'),
  frame('Backtrack and choose 3 first', 'Restore the entire 2 branch. The root chooses index two, marks it and starts [3].', callStack([['[]', 'F,F,F'], ['[3]', 'F,F,T']], {
    action: 'restore branch 2; used[2]=T; append 3',
  }), 'choose-three'),
  frame('Choose 1 after 3', 'At [3], index zero is unused. Mark it and append 1, forming [3,1].', callStack([['[]', 'F,F,F'], ['[3]', 'F,F,T'], ['[3,1]', 'T,F,T']], {
    action: 'used[0]=T; append 1',
  }), 'choose-three-one'),
  frame('Record 312', 'Only index one is unused. Append 2 and copy [3,1,2]. Restore index one after return.', callStack([['[]', 'F,F,F'], ['[3]', 'F,F,T'], ['[3,1]', 'T,F,T'], ['[3,1,2]', 'T,T,T']], {
    action: 'save 312; pop 2; used[1]=F',
    answer: '[123,132,213,231,312]',
  }), 'record-312'),
  frame('Backtrack and choose 2 after 3', 'Pop 1 and clear index zero. The [3] call chooses index one and appends 2.', callStack([['[]', 'F,F,F'], ['[3]', 'F,F,T'], ['[3,2]', 'F,T,T']], {
    action: 'pop 1; used[0]=F; used[1]=T; append 2',
  }), 'choose-three-two'),
  frame('Record 321 and finish', 'Only index zero remains. Append 1 and copy [3,2,1]. As recursion unwinds, restore the empty path and all-false flags.', callStack([['[]', 'F,F,F'], ['[3]', 'F,F,T'], ['[3,2]', 'F,T,T'], ['[3,2,1]', 'T,T,T']], {
    action: 'save 321; unwind and restore used=[F,F,F]',
    afterReturn: 'path=[]; used=[F,F,F]',
    result: '[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]',
  }), 'record-321'),
]);

const review = {
  pattern: 'Backtracking with one used flag per input index.',
  recognitionCue: 'Use this pattern when every ordering is required and each input position may appear exactly once in each complete path.',
  invariant: 'At every choose call, used[index] is true exactly when that input position has been selected in the current path. Path length is the next output position, and each returned branch restores both structures to their entry state.',
  stateModel: 'Maintain a mutable path, a boolean used array keyed by index, copied complete answers, and the recursion call stack; each branch performs mark, append, recurse, pop, unmark.',
  visualRationale: 'A progressively explored directed choice tree preserves all sixteen actual call entries and six output leaves. Indexed flags remain visible at the current prefix. Original path/flag stack rows stay in a disclosure with exact call-top and grid motion identities, while the final text distinguishes saved-leaf state from restored return state.',
  rejectedAlternatives: [
    'A six-leaf result tree shows outputs but can hide the used-index restoration that makes sibling branches legal.',
    'In-place swapping is a valid alternative algorithm but would not match the supplied used-list implementation.',
    'A factorial output table omits DFS order and the append/pop state transitions.',
  ],
  transferLesson: 'Input positions are reusable across sibling branches but forbidden within one path. Mark before recursion and unmark after return. Equal-valued positions remain distinct choices; unique value output requires a separate multiplicity state.',
  independentReview: '3.3 source-to-frame replay',
  reviewStatus: 'reviewed',
};

export default defineVisual('permutations', draft, review);

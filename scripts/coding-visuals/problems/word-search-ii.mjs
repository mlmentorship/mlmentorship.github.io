import { defineVisual, frame, grid, trie, visual } from '../primitives.mjs';

const board = [['o', 'a', 't'], ['e', 't', 'h'], ['e', 'a', 't']];
const trieNodes = [
  { key: 'root', label: 'root', x: 240, y: 28 },
  { key: 'o', label: 'o', x: 80, y: 82 },
  { key: 'e', label: 'e', x: 240, y: 82 },
  { key: 'p', label: 'p', x: 400, y: 82 },
  { key: 'oa', label: 'oa', x: 80, y: 136 },
  { key: 'ea', label: 'ea', x: 240, y: 136 },
  { key: 'pe', label: 'pe', x: 400, y: 136 },
  { key: 'oat', label: 'oat', x: 80, y: 190, terminal: true },
  { key: 'eat', label: 'eat', x: 240, y: 190, terminal: true },
  { key: 'pea', label: 'pea', x: 400, y: 190, terminal: true },
  { key: 'oath', label: 'oath', x: 80, y: 244, terminal: true },
];
const trieEdge = (from, to, label) => ({ key: `edge-${from}-${to}`, from, to, label });
const trieEdges = [
  trieEdge('root', 'o', 'o'), trieEdge('root', 'e', 'e'), trieEdge('root', 'p', 'p'),
  trieEdge('o', 'oa', 'a'), trieEdge('e', 'ea', 'a'), trieEdge('p', 'pe', 'e'),
  trieEdge('oa', 'oat', 't'), trieEdge('ea', 'eat', 't'), trieEdge('pe', 'pea', 'a'),
  trieEdge('oat', 'oath', 'h'),
];
const triePaths = ['oat', 'oath', 'eat', 'pea'].map((word) => ({ word, prefix: 'terminal' }));
const boardState = (row, col, path, extra = {}) => {
  const coordinates = new Map(path.map(([pathRow, pathCol], index) => [`${pathRow},${pathCol}`, index]));
  const marks = [];
  for (const [key, index] of coordinates) {
    const [pathRow, pathCol] = key.split(',').map(Number);
    marks.push({
      row: pathRow,
      col: pathCol,
      label: index === path.length - 1 ? 'search cursor' : `path ${index + 1}`,
      tone: index === path.length - 1 ? 'focus' : 'state',
      key: index === path.length - 1 ? 'board-cursor' : `path-cell-${index}`,
    });
  }
  if (!coordinates.has(`${row},${col}`)) {
    marks.push({ row, col, label: 'search cursor', tone: 'focus', key: 'board-cursor' });
  }
  return grid(board, marks, extra);
};

const draft = visual('Follow board cells and trie children together, mark the current path in place, emit terminal words once, and prune exhausted trie branches while backtracking.', [
  frame('Build the shared prefix trie', 'Insert words oat, oath, eat, and pea. oat is terminal at t while oath continues from that same t to h.', trie(triePaths, {
    nodes: trieNodes,
    edges: trieEdges,
    active: ['root'],
    width: 480,
    height: 275,
    board: '[[o,a,t],[e,t,h],[e,a,t]]',
    terminals: 'double borders mark oat, oath, eat, and pea',
    motion: [{ key: 'trie-cursor', kind: 'pointer', x: 240, y: 28, label: 'current root' }],
  }), 'build-trie'),
  frame('Start at board (0,0)=o', 'Root has child o, so descend into that trie node and temporarily mark board (0,0) as visited.', boardState(0, 0, [[0, 0]], {
    triePrefix: 'o',
    boardWrite: '(0,0): o -> #',
    answer: '[]',
  }), 'start-o'),
  frame('Extend the prefix to oa', 'From (0,0), right neighbor (0,1)=a matches the o-node child a; downward e has no o-node child and returns immediately.', boardState(0, 1, [[0, 0], [0, 1]], {
    triePrefix: 'o-a',
    rejectedNeighbor: '(1,0)=e is not a child after o',
    boardWrite: '(0,1): a -> #',
  }), 'extend-oa'),
  frame('Reach terminal word oat', 'From a, the direction loop checks down first: (1,1)=t matches. pop(None) removes and returns oat, so append oat exactly once.', boardState(1, 1, [[0, 0], [0, 1], [1, 1]], {
    triePrefix: 'o-a-t',
    terminal: 'pop word oat',
    answer: '[oat]',
  }), 'find-oat'),
  frame('Continue the shared prefix to oath', 'The oat trie node still has child h. From t, down a and visited-up fail before right (1,2)=h matches and emits oath.', boardState(1, 2, [[0, 0], [0, 1], [1, 1], [1, 2]], {
    triePrefix: 'o-a-t-h',
    terminal: 'pop word oath',
    answer: '[oat,oath]',
  }), 'find-oath'),
  frame('Restore cells and prune exhausted o branch', 'Backtracking restores h,t,a,o. Empty h, then t, a, and o nodes are removed because both terminal markers were popped.', boardState(0, 0, [], {
    restoration: '# -> h,t,a,o in return order',
    triePrune: 'root no longer has child o',
    answer: '[oat,oath]',
  }), 'prune-o'),
  frame('Reject an incomplete e prefix', 'At (1,0)=e the trie accepts e, but adjacent o,e,t do not match required child a, so restore e without emitting.', boardState(1, 0, [[1, 0]], {
    triePrefix: 'e',
    rejectedNeighbors: 'o, e, t are not child a',
    answer: '[oat,oath]',
  }), 'reject-first-e'),
  frame('Start the viable e path at (2,0)', 'The later e descends at the root, then right neighbor (2,1)=a follows the required child.', boardState(2, 1, [[2, 0], [2, 1]], {
    triePrefix: 'e-a',
    boardWrites: 'e -> #; a -> #',
    answer: '[oat,oath]',
  }), 'extend-ea'),
  frame('Reach terminal word eat', 'From a, the direction loop checks up (1,1)=t before right; it matches, pop(None) returns eat, and append it.', boardState(1, 1, [[2, 0], [2, 1], [1, 1]], {
    triePrefix: 'e-a-t',
    terminal: 'pop word eat',
    answer: '[oat,oath,eat]',
  }), 'find-eat'),
  frame('Finish after restoration and pruning', 'Restore t,a,e and prune their now-empty trie chain. Remaining root child p cannot start anywhere on the board, so pea is absent.', boardState(2, 0, [], {
    trieRemaining: 'p-e-a only',
    scanResult: 'no board cell p',
    result: '[oat,oath,eat]',
  }), 'finish'),
]);

const review = {
  pattern: 'Trie-guided grid backtracking with terminal deduplication and branch pruning.',
  recognitionCue: 'Use a trie when many dictionary words must be searched on the same board and their prefixes can share traversal work.',
  invariant: 'At each recursive call, the marked board path contains distinct adjacent cells spelling exactly the trie path to node; every emitted terminal has been removed, and every pruned trie branch can no longer produce an unseen word.',
  stateModel: 'Retain the shared mutable trie, current trie node, board coordinate, in-place visited marks, answer list, and recursion path; restore each board character before returning.',
  visualRationale: 'The first frame draws one node per shared prefix, including oat continuing to oath, then fixed board geometry shows a stable cursor and ordered path cells moving with trie-prefix labels, terminal pops, restoration, and pruning.',
  rejectedAlternatives: [
    'Running independent Word Search for every word repeats shared prefixes and ignores the supplied trie optimization.',
    'A board-only path animation hides why impossible prefixes stop and why found words are emitted once.',
    'A trie-only diagram hides adjacency, visited-cell exclusion, and backtracking restoration.',
  ],
  transferLesson: 'When many searches share prefixes, traverse input and trie in lockstep, encode path-local visited state reversibly, remove consumed outputs, and prune nodes only after their subtree is exhausted.',
  reviewStatus: 'reviewed',
};

export default defineVisual('word-search-ii', draft, review);

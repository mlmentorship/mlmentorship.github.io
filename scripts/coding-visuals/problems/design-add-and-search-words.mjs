import { defineVisual, frame, graph, visual } from '../primitives.mjs';

const nodes = ['root', 'b', 'ba', 'bat', 'd', 'da', 'dad', 'm', 'ma', 'mad'];
const edges = [
  'root -> b', 'b -> ba', 'ba -> bat',
  'root -> d', 'd -> da', 'da -> dad',
  'root -> m', 'm -> ma', 'ma -> mad',
];

const draft = visual('Follow one child for a literal, but recursively try every child for a dot.', [
  frame(
    'Store three terminal paths',
    'add_word stores bat, dad, and mad. Each of those final prefix nodes contains the None end marker.',
    graph(nodes, edges, { terminal: 'None at bat, dad, mad', words: 'bat, dad, mad' }),
    'store-word-trie',
  ),
  frame(
    'Branch at the wildcard',
    'search(".ad") starts match(0,root). Because word[0] is dot, any(...) recursively tries root children b, d, and m at index 1.',
    graph(nodes, edges, { start: 'root', frontier: ['match(1,b)', 'match(1,d)', 'match(1,m)'], query: '.ad', decision: 'dot -> every child' }),
    'branch-root-wildcard',
  ),
  frame(
    'Follow literals on the b branch',
    'At match(1,b), literal a follows b->ba. At index 2, literal d is absent because ba only has child t, so this branch returns false.',
    graph(nodes, edges, { start: 'ba', frontier: ['match(2,ba)', 'then backtrack'], query: '.ad', decision: 'd not in ba children {t}: false' }),
    'fail-b-branch',
  ),
  frame(
    'Backtrack to the d branch',
    'any(...) next calls match(1,d). Literal a follows d->da, then literal d follows da->dad.',
    graph(nodes, edges, { start: 'dad', frontier: ['match(3,dad)'], query: '.ad', path: 'root -> d -> da -> dad' }),
    'follow-d-branch',
  ),
  frame(
    'Require a terminal at full length',
    'Index 3 equals len(".ad"). dad has the None marker, so match returns true and any(...) short-circuits without needing the m branch.',
    graph(nodes, edges, { start: 'dad', frontier: [], terminal: 'dad contains None', result: 'search(".ad") = true' }),
    'return-wildcard-match',
  ),
]);

const review = {
  pattern: 'Trie search with depth-first branching at wildcard characters.',
  recognitionCue: 'Words are stored by prefix, but a query character may match any one letter, so literal traversal remains deterministic while wildcard positions create a search frontier.',
  invariant: 'match(index,node) is true exactly when some terminal word below node matches the query suffix starting at index; reaching query length succeeds only at a terminal node.',
  stateModel: 'Keep the trie plus recursive state (query index, trie node). A literal makes one recursive call, a dot tries each real child, and any short-circuits after the first true branch.',
  visualRationale: 'The full prefix topology stays fixed while the labelled DFS frontier moves from root to a failing b branch, backtracks, and reaches terminal dad*, exposing both branching and terminal-length logic.',
  rejectedAlternatives: [
    'Drawing dot as a literal trie node hides that it creates multiple recursive calls.',
    'A regular-expression label skips the trie traversal and branch-by-branch failure.',
    'Showing only the successful dad path omits the DFS backtrack that distinguishes wildcard search.',
  ],
  transferLesson: 'When one pattern symbol can choose multiple edges, define recursion by position and node, preserve exact base-case semantics, and short-circuit once any branch satisfies the suffix.',
  reviewStatus: 'reviewed',
};

export default defineVisual('design-add-and-search-words', draft, review);

import { defineVisual, frame, trie, visual } from '../primitives.mjs';

const nodes = [
	{ key: 'root', label: 'root', x: 240, y: 30 },
	{ key: 'b', label: 'b', x: 90, y: 90 },
	{ key: 'd', label: 'd', x: 240, y: 90 },
	{ key: 'm', label: 'm', x: 390, y: 90 },
	{ key: 'ba', label: 'ba', x: 90, y: 150 },
	{ key: 'da', label: 'da', x: 240, y: 150 },
	{ key: 'ma', label: 'ma', x: 390, y: 150 },
	{ key: 'bat', label: 'bat', x: 90, y: 210, terminal: true },
	{ key: 'dad', label: 'dad', x: 240, y: 210, terminal: true },
	{ key: 'mad', label: 'mad', x: 390, y: 210, terminal: true },
];
const edge = (from, to, label) => ({ key: `edge-${from}-${to}`, from, to, label });
const edges = [
	edge('root', 'b', 'b'), edge('root', 'd', 'd'), edge('root', 'm', 'm'),
	edge('b', 'ba', 'a'), edge('d', 'da', 'a'), edge('m', 'ma', 'a'),
	edge('ba', 'bat', 't'), edge('da', 'dad', 'd'), edge('ma', 'mad', 'd'),
];
const paths = ['bat', 'dad', 'mad'].map((word) => ({ word, prefix: 'terminal' }));
const state = (current, queued = [], extra = {}) => {
	const activeNode = nodes.find((node) => node.key === current);
	return trie(paths, {
		nodes,
		edges,
		active: [current],
		queued,
		width: 480,
		height: 245,
		words: 'bat,dad,mad',
		query: '.ad',
		motion: [{ key: 'trie-cursor', kind: 'pointer', x: activeNode.x, y: activeNode.y, label: `current ${current}` }],
		...extra,
	});
};

const draft = visual('A literal follows one child; a dot recursively tries each child until one full-length terminal path succeeds.', [
	frame('Store three terminal paths', 'bat, dad, and mad share root but have separate first-letter branches.', state('root', [], { terminal: 'bat,dad,mad' }), 'store-word-trie'),
	frame('Branch at the wildcard', 'At index 0, dot queues root children b, d, and m for index 1.', state('b', ['d', 'm'], { frontier: 'match(1,b), match(1,d), match(1,m)' }), 'branch-root-wildcard'),
	frame('Follow a on the b branch', 'Literal a moves b->ba at index 1.', state('ba', ['d', 'm'], { path: 'root->b->ba', next: 'literal d' }), 'follow-b-a'),
	frame('Fail and backtrack', 'ba has child t, not required d, so the b branch returns false and DFS tries d.', state('ba', ['d', 'm'], { decision: 'd not in {t}; false' }), 'fail-b-branch'),
	frame('Follow a on the d branch', 'Literal a moves d->da at index 1.', state('da', ['m'], { path: 'root->d->da' }), 'follow-d-a'),
	frame('Follow d to dad', 'Literal d moves da->dad at index 2.', state('dad', ['m'], { path: 'root->d->da->dad' }), 'follow-d-d'),
	frame('Require terminal at full length', 'Index 3 equals query length and dad terminal=yes, so any short-circuits before m.', state('dad', [], { baseCase: 'terminal=yes', result: 'search(".ad") = true' }), 'return-wildcard-match'),
]);

export default defineVisual('design-add-and-search-words', draft, {
	pattern: 'Trie search with DFS branching at wildcard characters.',
	recognitionCue: 'A stored-prefix query contains a symbol matching any one character.',
	invariant: 'match(index,node) is true exactly when a terminal descendant matches the remaining suffix.',
	stateModel: 'Trie plus recursive query index, current node, and wildcard branch frontier.',
	visualRationale: 'One shared root branches to all stored words; a stable cursor and queued-node fill expose wildcard DFS, failure, backtracking, and terminal success.',
	rejectedAlternatives: ['A dot node hides branching.', 'Regex skips trie mechanics.', 'A prefix table hides shared ancestry and backtracking.'],
	transferLesson: 'Branch recursion only at ambiguous symbols and preserve terminal semantics at exact query length.',
	independentReview: '3.4 source-to-frame replay',
	reviewStatus: 'reviewed',
});

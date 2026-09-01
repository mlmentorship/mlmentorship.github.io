import { defineVisual, frame, grid, visual } from '../primitives.mjs';

const rows = (edges, parent, sizes) => [
  ['node', '1', '2', '3', '4'],
  ['parent', ...parent],
  ['root size', ...sizes],
  ['accepted edges', edges || 'none', '', '', ''],
];
const state = (edges, parent, sizes, current, extra = {}) => grid(rows(edges, parent, sizes), current.map((col, index) => ({
  row: 0, col, label: index ? 'second' : 'first', tone: index ? 'state' : 'focus', key: index ? 'second-root' : 'first-root',
})), { example: 'edges=[[1,2],[3,4],[2,3],[4,2]]', ...extra });

const draft = visual('Accept an edge only when its endpoints have different representative roots; otherwise it closes a cycle.', [
  frame('Initialize singleton components', 'Nodes 1..4 each parent themselves and each root has size 1.', state('none',['1','2','3','4'],['1','1','1','1'],[1,2],{ currentEdge:'[1,2]' }), 'initialize-sets'),
  frame('Union edge 1-2', 'find(1)=1 and find(2)=2; attach root 2 under 1 and set size[1]=2.', state('1-2',['1','1','3','4'],['2','-','1','1'],[1,2],{ arithmetic:'1+1=2' }), 'union-1-2'),
  frame('Union edge 3-4', 'find(3)=3 and find(4)=4; attach root 4 under 3 and set size[3]=2.', state('1-2, 3-4',['1','1','3','3'],['2','-','2','-'],[3,4]), 'union-3-4'),
  frame('Find roots for edge 2-3', 'find(2) follows 2->1 while find(3)=3; roots 1 and 3 differ and both have size 2.', state('1-2, 3-4',['1','1','3','3'],['2','-','2','-'],[2,3],{ roots:'2->1; 3->3' }), 'find-2-3'),
  frame('Join the two components', 'Equal sizes need no swap; parent[3]=1 and size[1]=2+2=4.', state('1-2, 3-4, 2-3',['1','1','1','3'],['4','-','-','-'],[2,3],{ arithmetic:'2+2=4' }), 'union-2-3'),
  frame('Compress node 4', 'For candidate 4-2, find(4) sees 4->3->1 and path halving writes parent[4]=1; find(2)=1.', state('1-2, 3-4, 2-3',['1','1','1','1'],['4','-','-','-'],[4,2],{ parentBefore:'[1,1,1,3]', roots:'4->3->1; 2->1' }), 'compress-4'),
  frame('Reject the cycle-closing edge', 'Both roots equal 1, so union returns false before adding 4-2; the accepted path 4-3-2 already connects them.', state('1-2, 3-4, 2-3',['1','1','1','1'],['4','-','-','-'],[4,2],{ union:'false: 1 == 1', result:'[4,2]' }), 'return-redundant-edge'),
]);

export default defineVisual('redundant-connection', draft, {
  pattern:'Disjoint-set union with path halving and union by component size.',
  recognitionCue:'Undirected edges arrive incrementally and each needs an already-connected cycle check.',
  invariant:'Following parent links reaches one representative per component; size is authoritative at roots, and accepted edges join only different roots.',
  stateModel:'Parent and root-size arrays plus the current edge; accepted edges remain visible to explain the graph cycle.',
  visualRationale:'One compact grid aligns node, parent, root size, and accepted topology while endpoint cursors move through finds and unions.',
  rejectedAlternatives:['A final cycle skips union state.','Repeated DFS is a different algorithm.','A parent array alone hides the accepted graph path.'],
  transferLesson:'Compare roots before adding connectivity, attach the smaller tree under the larger, and compress paths during find.',
  independentReview:'3.4 source-to-frame replay', reviewStatus:'reviewed',
});

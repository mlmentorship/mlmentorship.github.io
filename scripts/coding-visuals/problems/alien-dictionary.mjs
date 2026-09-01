import { defineVisual, frame, graph, visual } from '../primitives.mjs';

const nodes = ['w', 'e', 'r', 't', 'f'].map((value) => ({ value, key: `letter-${value}` }));
const scene = (edges, extra = {}) => graph(nodes, edges, {
  words: '[wrt, wrf, er, ett, rftt]',
  ...extra,
});

const draft = visual('Use the first difference of each adjacent word pair as a directed edge, then emit zero-indegree letters with Kahn topological sorting.', [
  frame('Initialize every character', 'Create graph and indegree entries for w,e,r,t,f before adding edges, so isolated letters would also appear in the answer.', scene([], {
    indegree: 'w:0 e:0 r:0 t:0 f:0',
  }), 'initialize'),
  frame('Compare wrt with wrf', 'The shared prefix wr ends at the first difference t versus f, adding t->f and indegree(f)=1.', scene(['t -> f'], {
    comparison: 'w=w, r=r, then t!=f',
    indegree: 'w:0 e:0 r:0 t:0 f:1',
  }), 'edge-t-f'),
  frame('Compare wrf with er', 'The first characters differ, so add w->e and increment indegree(e) to 1.', scene(['t -> f', 'w -> e'], {
    comparison: 'w!=e',
    indegree: 'w:0 e:1 r:0 t:0 f:1',
  }), 'edge-w-e'),
  frame('Compare er with ett', 'The shared e is skipped; r versus t creates r->t and indegree(t)=1.', scene(['t -> f', 'w -> e', 'r -> t'], {
    comparison: 'e=e, then r!=t',
    indegree: 'w:0 e:1 r:0 t:1 f:1',
  }), 'edge-r-t'),
  frame('Compare ett with rftt', 'The first difference e versus r creates e->r. No invalid-prefix pair occurred.', scene(['t -> f', 'w -> e', 'r -> t', 'e -> r'], {
    comparison: 'e!=r',
    indegree: 'w:0 e:1 r:1 t:1 f:1',
    prefixGuard: 'not triggered',
  }), 'edge-e-r'),
  frame('Seed and pop w', 'Only w has indegree 0. Pop w, append it, remove w->e, and enqueue e when its indegree becomes 0.', scene(['t -> f', 'r -> t', 'e -> r'], {
    ready: '[e]',
    emitted: '[w]',
    update: 'indegree(e): 1 -> 0',
  }), 'pop-w'),
  frame('Pop e, then make r ready', 'Pop e, append it, remove e->r, and enqueue r at indegree 0.', scene(['t -> f', 'r -> t'], {
    ready: '[r]',
    emitted: '[w,e]',
    update: 'indegree(r): 1 -> 0',
  }), 'pop-e'),
  frame('Pop r, then make t ready', 'Pop r, append it, remove r->t, and enqueue t.', scene(['t -> f'], {
    ready: '[t]',
    emitted: '[w,e,r]',
    update: 'indegree(t): 1 -> 0',
  }), 'pop-r'),
  frame('Pop t, then make f ready', 'Pop t, append it, remove t->f, and enqueue f.', scene([], {
    ready: '[f]',
    emitted: '[w,e,r,t]',
    update: 'indegree(f): 1 -> 0',
  }), 'pop-t'),
  frame('Pop f and validate the order', 'All five characters were emitted, so len(order)=len(indegree)=5 and the valid alien alphabet is wertf.', scene([], {
    ready: '[]',
    emitted: '[w,e,r,t,f]',
    check: '5 == 5',
    result: 'wertf',
  }), 'finish'),
]);

const review = {
  pattern: 'Constraint extraction from adjacent sorted items followed by Kahn topological sort.',
  recognitionCue: 'Use this pattern when sorted composite values imply a hidden ordering among symbols and only the earliest differing position can establish precedence.',
  invariant: 'After processing each word pair, every recorded edge is a necessary character precedence; during Kahn traversal, ready contains exactly known zero-indegree un-emitted characters and emitted never violates an edge.',
  stateModel: 'Retain every character as a graph node, deduplicated outgoing-edge sets, indegree counts, a zero-indegree queue, and the emitted order.',
  visualRationale: 'A directed graph keeps real precedence topology visible while stable letter nodes persist as edges are added and removed; queue, indegree, and emitted labels make every safe topological move explicit.',
  rejectedAlternatives: [
    'Sorting characters by first appearance invents constraints not implied by adjacent words.',
    'A pairwise comparison table hides the graph paths and cycle-detection condition.',
    'DFS topological sort is valid but does not match the supplied zero-indegree queue transitions.',
  ],
  transferLesson: 'Extract only logically forced local precedence constraints, include unconstrained nodes, reject malformed prefix input, then topologically order and verify every node was emitted.',
  reviewStatus: 'reviewed',
};

export default defineVisual('alien-dictionary', draft, review);

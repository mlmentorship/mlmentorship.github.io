import { defineVisual, frame, queueGrid, visual } from '../primitives.mjs';

const edges = ['0-1', '0-2', '3-4'];
const adjacency = [['1', '2'], ['0'], ['0'], ['4'], ['3'], ['none']];
const scene = (extra = {}) => {
  const { frontier = [], ...metadata } = extra;
  return queueGrid(adjacency, frontier, {
    edgeTopology: `${edges.join(', ')}; node 5 isolated`,
    rowMeaning: 'row index = node; cells = adjacent nodes',
    ...metadata,
    motion: [
      { key: 'graph-topology', kind: 'state', x: 0, y: 0, label: edges.join(', ') },
      ...frontier.map((item, index) => ({ key: `frontier-${item}`, kind: 'frontier', x: index, y: 1, label: item })),
    ],
  });
};

const draft = visual('Each unseen outer-loop vertex starts exactly one DFS and marks exactly one previously uncounted connected component.', [
  frame(
    'Build the graph and initialize',
    'For 6 nodes and edges [[0,1],[0,2],[3,4]], build both directions. Start with seen = {}, components = 0.',
    scene({ adjacency: '0:[1,2]; 1:[0]; 2:[0]; 3:[4]; 4:[3]; 5:[]', seenState: '{}', components: '0' }),
    'initialize-components',
  ),
  frame(
    'Start component 1 at node 0',
    'Outer-loop start 0 is unseen. Increment components 0->1, add 0 to seen, and seed stack [0].',
    scene({ start: '0', visited: ['0'], frontier: ['0'], components: '1', stackState: '[0]' }),
    'start-component-1',
  ),
  frame(
    'Expand node 0',
    'Pop 0; neighbors 1 and 2 are unseen, so add and push both. seen = {0,1,2}; stack = [1,2].',
    scene({ start: '0', visited: ['0', '1', '2'], frontier: ['1', '2'], components: '1', stackState: '[1,2]' }),
    'expand-component-1-root',
  ),
  frame(
    'Pop node 2',
    'LIFO order pops 2. Its only neighbor 0 is seen, so add nothing and leave stack [1].',
    scene({ start: '2', visited: ['0', '1', '2'], frontier: ['1'], components: '1', stackState: '[1]', transition: '2->0 skipped: seen' }),
    'pop-component-1-node-2',
  ),
  frame(
    'Finish component 1 at node 1',
    'Pop 1. Its only neighbor 0 is seen, so the stack empties with {0,1,2} marked.',
    scene({ start: '1', visited: ['0', '1', '2'], components: '1', stackState: '[]', transition: '1->0 skipped: seen' }),
    'finish-component-1',
  ),
  frame(
    'Skip already seen starts',
    'Outer-loop starts 1 and 2 are in seen, so both take the continue branch and do not increment components.',
    scene({ visited: ['0', '1', '2'], components: '1', outerChecks: '1 seen: continue; 2 seen: continue' }),
    'skip-seen-1-2',
  ),
  frame(
    'Start component 2 at node 3',
    'Start 3 is unseen. Increment components 1->2, add 3, and seed stack [3].',
    scene({ start: '3', visited: ['0', '1', '2', '3'], frontier: ['3'], components: '2', stackState: '[3]' }),
    'start-component-2',
  ),
  frame(
    'Discover node 4',
    'Pop 3 and inspect neighbor 4. Add unseen 4 and push it, producing seen {0,1,2,3,4} and stack [4].',
    scene({ start: '3', visited: ['0', '1', '2', '3', '4'], frontier: ['4'], components: '2', stackState: '[4]' }),
    'expand-component-2',
  ),
  frame(
    'Finish component 2',
    'Pop 4; neighbor 3 is already seen, so the stack empties.',
    scene({ start: '4', visited: ['0', '1', '2', '3', '4'], components: '2', stackState: '[]', transition: '4->3 skipped: seen' }),
    'finish-component-2',
  ),
  frame(
    'Skip outer start 4',
    'Outer-loop start 4 is already seen, so take the continue branch without changing components.',
    scene({ visited: ['0', '1', '2', '3', '4'], components: '2', outerChecks: '4 seen: continue' }),
    'skip-seen-4',
  ),
  frame(
    'Start isolated component 3',
    'Start 5 is unseen, so increment components 2->3, add 5 to seen, and seed stack [5].',
    scene({ start: '5', visited: ['0', '1', '2', '3', '4', '5'], frontier: ['5'], components: '3', stackState: '[5]' }),
    'start-isolated-component',
  ),
  frame(
    'Pop isolated node 5',
    'Node 5 has an empty adjacency list. Pop it without discoveries, leaving the stack empty.',
    scene({ start: '5', visited: ['0', '1', '2', '3', '4', '5'], components: '3', stackState: '[]', transition: 'no neighbors' }),
    'finish-isolated-component',
  ),
  frame(
    'Return three components',
    'The outer loop has examined starts 0 through 5. Exactly three unseen starts launched floods, so return 3.',
    scene({ visited: ['0', '1', '2', '3', '4', '5'], components: '3', result: '3' }),
    'return-component-count',
  ),
]);

const review = {
  pattern: 'Connected-component counting by launching iterative DFS from every unseen vertex.',
  recognitionCue: 'Use it when an undirected graph may contain multiple disconnected groups, including isolated vertices, and the task asks how many maximal reachable groups exist.',
  invariant: 'After processing outer-loop starts below the current index, every vertex in their components is seen and components equals the number of DFS launches. A seen vertex can never launch another count.',
  stateModel: 'The minimal state is the adjacency list, global seen set, component counter, and per-flood stack. The trace includes every launch, discovery, pop, continue branch, and the isolated vertex.',
  visualRationale: 'Fixed adjacency rows keep both connected groups and the isolated node visible while the adjacent frontier, seen, current, and count labels explain the nested loops. A stable topology key preserves identity and stays legible at 390px.',
  rejectedAlternatives: [
    'Union-find was rejected because it does not match the supplied DFS implementation.',
    'Three colored blobs were rejected because color alone cannot explain DFS launches or isolated-node handling.',
    'A final component count was rejected because it hides why seen starts do not double-count a group.',
  ],
  transferLesson: 'A global visited set partitions a graph: every new search root contributes one component and consumes all vertices that must not contribute again. This transfers to islands, provinces, and cluster counting.',
  reviewStatus: 'reviewed',
};

export default defineVisual('number-of-connected-components', draft, review);

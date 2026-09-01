import { defineVisual, frame, linked, visual } from '../primitives.mjs';

const cache = (map, order, extra = {}) => linked(
  [
    { value: 'least', key: 'sentinel-least' },
    ...order.map((value) => ({ value, key: `cache-node-${value.split(':')[0]}` })),
    { value: 'most', key: 'sentinel-most' },
  ],
  {
    input: 'capacity=2; put(1,10), put(2,20), get(1), put(3,30), get(2)',
    map: map.length > 0 ? map.map(([key, value]) => `${key} -> ${value}`).join(', ') : '{}',
    links: 'each shown adjacency has both previous and next links',
    ...extra,
  },
);

const draft = visual('The map gives direct node access while the doubly linked order exposes the eviction candidate at its left edge.', [
  frame('Initialize empty cache', 'Create least and most sentinels, link them together, and start with an empty key-to-node map.', cache([], [], { size: '0 / 2' }), 'initialize'),
  frame('Append key 1', 'put(1,10) creates node 1, stores map[1]=node1, and inserts it immediately before the most sentinel.', cache([['1', 'node1(value=10)']], ['1:10'], { size: '1 / 2', rewires: 'least <-> 1 <-> most' }), 'put-one'),
  frame('Append key 2', 'put(2,20) creates node 2 and appends it at the most-recent edge; key 1 is now least recent.', cache([['1', 'node1(value=10)'], ['2', 'node2(value=20)']], ['1:10', '2:20'], { size: '2 / 2', rewires: '1 <-> 2 <-> most' }), 'put-two'),
  frame('Detach key 1 on get', 'get(1) finds node1 in O(1) through the map, then _remove reconnects least directly to node2.', cache([['1', 'node1(value=10)'], ['2', 'node2(value=20)']], ['2:20'], { detached: 'node1(value=10)', rewires: 'least <-> 2' }), 'detach-one'),
  frame('Reappend key 1 as most recent', '_append(node1) links it after node2 and before most; get(1) returns value 10.', cache([['1', 'node1(value=10)'], ['2', 'node2(value=20)']], ['2:20', '1:10'], { result: 'get(1) = 10', rewires: '2 <-> 1 <-> most' }), 'reappend-one'),
  frame('Append key 3 before checking capacity', 'put(3,30) first adds node3 to the map and most-recent edge. The size becomes 3, exceeding capacity 2.', cache([['1', 'node1(value=10)'], ['2', 'node2(value=20)'], ['3', 'node3(value=30)']], ['2:20', '1:10', '3:30'], { size: '3 / 2', decision: '3 > 2 -> evict least.next' }), 'append-three'),
  frame('Remove the least-recent node', 'least.next is node2. _remove(node2) reconnects least to node1; the map still points to node2 until the following delete.', cache([['1', 'node1(value=10)'], ['2', 'node2(value=20)'], ['3', 'node3(value=30)']], ['1:10', '3:30'], { detached: 'node2(value=20)', rewires: 'least <-> 1' }), 'detach-two'),
  frame('Delete evicted key 2 from the map', 'Delete map[2]. The map and list now contain exactly keys 1 and 3 in least-to-most order.', cache([['1', 'node1(value=10)'], ['3', 'node3(value=30)']], ['1:10', '3:30'], { size: '2 / 2', evicted: 'key 2' }), 'delete-two'),
  frame('Return a cache miss', 'get(2) does not find key 2 in the map, so it returns -1 without touching the linked order.', cache([['1', 'node1(value=10)'], ['3', 'node3(value=30)']], ['1:10', '3:30'], { lookup: '2 not in map', result: 'get(2) = -1' }), 'miss-two'),
]);

const review = {
  pattern: 'Hash map for direct lookup plus a sentinel-based doubly linked list for recency order.',
  recognitionCue: 'The interface requires both O(1) key lookup and O(1) least-recent eviction, which no single ordinary array, map, or queue provides alone.',
  invariant: 'Every map entry points to exactly one node between the sentinels, and list order is least-recent to most-recent; successful access moves that same key to the right edge.',
  stateModel: 'Retain capacity, the key-to-node map, least/most sentinels, and each node’s previous/next links. Remove and append are constant-size pointer rewires.',
  visualRationale: 'A sentinel-bounded linked chain gives every cache node a stable motion key as it moves, while adjacent visible map state shows direct key-to-node lookup through every mutation.',
  rejectedAlternatives: [
    'A map alone cannot identify the least-recent key in O(1) time.',
    'A queue alone cannot find and move an arbitrary accessed key in O(1) time.',
    'A final-state-only diagram hides the temporary over-capacity state and the separate list removal and map deletion.',
  ],
  transferLesson: 'Combine an index with an ordered linked structure whenever a system needs direct identity lookup plus constant-time promotion, removal, or boundary eviction.',
  reviewStatus: 'reviewed',
};

export default defineVisual('lru-cache', draft, review);

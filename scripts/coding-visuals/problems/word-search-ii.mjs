import { array, arrayMap, attention, bits, buckets, choices, defineVisual, dualWindow, frame, graph, grid, heap, intervals, linked, lru, mark, pendingReview, prefix, queueGrid, shapes, stack, table, tree, trie, visual } from '../primitives.mjs';

const draft = visual('A trie shares word prefixes and stops a board search as soon as the path leaves the trie.', [
    frame('Build shared search structure', 'All dictionary words enter one trie; each board path follows only a matching child.', trie([{ word: 'oath', prefix: 'o-a-t-h' }, { word: 'eat', prefix: 'e-a-t' }], { action: 'trie prefixes' })),
    frame('Walk the board and trie together', 'A board path that reaches o-a-t may continue to h; a path with no trie child stops.', grid([['o', 'a', 'a', 'n'], ['e', 't', 'a', 'e'], ['i', 'h', 'k', 'r'], ['i', 'f', 'l', 'v']], [{ row: 0, col: 0, label: 'o', tone: 'state' }, { row: 0, col: 1, label: 'a', tone: 'state' }, { row: 1, col: 1, label: 't', tone: 'focus' }, { row: 2, col: 1, label: 'h', tone: 'output' }], { path: 'oath' })),
    frame('Emit each terminal word once', 'The board finds oath and eat; failed prefixes never expand further.', grid([['o', 'a', 'a', 'n'], ['e', 't', 'a', 'e'], ['i', 'h', 'k', 'r'], ['i', 'f', 'l', 'v']], [], { result: '[oath,eat]' })),
  ]);

export default defineVisual('word-search-ii', draft, pendingReview(draft.objective));

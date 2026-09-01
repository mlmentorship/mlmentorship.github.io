import { defineVisual, frame, graph, visual } from '../primitives.mjs';

const draft = visual('Represent each prefix once and mark nodes whose root path is a complete word.', [
  frame(
    'Start with an empty root',
    'A new Trie contains only root={}. Insert("app") begins with node at root.',
    graph(['root'], [], { start: 'root', operation: 'insert("app")', prefix: '""' }),
    'initialize-root',
  ),
  frame(
    'Create the app prefix path',
    'setdefault creates root->a, a->ap, and ap->app as each character is consumed.',
    graph(['root', 'a', 'ap', 'app'], ['root -> a', 'a -> ap', 'ap -> app'], { start: 'app', operation: 'insert("app")', prefix: 'a -> ap -> app' }),
    'insert-app-path',
  ),
  frame(
    'Mark app as a full word',
    'After the last character, store the None end marker at node app. The path app is now both a prefix and a complete word.',
    graph(['root', 'a', 'ap', 'app'], ['root -> a', 'a -> ap', 'ap -> app'], { start: 'app', terminal: 'app contains None end marker', operation: 'insert complete' }),
    'mark-app-terminal',
  ),
  frame(
    'Reuse the shared prefix',
    'Insert("apple") follows existing a, ap, and app nodes, then creates appl and apple instead of duplicating the shared path.',
    graph(['root', 'a', 'ap', 'app', 'appl', 'apple'], ['root -> a', 'a -> ap', 'ap -> app', 'app -> appl', 'appl -> apple'], { start: 'apple', terminal: 'app and apple contain None', operation: 'insert("apple")' }),
    'insert-apple',
  ),
  frame(
    'Distinguish prefix from word',
    'search("ap") walks root->a->ap, but ap has no end marker, so full-word search returns false.',
    graph(['root', 'a', 'ap', 'app', 'appl', 'apple'], ['root -> a', 'a -> ap', 'ap -> app', 'app -> appl', 'appl -> apple'], { start: 'ap', query: 'search("ap")', terminal: 'ap has no None marker', outcome: 'false' }),
    'search-ap-false',
  ),
  frame(
    'Accept the same path as a prefix',
    'starts_with("ap") uses the same walk, but only requires the ap node to exist, so it returns true.',
    graph(['root', 'a', 'ap', 'app', 'appl', 'apple'], ['root -> a', 'a -> ap', 'ap -> app', 'app -> appl', 'appl -> apple'], { start: 'ap', query: 'starts_with("ap")', outcome: 'true' }),
    'prefix-ap-true',
  ),
  frame(
    'Find the complete word app',
    'search("app") walks to app and finds its None marker, so exact-word search returns true.',
    graph(['root', 'a', 'ap', 'app', 'appl', 'apple'], ['root -> a', 'a -> ap', 'ap -> app', 'app -> appl', 'appl -> apple'], { start: 'app', query: 'search("app")', terminal: 'app contains None', result: 'true' }),
    'search-app-true',
  ),
]);

const review = {
  pattern: 'Trie represented as a tree of character-child maps with explicit terminal markers.',
  recognitionCue: 'Many stored strings share prefixes and operations ask both exact-word and prefix membership, so prefix paths should be stored once and traversed character by character.',
  invariant: 'The root-to-node path spells that node’s prefix; a node has the None marker exactly when that prefix was inserted as a complete word.',
  stateModel: 'Keep one root dictionary and a current node while walking characters. Insert creates missing child maps and sets an end marker; queries reuse the same walk and differ only in terminal checking.',
  visualRationale: 'A real prefix-node graph shows shared root-to-prefix edges and separate terminal-marked nodes, making app versus ap and path reuse understandable without code or color.',
  rejectedAlternatives: [
    'Independent word rows falsely duplicate shared prefix nodes.',
    'A flat hash set supports exact words but does not expose prefix traversal or storage sharing.',
    'A character table hides parent-child topology and the semantic role of the end marker.',
  ],
  transferLesson: 'Use tries whenever work can be shared by prefixes; attach payload or terminal state to prefix nodes and stop searches as soon as a required child is absent.',
  reviewStatus: 'reviewed',
};

export default defineVisual('implement-trie', draft, review);

import { arrayMap, defineVisual, frame, mark, visual } from '../primitives.mjs';

const characters = ['e', 'a', 't', '|', 't', 'e', 'a', '|', 't', 'a', 'n'];
const example = 'words = ["eat", "tea", "tan"]';
const cursor = (index, label) => mark(index, label, 'focus', 'signature-cursor');
const groups = (entries, index, label, extra = {}) =>
  arrayMap(characters, entries, [cursor(index, label)], { example, mapLabel: '26-count tuple -> bucket', ...extra });

const draft = visual('Use each word\'s 26-letter frequency tuple as its bucket address.', [
  frame(
    'Initialize the group map',
    'The map is empty and the fresh 26-count array for "eat" is all zeros.',
    groups([], 0, 'eat[0]', { counts: 'a0 e0 t0; every other letter 0' }),
    'initialize-groups',
  ),
  frame(
    'Count e in eat',
    'The character e increments slot ord("e")-ord("a") from 0 to 1.',
    groups([], 0, 'e: 0 -> 1', { counts: 'a0 e1 t0; others 0' }),
    'eat-count-e',
  ),
  frame(
    'Count a in eat',
    'The character a increments its slot, preserving e:1.',
    groups([], 1, 'a: 0 -> 1', { counts: 'a1 e1 t0; others 0' }),
    'eat-count-a',
  ),
  frame(
    'Count t and append eat',
    'The completed tuple has a:1, e:1, t:1 and zeros elsewhere; append "eat" at that exact key.',
    groups([['(a1,e1,t1; others 0)', '[eat]']], 2, 't: 0 -> 1; append', { signature: '(a1,e1,t1; others 0)' }),
    'eat-append-bucket',
  ),
  frame(
    'Reset and count t in tea',
    'A fresh all-zero count array is created for "tea"; its first character sets t:1.',
    groups([['(a1,e1,t1; others 0)', '[eat]']], 4, 'new counts; t: 0 -> 1', { counts: 'a0 e0 t1; others 0' }),
    'tea-count-t',
  ),
  frame(
    'Count e in tea',
    'The second character sets e:1 while t remains 1.',
    groups([['(a1,e1,t1; others 0)', '[eat]']], 5, 'e: 0 -> 1', { counts: 'a0 e1 t1; others 0' }),
    'tea-count-e',
  ),
  frame(
    'Count a and reuse the bucket',
    'After a becomes 1, tea has the same complete 26-count tuple as eat, so append to the existing bucket.',
    groups([['(a1,e1,t1; others 0)', '[eat, tea]']], 6, 'a: 0 -> 1; append', { signature: '(a1,e1,t1; others 0)' }),
    'tea-append-existing-bucket',
  ),
  frame(
    'Reset and count t in tan',
    'A third fresh count array starts at zero; t becomes 1.',
    groups([['(a1,e1,t1; others 0)', '[eat, tea]']], 8, 'new counts; t: 0 -> 1', { counts: 'a0 n0 t1; others 0' }),
    'tan-count-t',
  ),
  frame(
    'Count a in tan',
    'The second character sets a:1 while t remains 1.',
    groups([['(a1,e1,t1; others 0)', '[eat, tea]']], 9, 'a: 0 -> 1', { counts: 'a1 n0 t1; others 0' }),
    'tan-count-a',
  ),
  frame(
    'Count n and create a bucket',
    'The completed tan tuple has a:1, n:1, t:1. It differs from the eat/tea key at e and n, so defaultdict creates a new bucket.',
    groups([
      ['(a1,e1,t1; others 0)', '[eat, tea]'],
      ['(a1,n1,t1; others 0)', '[tan]'],
    ], 10, 'n: 0 -> 1; append', { signature: '(a1,n1,t1; others 0)', branch: 'new key' }),
    'tan-create-new-bucket',
  ),
  frame(
    'Return the bucket values',
    'Insertion order leaves two map values: [eat, tea] for the shared tuple and [tan] for the distinct tuple.',
    groups([
      ['(a1,e1,t1; others 0)', '[eat, tea]'],
      ['(a1,n1,t1; others 0)', '[tan]'],
    ], 10, 'scan complete', { buckets: '2', result: '[["eat", "tea"], ["tan"]]' }),
    'return-groups',
  ),
]);

const review = {
  pattern: 'Hash map from a canonical 26-letter frequency tuple to a list of words.',
  recognitionCue: 'Many lowercase words must be partitioned by equal letter multiplicities while their original letter order is irrelevant.',
  invariant: 'After each word, every processed word appears once in the bucket keyed by its exact 26-count tuple; two words share a bucket exactly when they are anagrams.',
  stateModel: 'For the current word retain a reset 26-slot count array; globally retain groups[tuple(counts)] as the buckets. The word itself is appended only after its signature is complete.',
  visualRationale: 'The moving character cursor, visible sparse form of the exact 26-slot tuple, and tuple-to-bucket map show both signature construction and the existing-key/new-key branch.',
  rejectedAlternatives: [
    'A sorted-word bucket diagram uses a different O(m log m) key and would disagree with the supplied implementation.',
    'Lines connecting anagram words do not show the canonical key or how a new bucket is selected.',
    'Final buckets alone skip every count-array transition and the defaultdict branch.',
  ],
  transferLesson: 'Canonicalize each item into an equality-preserving key, then group by that key; this transfers to shifted-string groups, normalized records, and equivalence-class indexing.',
  reviewStatus: 'reviewed',
};

export default defineVisual('group-anagrams', draft, review);

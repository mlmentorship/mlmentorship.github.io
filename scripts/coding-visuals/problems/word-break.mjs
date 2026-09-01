import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const positions = ['0|', 'c', 'a', 't', 's', 'a', 'n', 'd', 'o', 'g'];
const example = 'text = "catsandog", words = ["cats", "dog", "sand", "and", "cat"]';
const scan = (index, label) => mark(index, label, 'focus', 'start-cursor');
const reached = (index, label = 'reachable') => mark(index, label, 'state', `reachable-${index}`);
const state = (start, reachable, extra = {}) => array(
  positions,
  [scan(start, `start=${start}`), ...reachable.map((index) => reached(index))],
  { example, positionMeaning: 'boundary after index characters; cell 0 is empty prefix', reachable: `{${reachable.join(',')}}`, ...extra },
);

const draft = visual('Only reachable boundaries may launch dictionary words; each full prefix match adds its ending boundary.', [
  frame('Initialize boundary 0', 'The empty prefix is reachable, so reachable={0} before scanning any start.', state(0, [0], { operation: 'initialize' }), 'initialize-reachable'),
  frame('Match cats at start 0', '"catsandog" starts with "cats" at 0, so add ending boundary 0+4=4.', state(0, [0,4], { word: 'cats', check: 'text[0:4] = "cats"', transition: 'add 4' }), 'match-cats'),
  frame('Reject dog and sand at start 0', 'Neither "dog" nor "sand" matches the prefix beginning at 0; reachable stays {0,4}.', state(0, [0,4], { words: 'dog, sand', checks: '"cat" != "dog"; "cats" != "sand"', transition: 'no change' }), 'miss-dog-sand-zero'),
  frame('Reject and, then match cat', '"and" misses at 0, but "cat" matches text[0:3], so add boundary 3.', state(0, [0,3,4], { words: 'and, cat', checks: '"cat" != "and"; text[0:3] = "cat"', transition: 'add 3' }), 'match-cat'),
  frame('Skip boundary 1', 'Start 1 is not in reachable, so the outer loop continues without testing any word.', state(1, [0,3,4], { branch: '1 not reachable: continue' }), 'skip-one'),
  frame('Skip boundary 2', 'Start 2 is not reachable, so no dictionary checks run.', state(2, [0,3,4], { branch: '2 not reachable: continue' }), 'skip-two'),
  frame('Reach 7 from boundary 3', 'At reachable start 3, cats and dog miss; "sand" matches text[3:7], so add 3+4=7.', state(3, [0,3,4,7], { checks: 'cats miss; dog miss; sand matches', transition: 'add 7' }), 'match-sand'),
  frame('Finish checks at boundary 3', 'Words "and" and "cat" also miss at start 3, so reachable remains {0,3,4,7}.', state(3, [0,3,4,7], { checks: 'and miss; cat miss', transition: 'no change' }), 'finish-three'),
  frame('Match and at boundary 4', 'At reachable start 4, cats, dog, and sand miss; "and" matches text[4:7], adding boundary 7 again.', state(4, [0,3,4,7], { checks: 'cats miss; dog miss; sand miss; and matches', transition: 'add 7 (already present)' }), 'match-and'),
  frame('Finish checks at boundary 4', '"cat" misses at start 4, so no new boundary is added.', state(4, [0,3,4,7], { check: 'cat miss', transition: 'no change' }), 'finish-four'),
  frame('Skip boundaries 5 and 6', 'Starts 5 and 6 are absent from reachable, so both iterations continue immediately.', state(6, [0,3,4,7], { branches: '5 not reachable; 6 not reachable' }), 'skip-five-six'),
  frame('Fail every word at boundary 7', 'Start 7 is reachable, but suffix "og" begins with none of cats, dog, sand, and, or cat.', state(7, [0,3,4,7], { checks: 'cats miss; dog miss; sand miss; and miss; cat miss', transition: 'no change' }), 'miss-at-seven'),
  frame('Skip boundaries 8 and 9', 'Neither 8 nor 9 is reachable, so the scan ends with reachable={0,3,4,7}.', state(9, [0,3,4,7], { branches: '8 not reachable; 9 not reachable' }), 'skip-eight-nine'),
  frame('Reject the full segmentation', 'len(text)=9 is not in reachable, so "catsandog" cannot be split entirely into dictionary words.', state(9, [0,3,4,7], { comparison: '9 not in {0,3,4,7}', result: 'false' }), 'return-false'),
]);

const review = {
  pattern: 'Forward reachability DP over string boundaries.',
  recognitionCue: 'The string must be segmented into reusable dictionary words, so each valid prefix endpoint can seed another exact prefix match.',
  invariant: 'Before scanning start s, reachable contains exactly the boundaries proven segmentable using dictionary words from earlier starts; only members may create new endpoints.',
  stateModel: 'Retain the text, dictionary, set of reachable boundaries, current start, and current word check. No segmentation path is needed for a boolean answer.',
  visualRationale: 'An indexed character strip with stable start and reachable-boundary keys shows prefix geometry, skipped starts, exact match spans, duplicate endpoints, and the missing final boundary.',
  rejectedAlternatives: [
    'A recursion tree repeats suffixes and obscures the memoized boundary state.',
    'A boolean table without characters hides which substring starts and ends at each dependency.',
    'Showing only "cat" + "sand" + failure skips alternative "cats" + "and" reachability and the actual loop branches.',
  ],
  transferLesson: 'Treat partial solutions as reachable boundaries and propagate only from proven states; this transfers to sentence segmentation, path reachability, and parsing with reusable tokens.',
  reviewStatus: 'reviewed',
};

export default defineVisual('word-break', draft, review);

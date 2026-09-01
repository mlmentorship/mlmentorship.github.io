import { arrayMap, defineVisual, frame, mark, visual } from '../primitives.mjs';

const values = ['1', '2', '3', '4', '100', '200'];
const example = 'nums = [100, 4, 200, 1, 3, 2]; set displayed in sorted order';
const candidate = (index, label) => mark(index, label, 'focus', 'candidate-cursor');
const end = (index, label) => mark(index, label, 'state', 'run-end');
const state = (entries, marks, extra = {}) =>
  arrayMap(values, entries, marks, { example, mapLabel: 'scalar state and membership query', ...extra });

const draft = visual('Only values without a predecessor start a run; then advance one value at a time.', [
  frame(
    'Build the membership set',
    'Deduplicate nums into {1,2,3,4,100,200}; best starts at 0. Sorted display does not change set membership or the answer.',
    state([['best', '0']], [candidate(0, 'next candidate 1')], { set: '{1,2,3,4,100,200}' }),
    'initialize-set',
  ),
  frame(
    'Recognize start 1',
    'For candidate 1, predecessor 0 is absent, so initialize end=1 and enter the while loop.',
    state([['best', '0'], ['query', '0 absent']], [candidate(0, 'start=1'), end(0, 'end=1')], { branch: 'start a run' }),
    'start-run-at-one',
  ),
  frame(
    'Advance end to 2',
    'end+1=2 is in the set, so the while loop moves end from 1 to 2.',
    state([['best', '0'], ['query', '2 present']], [candidate(0, 'start=1'), end(1, 'end: 1 -> 2')], { runLength: '2 - 1 + 1 = 2' }),
    'extend-run-to-two',
  ),
  frame(
    'Advance end to 3',
    'end+1=3 is present, so end moves from 2 to 3.',
    state([['best', '0'], ['query', '3 present']], [candidate(0, 'start=1'), end(2, 'end: 2 -> 3')], { runLength: '3 - 1 + 1 = 3' }),
    'extend-run-to-three',
  ),
  frame(
    'Advance end to 4',
    'end+1=4 is present, so end moves from 3 to 4.',
    state([['best', '0'], ['query', '4 present']], [candidate(0, 'start=1'), end(3, 'end: 3 -> 4')], { runLength: '4 - 1 + 1 = 4' }),
    'extend-run-to-four',
  ),
  frame(
    'Stop the run and update best',
    'end+1=5 is absent. The completed length is 4-1+1=4, so best changes from 0 to 4.',
    state([['best', '4'], ['query', '5 absent']], [candidate(0, 'start=1'), end(3, 'end=4')], { arithmetic: 'max(0, 4 - 1 + 1) = 4' }),
    'finish-main-run',
  ),
  frame(
    'Skip candidate 2',
    'For candidate 2, predecessor 1 is present. Continue immediately so the run [1,2,3,4] is not counted again.',
    state([['best', '4'], ['query', '1 present']], [candidate(1, 'skip start=2'), end(3, 'completed end=4')], { branch: 'continue' }),
    'skip-two',
  ),
  frame(
    'Skip candidate 3',
    'For candidate 3, predecessor 2 is present, so continue.',
    state([['best', '4'], ['query', '2 present']], [candidate(2, 'skip start=3'), end(3, 'completed end=4')], { branch: 'continue' }),
    'skip-three',
  ),
  frame(
    'Skip candidate 4',
    'For candidate 4, predecessor 3 is present, so continue.',
    state([['best', '4'], ['query', '3 present']], [candidate(3, 'skip start=4'), end(3, 'completed end=4')], { branch: 'continue' }),
    'skip-four',
  ),
  frame(
    'Measure singleton 100',
    'For 100, predecessor 99 is absent, but successor 101 is also absent. Its length is 100-100+1=1, so best stays 4.',
    state([['best', '4'], ['queries', '99 absent; 101 absent']], [candidate(4, 'start=100'), end(4, 'end=100')], { arithmetic: 'max(4, 100 - 100 + 1) = 4' }),
    'measure-one-hundred',
  ),
  frame(
    'Measure singleton 200',
    'For 200, predecessor 199 and successor 201 are absent. Its length is 1, so best remains 4.',
    state([['best', '4'], ['queries', '199 absent; 201 absent']], [candidate(5, 'start=200'), end(5, 'end=200')], { arithmetic: 'max(4, 200 - 200 + 1) = 4' }),
    'measure-two-hundred',
  ),
  frame(
    'Return the longest length',
    'All set values are either a measured run start or were skipped because their predecessor exists. Return best=4.',
    state([['best', '4']], [mark(0, 'best run start', 'output', 'candidate-cursor'), mark(3, 'best run end', 'output', 'run-end')], { longestRun: '[1,2,3,4]', result: '4' }),
    'return-best',
  ),
]);

const review = {
  pattern: 'Hash-set membership with run-start filtering.',
  recognitionCue: 'The input is unsorted and asks for consecutive integer values, not consecutive positions, while near-linear time rules out sorting as the intended mechanism.',
  invariant: 'A run is expanded only from its unique smallest value, identified by missing predecessor; best is the maximum completed run length seen so far.',
  stateModel: 'Retain the deduplicated value set, outer candidate start, moving end for the current run, and scalar best. No input ordering or visited set is required.',
  visualRationale: 'A sorted-for-reading set strip preserves actual membership while stable candidate and end keys visibly separate skipped interior values from forward run expansion.',
  rejectedAlternatives: [
    'Sorting the original array makes adjacency easy but depicts an O(n log n) algorithm instead of the supplied set solution.',
    'A graph with edges between consecutive values adds topology that membership queries already express more simply.',
    'A final highlighted run hides the predecessor gate that prevents repeated expansion and establishes average O(n) work.',
  ],
  transferLesson: 'Before expanding a component, find a unique boundary that only one member can satisfy; this avoids duplicate work in runs, intervals, and component scans.',
  reviewStatus: 'reviewed',
};

export default defineVisual('longest-consecutive-sequence', draft, review);

import { defineVisual, frame, intervals, visual } from '../primitives.mjs';

const scan = (label, x, y) => [{ key: 'scan-interval', kind: 'interval', x, y, label }];
const range = (label, start, end, tone = 'neutral') => ({ label, start, end, tone });

const draft = visual('Sort ranges by start; compare each range with the last merged end, extending on overlap or appending on a gap.', [
  frame('Sort the concrete input', 'Input [[8,10],[1,3],[2,6],[15,18]] becomes [[1,3],[2,6],[8,10],[15,18]], so possible overlaps are adjacent.', intervals([
    range('1: [1,3]', 1, 3, 'focus'),
    range('2: [2,6]', 2, 6),
    range('3: [8,10]', 8, 10),
    range('4: [15,18]', 15, 18),
  ], {
    max: 18,
    order: 'sorted by start',
    motion: scan('[1,3]', 1, 0),
  }), 'sort'),
  frame('Seed the merged output', 'Copy ordered[0] = [1,3]. The last merged end is now 3.', intervals([
    range('merged [1,3]', 1, 3, 'output'),
    range('next [2,6]', 2, 6, 'focus'),
    range('pending [8,10]', 8, 10),
    range('pending [15,18]', 15, 18),
  ], {
    max: 18,
    state: 'merged = [[1,3]]',
    motion: scan('[2,6]', 2, 1),
  }), 'seed'),
  frame('Extend the overlapping end', 'For [2,6], start 2 <= last end 3, so set the last end to max(3,6) = 6.', intervals([
    range('merged [1,6]', 1, 6, 'output'),
    range('current [2,6]', 2, 6, 'focus'),
    range('next [8,10]', 8, 10),
    range('pending [15,18]', 15, 18),
  ], {
    max: 18,
    branch: '2 <= 3 -> overlap',
    state: 'merged = [[1,6]]',
    motion: scan('[2,6]', 2, 1),
  }), 'merge-overlap'),
  frame('Append after the first gap', 'For [8,10], start 8 > last end 6, so append a new merged range instead of extending [1,6].', intervals([
    range('merged [1,6]', 1, 6, 'output'),
    range('merged [8,10]', 8, 10, 'output'),
    range('current [8,10]', 8, 10, 'focus'),
    range('next [15,18]', 15, 18),
  ], {
    max: 18,
    branch: '8 > 6 -> append',
    state: 'merged = [[1,6],[8,10]]',
    motion: scan('[8,10]', 8, 2),
  }), 'append-eight-ten'),
  frame('Append after the second gap', 'For [15,18], start 15 > last end 10, so append it. Every sorted interval has now been consumed.', intervals([
    range('merged [1,6]', 1, 6, 'output'),
    range('merged [8,10]', 8, 10, 'output'),
    range('merged [15,18]', 15, 18, 'output'),
    range('current [15,18]', 15, 18, 'focus'),
  ], {
    max: 18,
    branch: '15 > 10 -> append',
    motion: scan('[15,18]', 15, 3),
    result: '[[1,6],[8,10],[15,18]]',
  }), 'append-fifteen-eighteen'),
]);

const review = {
  pattern: 'Sort intervals by start, then scan while maintaining the last merged interval.',
  recognitionCue: 'Use it when arbitrary ranges must be coalesced by overlap; sorting by start guarantees that any range able to overlap the current merged component appears before a later gap.',
  invariant: 'Before each current interval, merged is sorted, disjoint, and exactly covers all processed ranges; only merged[-1] can overlap current because starts are nondecreasing.',
  stateModel: 'The minimal scan state is the sorted intervals, current index, and merged output whose final element is mutable. The timeline fixes all endpoints on one common number line.',
  visualRationale: 'Aligned interval tracks make overlap and gaps geometric rather than verbal. Explicit comparison equations and ordered labels remain readable without color or playback.',
  rejectedAlternatives: [
    'A start/end table was rejected because it hides physical overlap and gaps.',
    'A graph of pairwise overlaps was rejected because sorting removes the need to compare all pairs.',
    'A final merged picture was rejected because it omits the seed, extend, and append branches.',
  ],
  transferLesson: 'Sorting can turn a global overlap problem into a local frontier check: preserve a completed prefix and keep only the final component open for possible extension.',
  reviewStatus: 'reviewed',
};

export default defineVisual('merge-intervals', draft, review);

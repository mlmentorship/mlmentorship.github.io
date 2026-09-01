import { defineVisual, frame, intervals, visual } from '../primitives.mjs';

const scan = (label, x, y) => [{ key: 'scan-interval', kind: 'interval', x, y, label }];
const range = (label, start, end, tone = 'neutral') => ({ label, start, end, tone });

const draft = visual('Partition sorted ranges into a copied prefix, one growing overlap span, and an untouched suffix.', [
  frame('Initialize the inserted range', 'Use intervals [[1,2],[3,5],[6,7],[8,10],[12,16]] and new [4,8]. Start answer = [], index = 0, span = [4,8].', intervals([
    range('current [1,2]', 1, 2, 'focus'),
    range('[3,5]', 3, 5),
    range('new span [4,8]', 4, 8, 'state'),
    range('[6,7]', 6, 7),
    range('[8,10]', 8, 10),
    range('[12,16]', 12, 16),
  ], {
    max: 16,
    state: 'answer = []; span = [4,8]',
    motion: scan('[1,2]', 1, 0),
  }), 'initialize'),
  frame('Copy the interval fully before', '[1,2] ends at 2 < span start 4, so append [1,2] unchanged and advance index to [3,5].', intervals([
    range('answer [1,2]', 1, 2, 'output'),
    range('current [3,5]', 3, 5, 'focus'),
    range('span [4,8]', 4, 8, 'state'),
    range('pending [6,7]', 6, 7),
    range('pending [8,10]', 8, 10),
    range('suffix [12,16]', 12, 16),
  ], {
    max: 16,
    branch: '2 < 4 -> copy prefix',
    motion: scan('[3,5]', 3, 1),
  }), 'copy-prefix'),
  frame('Merge the first overlap', '[3,5] starts at 3 <= span end 8, so span becomes [min(4,3), max(8,5)] = [3,8].', intervals([
    range('answer [1,2]', 1, 2, 'output'),
    range('current [3,5]', 3, 5, 'focus'),
    range('growing span [3,8]', 3, 8, 'state'),
    range('next [6,7]', 6, 7),
    range('pending [8,10]', 8, 10),
    range('suffix [12,16]', 12, 16),
  ], {
    max: 16,
    arithmetic: '[min(4,3), max(8,5)] = [3,8]',
    motion: scan('[3,5]', 3, 1),
  }), 'merge-three-five'),
  frame('Consume a contained overlap', '[6,7] starts at 6 <= 8, but [min(3,6), max(8,7)] stays [3,8]. Advance without emitting a second range.', intervals([
    range('answer [1,2]', 1, 2, 'output'),
    range('growing span [3,8]', 3, 8, 'state'),
    range('current [6,7]', 6, 7, 'focus'),
    range('next [8,10]', 8, 10),
    range('suffix [12,16]', 12, 16),
  ], {
    max: 16,
    arithmetic: '[min(3,6), max(8,7)] = [3,8]',
    motion: scan('[6,7]', 6, 2),
  }), 'merge-six-seven'),
  frame('Extend through a touching overlap', '[8,10] starts at 8 <= span end 8, so closed intervals overlap at 8 and span extends to [3,10].', intervals([
    range('answer [1,2]', 1, 2, 'output'),
    range('growing span [3,10]', 3, 10, 'state'),
    range('current [8,10]', 8, 10, 'focus'),
    range('next [12,16]', 12, 16),
  ], {
    max: 16,
    arithmetic: '[min(3,8), max(8,10)] = [3,10]',
    motion: scan('[8,10]', 8, 3),
  }), 'merge-eight-ten'),
  frame('Stop at the suffix gap', '[12,16] starts at 12 > span end 10, so overlap scanning stops. Return prefix + [3,10] + untouched suffix.', intervals([
    range('answer [1,2]', 1, 2, 'output'),
    range('inserted [3,10]', 3, 10, 'output'),
    range('suffix [12,16]', 12, 16, 'output'),
    range('current [12,16]', 12, 16, 'focus'),
  ], {
    max: 16,
    branch: '12 > 10 -> suffix begins',
    motion: scan('[12,16]', 12, 4),
    result: '[[1,2],[3,10],[12,16]]',
  }), 'return-result'),
]);

const review = {
  pattern: 'A linear three-phase interval scan over sorted, non-overlapping input: prefix, overlap span, suffix.',
  recognitionCue: 'Use it when inserting one range into already sorted disjoint ranges; their order partitions them into intervals ending before, overlapping, or starting after the growing span.',
  invariant: 'answer contains every interval strictly before the span, span is the union of the new interval and all consumed overlaps, and intervals from index onward remain untouched and sorted.',
  stateModel: 'The minimal state is answer, index, and mutable span endpoints start and end. The shared number line shows the current interval moving through each of the three groups.',
  visualRationale: 'Aligned interval tracks expose before, overlap, containment, touching, and after relations directly. Branch equations and labels preserve meaning in monochrome static output.',
  rejectedAlternatives: [
    'Calling the general merge routine was rejected because it adds sorting and hides the useful pre-sorted three-phase structure.',
    'A table of endpoint comparisons was rejected because it makes overlap geometry harder to see.',
    'A single before-and-after diagram was rejected because it skips repeated span growth and the suffix stopping condition.',
  ],
  transferLesson: 'When ordered input surrounds one mutable object, scan an immutable prefix, absorb the contiguous interaction region into that object, then reuse the untouched suffix.',
  independentReview: '3.3 source-to-frame replay',
  reviewStatus: 'reviewed',
};

export default defineVisual('insert-interval', draft, review);

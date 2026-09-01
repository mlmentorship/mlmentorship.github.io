import { defineVisual, frame, intervals, visual } from '../primitives.mjs';

const sorted = [
  { label: '[2,4]', start: 2, end: 4 },
  { label: '[7,10]', start: 7, end: 10 },
  { label: '[9,11]', start: 9, end: 11 },
  { label: '[12,14]', start: 12, end: 14 },
];

const draft = visual('Sorting by start time makes the first overlap appear between adjacent meetings, so Python all can stop at the first failed boundary check.', [
  frame(
    'Start with unsorted meetings',
    'Use intervals [[7,10],[2,4],[12,14],[9,11]]. Their input order does not reveal the chronological neighbor checks.',
    intervals([
      { label: '[7,10]', start: 7, end: 10 },
      { label: '[2,4]', start: 2, end: 4 },
      { label: '[12,14]', start: 12, end: 14 },
      { label: '[9,11]', start: 9, end: 11 },
    ], { max: 14, input: '[[7,10],[2,4],[12,14],[9,11]]' }),
    'unsorted-intervals',
  ),
  frame(
    'Sort lexicographically',
    'intervals.sort() orders starts as [[2,4],[7,10],[9,11],[12,14]]. Equal starts would be ordered by end, but this input has distinct starts.',
    intervals(sorted, { max: 14, sorted: '[[2,4],[7,10],[9,11],[12,14]]' }),
    'sort-by-start',
  ),
  frame(
    'Check boundary at index 1',
    'Compare previous end 4 with current start 7. Since 4 <= 7, [2,4] finishes before [7,10] begins, and all continues.',
    intervals([
      { ...sorted[0], tone: 'state' },
      { ...sorted[1], tone: 'focus' },
      sorted[2],
      sorted[3],
    ], { max: 14, index: '1', boundary: '4 <= 7: true' }),
    'check-index-1',
  ),
  frame(
    'Check boundary at index 2',
    'Compare previous end 10 with current start 9. The test 10 <= 9 is false, and the bars overlap from time 9 to 10.',
    intervals([
      sorted[0],
      { ...sorted[1], tone: 'warning' },
      { ...sorted[2], tone: 'warning' },
      sorted[3],
    ], { max: 14, index: '2', boundary: '10 <= 9: false', overlap: '[9,10]' }),
    'check-index-2',
  ),
  frame(
    'Short-circuit and return false',
    'Python all stops after the false index-2 comparison, so index 3 is not evaluated. One person cannot attend [7,10] and [9,11].',
    intervals([
      sorted[0],
      { ...sorted[1], tone: 'warning' },
      { ...sorted[2], tone: 'warning' },
      sorted[3],
    ], { max: 14, skipped: 'index 3: 11 <= 12 not evaluated', result: 'false' }),
    'return-false',
  ),
]);

const review = {
  pattern: 'Sort intervals by start, then scan adjacent previous-end/current-start boundaries with early exit.',
  recognitionCue: 'Use it when one resource must handle every interval and any temporal overlap makes the schedule invalid.',
  invariant: 'Before checking index i, all adjacent boundaries before i are non-overlapping. In start-sorted order, any overlap must include an adjacent pair, so the first failed comparison proves impossibility.',
  stateModel: 'The minimal state after in-place sorting is the interval array and generator index. Each step compares intervals[i-1][1] <= intervals[i][0], and all short-circuits on false.',
  visualRationale: 'Aligned timeline bars expose actual duration and the overlap [9,10], while labels print the exact adjacent inequality and skipped generator step. The authored frames keep the same interval identities through sorting and comparison.',
  rejectedAlternatives: [
    'A start/end table was rejected because it hides the geometric overlap.',
    'A sweep-line event counter was rejected because it depicts a different algorithm.',
    'Only the conflicting pair was rejected because it omits sorting, the successful first check, and all short-circuiting.',
  ],
  transferLesson: 'Sort to make a global conflict local, then compare neighboring boundaries and stop at the first violation. This transfers to overlap validation, booking conflicts, and disjoint-range checks.',
  reviewStatus: 'reviewed',
};

export default defineVisual('meeting-rooms', draft, review);

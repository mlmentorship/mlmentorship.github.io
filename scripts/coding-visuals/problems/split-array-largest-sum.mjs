import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const nums = ['7', '2', '5', '10', '8'];
const searchScene = (mid, marks, extra) => array(nums, marks, {
  ...extra,
  motion: [
    { key: 'answer-midpoint', kind: 'pointer', x: mid, y: 0, label: `limit ${mid}` },
  ],
});

const draft = visual('Binary-search the first feasible capacity because the greedy number of contiguous parts never increases when capacity grows.', [
  frame('Initialize answer bounds', 'For nums = [7,2,5,10,8] and k = 2, no limit below max(nums)=10 can hold 10, while sum(nums)=32 always fits in one part.', searchScene(21, [], {
    bounds: 'lo = 10, hi = 32',
    midpoint: '(10 + 32) // 2 = 21',
    partition: 'not tested yet',
  }), 'initialize'),
  frame('Test limit 21', 'Greedy accumulates 7+2+5=14, then 14+10 exceeds 21, so it cuts before 10; [7,2,5] | [10,8] uses 2 parts and is feasible.', searchScene(21, [mark(3, 'cut before', 'focus', 'greedy-cut')], {
    partition: '[7,2,5] | [10,8]',
    sums: '14, 18',
    decision: '2 <= k, so hi: 32 -> 21',
    nextBounds: 'lo = 10, hi = 21',
  }), 'test-21'),
  frame('Test limit 15', 'Greedy cuts before 10 because 14+10 exceeds 15, then before 8 because 10+8 exceeds 15; three parts are too many.', searchScene(15, [
    mark(3, 'cut before', 'focus', 'greedy-cut-1'),
    mark(4, 'cut before', 'warning', 'greedy-cut-2'),
  ], {
    midpoint: '(10 + 21) // 2 = 15',
    partition: '[7,2,5] | [10] | [8]',
    sums: '14, 10, 8',
    decision: '3 > k, so lo: 10 -> 16',
    nextBounds: 'lo = 16, hi = 21',
  }), 'test-15'),
  frame('Test limit 18', 'The first cut occurs before 10 and [10,8] reaches exactly 18; two parts are feasible, so keep 18 and discard larger limits.', searchScene(18, [mark(3, 'cut before', 'focus', 'greedy-cut')], {
    midpoint: '(16 + 21) // 2 = 18',
    partition: '[7,2,5] | [10,8]',
    sums: '14, 18',
    decision: '2 <= k, so hi: 21 -> 18',
    nextBounds: 'lo = 16, hi = 18',
  }), 'test-18'),
  frame('Reject limit 17', 'After the cut before 10, adding 8 would make 18 > 17, forcing a third part; therefore 17 and every smaller candidate are infeasible.', searchScene(17, [
    mark(3, 'cut before', 'focus', 'greedy-cut-1'),
    mark(4, 'cut before', 'warning', 'greedy-cut-2'),
  ], {
    midpoint: '(16 + 18) // 2 = 17',
    partition: '[7,2,5] | [10] | [8]',
    sums: '14, 10, 8',
    decision: '3 > k, so lo: 16 -> 18',
    nextBounds: 'lo = 18, hi = 18',
  }), 'test-17'),
  frame('Return the first feasible limit', 'The search converges at lo = hi = 18. Greedy proves 18 feasible, while the rejected 17 proves no smaller maximum part sum works.', searchScene(18, [mark(3, 'optimal cut', 'output', 'greedy-cut')], {
    partition: '[7,2,5] | [10,8]',
    bounds: 'lo = hi = 18',
    result: '18',
  }), 'result'),
]);

const review = {
  pattern: 'Binary search on a monotone answer with a greedy feasibility scan.',
  recognitionCue: 'Use it when asked to minimize a numeric capacity or maximum load, and a candidate limit can be checked greedily with feasibility changing only once as the limit increases.',
  invariant: 'Every limit below lo is infeasible, at least one feasible answer lies at or below hi, and parts_needed(limit) is nonincreasing; each midpoint decision preserves the smallest feasible limit inside [lo, hi].',
  stateModel: 'The minimal search state is lo, hi, midpoint, and the greedy scan state used, running sum, and cut positions. The fixed array and explicit partitions show the evidence behind every bound movement.',
  visualRationale: 'The actual indexed values with cut markers expose contiguous greedy packing, while bound arithmetic explains the outer binary search; this is simpler than plotting every capacity and remains readable without color or playback.',
  rejectedAlternatives: [
    'A number-line-only binary search was rejected because it hides how feasibility is computed.',
    'A partition tree was rejected because the greedy predicate avoids enumerating all possible cuts.',
    'A final optimal partition alone was rejected because it does not prove that limit 17 is infeasible.',
  ],
  transferLesson: 'For minimize-the-maximum problems, search the answer when a greedy capacity check is monotone: feasible moves the upper bound down, infeasible moves the lower bound above the candidate.',
  reviewStatus: 'reviewed',
};

export default defineVisual('split-array-largest-sum', draft, review);

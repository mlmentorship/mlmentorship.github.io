import { defineVisual, frame, intervals, visual } from '../primitives.mjs';

const example = 'intervals = [[1,3], [3,5], [2,4], [5,7]]';
const sorted = [
  { label: '[1,3]', start: 1, end: 3 },
  { label: '[2,4]', start: 2, end: 4 },
  { label: '[3,5]', start: 3, end: 5 },
  { label: '[5,7]', start: 5, end: 7 },
];

function scene(candidate, tones, extra = {}) {
  return intervals(
    sorted.map((item, index) => ({ ...item, tone: tones[index] ?? 'neutral' })),
    {
      max: 7,
      example,
      sortedByEnd: '[1,3], [2,4], [3,5], [5,7]',
      ...extra,
      motion: [
        { key: 'candidate', kind: 'pointer', x: candidate, y: 0, label: `candidate ${candidate + 1}` },
        { key: 'last-end', kind: 'boundary', x: Number.isFinite(extra.lastEnd) ? extra.lastEnd : 0, y: 1, label: `last_end ${extra.lastEnd ?? '-inf'}` },
      ],
    },
  );
}

const draft = visual('Sort by end time; keep a candidate exactly when its start reaches the last kept end.', [
  frame(
    'Initialize after sorting',
    'Sorting by end gives [1,3], [2,4], [3,5], [5,7]. Start with last_end=-inf and removed=0.',
    scene(0, ['focus'], { lastEnd: '-inf', removed: '0' }),
    'initialize-end-order',
  ),
  frame(
    'Keep the first interval',
    'For [1,3], 1 < -inf is false. Keep it and change last_end from -inf to 3.',
    scene(0, ['state'], { comparison: '1 < -inf: false', lastEnd: 3, removed: '0', decision: 'keep [1,3]' }),
    'keep-one-three',
  ),
  frame(
    'Remove the overlap',
    'For [2,4], 2 < last_end 3 is true. Remove it, increment removed to 1, and leave last_end at 3.',
    scene(1, ['state', 'warning'], { comparison: '2 < 3: true', lastEnd: 3, removed: '1', decision: 'remove [2,4]' }),
    'remove-two-four',
  ),
  frame(
    'Keep a touching interval',
    'For [3,5], 3 < last_end 3 is false. Touching at 3 is allowed, so keep it and set last_end=5.',
    scene(2, ['state', 'warning', 'focus'], { comparison: '3 < 3: false', lastEnd: 5, removed: '1', decision: 'keep [3,5]' }),
    'keep-three-five',
  ),
  frame(
    'Keep the final interval',
    'For [5,7], 5 < last_end 5 is false. Keep it and set last_end=7.',
    scene(3, ['state', 'warning', 'state', 'focus'], { comparison: '5 < 5: false', lastEnd: 7, removed: '1', decision: 'keep [5,7]' }),
    'keep-five-seven',
  ),
  frame(
    'Return the removal count',
    'Exactly [2,4] was removed; [1,3], [3,5], and [5,7] remain pairwise non-overlapping.',
    scene(3, ['output', 'warning', 'output', 'output'], { lastEnd: 7, kept: '3', result: '1' }),
    'return-one-removal',
  ),
]);

const review = {
  pattern: 'Greedy interval scheduling after sorting candidates by ascending end time.',
  recognitionCue: 'The task asks for the fewest removals needed to leave non-overlapping ranges, which is equivalent to keeping the largest compatible subset.',
  invariant: 'After each candidate, the kept intervals do not overlap, last_end is the end of the latest kept interval, and the greedy choices leave at least as much future room as any alternative of the same size.',
  stateModel: 'Retain only the end-sorted intervals, the moving candidate, last_end, and removed count. The complete kept list is shown for explanation but is not required by the implementation.',
  visualRationale: 'Ranges drawn to scale on one time axis make overlap and endpoint touching visible; a moving candidate and last_end boundary explain each keep/remove branch.',
  rejectedAlternatives: [
    'A prose iteration table hides the geometric meaning of start < last_end.',
    'An overlap graph adds edges and a graph-selection problem when one timeline boundary is sufficient.',
    'Sorting by start and merging depicts a different goal and does not justify the earliest-finish exchange.',
  ],
  transferLesson: 'When selecting the most compatible time ranges, an earlier finishing accepted choice never blocks a range that a later-finishing choice could keep; reuse this exchange argument for activity and reservation scheduling.',
  independentReview: '3.3 source-to-frame replay',
  reviewStatus: 'reviewed',
};

export default defineVisual('non-overlapping-intervals', draft, review);

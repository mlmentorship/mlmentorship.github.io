import { bars, defineVisual, frame, visual } from '../primitives.mjs';

const heights = [1, 8, 6, 2, 5, 4, 8, 3, 7];
const labels = (left, right) => heights.map((height, index) => {
  if (index === left) return `L=${height}`;
  if (index === right) return `R=${height}`;
  return String(height);
});
const pointerMotion = (left, right) => [
  { key: 'left', kind: 'pointer', x: left, y: heights[left], label: `L at ${left}` },
  { key: 'right', kind: 'pointer', x: right, y: heights[right], label: `R at ${right}` },
];
const candidate = (left, right, best, move, reason, result) => bars(heights, {
  labels: labels(left, right),
  area: {
    left,
    right,
    height: Math.min(heights[left], heights[right]),
    label: `${right - left} x min(${heights[left]}, ${heights[right]}) = ${(right - left) * Math.min(heights[left], heights[right])}`,
  },
  coveredRange: `[${left}..${right}]`,
  best,
  move,
  reason,
  ...(result ? { result } : {}),
  motion: pointerMotion(left, right),
});

const draft = visual('Evaluate both boundary walls, then move only the shorter wall because keeping it cannot produce a taller container at a smaller width.', [
  frame(
    'Initialize at both ends',
    'For heights [1, 8, 6, 2, 5, 4, 8, 3, 7], L=0 and R=8 give width 8 and area 8. Height 1 is limiting, so move L right.',
    candidate(0, 8, 'max(0, 8) = 8', 'L: 0 -> 1', 'left height 1 < right height 7'),
    'range-0-8',
  ),
  frame(
    'Find area 49',
    'L=1 and R=8 give width 7 and area 49, the new best. Height 7 is limiting, so move R left.',
    candidate(1, 8, 'max(8, 49) = 49', 'R: 8 -> 7', 'right height 7 < left height 8'),
    'range-1-8',
  ),
  frame(
    'Discard right height 3',
    'L=1 and R=7 give width 6 and area 18. The right wall is shorter, so only moving R can possibly raise the limiting height.',
    candidate(1, 7, 'max(49, 18) = 49', 'R: 7 -> 6', 'right height 3 < left height 8'),
    'range-1-7',
  ),
  frame(
    'Handle equal height 8',
    'L=1 and R=6 give width 5 and area 40. Equal heights cannot improve if either one is kept at a smaller width; the implementation moves R.',
    candidate(1, 6, 'max(49, 40) = 49', 'R: 6 -> 5', 'equal heights take the else branch'),
    'range-1-6',
  ),
  frame(
    'Discard right height 4',
    'L=1 and R=5 give width 4 and area 16. The right wall limits the area, so move R left.',
    candidate(1, 5, 'max(49, 16) = 49', 'R: 5 -> 4', 'right height 4 < left height 8'),
    'range-1-5',
  ),
  frame(
    'Discard right height 5',
    'L=1 and R=4 give width 3 and area 15. The right wall is still shorter, so move R left.',
    candidate(1, 4, 'max(49, 15) = 49', 'R: 4 -> 3', 'right height 5 < left height 8'),
    'range-1-4',
  ),
  frame(
    'Discard right height 2',
    'L=1 and R=3 give width 2 and area 4. The right wall limits the area, so move R left.',
    candidate(1, 3, 'max(49, 4) = 49', 'R: 3 -> 2', 'right height 2 < left height 8'),
    'range-1-3',
  ),
  frame(
    'Meet and return the best',
    'L=1 and R=2 give width 1 and area 6. Move R to L; no pair remains, and the maximum area stays 49.',
    candidate(1, 2, 'max(49, 6) = 49', 'R: 2 -> 1; pointers meet', 'right height 6 < left height 8', '49'),
    'range-1-2',
  ),
]);

export default defineVisual('container-with-most-water', draft, {
  pattern: 'Greedy two pointers at opposite ends of a height array.',
  recognitionCue: 'Choose two boundary positions to maximize width times the smaller boundary height; moving inward always loses width, so only a taller limiting wall can compensate.',
  invariant: 'Before each iteration, best is the largest area among discarded boundary pairs. Moving the shorter wall cannot discard a better pair because every pair that keeps it has smaller width and height no greater than that same short wall.',
  stateModel: 'Keep left and right indices plus the best area seen. Current width and limiting height are derived from those indices, so no window contents or auxiliary collection is needed.',
  visualRationale: 'Vertical bars preserve the height geometry, the measured rectangle prints width x limiting height, L and R labels stay visible on the boundary bars, and authored pointer motion keys track both boundaries across every iteration.',
  rejectedAlternatives: [
    'A plain indexed array shows pointer positions but hides that the shorter vertical wall determines water height.',
    'A pair-by-pair area table lists arithmetic without exposing the geometric discard argument.',
    'A brute-force matrix of all pairs visualizes O(n^2) work rather than the supplied greedy scan.',
  ],
  transferLesson: 'When moving either boundary worsens one factor, move the boundary responsible for the current bottleneck; retaining that bottleneck cannot improve the objective as the other factor shrinks.',
  reviewStatus: 'reviewed',
});

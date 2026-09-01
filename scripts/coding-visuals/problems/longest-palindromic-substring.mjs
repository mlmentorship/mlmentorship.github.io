import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const text = ['b', 'a', 'b', 'a', 'd'];

function centerFrame(middle, left, right, details, best, result) {
  return array(text, [
    mark(left, `L=${left}`, result ? 'output' : 'state', 'left'),
    ...(right === left ? [] : [mark(right, `R=${right}`, result ? 'output' : 'focus', 'right')]),
    mark(middle, `middle=${middle}`, 'focus', 'middle'),
  ], {
    input: 'babad',
    calls: `expand(${middle},${middle}); expand(${middle},${middle + 1})`,
    expansions: details,
    bestRange: best,
    ...(result ? { result } : {}),
  });
}

const draft = visual('Every odd or even palindrome is discovered by expanding its unique center; update the saved range only when an expansion is strictly longer.', [
  frame(
    'Initialize best range',
    'For text babad, best_left = best_right = 0, so the initial saved palindrome is b.',
    array(text, [mark(0, 'best [0,0]', 'state', 'middle')], {
      input: 'babad',
      bestRange: '[0,0] -> b',
    }),
    'initialize',
  ),
  frame('Process middle 0', 'Odd expansion accepts b but ties the saved length; even expansion compares b with a and stops.', centerFrame(0, 0, 0, 'odd [0,0]=b; even b!=a', '[0,0] -> b'), 'middle-0'),
  frame('Process middle 1', 'Odd expansion accepts a, then bab. Width 2 is greater than saved width 0, so best becomes [0,2]; even a!=b.', centerFrame(1, 0, 2, 'odd a -> bab -> bounds stop; even a!=b', '[0,2] -> bab'), 'middle-1'),
  frame('Process middle 2', 'Odd expansion accepts b, then aba. Its width 2 ties bab, so the strict greater-than branch keeps bab; even b!=a.', centerFrame(2, 1, 3, 'odd b -> aba -> b!=d; even b!=a', '[0,2] -> bab'), 'middle-2'),
  frame('Process middle 3', 'Odd expansion accepts a then b!=d; even compares a with d and stops. Neither candidate beats bab.', centerFrame(3, 3, 3, 'odd a -> b!=d; even a!=d', '[0,2] -> bab'), 'middle-3'),
  frame('Process middle 4 and return', 'Odd expansion accepts d and reaches the boundary; even starts out of bounds. Slice [0:3] returns bab.', centerFrame(4, 0, 2, 'odd d -> boundary; even right=5 out of bounds', '[0,2] -> bab', 'bab'), 'middle-4'),
]);

export default defineVisual('longest-palindromic-substring', draft, {
  pattern: 'Expand around every character center and every gap center.',
  recognitionCue: 'The answer is a contiguous palindrome, whose symmetry guarantees one unique odd character center or even gap center.',
  invariant: 'Within expand(left,right), text[left:right+1] is palindromic before the pointers move outward. After each completed center, the saved range is the longest palindrome seen at any processed center.',
  stateModel: 'Keep the current middle, expanding left/right boundaries, and global best_left/best_right. No substring copies or DP table are required.',
  visualRationale: 'The actual indexed string with stable middle, left, and right keys shows center geometry, outward movement, mismatch stops, and the strict tie rule; printed ranges preserve meaning without color.',
  rejectedAlternatives: [
    'A final highlighted substring hides the center scans and why aba does not replace bab.',
    'A two-dimensional palindrome DP grid uses more space than the supplied center expansion.',
    'A list of substrings obscures symmetry and repeats character comparisons.',
  ],
  transferLesson: 'When validity is symmetric around a center, enumerate both center types and expand until the invariant breaks; change only the aggregation to find a longest value, count all values, or validate radii.',
  reviewStatus: 'reviewed',
});

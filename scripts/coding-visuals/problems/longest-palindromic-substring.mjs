import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const text = ['b', 'a', 'b', 'a', 'd'];

function centerFrame(middle, left, right, details, best, result) {
  return array(text, [
    mark(left, result ? 'bL' : 'aL', result ? 'output' : 'state', 'left'),
    ...(right === left ? [] : [mark(right, result ? 'bR' : 'aR', result ? 'output' : 'focus', 'right')]),
    mark(middle, 'm', 'focus', 'middle'),
  ], {
    input: 'babad',
    fitArray: true,
    arrayColumns: 5,
    savedRange: { start: best[0], end: best[1] },
    calls: `expand(${middle},${middle}); expand(${middle},${middle + 1})`,
    acceptedOdd: result ? `[${middle},${middle + 1}) = d` : `[${left},${right + 1})`,
    expansions: details,
    bestRange: `[${best[0]},${best[1]}) = ${text.slice(best[0], best[1]).join('')}`,
    evenContrast: 'cbbd: gap at 1/2 accepts [1,3) = bb',
    ...(result ? { result } : {}),
  });
}

const draft = visual('Expand every character center and every gap center. Save only a strictly longer accepted range.', [
  frame(
    'Initialize best range',
    'For babad, save inclusive [0,0], the first b. The underline uses half-open range [0,1).',
    array(text, [mark(0, 'm', 'state', 'middle')], {
      input: 'babad',
      fitArray: true,
      arrayColumns: 5,
      savedRange: { start: 0, end: 1 },
      bestRange: '[0,1) = b',
      evenContrast: 'cbbd: gap at 1/2 accepts [1,3) = bb',
    }),
    'initialize',
  ),
  frame('Process middle 0', 'Odd center zero accepts b. Even center zero rejects b versus a. The saved range does not change.', centerFrame(0, 0, 0, 'odd b; even b!=a', [0,1]), 'middle-0'),
  frame('Process middle 1', 'Odd center one accepts a, then bab. Save inclusive [0,2]. Its next radius is outside the string.', centerFrame(1, 0, 2, 'odd a, then bab, then bounds stop; even a!=b', [0,3]), 'middle-1'),
  frame('Process middle 2', 'Odd center two accepts b, then aba. Equal width does not replace bab. The next endpoints b and d disagree.', centerFrame(2, 1, 3, 'odd b, then aba, then b!=d; even b!=a', [0,3]), 'middle-2'),
  frame('Process middle 3', 'Odd center three accepts a. Its next endpoints b and d disagree. Even endpoints a and d also disagree.', centerFrame(3, 3, 3, 'odd a, then b!=d; even a!=d', [0,3]), 'middle-3'),
  frame('Process middle 4 and return', 'Odd center four accepts d. Its next radius and the even call exceed the boundary. Return saved bab, not d.', centerFrame(4, 0, 2, 'odd d, then boundary; even right=5 outside', [0,3], 'bab'), 'middle-4'),
]);

export default defineVisual('longest-palindromic-substring', draft, {
  pattern: 'Expand around every character center and every gap center.',
  recognitionCue: 'The answer is a contiguous palindrome, whose symmetry guarantees one unique odd character center or even gap center.',
  invariant: 'A successful guard certifies text[left:right+1] before updates. After the pointers move, the next candidate is untested. After each center, the saved range is a longest accepted range so far.',
  stateModel: 'Keep the current middle, candidate left/right endpoints and saved inclusive best endpoints. The scan uses constant auxiliary pointer state; returning text copies the saved slice.',
  visualRationale: 'The fitted indexed string retains every original pointer identity. A separate saved-range underline distinguishes bab from the tied accepted aba. Compact aL/aR labels denote accepted endpoints, final bL/bR denote saved endpoints, and the cbbd contrast makes the even-center call explicit.',
  rejectedAlternatives: [
    'A final highlighted substring hides the center scans and why aba does not replace bab.',
    'A two-dimensional palindrome DP grid uses more space than the supplied center expansion.',
    'A list of substrings obscures symmetry and repeats character comparisons.',
  ],
  transferLesson: 'When validity is symmetric around a center, enumerate both center types and expand until the invariant breaks; change only the aggregation to find a longest value, count all values, or validate radii.',
  reviewStatus: 'reviewed',
});

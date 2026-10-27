import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const chars = ['A', ',', 'space', 'b', 'space', 'a', '!'];
const state = (left, right, extra = {}) => array(chars, [
  mark(left, 'L', 'state', 'left-pointer'),
  mark(right, 'R', 'focus', 'right-pointer'),
], { coveredRange: `[${left}..${right}]`, fitArray: true, arrayColumns: 7, ...extra });

const draft = visual('Keep raw text in place, skip non-alphanumeric endpoints one at a time, and move both pointers only after a case-insensitive match.', [
  frame('Initialize at both raw endpoints', 'For "A, b a!", L=0 points to A and R=6 points to !.', state(0, 6, {
    action: 'initialize opposite-end pointers',
  }), 'initialize'),
  frame('Skip punctuation on the right', 'L points to alphanumeric A. R points to !, so decrement only R from 6 to 5.', state(0, 5, {
    movement: 'R: 6 -> 5 because ! is not alphanumeric',
  }), 'skip-right-punctuation'),
  frame('Compare the first pair', 'A and a match after lowercase comparison. Advance L from 0 to 1 and R from 5 to 4.', state(1, 4, {
    comparison: 'a == a',
    movement: 'L: 0 -> 1; R: 5 -> 4 after match',
  }), 'match-a'),
  frame('Skip the comma on the left', 'At L=1, comma is not alphanumeric. Increment only L to 2.', state(2, 4, {
    movement: 'L: 1 -> 2 because comma is not alphanumeric',
  }), 'skip-comma'),
  frame('Skip the space on the left', 'At L=2, space is not alphanumeric, so L advances again to index 3.', state(3, 4, {
    movement: 'L: 2 -> 3 because space is not alphanumeric',
  }), 'skip-left-space'),
  frame('Skip the space on the right', 'L points to b and R=4 points to a space. Decrement only R to 3.', state(3, 3, {
    movement: 'R: 4 -> 3 because space is not alphanumeric',
  }), 'skip-right-space'),
  frame('Stop when pointers meet', 'Now L=R=3 at b, so left < right is false. Every outer pair matched, so return True.', state(3, 3, {
    guard: '3 < 3 is false',
    result: 'true',
  }), 'finish'),
]);

const review = {
  pattern: 'Two pointers moving inward over filtered endpoint characters.',
  recognitionCue: 'Use opposite-end pointers when equality must hold symmetrically after ignoring characters, case, or other endpoint noise.',
  invariant: 'Before each loop iteration, every accepted alphanumeric pair outside [left,right] matches case-insensitively; discarded outside characters are non-alphanumeric and cannot affect the answer.',
  stateModel: 'Keep only left and right indices into the original text; skip exactly one invalid endpoint per iteration, compare valid endpoints, and move both only after equality.',
  visualRationale: 'The full raw character array preserves punctuation and spacing while stable L and R keys visibly move inward; range, branch reason, and direction labels make each boundary movement explicit.',
  rejectedAlternatives: [
    'Building a normalized copy and reversing it is concise but uses O(n) extra space and hides endpoint skipping.',
    'A normalized-only array erases the punctuation branches executed by the supplied implementation.',
    'A prose comparison list cannot show that only one pointer moves on each skip branch.',
  ],
  transferLesson: 'When irrelevant data appears at either boundary, repair one endpoint at a time until both are comparable; only then apply the symmetric predicate and advance both.',
  reviewStatus: 'reviewed',
};

export default defineVisual('valid-palindrome', draft, review);

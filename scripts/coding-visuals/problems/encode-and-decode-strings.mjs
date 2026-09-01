import { array, defineVisual, frame, mark, visual } from '../primitives.mjs';

const data = '4#lint1##0#'.split('');
const cursor = (index, label, extra = {}) => array(data, [mark(index, label, extra.result ? 'output' : 'focus', 'decoder-index')], {
  encoded: '4#lint1##0#',
  ...extra,
});

const draft = visual('A decimal length before # makes each payload boundary explicit, even when the payload contains # or is empty.', [
  frame('Encode lint', 'The first input string has length 4, so encoding emits 4#lint.', array(['lint', '#', ''], [mark(0, 'encode item 0', 'focus', 'codec-cursor')], { emitted: '4#lint' }), 'encode-lint'),
  frame('Encode #', 'The second payload is the delimiter character itself, but its length prefix makes 1## unambiguous.', array(['lint', '#', ''], [mark(1, 'encode item 1', 'focus', 'codec-cursor')], { emitted: '4#lint1##' }), 'encode-hash'),
  frame('Encode the empty string', 'The empty payload has length 0, so it emits 0# and the complete data is 4#lint1##0#.', array(['lint', '#', ''], [mark(2, 'encode item 2', 'focus', 'codec-cursor')], { emitted: '4#lint1##0#' }), 'encode-empty'),
  frame('Decode length 4', 'At index 0, separator=1 and length=int(data[0:1])=4. Move index to 2.', cursor(0, 'index=0', { separator: '1', length: '4', nextPayload: 'data[2:6]' }), 'decode-length-4'),
  frame('Consume lint', 'Append data[2:6] = lint, then index += 4 moves from 2 to 6.', cursor(6, 'index=6', { decoded: '["lint"]', consumed: '[2:6] -> lint' }), 'decode-lint'),
  frame('Decode payload #', 'From index 6, separator=7 and length=1. Move to 8, append data[8:9] = #, then advance to 9.', cursor(9, 'index=9', { decoded: '["lint", "#"]', parse: 'length data[6:7]=1; payload data[8:9]=#' }), 'decode-hash'),
  frame('Decode the empty payload', 'From index 9, separator=10 and length=0. Move to 11, append data[11:11] = empty, and the loop ends.', cursor(10, 'separator=10', { decoded: '["lint", "#", ""]', parse: 'length data[9:10]=0; payload data[11:11]=""', finalIndex: '11 = len(data)', result: '["lint", "#", ""]' }), 'decode-empty'),
]);

export default defineVisual('encode-and-decode-strings', draft, {
  pattern: 'Self-delimiting length-prefix framing.',
  recognitionCue: 'Arbitrary strings, including empty strings and delimiter characters, must be concatenated and later recovered without escaping ambiguity.',
  invariant: 'At the start of each decode loop, index points to the first decimal length digit. After locating # and consuming exactly length payload characters, index points to the next length or end of data.',
  stateModel: 'Encoding needs each text and its length. Decoding keeps the encoded data, index, separator, parsed length, and output list; payload content never affects boundary detection.',
  visualRationale: 'The exact encoded character stream and stable decoder-index key show header scan, payload slice, and cursor advance for every item, including # and the zero-length slice.',
  rejectedAlternatives: [
    'Joining with # fails when a payload contains # and cannot distinguish some empty-string lists.',
    'Escaping delimiters adds a second grammar and more error-prone scanning.',
    'A result-only round trip hides the separator and slice arithmetic.',
  ],
  transferLesson: 'Prefix variable-length records with a parseable size so payload bytes remain opaque; this framing pattern transfers to network protocols, file formats, and binary serialization.',
  reviewStatus: 'reviewed',
});

import assert from 'node:assert/strict';

export const mentalPictures = {
  'number-of-islands-state': {
    operation: 'mark-before-enqueue', grid: [[1, 1], [1, 1]],
    review: 'A shared-cell table hides two live discovery routes. Show both neighbors reaching the same bottom-right object, then keep one queue entry after the second route sees its discovery mark.',
    prediction: 'Would marking only when the shared cell is popped stop both earlier neighbors from enqueueing it?',
  },
  'pacific-atlantic-water-flow-state': {
    operation: 'reversed-height-admission', cases: [[2, 5], [2, 1]],
    review: 'A comparison table hides the direction reversal. Put heights on one vertical ruler and walk from ocean-side two toward five, refusing the lower one as a possible reverse source.',
    prediction: 'Is an equal-height neighbor allowed by the reversed search, or must every step rise strictly?',
  },
  'rotate-image-state': {
    operation: 'transpose-then-row-reverse', grid: [[1, 2], [3, 4]],
    review: 'Appended before/after rows hide the two coordinate operations. Preserve every tile through an actual transpose and then a horizontal row reversal, following original one to its final top-right position.',
    prediction: 'Where does original top-left one end after both coordinate changes?',
  },
  'rotting-oranges-state': {
    operation: 'simultaneous-infection-waves', arrivals: [[0, 1, 2], [1, 2, 'wall'], ['wall', 3, 4]],
    review: 'A time grid alone obscures why counts are not minutes. Group the same cell identities into five arrival buckets, separating the initial seed from four simultaneous infection waves.',
    prediction: 'Do the two cells arriving at minute one require two successive minutes?',
  },
  'set-matrix-zeroes-state': {
    operation: 'frozen-zero-obligations', grid: [[1, 0, 1], [1, 1, 1]],
    review: 'Appended matrices omit ownership of the obligations. Enclose only original row zero and column one, then preserve the lower corner ones after new output zeros appear.',
    prediction: 'What incorrect extra columns would be added by rescanning the newly zeroed top row?',
  },
  'spiral-matrix-state': {
    operation: 'crossed-spiral-boundary', grid: [[1, 2, 3]],
    review: 'Boundary tokens omit the consumed one-row object. Sweep all three positions once and show top moving beyond the unchanged bottom before the opposite sweep is considered.',
    prediction: 'Which already-emitted values would be duplicated if the bottom sweep ignored top exceeding bottom?',
  },
  'binary-precision-and-recall-state': {
    operation: 'select-denominator',
    review: 'A confusion table hides the membership in each denominator. Reuse the same eight true-positive identities while selecting actual positives for recall and predicted positives for precision.',
    prediction: 'Which thirty-two cases enter the precision denominator but not the recall denominator?',
  },
  'causal-attention-state': {
    operation: 'causal-weight-products', query: 1, values: [1, 2, 100],
    review: 'Zero table entries can be mistaken for missing inputs. Keep the large future value present but send it through its exactly zero weight before combining contributions.',
    prediction: 'If the future scalar grows from one hundred to one thousand, does its masked contribution change?',
    caption: 'At query one, weights .4,.6,0 act on illustrative scalar values 1,2,100. Products .4,1.2,0 sum to 1.6. The future value stays present but contributes zero.',
  },
  'counting-bits-state': {
    operation: 'halved-popcount-reuse', value: 13,
    review: 'A shifted token row omits ownership of the reused result. Preserve the three high bits as six and route the separated low bit into the final count addition.',
    prediction: 'Which count is reused after dropping the low one bit of binary 1101?',
  },
  'cross-entropy-from-logits-state': {
    operation: 'shared-normalizer-target', logits: [2, 0],
    review: 'Two subtraction rows hide one shared normalizer. Fork the same log-normalizer into two target-specific subtraction routes instead of implying separate logits or separate normalizers.',
    prediction: 'With logits fixed, which quantity changes when the target changes from class zero to class one?',
  },
  'mini-batches-state': {
    operation: 'bounded-batch-membership', count: 7, size: 3,
    review: 'Slice rows hide continuity and the short final group. Keep all seven input identities on one lane and enclose exactly three, three, and one without duplicating or dropping the tail.',
    prediction: 'Which real input identity disappears if the one-item final slice is discarded?',
  },
  'missing-number-state': {
    operation: 'xor-range-pairing', count: 3, observed: [3, 0, 1],
    review: 'Range/input rows hide cancellation identity. Align equal range and observation terms in pairs, leaving expected value two as the only unpaired term.',
    prediction: 'Does reordering the observed values change which expected term remains unpaired?',
  },
  'number-of-1-bits-state': {
    operation: 'clear-lowest-one', value: 12,
    review: 'Binary strings omit the exact bit removed each pass. Keep four bit positions fixed and mark the lowest remaining one before it disappears in the next snapshot.',
    prediction: 'Does an empty low position require an extra clearing pass?',
  },
  'pad-variable-length-sequences-state': {
    operation: 'length-owned-mask', sequences: [[0, 7], [4]], padding: 0,
    review: 'Value/validity strings hide identical zeros with different ownership. Draw both zeros at equal size but attach validity to original length, not value equality.',
    prediction: 'Would masking every zero incorrectly erase a genuine token from the first sequence?',
  },
  'pairwise-squared-distances-state': {
    operation: 'distance-dot-identity', points: [[1, 2], [3, 2]],
    review: 'Norm/dot tokens hide the displacement they equal. Put both vectors in the same plane, measure their two-unit gap, then reconcile squared distance four with the row-norm identity.',
    prediction: 'If both points move upward together, does their displacement change even though their individual norms change?',
  },
  'product-of-array-except-self-state': {
    operation: 'excluded-zero-product', values: [2, 0, 4], index: 1,
    review: 'Prefix/suffix rows obscure the zero counterexample. Cross only the current middle zero out of the selected factors and combine the two surviving nonzero sides into eight.',
    prediction: 'Why is the output at the zero position nonzero while the other two outputs are zero?',
  },
  'reverse-bits-state': {
    operation: 'fixed-width-bit-mirror', width: 32, value: 1,
    review: 'Endpoint bit labels hide the retained leading positions. Show all thirty-two fixed slots and carry the sole low bit across to high position thirty-one.',
    prediction: 'Would reversing only the written character one preserve the required thirty-two-bit position mapping?',
  },
  'stable-softmax-state': {
    operation: 'stable-shift-invariance', logits: [1000, 1001],
    review: 'Probability shares alone hide the overflow-avoiding transformation. Carry both large logits through the same maximum subtraction before creating their unchanged normalized shares.',
    prediction: 'Does adding another thousand to both original logits change their shifted difference or their probability shares?',
  },
  'subarray-sum-equals-k-state': {
    operation: 'query-before-insert', mode: 'prefix', values: [0], target: 0,
    review: 'Frequency rows hide the order-sensitive count. Query the seeded empty-prefix receipt once, then update the stored count without performing a second query.',
    prediction: 'Which spurious extra match appears if the current zero prefix is inserted before its query?',
  },
  'sum-of-two-integers-state': {
    operation: 'xor-carry-trace', states: [[5, 3], [6, 2], [4, 4], [0, 8], [8, 0]],
    review: 'Decimal pairs hide the carried bit overlap. Preserve the low four positions for provisional sum and shifted carry separately through every simultaneous update.',
    prediction: 'At the four-and-four snapshot, why does XOR become zero while the carry moves to eight?',
  },
  'two-sum-state': {
    operation: 'query-before-insert', mode: 'complement', values: [3, 3], target: 6,
    review: 'Lookup/store tokens hide whether the current position is already in memory. Query the empty map at index zero before storing it; only index one may find that earlier partner.',
    prediction: 'Could index zero incorrectly match itself if its three were stored before the first lookup?',
  },
  'decode-string-state': {
    operation: 'nested-repeat-payload', inner: 'b', innerCount: 3, prefix: 'a', outerCount: 2,
    review: 'Completed strings hide nested ownership. Enclose the three inner b copies inside one outer a-plus-bbb payload, then duplicate the completed payload as a whole.',
    prediction: 'Would repeating the outer a before resolving its inner b group produce the same payload?',
  },
  'encode-and-decode-strings-state': {
    operation: 'length-delimited-payload', payload: 'a#b', separator: '#',
    review: 'Length and payload rows hide the exact span. Keep all encoded characters in one lane and enclose exactly three payload characters, including their internal separator-looking symbol.',
    prediction: 'Which hash is the length separator, and which hash must remain inside the decoded string?',
  },
  'implement-trie-state': {
    operation: 'trie-prefix-terminal', word: 'apple', prefix: 'app',
    review: 'Path/terminal cells conceal unchanged shared nodes. Draw the same apple chain before and after inserting app and alter only the app terminal flag.',
    prediction: 'Does inserting app require replacing the existing apple path or deleting its terminal flag?',
  },
  'design-add-and-search-words-state': {
    operation: 'wildcard-terminal-search', word: 'bad',
    review: 'A matching path is not a successful search without a terminal node. Compare the prefix-only ba stop with b.d continuing through the stored a child to terminal d.',
    prediction: 'Why is the existing ba path insufficient while b.d reaches a stored word?',
  },
  'word-search-ii-state': {
    operation: 'emit-and-continue-trie', words: ['a', 'ab'], board: ['a', 'b'],
    review: 'Terminal/child rows can suggest stopping after the first emission. Keep both terminal nodes and the live child edge so emitting a does not lose ab.',
    prediction: 'Which valid word disappears if visiting terminal a returns immediately instead of exploring its b child?',
  },
  'word-search-state': {
    operation: 'used-board-cell-refusal', board: ['A', 'B'], requested: [0, 1, 0],
    review: 'Letter tokens hide physical cell identity. Keep the two real board cells above the three requested visits, marking the final request as reuse of cell zero rather than a new A.',
    prediction: 'Does the final matching A create a third board cell, or request the already-used first cell again?',
  },
  'valid-palindrome-state': {
    operation: 'normalized-end-comparison', input: 'A,b:a',
    review: 'A normalized token row omits the filtering routes. Drop punctuation connections, normalize the retained A, and compare the two resulting a endpoints.',
    prediction: 'Would comparing raw uppercase A against lowercase a without normalization preserve this answer?',
  },
  'valid-parentheses-state': {
    operation: 'top-opener-refusal', stack: ['(', '['], incoming: ')',
    review: 'Required/actual tokens hide stack order. Place the mismatching square opener on top and the tempting matching round opener below the only permitted comparison.',
    prediction: 'Can the closer skip the top square opener to use the round opener deeper in the stack?',
  },
  'meeting-rooms-state': {
    operation: 'next-interval-fit', cases: [{ kept: [8, 10], next: [10, 12] }, { kept: [8, 10], next: [9, 12] }],
    title: 'An equal endpoint touches; a strictly earlier start overlaps',
    review: 'A comparison table does not show the interiors. Put both interval tracks on one shared time ruler and preserve the empty intersection at equality.',
    prediction: 'Which positive-length overlap appears when the next start moves from ten to nine?',
  },
  'non-overlapping-intervals-state': {
    operation: 'next-interval-fit', cases: [{ kept: [1, 3], next: [3, 4] }, { kept: [1, 5], next: [3, 4] }],
    title: 'An earlier retained finish leaves the next interval available',
    review: 'Yes/no cells hide the opportunity preserved by the exchange. Keep the next interval fixed while extending the retained finish from three to five.',
    prediction: 'Which future interval becomes impossible if the retained finish moves from three to five?',
  },
  'insert-interval-state': {
    operation: 'interval-stage-union',
    stages: [{ label: 'New [2,5] lies between the sorted neighbors', spans: [[1, 3], [2, 5], [6, 9]] }, { label: 'Merge the overlap; preserve separated [6,9]', spans: [[1, 5], [6, 9]] }], expected: [[1, 5], [6, 9]],
    review: 'Interval tokens hide overlap and separation. Align every span on one ruler so only the first neighbor joins the new interval.',
    prediction: 'What gap prevents interval six-to-nine from entering the first merged span?',
  },
  'merge-intervals-state': {
    operation: 'interval-stage-union',
    stages: [{ label: 'Sorted spans awaiting comparison', spans: [[1, 4], [2, 6], [5, 7]] }, { label: 'First merge extends active end to six', spans: [[1, 6], [5, 7]] }, { label: 'Pending five-to-seven overlaps updated end six', spans: [[1, 7]] }], expected: [[1, 7]],
    review: 'An updated endpoint token omits the newly available overlap. Keep all spans on the same ruler and grow the active right edge before testing five-to-seven.',
    prediction: 'Would comparing the final start five against the obsolete end four preserve the correct union?',
  },
  'merge-k-sorted-lists-state': {
    operation: 'advance-winning-head', lists: [[1, 9], [4], [2]],
    review: 'Head rows obscure list ownership. Advance only the list that supplied one, then compare its exposed nine against the unchanged heads four and two.',
    prediction: 'After emitting one, why is its successor nine not automatically the next output?',
  },
  'network-delay-time-state': {
    operation: 'stale-distance-entry', best: 2, queued: [2, 6],
    review: 'Expand/skip cells hide one shared authoritative record. Compare both queued estimates against the same best-distance store, stopping the stale six before expansion.',
    prediction: 'Does a later-popped six overwrite the already-known shorter distance two?',
  },
  'best-time-to-buy-and-sell-stock-state': {
    operation: 'saved-witness-trace', mode: 'profit', values: [3, 1, 5, 2], expected: [[0, 0, 4, 1], [0, 0, 4, 4]],
    title: 'A worse selling day does not erase the best completed trade',
    review: 'A current/saved pair hides persistence through time. Plot both profit traces on the same scale and retain the earlier four when the last current trade falls to one.',
    prediction: 'Does the final selling price two remove the earlier opportunity to buy at one and sell at five?',
  },
  'maximum-subarray-state': {
    operation: 'saved-witness-trace', mode: 'sum', values: [-2, 3, 4, -10], expected: [[-2, 3, 7, -3], [-2, 3, 7, 7]],
    title: 'The current suffix can lose while the saved optimum survives',
    review: 'A final pair of sums omits the completed witness. Carry the saved seven horizontally after the current suffix falls below zero.',
    prediction: 'Must the best subarray include the final minus ten?',
  },
  'longest-palindromic-substring-state': {
    operation: 'saved-witness-trace', mode: 'palindrome', values: ['level', 'aba'], expected: [[5, 3], [5, 5]],
    title: 'A later shorter palindrome does not replace the saved best',
    review: 'Two length bars omit the update history. Use real palindrome witnesses and carry five through a later length-three candidate.',
    prediction: 'Does finding aba later make the earlier level witness cease to be a palindrome?',
  },
  'meeting-rooms-ii-state': {
    operation: 'saved-witness-trace', mode: 'rooms', values: [.25, 1, 3.5], intervals: [[0, 2], [.5, 1.5], [3, 4]], expected: [[1, 2, 1], [1, 2, 2]],
    title: 'The saved room peak survives a smaller final active heap',
    review: 'Final and peak bars hide the occupancy history. Count the actual meetings at three ordered checkpoints and retain the two-room peak after occupancy returns to one.',
    prediction: 'Can the smaller occupancy at the last checkpoint retroactively eliminate the earlier need for two rooms?',
  },
  'longest-repeating-character-replacement-state': {
    operation: 'replacement-window-witness', windows: ['AABA', 'ABBA'], budget: 1,
    review: 'Validity tokens hide the edit demand. Mark the nonmajority characters in each actual window and keep the first valid length-four witness after the later window needs two edits.',
    prediction: 'Does the later invalid ABBA window invalidate the saved AABA witness?',
  },
  'maximum-product-subarray-state': {
    operation: 'negative-extrema-flip', previous: [3, -6], multiplier: -4,
    review: 'Candidate products hide the sign-flipped ownership. Route both previous extrema through the same negative multiplier so the old minimum becomes the new maximum.',
    prediction: 'Which twenty-four-point candidate disappears if only the previous maximum is retained?',
  },
  'longest-substring-without-repeating-characters-state': {
    operation: 'nondecreasing-window-boundary', string: 'abba', previousLeft: 2, previousOccurrence: 0,
    review: 'Timestamp dots hide the reopened duplicate. Draw the rejected left-one window containing both b copies beside the retained left-two window.',
    prediction: 'Which repeated character reenters if the final a moves the left boundary backward to one?',
  },
  'jump-game-state': {
    operation: 'reachable-jump-frontier', values: [1, 0, 9],
    review: 'Yes/no tokens hide the unreachable donation. Enclose only the reachable prefix and leave the large nine outside the frontier it cannot extend.',
    prediction: 'Can the value nine contribute a jump before its own position becomes reachable?',
  },
  'longest-consecutive-sequence-state': {
    operation: 'run-start-membership', values: [1, 2, 3, 8],
    review: 'Predecessor flags hide the overlapping run. Draw one scan through 1,2,3 and a separate start at eight, leaving two and three present but without independent start marks.',
    prediction: 'Does rejecting two as a new run start remove two from the existing one-to-three run?',
  },
  'daily-temperatures-state': {
    operation: 'temperature-versus-index', temperatures: [70, 68, 69, 80],
    review: 'Subtraction tokens conflate temperature and time. Put temperatures on the vertical axis and daily positions on the horizontal axis so a ten-degree rise and three-day wait occupy different dimensions.',
    prediction: 'If the warmer day were ninety degrees at the same index three, would the waiting time change?',
  },
  'count-number-of-nice-subarrays-state': {
    operation: 'nested-start-membership', values: [2, 2, 2, 1, 1, 1], exact: 3,
    review: 'Difference bars hide which starts cancel. Keep all six start identities at the fixed final endpoint and remove exactly the two starts already counted by the lower at-most bound.',
    prediction: 'Which starts remain after subtracting the at-most-two group from the at-most-three group?',
  },
  'minimum-window-substring-state': {
    operation: 'character-copy-counts', words: ['AABC', 'ABC'], mode: 'minimum',
    review: 'A count table does not make the missing copy tangible. Give every required character its own mark, then leave the second required A slot unfilled in ABC.',
    prediction: 'Can one observed A satisfy both required A copies merely because all character types appear?',
    footer: 'ABC misses one required A despite containing all three character types.',
  },
  'permutation-in-string-state': {
    operation: 'character-copy-counts', words: ['aab', 'abb'], mode: 'exact',
    review: 'Membership is not multiplicity. Align physical copies in the same character bins so the missing a and extra b remain visible.',
    prediction: 'Does sharing the set of letters a and b make abb a permutation of aab?',
    footer: 'The missing a and extra b defeat exact count equality.',
  },
  'valid-anagram-state': {
    operation: 'character-copy-counts', words: ['aab', 'abb'], mode: 'exact',
    review: 'Equal lengths hide unequal inventories. Preserve the three copies in each word but place them by character so the inventories do not align.',
    prediction: 'Which character count must change to turn abb into an anagram of aab?',
    footer: 'Both lengths are three; their per-character inventories differ.',
  },
  'group-anagrams-state': {
    operation: 'character-copy-counts', words: ['ab', 'ba', 'aa'], mode: 'groups', expectedGroups: 2,
    review: 'Signature strings can feel arbitrary. Put copies from all three words into shared a/b bins so ab and ba coincide while aa occupies a different inventory.',
    prediction: 'Would changing the order of letters within aa move it into the ab/ba group?',
    footer: 'ab and ba share one inventory; aa supplies a second group.',
  },
  'contains-duplicate-state': {
    operation: 'set-entry-collapse', values: [1, 2, 1, 3],
    review: 'Unequal length tokens hide the equality merge. Route both separate input-one positions into the same single set entry.',
    prediction: 'Which two input positions must collapse onto one set entry?',
  },
  'coin-change-state': {
    operation: 'coin-predecessor', amount: 3, coins: [2, 3], costs: [0, 4, 1, 1],
    review: 'Candidate arithmetic hides the predecessor boundary. Route amount three to amount one and amount zero, preserving the unreachable sentinel on the first route.',
    prediction: 'Does an arithmetic candidate through unreachable amount one become a valid solution merely by adding one coin?',
  },
  'combination-sum-iv-state': {
    operation: 'last-step-paths', target: 4, expected: [3, 2],
    review: 'Count bars conceal ordered-sequence membership. Enumerate the five sequences and distinguish their last values so the disjoint predecessor groups are explicit.',
    prediction: 'Do 1,1,2 and 2,1,1 belong to the same final-value group?',
  },
  'decode-ways-state': {
    operation: 'decimal-pair-gate', pairs: ['10', '01', '27'],
    review: 'Validity cells hide digit order. Preserve the two-character strings at the gate so zero in the second position differs from an illegal leading zero.',
    prediction: 'Why is ten accepted while zero-one is not, even though both contain a zero?',
  },
  'edit-distance-state': {
    operation: 'edit-diagonal-cost', pairs: [['a', 'b'], ['a', 'a']],
    review: 'A diagonal token is not the operation. Pass the empty-prefix cost through a mismatch toll of one and a matching toll of zero.',
    prediction: 'If the two characters match, which edit is paid by inheriting the diagonal?',
  },
  'house-robber-ii-state': {
    operation: 'circular-endpoint-cases', values: [2, 3, 2],
    review: 'A two-case matrix omits the circular edge that forbids taking both endpoints. Draw that edge, then the two linear cases with opposite endpoints excluded.',
    prediction: 'Why must the two case answers be maximized rather than added?',
    caption: 'These circular houses have values 2,3,2. Excluding either endpoint gives a linear problem. Each case scores three; adding the case answers would select the same middle house twice.',
  },
  'longest-increasing-path-in-a-matrix-state': {
    operation: 'shared-increasing-suffix', grid: [[1, 3, 4], [0, 2, 0]], starts: [[0, 0], [1, 1]], suffix: [[0, 1], [0, 2]],
    review: 'Repeated suffix text can suggest copied rather than shared state. Draw both valid adjacent entry routes into the same physical three-to-four suffix.',
    prediction: 'Does the memoized suffix length at three depend on whether the entry route started at one or two?',
    caption: 'In this illustrative grid, starts one and two both reach the same adjacent suffix 3,4. Its length two is cached at cell three and reused.',
  },
  'palindromic-substrings-state': {
    operation: 'palindrome-occurrence-spans', string: 'aaa',
    review: 'Length totals hide location identity. Draw all six start-end intervals on the same three positions, retaining separate occurrences of equal text.',
    prediction: 'Why do the three one-character intervals count separately even though each contains a?',
  },
  'partition-equal-subset-sum-state': {
    operation: 'old-snapshot-extension', old: [0], item: 2,
    review: 'The reuse-bug table does not show its feedback edge. Extend only old zero with one current two and refuse to feed the newly created two back through that same item.',
    prediction: 'Which illegal same-item feedback would create four while processing just one two?',
  },
  'permutations-state': {
    operation: 'unused-position-availability', values: [1, 2, 3], used: [0, 2],
    review: 'A remaining-index token hides why backward source order is legal. Preserve all three source positions and leave middle position one available after choosing zero and two.',
    prediction: 'Is unused position one blocked just because its source index precedes the last chosen index two?',
  },
  'subsets-state': {
    operation: 'binary-choice-leaves', values: [1, 2],
    review: 'A decision matrix can hide excluded branches. Enumerate every binary decision path and connect both include and exclude outcomes to their actual subset memberships.',
    prediction: 'Which two decisions produce the empty subset rather than skipping the decision process?',
  },
  'unique-paths-state': {
    operation: 'grid-final-move', rows: 3, columns: 3,
    review: 'Two count bars omit the actual paths. Draw all six monotone grid routes and distinguish the last incoming edge so the upper and left predecessor groups cannot overlap.',
    prediction: 'If the destination\'s upper neighbor is blocked, which final-move group disappears?',
  },
  'word-break-state': {
    operation: 'word-boundary-alternatives', string: 'cars', words: ['car', 'ca', 'rs'], splits: [3, 2],
    review: 'A validity table obscures the competing boundaries. Mark the greedy cut after car beside the valid cut after ca, retaining the same four character positions.',
    prediction: 'Which suffix becomes reachable when the first cut moves from position three to position two?',
  },
  'same-tree-state': {
    operation: 'tree-shape-comparison', trees: [{ '': 1, L: 2, R: null }, { '': 1, L: null, R: 2 }],
    review: 'A child-position table hides the object shape. Draw the shared values in opposite child positions with explicit missing children.',
    prediction: 'Can the same values-only preorder distinguish these left-child and right-child trees?',
  },
  'serialize-and-deserialize-binary-tree-state': {
    operation: 'tree-shape-comparison', serialized: true, trees: [{ '': 1, L: 2, R: null }, { '': 1, L: null, R: 2 }],
    review: 'Encoding rows omit what the null tokens preserve. Draw both shapes and their different null-marked preorder strings together.',
    prediction: 'Which encoding token tells the decoder that the second tree has no left child?',
  },
  'subtree-of-another-tree-state': {
    operation: 'tree-shape-comparison', subtree: true, trees: [{ '': 5, L: 3, R: null }, { '': 5, L: 3, R: 7 }],
    review: 'Equal root cells can overshadow the extra child. Keep the matching root and left branch identical while exposing the candidate-only right node.',
    prediction: 'Does the extra right child disappear merely because both candidate roots equal five?',
  },
  'invert-binary-tree-state': {
    operation: 'tree-shape-comparison', invert: true,
    trees: [{ '': 1, L: 2, LL: 4, R: 3, RR: 5 }, { '': 1, L: 3, LL: 5, R: 2, RR: 4 }],
    review: 'A three-value swap cannot distinguish moving subtrees from changing stored values. Add descendants and mirror every child link while preserving each value.',
    prediction: 'Would swapping only stored values two and three move their descendants to the correct mirrored positions?',
    caption: 'Each node swaps its child links. Root one keeps its value, while nodes two and four move together. Descendant links also mirror; this is not merely a value swap.',
  },
  'validate-binary-search-tree-state': {
    operation: 'inherited-ancestor-bound', tree: { '': 10, L: 5, LR: 12 }, lower: 5, upper: 10, value: 12,
    review: 'A local/global comparison table hides the inherited region. Enclose the ancestor-left subtree and put twelve outside its allowed five-to-ten interval.',
    prediction: 'Why does satisfying parent five still fail the upper bound inherited from ancestor ten?',
  },
  'lowest-common-ancestor-in-a-bst-state': {
    operation: 'bst-stop-boundary', tree: { '': 6, L: 2, R: 8 }, targets: [[2, 8], [6, 8]],
    review: 'Direction words do not expose the search paths. Draw a split at six beside the equality case, where one target already occupies that ancestor.',
    prediction: 'If one target is the current node itself, must both targets lie in different child subtrees?',
  },
  'kth-smallest-element-in-a-bst-state': {
    operation: 'inorder-versus-push', tree: { '': 2, L: 1, R: 3 }, stack: [2, 1], rank: 1,
    review: 'Push and visit rows are too easy to conflate. Preserve the actual tree between the stack insertion order and the sorted inorder emission order.',
    prediction: 'For rank one, is the root pushed first automatically the first value visited?',
  },
  'linked-list-cycle-state': {
    operation: 'pointer-identity-meeting', value: 7,
    review: 'A table can depict two copies of the same supposed identity. Draw two equal-valued objects for the false case, but one shared object receiving both pointers for the true case.',
    prediction: 'Do matching stored sevens prove that slow and fast reference the same object?',
  },
  'remove-nth-node-from-end-state': {
    operation: 'linked-chain-frames', title: 'The dummy link can skip the head just like any other target',
    rows: [{ label: 'Before: D is the dummy predecessor; n = 3', chain: ['D', 1, 2, 3, 'null'] }, { label: 'After D.next skips one; return head two', chain: ['D', 2, 3, 'null'] }],
    review: 'Before/after tokens hide the actual pointer splice. Include the dummy and terminated chain so deleting the head changes the same predecessor-next link as an interior deletion.',
    prediction: 'Which next link changes when the removed node is the original head?',
  },
  'reverse-linked-list-state': {
    operation: 'linked-chain-frames', title: 'Save the suffix reference before redirecting the current link',
    rows: [{ label: 'Original chain; save next = two', chain: [1, 2, 3, 'null'] }, { label: 'Redirect one toward the processed prefix', chain: [1, 'null'] }, { label: 'The saved two still reaches the unprocessed suffix', chain: [2, 3, 'null'] }],
    review: 'A saved-next token does not show what it protects. Draw the processed and remaining chains separately after the old one-to-two link is overwritten.',
    prediction: 'Without saving two first, which reference would still reach the unprocessed suffix after one points to null?',
  },
  'reorder-list-state': {
    operation: 'linked-chain-frames', title: 'Detach and reverse the second half before weaving',
    rows: [{ label: 'Original list', chain: [1, 2, 3, 4, 5, 'null'] }, { label: 'First half stops after three', chain: [1, 2, 3, 'null'] }, { label: 'Detached second half reverses to five, four', chain: [5, 4, 'null'] }, { label: 'Weave without reconnecting the old middle edge', chain: [1, 5, 2, 4, 3, 'null'] }],
    review: 'Sequence rows omit the termination boundary. Show both detached null-terminated halves before the final alternating links so the old middle edge cannot silently form a cycle.',
    prediction: 'What old link must be removed before alternating the two halves?',
  },
  'merge-two-sorted-lists-state': {
    operation: 'linked-chain-frames', title: 'One splice attaches the already-sorted surviving chain',
    rows: [{ label: 'One input is exhausted', chain: ['null'] }, { label: 'The other input retains its existing suffix links', chain: [4, 6, 9, 'null'] }, { label: 'Attach that same surviving head to the output tail', chain: ['tail', 4, 6, 9, 'null'] }],
    review: 'Remainder text hides why no further comparisons are needed. Draw the untouched suffix as an already-linked chain and change only the output-tail connection.',
    prediction: 'Must the links four-to-six and six-to-nine be rebuilt after the other list is exhausted?',
  },
  'min-stack-state': {
    operation: 'minimum-history-pop', values: [5, 2, 2],
    review: 'Duplicated minima need duplicated owned snapshots, not distinct-value storage. Align each value with its minimum history and remove only the top pair.',
    prediction: 'After popping one two, which surviving snapshot still establishes minimum two?',
  },
  'largest-rectangle-in-histogram-state': {
    operation: 'histogram-sentinel-retained', heights: [2, 1, 2],
    review: 'No-unresolved-height text can suggest an empty stack. Show the final zero closing all positive spans and remaining as one real stack entry.',
    prediction: 'Does resolving every positive-height rectangle mean the actual stack has no entries?',
  },
  'lru-cache-state': {
    operation: 'recency-frontier', states: [[1, 2], [2, 1], [1, 3]],
    review: 'Ordered key rows omit the read-induced age change. Put the same keys on an oldest-to-newest rail, then show two outside the capacity wall after reading one.',
    prediction: 'Without get(1), which key would the insertion of three evict instead?',
  },
  'alien-dictionary-state': {
    operation: 'first-unequal-character', words: ['wrt', 'wrf'],
    review: 'Selected table cells do not distinguish shared evidence from the decisive mismatch. Keep equal prefix columns inert and route only t and f into the ordering edge.',
    prediction: 'Would a later mismatch supply a second edge after the first unequal character already decides this word pair?',
  },
  'balanced-binary-tree-state': {
    operation: 'child-return-join', mode: 'failure', children: ['FAIL', 2], expected: 'FAIL',
    review: 'FAIL beside a numeric height can look like another comparable value. Pass both returned packets into a failure gate that preserves FAIL instead of computing a parent height.',
    prediction: 'Can an arbitrarily large valid height from the other child repair the failed child?',
  },
  'maximum-depth-of-binary-tree-state': {
    operation: 'child-return-join', mode: 'depth', children: [2, 4], expected: 5,
    review: 'Arithmetic tokens hide the mutually exclusive paths. Merge child depth packets with max, then add this one parent rather than adding both child paths.',
    prediction: 'Why does adding two and four count nodes from two different root-to-leaf routes?',
  },
  'binary-tree-level-order-traversal-state': {
    operation: 'frozen-queue-level',
    review: 'Queue rows hide which generation owns new entries. Retain the two-node level budget while coloring appended children as the next generation.',
    prediction: 'After A appends two children, does the current level budget become the new queue length three?',
    caption: 'A and B start this illustrative level. A appends A1 and A2; B is a leaf. Process exactly A and B, then leave the new children for the next level.',
  },
  'binary-tree-maximum-path-sum-state': {
    operation: 'branching-return', root: 10, gains: [4, 6],
    review: 'Two sums do not expose their different path contracts. Draw a complete path using both children separately from the one-child route that can continue through the parent.',
    prediction: 'Can the caller extend the twenty-point two-child path through another parent without revisiting this node?',
  },
  'clone-graph-state': {
    operation: 'clone-identity-boundary', labels: ['A', 'B'],
    review: 'A copy-map table hides the graph boundary. Keep original and copied cycles in separate enclosures and make the back edge reuse the already allocated A copy.',
    prediction: 'If B-prime links to original A instead, which enclosure does that supposed clone leave?',
  },
  'construct-tree-from-preorder-and-inorder-traversal-state': {
    operation: 'root-membership-split', preorder: [3, 9, 20, 15, 7], inorder: [9, 3, 15, 20, 7],
    review: 'Token partitions omit the cross-traversal identity match. Connect preorder root three to its inorder position and enclose the actual left and right memberships.',
    prediction: 'Can preorder position alone determine how many of these nodes belong to the left subtree?',
  },
  'course-schedule-state': {
    operation: 'empty-ready-counterexample',
    review: 'Two empty-queue cells hide the unprocessed cycle. Show a completed chain beside an emitted isolated A with B and C still blocking each other.',
    prediction: 'When the ready queue is empty but B and C still wait on each other, have all three courses completed?',
    caption: 'An empty ready queue occurs after both a completed three-course chain and a blocked B-C cycle. Success additionally requires emitting every course.',
  },
  'graph-valid-tree-state': {
    operation: 'component-membership', labels: ['A', 'B', 'C', 'D'], edges: [[0, 1], [1, 2], [2, 0]], expected: 2,
    review: 'A triangle and orphan need their joint accounting exposed. Process the closing third edge without merging D so n-1 edges visibly coexist with a cycle and two components.',
    prediction: 'Does the third edge make isolated D reachable from the triangle?',
    footer: 'Three edges for four nodes still leave a cycle and isolated D.',
  },
  'number-of-connected-components-state': {
    operation: 'component-membership', labels: [0, 1, 2, 3], edges: [[0, 1], [2, 3], [1, 0]], expected: 2,
    review: 'Count tokens hide the unchanged groups. Keep both component enclosures fixed when the repeated 1-0 union names two nodes already together.',
    prediction: 'Which actual group would disappear if the duplicate union incorrectly decremented two to one?',
    footer: 'The duplicate 1-0 union leaves both existing groups intact.',
  },
  'redundant-connection-state': {
    operation: 'component-membership', labels: [1, 2, 3], edges: [[0, 1], [1, 2], [0, 2]], expected: 1,
    review: 'Same-root words do not show the closing route. Preserve the accepted chain and add the candidate 1-3 edge as a cycle, not another component merge.',
    prediction: 'Can joining endpoints already in the same enclosure reduce the component count again?',
    footer: 'Edge 1-3 closes the existing route; it does not merge two groups.',
  },
  '3sum-state': {
    operation: 'monotone-pair-bound', fixed: -4, left: -2, partners: [1, 2], target: 0,
    review: 'A rejection table hides why one comparison eliminates every smaller partner. Put all attainable sums below the target on one ruler and mark the largest failing partner.',
    prediction: 'If the largest available partner still gives a negative sum, can a smaller partner make this fixed pair reach zero?',
  },
  'binary-search-state': {
    operation: 'candidate-cut', values: [1, 3, 5, 7, 9], target: 7,
    stages: [{ label: 'Sorted candidates; target seven', keep: [0, 1, 2, 3, 4] }, { label: 'Middle five is too small: reject its entire prefix', keep: [3, 4] }],
    review: 'Out tokens hide a contiguous eliminated region. Keep every input position fixed and remove the whole prefix justified by the middle comparison.',
    prediction: 'Why can the value three be rejected without comparing it directly with seven?',
  },
  'find-minimum-in-rotated-sorted-array-state': {
    operation: 'candidate-cut', values: [3, 4, 1, 2], minimum: true,
    stages: [{ label: 'Rotated candidates; rightmost value two', keep: [0, 1, 2, 3] }, { label: 'Middle four exceeds right two: discard its prefix', keep: [2, 3] }, { label: 'Middle one is below right two: keep that middle', keep: [2] }],
    review: 'Ledger rows omit the asymmetric inclusive bound. Preserve the middle-one identity through the second cut instead of discarding both compared positions.',
    prediction: 'What happens to the answer if the low-side comparison uses right = middle - 1?',
  },
  'search-in-rotated-sorted-array-state': {
    operation: 'candidate-cut', values: [4, 5, 6, 7, 0, 1, 2], target: 0,
    stages: [{ label: 'Middle seven ends the sorted left half', keep: [0, 1, 2, 3, 4, 5, 6] }, { label: 'Zero is outside four through seven: keep the other half', keep: [4, 5, 6] }],
    review: 'A shorter output row loses original positions. Shade the ineligible sorted half in place so its range test, not global sorting, explains the cut.',
    prediction: 'Would target five justify discarding this same left half?',
  },
  'container-with-most-water-state': {
    operation: 'shorter-wall-bound', lower: 2, widths: [3, 2],
    review: 'Arithmetic tokens omit the physical upper bound. Keep the short wall fixed while narrowing the floor; even an arbitrarily tall replacement cannot raise the water above two.',
    prediction: 'Can keeping the short wall and reducing width ever beat the current area six?',
  },
  'koko-eating-bananas-state': {
    operation: 'ceiling-hours', piles: [3, 6, 7, 11], speeds: [3, 4], hours: 8, expected: [10, 8],
    review: 'Total-hour bars hide ceiling division. Partition each pile into full-hour boxes, retaining a partially filled final box as one whole hour.',
    prediction: 'At speed four, does the last three-banana remainder of pile eleven cost zero hours or one?',
  },
  'split-array-largest-sum-state': {
    operation: 'greedy-capacity', values: [7, 2, 5, 10, 8], capacities: [17, 18], allowed: 2, expected: [3, 2],
    review: 'Group-count bars do not show the capacity predicate. Pack the same ordered items into bounded rows so ten and eight visibly fit together only at eighteen.',
    prediction: 'Which cut disappears when capacity rises from seventeen to eighteen?',
  },
  'kth-largest-element-state': {
    operation: 'retained-min-heap', entries: [[7, 7], [2, 2], [9, 9], [4, 4]], count: 3, rootAnswer: true,
    review: 'A sorted list disguises the heap boundary. Draw the actual retained min-heap separately from the discarded global minimum; its root is the weakest surviving top-three value.',
    prediction: 'Why is the answer four rather than the smallest observed value two?',
  },
  'top-k-frequent-elements-state': {
    operation: 'frequency-buckets', entries: [[9, 3], [1, 5], [8, 4]], count: 2,
    review: 'Value-frequency tokens hide the bucket coordinate and scan. Place each distinct value once in its frequency bucket, then stop descending after the two higher buckets supply the answer.',
    prediction: 'If nine becomes ninety without changing its frequency, does it enter the retained top two?',
  },
  'longest-increasing-subsequence-state': {
    operation: 'tails-versus-witness', values: [3, 5, 7, 2],
    review: 'Tails rows resemble a saved sequence. Preserve source indices under the updated tails and draw the actual length-three witness on the original timeline.',
    prediction: 'Why can tail two at input index three not precede tail five at input index one in an actual subsequence?',
  },
  'find-median-from-data-stream-state': {
    operation: 'heap-order-gate', cases: [[5, 1], [1, 5]],
    review: 'Yes/no cells obscure the second heap invariant. Place both heap extrema on one number line while keeping their equal sizes explicit; the reversed case crosses the lower/upper boundary.',
    prediction: 'Can equal heap sizes alone guarantee that every lower-half value precedes every upper-half value?',
  },
  'top-k-scores-state': {
    operation: 'subset-rank-order', scores: [.2, .9, .4, .8], selected: [3, 1],
    review: 'Index rows hide the second operation. Keep the correct selected identities while exchanging their positions to produce descending score order.',
    prediction: 'Does selecting the right two indices guarantee that the first returned score is the larger one?',
  },
  'climbing-stairs-state': {
    operation: 'last-step-paths', target: 5,
    review: 'A divided bar states the recurrence but hides its bijection. Display every path and color its final move so removing that move exposes the smaller subproblem.',
    prediction: 'After removing the final two-step move, how many stairs remain in every path of that group?',
  },
  'course-schedule-ii-state': {
    operation: 'prerequisite-counter',
    review: 'Unmet-count tokens omit the edges that supply the decrements. Preserve both parent identities and release course three only after both have emitted.',
    prediction: 'If prerequisite two never emits, can emitting prerequisite one twice make course three ready?',
  },
  'house-robber-state': {
    operation: 'nonadjacent-choice', values: [2, 7, 9, 3],
    review: 'Profit bars hide which prefix taking the last house may reuse. Draw the actual selected houses in both branches so the adjacent exclusion remains visible.',
    prediction: 'Why would adding the last three to the saved eleven select adjacent houses?',
  },
  'longest-common-subsequence-state': {
    operation: 'conditional-diagonal',
    review: 'A highlighted cell alone does not identify its predecessor. Connect the matching B cell to the shorter-prefix diagonal it extends.',
    prediction: 'If the last characters differ, is the diagonal-plus-one transition still permitted?',
  },
  'combination-sum-state': {
    operation: 'ordered-choice-path', paths: [[0, 0, 1], [1, 0, 0]], strict: false,
    review: 'Index tokens hide the traversal constraint. Trace one valid reused choice and the backward path that would emit the same combination a second time.',
    prediction: 'Why is repeating index zero allowed while returning from index one to zero is not?',
  },
  'bayes-positive-test-denominator': {
    operation: 'condition-population',
    review: 'The result-only cohorts hide the base rate. Preserve cohort identity through unequal selection and normalization.',
    prediction: 'If disease prevalence rises, which input cohort changes before the test acts?',
    population: [100, 99900], rates: [.99, .01], peoplePerMark: 100,
  },
  'ab-test-mde-sample-size': {
    operation: 'square-spread',
    review: 'Sample-budget bars do not show why the factor is squared. Turn doubled noise width into four equal variance tiles.',
    prediction: 'Tripling noise requires how many baseline sample budgets under the same assumptions?',
    noise: [1, 2],
  },
  'attention-keys-choose-values-contribute': {
    operation: 'weighted-values',
    review: 'Endpoint bars omit the multiplication. Let weight set width and scalar value set height so contribution is area.',
    prediction: 'What becomes of the second contribution if its value is zero but its weight stays unchanged?',
    weights: [.8, .2], values: [1, 10], unit: .2, normalized: true,
  },
  'class-imbalance-sampling-prior': {
    operation: 'biased-sample', review: 'A deployment-only population omits the sampling intervention. Pull a balanced batch out of the unchanged imbalanced world.',
    prediction: 'Does drawing equal class counts change the deployment base rate?', counts: [10, 990], batch: [10, 10],
  },
  'confusion-matrix-denominators': {
    operation: 'select-denominator', review: 'A table hides membership. Select two different subsets of the same cases while preserving the eight shared hits.',
    prediction: 'Which extra cases enter precision but not recall?',
  },
  'calibration-reliability-gap': {
    operation: 'forecast-count', review: 'Observed outcomes alone omit the forecast. Compare the expected aggregate count with the actual aggregate count on one scale.',
    prediction: 'If 80 cases were positive, would this bin still show the same calibration gap?', expected: 80, observed: 60, total: 100,
  },
  'contextual-bandit-partial-feedback-support': {
    operation: 'logged-support', review: 'A missing value needs a missing path, not another ledger entry. Give the zero-probability action no route to an observed reward.',
    prediction: 'Can multiplying observed A and B records reveal C rewards?', actions: ['A', 'B', 'C'], probabilities: [.6, .4, 0],
  },
  'continuous-batching-reuses-finished-slot': {
    operation: 'replace-slot', review: 'Ledger rows obscure persistence. Keep A in the same slot while B exits and C enters the released slot.',
    prediction: 'Must A finish before C can occupy B’s released slot?',
  },
  'fixed-budget-ranking-funnel': {
    operation: 'nested-cohorts', review: 'Three bars do not show survivors. Reuse candidate positions through two filters, with one tile representing 100 candidates.',
    prediction: 'Can final scoring recover a candidate removed upstream?', counts: [10000, 1000, 100], unit: 100, labels: ['Retrieved', 'Shortlist', 'Scored'], ending: 'Only survivors receive expensive scoring.',
  },
  'agent-safety-authority-loop': {
    operation: 'bounded-capacity', review: 'An overflow result omits the proposal being inserted. Preserve the committed prefix and expose the ten-unit excess at the wall.',
    prediction: 'Does refusing the proposal reverse the committed effects?', mode: 'accumulate', committed: 60, proposed: 50, capacity: 100,
  },
  'disagreement-evidence-commit-loop': {
    operation: 'bounded-capacity', review: 'Memory is a physical admission boundary. Align both cache choices with the same eight-slot wall and leave the ninth unit outside.',
    prediction: 'Can a speed advantage make the ninth memory unit fit?', mode: 'choose', labels: ['Cache A', 'Cache B'], required: [9, 6], available: [8, 8], unit: 'GB',
  },
  'reasoning-admission-and-capacity': {
    operation: 'bounded-capacity', review: 'Yes/no tokens hide independent resources. Make compute and KV separate banks whose gates must both pass.',
    prediction: 'Can two free compute units supply the missing KV page?', mode: 'all', labels: ['Compute', 'KV pages'], required: [4, 6], available: [6, 5], unit: 'units',
  },
  'ecosystem-serving-stages': {
    operation: 'nested-cohorts', review: 'Flat groups hide nested selection. Retain the same 100 candidate identities through eligibility and exposure filters.',
    prediction: 'What reward is observed for an eligible candidate that was never shown?', counts: [100, 20, 5], unit: 1, labels: ['Retrieved', 'Eligible', 'Shown'], ending: 'Unshown is not a negative outcome.',
  },
  'backprop-forward-reverse-trace': {
    operation: 'arithmetic-paths', review: 'A final pair of gradients hides the cached multipliers. Fork the upstream signal through the other forward input.',
    prediction: 'If x doubles while the upstream signal stays fixed, which parameter gradient doubles?',
    paths: [{ label: 'Forward: a times x', inputs: [3, 2], operator: 'multiply', result: 6 }, { label: 'To a: upstream times cached x', inputs: [4, 2], operator: 'multiply', result: 8 }, { label: 'To x: upstream times cached a', inputs: [4, 3], operator: 'multiply', result: 12 }],
  },
  'logistic-gradient-cancellation': {
    operation: 'arithmetic-paths', review: 'Cancellation should leave a visible surviving route. Trace a concrete positive-label gradient through the sigmoid and then the input.',
    prediction: 'If the input is zero, can the weight receive this nonzero logit gradient?',
    paths: [{ label: 'Positive label: p = 0.2', inputs: [-5, .16], operator: 'multiply', result: -.8 }, { label: 'The weight still receives x = 3', inputs: [-.8, 3], operator: 'multiply', result: -2.4 }],
    caption: 'For a positive label, the two logit-gradient factors multiply to -0.8. The weight still needs the input factor. At x=3, its gradient is -2.4.',
  },
  'ece-weighted-bin-contributions': {
    operation: 'weighted-values', review: 'Equal results hide different causes. Make bin share and gap orthogonal dimensions of equal contribution areas.',
    prediction: 'What happens if only the first bin doubles in size?', weights: [.1, .4], values: [.4, .1], unit: .01, widthUnit: .1, heightUnit: .1,
    widthLabel: 'bin share', heightLabel: 'gap', totalLabel: 'Shown contributions',
  },
  'gradient-magnitude-through-depth': {
    operation: 'gain-cascade', review: 'Power curves conceal local repetition. Pass the same unit signal through ten identical gates and let its width compound.',
    prediction: 'Is depth alone sufficient to say whether this signal grows?', gains: [.8, 1.2], count: 10,
  },
  'loss-estimation-vs-decision': {
    operation: 'weighted-values', review: 'A probability is not an action. Show error probability multiplying error cost and choose between the resulting areas.',
    prediction: 'Would the same score choose the same action if false-positive cost fell to one?',
    weights: [.2, .8], values: [10, 1], unit: .2, normalized: true, combine: false, widthLabel: 'error chance', heightLabel: 'error cost', verdict: 'Negative action costs less',
  },
  'lstm-coordinate-gated-state-update': {
    operation: 'weighted-values', review: 'The two gates are independent valves, not a normalized mixture. Preserve the old-state and candidate contributions until they merge.',
    prediction: 'If the input gate closes, which contribution remains?', weights: [.8, .5], values: [2, 3], expected: [1.6, 1.5], unit: .1, widthLabel: 'gate', heightLabel: 'state',
  },
  'gradient-boosting-residual-update': {
    operation: 'residual-step', review: 'A ledger obscures distance. Put prediction and target on one ruler, then move only one tenth of the residual.',
    prediction: 'Does the new tree add the full target or a fraction of what remains?', predictionValue: 7, targetValue: 10, shrinkage: .1,
  },
  'logit-odds-probability-map': {
    operation: 'linked-scales', review: 'Three number rows hide spacing. Connect the same candidates through equally spaced logits, multiplicative odds, and compressed probabilities.',
    prediction: 'Does another unit logit step add the same probability near the upper end?', logits: [0, 1, 2],
  },
  'adversarial-threat-set-boundary': {
    operation: 'geometric-action', mode: 'threat', review: 'A line between points hides the allowance. Draw both decision regions and the permitted horizontal interval around the original point.',
    prediction: 'Can an allowance of 0.2 cross the boundary from x=0.3?', allowance: .5,
  },
  'estimator-error-reference-points': {
    operation: 'geometric-action', mode: 'spread', review: 'One estimate does not illustrate repeated sampling. Add an explicit four-outcome experiment with separate truth, mean, and spread.',
    prediction: 'If all four estimates move together, does their spread around their own mean change?', outcomes: [[2, -1], [2, 1], [4, -1], [4, 1]],
    caption: 'This illustrative estimator has four equally likely outcomes. Its mean is (3,0), one unit from truth (2,0). Squared spread around its mean is 2. Mean squared error is 3.',
  },
  'determinant-signed-area-collapse': {
    operation: 'geometric-action', mode: 'orientation', review: 'A filled polygon cannot explain sign. Trace both column orders around the same parallelogram.',
    prediction: 'Does reversing the column order change the floor area?',
  },
  'spectral-theorem-change-basis': {
    operation: 'geometric-action', mode: 'stretch', review: 'A token multiplication does not show independent axes. Stretch the actual coordinate displacements by different eigenvalues.',
    prediction: 'Does scaling the first eigen-coordinate require mixing it with the second?', input: [3, 1], gains: [2, 3],
  },
  'embedding-normalization-metric-equivalence': {
    operation: 'geometric-action', mode: 'normalize', review: 'Ranking bars hide angle and length. Pull candidates onto a unit circle and compare their query-axis projections.',
    prediction: 'After normalization, can increasing only A’s original length improve its cosine?', candidates: [[10, 10], [1, 0]],
  },
  'uncertainty-within-between-models': {
    operation: 'geometric-action', mode: 'mixture-spread', review: 'Mean markers hide each model’s outcomes. Separate within-model displacement from movement between model centers.',
    prediction: 'Can individually concentrated models form a wide mixture?', means: [1, 4], within: .4,
    caption: 'Each model has two equally likely outcomes. Both lie 0.4 from that model mean. Within-model variance is 0.16. Between-model variance is 2.25. Their sum is 2.41.',
  },
  'covariance-centered-product-signs': {
    operation: 'geometric-action', mode: 'centered-products', review: 'A sign table hides joint deviations. Put observations in the four centered quadrants and on the zero-contribution axis.',
    prediction: 'Which sign changes if only the centered x deviation changes sign?',
  },
  'global-norm-clipping-radial-projection': {
    operation: 'geometric-action', mode: 'clip', review: 'Two vectors do not expose the constraint. Add the norm boundary and pull the original point inward along its existing ray.',
    prediction: 'Does clipping rotate this gradient toward an axis?', radius: 2,
  },
  'ols-orthogonal-projection': {
    operation: 'geometric-action', mode: 'projection', review: 'A right-angle symbol states optimality without demonstrating it. Add a feasible rail and a longer competing fit-to-observation tether.',
    prediction: 'Can another coefficient representation change the same projected prediction?', competitor: [1, 0],
  },
  'matrix-columns-define-map': {
    operation: 'geometric-action', mode: 'cancel', review: 'Origin vectors obscure head-to-tail cancellation. Send one nonzero contribution out and the other back to the origin.',
    prediction: 'Can a nonzero coefficient vector disappear under dependent columns?',
  },
  'point-in-time-availability-boundary': {
    operation: 'event-clock', mode: 'availability', review: 'Isolated timestamp dots hide the unavailable interval. Carry one record from event to availability across the prediction boundary.',
    prediction: 'Can a later-arriving record be used because its event happened earlier?',
  },
  'multimodal-stale-frame-gate': {
    operation: 'event-clock', mode: 'age', review: 'Age bars do not show identity. Carry the same stamped frame through transport and measure both ages from the same question.',
    caption: 'Capture age, not arrival delay, determines freshness.',
    prediction: 'Can fast delivery refresh an old capture?', ages: [4.3, .1],
  },
  'video-ecosystem-delayed-effects': {
    operation: 'event-clock', mode: 'delayed-effect', review: 'One event list hides two response speeds. Split viewer and creator lanes across the short-experiment stop boundary.',
    prediction: 'Can the early readout rule out an effect whose observation lies after the test?',
  },
  'decision-threshold-action-regions': {
    operation: 'policy-bands', review: 'One policy map can look universal. Keep one score fixed while changing the declared action regions.',
    prediction: 'Does a fixed score contain a policy-independent action?', score: .85, policies: [[.2, .8], [.4, .9]],
    caption: 'The same score can fall in different policy regions. Here, 0.85 means block under A and review under B. Thresholds require a declared policy.',
  },
  'fixed-capacity-portfolio-guardrails': {
    operation: 'budget-transfer', review: 'The transfer should be countable and the protected accounts visible. Reassign six identified budget units while retaining the other 94.',
    prediction: 'Which account gives up the six points, and which two accounts do not move?',
  },
  'adamw-decoupled-shrinkage': {
    operation: 'decay-bypass', review: 'Equal result bars omit the route. Put coupled decay through the preconditioner and let direct decay visibly bypass it.',
    prediction: 'Which route changes when only the frozen preconditioner changes?', scales: [1, 10], base: .001,
  },
  'regularization-diagnose-and-target': {
    operation: 'coefficient-restraint', review: 'Replace the diagnostic table with exact coefficient motion. Enlarge the weak coordinate so stopping at zero differs visibly from scaling.',
    prediction: 'Which L1 coefficient returns when the threshold falls below 0.1?',
  },
  'cold-start-signal-handoff': {
    operation: 'density-update', review: 'A final density hides evidence weighting. Keep the prior, show the three observations, and reshape the whole density rather than declaring a point estimate.',
    prediction: 'Do three observations make every probability except 0.6 impossible?', successes: 2, failures: 1,
  },
  'dropout-inverted-expectation': {
    operation: 'random-branch', review: 'Arithmetic does not distinguish a draw from its expectation. Branch one activation into two alternative realizations and reunite only their weighted average.',
    prediction: 'Does one pass return the mean of the two branches?', input: 4, keep: .5,
  },
  'gmm-soft-responsibility': {
    operation: 'split-observation', review: 'A stacked percentage hides observation identity. Divide one unit of observation mass between two updates without duplicating it.',
    prediction: 'If both branches received a full observation, how much total mass would be counted?', fractions: [.7, .3],
  },
  'mutual-information-noisy-channel-budget': {
    operation: 'binary-channel', review: 'An information budget hides conditional dependence. Route fair input bits mostly straight through, with thin crossover paths for flips.',
    prediction: 'If both outgoing paths became equally likely, would the input still predict the output?', flip: .1,
  },
  'clt-data-vs-sample-means': {
    operation: 'average-distribution', review: 'Two densities omit averaging itself. Gather ten skewed observations into one mean while retaining the original source density.',
    prediction: 'Does collecting means alter the distribution of an individual future observation?', draws: [.03, .14, .29, .45, .61, .89, 1.1, 1.42, 2.1, 2.97],
    caption: 'The original exponential population stays unchanged. Ten independent observations produce one mean. Here, its exact Gamma density is narrower. Its population variance is one tenth.',
  },
  'cnn-spatial-channel-pyramid': {
    operation: 'channel-pooling', review: 'Shape tokens obscure the reduction axes. Collapse each actual spatial map into its own output while preserving channel color and identity.',
    prediction: 'How many outputs remain when only the spatial axes are averaged?', channels: 3, side: 4,
  },
  'exponential-family-sufficient-stat-moment-match': {
    operation: 'sufficient-compression', review: 'A sum ledger hides what is discarded. Permute the observed bits, then compress both orders into the same count pair.',
    prediction: 'Can the Bernoulli likelihood distinguish these two orders under the stated independent model?', outcomes: [1, 0, 1, 1, 0], permutation: [1, 0, 3, 4, 2],
  },
  'fp16-bf16-bit-budget': {
    operation: 'bit-reassignment', review: 'One format alone hides the tradeoff. Reassign three of the same sixteen physical bit slots from fraction to exponent.',
    prediction: 'Can this fixed-size layout add exponent bits without taking them from another field?', formats: [[1, 5, 10], [1, 8, 7]],
  },
  'label-smoothing-finite-confidence-optimum': {
    operation: 'target-redistribution', review: 'A two-part bar hides where the uniform mass goes. Spread ten of one hundred target units across all five classes, including the true one.',
    prediction: 'Does the true class also receive its share of the uniform mass?', classes: 5, retained: 90, redistributed: 10,
  },
  'capacity-regimes-double-descent': {
    operation: 'interpolation-freedom', review: 'An error curve can become another memorized shape. Pin two continuous fits to the same observations and expose their unconstrained between-point disagreement.',
    prediction: 'Do the three observed pins determine the prediction halfway between them?',
    headline: 'Two perfect training fits disagree between observed points',
    caption: 'Both curves pass through the three observed points. One shared deformation parameter changes predictions between observations. Training fit alone cannot choose between them. The derivation shows a possible double-descent regime.',
    gains: [0, 2], observed: [0, 1, 2], probe: .5,
  },
  'mle-bernoulli-parameter-sweep': {
    operation: 'likelihood-candidates', review: 'A final peak hides what is held fixed. Pass the same five outcomes through three candidate probabilities and multiply their evidence factors.',
    prediction: 'Why do the two failures penalize the 0.9 candidate?', outcomes: [1, 0, 1, 1, 0], candidates: [.2, .6, .9],
  },
  'activation-gradient-regions': {
    operation: 'local-slope', review: 'Derivative curves do not show what a local slope measures. Draw forward responses and equal input nudges whose vertical changes differ.',
    prediction: 'Where does the same small input move produce almost no output change?',
  },
  'bellman-stochastic-backup': {
    operation: 'weighted-values', review: 'The best leaf is not the chance-node answer. Make both return contributions occupy countable probability-times-return areas.',
    prediction: 'Can the value become two while the lower-return branch still has positive probability?', weights: [.75, .25], values: [2, 1], expected: [1.5, .25], normalized: true, widthUnit: .25, heightUnit: .5, unit: .125,
    totalLabel: 'A chance node combines both possible outcomes.',
  },
  'belief-propagation-branch-summary': {
    operation: 'weighted-values', review: 'Show marginalization as gathering both weighted hidden-state contributions, rather than choosing a hidden state.',
    prediction: 'What contribution disappears if the second hidden state is wrongly discarded?', weights: [.3, .7], values: [2, .5], expected: [.6, .35], normalized: true, widthUnit: .1, heightUnit: .5, unit: .05,
    totalLabel: 'The message is a sum, before later normalization.',
  },
  'content-profile-scores-new-item': {
    operation: 'weighted-values', review: 'A dot-product ledger hides feature alignment. Give the two aligned coordinate products their own equal-scale areas.',
    prediction: 'Can swapping the item coordinates change the match score?', weights: [2, 1], values: [1, 3], expected: [2, 3], widthUnit: 1, heightUnit: 1, unit: 1, widthLabel: 'interest', heightLabel: 'item feature',
  },
  'ring-attention-fixed-query-kv-cycle': {
    operation: 'weighted-values', review: 'Equal local averages obscure unequal evidence. Weight each local answer by its share of the combined normalizer.',
    prediction: 'Why is the answer not the equal average six?', weights: [.2, .8], values: [2, 10], expected: [.4, 8], normalized: true, widthUnit: .2, heightUnit: 2, unit: .4,
    totalLabel: 'Normalizers 2 and 8 supply these unequal weights.',
  },
  'pca-rank-one-projection': {
    operation: 'geometric-action', mode: 'projection', review: 'Make the retained line a physical constraint. Drop the observation onto it and retain the discarded perpendicular tether.',
    prediction: 'Does the perpendicular displacement remain in the rank-one reconstruction?', competitor: [1, 0],
  },
  'relu-second-moment-through-depth': {
    operation: 'gain-cascade', review: 'Initialization is a repeated moment multiplier, not a promise of universal stability. Carry the second moment through ten layers.',
    prediction: 'Can repeatedly losing half the second moment preserve its initial scale?', gains: [.5, 1], count: 10,
  },
  'importance-weights-effective-sample-size': {
    operation: 'weighted-draws', review: 'Repeated weight is easily mistaken for repeated evidence. Keep exactly five observation identities while enlarging their influence areas.',
    prediction: 'Does enlarging the first draw create another independent observation?', weights: [6, 1, 1, 1, 1],
  },
  'position-bias-inverse-examination-weight': {
    operation: 'weighted-draws', review: 'Keep one observed click visibly singular while its inverse-examination contribution expands to five units.',
    prediction: 'Does a weight of five mean that five clicks were observed?', weights: [5], examination: .2,
  },
  'flow-local-volume-density': {
    operation: 'mass-expansion', review: 'Trace four persistent probability marks through a doubled domain. Expand space without multiplying observations or mass.',
    prediction: 'What happens to density if the same mass occupies twice the area?',
  },
  'policy-gradient-advantage-sign': {
    operation: 'baseline-displacement', review: 'A negative bar is a memorized sign. Measure the actual return backward from the baseline on the same ruler.',
    prediction: 'Can a positive raw return still discourage the sampled action?', observed: 3, baseline: 5,
  },
  'naive-bayes-evidence-ledger': {
    operation: 'odds-update', review: 'Preserve the comparison class while independent evidence doubles the other class weight twice.',
    prediction: 'What assumption permits the second likelihood ratio to multiply the first?', odds: [1, 2, 4], reference: 4,
  },
  'scaling-fixed-compute-allocation': {
    operation: 'rectangle-exchange', review: 'A hyperbola hides the conserved resource. Reshape the same eight compute tiles by exchanging parameter and token axes.',
    prediction: 'Can both dimensions double while these eight tiles stay fixed?', shapes: [[2, 4], [4, 2]], labels: ['parameters', 'tokens'],
  },
  'residual-add-gradient-split': {
    operation: 'copied-gradient', review: 'At addition, an adjoint is copied, not divided like probability mass. Follow full upstream sensitivity into both routes before their input contributions add.',
    prediction: 'Does the identity path receive only half of the upstream gradient?', upstream: 2.2, derivative: .1,
  },
  'potential-shaping-cancels-backtracking': {
    operation: 'potential-loop', review: 'Show the return trip undoing the same potential gain. A closed physical loop makes the cancellation memorable.',
    prediction: 'Can cycling between these two states accumulate undiscounted potential reward?', potentials: [2, 5, 2],
  },
  'gae-residual-weight-trace': {
    operation: 'backward-fold', review: 'The weighted sum hides its time direction. Pull the future accumulator backward, decay it, then add the current residual.',
    prediction: 'Which residual crosses two decay boundaries before reaching the first time step?', residuals: [1, 2, -1], decay: .8,
  },
  'bert-mlm-selection-and-corruption': {
    operation: 'target-corruption', review: 'Target membership and visible replacement are different sets. Keep token identities through selection and show unchanged targets inside the selected group.',
    prediction: 'Can an unchanged visible token still contribute a prediction loss?', total: 1000, counts: [120, 15, 15, 850], unit: 5,
  },
  'bootstrap-resampling-unit': {
    operation: 'cluster-resample', review: 'User identities must carry their paired events. Duplicate an entire selected user cluster, not unrelated event rows.',
    prediction: 'Which events travel together when user A is sampled twice?', original: ['A', 'B', 'C'], selected: ['A', 'C', 'A'],
  },
  'bptt-detach-cuts-gradient-not-state': {
    operation: 'gradient-boundary', review: 'A table cannot distinguish forward transport from reverse sensitivity. Let the hidden value cross a wall while the returning gradient stops.',
    prediction: 'Does stopping the gradient reset the hidden value to zero?', value: 2,
  },
  'ctc-blank-preserves-repeat': {
    operation: 'alignment-collapse', review: 'The blank is a separator before it is deleted. Track the two surviving a identities across both ordered transformations.',
    prediction: 'Would deleting the blank before merging leave the same number of a outputs?', path: ['a', 'a', '_', 'a'],
  },
  'im2col-patches-become-columns': {
    operation: 'patch-columns', review: 'The matrix alone hides overlap. Select two neighboring patches from one image and preserve each pixel identity in the resulting columns.',
    prediction: 'Why do the middle image pixels appear in both output columns?', image: [[1, 2, 3], [4, 5, 6]],
  },
  'gqa-eight-queries-two-kv-groups': {
    operation: 'shared-memory', review: 'A list can imply merged outputs. Fan eight distinct queries out from two shared KV stores and retain eight distinct result identities.',
    prediction: 'Does sharing two KV groups reduce eight query outputs to two?', groups: 2, perGroup: 4,
  },
  'multi-head-attention-split-recombine': {
    operation: 'feature-partition', review: 'Splitting boxes can accidentally split time. Cut the feature columns while keeping every token row in each head, then reassemble them.',
    prediction: 'Which head loses access to token two after this feature split?', tokens: 3, features: 4, heads: 2,
  },
  'kv-cache-one-write-growing-read': {
    operation: 'cache-frontier', review: 'Different-height bars hide cache identity. Preserve seven old pairs, append the eighth, and visibly read all eight.',
    prediction: 'How many old pairs are read when only one new pair is written?', pairs: 8,
  },
  'kv-cache-newest-row-equivalence': {
    operation: 'cache-frontier', review: 'The access row should remain intact even when old projections are reused. Draw three stored keys and one newly written key under the single new query.',
    prediction: 'Does reusing the earlier keys remove them from the newest query?', pairs: 4,
  },
  'attention-causal-mask-row': {
    operation: 'causal-frontier', review: 'Zeros in a row look like ordinary small weights. Put an explicit time barrier before future positions and stop their contribution routes.',
    prediction: 'Can a future key contribute a tiny positive amount through this mask?', weights: [.2, .3, .5, 0, 0], query: 2,
  },
  'mixup-cutmix-input-target-coupling': {
    operation: 'clipped-image-mass', review: 'The target must follow the pixels actually inserted. Clip a sampled rectangle at the image edge and count only its thirty visible pixels.',
    prediction: 'Should the target use the sampled 42 pixels or the realized 30?', side: 10, patch: [5, 2, 7, 6],
    caption: 'The sampled patch extends beyond the image. Clipping leaves 30 of 100 pixels. Its target weight is 0.3. The retained base contributes 0.7.',
  },
  'pruning-shape-versus-zeros': {
    operation: 'remove-structure', review: 'Count physical matrix slots before and after removal. Zeroing values keeps nine slots; deleting a row leaves six.',
    prediction: 'Does filling a channel with zeros change the dense matrix shape?', values: [[1, 0, 2], [0, 0, 0], [3, 0, 4]], removed: 1,
  },
  'domain-adaptation-shift-assumptions': {
    operation: 'relabel-population', review: 'Input matching is insufficient when labels flip. Keep the same four input identities while changing only their class colors.',
    prediction: 'Can a method checking only the input counts detect this reversed rule?', inputs: [0, 0, 1, 1], labels: [[0, 0, 1, 1], [1, 1, 0, 0]],
  },
  'grpo-group-relative-signal': {
    operation: 'centered-group', review: 'Success markers alone imply useful preference. Subtract the shared group level from each equally rewarded output and show every residual vanish.',
    prediction: 'Which output receives a positive centered preference when all rewards equal one?', rewards: [1, 1, 1, 1],
  },
  'normalization-shared-statistics': {
    operation: 'axis-pooling', review: 'The mean table hides the pooling axis. Encircle rows and then columns of the same four values before computing the two different summaries.',
    prediction: 'Would exchanging the two samples change which features share a BatchNorm mean?', values: [[1, 3], [5, 7]],
  },
  'streaming-metrics-counts-before-f1': {
    operation: 'count-pooling', review: 'Metric averaging loses the observation denominator. Concatenate the three categories of case identities before computing one nonlinear ratio.',
    prediction: 'Why does the one-case shard not receive half the pooled denominator?', shards: [[8, 2, 2], [1, 0, 0]],
  },
  'kmeans-assign-then-update': {
    operation: 'assigned-centers', review: 'Move each center to the balance point of its assigned observations, while keeping the assignment colors fixed.',
    prediction: 'During this centroid update, can a point silently change its assigned cluster?', points: [[0, 2], [8, 10]], before: [0, 10],
  },
  'lora-zero-impact-first-gradient': {
    operation: 'arithmetic-paths', review: 'A zero output does not sever every sensitivity route. Compare the factor that vanishes in each initial derivative.',
    prediction: 'Which factor has a nonzero first gradient despite the zero adapter output?', paths: [
      { label: 'Forward: B times Ax', inputs: [0, 2], operator: 'multiply', result: 0 },
      { label: 'B sensitivity: upstream times Ax', inputs: [3, 2], operator: 'multiply', result: 6 },
      { label: 'A sensitivity: upstream times Bx', inputs: [3, 0], operator: 'multiply', result: 0 },
    ],
  },
  'gp-conditioning-local-uncertainty': {
    operation: 'uncertainty-pin', review: 'A single standard-deviation curve hides the constrained prediction envelope. Pin the whole symmetric band at the observed input and let it reopen with distance.',
    prediction: 'Does one noiseless observation eliminate uncertainty everywhere?',
    caption: 'The noiseless observation pins uncertainty at zero. The symmetric band shows plus or minus one posterior standard deviation. This unit RBF example reopens away from the observed input.',
  },
  'distillation-temperature-dark-knowledge': {
    operation: 'soften-mass', review: 'A rounded table does not expose redistributed probability. Recompute exact softmax masses and transfer width from the winner to both alternatives.',
    prediction: 'Does heating these logits change which class is largest?', logits: [2, 1, 0], temperatures: [1, 2],
  },
  'shap-all-orderings-credit': {
    operation: 'ordering-credit', review: 'Abstract formulas hide the order-dependent marginal. Let the same total payoff be allocated in both feature orders, then average each feature’s two gains.',
    prediction: 'Can choosing only one feature order give a different attribution?', values: [0, 2, 1, 6],
    caption: 'This illustrative game has values 0, 2, 1, and 6 for empty, A, B, and both. Order changes marginal credit. Averaging both orders gives A 3.5 and B 2.5.',
  },
  'decoding-fixed-count-adaptive-mass': {
    operation: 'mass-cutoffs', review: 'A stacked mass bar does not expose membership. Keep one hundred probability units and select the same token identities under two different rules.',
    prediction: 'Why must the 0.8 cutoff include a third token?', counts: [45, 25, 15, 10, 5], selected: [2, 3],
  },
  'forward-backward-viterbi-chain-objectives': {
    operation: 'joint-paths', review: 'Separate most-probable marginals can imply a most-probable path. Show the four complete-path masses and pool them along each time axis.',
    prediction: 'Is the path AB formed by the marginal winners the largest single path?', masses: [.35, .3, .05, .3],
  },
  'learning-rate-range-to-schedule': {
    operation: 'iterative-step', review: 'A sweep curve hides the local stability mechanism. Run three update gains from the same quadratic coordinate and show contraction, oscillation, or growth.',
    prediction: 'Does crossing the minimum necessarily mean the iterations diverge?', rates: [.1, .9, 1.1], steps: 3,
    caption: 'For loss x squared, the update multiplies x by one minus twice the learning rate. These three-step fixtures contract, oscillate, or grow. A range test still requires observed task evidence.',
  },
  'l1-l2-first-contact-geometry': {
    operation: 'constraint-contact', review: 'Endpoints alone do not show why one coordinate disappears. Project the same target onto an actual diamond and circle with equal geometric scales.',
    prediction: 'Which boundary has a corner on a coordinate axis?', target: [2, .4],
  },
  'q-learning-off-policy-backup': {
    operation: 'arithmetic-paths', review: 'The behavior and backup choices should be different operations. Keep the sampled value two visible while the maximizing target route selects five.',
    prediction: 'Does the backup use the behavior action simply because it was sampled?', paths: [
      { label: 'Backup chooses max; behavior chose two', inputs: [2, 5], operator: 'maximum', result: 5 },
      { label: 'Discount the selected next value', inputs: [.9, 5], operator: 'multiply', result: 4.5 },
      { label: 'Add the immediate reward', inputs: [1, 4.5], operator: 'add', result: 5.5 },
    ],
  },
  'autograd-shared-node-accumulate-then-propagate': {
    operation: 'arithmetic-paths', review: 'Propagating a partial contribution early loses the shared-node sum. Make accumulation precede the local derivative gate.',
    prediction: 'What contribution is missing if the first branch propagates before the second arrives?', paths: [
      { label: 'Collect both incoming contributions', inputs: [12, 2], operator: 'add', result: 14 },
      { label: 'Then propagate through du/dx = 3', inputs: [14, 3], operator: 'multiply', result: 42 },
    ],
  },
  'reparameterization-gradient-path': {
    operation: 'fixed-noise', review: 'The noise must remain external while the transformation moves. Pin one epsilon draw and move only the location parameter on the output ruler.',
    prediction: 'Is another noise sample needed to differentiate this one transformation?', epsilon: -1, sigma: .5, means: [2, 3],
    caption: 'One fixed noise draw is minus one. With scale 0.5, changing the mean from two to three moves the sample from 1.5 to 2.5. Gradients follow the deterministic transformation.',
  },
  'fp16-loss-scaling-ledger': {
    operation: 'scale-before-round', review: 'Rounding is an irreversible threshold, not a reversible algebraic step. Compare scaling before the representation floor with multiplying a zero afterward.',
    prediction: 'Can multiplying the already rounded zero recover the original gradient?', exponent: -30, scaleExponent: 20,
    caption: 'The gradient is two to the minus thirty. FP16 rounds it to zero. First scale by 2^20, giving 2^-10. Unscaling then recovers the original value.',
  },
  'ranking-metrics-per-rank-credit': {
    operation: 'rank-credit', review: 'Equal final values hide different collection rules. Follow the same relevance identities and show one metric stopping at the first hit.',
    prediction: 'Which metric can change if a later relevant item moves?', relevant: [false, true, false, true],
  },
  'lambdarank-swap-impact-weight': {
    operation: 'rank-displacement', review: 'A bar comparison hides rank location. Move the same relevant item through two equal-distance swaps on a fixed discount ladder.',
    prediction: 'Does moving by one rank cost the same near the top and near the bottom?', moves: [[1, 2], [9, 10]],
  },
  'quantization-scale-granularity': {
    operation: 'quantization-ruler', review: 'The outlier changes representable spacing, not the original small values. Snap the same two signs onto coarse and fine rulers.',
    prediction: 'Can the coarse shared scale distinguish these two small signs?', values: [-.5, .5], scales: [2, .5 / 127],
  },
  'ppo-clipped-surrogate-plateaus': {
    operation: 'one-sided-plateau', review: 'A clipped objective is not a ratio wall. Put an actual ratio outside the clipping interval on a flat objective segment.',
    prediction: 'Is the ratio forbidden from reaching 1.6 when its objective contribution plateaus?', probes: [.6, 1.6],
  },
  'd-separation-conditioning-switch': {
    operation: 'conditioning-paths', review: 'A truth table hides the graph motif. Draw chain and collider directions before marking how observing their middle changes connectivity.',
    prediction: 'Does observing the middle node always block the path?',
  },
  'causal-confounding-dag': {
    operation: 'confounding-intervention', review: 'Association is a surviving common-cause route. Compare observational assignment with an intervention that cuts the assignment edge without changing the outcome edge.',
    prediction: 'Which arrow is removed by assigning messages independently of prior engagement?',
    caption: 'Prior engagement affects messages and churn in this graph. An intervention cuts engagement’s assignment arrow into messages. Association alone does not establish a messages-to-churn effect.',
  },
  'actor-critic-two-objective-flow': {
    operation: 'actor-critic-routes', review: 'Differentiate shared features from the detached advantage target. Show both head gradients reaching the trunk while the actor-to-advantage route stops.',
    prediction: 'Does stopping the advantage route also stop the actor from updating shared features?',
  },
  'explain-backprop-one-loss-all-gradients': {
    operation: 'reverse-fanout', review: 'One scalar objective should initiate one reverse flow. Follow a shared sensitivity into two parameter derivatives rather than restarting from the loss.',
    prediction: 'Does each parameter require a separate complete reverse traversal?',
    caption: 'This illustrative graph uses u=a+3b and loss 2u. One backward flow gives shared sensitivity two, then parameter gradients two and six. Shared contributions accumulate before propagation.',
  },
  'pre-norm-decoder-residual-stream': {
    operation: 'forward-bypass', review: 'The normalized branch must not replace the raw stream. Carry one unchanged input along a bypass and add the learned correction at the end.',
    prediction: 'If the learned correction is zero, does normalization still replace the original stream?',
    caption: 'The original residual stream bypasses normalization. The learned branch reads normalized input, then adds its correction back to the unchanged stream. Setting the correction to zero preserves the input.',
  },
  'diffusion-training-sampling-asymmetry': {
    operation: 'dependent-generation', review: 'A training example samples one time; generation requires the predecessor result. Compare an independent sampled-time loss with a dependent reverse chain.',
    prediction: 'Can the second reverse transition run without the first transition’s output?',
  },
  'hmm-hidden-chain-local-emissions': {
    operation: 'hidden-emissions', review: 'Keep the horizontal dependency in the hidden state lane and make every observed emission a local vertical child.',
    prediction: 'Does a direct arrow between neighboring observations belong in this HMM?',
  },
  'em-lower-bound-climb': {
    operation: 'bound-climb', review: 'A pair of curves can hide the monotonicity mechanism. Mark their contact, climb the supporting bound, and then read the higher true objective.',
    prediction: 'Why is the true objective at the new bound maximum no lower than the old objective?', current: .5, next: .25,
  },
  'kge-antisymmetry-score-swap': {
    operation: 'complex-turn', review: 'A sign ledger does not reveal the orientation change. Multiply unit complex factors as quarter-turns and expose the real-axis endpoint after swapping.',
    prediction: 'Why does conjugating the swapped head reverse the rotation?', factors: [[1, 1, 0], [0, 1, -1]],
  },
  'bandit-three-selection-rules': {
    operation: 'uncertainty-choice', review: 'The decision must reflect the operation applied to each same arm. Compare mean locations, uncertainty bonuses, and one posterior draw on common rulers.',
    prediction: 'Can the lower estimated mean win an uncertainty-aware choice?', values: [[.65, .6], [.67, .72], [.62, .75]],
  },
  'crf-global-path-scoring': {
    operation: 'path-score-movement', review: 'The transition penalty is backward motion from the accumulated emission score. Follow complete paths rather than choosing each emission independently.',
    prediction: 'Does the path with the best local emissions still win after its transition penalty?', paths: [[4, -5], [3, 0], [2, 0]], labels: ['AA', 'AB', 'BB'],
  },
  'multitask-gradient-diagnosis': {
    operation: 'opposed-directions', review: 'Equal norm markers must preserve direction. Join the unit vectors head-to-tail so their cancellation is visible.',
    prediction: 'Does matching the two norms resolve their opposing update directions?',
  },
  'fm-shared-embeddings-unseen-pair': {
    operation: 'weighted-values', review: 'An unseen pair should reuse matching coordinate products, not acquire a fictional observation. Collect the two shared embedding contributions.',
    prediction: 'Does estimating the interaction prove that this feature pair was observed?', weights: [1, 2], values: [3, 1], expected: [3, 2], widthUnit: 1, heightUnit: 1, unit: 1, widthLabel: 'A coordinate', heightLabel: 'B coordinate', totalLabel: 'Shared embeddings supply an estimated pair weight.',
  },
  'mf-missing-rating-dot-product': {
    operation: 'weighted-values', review: 'Keep the missing rating distinct from a recorded value. Gather aligned user-item coordinate products into an explicitly estimated result.',
    prediction: 'Which observation would be required to call five a recorded rating?', weights: [1, 2], values: [3, 1], expected: [3, 2], widthUnit: 1, heightUnit: 1, unit: 1, widthLabel: 'user factor', heightLabel: 'item factor', totalLabel: 'Five is a prediction, not an observed rating.',
  },
  'attention-scaling-variance-trace': {
    operation: 'variance-square', review: 'Dividing a standard deviation scales both dimensions of its variance square. Keep sixty-four independent unit contributions visible before shrinking.',
    prediction: 'Why does dividing the dot product by eight divide its variance by sixty-four?', deviations: [8, 1],
  },
  'cross-entropy-confident-wrong-gradient': {
    operation: 'saturated-gradient', review: 'Wrong confidence and saturation are separate mechanisms. Compare the full BCE sensitivity with squared error passing through the small sigmoid gate.',
    prediction: 'At predicted probability 0.01 for a positive label, which gradient remains large?', probability: .01,
    caption: 'The positive label has predicted probability 0.01. Cross-entropy gives absolute logit gradient 0.99. Squared error also passes through the sigmoid slope, giving 0.019602. Confidence alone does not explain gradient size.',
  },
  'factor-model-covariance-decomposition': {
    operation: 'shared-factor-area', review: 'Off-diagonal dependence should be the same shared cause appearing in both assets. Build all four covariance products from one factor and distinguish independent diagonal noise.',
    prediction: 'Can independent asset noise add an off-diagonal covariance term?', loadings: [1, 2],
  },
  'gradient-accumulation-state-boundary': {
    operation: 'parameter-version', review: 'Four passes should carry the same parameter version until the accumulated gradient crosses one optimizer boundary.',
    prediction: 'Which microbatch in this accumulation cycle sees parameter version v1?', gradients: [1, 2, 3, 4],
  },
  'als-alternating-independent-solves': {
    operation: 'frozen-factor', review: 'Alternation is a freeze-and-solve operation, not two moving factor tables. Hold one rank-one item vector while two independent user rows settle.',
    prediction: 'Which factor must stay fixed while these user rows are solved?', ratings: [[2, 4], [3, 6]], fixed: [1, 2],
    caption: 'This rank-one fixture holds item factors (1,2) fixed. Independent user rows solve to two and three, reconstructing ratings (2,4) and (3,6). The next alternating stage may update item factors.',
  },
  'optimizer-sharpness-reparameterization': {
    operation: 'coordinate-relabel', review: 'Keep one physical scalar prediction while changing its ruler labels. The same displacement receives a smaller coordinate value and a larger coordinate curvature.',
    prediction: 'Does relabeling a=2b alter the represented scalar prediction?', factor: 2,
  },
  'discrete-gradients-forward-backward-paths': {
    operation: 'hard-forward-surrogate', review: 'The hard forward gate stays unchanged when a surrogate reverse route is declared. Draw both directions separately and name the surrogate explicitly.',
    prediction: 'Does choosing a straight-through gradient smooth the actual forward threshold?', input: .2,
  },
  'markov-chain-stationary-flow-ledger': {
    operation: 'balanced-reservoirs', review: 'Transition rates differ because reservoir masses differ. Draw unequal stationary stores with equal opposite mass flows.',
    prediction: 'Can the stationary distribution stay fixed while individual states continue moving?', masses: [2 / 3, 1 / 3], rates: [.2, .4],
  },
  'jacobian-transpose-pullback': {
    operation: 'weighted-jacobian', review: 'Each input column collects from every output. Give local derivatives height and upstream output sensitivities width before gathering each column.',
    prediction: 'Which input loses a contribution if only one output row is propagated?', jacobian: [[1, 2], [3, 4]], upstream: [2, 1],
  },
  'perplexity-tokenizer-denominator': {
    operation: 'token-buckets', review: 'The same text occupies different numbers of token buckets. Hold total code length fixed and contrast per-token and per-character denominators.',
    prediction: 'Can equal total code length yield different perplexities under different tokenizations?', letters: ['a', 'b', 'c', 'd'], groups: [1, 2], bits: 4,
    caption: 'Both tokenizations encode the same four characters using four illustrative bits. Four tokens give perplexity two; two tokens give perplexity four. Both use one bit per character.',
  },
  'positive-definite-boundedness': {
    operation: 'unbounded-valley', review: 'A flat line should be an escape route in parameter space. Draw the zero-loss center of an indefinitely extending sublevel strip.',
    prediction: 'Can moving arbitrarily far along y increase x squared when x stays zero?',
  },
  'precision-recall-f1-harmonic-penalty': {
    operation: 'reciprocal-rates', review: 'Show the reciprocal cost hidden in the harmonic mean. A weak rate creates a large contribution to the reciprocal denominator.',
    prediction: 'Which reciprocal term dominates when precision is 0.9 but recall is 0.1?', rates: [.9, .1],
  },
  'distribution-observation-unit': {
    operation: 'event-count-wait', review: 'Use one event stream to define two different observations: count marks in a fixed window, or measure the gap to the next event.',
    prediction: 'Does the waiting-time ruler produce integer-valued observations?', waits: [.17, 1.42, 3.08],
  },
  'random-forest-correlation-floor': {
    operation: 'shared-noise-floor', review: 'Correlated error is a retained common component. Average away private variance while keeping the same common tiles in every ensemble.',
    prediction: 'Which component remains even as the number of trees becomes very large?', trees: [1, 4, 16], common: .2, private: .8,
  },
  'bm25-saturation-length-penalty': {
    operation: 'document-matches', review: 'Term frequency and length must describe actual documents. Preserve four highlighted matches while adding nonmatching words, then compare eight matches at the original length.',
    prediction: 'Do extra nonmatching words improve the same four-match score?', documents: [[4, 1], [8, 1], [4, 2]],
  },
  'warmup-cosine-three-checkpoints': {
    operation: 'fixed-floor', review: 'Separate the persistent floor from the decaying headroom. The midpoint retains half of the 0.8 headroom, not half of the total one.',
    prediction: 'Why is the post-warmup midpoint 0.6 rather than 0.5?', floor: .2, phases: [0, .5, 1],
  },
  'checkpoint-boundary-recompute-balance': {
    operation: 'saved-rebuilt-states', review: 'A minimum curve hides competing inventories. Count saved segment boundaries and the currently rebuilt segment on separate physical rails.',
    prediction: 'Why does saving fewer boundaries eventually require more concurrent rebuilt states?', layers: 16, segments: [1, 4, 8],
  },
  'intervals-two-randomness-models': {
    operation: 'repeat-or-condition', review: 'Keep truth fixed while repeated intervals move, then keep one dataset fixed while posterior uncertainty spans possible parameters.',
    prediction: 'Which panel repeats the data, and which conditions on one observed dataset?', successes: 3, failures: 2,
  },
  'kernel-trick-exact-shortcut': {
    operation: 'equal-kernel-area', review: 'The shortcut must end at the same quantity. Count thirty-six unit contributions through explicit features and through the compact square.',
    prediction: 'Does the compact route approximate the expanded feature result?', products: [4, 4, 8, 8, 8, 4], base: 6,
  },
  'kl-direction-candidate-ranking': {
    operation: 'averaging-population', review: 'The averaging distribution should be physically different. Keep P fixed in forward comparisons, then draw each candidate’s different population in reverse comparisons.',
    prediction: 'Which distribution supplies the averaging weights in reverse KL?', target: [.9, .1], candidates: [[.7, .3], [.99, .01]],
  },
  'gcn-neighborhood-to-matrix-row': {
    operation: 'neighborhood-messages', review: 'One node update gathers only its declared neighborhood. Add an isolated counterexample whose large feature must not leak into the aggregation.',
    prediction: 'Can the isolated node’s value enter B without an edge?', features: [1, 2, 3, 100],
    caption: 'This illustrative B update averages features from A, B, and C, giving two. Isolated D has value 100 but no message route. The exact weighting depends on the declared GCN.',
  },
  'gpu-roofline-batching-ridge': {
    operation: 'bandwidth-ceiling', review: 'Arithmetic intensity is work per transferred byte. Keep bandwidth fixed while each byte supports one or two work units, and label the result as a ceiling.',
    prediction: 'Does a doubled memory ceiling prove an observed throughput doubled?', intensities: [1, 2],
  },
  'z-loss-anchors-common-offset': {
    operation: 'normalizer-shift', review: 'A common offset expands every unnormalized mass by the same factor. Renormalize both into identical shares while their log-normalizers remain different.',
    prediction: 'Which quantity changes even when every softmax probability stays fixed?', logits: [0, 1], shift: 2,
  },
  'batched-top-k-running-invariant': {
    operation: 'retained-frontier', review: 'Keep candidate identities outside the retained frontier. A new eight can replace seven, but discarded three cannot later overtake it.',
    prediction: 'Why is previously discarded three unnecessary when eight arrives?', values: [9, 3, 7, 2, 8], retained: [[9, 7], [9, 8]],
  },
  'resnet-shortcut-shape-contract': {
    operation: 'shape-contract', review: 'Addition should be physically blocked by mismatched axes. Compare unmatched spatial/channel inventories with the projected shortcut that exactly matches the branch.',
    prediction: 'Does matching only the spatial dimensions make this residual addition valid?', input: [8, 8, 16], output: [4, 4, 32],
  },
  'linear-attention-two-escape-routes': {
    operation: 'feature-summary', review: 'The kernel route stores sufficient feature sums before queries arrive. Work one exact positive-feature example, while retaining the distinct position-compression route in the derivation.',
    prediction: 'Which sums can be retained without storing a query-by-position matrix?', features: [[1, 0], [0, 1], [1, 1]], values: [2, 4, 6], query: [1, 2],
    caption: 'This positive-feature example stores S=(8,10) and z=(2,2). Query features (1,2) read 28/6. This is its declared kernel rule, not exact softmax. Position compression is a separate route.',
  },
  'positional-relative-mechanisms': {
    operation: 'position-entry-point', review: 'Position can change coordinates before a dot product or change the score afterward. Draw the two routes with different entry points.',
    prediction: 'Does an ALiBi penalty rotate the stored query and key vectors?', distances: [0, 1, 2], slope: .5,
  },
  'rnnt-two-axis-alignment-lattice': {
    operation: 'alignment-axes', review: 'Replace the state table with a traversable lattice. Two token moves share one time coordinate before blank moves advance audio time.',
    prediction: 'Does emitting a second token require advancing to another audio frame?', path: [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]],
  },
  'asr-alignment-decoder-choices': {
    operation: 'alignment-axes', review: 'Keep the time coordinate visibly fixed across a token edge, then move it only across the blank edge.',
    prediction: 'Which edge changes audio time rather than output position?', path: [[2, 1], [2, 2], [3, 2]],
  },
  'roc-pr-same-operating-point': {
    operation: 'denominator-members', review: 'The rare-positive denominator must remain countable. Reuse the same nine hit identities in ten actual positives and 108 predicted positives.',
    prediction: 'Why can an approximately one-percent false-positive rate still give low precision?',
  },
  'rope-relative-rotation': {
    operation: 'common-rotation', review: 'Rotate both actual direction arrows together and retain their right angle. A shared rotation preserves their dot product, not every model output.',
    prediction: 'Does the ninety-degree angle change when both arrows rotate forty-five degrees?', angle: Math.PI / 4,
  },
  'attention-source-routing': {
    operation: 'query-rows', review: 'Route three source values into each of two distinct query destinations. Output count follows the destination query rows.',
    prediction: 'Do three source keys create a third output when there are only two queries?', sources: 3, queries: 2,
  },
  'semantic-segmentation-boundary-fusion': {
    operation: 'lost-boundary', review: 'Pool two different boundary patterns into the same coarse value. Enlarge that one value and show the original distinction cannot be recovered.',
    prediction: 'Which original pattern can be recovered from the coarse value alone?', patterns: [[0, 0, 1, 1], [0, 1, 0, 1]],
  },
  'packed-sequence-mask-boundaries': {
    operation: 'packed-isolation', review: 'Earlier physical position is insufficient authority to attend. Draw two causal blocks and a sealed cross-example boundary.',
    prediction: 'Can B1 read A1 merely because A1 appears earlier in the packed buffer?',
  },
  'momentum-alternating-components': {
    operation: 'velocity-fold', review: 'Persistent and alternating directions should drive the same recurrence. Track each retained velocity before adding the new gradient.',
    prediction: 'Which sequence keeps reinforcing the same direction under beta 0.5?', gradients: [[1, 1, 1, 1], [1, -1, 1, -1]], beta: .5,
  },
  'same-mesh-axis-diagonal-gap': {
    operation: 'missing-cross-products', review: 'Make all four required block slots physical. Fill only diagonal products and leave cross terms empty rather than pretending addition creates them.',
    prediction: 'Can adding the two diagonal products create A0B1?',
  },
  'softmax-ce-logit-update': {
    operation: 'subtract-gradient', review: 'Negative sensitivity should move the true logit upward under descent. Draw actual before-and-after coordinates for all three logits.',
    prediction: 'Why does a negative gradient raise the true-class logit?', gradients: [-2 / 3, 1 / 3, 1 / 3], rate: .3,
  },
  'sparse-attention-reach-patterns': {
    operation: 'two-hop-relay', review: 'One missing direct entry remains missing in one layer. Track B information into G, then into A on the next layer.',
    prediction: 'Can A obtain B’s information through G in the first layer?',
  },
  'speaker-verification-threshold-tradeoff': {
    operation: 'threshold-members', review: 'Move a threshold across persistent genuine and impostor identities. Count who changes from accepted to rejected instead of memorizing two density shapes.',
    prediction: 'Which genuine speaker is rejected when the threshold rises to one?', genuine: [-.2, .3, .8, 1.2, 1.8], impostors: [-1.8, -1.2, -.7, -.1, .6], thresholds: [0, 1],
    caption: 'These ten illustrative scores overlap. Raising the threshold from zero to one rejects the remaining impostor but also two previously accepted genuine speakers. They are not measured operating rates.',
  },
  'svm-support-vectors-set-boundary': {
    operation: 'nonzero-contributions', review: 'Keep all four training examples present, but stop routes whose dual weights are exactly zero.',
    prediction: 'Does a zero dual contribution delete the corresponding training observation?', weights: [.7, 0, .4, 0],
  },
  'bpe-corpus-merge-trace': {
    operation: 'fixed-merge-order', review: 'The new word obeys a previously learned ordering. Retain character identity while consuming the three fixed merge instructions.',
    prediction: 'Does encoding lower relearn pair priorities from that single word?', stages: [['l', 'o', 'w', 'e', 'r'], ['lo', 'w', 'e', 'r'], ['low', 'e', 'r'], ['low', 'er']],
  },
  'transformer-two-mixing-axes': {
    operation: 'mixing-axes', review: 'Rotate the connectivity emphasis without changing the token-feature grid. Attention connects token rows; feed-forward transformations connect features inside one row.',
    prediction: 'Can one feed-forward application exchange information between different token rows?',
  },
  'nonlinear-embedding-local-vs-global': {
    operation: 'local-gap', review: 'Preserve each close pair while expanding the displayed gap between pairs. The drawing’s global separation should visibly change without altering the chosen local neighbors.',
    prediction: 'Does the changed page gap establish a changed original-space distance?', layouts: [[65, 95, 170, 200], [35, 65, 280, 310]],
  },
  'uat-existence-error-band': {
    operation: 'piecewise-fit', review: 'Show the actual approximating segments and separately magnify their error. Existence of this fit must not be mistaken for successful training or generalization.',
    prediction: 'Does constructing these four segments prove a training algorithm will find them?', segments: 4, tolerance: .02,
  },
  'value-policy-action-interface': {
    operation: 'equal-value-policies', review: 'Keep equal action values fixed while two declared selection rules produce different actions or distributions.',
    prediction: 'Which rule breaks the tie, and which rule preserves equal probability?',
  },
  'vision-patches-to-positioned-tokens': {
    operation: 'patch-token-inventory', review: 'Carry four spatial patch identities into sequence positions, then prepend a visibly additional class token.',
    prediction: 'Why does a two-by-two patch grid produce five sequence positions here?',
  },
  'weight-decay-adaptive-paths': {
    operation: 'decay-bypass', review: 'Follow the coupled route through distinct frozen scales and the direct route around them. Equal weights alone do not equalize coupled shrinkage.',
    prediction: 'Which route changes if the preconditioner grows from one to ten?', scales: [1, 10], base: .001,
  },
  'why-dropout-shared-subnetworks': {
    operation: 'shared-mask-weights', review: 'Keep parameter identities fixed under both masks. Masked paths close without creating independently trained copies.',
    prediction: 'Does the second mask introduce another independently trained w3?', masks: [[1, 0, 1], [0, 1, 1]],
  },
  'word-analogy-shared-displacement': {
    operation: 'translated-displacement', review: 'Move the same displacement arrow to a new origin. Its endpoint is a query coordinate before vocabulary search chooses an actual word.',
    prediction: 'Does reaching coordinate (3,1) itself identify a vocabulary word?', points: [[0, 1], [1, 2], [2, 0], [3, 1]],
  },
  'product-metric-decision-authority': {
    operation: 'serial-gates', review: 'The launch rule is conjunction, not voting. Pass completion and cost, then stop the same release token at the latency gate.',
    prediction: 'Can two passing gates compensate for latency 2.1 above its limit two?', lanes: [{ labels: ['gain', 'cost', 'latency'], states: [true, true, false], detail: 'Gain 4.66; cost 1.97; latency 2.1. Release blocked.' }],
  },
  'quality-speed-proportionate-path': {
    operation: 'serial-gates', review: 'A protected record has its own required route. Ordinary success cannot turn its failed record-rule gate into a pass.',
    prediction: 'Does ordinary-case success authorize the protected-case violation?', lanes: [{ labels: ['speed', 'record rule'], states: [true, true], detail: 'Ordinary case passes both.' }, { labels: ['speed', 'record rule'], states: [true, false], detail: 'Protected case stops at its failed record rule.' }],
  },
  'ml-ab-test-launch-gates': {
    operation: 'serial-gates', review: 'Show the two candidates stopped at different gates. Statistical evidence does not replace practical benefit or safety.',
    prediction: 'Which failed requirement remains after the nonzero-effect check passes?', lanes: [{ labels: ['nonzero', 'worth it', 'safe'], states: [true, false, true], detail: 'Candidate A lacks practical value.' }, { labels: ['nonzero', 'worth it', 'safe'], states: [true, true, false], detail: 'Candidate B violates safety.' }],
  },
  'post-training-policy-gate': {
    operation: 'serial-gates', review: 'Keep task completion and authorization as separate gates on the same trial. Unauthorized success remains a failed trial.',
    prediction: 'Can successful completion override missing account authority?', lanes: [{ labels: ['task', 'authority'], states: [true, true], detail: 'Authorized successful trial passes.' }, { labels: ['task', 'authority'], states: [true, false], detail: 'Unauthorized successful trial is rejected.' }],
  },
  'distributed-recovery-consistency-gate': {
    operation: 'serial-gates', review: 'A recovery bundle needs all versions aligned. Trace the second rank’s matching weights through a stale optimizer gate.',
    prediction: 'Can correct v4 weights repair the stale v3 optimizer state?', lanes: [{ labels: ['model v4', 'optimizer v4', 'data p8'], states: [true, true, true], detail: 'Rank zero has a consistent bundle.' }, { labels: ['model v4', 'optimizer v3', 'data p8'], states: [true, false, true], detail: 'Rank one cannot restart from this mixed bundle.' }],
  },
  'foundation-data-deletion-blast-radius': {
    operation: 'evidence-chain', review: 'Source removal is only one known stage. Keep checkpoint and released-model influence as unresolved branches rather than completed deletion claims.',
    prediction: 'Does excluding D7 from new data prove its old model influence was removed?', labels: ['new data', 'old checkpoint', 'released model'], states: ['verified', 'unknown', 'unknown'],
  },
  'adaptive-red-team-release-loop': {
    operation: 'evidence-chain', review: 'One reproduced direct attack is now blocked. Preserve quoted and tool-output paths as separate untested evidence obligations.',
    prediction: 'Which attack paths still lack an observed post-fix result?', labels: ['direct attack', 'quoted attack', 'tool output'], states: ['verified', 'unknown', 'unknown'],
  },
  'coding-assistant-evidence-ladder': {
    operation: 'evidence-chain', review: 'A test result should not flow automatically into durable user-value claims. Put each unobserved adoption stage behind its own evidence boundary.',
    prediction: 'Which receipt would be required before claiming continued use?', labels: ['tested contract', 'acceptance', 'continued use', 'durable value'], states: ['verified', 'unknown', 'unknown', 'unknown'],
  },
  'policy-label-observation-loop': {
    operation: 'observed-labels', review: 'Mature positives and negatives provide actual label routes. Pending and censored records stop before the label bucket without becoming zeros.',
    prediction: 'Does a censored record belong in the observed negative bucket?', labels: ['mature +', 'mature -', 'pending', 'censored'], values: ['1', '0', '?', '?'],
  },
  'fraud-policy-label-coverage': {
    operation: 'observed-labels', review: 'Policy affects whether an outcome can be observed. Keep approved outcomes, pending maturity, and declined missing outcomes on distinct routes.',
    prediction: 'Can a declined payment be labeled legitimate from an unobserved repayment?', labels: ['mature approval', 'pending', 'declined'], values: ['0/1', '?', '?'],
  },
  'training-loop-persistent-state': {
    operation: 'separate-state-rails', review: 'Model mode and gradient buffer are independent persistent state. Let each action change only its own rail.',
    prediction: 'After zeroing gradients in evaluation mode, which state still needs repair?', states: [['train', 'old'], ['eval', 'old'], ['eval', 'zero'], ['train', 'zero']], actions: ['initial', 'eval()', 'zero()', 'train()'],
  },
  'model-not-learning-diagnostic-path': {
    operation: 'broken-update', review: 'A nonzero gradient is not an applied update. Stop the route before delta weights and preserve the unchanged loss readout.',
    prediction: 'Which observation distinguishes gradient computation from a real parameter step?', gradient: .8, change: 0, loss: 1.1,
  },
  'multi-team-platform-contract-boundary': {
    operation: 'compare-swap-owner', review: 'A version gate grants ownership only to the matching attempt. Carry A’s accepted version into B’s stale comparison and keep A’s result untouched.',
    prediction: 'Can B overwrite A while still expecting version zero?',
  },
  'enterprise-agent-three-plane-boundary': {
    operation: 'idempotent-receipt', review: 'The same idempotency identity must return one committed receipt after reply loss. Keep the account effect visibly singular across both requests.',
    prediction: 'Does a lost response authorize creating a second payment?',
  },
  'rag-recall-precision-funnel': {
    operation: 'retained-evidence', review: 'Carry the two required passages through selection. A removed passage stays outside the later reranker and final context.',
    prediction: 'Can reranking recover evidence B after the cap removed it?', stages: [['A', 'B'], ['A'], ['A']], labels: ['retrieved', 'after cap', 'reranked'],
  },
  'llm-inference-two-budget-scheduler': {
    operation: 'persistent-refreshing', review: 'Draw the persistent twelve-page ownership inventory separately from the eight reusable work slots. A new iteration refreshes only the work rail.',
    prediction: 'Do free work slots release the nine pages already reserved by A and B?', pages: ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', '-', '-', '-'], work: ['D', 'D', 'D', 'P', 'P', 'P', 'P', 'P'],
  },
  'reasoning-fixed-budget-reclaim': {
    operation: 'node-reassignment', review: 'Carry the same eight node identities across serving pressure. Reclaim the two offline nodes only after their checkpoint-and-pause boundary.',
    prediction: 'Do quarterly accelerator hours create another free node in this instantaneous pool?', before: ['S', 'S', 'S', 'S', 'S', 'S', 'O', 'O'], after: ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
  },
  'offline-online-gap-diagnostic-order': {
    operation: 'assignment-membership', review: 'Intention-to-treat preserves the full assigned group. Circle the forty fallback identities as changed delivery, not removal from the experimental denominator.',
    prediction: 'Does excluding fallback users preserve the originally assigned estimand?', total: 100, served: 60, unit: 1,
  },
  'youtube-recommender-selective-feedback-loop': {
    operation: 'assignment-membership', review: 'Selection creates exposure evidence for only ten candidates. Keep the 890 unshown candidates visibly present without assigning negative outcomes.',
    prediction: 'Do unshown eligible candidates supply observed negative responses?', total: 900, served: 10, unit: 10,
  },
  'chatbot-randomization-unit': {
    operation: 'conversation-assignment', review: 'One outcome encloses all three turns. Contrast one conserved conversation assignment with an A-B-A sequence that contaminates the same outcome.',
    prediction: 'Can this one conversation’s outcome isolate A after its middle turn uses B?',
  },
  'hierarchical-all-reduce-path': {
    operation: 'hierarchical-sum', review: 'Keep the rank inputs separate until local grouping. Combine four and six once, then copy the completed ten into every rank destination.',
    prediction: 'Does distributing the completed ten require adding it again?', inputs: [1, 3, 2, 4],
  },
  'ring-all-reduce-chunk-trace': {
    operation: 'ring-reduction', review: 'Distinguish a growing partial from circulating completed data. The second route copies ten rather than summing four tens.',
    prediction: 'After reduction reaches ten, should all-gather change its value?', inputs: [1, 2, 3, 4],
  },
  'agent-review-control-loop': {
    operation: 'strict-boundary', review: 'Make equality a distinct occupied boundary. Reject the boundary point as well as the expired region; only the strictly positive side passes.',
    prediction: 'Which result follows when expiry minus now equals zero?', positions: [-1, 0, 1], accepted: [false, false, true], labels: ['expired', 'equal now', 'future'],
  },
  'nms-greedy-suppression': {
    operation: 'suppression-route', review: 'Retain the overlapping three-box chain and make the removed B route visibly unable to suppress C.',
    prediction: 'Does B’s overlap with C matter after A has already removed B?',
  },
  'ecosystem-delayed-feedback-loop': {
    operation: 'clock-after-boundary', review: 'Keep the short-run observation boundary between viewer response and delayed creator response. The latter remains outside the first readout.',
    prediction: 'Which response has not yet been observed when short-run analysis closes?', events: [{ value: 0, label: 'allocation' }, { value: 1, label: 'viewer response' }, { value: 2, label: 'short-run close' }, { value: 3, label: 'creator response' }], boundary: 2,
  },
  'agent-governed-action-path': {
    operation: 'serial-gates', review: 'Execution has an authorization boundary, and a claimed effect has a receipt boundary. Pass both only on the fully evidenced route.',
    prediction: 'Can a proposed action dispatch before its exact authorization?', lanes: [{ labels: ['proposal', 'authority', 'receipt'], states: [true, false, false], detail: 'A proposal alone stops before execution.' }, { labels: ['proposal', 'authority', 'receipt'], states: [true, true, true], detail: 'The authorized action has an outcome receipt.' }],
  },
  'ivf-pq-query-funnel': {
    operation: 'unscanned-partition', review: 'Put the actual nearest member physically inside excluded C. Improving representations within A and B cannot open the absent scan route.',
    prediction: 'Can more accurate PQ distances inside A and B expose the true neighbor in C?',
  },
  'ml-role-activity-profiles': {
    operation: 'role-budget', review: 'Use one hundred shared units per fictional role so the activity mix changes while total time stays fixed.',
    prediction: 'Does the lead example have more total time, or a different allocation?', values: [[60, 15, 25], [10, 65, 25], [20, 20, 60]], labels: ['RS', 'MLE', 'lead'],
  },
  'autoregressive-diffusion-sampling-steps': {
    operation: 'committed-provisional', review: 'Retain committed prefix identities only on the autoregressive lane. Mark every diffusion position provisional until the completed sample.',
    prediction: 'Can the middle diffusion sample be treated as a committed left prefix?',
  },
  'coding-assistant-two-latency-paths': {
    operation: 'versioned-reply', review: 'Carry the v7 request identity through the intervening v8 edit. The fast v7 reply must stop at the version-match gate.',
    prediction: 'Does arriving quickly make the v7 answer applicable to v8?',
  },
  'cot-signal-is-not-faithfulness': {
    operation: 'separate-state-rails', review: 'Change the explanation rail while preserving the prohibited action rail. A calmer text appearance does not alter the committed write.',
    prediction: 'Which rail actually records whether the prohibited write changed?', states: [['alarm', 'write'], ['calm', 'write']], actions: ['before', 'after'], labels: ['explanation', 'action'], good: [['calm'], []], footer: 'The explanation changed; the prohibited action did not.',
  },
  'fused-moe-triton-sram-roofline': {
    operation: 'memory-roundtrip', review: 'The intermediate should follow an actual out-and-back HBM route in the unfused case, then stay on chip in the fitting fused case. Inputs and outputs still cross the boundary.',
    prediction: 'Which traffic remains after the intermediate round trip disappears?',
  },
  'coding-interview-method-state': {
    operation: 'complement-before-store', review: 'The current value queries an earlier identity before inserting itself. Preserve the distinct saved index rather than matching a freshly stored current index.',
    prediction: 'Which two original indices are returned when saved value two meets current value four?',
  },
  'content-moderation-governed-decision-loop': {
    operation: 'policy-region', review: 'Place the score inside the explicitly versioned P7 action region. The score is a coordinate, not an intrinsic publishing permission.',
    prediction: 'Which P7 region contains 0.62?', score: .62, boundaries: [.4, .8], actions: ['allow', 'review', 'block'],
  },
  'paper-critique-prioritization-funnel': {
    operation: 'matched-comparison', review: 'Arrange task accuracy and compute on separate axes. Keep the missing baseline-at-two-times-compute cell open rather than attributing the full reported gain.',
    prediction: 'Which missing result would isolate benefit at the new method’s resource budget?', labels: ['baseline 1x', 'baseline 2x', 'new at 2x'], values: [0, null, 4], suffix: 'points above A',
  },
  'cross-validation-preserve-structure': {
    operation: 'group-cut', review: 'Move the cut so all three identities of one user stay on one side. Contrast a random row assignment that divides the same user across train and test.',
    prediction: 'Which assignment keeps user A from appearing on both sides?',
  },
  'dbscan-neighborhood-expansion': {
    operation: 'core-border-source', review: 'Show E joining D’s neighborhood without acquiring D’s expansion authority. F remains disconnected; counts explicitly include self.',
    prediction: 'Can border E expand the cluster when its neighborhood count is only two?',
  },
  'frontier-training-divergence-zoom': {
    operation: 'first-failure', review: 'Traverse actual rank-time cells. Pin the first nonfinite cell at rank one, step two before showing the later shared symptoms.',
    prediction: 'Do the three bad cells at step three identify where failure began?',
  },
  'work-prioritization-counterfactual-portfolio': {
    operation: 'alternative-whole', review: 'Put two feasible fixed-capacity portfolios side by side. Compatible B and C share the alternative whole, not two unrelated opportunities added without permission.',
    prediction: 'Does past effort on A change its forward value eight versus eleven?', values: [8, 11], labels: ['A alone', 'compatible B + C'],
  },
  'decision-tree-partitions-feature-space': {
    operation: 'conditional-spatial-cut', review: 'Draw the root boundary across the whole feature plane and the second cut only on its right branch.',
    prediction: 'Does changing y above one move any x<=0 point out of A?',
  },
  'ethical-decision-process-outcome': {
    operation: 'clock-after-boundary', review: 'The committed decision boundary separates original evidence from later luck. Outcome observation cannot travel back into the decision-time evidence set.',
    prediction: 'Was the later unexpected outcome available to the original decision maker?', events: [{ value: 0, label: 'available evidence' }, { value: 1, label: 'decision committed' }, { value: 2, label: 'later outcome' }], boundary: 1,
  },
  'ablation-rival-hypotheses': {
    operation: 'matched-comparison', review: 'Make the original-to-matched resource gain distinct from the small matched-to-method gap. Only the latter isolates the component under this setup.',
    prediction: 'Which gap is the matched component effect: three or 0.2 points?', labels: ['original', 'matched control', 'method'], values: [80, 82.8, 83], suffix: 'score',
  },
  'coding-product-authority-boundary': {
    operation: 'authority-rails', review: 'Shared repository context reaches all four modes. Execution routes close in three modes and open only for a scoped isolated agent.',
    prediction: 'Does having the same repository context grant chat execution authority?', labels: ['inline', 'chat', 'review', 'agent'], allowed: [false, false, false, true],
  },
  'feature-store-two-time-contract': {
    operation: 'availability-before', review: 'Draw B’s event and arrival on one clock around noon. Only A has crossed the availability boundary before the historical query.',
    prediction: 'Can B’s earlier event time override its later 12:10 availability?',
  },
  'ml-monitoring-observability-layers': {
    operation: 'clock-after-boundary', review: 'Separate immediately known freshness failure from outcome labels that mature later. Unknown future quality is not a delay in detecting the current contract breach.',
    prediction: 'Which failure is already knowable before quality outcomes mature?', events: [{ value: 0, label: 'snapshot stale' }, { value: 1, label: 'freshness failure known' }, { value: 2, label: 'maturation wait' }, { value: 3, label: 'quality readout' }], boundary: 1,
  },
  'ecosystem-experiment-interference': {
    operation: 'shared-supply-return', review: 'Keep one shared creator reservoir between the assigned cells. Changed supply returns to both future treatment and future control.',
    prediction: 'Which return route breaks a simple no-interference interpretation?',
  },
  'spotify-two-level-shelf-composition': {
    operation: 'duplicate-penalty', review: 'Expose the repeated song identity in A+B before applying its four-point penalty. A+C retains ten without that repeated item.',
    prediction: 'Why does choosing the two highest-scoring shelves lose to A+C?',
  },
  'encoder-once-decoder-loop': {
    operation: 'fixed-source-growing-prefix', review: 'Carry the same source-memory identity across three decode steps while extending only the generated prefix.',
    prediction: 'Which inventory grows between step one and step three?',
  },
  'agent-eval-diagnostic-flywheel': {
    operation: 'serial-gates', review: 'Carry the same trial through task and authority checks. Passing task tests does not open the unauthorized-write gate.',
    prediction: 'Can this task-test pass make the joint trial pass?', lanes: [{ labels: ['task test', 'authority'], states: [true, false], detail: 'The unauthorized write vetoes the completed task.' }],
  },
  'search-ranker-evidence-flywheel': {
    operation: 'weighted-tail', review: 'Preserve the ninety/ten traffic weights while separating the failing tail from its larger head contribution.',
    prediction: 'Does weighted score 0.88 clear the tail floor 0.78?', weights: [.9, .1], values: [[.8, .8], [.9, .7]], floor: .78,
  },
  'independent-confirmation-vs-adaptive-reuse': {
    operation: 'clock-after-boundary', review: 'Keep benchmark-guided selection left of the lock and genuinely fresh confirmation right of it.',
    prediction: 'Is the feedback used to choose the model independent confirmation of that choice?', events: [{ value: 0, label: 'benchmark feedback' }, { value: 1, label: 'feedback guides choice' }, { value: 2, label: 'selection locked' }, { value: 3, label: 'fresh confirmation' }], boundary: 2,
  },
  'collective-failure-commit-boundary': {
    operation: 'restore-common', review: 'Mixed local progress cannot be a collective restart point. Both rank paths return to the same committed v4 before retrying.',
    prediction: 'Can rank zero keep v5 while rank one restores v4 for the retry?',
  },
  'llm-adaptation-layers': {
    operation: 'serial-gates', review: 'Schema validity is only one gate. Stale knowledge and failed policy retain their own blocking states.',
    prediction: 'Does valid JSON establish correct knowledge or authorized behavior?', lanes: [{ labels: ['schema', 'knowledge', 'policy'], states: [true, false, false], detail: 'Correct formatting cannot release this stale, prohibited response.' }],
  },
  'fine-tuning-two-decisions': {
    operation: 'independent-choices', review: 'Selection of a training signal and parameter scope should have separate rails. Cross all four combinations instead of treating LoRA as a supervision type.',
    prediction: 'Does choosing preference pairs force either full-scope or LoRA updates?',
  },
  'as-five-cross-cutting-signals': {
    operation: 'independent-evidence', review: 'Two observed signals remain observations, while three other skills stay unresolved. None should inherit evidence from confident presentation.',
    prediction: 'Which signals remain unknown after this framing-and-testing follow-up?', labels: ['framing', 'testing', 'scaling', 'leading', 'impact'], states: ['verified', 'verified', 'unknown', 'unknown', 'unknown'],
  },
  'flashattention-sram-tile-stream': {
    operation: 'stable-tile-merge', review: 'Discard local scores only after preserving the running maximum, denominator, and weighted numerator. Verify their rescaled merge against the complete softmax.',
    prediction: 'Which retained quantities let a second tile merge without keeping the first score tile?', logits: [[0, 2], [1]], values: [[1, 3], [5]],
    caption: 'These illustrative tiles retain a maximum, normalizer, and weighted-value accumulator. Stable rescaling merges both tiles to the same weighted output as complete softmax. Temporary score storage can then be discarded.',
  },
  'flashattention-standard-hbm-path': {
    operation: 'memory-roundtrip', review: 'Remove the out-and-back score-intermediate route, not all HBM activity. The surviving input/output route remains explicit.',
    prediction: 'Are QKV reads and output writes removed with temporary score storage?',
  },
  'foundation-data-curation-lineage': {
    operation: 'serial-gates', review: 'Accepted membership needs its own consumption receipt. Preserve rejected R18’s denial instead of treating absence as missing provenance.',
    prediction: 'Does manifest membership alone establish actual training consumption?', lanes: [{ labels: ['policy', 'manifest', 'read receipt'], states: [true, true, true], detail: 'R17 has permission, membership, and actual consumption evidence.' }, { labels: ['policy', 'manifest', 'read receipt'], states: [false, false, false], detail: 'R18 retains its denial; no membership or read is claimed.' }],
  },
  'interview-evidence-to-drill': {
    operation: 'independent-evidence', review: 'A known time constraint may select a drill; an unknown tool constraint must still prompt a permission question.',
    prediction: 'Does knowing the time budget grant unknown tool permission?', labels: ['time budget', 'tool permission'], states: ['verified', 'unknown'],
  },
  'proof-of-work-three-checks': {
    operation: 'independent-evidence', review: 'Patch evidence establishes authorship without automatically attributing decision or rollout ownership.',
    prediction: 'Which additional evidence would establish personal rollout responsibility?', labels: ['personal patch', 'decision ownership', 'rollout ownership'], states: ['verified', 'unknown', 'unknown'],
  },
  'fsdp-wrapped-unit-lifecycle': {
    operation: 'transient-gather', review: 'Follow one rank’s retained quarter into a temporary complete wrapped unit and back to its same resting shard. Gathering is a lifetime peak, not permanent replication of the whole model.',
    prediction: 'Does sharded resting storage eliminate the active unit’s temporary gather?',
  },
  'gan-alternating-update-ownership': {
    operation: 'gradient-ownership', review: 'Freeze D parameter updates without breaking the differentiable path through D on the G step. Stop fake-to-G gradients only on the D step.',
    prediction: 'Can gradients pass through D while D’s weights remain frozen?',
  },
  'hallucination-claim-release-gates': {
    operation: 'serial-gates', review: 'Factual support and release authority operate on the same claim but require independent receipts.',
    prediction: 'Does factual support authorize a private claim’s disclosure?', lanes: [{ labels: ['support', 'release'], states: [true, true], detail: 'Public supported claim is releasable.' }, { labels: ['support', 'release'], states: [true, false], detail: 'Private supported claim stops at release permission.' }],
  },
  'llm-eval-calibration-loop': {
    operation: 'count-floor', review: 'One trial mark is one joint result. Place the required fourteen-pass boundary after the thirteenth green identity.',
    prediction: 'Which single additional joint result is needed to reach seventy percent?', total: 20, passed: 13, required: 14,
  },
  'asr-adaptation-boundaries': {
    operation: 'replace-one-state', review: 'A modular lexicon change leaves the acoustic core identity untouched. Do not depict all adaptation as retraining all model parts.',
    prediction: 'Which component remains fixed in this modular adaptation?', states: [['acoustic fixed', 'lexicon old'], ['acoustic fixed', 'lexicon new']], changed: 1,
  },
  'model-gain-evidence-decision': {
    operation: 'gain-floor', review: 'Put the half-point estimate on the same ruler as the one-point practical threshold. Positive is not synonymous with worthwhile.',
    prediction: 'Does a statistically positive half point cross the one-point practical floor?', estimate: .5, required: 1,
  },
  'beam-search-live-finished-transition': {
    operation: 'normalized-ranking', review: 'Keep both complete beam identities while changing only the declared scoring normalization. Stop expansion at EOS.',
    prediction: 'Which finished beam wins after dividing log probability by its counted length?', probabilities: [.18, .15], lengths: [2, 3], labels: ['A EOS', 'A x EOS'],
  },
  'knn-index-decision-funnel': {
    operation: 'count-versus-size', review: 'Reduce the number of read objects separately from bytes per object. Preserve four baseline vectors while the two mechanisms target orthogonal dimensions.',
    prediction: 'Which change saves representation bytes without reducing scanned-vector count?',
    caption: 'Illustrative vector blocks distinguish read count from representation size. IVF scans fewer vectors; PQ stores fewer bytes per vector. Neither recovers a candidate excluded by the scan decision.',
  },
  'black-box-discriminating-probes': {
    operation: 'hypothesis-separation', review: 'Route the two hypotheses into the same result under a weak probe and different results under a strong probe.',
    prediction: 'Which probe has different predictions under the two rival hypotheses?',
  },
  'project-stop-forward-evidence-ledger': {
    operation: 'future-net', review: 'Follow forward value three minus remaining cost five to net minus two. Keep spent twenty outside the forward arithmetic route.',
    prediction: 'Can sunk twenty change the additional net value of continuing?', value: 3, remaining: 5, sunk: 20,
  },
  'senior-principal-delegated-coherence': {
    operation: 'scope-join', review: 'Local implementation ends within one owner’s boundary. A shared-interface change crosses into multiple affected owners who must align.',
    prediction: 'Which decision requires alignment beyond the local implementation owner?',
  },
  'reactive-pretraining-evidence-loop': {
    operation: 'recoverable-fork', review: 'Fork both continuations from one committed identity. The probe passes through an evidence gate before it could replace the recoverable mainline.',
    prediction: 'Does a promising probe erase the committed mainline before promotion?', labels: ['hot mainline', 'bounded probe'],
  },
  'judge-calibration-gates': {
    operation: 'serial-gates', review: 'A high quality score still stops at a hard-rule failure. The order-flip case passes that rule but stops at a separate calibration review.',
    prediction: 'Can quality 0.9 override either the hard failure or order sensitivity?', lanes: [{ labels: ['quality', 'hard rule', 'stable order'], states: [true, false, true], detail: 'Hard violation: hold despite quality 0.9.' }, { labels: ['quality', 'hard rule', 'stable order'], states: [true, true, false], detail: 'Order flip: review despite quality 0.9.' }],
  },
  'healthcare-llm-clinician-gate': {
    operation: 'serial-gates', review: 'The exact approval bundle binds patient, encounter, and draft. Matching the patient alone cannot release a new encounter or draft.',
    prediction: 'Does approval of P7/E4/D2 authorize current P7/E5/D3?', lanes: [{ labels: ['patient P7', 'encounter', 'draft'], states: [true, false, false], detail: 'E5/D3 differs from approved E4/D2; release is blocked.' }],
  },
  'llm-eval-chain-of-trust': {
    operation: 'serial-gates', review: 'A citation count proxy has no authority to clear independent account permission checks.',
    prediction: 'Can five citations authorize the new response’s prohibited account handling?', lanes: [{ labels: ['citations', 'authority'], states: [true, true], detail: 'Baseline two citations retains authority.' }, { labels: ['citations', 'authority'], states: [true, false], detail: 'New five-citation response fails authority.' }],
  },
  'llm-cost-compounds-across-agent-calls': {
    operation: 'cached-and-output', review: 'Keep a repeated-input component on each call and a fresh-output component alongside it. Caching cannot remove the generated-output ledger.',
    prediction: 'Which cost component repeats freshly even when all three inputs are cached?',
  },
  'llm-security-influence-authority-path': {
    operation: 'serial-gates', review: 'Untrusted document influence can propose an action, but it is not a credential that opens the host authorization gate.',
    prediction: 'Can a retrieved export instruction supply its own dispatch authority?', lanes: [{ labels: ['proposal', 'authority'], states: [true, false], detail: 'Untrusted suggested export stops before authorized dispatch.' }],
  },
  'long-context-three-bottlenecks': {
    operation: 'logical-growth', review: 'Show denser physical packing beside a longer logical sequence. Reclaiming gaps does not remove the additional logical pages or attention reads.',
    prediction: 'Does packing away allocation gaps make a longer logical context shorter?',
    caption: 'Illustrative block identities separate physical allocation gaps from logical length. Paging reduces waste, while a longer context still adds logical KV data and attention reads.',
  },
  'loss-spike-trace-taxonomy': {
    operation: 'different-traces', review: 'Keep the training trace flat while the evaluation trace changes. A discrepancy identifies which measurement path to inspect, not a proved optimizer failure.',
    prediction: 'Which evidence would be missing before blaming the stable training update path?',
    caption: 'The illustrative shapes show a stable training trace and an evaluation-only spike, not measured loss magnitudes. Investigate evaluation mode, data, and measurement before inferring training instability.',
  },
  'activation-patching-controlled-contrast': {
    operation: 'replace-one-state', review: 'Only activation A is restored from the clean run. Carry the corrupted other-state identity unchanged into the patched run.',
    prediction: 'Which state must remain corrupted to isolate the declared intervention?', states: [['A clean', 'other clean'], ['A corrupt', 'other corrupt'], ['A clean', 'other corrupt']], changed: 0,
  },
  'microannealing-controlled-fork': {
    operation: 'serial-gates', review: 'Domain improvement and preservation are separate promotion requirements. A failed preservation result stops the probe before adoption.',
    prediction: 'Can improved domain loss waive the preservation failure?', lanes: [{ labels: ['domain gain', 'preserve'], states: [true, false], detail: 'The probe stays unpromoted after its preservation failure.' }],
  },
  'moe-top-two-routing-accounting': {
    operation: 'stored-executed', review: 'Retain all expert-weight identities in the store row. Only E1 and E3 receive this token’s execution routes.',
    prediction: 'Are E2 and E4 removed from memory because this token does not execute them?', active: [true, false, true, false],
  },
  'ml-lineage-two-way-incident-trace': {
    operation: 'independent-evidence', review: 'Lineage establishes possible exposure. Actual consumption and actual harm require separate observed receipts.',
    prediction: 'Does descent from the shard prove the model actually consumed it or was harmed?', labels: ['descendant', 'consumed', 'harmed'], states: ['verified', 'unknown', 'unknown'],
  },
  'model-organism-policy-test': {
    operation: 'independent-evidence', review: 'Retain the known-trigger fix as an observed result while new paraphrases and contexts stay unresolved.',
    prediction: 'Does blocking the known trigger establish safety for untested contexts?', labels: ['known trigger fixed', 'new contexts'], states: ['verified', 'unknown'],
  },
  'ambitious-project-three-evidence-test': {
    operation: 'contribution-boundaries', review: 'Personal decision, shared execution, and team outcome occupy different ownership boundaries. Do not merge them into one personal-result claim.',
    prediction: 'Does owning the design decision establish sole ownership of the team outcome?',
  },
  'overrated-technique-claim-stress-test': {
    operation: 'serial-gates', review: 'A result tests the bounded critique only when task and resource budget match its declared scope.',
    prediction: 'Which contrary result could actually update the matched-task claim?', lanes: [{ labels: ['same task', 'same budget'], states: [true, true], detail: 'The matched result can test and overturn the bounded claim.' }, { labels: ['same task', 'same budget'], states: [false, false], detail: 'Other task and higher budget leave the original comparison unresolved.' }],
  },
  'multi-agent-ctde-information-boundary': {
    operation: 'training-information', review: 'Joint state reaches the training critic, but deployed actors retain only their allowed local observations.',
    prediction: 'Can the deployed actor call the centralized training critic?',
  },
  'multimodal-architectures-meeting-point': {
    operation: 'different-meeting-points', review: 'Move the actual join before or after modality-specific processing. The topology should show when cross-modal interactions become possible.',
    prediction: 'Which architecture keeps image and text processing separate until the final join?',
  },
  'negative-hardness-label-gate': {
    operation: 'serial-gates', review: 'Similarity establishes hardness, while verified irrelevance establishes negative eligibility. A relevant answer stops at the second gate.',
    prediction: 'Can a highly similar relevant answer be a valid hard negative?', lanes: [{ labels: ['similar', 'negative'], states: [true, false], detail: 'Relevant answer: reject as a negative.' }, { labels: ['similar', 'negative'], states: [true, true], detail: 'Verified irrelevant distractor can be a hard negative.' }],
  },
  'training-recipe-evidence-gates': {
    operation: 'broken-update', review: 'Preserve the gap between gradient computation and the optimizer-bound parameter change.',
    prediction: 'What receipt is missing when gradient is 0.8 but weight change is zero?', gradient: .8, change: 0, loss: 1.1,
  },
  'detector-candidate-resolution': {
    operation: 'truth-versus-predictions', review: 'Training assignment receives ground-truth supervision; inference suppression receives only predictions. The operations cannot borrow each other’s information.',
    prediction: 'Does inference NMS require ground-truth boxes?',
  },
  'accelerator-critical-path-trace': {
    operation: 'parallel-finish', review: 'Start both dependencies at the same clock origin. The finish marker follows their maximum, not the sum or a noncritical improvement.',
    prediction: 'Does halving the sixty-unit parallel task shorten the hundred-unit finish?', cases: [[100, 60], [100, 30], [80, 60]],
  },
  'paged-attention-block-table-indirection': {
    operation: 'logical-physical-pages', review: 'Scatter the selected physical page identities but preserve logical sequence order through the explicit mapping.',
    prediction: 'Does physical page two become the first sequence block because its address is smaller?', pages: [7, 2, 9],
  },
  'people-also-bought-three-relations': {
    operation: 'serial-gates', review: 'Co-purchase strength and physical fit are independent relations. High co-buy evidence cannot open a failed compatibility gate.',
    prediction: 'Can the more frequently co-purchased case be eligible if it does not fit?', lanes: [{ labels: ['co-buy', 'fits phone'], states: [true, false], detail: 'High co-buy case A fails fit and is excluded.' }, { labels: ['co-buy', 'fits phone'], states: [true, true], detail: 'Lower co-buy case B is compatible and eligible.' }],
  },
  'personalized-search-request-learning-loop': {
    operation: 'retained-evidence', review: 'Keep B’s candidate identity through retrieval, then show its removal at the cap. The final ranker has no route to that removed object.',
    prediction: 'Can the final ranker score relevant B after the upstream cap removed it?', stages: [['A', 'B'], ['A'], ['A']], labels: ['retrieved', 'after cap', 'final ranker'],
  },
  'pipeline-fill-drain-bubble': {
    operation: 'pipeline-wave', review: 'Trace each microbatch diagonally through three stages. Empty cells are stage capacity inside the same five elapsed slots, not extra elapsed time.',
    prediction: 'Should six idle stage-slots be added to the five-slot wall clock?',
  },
  'plan-70b-parallel-accounting': {
    operation: 'distinct-examples', review: 'Count physical example identities across two data replicas and four sixteen-example microbatches. Tensor and pipeline subdivisions reuse these identities.',
    prediction: 'Does multiplying by TP eight count any additional distinct example?', replicas: 2, batches: 4, batchSize: 16,
  },
  'reward-model-distribution-shift': {
    operation: 'independent-evidence', review: 'Validation belongs to its observed response distribution. The optimized distribution requires new evidence rather than inheriting the old accuracy receipt.',
    prediction: 'Does original-distribution validation certify the newly shifted response region?', labels: ['original response accuracy', 'shifted response accuracy'], states: ['verified', 'unknown'],
  },
  'prefill-decode-weight-reuse': {
    operation: 'one-load-many-computes', review: 'One weight tile fans out to four token computations. Reusing its load does not remove token-specific arithmetic or cache reads.',
    prediction: 'Which per-token work remains after the weight load is shared?',
  },
  'technical-project-decision-spine': {
    operation: 'parallel-finish', review: 'An awaited hundred-unit dependency keeps the response finish fixed while model time shrinks from sixty to thirty.',
    prediction: 'Which dependency remains on the end-to-end critical path?', cases: [[100, 60], [100, 30]],
  },
  'profiling-exposed-collective-tail': {
    operation: 'overlap-tail', review: 'Place forty communication units and thirty useful-compute units on the same clock. Only their uncovered ten-unit tail extends the step.',
    prediction: 'Is the exposed delay forty units or ten?', communication: 40, covered: 30,
  },
  'legal-rag-claim-lineage': {
    operation: 'serial-gates', review: 'Resolve each citation first, then test support for the exact claim. Working links cannot supply absent semantic support.',
    prediction: 'Does a resolvable but irrelevant citation clear exact-claim support?', lanes: [{ labels: ['resolves', 'supports'], states: [true, true], detail: 'Citation A resolves and supports the exact claim.' }, { labels: ['resolves', 'supports'], states: [true, false], detail: 'Citation B resolves but cannot support this claim.' }],
  },
  'rag-index-request-paths': {
    operation: 'versioned-chunk', review: 'Carry the same C7/v3 evidence identity from versioned document through index and request to the released claim record.',
    prediction: 'Does a citation string alone prove the retrieved evidence was C7 version three?',
  },
  'personalization-three-feature-clocks': {
    operation: 'age-versus-latency', review: 'Draw the long snapshot-to-response tether and short request-to-response tether on one clock without confusing their endpoints.',
    prediction: 'Does request latency 0.1 make the ten-unit-old feature snapshot fresh?',
  },
  'recsys-llm-bounded-augmentation': {
    operation: 'serial-gates', review: 'Generated suggestions must satisfy both deadline and catalog membership before displacing the bounded baseline.',
    prediction: 'Which cases retain fallback even when one requirement passes?', lanes: [{ labels: ['deadline', 'catalog'], states: [true, false], detail: 'Timely unknown item: fallback.' }, { labels: ['deadline', 'catalog'], states: [false, true], detail: 'Valid but late item: fallback.' }, { labels: ['deadline', 'catalog'], states: [true, true], detail: 'Timely valid item: eligible.' }],
  },
  'inference-cost-measure-test-compound-loop': {
    operation: 'accepted-baseline', review: 'Only a quality-qualified saving changes the retained baseline. The six-unit proposal remains rejected beside the unchanged eight-unit accepted state.',
    prediction: 'Which cost becomes the baseline after the cheaper proposal fails tail quality?', values: [10, 8, 6], accepted: [true, true, false],
  },
  'fair-comparison-evidence-ledger': {
    operation: 'opportunity-count', review: 'Ten versus one hundred search identities are unequal tuning opportunities. Do not infer matched method quality from the two selected best scores.',
    prediction: 'Which missing matched search would isolate method benefit?', counts: [10, 100], labels: ['Method A: ten trials', 'Method B: one hundred trials'],
  },
  'environment-grader-contract-handoff': {
    operation: 'serial-gates', review: 'A reached goal and a violated allowed-tool process remain separate recorded results in the same run.',
    prediction: 'Can the successful final state waive the prohibited process?', lanes: [{ labels: ['goal', 'allowed process'], states: [true, false], detail: 'The run reaches its goal but fails the declared process contract.' }],
  },
  'rlhf-dpo-training-paths': {
    operation: 'preference-routes', review: 'Both routes start from preference pairs. The direct DPO route bypasses separate reward fitting and policy rollouts rather than pretending those stages failed.',
    prediction: 'Which stages are absent from this direct DPO route?',
  },
  'robotics-policy-dagger-recovery': {
    operation: 'label-visited-state', review: 'Attach the expert action to the learner’s actual drift-state identity, not to the original demonstration state.',
    prediction: 'Which state should receive the corrective expert label after learner drift?',
  },
  'oversight-control-assumption-test': {
    operation: 'independent-evidence', review: 'Detection and containment require independent tests. Preserve the untested cross-claim in each cohort.',
    prediction: 'Does competent detection itself establish adversarial containment?', labels: ['eval test: detection', 'eval test: containment', 'control test: detection', 'control test: containment'], states: ['verified', 'unknown', 'unknown', 'verified'],
  },
  'ambiguous-problem-scope-loop': {
    operation: 'independent-evidence', review: 'The complaint slice establishes observed scope only there. A discriminating sample elsewhere is still missing.',
    prediction: 'What observation would justify widening beyond the complaint slice?', labels: ['complaint slice', 'other slices'], states: ['verified', 'unknown'],
  },
  'speculative-decoding-first-rejection-boundary': {
    operation: 'prefix-correction', review: 'Carry A through commitment, remove rejected B, and invalidate dependent C before inserting a corrected replacement.',
    prediction: 'Can draft C remain committed unchanged after its preceding draft B is rejected?',
  },
  'streaming-asr-provisional-to-final': {
    operation: 'revisable-words', review: 'Keep both earlier hypotheses provisional. Only the explicit final receipt changes the ice-cream tokens into committed output.',
    prediction: 'Can the first visible I scream be treated as final words?',
  },
  'strong-scaling-wall-time-cost-knee': {
    operation: 'device-time-area', review: 'Use actual device count as one side and elapsed hours as the other. The faster sixteen-device run has greater total device-hour area.',
    prediction: 'Which run finishes sooner but uses more device-hours?', cases: [[8, 2], [16, 1.5]],
  },
  'synthetic-data-two-independent-evidence-gates': {
    operation: 'clock-after-boundary', review: 'Place training-row filtering before selection lock and independent held-out confirmation after it.',
    prediction: 'Can reused training-filter feedback serve as fresh final confirmation?', events: [{ value: 0, label: 'training-row filter feedback' }, { value: 1, label: 'selection locked' }, { value: 2, label: 'fresh held-out confirmation' }], boundary: 1,
  },
  'tensor-parallel-ffn-handoff': {
    operation: 'partial-coordinate-sum', review: 'Two channel shards contribute to one output coordinate. Their four and six merge into one ten-valued coordinate rather than a two-coordinate concatenation.',
    prediction: 'Does concatenating four and six reconstruct the complete output coordinate?',
  },
  'test-time-compute-verifier-bottleneck': {
    operation: 'serial-gates', review: 'Separate correct generation from verifier selection for both candidate identities. A correct rejected answer and wrong selected answer both fail the desired joint result.',
    prediction: 'Can additional generation alone repair a verifier that chooses B instead of correct A?', lanes: [{ labels: ['correct', 'selected'], states: [true, false], detail: 'Correct A is discarded.' }, { labels: ['correct', 'selected'], states: [false, true], detail: 'Wrong B is selected; the released result remains wrong.' }],
  },
  'train-100b-parallelism-map': {
    operation: 'partitioned-object', review: 'Mark the object that each parallelism axis partitions. Splitting layers, tensors, or optimizer state does not create distinct examples.',
    prediction: 'Which axis actually partitions distinct input examples?',
  },
  'transformer-training-inference-ledgers': {
    operation: 'stored-executed', review: 'All four stored expert identities remain visible while this token opens only two execution paths.',
    prediction: 'Does top-two execution cut total stored experts to two?', active: [true, false, true, false],
  },
  'two-tower-offline-online-split': {
    operation: 'offline-index-live-query', review: 'Place item encoding before the request boundary and retain the resulting index. The live query reaches that stored index without re-encoding every item.',
    prediction: 'Which encoding happens during this live request?',
  },
  'two-tower-cross-encoder-fanout': {
    operation: 'opportunity-count', review: 'The cross-encoder receives only the twenty retrieved candidate identities, not all one thousand corpus items.',
    prediction: 'Can joint encoding rescue an item absent from the twenty retrieved candidates?', counts: [1000, 20], units: [20, 1], labels: ['Corpus: one mark twenty items', 'Joint scoring: one mark one pair'],
  },
  'vae-training-and-generation-paths': {
    operation: 'different-latent-source', review: 'Training reaches latent z through an encoder of observed x. Generation starts from the prior and cannot borrow an absent observation.',
    prediction: 'Which latent source is used when no observed input exists?',
  },
  'speculative-decoding-weight-read-ledger': {
    operation: 'cost-per-output', review: 'Preserve total speculative work 1.3 while changing the number of accepted output identities sharing it.',
    prediction: 'Why can one accepted token cost more than the one-unit baseline?', total: 1.3, outputs: [2, 1], baseline: 1,
  },
  'cross-validation-fit-the-question': {
    operation: 'split-dependencies', review: 'Show new-user and future-time boundaries on separate rails. A random split cannot claim both protections by default.',
    prediction: 'Which boundary matches future deployment rather than unseen users?',
  },
  'wsd-checkpoint-lineage': {
    operation: 'recoverable-fork', review: 'Keep hot continuation and cooldown evaluation as children of the same committed checkpoint without equating branch evidence with promotion.',
    prediction: 'Does evaluating the cooldown branch automatically replace the hot continuation?', labels: ['hot continuation', 'cooldown branch'],
  },
  'coding-product-mode-boundaries': {
    operation: 'authority-rails', review: 'All modes retain context access, but only the isolated agent’s scoped tool route opens.',
    prediction: 'Does shared context make inline, chat, or review an executing mode?', labels: ['inline', 'chat', 'review', 'agent'], allowed: [false, false, false, true],
  },
  'coding-product-task-lifecycle': {
    operation: 'separate-state-rails', review: 'Cancellation revokes future dispatch before a worker-stop receipt proves termination. Keep the intermediate worker state unresolved.',
    prediction: 'Does withdrawn dispatch authority itself prove the worker has stopped?', states: [['allowed', 'running'], ['revoked', '?'], ['revoked', 'stopped']], actions: ['before', 'cancel', 'receipt'], good: [['revoked'], ['stopped']], footer: 'Revocation precedes termination evidence; neither means patch verification.',
  },
  'foundation-data-build-planes': {
    operation: 'serial-gates', review: 'A completed artifact crosses both permission and validation gates before publication. Either missing requirement blocks the same artifact.',
    prediction: 'Can completed shard files publish before both permission and validation pass?', lanes: [{ labels: ['built', 'policy', 'validation'], states: [true, true, false], detail: 'Built and permitted but invalid: hold.' }, { labels: ['built', 'policy', 'validation'], states: [true, false, true], detail: 'Built and valid but unpermitted: hold.' }, { labels: ['built', 'policy', 'validation'], states: [true, true, true], detail: 'All required artifact gates pass.' }],
  },
  'multimodal-path-boundaries': {
    operation: 'independent-evidence', review: 'Authorization, dispatch, and effect are separately evidenced events. A correlated device receipt establishes effect rather than permission alone.',
    title: 'Permission differs from occurrence', caption: '',
    prediction: 'Does permission establish occurrence?', labels: ['permission', 'dispatch', 'receipt'], states: ['verified', 'verified', 'verified'],
  },
  'multimodal-turn-lifecycle': {
    operation: 'epoch-members', review: 'Carry the active epoch through late-packet rejection. Rejecting seven under active eight cannot retire or rewind the new epoch.',
    title: 'Retired packets stay rejected', footer: 'Late seven cannot rewind eight.',
    caption: '',
    prediction: 'Can retired packets reenter?',
  },
  'video-publication-evidence': {
    operation: 'serial-gates', review: 'Rollback binds model, features, and policy to a compatible versioned bundle. An old model alone cannot open mismatched feature and policy gates.',
    prediction: 'Does restoring model v1 alone authorize feature/policy v2?', lanes: [{ labels: ['model v1', 'features v1', 'policy v1'], states: [true, false, false], detail: 'Unsafe rollback still has features/policy v2.' }, { labels: ['model v1', 'features v1', 'policy v1'], states: [true, true, true], detail: 'Compatible restored v1 bundle passes.' }],
  },
  'infonce-positive-candidate-set': {
    operation: 'positive-in-whole', review: 'Carry the declared positive identity into its own denominator alongside both negative candidates. Semantic relevance cannot silently change the declared labels.',
    prediction: 'Is the declared positive excluded from the denominator candidate set?',
  },
  'fp16-safe-update-order': {
    operation: 'scaled-clipping-order', review: 'Run the same scaled gradient through both orders. The wrong order produces one hundredth of the correct clipped gradient; a nonfinite route remains unable to step.',
    prediction: 'Which order leaves the intended clipped gradient one instead of 0.01?', gradient: 10, scale: 100, limit: 1,
    caption: 'Illustrative gradient ten is scaled by one hundred. Unscale before clipping gives one; clipping the scaled gradient first gives 0.01 after unscaling. A nonfinite result must skip the parameter step.',
  },
  'stairs-final-move-partition': {
    operation: 'last-step-paths', review: 'Enumerate all eight actual five-stair paths and sort by their last move. The five and three groups are disjoint sets, not overlapping recurrence terms.',
    prediction: 'Can any path appear in both final-one and final-two groups?', target: 5,
  },
  'course-prerequisite-fanout': {
    operation: 'prerequisite-counter', review: 'Keep both prerequisite identities present while the unmet counter falls two, one, zero. Only zero opens course three’s ready gate.',
    prediction: 'Is course three ready after only prerequisite one is emitted?',
  },
  'robber-take-skip-boundaries': {
    operation: 'nonadjacent-choice', review: 'Taking house three borrows only the prefix ending before its neighbor. The skipped branch retains the complete earlier eleven-valued witness.',
    prediction: 'Can taking house three add its value to the eleven that already includes neighboring house two?', values: [2, 7, 9, 3],
  },
  'lcs-conditional-parent-cells': {
    operation: 'conditional-diagonal', review: 'Make the match-to-diagonal and mismatch-to-up/left arrows different. The highlighted two must inherit the selected diagonal one plus one.',
    prediction: 'Which predecessor is extended when B matches B?',
  },
  'subset-sum-old-snapshot': {
    operation: 'old-snapshot-extension', review: 'Carry only old zero and one through the current five-valued item. Publish new five and six after extension; never feed new five back through the same item.',
    prediction: 'Why is ten absent when this item five can be used only once?', old: [0, 1], item: 5,
  },
  'dsu-edges-versus-parents': {
    operation: 'graph-versus-parents', review: 'Keep original accepted graph edges fixed above both parent-bookkeeping stages. Only node four’s parent pointer changes to root one.',
    prediction: 'Does redirecting four’s parent add edge four-one to the original problem graph?',
  },
  'subsets-increasing-index-tree': {
    operation: 'ordered-choice-path', review: 'Carry the same subset membership through two orders but open emission only for strictly increasing choice indices.',
    prediction: 'Why is the backward one-then-zero path a duplicate rather than a new subset?', paths: [[0, 1], [1, 0]], strict: true,
  },
};

function eventClock(picture, model, tools) {
  const { text, line, circle, color, fill, number } = tools;
  let body = '';
  if (picture.mode === 'availability') {
    const times = model.events.map(event => event[1]);
    const position = value => 35 + (value - times[0]) / (times[2] - times[0]) * 290;
    assert(times[0] < model.boundary && times[2] > model.boundary);
    const labels = ['Event', 'Prediction', 'Available'];
    body += text(20, 27, 'One record, two different timestamps', { width: 320, weight: 700 })
      + `<rect x="35" y="181" width="290" height="42" style="fill:${fill('warning')};opacity:.7"></rect>`
      + line(35, 202, 325, 202);
    for (const [index, time] of times.entries()) {
      const horizontal = position(time);
      const minutes = Math.round(time * 60);
      const stamp = `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, '0')}`;
      body += text(horizontal, 108, labels[index], { width: 98, anchor: index === 0 ? 'start' : index === 2 ? 'end' : 'middle', weight: 700, tone: index === 1 ? 'warning' : 'input' })
        + circle(horizontal, 202, 7, index === 1 ? 'warning' : 'input', true)
        + text(horizontal, 265, stamp, { width: 100, anchor: index === 0 ? 'start' : index === 2 ? 'end' : 'middle', weight: 700 });
    }
    body += line(position(model.boundary), 136, position(model.boundary), 324, 'warning', true)
      + text(20, 366, 'The noon decision lands before availability', { width: 320, size: 18, weight: 700, tone: 'warning' })
      + text(20, 430, 'An early event is not an available feature.', { width: 320, tone: 'soft' });
    return { body, height: 477 };
  }
  if (picture.mode === 'age') {
    assert.deepEqual(picture.ages, model.rows.map(row => row[1]));
    const position = age => 325 - age / picture.ages[0] * 290;
    body += text(20, 27, 'Delivery does not refresh the captured scene', { width: 320, weight: 700 });
    for (const [stage, age] of picture.ages.entries()) {
      const horizontal = position(age);
      body += `<rect x="${horizontal - 27}" y="122" width="54" height="45" rx="3" style="fill:${fill('input')};stroke:${color('input')};stroke-width:2"></rect>`
        + circle(horizontal - 9, 137, 4, 'input', true)
        + `<path d="M${horizontal - 20} 159l12-11 9 7 8-5 12 9" style="fill:none;stroke:${color('input')};stroke-width:2"></path>`
        + text(stage ? 340 : 20, 94, stage ? 'Arrives' : 'Captured', { width: 120, anchor: stage ? 'end' : 'start', weight: 700, tone: 'input' });
    }
    body += arrowBetween([70, 145], [285, 145], 'input', tools, 2)
      + text(180, 193, 'Same frame', { width: 160, anchor: 'middle', tone: 'input' })
      + line(325, 213, 325, 388, 'warning', true)
      + arrowBetween([35, 271], [325, 271], 'warning', tools, 3)
      + text(20, 252, 'Content age: 4.30 s', { width: 290, weight: 700, tone: 'warning' })
      + line(position(picture.ages[1]), 338, 325, 338, 'output')
      + circle(position(picture.ages[1]), 338, 3, 'output', true)
      + text(20, 366, 'Arrival age: 0.10 s', { width: 290, weight: 700, tone: 'output' })
      + text(340, 423, 'Question time', { width: 170, anchor: 'end', weight: 700, tone: 'warning' });
    return { body, height: 459 };
  }
  assert.equal(picture.mode, 'delayed-effect');
  const position = index => 35 + index * 96;
  assert.deepEqual(model.events.map(event => event[1]), [0, 1, 2, 3]);
  body += text(20, 27, 'Fast feedback and slow response share a clock', { width: 320, weight: 700 })
    + text(position(2), 80, 'Short test ends', { width: 110, anchor: 'middle', weight: 700, tone: 'warning' })
    + line(position(2), 115, position(2), 365, 'warning', true)
    + text(20, 126, 'Viewer response', { width: 170, tone: 'input', weight: 700 })
    + text(20, 292, 'Creator response', { width: 170, tone: 'output', weight: 700 })
    + circle(position(0), 233, 8, 'focus', true)
    + arrowBetween([position(0), 233], [position(1), 166], 'input', tools, 2)
    + arrowBetween([position(0), 233], [position(3), 330], 'output', tools, 2)
    + circle(position(1), 166, 9, 'input', true) + circle(position(3), 330, 9, 'output', true)
    + text(20, 221, 'Policy changes', { width: 150, tone: 'focus' })
    + text(20, 415, 'The early stop misses the later response', { width: 320, size: 18, weight: 700 })
    + text(20, 472, 'Event order only; no duration estimate is implied.', { width: 320, tone: 'soft' });
  return { body, height: 519 };
}

function policyBands(picture, model, tools) {
  const { text, line, circle, color, fill, number } = tools;
  assert.deepEqual(picture.policies[0], [.2, .8]);
  const names = ['allow', 'review', 'block'];
  const tones = ['output', 'focus', 'warning'];
  let body = text(20, 27, 'The score stays fixed. The policy changes.', { width: 320, weight: 700 });
  for (const [phase, boundaries] of picture.policies.entries()) {
    const vertical = 130 + phase * 202;
    const ends = [0, ...boundaries, 1];
    const selected = picture.score < boundaries[0] ? 0 : picture.score < boundaries[1] ? 1 : 2;
    body += text(20, vertical - 50, `Illustrative policy ${phase ? 'B' : 'A'}`, { width: 320, weight: 700 });
    for (let region = 0; region < 3; region++) {
      body += `<rect data-picture-policy="${phase},${region}" x="${20 + ends[region] * 320}" y="${vertical}" width="${(ends[region + 1] - ends[region]) * 320}" height="33" style="fill:${fill(tones[region])};stroke:${color(tones[region])};stroke-width:1.75"></rect>`;
    }
    const horizontal = 20 + picture.score * 320;
    body += line(horizontal, vertical - 22, horizontal, vertical + 49, tones[selected])
      + circle(horizontal, vertical - 22, 6, tones[selected], true)
      + text(20, vertical + 88, `${number(picture.score)}: ${names[selected]}`, { width: 320, size: 22, weight: 700, tone: tones[selected] });
  }
  body += text(20, 477, 'Green: allow. Brown: review. Red: block.', { width: 320, tone: 'soft' });
  return { body, height: 521 };
}

function budgetTransfer(picture, model, tools) {
  const { text, circle, number } = tools;
  const before = model.rows.map(row => row[1]);
  const after = before.map((value, index) => value + (index === model.transfer.to ? model.transfer.value : index === model.transfer.from ? -model.transfer.value : 0));
  assert.equal(before.reduce((sum, value) => sum + value, 0), 100);
  assert.equal(after.reduce((sum, value) => sum + value, 0), 100);
  for (const index of model.transfer.protected) assert.equal(before[index], after[index]);
  const tones = model.rows.map(row => row[2]);
  let body = text(20, 27, 'The same 100 budget points before and after', { width: 320, weight: 700 });
  for (const [phase, values] of [before, after].entries()) {
    const horizontal = 20 + phase * 180;
    const classify = index => {
      let total = 0;
      const group = values.findIndex(value => { total += value; return index < total; });
      return { tone: tones[group] };
    };
    body += text(horizontal, 77, phase ? 'After' : 'Before', { width: 140, weight: 700 })
      + dotField(100, 10, horizontal + 6, 105, 13.5, classify, tools, `budget-${phase}`);
    for (let index = 50; index < 56; index++) body += circle(horizontal + 6 + index % 10 * 13.5, 105 + Math.floor(index / 10) * 13.5, 6, 'focus');
  }
  body += arrowBetween([95, 277], [265, 277], 'focus', tools, 2)
    + text(180, 316, 'Reassign six circled points', { width: 320, anchor: 'middle', weight: 700, tone: 'focus' });
  for (const [index, row] of model.rows.entries()) {
    const vertical = 365 + index * 43;
    body += text(20, vertical, row[0], { width: 160, weight: 700, tone: row[2] })
      + text(340, vertical, model.transfer.protected.includes(index) ? `${before[index]}; protected` : `${before[index]} to ${after[index]}`, { width: 155, anchor: 'end', tone: row[2] });
  }
  return { body, height: 525 };
}

function decayBypass(picture, model, tools) {
  const { text, line, circle, color, fill, number } = tools;
  const coupled = picture.scales.map(scale => picture.base / scale);
  assert.deepEqual([...coupled, picture.base], model.rows.map(row => row[1]));
  let body = text(20, 27, 'Shrinkage: coupled or direct', { width: 320, weight: 700 })
    + text(20, 54, 'Same weights; frozen preconditioners', { width: 320, tone: 'soft' });
  for (const [lane, scale] of picture.scales.entries()) {
    const vertical = 100 + lane * 189;
    body += text(20, vertical - 24, `Frozen gradient scale: ${scale}`, { width: 320, weight: 700 })
      + circle(48, vertical + 31, 21, 'input') + text(48, vertical + 36, 'w', { width: 36, anchor: 'middle', weight: 700 })
      + `<path d="M128 ${vertical + 8}h70l18 23-18 23h-70l-18-23Z" style="fill:${fill('focus')};stroke:${color('focus')};stroke-width:1.75"></path>`
      + text(163, vertical + 36, `/ ${scale}`, { width: 70, anchor: 'middle', weight: 700, tone: 'focus' })
      + arrowBetween([70, vertical + 31], [108, vertical + 31], 'input', tools, 2)
      + arrowBetween([216, vertical + 31], [260, vertical + 31], 'input', tools, 2)
      + text(340, vertical + 36, number(coupled[lane]), { width: 95, anchor: 'end', weight: 700, tone: 'input' })
      + `<path d="M48 ${vertical + 54}V${vertical + 101}H291" style="fill:none;stroke:${color('output')};stroke-width:2.5"></path>`
      + text(340, vertical + 107, number(picture.base), { width: 95, anchor: 'end', weight: 700, tone: 'output' });
  }
  body += text(20, 491, 'Blue: coupled through the gradient scale', { width: 320, tone: 'input', weight: 700 })
    + text(20, 536, 'Green: direct decay bypasses that scale', { width: 320, tone: 'output', weight: 700 });
  return { body, height: 580 };
}

function arrowBetween(from, to, tone, tools, width = 2) {
  const horizontal = to[0] - from[0];
  const vertical = to[1] - from[1];
  const length = Math.hypot(horizontal, vertical);
  if (!length) return tools.circle(...to, 5, tone, true);
  const along = [horizontal / length, vertical / length];
  const cross = [-along[1], along[0]];
  const wing = (side) => [to[0] - along[0] * 8 + cross[0] * 4 * side, to[1] - along[1] * 8 + cross[1] * 4 * side];
  return `<path d="M${from.join(' ')}L${to.join(' ')}M${wing(1).join(' ')}L${to.join(' ')}L${wing(-1).join(' ')}" style="fill:none;stroke:${tools.color(tone)};stroke-width:${width};stroke-linejoin:round"></path>`;
}

function geometricAction(picture, model, tools) {
  const { text, line, circle, color, fill, number } = tools;
  const mode = picture.mode;
  const minimum = mode === 'centered-products' ? -2 : model.domain?.[0] ?? 0;
  const maximum = mode === 'centered-products' ? 2 : mode === 'stretch' ? 7 : model.domain?.[1] ?? 5;
  const unit = 288 / (maximum - minimum);
  const point = ([horizontal, vertical]) => [34 + (horizontal - minimum) * unit, 330 - (vertical - minimum) * unit];
  const polygon = (points, tone) => `<path d="${points.map((value, index) => `${index ? 'L' : 'M'}${point(value).join(' ')}`).join('')}Z" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.75"></path>`;
  const mark = (value, tone, radius = 6) => circle(...point(value), radius, tone, true);
  let body = '';
  if (!['orientation', 'normalize', 'mixture-spread'].includes(mode)) {
    body += `<g opacity=".22">`;
    for (let coordinate = Math.ceil(minimum); coordinate <= maximum; coordinate++) {
      body += line(...point([coordinate, minimum]), ...point([coordinate, maximum])) + line(...point([minimum, coordinate]), ...point([maximum, coordinate]));
    }
    body += '</g>' + line(...point([minimum, 0]), ...point([maximum, 0])) + line(...point([0, minimum]), ...point([0, maximum]));
  }
  if (mode === 'projection') {
    const observed = model.vectors[0][1];
    const fitted = model.vectors[1][1];
    const residual = observed.map((value, index) => value - fitted[index]);
    assert.equal(residual[0], 0);
    assert(Math.hypot(...observed.map((value, index) => value - picture.competitor[index])) > Math.hypot(...residual));
    body += `<path d="M${point([0, 0]).join(' ')}L${point([maximum, 0]).join(' ')}" style="stroke:${color('output')};stroke-width:14;opacity:.2"></path>`
      + line(...point(picture.competitor), ...point(observed), 'neutral', true)
      + arrowBetween(point(fitted), point(observed), 'warning', tools, 3)
      + mark(picture.competitor, 'neutral') + mark(observed, 'input', 8) + mark(fitted, 'output', 8);
    const [footX, footY] = point(fitted);
    body += `<path d="M${footX - 11} ${footY}v-11h11" style="fill:none;stroke:${color('output')};stroke-width:2"></path>`
      + text(20, 27, 'Allowed fits lie on the green rail', { width: 320, weight: 700 })
      + text(340, point(observed)[1] - 23, 'Observed (3,2)', { width: 190, anchor: 'end', weight: 700, tone: 'input' })
      + text(20, 385, 'Other tethers are longer', { width: 320, weight: 700, tone: 'soft' })
      + text(20, 431, 'Shortest gap: perpendicular to the rail', { width: 320, size: 18, weight: 700, tone: 'output' });
    return { body, height: 479 };
  }
  if (mode === 'clip') {
    const [original, clipped] = model.vectors.map(vector => vector[1]);
    const scale = picture.radius / Math.hypot(...original);
    assert(clipped.every((value, index) => Math.abs(value - original[index] * scale) < 1e-12));
    const origin = point([0, 0]);
    for (const [radius, tone] of [[5, 'neutral'], [picture.radius, 'output']]) body += `<path d="M${origin[0] + radius * unit} ${origin[1]}A${radius * unit} ${radius * unit} 0 0 0 ${origin[0]} ${origin[1] - radius * unit}" style="fill:none;stroke:${color(tone)};stroke-width:2;stroke-dasharray:4 4"></path>`;
    body += arrowBetween(origin, point(original), 'input', tools, 2)
      + arrowBetween(point(original), point(clipped), 'warning', tools, 3)
      + arrowBetween(origin, point(clipped), 'output', tools, 3)
      + mark(original, 'input') + mark(clipped, 'output')
      + text(20, 27, 'Pull inward along the same ray', { width: 320, weight: 700 })
      + text(20, 391, '(3,4) becomes (1.2,1.6)', { width: 320, size: 20, weight: 700 })
      + text(20, 438, 'Length 5 becomes 2. Direction stays.', { width: 320, weight: 700, tone: 'output' });
    return { body, height: 470 };
  }
  if (mode === 'threat') {
    const [original, perturbed] = model.vectors.map(vector => vector[1]);
    assert(Math.hypot(...original.map((value, index) => value - perturbed[index])) <= picture.allowance + 1e-12);
    const boundary = point([0, 0])[0];
    body += `<rect x="34" y="42" width="${boundary - 34}" height="288" style="fill:${fill('warning')};opacity:.65"></rect><rect x="${boundary}" y="42" width="${322 - boundary}" height="288" style="fill:${fill('output')};opacity:.65"></rect>`
      + line(...point([0, minimum]), ...point([0, maximum]), 'warning', true)
      + line(...point([original[0] - picture.allowance, original[1] - .65]), ...point([original[0] + picture.allowance, original[1] - .65]), 'focus')
      + arrowBetween(point(original), point(perturbed), 'warning', tools, 3) + mark(original, 'input', 6) + mark(perturbed, 'warning', 6)
      + text(20, 27, 'An allowed move can cross the boundary', { width: 320, weight: 700 })
      + line(...point(original), 260, 100, 'input') + text(340, 85, 'Original', { width: 125, anchor: 'end', tone: 'input', weight: 700 })
      + line(...point(perturbed), 75, 235, 'warning') + text(20, 263, 'Perturbed', { width: 125, tone: 'warning', weight: 700 })
      + text(20, 391, `Horizontal allowance: ${picture.allowance}`, { width: 320, weight: 700 })
      + text(20, 438, 'Small is defined by the declared threat set.', { width: 320, tone: 'soft' });
    return { body, height: 480 };
  }
  if (mode === 'spread') {
    const mean = [0, 1].map(coordinate => picture.outcomes.reduce((sum, outcome) => sum + outcome[coordinate], 0) / picture.outcomes.length);
    const truth = model.vectors[0][1];
    const variance = picture.outcomes.reduce((sum, outcome) => sum + outcome.reduce((error, value, index) => error + (value - mean[index]) ** 2, 0), 0) / picture.outcomes.length;
    assert.deepEqual(mean, model.vectors[1][1]);
    assert.equal(variance, 2);
    for (const outcome of picture.outcomes) body += line(...point(mean), ...point(outcome), 'neutral', true) + mark(outcome, 'neutral', 6);
    body += arrowBetween(point(truth), point(mean), 'warning', tools, 3) + mark(truth, 'input', 8) + mark(mean, 'warning', 8)
      + text(20, 27, 'Repeat the estimator, not the truth', { width: 320, weight: 700 })
      + text(20, 91, 'Gray points: four possible estimates', { width: 320, tone: 'soft' })
      + text(20, 378, 'Truth to mean: bias', { width: 320, weight: 700, tone: 'warning' })
      + text(20, 420, 'Mean to outcomes: spread', { width: 320, weight: 700, tone: 'soft' })
      + text(20, 465, 'Mean squared error: 1 + 2 = 3', { width: 320, size: 18, weight: 700 });
    return { body, height: 501 };
  }
  if (mode === 'centered-products') {
    for (const horizontal of [-1, 1]) for (const vertical of [-1, 1]) {
      const product = horizontal * vertical;
      const tone = product > 0 ? 'output' : 'warning';
      body += circle(...point([horizontal, vertical]), 22, tone) + text(point([horizontal, vertical])[0], point([horizontal, vertical])[1] + 6, product > 0 ? '+1' : '-1', { width: 40, anchor: 'middle', weight: 700, tone });
    }
    body += mark([0, 1], 'neutral', 7)
      + text(20, 27, 'Center first. Then multiply deviations.', { width: 320, weight: 700 })
      + text(20, 381, 'Same side: positive contribution', { width: 320, tone: 'output', weight: 700 })
      + text(20, 423, 'Opposite sides: negative contribution', { width: 320, tone: 'warning', weight: 700 })
      + text(20, 465, 'On a centered axis: zero contribution', { width: 320, tone: 'soft' });
    return { body, height: 501 };
  }
  if (mode === 'stretch') {
    const result = picture.input.map((value, index) => value * picture.gains[index]);
    assert.deepEqual(result, model.rows.at(-1)[1].map(Number));
    const corner = [result[0], picture.input[1]];
    body += arrowBetween(point([0, 0]), point(picture.input), 'input', tools, 2)
      + arrowBetween(point(picture.input), point(corner), 'focus', tools, 3)
      + arrowBetween(point(corner), point(result), 'output', tools, 3)
      + mark(picture.input, 'input') + mark(result, 'output')
      + text(20, 27, 'In the eigenbasis, each axis scales alone', { width: 320, weight: 700 })
      + text(20, 382, 'First coordinate: 3 x 2 = 6', { width: 320, weight: 700, tone: 'focus' })
      + text(20, 424, 'Second coordinate: 1 x 3 = 3', { width: 320, weight: 700, tone: 'output' })
      + text(20, 470, 'This diagonal example already uses that basis.', { width: 320, tone: 'soft' });
    return { body, height: 512 };
  }
  if (mode === 'cancel') {
    const [first, second, result] = model.vectors.map(vector => vector[1]);
    assert.deepEqual(first.map((value, index) => value + second[index]), result);
    body += arrowBetween(point([0, 0]), point(first), 'input', tools, 4)
      + arrowBetween(point(first), point(result), 'warning', tools, 2)
      + circle(...point([0, 0]), 11, 'output')
      + text(20, 27, 'A nonzero input can travel out and back', { width: 320, weight: 700 })
      + text(20, 387, 'One column contribution cancels the other', { width: 320, weight: 700 })
      + text(20, 443, '(1,-1) in. Zero out.', { width: 320, size: 22, weight: 700, tone: 'output' });
    return { body, height: 479 };
  }
  if (mode === 'orientation') {
    const [first, second] = model.vectors.map(vector => vector[1]);
    const determinant = first[0] * second[1] - first[1] * second[0];
    assert.equal(determinant, 4);
    body += text(20, 27, 'Same floor. Opposite traversal order.', { width: 320, weight: 700 });
    for (const order of [0, 1]) {
      const origin = [32 + order * 180, 236];
      const local = ([horizontal, vertical]) => [origin[0] + horizontal * 36, origin[1] - vertical * 36];
      const vertices = [[0, 0], order ? second : first, first.map((value, index) => value + second[index]), order ? first : second];
      body += `<path d="${vertices.map((vertex, index) => `${index ? 'L' : 'M'}${local(vertex).join(' ')}`).join('')}Z" style="fill:${fill('output')};stroke:${color('output')};stroke-width:1.75"></path>`;
      for (let index = 0; index < vertices.length; index++) {
        const from = local(vertices[index]);
        const to = local(vertices[(index + 1) % vertices.length]);
        const middle = from.map((value, coordinate) => value * .35 + to[coordinate] * .65);
        body += arrowBetween(from, middle, order ? 'warning' : 'input', tools, 2);
      }
      body += text(origin[0] + 54, 300, `det ${order ? '-' : '+'}${determinant}`, { width: 140, anchor: 'middle', size: 24, weight: 700, tone: order ? 'warning' : 'input' });
    }
    body += text(20, 371, 'Both polygons contain four square units.', { width: 320, weight: 700 })
      + text(20, 423, 'Order changes orientation, not unsigned area.', { width: 320, tone: 'soft' });
    return { body, height: 470 };
  }
  if (mode === 'mixture-spread') {
    const horizontal = value => 35 + value * 58;
    const center = picture.means.reduce((sum, value) => sum + value, 0) / picture.means.length;
    const withinVariance = picture.within ** 2;
    const betweenVariance = picture.means.reduce((sum, value) => sum + (value - center) ** 2, 0) / picture.means.length;
    assert(Math.abs(withinVariance + betweenVariance - 2.41) < 1e-12);
    body += text(20, 27, 'Narrow models can disagree widely', { width: 320, weight: 700 }) + line(35, 182, 325, 182);
    for (const [index, mean] of picture.means.entries()) {
      const tone = index ? 'warning' : 'input';
      body += line(horizontal(mean - picture.within), 182, horizontal(mean + picture.within), 182, tone)
        + circle(horizontal(mean - picture.within), 182, 6, tone, true) + circle(horizontal(mean + picture.within), 182, 6, tone, true)
        + circle(horizontal(mean), 120, 7, tone, true) + line(horizontal(mean), 130, horizontal(mean), 172, tone, true)
        + text(horizontal(mean), 91, `Model ${index ? 'B' : 'A'}`, { width: 120, anchor: 'middle', weight: 700, tone });
      body += arrowBetween([horizontal(center), 306], [horizontal(mean), 227], tone, tools, 2);
    }
    body += circle(horizontal(center), 306, 8, 'output', true)
      + text(180, 355, 'Mixture mean: 2.5', { width: 300, anchor: 'middle', weight: 700, tone: 'output' })
      + text(20, 411, 'Within: 0.16. Between: 2.25.', { width: 320, weight: 700 })
      + text(20, 460, 'Both sources contribute to uncertainty.', { width: 320, tone: 'soft' });
    return { body, height: 501 };
  }
  assert.equal(mode, 'normalize');
  body += text(20, 27, 'Remove length. Keep direction.', { width: 320, weight: 700 });
  const center = [110, 272];
  const radius = 125;
  body += `<path d="M${center[0] + radius} ${center[1]}A${radius} ${radius} 0 0 0 ${center[0]} ${center[1] - radius}" style="fill:none;stroke:${color('neutral')};stroke-width:2;stroke-dasharray:4 4"></path>`
    + line(center[0], center[1], 322, center[1]);
  for (const [index, candidate] of picture.candidates.entries()) {
    const normalized = candidate.map(value => value / Math.hypot(...candidate));
    const end = [center[0] + normalized[0] * radius, center[1] - normalized[1] * radius];
    const tone = index ? 'output' : 'input';
    body += arrowBetween(center, end, tone, tools, 3) + circle(...end, 7, tone, true)
      + line(end[0], end[1], end[0], center[1] + 18, tone, true)
      + text(20, 357 + index * 62, `${index ? 'B' : 'A'}: raw dot ${candidate[0]}; cosine ${number(normalized[0])}`, { width: 320, weight: 700, tone });
  }
  body += text(20, 481, 'The diagonal loses its length advantage.', { width: 320, tone: 'soft' });
  return { body, height: 523 };
}

function dotField(count, columns, horizontal, vertical, spacing, classify, tools, stage = '') {
  let body = '';
  for (let index = 0; index < count; index++) {
    const { tone = 'neutral', active = true, square = false } = classify(index);
    const centerX = horizontal + index % columns * spacing;
    const centerY = vertical + Math.floor(index / columns) * spacing;
    const radius = Math.min(4, spacing * .29);
    const shape = square ? `<rect x="${centerX - radius}" y="${centerY - radius}" width="${2 * radius}" height="${2 * radius}" style="fill:${tools.color(tone)};stroke:${tools.color(tone)}"></rect>` : tools.circle(centerX, centerY, radius, tone, true);
    body += `<g data-picture-member="${stage},${index}" data-picture-active="${active}" opacity="${active ? 1 : .16}">${shape}</g>`;
  }
  return body;
}

function biasedSample(picture, model, tools) {
  const { text, line, circle } = tools;
  assert.deepEqual(picture.counts, model.rows.map(row => row[1]));
  const total = picture.counts.reduce((sum, value) => sum + value, 0);
  let body = text(20, 26, 'The deployment world stays imbalanced', { width: 320, weight: 700 });
  body += dotField(total, 50, 23.2, 67, 6.4, index => ({ tone: index < picture.counts[0] ? 'output' : 'input', square: index >= picture.counts[0] }), tools, 'world');
  body += text(20, 222, '10 positive; 990 negative', { width: 320, weight: 700 })
    + text(20, 264, 'Choose a balanced batch', { width: 320, size: 18, weight: 700 })
    + line(90, 280, 90, 311, 'output') + line(270, 280, 270, 311, 'input');
  for (const [cohort, count] of picture.batch.entries()) {
    assert(count <= picture.counts[cohort]);
    body += dotField(count, 5, 40 + cohort * 180, 335, 24, () => ({ tone: cohort ? 'input' : 'output', square: Boolean(cohort) }), tools, `batch-${cohort}`)
      + text(90 + cohort * 180, 407, `${count} ${cohort ? 'negative' : 'positive'}`, { width: 145, anchor: 'middle', weight: 700, tone: cohort ? 'input' : 'output' });
  }
  body += circle(30, 452, 7, 'neutral') + text(48, 457, 'Sampling changes the batch, not the world', { width: 292, tone: 'soft' });
  return { body, height: 497 };
}

function selectDenominator(picture, model, tools) {
  const { text, color, fill, number } = tools;
  const [[hits, misses], [alarms, correctNegatives]] = model.values;
  assert([hits, misses, alarms, correctNegatives].every(Number.isInteger));
  const total = hits + misses + alarms + correctNegatives;
  const columns = Math.ceil(Math.sqrt(total));
  const spacing = Math.min(13, 135 / columns);
  const denominators = [hits + misses, hits + alarms];
  let body = text(20, 26, 'Same cases. Same hits. Different groups.', { width: 320, weight: 700 });
  for (const [selection, denominator] of denominators.entries()) {
    const horizontal = 20 + selection * 180;
    body += text(horizontal, 69, selection ? 'Predicted +' : 'Actual +', { width: 145, weight: 700 });
    body += dotField(total, columns, horizontal + 6, 96, spacing, index => {
      const actualPositive = index < hits + misses;
      const predictedPositive = index < hits || index >= hits + misses && index < hits + misses + alarms;
      return { tone: index < hits ? 'output' : 'warning', active: selection ? predictedPositive : actualPositive, square: !actualPositive };
    }, tools, `denominator-${selection}`);
    const vertical = 121 + Math.ceil(total / columns) * spacing;
    body += text(horizontal, vertical, `${hits} / ${denominator}`, { width: 145, size: 26, weight: 700, tone: 'output' })
      + `<rect x="${horizontal}" y="${vertical + 20}" width="140" height="24" style="fill:${fill('neutral')};stroke:${color('neutral')}"></rect><rect data-picture-share="${selection}" data-picture-numerator="${hits}" data-picture-denominator="${denominator}" x="${horizontal}" y="${vertical + 20}" width="${140 * hits / denominator}" height="24" style="fill:${color('output')}"></rect>`
      + text(horizontal, vertical + 76, `${number(100 * hits / denominator)}%`, { width: 145, size: 22, weight: 700 });
  }
  const bottom = 258 + Math.ceil(total / columns) * spacing;
  body += text(20, bottom, `${hits} green hits survive both selections`, { width: 320, weight: 700, tone: 'output' })
    + text(20, bottom + 45, 'Changing the denominator changes the question', { width: 320, tone: 'soft' });
  return { body, height: bottom + 91 };
}

function forecastCount(picture, model, tools) {
  const { text, color, fill, line } = tools;
  assert.equal(picture.observed, model.rows[0][1]);
  assert.equal(picture.total, model.rows.reduce((sum, row) => sum + row[1], 0));
  let body = text(20, 27, 'Every case received the same 80% forecast', { width: 320, weight: 700 });
  for (const [index, count] of [picture.expected, picture.observed].entries()) {
    const vertical = 92 + index * 149;
    body += text(20, vertical, index ? 'Observed positive count' : 'Forecast expected count', { width: 320, weight: 700 })
      + `<rect x="20" y="${vertical + 22}" width="320" height="48" style="fill:${fill('neutral')};stroke:${color('neutral')}"></rect>`;
    for (let unit = 0; unit < count; unit++) body += `<rect data-picture-calibration="${index}" x="${20 + unit % 20 * 16}" y="${vertical + 22 + Math.floor(unit / 20) * 9.6}" width="16" height="9.6" style="fill:${index ? color('output') : fill('input')};stroke:${color(index ? 'output' : 'input')};stroke-width:.7"></rect>`;
    body += text(340, vertical + 105, `${count} of ${picture.total}`, { width: 170, anchor: 'end', size: 22, weight: 700, tone: index ? 'output' : 'input' });
  }
  body += line(20 + picture.observed / picture.total * 320, 379, 20 + picture.expected / picture.total * 320, 379, 'warning')
    + text(20, 418, '20 too high in this aggregate', { width: 320, size: 20, weight: 700, tone: 'warning' })
    + text(20, 463, 'Expected counts do not identify individual outcomes', { width: 320, tone: 'soft' });
  return { body, height: 510 };
}

function loggedSupport(picture, model, tools) {
  const { text, circle, line, color, fill } = tools;
  assert.equal(picture.probabilities.reduce((sum, value) => sum + value, 0), 1);
  assert.equal(picture.probabilities[2], 0);
  let body = text(20, 27, 'The logger decides which paths exist', { width: 320, weight: 700 });
  for (const [index, probability] of picture.probabilities.entries()) {
    const horizontal = 65 + index * 115;
    const tone = probability ? 'input' : 'warning';
    body += circle(horizontal, 104, 24, tone)
      + text(horizontal, 110, picture.actions[index], { width: 44, anchor: 'middle', weight: 700, size: 20 })
      + text(horizontal, 164, probability, { width: 95, anchor: 'middle', weight: 700, tone });
    if (probability) body += `<path data-picture-support="${index}" d="M${horizontal} 180V278" style="fill:none;stroke:${color(tone)};stroke-width:${probability * 30}"></path>`;
    else body += line(horizontal, 180, horizontal, 211, tone, true) + line(horizontal - 9, 218, horizontal + 9, 236, tone) + line(horizontal + 9, 218, horizontal - 9, 236, tone) + line(horizontal, 243, horizontal, 278, tone, true);
    body += `<rect x="${horizontal - 45}" y="290" width="90" height="54" rx="4" style="fill:${fill(tone)};stroke:${color(tone)}"></rect>`
      + text(horizontal, 322, probability ? 'seen' : '?', { width: 80, anchor: 'middle', size: 20, weight: 700, tone });
  }
  body += text(20, 398, 'No path means no observed reward', { width: 320, weight: 700 })
    + text(20, 443, 'Unknown is not zero', { width: 320, size: 22, weight: 700, tone: 'warning' });
  return { body, height: 471 };
}

function replaceSlot(picture, model, tools) {
  const { text, color, fill, line, circle } = tools;
  assert.deepEqual(model.rows.map(row => row[1]), [['A', 'B'], ['A', 'free'], ['A', 'C']]);
  let body = text(20, 26, 'A keeps its slot while B leaves', { width: 320, weight: 700 });
  for (const [phase, names] of [['A', 'B'], ['A', 'C']].entries()) {
    const vertical = phase ? 314 : 88;
    body += text(phase ? 130 : 20, vertical - 20, phase ? 'Next iteration' : 'This iteration', { width: phase ? 200 : 320, tone: 'soft' });
    for (const [slot, name] of names.entries()) {
      const horizontal = 25 + slot * 185;
      const tone = slot ? phase ? 'output' : 'focus' : 'input';
      body += `<rect data-picture-slot="${phase},${slot}" data-picture-job="${name}" x="${horizontal}" y="${vertical}" width="125" height="59" rx="4" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`
        + text(horizontal + 62.5, vertical + 38, name, { width: 110, anchor: 'middle', size: 26, weight: 700, tone });
    }
  }
  body += line(87.5, 159, 87.5, 297, 'input') + line(82.5, 289, 87.5, 297, 'input') + line(92.5, 289, 87.5, 297, 'input')
    + text(111, 218, 'still running', { width: 88, tone: 'input' })
    + `<path d="M272.5 159Q272.5 192 320 210" style="fill:none;stroke:${color('focus')};stroke-width:2"></path>`
    + circle(320, 210, 18, 'focus') + text(320, 215, 'B', { width: 30, anchor: 'middle', weight: 700, tone: 'focus' })
    + text(245, 184, 'done', { width: 85, tone: 'focus' })
    + circle(218, 237, 18, 'output') + text(218, 242, 'C', { width: 30, anchor: 'middle', weight: 700, tone: 'output' })
    + line(230, 251, 272.5, 297, 'output') + line(265, 294, 272.5, 297, 'output') + line(272.5, 289, 272.5, 297, 'output')
    + text(20, 421, 'A stays. The freed slot receives C.', { width: 320, weight: 700 });
  return { body, height: 449 };
}

function nestedCohorts(picture, model, tools) {
  const { text, line } = tools;
  const whole = picture.counts[0] / picture.unit;
  assert.equal(whole, 100);
  assert(picture.counts.every((count, index) => Number.isInteger(count / picture.unit) && (!index || count <= picture.counts[index - 1])));
  if (model.kind === 'bars') assert.deepEqual(picture.counts, model.rows.map(row => row[1]));
  else assert.deepEqual(picture.counts, [model.rows.reduce((sum, row) => sum + row[1], 0), model.rows[0][1] + model.rows[1][1], model.rows[0][1]]);
  let body = text(20, 27, `Each mark: ${picture.unit} candidate${picture.unit === 1 ? '' : 's'}`, { width: 320, weight: 700 });
  for (const [stage, count] of picture.counts.entries()) {
    const horizontal = 20 + stage * 115;
    const tone = ['input', 'focus', 'output'][stage];
    body += text(horizontal, 75, picture.labels[stage], { width: 92, weight: 700, tone })
      + dotField(whole, 10, horizontal + 4, 122, 9.2, index => ({ active: index < count / picture.unit, tone }), tools, `filter-${stage}`)
      + text(horizontal + 45, 261, count, { width: 100, anchor: 'middle', size: 24, weight: 700, tone });
    if (stage < 2) body += line(horizontal + 46, 291, horizontal + 140, 291, tone) + line(horizontal + 132, 286, horizontal + 140, 291, tone) + line(horizontal + 132, 296, horizontal + 140, 291, tone);
  }
  body += text(20, 348, picture.ending, { width: 320, size: 18, weight: 700 })
    + text(20, 405, 'Faded candidates were filtered out, not relabeled.', { width: 320, tone: 'soft' });
  return { body, height: 451 };
}

function boundedCapacity(picture, model, tools) {
  const { text, line, color, fill, number, circle } = tools;
  let body = '';
  if (picture.mode === 'accumulate') {
    const total = picture.committed + picture.proposed;
    assert.equal(total, model.rows[1][1]);
    assert.equal(picture.capacity, model.limit);
    const scale = 320 / total;
    body += text(20, 27, 'A proposal enters an already-used budget', { width: 320, weight: 700 });
    for (const [index, value] of [picture.committed, picture.proposed].entries()) {
      const horizontal = index ? 196 : 20;
      const tone = index ? 'focus' : 'input';
      body += text(horizontal, 84, index ? 'Proposed' : 'Committed', { width: 145, weight: 700, tone })
        + `<rect x="${horizontal}" y="110" width="${value * scale}" height="36" style="fill:${fill(tone)};stroke:${color(tone)}"></rect>`
        + text(horizontal, 181, value, { width: 145, size: 26, weight: 700, tone });
    }
    body += text(20, 237, 'Try to fit both', { width: 320, size: 18, weight: 700 });
    const segments = [[picture.committed, 'input'], [picture.capacity - picture.committed, 'focus'], [total - picture.capacity, 'warning']];
    let horizontal = 20;
    for (const [value, tone] of segments) {
      body += `<rect data-picture-allocation="${tone}" data-picture-amount="${value}" x="${horizontal}" y="270" width="${value * scale}" height="48" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`;
      horizontal += value * scale;
    }
    body += line(20 + picture.capacity * scale, 253, 20 + picture.capacity * scale, 340, 'warning')
      + text(20, 372, `${picture.committed} + ${picture.proposed} = ${total}`, { width: 320, size: 24, weight: 700 })
      + text(20, 418, `${total - picture.capacity} units cannot pass the ${picture.capacity}-unit wall`, { width: 320, weight: 700, tone: 'warning' })
      + text(20, 470, 'Refusal does not erase the committed prefix.', { width: 320, tone: 'soft' });
    return { body, height: 516 };
  }
  assert.equal(picture.required.length, picture.available.length);
  const maximum = Math.max(...picture.required, ...picture.available);
  const unitWidth = Math.min(42, 288 / maximum);
  const fits = picture.required.map((need, index) => need <= picture.available[index]);
  body += text(20, 27, picture.mode === 'all' ? 'Two banks. Two gates. Both must pass.' : 'The same wall applies to both choices', { width: 320, weight: 700 });
  for (const [index, need] of picture.required.entries()) {
    const vertical = 91 + index * 156;
    const available = picture.available[index];
    const tone = fits[index] ? 'output' : 'warning';
    body += text(20, vertical, `${picture.labels[index]}: ${need} / ${available} ${picture.unit}`, { width: 320, weight: 700, tone });
    for (let unit = 0; unit < Math.max(need, available); unit++) {
      const horizontal = 28 + unit * unitWidth;
      body += `<rect data-picture-resource="${index},${unit}" data-picture-requested="${unit < need}" data-picture-capacity="${unit < available}" x="${horizontal}" y="${vertical + 29}" width="${unitWidth - 3}" height="39" rx="2" style="fill:${unit < need ? fill(unit < available ? 'input' : 'warning') : 'none'};stroke:${color(unit < available ? 'neutral' : 'warning')};stroke-width:${unit < available ? 1.5 : 2.5};${unit < available ? '' : 'stroke-dasharray:3 3'}"></rect>`;
      if (unit < need) body += circle(horizontal + (unitWidth - 3) / 2, vertical + 48.5, 4, unit < available ? 'input' : 'warning', true);
    }
    body += line(28 + available * unitWidth - 2, vertical + 19, 28 + available * unitWidth - 2, vertical + 82, 'warning')
      + text(20, vertical + 116, fits[index] ? 'Fits this bank' : 'Stops at the wall', { width: 320, tone, weight: 700 });
  }
  const verdict = picture.mode === 'all' ? fits.every(Boolean) ? 'Admit: both gates pass' : 'Do not admit: one gate fails' : `Choose ${picture.labels[fits.findIndex(Boolean)]}`;
  body += text(20, 445, verdict, { width: 320, size: 20, weight: 700, tone: picture.mode === 'all' ? 'warning' : 'output' });
  return { body, height: 484 };
}

function conditionPopulation(picture, model, tools) {
  const { text, circle, line, color, fill, number } = tools;
  const positives = picture.population.map((count, index) => count * picture.rates[index]);
  assert.deepEqual(positives, model.rows.map(row => row[1]));
  const total = picture.population.reduce((sum, count) => sum + count, 0);
  const selected = positives.reduce((sum, count) => sum + count, 0);
  const tones = ['output', 'input'];
  let body = text(20, 26, `Before the test: ${total} people`, { width: 320, weight: 700 })
    + text(20, 53, `Each dot: ${picture.peoplePerMark} people`, { width: 320, tone: 'soft' });
  let mark = 0;
  for (const [cohort, count] of picture.population.entries()) {
    assert(Number.isInteger(count / picture.peoplePerMark));
    for (let index = 0; index < count / picture.peoplePerMark; index++, mark++) {
      body += `<g data-picture-population="${cohort}">${circle(23.2 + mark % 50 * 6.4, 76 + Math.floor(mark / 50) * 6.4, 2, tones[cohort], true)}</g>`;
    }
  }
  body += text(20, 224, '100 diseased', { width: 150, weight: 700, tone: tones[0] })
    + text(340, 224, '99900 healthy', { width: 165, anchor: 'end', weight: 700, tone: tones[1] });
  body += text(20, 268, 'Select positive results', { width: 320, size: 18, weight: 700 });
  for (const [cohort, rate] of picture.rates.entries()) {
    const horizontal = 20 + cohort * 180;
    const tone = tones[cohort];
    body += text(horizontal, 300, `${number(rate * 100)}% pass this filter`, { width: 145, tone })
      + `<rect x="${horizontal}" y="325" width="140" height="20" style="fill:${fill('neutral')};stroke:${color('neutral')}"></rect><rect data-picture-survival="${cohort}" data-input="${picture.population[cohort]}" data-rate="${rate}" data-output="${positives[cohort]}" x="${horizontal}" y="325" width="${140 * rate}" height="20" style="fill:${color(tone)}"></rect>`
      + line(horizontal + 70, 355, horizontal + 70, 378, tone)
      + line(horizontal + 65, 372, horizontal + 70, 378, tone)
      + line(horizontal + 75, 372, horizontal + 70, 378, tone)
      + text(horizontal + 70, 414, positives[cohort], { width: 140, anchor: 'middle', size: 26, weight: 700, tone });
  }
  const share = positives[0] / selected;
  body += text(20, 461, 'Now the positive group is the whole', { width: 320, weight: 700 })
    + `<rect x="20" y="480" width="320" height="28" style="fill:${fill(tones[1])};stroke:${color(tones[1])}"></rect><rect data-picture-posterior="${share}" x="20" y="480" width="${share * 320}" height="28" style="fill:${color(tones[0])}"></rect>`
    + text(20, 542, `${positives[0]} / ${selected} = ${number(share * 100)}%`, { width: 320, size: 22, weight: 700, tone: tones[0] });
  return { body, height: 562 };
}

function squareSpread(picture, model, tools) {
  const { text, line, color, fill, circle, number } = tools;
  assert.deepEqual(picture.noise.map(noise => noise ** 2), model.rows.map(row => row[1]));
  let body = text(20, 27, 'Same effect; same error targets', { width: 320, weight: 700 })
    + text(20, 61, 'Noise is a width. Variance is its square.', { width: 320, size: 16 });
  for (const [index, noise] of picture.noise.entries()) {
    const horizontal = index ? 184 : 36;
    const bottom = 260;
    const unit = 72;
    const tone = index ? 'output' : 'input';
    body += `<g data-picture-square="${noise}" data-picture-variance="${noise ** 2}">`;
    for (let row = 0; row < noise; row++) for (let column = 0; column < noise; column++) {
      body += `<rect x="${horizontal + column * unit}" y="${bottom - (row + 1) * unit}" width="${unit}" height="${unit}" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.75"></rect>`
        + circle(horizontal + column * unit + unit / 2, bottom - (row + .5) * unit, 5, tone, true);
    }
    body += '</g>' + line(horizontal, bottom + 12, horizontal + noise * unit, bottom + 12, tone)
      + text(horizontal + noise * unit / 2, bottom + 39, `noise ${noise}`, { width: noise * unit + 20, anchor: 'middle', weight: 700, tone })
      + text(horizontal + noise * unit / 2, 346, `${noise ** 2}x samples`, { width: 145, anchor: 'middle', size: 20, weight: 700, tone });
  }
  body += text(20, 397, 'Four equal tiles need four equal budgets', { width: 320, weight: 700 })
    + text(20, 445, 'Relative scaling, not a sample-size prescription', { width: 320, tone: 'soft' });
  return { body, height: 488 };
}

function weightedValues(picture, model, tools) {
  const { text, color, fill, line, number } = tools;
  const contributions = picture.weights.map((weight, index) => weight * picture.values[index]);
  const expected = picture.expected ?? model.rows.map(row => row[1]);
  assert(contributions.every((value, index) => Math.abs(value - expected[index]) < 1e-12));
  if (picture.normalized) assert(Math.abs(picture.weights.reduce((sum, weight) => sum + weight, 0) - 1) < 1e-12);
  const widthUnit = picture.widthUnit ?? picture.unit;
  const heightUnit = picture.heightUnit ?? 1;
  assert(Math.abs(widthUnit * heightUnit - picture.unit) < 1e-12);
  const tileWidth = Math.min(26, 128 / Math.max(...picture.weights.map(weight => weight / widthUnit)));
  const tileHeight = Math.min(20, 144 / Math.max(...picture.values.map(value => value / heightUnit)));
  let body = text(20, 26, `Width: ${picture.widthLabel ?? 'weight'}. Height: ${picture.heightLabel ?? 'value'}.`, { width: 320, weight: 700 });
  const tones = ['input', 'output'];
  for (const [index, weight] of picture.weights.entries()) {
    const horizontal = index ? 210 : 20;
    const unitsWide = Math.round(weight / widthUnit);
    const unitsTall = Math.round(picture.values[index] / heightUnit);
    assert(Math.abs(unitsWide * widthUnit - weight) < 1e-12 && Math.abs(unitsTall * heightUnit - picture.values[index]) < 1e-12);
    body += text(horizontal, 66, `${picture.widthLabel ?? 'weight'} ${number(weight)}`, { width: 140, weight: 700, tone: tones[index] });
    for (let row = 0; row < unitsTall; row++) for (let column = 0; column < unitsWide; column++) {
      body += `<rect data-picture-contribution="${index}" x="${horizontal + column * tileWidth}" y="${244 - (row + 1) * tileHeight}" width="${tileWidth}" height="${tileHeight}" style="fill:${fill(tones[index])};stroke:${color(tones[index])};stroke-width:1.5"></rect>`;
    }
    body += text(horizontal, 278, `${picture.heightLabel ?? 'value'} ${number(picture.values[index])}`, { width: 140, tone: tones[index] })
      + text(horizontal, 313, `area ${number(contributions[index])}`, { width: 140, size: 20, weight: 700, tone: tones[index] });
    if (picture.combine !== false) body += line(horizontal + 30, 332, index ? 242 : 98, 362, tones[index]);
  }
  if (picture.combine === false) {
    body += text(20, 397, picture.verdict, { width: 320, size: 20, weight: 700, tone: 'output' })
      + text(20, 453, 'Compare actions; do not add their costs.', { width: 320, tone: 'soft' });
    return { body, height: 493 };
  }
  const total = contributions.reduce((sum, value) => sum + value, 0);
  let position = 20;
  for (const [index, contribution] of contributions.entries()) {
    for (let unit = 0; unit < Math.round(contribution / picture.unit); unit++) {
      const width = 320 / Math.round(total / picture.unit);
      body += `<rect x="${position}" y="376" width="${width}" height="25" style="fill:${fill(tones[index])};stroke:${color(tones[index])};stroke-width:1.25"></rect>`;
      position += width;
    }
  }
  body += text(20, 439, `${contributions.map(number).join(' + ')} = ${number(total)}`, { width: 320, size: 22, weight: 700 })
    + text(20, 478, `Each tile contributes ${picture.unit}`, { width: 320, tone: 'soft' });
  if (picture.totalLabel) body += text(20, 515, picture.totalLabel, { width: 320, tone: 'soft' });
  return { body, height: picture.totalLabel ? 541 : 501 };
}

function arithmeticPaths(picture, model, tools) {
  const { text, line, circle, color, fill, number } = tools;
  let body = '';
  for (const [index, path] of picture.paths.entries()) {
    const vertical = 30 + index * 149;
    const calculated = path.operator === 'multiply' ? path.inputs.reduce((product, value) => product * value, 1) : path.operator === 'add' ? path.inputs.reduce((sum, value) => sum + value, 0) : path.operator === 'maximum' ? Math.max(...path.inputs) : path.inputs[0] - path.inputs[1];
    assert(Math.abs(calculated - path.result) < 1e-12);
    assert.equal(path.inputs.length, 2);
    const tone = index ? 'output' : 'input';
    body += text(20, vertical, path.label, { width: 320, weight: 700, tone });
    for (const [operand, value] of path.inputs.entries()) {
      const horizontal = 52 + operand * 112;
      body += circle(horizontal, vertical + 56, 27, tone)
        + text(horizontal, vertical + 62, number(value), { width: 49, anchor: 'middle', weight: 700 })
        + line(horizontal + 29, vertical + 56, horizontal + (operand ? 57 : 36), vertical + 56, tone);
    }
    body += text(108, vertical + 62, path.operator === 'multiply' ? 'x' : path.operator === 'add' ? '+' : path.operator === 'maximum' ? 'max' : '-', { width: path.operator === 'maximum' ? 43 : 28, anchor: 'middle', size: path.operator === 'maximum' ? 15 : 22, weight: 700 })
      + `<path d="M236 ${vertical + 49}l10 7-10 7" style="fill:none;stroke:${color(tone)};stroke-width:2"></path>`
      + `<rect data-picture-arithmetic="${index}" data-picture-result="${path.result}" x="263" y="${vertical + 28}" width="76" height="56" rx="4" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`
      + text(301, vertical + 63, number(path.result), { width: 68, anchor: 'middle', size: 20, weight: 700, tone });
  }
  return { body, height: picture.paths.length * 149 + 4 };
}

function gainCascade(picture, model, tools) {
  const { text, color, number, line } = tools;
  let body = text(20, 26, 'The same signal passes every local gate', { width: 320, weight: 700 });
  for (const [lane, gain] of picture.gains.entries()) {
    const vertical = 141 + lane * 196;
    const tone = lane ? 'warning' : 'input';
    body += text(20, vertical - 67, `Every gate multiplies by ${gain}`, { width: 320, weight: 700, tone });
    for (let index = 0; index < picture.count; index++) {
      const before = gain ** index;
      const after = gain ** (index + 1);
      const horizontal = 28 + index * 28;
      const scale = 8;
      body += `<path data-picture-gain="${lane},${index}" data-picture-before="${before}" data-picture-after="${after}" d="M${horizontal} ${vertical - before * scale / 2}L${horizontal + 28} ${vertical - after * scale / 2}V${vertical + after * scale / 2}L${horizontal} ${vertical + before * scale / 2}Z" style="fill:${color(tone)};opacity:.65"></path>`
        + line(horizontal + 28, vertical - 34, horizontal + 28, vertical + 34, 'neutral');
    }
    const result = gain ** picture.count;
    assert(Math.abs(result - model.series[lane][1].at(-1)[1]) < 1e-12);
    body += text(28, vertical + 64, '1 in', { width: 90, tone: 'soft' })
      + text(324, vertical + 64, `${number(result)} out`, { width: 170, anchor: 'end', weight: 700, tone });
  }
  body += text(20, 451, 'Ten gates can shrink or amplify.', { width: 320, size: 18, weight: 700 });
  return { body, height: 482 };
}

function residualStep(picture, model, tools) {
  const { text, line, circle, color, number } = tools;
  const residual = picture.targetValue - picture.predictionValue;
  const step = residual * picture.shrinkage;
  const after = picture.predictionValue + step;
  assert.equal(Number(model.rows[0][1].at(-1)), residual);
  assert.equal(Number(model.rows[2][1].at(-1)), after);
  const position = value => 34 + value * 28;
  let body = text(20, 27, 'Fit the remaining gap, not the whole target', { width: 320, weight: 700 });
  for (const [phase, prediction] of [picture.predictionValue, after].entries()) {
    const vertical = 119 + phase * 188;
    body += text(20, vertical - 38, phase ? 'After one shrunk correction' : 'Before the new tree', { width: 320, weight: 700 });
    body += line(34, vertical, 314, vertical)
      + `<path d="M${position(prediction)} ${vertical}H${position(picture.targetValue)}" style="stroke:${color('warning')};stroke-width:12;opacity:.2"></path>`
      + circle(position(prediction), vertical, 7, 'input', true) + circle(position(picture.targetValue), vertical, 7, 'output', true)
      + text(position(prediction), vertical + 39, prediction, { width: 100, anchor: 'end', weight: 700, tone: 'input' })
      + text(330, vertical - 16, '10', { width: 55, anchor: 'end', tone: 'output', weight: 700 })
      + text(34, vertical + 39, '0', { width: 40, tone: 'soft' });
  }
  body += text(20, 222, `${number(residual)} remaining x ${picture.shrinkage} = ${number(step)}`, { width: 320, size: 20, weight: 700, tone: 'focus' })
    + text(20, 411, 'The correction closes only part of the gap.', { width: 320, weight: 700 });
  return { body, height: 454 };
}

function linkedScales(picture, model, tools) {
  const { text, line, circle, number } = tools;
  const odds = picture.logits.map(value => Math.exp(value));
  const probabilities = odds.map(value => value / (1 + value));
  const rows = [picture.logits, odds, probabilities];
  const maxima = [2, Math.exp(2), 1];
  const tones = ['input', 'focus', 'output'];
  let body = text(20, 27, 'Follow the same three candidates', { width: 320, weight: 700 });
  for (const [stage, values] of rows.entries()) {
    const vertical = 112 + stage * 145;
    body += text(20, vertical - 34, ['Logit: equal steps', 'Odds: multiply by e', 'Probability: compress toward one'][stage], { width: 320, weight: 700 })
      + line(35, vertical, 325, vertical);
    for (const [index, value] of values.entries()) {
      const horizontal = 35 + value / maxima[stage] * 290;
      body += circle(horizontal, vertical, 6, tones[index], true)
        + text(horizontal, vertical + (stage === 2 && index === 1 ? -15 : 34), number(value), { width: 80, anchor: index === 2 ? 'end' : 'middle', weight: 700, tone: tones[index] });
      if (stage < 2) {
        const nextHorizontal = 35 + rows[stage + 1][index] / maxima[stage + 1] * 290;
        body += line(horizontal, vertical + 44, nextHorizontal, vertical + 87, tones[index], true);
      }
    }
  }
  body += text(20, 489, 'Equal logit gaps become unequal probability gaps.', { width: 320, tone: 'soft' });
  return { body, height: 537 };
}

function densityArea(points, baseline, horizontal, verticalScale, tone, tools) {
  return `<path d="M${horizontal(points[0][0])} ${baseline}${points.map(([value, density]) => `L${horizontal(value)} ${baseline - density * verticalScale}`).join('')}L${horizontal(points.at(-1)[0])} ${baseline}Z" style="fill:${tools.fill(tone)};stroke:${tools.color(tone)};stroke-width:2"></path>`;
}

function densityUpdate(picture, model, tools) {
  const { text, circle, line, color, number } = tools;
  assert.equal(picture.successes, 2);
  assert.equal(picture.failures, 1);
  assert(Math.abs(12 * (1 / 3 - 1 / 4) - 1) < 1e-12);
  const horizontal = value => 35 + value * 290;
  let body = text(20, 27, 'Evidence reshapes a distribution', { width: 320, weight: 700 });
  body += text(20, 70, 'Prior: every probability has equal density', { width: 320, tone: 'soft' })
    + densityArea(model.series[0][1], 181, horizontal, 62, 'input', tools)
    + text(20, 219, 'Two successes; one failure', { width: 320, weight: 700 });
  for (let index = 0; index < picture.successes + picture.failures; index++) {
    const tone = index < picture.successes ? 'output' : 'warning';
    body += circle(113 + index * 67, 254, 14, tone, true);
    if (index >= picture.successes) body += line(107 + index * 67, 254, 119 + index * 67, 254, 'text');
  }
  body += text(20, 310, 'Multiply by p x p x (1-p), then normalize', { width: 320, weight: 700 })
    + densityArea(model.series[1][1], 465, horizontal, 62, 'output', tools)
    + line(horizontal(.6), 342, horizontal(.6), 472, 'output', true)
    + text(20, 507, 'Mean 0.6; uncertainty remains', { width: 320, size: 20, weight: 700, tone: 'output' });
  return { body, height: 543 };
}

function randomBranch(picture, model, tools) {
  const { text, circle, line, number } = tools;
  const outputs = [0, picture.input / picture.keep];
  assert.deepEqual(outputs, model.rows[1][1].map(Number));
  const expected = (1 - picture.keep) * outputs[0] + picture.keep * outputs[1];
  assert.equal(expected, picture.input);
  let body = text(20, 27, 'One draw follows one branch', { width: 320, weight: 700 })
    + circle(180, 95, 25, 'input', true) + text(180, 102, picture.input, { width: 45, anchor: 'middle', size: 20, weight: 700, tone: 'text' });
  for (const [index, output] of outputs.entries()) {
    const horizontal = index ? 280 : 80;
    const probability = index ? picture.keep : 1 - picture.keep;
    const radius = output ? 25 * Math.sqrt(output / picture.input) : 7;
    const tone = index ? 'output' : 'neutral';
    body += arrowBetween([180, 125], [horizontal, 233 - radius], tone, tools, 2)
      + text(index ? 340 : 20, 148, `${probability * 100}% chance`, { width: 110, anchor: index ? 'end' : 'start', size: 14, weight: 700 })
      + circle(horizontal, 247, radius, tone, Boolean(output))
      + text(horizontal, 311, output, { width: 90, anchor: 'middle', size: 26, weight: 700, tone });
  }
  body += text(20, 380, 'Neither realization equals four', { width: 320, weight: 700 })
    + text(20, 429, `${1 - picture.keep} x 0 + ${picture.keep} x ${outputs[1]} = ${expected}`, { width: 320, size: 20, weight: 700, tone: 'output' })
    + text(20, 479, 'Only the expectation is preserved.', { width: 320, tone: 'soft' });
  return { body, height: 514 };
}

function splitObservation(picture, model, tools) {
  const { text, circle, color, fill, number } = tools;
  assert.deepEqual(picture.fractions, model.rows.map(row => row[1]));
  assert.equal(picture.fractions.reduce((sum, value) => sum + value, 0), 1);
  let body = text(20, 27, 'One observation. One unit of total mass.', { width: 320, weight: 700 })
    + circle(180, 103, 30, 'focus', true)
    + text(180, 109, '1', { width: 40, anchor: 'middle', size: 22, weight: 700 });
  for (const [index, fraction] of picture.fractions.entries()) {
    const horizontal = index ? 280 : 80;
    const tone = index ? 'output' : 'input';
    const radius = Math.sqrt(fraction) * 30;
    body += arrowBetween([180, 141], [horizontal, 259 - radius], tone, tools, fraction * 10)
      + `<circle data-picture-fraction="${fraction}" cx="${horizontal}" cy="275" r="${radius}" style="fill:${color(tone)};stroke:${color(tone)}"></circle>`
      + text(horizontal, 335, fraction, { width: 120, anchor: 'middle', size: 26, weight: 700, tone })
      + text(horizontal, 380, `Update ${index ? 'B' : 'A'}`, { width: 130, anchor: 'middle', weight: 700, tone });
  }
  body += text(20, 442, 'Fractional updates, not two full observations', { width: 320, weight: 700 });
  return { body, height: 488 };
}

function binaryChannel(picture, model, tools) {
  const { text, circle, color, number } = tools;
  const information = 1 + picture.flip * Math.log2(picture.flip) + (1 - picture.flip) * Math.log2(1 - picture.flip);
  assert(Math.abs(information - model.rows[0][1]) < 1e-12);
  let body = text(20, 27, 'Fair bits can still predict each other', { width: 320, weight: 700 });
  for (let input = 0; input < 2; input++) for (let output = 0; output < 2; output++) {
    const joint = .5 * (input === output ? 1 - picture.flip : picture.flip);
    const from = [90, 132 + input * 176];
    const to = [270, 132 + output * 176];
    body += `<path data-picture-channel="${input},${output}" data-picture-joint="${joint}" d="M${from.join(' ')}C160 ${from[1]} 200 ${to[1]} ${to.join(' ')}" style="fill:none;stroke:${color(input === output ? 'input' : 'warning')};stroke-width:${joint * 30};opacity:.75"></path>`;
  }
  for (let bit = 0; bit < 2; bit++) for (const horizontal of [65, 295]) body += circle(horizontal, 132 + bit * 176, 24, bit ? 'output' : 'input') + text(horizontal, 139 + bit * 176, bit, { width: 40, anchor: 'middle', size: 22, weight: 700 });
  body += text(20, 90, 'Input: 50/50', { width: 145, weight: 700 })
    + text(340, 90, 'Output: 50/50', { width: 145, anchor: 'end', weight: 700 })
    + text(20, 384, 'Thick paths: 90% retain the input bit', { width: 320, tone: 'input', weight: 700 })
    + text(20, 428, 'Thin paths: 10% flip it', { width: 320, tone: 'warning', weight: 700 })
    + text(20, 481, `Information retained: ${number(information)} bits`, { width: 320, size: 20, weight: 700 });
  return { body, height: 527 };
}

function averageDistribution(picture, model, tools) {
  const { text, circle, line, number } = tools;
  assert.equal(picture.draws.length, 10);
  const mean = picture.draws.reduce((sum, value) => sum + value, 0) / picture.draws.length;
  assert(Math.abs(mean - 1) < 1e-12);
  const horizontal = value => 35 + value / 4 * 290;
  const original = Array.from({ length: 81 }, (_, index) => [index / 20, Math.exp(-index / 20)]);
  const means = Array.from({ length: 81 }, (_, index) => { const value = index / 20; return [value, 10 ** 10 / 362880 * value ** 9 * Math.exp(-10 * value)]; });
  let body = text(20, 27, 'The population does not become an average', { width: 320, weight: 700 })
    + text(20, 77, 'Original exponential population', { width: 320, weight: 700, tone: 'input' })
    + densityArea(original, 180, horizontal, 72, 'input', tools);
  for (const value of picture.draws) body += circle(horizontal(value), 235, 4, 'input', true) + line(horizontal(value), 245, horizontal(mean), 291, 'neutral', true);
  body += circle(horizontal(mean), 304, 7, 'output', true)
    + text(340, 284, 'Ten draws', { width: 150, anchor: 'end', tone: 'input' })
    + text(340, 320, 'One mean', { width: 150, anchor: 'end', tone: 'output', weight: 700 })
    + text(20, 357, 'Distribution of means of ten', { width: 320, weight: 700, tone: 'output' })
    + densityArea(means, 466, horizontal, 72, 'output', tools)
    + text(20, 515, 'Same center. One tenth the variance.', { width: 320, size: 18, weight: 700 });
  return { body, height: 555 };
}

function channelPooling(picture, model, tools) {
  const { text, color, fill, circle } = tools;
  assert.equal(picture.channels, model.rows[0][1].length);
  let body = text(20, 27, 'Collapse space, not channels', { width: 320, weight: 700 });
  for (let channel = 0; channel < picture.channels; channel++) {
    const horizontal = 20 + channel * 115;
    const tone = ['input', 'focus', 'output'][channel];
    let total = 0;
    body += text(horizontal + 43, 76, `Channel ${channel + 1}`, { width: 100, anchor: 'middle', weight: 700, tone });
    for (let row = 0; row < picture.side; row++) for (let column = 0; column < picture.side; column++) {
      const value = channel + (row + column) % 2 * 2;
      total += value;
      body += `<rect data-picture-pool="${channel},${row},${column}" data-picture-value="${value}" x="${horizontal + column * 22}" y="${112 + row * 22}" width="22" height="22" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1"></rect>`
        + text(horizontal + column * 22 + 11, 128 + row * 22, value, { width: 20, anchor: 'middle', tone });
    }
    const mean = total / picture.side ** 2;
    assert.equal(mean, channel + 1);
    body += arrowBetween([horizontal + 44, 219], [horizontal + 44, 313], tone, tools, 2)
      + text(horizontal + 44, 267, '/ 16', { width: 100, anchor: 'middle', tone, weight: 700 })
      + circle(horizontal + 44, 351, 27, tone, true)
      + text(horizontal + 44, 358, mean, { width: 46, anchor: 'middle', size: 22, weight: 700 });
  }
  body += text(20, 423, 'Sixteen spatial values become one per channel', { width: 320, weight: 700 })
    + text(20, 479, 'Three channels still produce three outputs.', { width: 320, tone: 'soft' });
  return { body, height: 522 };
}

function sufficientCompression(picture, model, tools) {
  const { text, circle, line, color, fill } = tools;
  const successes = picture.outcomes.reduce((sum, value) => sum + value, 0);
  const count = picture.outcomes.length;
  assert.equal(successes / count, Number(model.rows.at(-1)[1].at(-1)));
  let body = text(20, 27, 'Two orders, one Bernoulli likelihood', { width: 320, weight: 700 });
  for (const [phase, order] of [picture.outcomes.map((_, index) => index), picture.permutation].entries()) {
    const vertical = 103 + phase * 101;
    for (const [position, index] of order.entries()) {
      const value = picture.outcomes[index];
      const horizontal = 45 + position * 66;
      body += circle(horizontal, vertical, 19, value ? 'output' : 'neutral', Boolean(value))
        + text(horizontal, vertical + 6, value, { width: 32, anchor: 'middle', weight: 700 });
    }
  }
  body += arrowBetween([105, 240], [160, 300], 'input', tools, 2) + arrowBetween([255, 240], [200, 300], 'input', tools, 2)
    + text(180, 337, `${successes} successes; ${count} trials`, { width: 300, anchor: 'middle', size: 22, weight: 700 });
  for (let index = 0; index < count; index++) body += `<rect x="${35 + index * 58}" y="375" width="58" height="34" style="fill:${fill(index < successes ? 'output' : 'neutral')};stroke:${color(index < successes ? 'output' : 'neutral')};stroke-width:1.75"></rect>`;
  body += text(20, 458, `${successes} / ${count} = ${successes / count}`, { width: 320, size: 24, weight: 700, tone: 'output' })
    + text(20, 505, 'Order is discarded; the sufficient counts remain.', { width: 320, tone: 'soft' });
  return { body, height: 550 };
}

function bitReassignment(picture, model, tools) {
  const { text, color, fill, line } = tools;
  assert.deepEqual(picture.formats[1], [1, model.rows[0][1], model.rows[1][1]]);
  assert(picture.formats.every(counts => counts.reduce((sum, value) => sum + value, 0) === 16));
  const tones = ['neutral', 'input', 'output'];
  let body = text(20, 27, 'Sixteen slots. Three change jobs.', { width: 320, weight: 700 });
  for (const [format, counts] of picture.formats.entries()) {
    const vertical = 111 + format * 182;
    body += text(20, vertical - 28, format ? 'BF16' : 'FP16', { width: 320, size: 20, weight: 700 });
    let slot = 0;
    for (const [field, count] of counts.entries()) for (let index = 0; index < count; index++, slot++) {
      body += `<rect data-picture-bit="${format},${slot}" data-picture-field="${field}" x="${20 + slot * 20}" y="${vertical}" width="19" height="36" style="fill:${fill(tones[field])};stroke:${color(tones[field])};stroke-width:${slot >= 6 && slot <= 8 ? 2.5 : 1.25}"></rect>`;
    }
    body += text(20, vertical + 78, `1 sign; ${counts[1]} exponent; ${counts[2]} fraction`, { width: 320, weight: 700 });
  }
  body += arrowBetween([170, 212], [170, 259], 'focus', tools, 3)
    + text(20, 446, 'Blue buys range. Green buys precision.', { width: 320, weight: 700 })
    + text(20, 493, 'The total bit budget never grows.', { width: 320, tone: 'soft' });
  return { body, height: 533 };
}

function targetRedistribution(picture, model, tools) {
  const { text, color, fill, circle } = tools;
  const perClass = picture.redistributed / picture.classes;
  const counts = Array.from({ length: picture.classes }, (_, index) => perClass + (index ? 0 : picture.retained));
  assert.deepEqual([counts[0] / 100, counts.slice(1).reduce((sum, count) => sum + count, 0) / 100], model.rows.map(row => row[1]));
  let body = text(20, 27, 'Keep ninety. Spread ten across all five.', { width: 320, weight: 700 });
  body += dotField(100, 10, 124, 81, 12, index => ({ tone: index < picture.retained ? 'output' : 'focus' }), tools, 'target-before')
    + text(20, 234, 'The true class receives uniform mass too', { width: 320, weight: 700 });
  for (const [index, count] of counts.entries()) {
    const horizontal = 20 + index * 65;
    body += text(horizontal + 26, 291, `${index + 1}`, { width: 50, anchor: 'middle', weight: 700 })
      + dotField(count, 10, horizontal + 3, 321, 5.2, unit => ({ tone: index === 0 && unit < picture.retained ? 'output' : 'focus' }), tools, `target-after-${index}`)
      + text(horizontal + 26, 425, count / 100, { width: 61, anchor: 'middle', weight: 700, tone: index ? 'focus' : 'output' });
    body += arrowBetween([180, 256], [horizontal + 26, 268], 'focus', tools, 1.5);
  }
  body += text(20, 489, '90 + 2 = 92 for the true class', { width: 320, size: 20, weight: 700, tone: 'output' });
  return { body, height: 530 };
}

function interpolationFreedom(picture, model, tools) {
  const { text, circle, line, color, fill, number } = tools;
  const value = (input, gain) => input * (2 - input) + gain * input * (input - 1) * (input - 2);
  const horizontal = input => 35 + input / 2 * 290;
  const vertical = output => 320 - (output + .5) / 2.3 * 250;
  const tones = ['input', 'output'];
  let body = text(20, 27, 'Observed points pin the material in place', { width: 320, weight: 700 })
    + line(35, vertical(0), 325, vertical(0));
  for (const [family, gain] of picture.gains.entries()) {
    const points = Array.from({ length: 81 }, (_, index) => [index / 40, value(index / 40, gain)]);
    for (const input of picture.observed) assert.equal(value(input, gain), input === 1 ? 1 : 0);
    const upper = points.map(([input, output]) => [horizontal(input), vertical(output + .035)]);
    const lower = [...points].reverse().map(([input, output]) => [horizontal(input), vertical(output - .035)]);
    body += `<path data-picture-ribbon="${family}" data-picture-parameter="${gain}" d="${[...upper, ...lower].map((point, index) => `${index ? 'L' : 'M'}${point.join(' ')}`).join('')}Z" style="fill:${fill(tones[family])};stroke:${color(tones[family])};stroke-width:2"></path>`;
  }
  for (const input of picture.observed) body += circle(horizontal(input), vertical(value(input, 0)), 8, 'focus', true)
    + text(horizontal(input), 359, input, { width: 70, anchor: input === 0 ? 'start' : input === 2 ? 'end' : 'middle', tone: 'soft' });
  const results = picture.gains.map(gain => value(picture.probe, gain));
  assert.deepEqual(results, [.75, 1.5]);
  body += line(horizontal(picture.probe), vertical(results[1]), horizontal(picture.probe), vertical(results[0]), 'warning', true);
  for (const [index, result] of results.entries()) body += circle(horizontal(picture.probe), vertical(result), 5, tones[index], true);
  body += text(20, 412, 'Between the pins, predictions can separate', { width: 320, size: 18, weight: 700 })
    + text(20, 468, 'At x = 0.5: 0.75 or 1.5', { width: 320, size: 20, weight: 700, tone: 'warning' })
    + text(20, 516, 'Both have zero error on the observed pins.', { width: 320, tone: 'soft' });
  return { body, height: 556 };
}

function likelihoodCandidates(picture, model, tools) {
  const { text, color, fill, circle, number } = tools;
  const successes = picture.outcomes.reduce((sum, value) => sum + value, 0);
  assert.equal(successes, 3);
  assert.equal(picture.outcomes.length - successes, 2);
  const likelihoods = picture.candidates.map(candidate => picture.outcomes.reduce((product, outcome) => product * (outcome ? candidate : 1 - candidate), 1));
  assert(likelihoods[1] > likelihoods[0] && likelihoods[1] > likelihoods[2]);
  let body = text(20, 27, 'Hold these five observed outcomes fixed', { width: 320, weight: 700 });
  for (const [index, outcome] of picture.outcomes.entries()) body += circle(111 + index * 40, 77, 12, outcome ? 'output' : 'warning') + text(111 + index * 40, 82, outcome, { width: 20, anchor: 'middle', weight: 700 });
  for (const [lane, candidate] of picture.candidates.entries()) {
    const vertical = 153 + lane * 115;
    const tone = lane === 1 ? 'output' : 'input';
    body += text(20, vertical + 18, `p=${candidate}`, { width: 75, weight: 700, tone });
    for (const [index, outcome] of picture.outcomes.entries()) {
      const factor = outcome ? candidate : 1 - candidate;
      const horizontal = 100 + index * 40;
      body += `<rect x="${horizontal}" y="${vertical}" width="22" height="42" style="fill:${fill('neutral')};stroke:${color('neutral')}"></rect><rect data-picture-likelihood="${lane},${index}" data-picture-factor="${factor}" x="${horizontal}" y="${vertical + 42 * (1 - factor)}" width="22" height="${42 * factor}" style="fill:${color(outcome ? tone : 'warning')}"></rect>`;
    }
    body += text(340, vertical + 85, `Product: ${number(likelihoods[lane])}`, { width: 250, anchor: 'end', weight: 700, tone });
  }
  body += text(20, 540, 'Multiply successes and failures for each candidate.', { width: 320, tone: 'soft' });
  return { body, height: 585 };
}

function localSlope(picture, model, tools) {
  const { text, line, circle, color, number } = tools;
  const horizontal = value => 35 + (value + 4) / 8 * 290;
  let body = text(20, 27, 'A flat forward response gives a small slope', { width: 320, weight: 700 });
  const specifications = [
    { name: 'Sigmoid: output zero to one', baseline: 218, scale: 110, operation: value => 1 / (1 + Math.exp(-value)), derivative: value => { const probability = 1 / (1 + Math.exp(-value)); return probability * (1 - probability); }, probes: [-3, 0, 3] },
    { name: 'ReLU: output zero to four', baseline: 462, scale: 32, operation: value => Math.max(0, value), derivative: value => value < 0 ? 0 : 1, probes: [-2, 2] },
  ];
  for (const [index, specification] of specifications.entries()) {
    const tone = index ? 'output' : 'input';
    body += text(20, index ? 301 : 79, specification.name, { width: 320, weight: 700, tone })
      + line(35, specification.baseline, 325, specification.baseline)
      + `<path d="${Array.from({ length: 81 }, (_, position) => { const value = -4 + position / 10; return `${position ? 'L' : 'M'}${horizontal(value)} ${specification.baseline - specification.operation(value) * specification.scale}`; }).join('')}" style="fill:none;stroke:${color(tone)};stroke-width:2"></path>`;
    for (const probe of specification.probes) {
      const output = specification.operation(probe);
      const slope = specification.derivative(probe);
      const start = [horizontal(probe - .4), specification.baseline - (output - slope * .4) * specification.scale];
      const end = [horizontal(probe + .4), specification.baseline - (output + slope * .4) * specification.scale];
      body += line(...start, ...end, 'warning') + circle(horizontal(probe), specification.baseline - output * specification.scale, 4, tone, true)
        + text(horizontal(probe), specification.baseline + 36, number(slope), { width: 85, anchor: 'middle', weight: 700, tone });
    }
  }
  body += text(20, 550, 'Numbers are local slopes, not forward outputs.', { width: 320, tone: 'soft' });
  return { body, height: 594 };
}

function weightedDraws(picture, model, tools) {
  const { text, circle, line, number } = tools;
  const count = picture.weights.length;
  const total = picture.weights.reduce((sum, weight) => sum + weight, 0);
  const squares = picture.weights.reduce((sum, weight) => sum + weight ** 2, 0);
  let body = text(20, 27, 'Observation identity does not multiply', { width: 320, weight: 700 });
  for (const [index, weight] of picture.weights.entries()) {
    const horizontal = count === 1 ? 180 : 40 + index * 70;
    body += circle(horizontal, 106, 8, 'input', true)
      + text(horizontal, 151, `#${index + 1}`, { width: 60, anchor: 'middle', tone: 'soft' })
      + arrowBetween([horizontal, 178], [horizontal, 267 - 14 * Math.sqrt(weight)], 'input', tools)
      + circle(horizontal, 283, 14 * Math.sqrt(weight), 'output')
      + text(horizontal, 289, weight, { width: 48, anchor: 'middle', weight: 700, tone: 'output' });
  }
  body += text(20, 374, `${count} observed ${count === 1 ? 'click' : 'draws'}; total weight ${total}`, { width: 320, weight: 700 });
  if (picture.examination) {
    assert.equal(1 / picture.examination, total);
    body += text(20, 431, `Examination ${picture.examination}: weight = ${total}`, { width: 320, tone: 'output', weight: 700 });
  } else {
    assert.equal(total ** 2 / squares, 2.5);
    body += text(20, 431, `ESS = ${total} squared / ${squares} = ${number(total ** 2 / squares)}`, { width: 320, tone: 'output', weight: 700 });
  }
  return { body, height: 480 };
}

function massExpansion(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  let body = text(20, 27, 'Stretch the space, not the probability mass', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const width = 145 * (stage + 1);
    const vertical = 102 + stage * 226;
    const tone = stage ? 'output' : 'input';
    body += `<rect data-picture-area="${stage + 1}" x="35" y="${vertical}" width="${width}" height="88" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`;
    for (let index = 0; index < 4; index++) body += `<g data-picture-mass-member="${index}" data-picture-stage="${stage}">${circle(35 + width * ((index % 2) * .5 + .25), vertical + 22 + Math.floor(index / 2) * 44, 7, tone, true)}</g>`;
    body += text(20, vertical + 127, `Mass 1; area ${stage + 1}; density ${stage ? '0.5' : '1'}`, { width: 320, weight: 700, tone });
  }
  body += text(20, 519, 'Four marks still carry four quarters of the mass.', { width: 320, tone: 'soft' });
  return { body, height: 565 };
}

function baselineDisplacement(picture, model, tools) {
  const { text, line, circle } = tools;
  const horizontal = value => 35 + value * 48;
  const advantage = picture.observed - picture.baseline;
  assert.equal(advantage, -2);
  let body = text(20, 27, 'Measure from the baseline, not from zero', { width: 320, weight: 700 }) + line(35, 198, 323, 198);
  for (let value = 0; value <= 6; value++) body += line(horizontal(value), 193, horizontal(value), 205) + text(horizontal(value), 241, value, { width: 30, anchor: 'middle', tone: 'soft' });
  body += circle(horizontal(picture.observed), 198, 9, 'input', true) + circle(horizontal(picture.baseline), 198, 9, 'focus', true)
    + text(horizontal(picture.observed), 158, 'return 3', { width: 110, anchor: 'middle', tone: 'input', weight: 700 })
    + text(340, 112, 'baseline 5', { width: 120, anchor: 'end', tone: 'focus', weight: 700 })
    + arrowBetween([horizontal(picture.baseline), 290], [horizontal(picture.observed), 290], 'warning', tools, 4)
    + text(20, 365, 'Move backward by two: advantage -2', { width: 320, weight: 700, tone: 'warning' })
    + text(20, 426, 'Positive return; below-baseline action.', { width: 320, tone: 'soft' });
  return { body, height: 471 };
}

function oddsUpdate(picture, model, tools) {
  const { text, circle, color, fill, number } = tools;
  let body = text(20, 27, 'Independent evidence changes relative weight', { width: 320, weight: 700 });
  for (const [stage, odds] of picture.odds.entries()) {
    const vertical = 109 + stage * 152;
    const share = odds / (odds + picture.reference);
    body += text(20, vertical - 30, ['Prior odds 1:4', 'After one likelihood ratio of two', 'After a second independent ratio of two'][stage], { width: 320, weight: 700 })
      + `<rect x="20" y="${vertical}" width="320" height="32" style="fill:${fill('neutral')};stroke:${color('neutral')}"></rect><rect data-picture-odds="${odds}" x="20" y="${vertical}" width="${320 * share}" height="32" style="fill:${color('input')}"></rect>`
      + text(20, vertical + 69, `${odds}:${picture.reference} becomes ${number(share * 100)}%`, { width: 320, tone: 'input', weight: 700 });
    if (stage < 2) body += arrowBetween([320, vertical + 80], [320, vertical + 111], 'input', tools);
  }
  body += text(20, 543, 'Correlated evidence needs a different joint model.', { width: 320, tone: 'soft' });
  return { body, height: 591 };
}

function rectangleExchange(picture, model, tools) {
  const { text, color, fill } = tools;
  const products = picture.shapes.map(shape => shape[0] * shape[1]);
  assert(products.every(product => product === 8));
  let body = text(20, 27, 'Reshape the same eight compute tiles', { width: 320, weight: 700 });
  for (const [stage, shape] of picture.shapes.entries()) {
    const vertical = 103 + stage * 300;
    const tone = stage ? 'output' : 'input';
    for (let row = 0; row < shape[1]; row++) for (let column = 0; column < shape[0]; column++) body += `<rect data-picture-compute-tile="${stage},${row},${column}" x="${55 + column * 55}" y="${vertical + row * 55}" width="55" height="55" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>`;
    body += text(20, vertical - 24, `${shape[0]} ${picture.labels[0]} x ${shape[1]} ${picture.labels[1]}`, { width: 320, weight: 700, tone });
  }
  body += text(20, 571, 'Twice the width leaves half the height.', { width: 320, tone: 'soft' });
  return { body, height: 617 };
}

function copiedGradient(picture, model, tools) {
  const { text, circle, number } = tools;
  const residual = picture.upstream * picture.derivative;
  assert(Math.abs(picture.upstream + residual - 2.42) < 1e-12);
  let body = text(20, 27, 'Addition copies the upstream sensitivity', { width: 320, weight: 700 })
    + circle(180, 108, 28, 'focus') + text(180, 114, picture.upstream, { width: 49, anchor: 'middle', weight: 700 });
  for (let branch = 0; branch < 2; branch++) {
    const horizontal = branch ? 280 : 80;
    const tone = branch ? 'output' : 'input';
    body += arrowBetween([180, 139], [horizontal, 226], tone, tools, 3)
      + text(horizontal, 258, '2.2', { width: 80, anchor: 'middle', weight: 700, tone })
      + text(horizontal, 305, branch ? 'x 0.1' : 'x 1', { width: 100, anchor: 'middle', weight: 700 })
      + arrowBetween([horizontal, 326], [horizontal, 372], tone, tools)
      + text(horizontal, 409, number(branch ? residual : picture.upstream), { width: 110, anchor: 'middle', size: 22, weight: 700, tone })
      + arrowBetween([horizontal, 433], [180, 482], tone, tools);
  }
  body += text(180, 525, '2.2 + 0.22 = 2.42', { width: 320, anchor: 'middle', size: 22, weight: 700 });
  return { body, height: 562 };
}

function potentialLoop(picture, model, tools) {
  const { text, circle, number } = tools;
  const positions = [[65, 155], [285, 155], [65, 344]];
  let body = text(20, 27, 'The return trip repays the potential gain', { width: 320, weight: 700 });
  for (const [index, potential] of picture.potentials.entries()) body += circle(...positions[index], 24, index === 1 ? 'output' : 'input') + text(positions[index][0], positions[index][1] + 6, potential, { width: 40, anchor: 'middle', weight: 700 });
  body += arrowBetween([93, 155], [253, 155], 'output', tools, 3) + text(175, 123, '+3', { width: 80, anchor: 'middle', tone: 'output', weight: 700 })
    + arrowBetween([267, 178], [89, 324], 'warning', tools, 3) + text(228, 287, '-3', { width: 80, anchor: 'middle', tone: 'warning', weight: 700 })
    + text(20, 427, 'Same state: potential 2 again', { width: 320, weight: 700 })
    + text(20, 478, 'Undiscounted loop reward: +3 - 3 = 0', { width: 320, tone: 'output', weight: 700 });
  assert.equal(picture.potentials.at(-1) - picture.potentials[0], 0);
  return { body, height: 526 };
}

function backwardFold(picture, model, tools) {
  const { text, circle, number } = tools;
  let accumulator = 0;
  let body = text(20, 27, 'Walk backward: decay the future, add today', { width: 320, weight: 700 });
  for (let index = picture.residuals.length - 1; index >= 0; index--) {
    const vertical = 139 + index * 154;
    const future = accumulator;
    accumulator = picture.residuals[index] + picture.decay * future;
    body += circle(52, vertical, 23, 'input') + text(52, vertical + 6, picture.residuals[index], { width: 42, anchor: 'middle', weight: 700 })
      + arrowBetween([79, vertical], [193, vertical], 'input', tools)
      + text(228, vertical + 7, number(accumulator), { width: 100, anchor: 'middle', weight: 700, tone: 'output' })
      + text(20, vertical + 54, `${picture.residuals[index]} + 0.8 x ${number(future)}`, { width: 310, tone: 'soft' });
    if (index > 0) body += arrowBetween([290, vertical - 10], [290, vertical - 115], 'output', tools, 3)
      + text(251, vertical - 61, 'x .8', { width: 70, anchor: 'end', tone: 'output', size: 16 });
  }
  assert(Math.abs(accumulator - 1.96) < 1e-12);
  body += text(20, 560, 'Future residuals cross more decay boundaries.', { width: 320, tone: 'soft' });
  return { body, height: 606 };
}

function targetCorruption(picture, model, tools) {
  const { text, circle, number } = tools;
  assert.equal(picture.counts.reduce((sum, count) => sum + count, 0), picture.total);
  assert.deepEqual(picture.counts, model.rows.map(row => row[1]));
  let body = text(20, 27, 'Target selection survives different corruptions', { width: 320, weight: 700 });
  const tones = ['focus', 'warning', 'output', 'neutral'];
  const targets = picture.counts.slice(0, 3).reduce((sum, value) => sum + value, 0);
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 109 + stage * 248;
    body += text(20, vertical - 28, stage ? 'After corruption: the same 150 targets' : 'Before corruption: select 150 of 1,000', { width: 320, weight: 700 });
    let group = 0;
    let boundary = picture.counts[0] / picture.unit;
    for (let index = 0; index < picture.total / picture.unit; index++) {
      while (index >= boundary) { group++; boundary += picture.counts[group] / picture.unit; }
      const tone = stage ? tones[group] : index < targets / picture.unit ? 'input' : 'neutral';
      body += `<g data-picture-token-block="${index}" data-picture-stage="${stage}" data-picture-target="${index < targets / picture.unit}">${circle(42 + index % 20 * 14, vertical + Math.floor(index / 20) * 14, 3, tone, true)}</g>`;
    }
  }
  body += text(20, 543, '120 masked; 15 random; 15 unchanged', { width: 320, weight: 700 })
    + text(20, 595, 'Each mark is five tokens. All 150 remain targets.', { width: 320, tone: 'soft' });
  return { body, height: 644 };
}

function clusterResample(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  const tones = ['input', 'state', 'output'];
  let body = text(20, 27, 'Sample the user; carry both paired events', { width: 320, weight: 700 });
  for (const [stage, users] of [picture.original, picture.selected].entries()) {
    const vertical = 124 + stage * 240;
    body += text(20, vertical - 46, stage ? 'One bootstrap sample: A, C, A' : 'Original user clusters: A, B, C', { width: 320, weight: 700 });
    for (const [index, user] of users.entries()) {
      const horizontal = 20 + index * 110;
      const tone = tones[picture.original.indexOf(user)];
      body += `<rect x="${horizontal}" y="${vertical}" width="100" height="104" rx="4" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`
        + text(horizontal + 50, vertical + 29, user, { width: 80, anchor: 'middle', weight: 700, tone });
      for (let event = 0; event < 2; event++) body += circle(horizontal + 28 + event * 45, vertical + 67, 16, tone)
        + text(horizontal + 28 + event * 45, vertical + 72, `${user.toLowerCase()}${event + 1}`, { width: 31, anchor: 'middle', size: 14, weight: 700 });
    }
  }
  body += text(20, 531, 'A is repeated as a whole; B is absent.', { width: 320, tone: 'soft' });
  return { body, height: 578 };
}

function gradientBoundary(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'The value crosses; the gradient stops', { width: 320, weight: 700 })
    + line(180, 114, 180, 229, 'warning', true) + line(180, 278, 180, 400, 'warning', true)
    + text(180, 86, 'detach boundary', { width: 240, anchor: 'middle', weight: 700, tone: 'warning' });
  for (const horizontal of [68, 292]) body += circle(horizontal, 180, 30, 'input') + text(horizontal, 186, `h=${picture.value}`, { width: 54, anchor: 'middle', weight: 700, tone: 'input' });
  body += arrowBetween([102, 180], [255, 180], 'input', tools, 3)
    + text(20, 257, 'Forward hidden value: unchanged', { width: 320, weight: 700, tone: 'input' })
    + arrowBetween([292, 342], [192, 342], 'warning', tools, 3)
    + line(170, 332, 190, 352, 'warning') + line(190, 332, 170, 352, 'warning')
    + text(20, 445, 'No reverse sensitivity reaches the earlier graph.', { width: 320, tone: 'warning', weight: 700 });
  return { body, height: 505 };
}

function alignmentCollapse(picture, model, tools) {
  const { text, circle, line } = tools;
  const stages = [picture.path.map((value, index) => ({ value, index })), [{ value: 'a', index: 0 }, { value: '_', index: 2 }, { value: 'a', index: 3 }], [{ value: 'a', index: 0 }, { value: 'a', index: 3 }]];
  let body = text(20, 27, 'The blank separates two equal outputs', { width: 320, weight: 700 });
  for (const [stage, values] of stages.entries()) {
    const vertical = 138 + stage * 171;
    body += text(20, vertical - 45, ['Alignment: a, a, blank, a', 'Merge consecutive repeats first', 'Then remove the blank'][stage], { width: 320, weight: 700 });
    for (const [position, value] of values.entries()) {
      const horizontal = 65 + position * 75;
      const tone = value.value === '_' ? 'warning' : value.index < 2 ? 'input' : 'output';
      body += `<g data-picture-alignment-member="${value.index}" data-picture-stage="${stage}">${circle(horizontal, vertical, 24, tone)}${text(horizontal, vertical + 7, value.value, { width: 40, anchor: 'middle', size: 24, weight: 700, tone })}</g>`;
    }
  }
  assert.equal(stages.at(-1).map(item => item.value).join(''), 'aa');
  body += text(20, 550, 'Delete the separator only after merging repeats.', { width: 320, tone: 'soft' });
  return { body, height: 598 };
}

function patchColumns(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Overlapping patches reuse the same pixels', { width: 320, weight: 700 });
  for (let row = 0; row < 2; row++) for (let column = 0; column < 3; column++) {
    const value = picture.image[row][column];
    body += `<rect x="${54 + column * 80}" y="${102 + row * 70}" width="80" height="70" style="fill:${fill(column === 1 ? 'focus' : 'neutral')};stroke:${color('neutral')}"></rect>`
      + text(94 + column * 80, 145 + row * 70, value, { width: 60, anchor: 'middle', size: 25, weight: 700 });
  }
  for (let patch = 0; patch < 2; patch++) {
    const tone = patch ? 'output' : 'input';
    body += `<rect x="${51 + patch * 80}" y="${99 + patch * 6}" width="166" height="146" style="fill:none;stroke:${color(tone)};stroke-width:3"></rect>`
      + text(110 + patch * 140, 324, `patch ${patch}`, { width: 110, anchor: 'middle', weight: 700, tone });
    const column = picture.image.flatMap(row => row.slice(patch, patch + 2));
    assert.deepEqual(column, model.values.map(row => row[patch]));
    for (const [index, value] of column.entries()) body += `<rect data-picture-patch="${patch}" data-picture-pixel="${value}" x="${82 + patch * 140}" y="${351 + index * 49}" width="56" height="49" style="fill:${fill(value === 2 || value === 5 ? 'focus' : tone)};stroke:${color(tone)};stroke-width:1.5"></rect>`
      + text(110 + patch * 140, 383 + index * 49, value, { width: 45, anchor: 'middle', size: 22, weight: 700 });
  }
  body += text(20, 598, 'Pixels 2 and 5 belong to both columns.', { width: 320, tone: 'soft' });
  return { body, height: 644 };
}

function sharedMemory(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  let body = text(20, 27, 'Share the store, preserve every query output', { width: 320, weight: 700 });
  for (let group = 0; group < picture.groups; group++) {
    const vertical = 150 + group * 264;
    const tone = group ? 'output' : 'input';
    body += `<rect x="20" y="${vertical - 31}" width="94" height="63" rx="4" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`
      + text(67, vertical - 4, `KV ${group + 1}`, { width: 85, anchor: 'middle', weight: 700, tone });
    for (let query = 0; query < picture.perGroup; query++) {
      const number = group * picture.perGroup + query + 1;
      const row = vertical - 84 + query * 56;
      body += arrowBetween([119, vertical], [176, row], tone, tools)
        + circle(200, row, 19, tone) + text(200, row + 5, `Q${number}`, { width: 36, anchor: 'middle', size: 14, weight: 700 })
        + arrowBetween([223, row], [274, row], tone, tools)
        + `<g data-picture-query-output="${number}">${circle(299, row, 19, tone)}${text(299, row + 5, `O${number}`, { width: 35, anchor: 'middle', size: 14, weight: 700 })}</g>`;
    }
  }
  assert.equal(picture.groups * picture.perGroup, 8);
  body += text(20, 571, 'Two stores; eight independently computed outputs.', { width: 320, tone: 'soft' });
  return { body, height: 619 };
}

function featurePartition(picture, model, tools) {
  const { text, color, fill } = tools;
  const half = picture.features / picture.heads;
  assert.equal(half, 2);
  let body = text(20, 27, 'Every head keeps every token row', { width: 320, weight: 700 });
  for (let stage = 0; stage < 3; stage++) {
    const vertical = 110 + stage * 199;
    body += text(20, vertical - 26, ['Input: three tokens, four features', 'Split columns into two heads', 'Concatenate feature columns again'][stage], { width: 320, weight: 700 });
    for (let token = 0; token < picture.tokens; token++) for (let feature = 0; feature < picture.features; feature++) {
      const head = Math.floor(feature / half);
      const horizontal = 64 + feature * 51 + (stage === 1 && head ? 24 : 0);
      const tone = head ? 'output' : 'input';
      body += `<rect data-picture-feature="${token},${feature}" data-picture-stage="${stage}" x="${horizontal}" y="${vertical + token * 39}" width="51" height="39" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>`;
    }
  }
  body += text(20, 678, 'The split changes features, not sequence membership.', { width: 320, tone: 'soft' });
  return { body, height: 728 };
}

function cacheFrontier(picture, model, tools) {
  const { text, circle, color, fill, line } = tools;
  let body = text(20, 27, 'One new write; read the complete prefix', { width: 320, weight: 700 })
    + circle(180, 101, 27, 'focus') + text(180, 107, `Q${picture.pairs - 1}`, { width: 47, anchor: 'middle', weight: 700 });
  assert(Number.isInteger(picture.pairs) && picture.pairs > 0 && picture.pairs <= 8);
  const spacing = 320 / picture.pairs;
  const width = Math.min(70, spacing - 4);
  for (let pair = 0; pair < picture.pairs; pair++) {
    const horizontal = 20 + pair * spacing;
    const vertical = 243;
    const tone = pair === picture.pairs - 1 ? 'output' : 'input';
    body += arrowBetween([180, 134], [horizontal + width / 2, vertical - 9], tone, tools)
      + `<rect data-picture-cache-pair="${pair}" data-picture-new-write="${pair === picture.pairs - 1}" x="${horizontal}" y="${vertical}" width="${width}" height="56" rx="4" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`
      + text(horizontal + width / 2, vertical + 34, `KV${pair}`, { width: width - 6, anchor: 'middle', weight: 700, tone });
  }
  body += text(20, picture.pairs > 4 ? 477 : 375, `${picture.pairs - 1} retained; one appended; all ${picture.pairs} read`, { width: 320, weight: 700 })
    + text(20, picture.pairs > 4 ? 541 : 435, 'Reusing projections does not remove attention access.', { width: 320, tone: 'soft' });
  return { body, height: picture.pairs > 4 ? 591 : 486 };
}

function causalFrontier(picture, model, tools) {
  const { text, circle, line, color } = tools;
  assert.equal(picture.weights.reduce((sum, weight) => sum + weight, 0), 1);
  assert(picture.weights.slice(picture.query + 1).every(weight => weight === 0));
  let body = text(20, 27, 'Future values cannot cross the time barrier', { width: 320, weight: 700 })
    + text(20, 86, 'Query at position two', { width: 320, weight: 700, tone: 'focus' })
    + line(226, 127, 226, 400, 'warning', true);
  for (const [position, weight] of picture.weights.entries()) {
    const horizontal = 40 + position * 69;
    const tone = weight ? 'input' : 'neutral';
    body += circle(horizontal, 176, 17, tone) + text(horizontal, 182, position, { width: 30, anchor: 'middle', weight: 700 });
    if (weight) body += arrowBetween([horizontal, 198], [horizontal, 292], 'input', tools, 2 + weight * 12);
    else body += line(horizontal, 233, horizontal, 275, 'warning', true) + line(horizontal - 7, 273, horizontal + 7, 287, 'warning') + line(horizontal + 7, 273, horizontal - 7, 287, 'warning');
    body += text(horizontal, 350, weight, { width: 62, anchor: 'middle', weight: 700, tone: weight ? 'input' : 'warning' });
  }
  body += text(20, 441, 'Allowed weights sum to one; future weights are zero.', { width: 320, weight: 700 })
    + text(20, 507, 'Mask the logits before normalizing the visible prefix.', { width: 320, tone: 'soft' });
  return { body, height: 561 };
}

function clippedImageMass(picture, model, tools) {
  const { text, color, fill } = tools;
  let count = 0;
  let body = text(20, 27, 'The target follows the realized clipped area', { width: 320, weight: 700 });
  for (let row = 0; row < picture.side; row++) for (let column = 0; column < picture.side; column++) {
    const inserted = column >= picture.patch[0] && column < picture.patch[0] + picture.patch[2] && row >= picture.patch[1] && row < picture.patch[1] + picture.patch[3];
    if (inserted) count++;
    const tone = inserted ? 'output' : 'input';
    body += `<rect data-picture-inserted-pixel="${inserted}" x="${45 + column * 25}" y="${110 + row * 25}" width="25" height="25" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1"></rect>`;
  }
  body += `<rect x="${45 + picture.patch[0] * 25}" y="${110 + picture.patch[1] * 25}" width="${picture.patch[2] * 25}" height="${picture.patch[3] * 25}" style="fill:none;stroke:${color('warning')};stroke-width:2;stroke-dasharray:5 4"></rect>`
    + text(20, 415, 'Sampled 7 x 6; realized 5 x 6', { width: 320, weight: 700 })
    + text(20, 465, '30 inserted pixels / 100 = 0.3', { width: 320, weight: 700, tone: 'output' });
  assert.equal(count, 30);
  assert.equal(count / picture.side ** 2, model.rows[1][1]);
  return { body, height: 513 };
}

function removeStructure(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Empty values still occupy physical slots', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 111 + stage * 274;
    const rows = picture.values.filter((row, index) => stage === 0 || index !== picture.removed);
    body += text(20, vertical - 29, stage ? 'Remove the whole zero channel: 2 x 3' : 'Only zero the values: still 3 x 3', { width: 320, weight: 700 });
    for (const [row, values] of rows.entries()) for (const [column, value] of values.entries()) {
      const tone = value ? 'input' : 'neutral';
      body += `<rect data-picture-pruning-slot="${stage},${row},${column}" x="${75 + column * 70}" y="${vertical + row * 62}" width="70" height="62" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>`
        + text(110 + column * 70, vertical + row * 62 + 38, value, { width: 60, anchor: 'middle', size: 24, tone });
    }
  }
  body += text(20, 572, 'Nine slots become six only after structural removal.', { width: 320, tone: 'soft' });
  return { body, height: 622 };
}

function relabelPopulation(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'The inputs match; their label rules reverse', { width: 320, weight: 700 });
  for (const [domain, labels] of picture.labels.entries()) {
    const vertical = 151 + domain * 212;
    body += text(20, vertical - 59, `Domain ${domain ? 'B' : 'A'}: two zeros, two ones`, { width: 320, weight: 700 });
    for (const [index, label] of labels.entries()) {
      const horizontal = 55 + index * 83;
      const tone = label ? 'output' : 'input';
      body += `<g data-picture-domain-member="${index}" data-picture-domain="${domain}">${circle(horizontal, vertical, 22, tone)}${text(horizontal, vertical + 7, picture.inputs[index], { width: 40, anchor: 'middle', size: 22, weight: 700 })}</g>`
        + text(horizontal, vertical + 64, `y=${label}`, { width: 75, anchor: 'middle', tone, weight: 700 });
    }
  }
  body += text(20, 489, 'Equal input frequencies cannot reveal this label shift.', { width: 320, tone: 'soft' });
  return { body, height: 540 };
}

function centeredGroup(picture, model, tools) {
  const { text, circle, line } = tools;
  const mean = picture.rewards.reduce((sum, reward) => sum + reward, 0) / picture.rewards.length;
  let body = text(20, 27, 'Every output succeeds; no output beats its peers', { width: 320, weight: 700 });
  for (const [index, reward] of picture.rewards.entries()) {
    const horizontal = 55 + index * 83;
    const centered = reward - mean;
    assert.equal(centered, 0);
    body += circle(horizontal, 140, 23, 'output') + text(horizontal, 147, reward, { width: 40, anchor: 'middle', weight: 700, size: 22 })
      + arrowBetween([horizontal, 172], [horizontal, 273], 'output', tools)
      + text(horizontal + 22, 219, '-1', { width: 34, anchor: 'middle', weight: 700, tone: 'soft' })
      + circle(horizontal, 326, 7, 'neutral') + text(horizontal, 382, centered, { width: 60, anchor: 'middle', size: 24, weight: 700, tone: 'warning' });
  }
  body += text(20, 448, 'Centered reward is zero for all four.', { width: 320, weight: 700 })
    + text(20, 507, 'No relative ordering is supplied by these rewards.', { width: 320, tone: 'soft' });
  return { body, height: 559 };
}

function axisPooling(picture, model, tools) {
  const { text, color, fill, number } = tools;
  let body = text(20, 27, 'Change the pooling axis, not the data', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 129 + stage * 260;
    body += text(20, vertical - 48, stage ? 'BatchNorm: pool each feature column' : 'LayerNorm: pool within each sample row', { width: 320, weight: 700 });
    for (let row = 0; row < 2; row++) for (let column = 0; column < 2; column++) {
      const tone = (stage ? column : row) ? 'output' : 'input';
      body += `<rect x="${55 + column * 88}" y="${vertical + row * 68}" width="88" height="68" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`
        + text(99 + column * 88, vertical + row * 68 + 43, picture.values[row][column], { width: 75, anchor: 'middle', size: 25, weight: 700 });
    }
    for (let group = 0; group < 2; group++) {
      const values = stage ? picture.values.map(row => row[group]) : picture.values[group];
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      const tone = group ? 'output' : 'input';
      body += stage ? text(99 + group * 88, vertical + 181, `mean ${mean}`, { width: 95, anchor: 'middle', weight: 700, tone }) : text(340, vertical + group * 68 + 42, `mean ${mean}`, { width: 91, anchor: 'end', weight: 700, tone });
    }
  }
  return { body, height: 617 };
}

function countPooling(picture, model, tools) {
  const { text, circle, number } = tools;
  const merged = picture.shards[0].map((value, category) => value + picture.shards[1][category]);
  assert.deepEqual(merged, model.values.at(-1));
  const tones = ['output', 'warning', 'input'];
  let body = text(20, 27, 'Pool the cases before computing the ratio', { width: 320, weight: 700 });
  for (const [stage, counts] of [...picture.shards, merged].entries()) {
    const vertical = 121 + stage * 156;
    body += text(20, vertical - 35, ['Shard A', 'Shard B', 'Combined cases'][stage], { width: 320, weight: 700 });
    for (const [category, count] of counts.entries()) {
      const horizontal = 25 + category * 112;
      for (let index = 0; index < count; index++) body += circle(horizontal + index % 3 * 21, vertical + Math.floor(index / 3) * 21, 6, tones[category], true);
      body += text(horizontal, vertical + 95, `${['TP', 'FP', 'FN'][category]} ${count}`, { width: 100, weight: 700, tone: tones[category] });
    }
  }
  const f1 = 2 * merged[0] / (2 * merged[0] + merged[1] + merged[2]);
  body += text(20, 606, `F1 = 18 / 22 = ${number(f1)}`, { width: 320, weight: 700, tone: 'output' });
  return { body, height: 651 };
}

function assignedCenters(picture, model, tools) {
  const { text, circle, line } = tools;
  const horizontal = value => 35 + value * 29;
  let body = text(20, 27, 'Fixed assignments pull each center to their mean', { width: 320, weight: 700 });
  for (const [cluster, points] of picture.points.entries()) {
    const vertical = 151 + cluster * 237;
    const mean = points.reduce((sum, value) => sum + value, 0) / points.length;
    const tone = cluster ? 'output' : 'input';
    body += line(35, vertical, 325, vertical);
    for (const point of points) body += circle(horizontal(point), vertical, 7, tone, true) + text(horizontal(point), vertical + 36, point, { width: 55, anchor: point === 10 ? 'end' : 'middle', tone });
    body += circle(horizontal(picture.before[cluster]), vertical - 58, 11, tone)
      + arrowBetween([horizontal(picture.before[cluster]), vertical - 58], [horizontal(mean), vertical - 58], tone, tools)
      + circle(horizontal(mean), vertical - 58, 9, tone, true)
      + text(20, vertical + 100, `Cluster ${cluster + 1}: mean ${mean}`, { width: 320, weight: 700, tone });
  }
  return { body, height: 531 };
}

function uncertaintyPin(picture, model, tools) {
  const { text, line, circle, color, fill } = tools;
  const horizontal = value => 35 + (value + 3) / 6 * 290;
  let body = text(20, 27, 'An observation pins the band only locally', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 178 + stage * 268;
    const tone = stage ? 'output' : 'input';
    const points = stage ? model.series[1][1] : [[-3, 1], [3, 1]];
    const upper = points.map(([value, width]) => [horizontal(value), vertical - width * 67]);
    const lower = [...points].reverse().map(([value, width]) => [horizontal(value), vertical + width * 67]);
    body += text(20, vertical - 111, stage ? 'Posterior: zero width at the observed input' : 'Prior: equal uncertainty across inputs', { width: 320, weight: 700 })
      + `<path d="${[...upper, ...lower].map((point, index) => `${index ? 'L' : 'M'}${point.join(' ')}`).join('')}Z" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></path>`
      + line(35, vertical, 325, vertical, 'neutral', true);
    if (stage) body += circle(horizontal(0), vertical, 8, 'focus', true);
  }
  assert.equal(Math.sqrt(1 - Math.exp(0)), 0);
  body += text(20, 590, 'The envelope reopens away from the observation.', { width: 320, tone: 'soft' });
  return { body, height: 639 };
}

function softenMass(picture, model, tools) {
  const { text, color, fill, number } = tools;
  let body = text(20, 27, 'Heating redistributes mass without changing order', { width: 320, weight: 700 });
  const tones = ['input', 'output', 'state'];
  for (const [stage, temperature] of picture.temperatures.entries()) {
    const vertical = 151 + stage * 227;
    const exponentials = picture.logits.map(value => Math.exp(value / temperature));
    const denominator = exponentials.reduce((sum, value) => sum + value, 0);
    const probabilities = exponentials.map(value => value / denominator);
    assert(probabilities[0] > probabilities[1] && probabilities[1] > probabilities[2]);
    body += text(20, vertical - 51, `Temperature ${temperature}: same three classes`, { width: 320, weight: 700 });
    let position = 20;
    for (const [index, probability] of probabilities.entries()) {
      body += `<rect data-picture-softened-class="${index}" data-picture-temperature="${temperature}" data-picture-probability="${probability}" x="${position}" y="${vertical}" width="${probability * 320}" height="55" style="fill:${fill(tones[index])};stroke:${color(tones[index])};stroke-width:1.5"></rect>`;
      position += probability * 320;
      body += text(20 + index * 111, vertical + 105, `${String.fromCharCode(65 + index)} ${number(probability)}`, { width: 104, weight: 700, tone: tones[index] });
    }
  }
  body += text(20, 554, 'A stays largest; B and C gain probability.', { width: 320, tone: 'soft' });
  return { body, height: 601 };
}

function orderingCredit(picture, model, tools) {
  const { text, color, fill, number } = tools;
  const [empty, aloneA, aloneB, both] = picture.values;
  const credits = [[aloneA - empty, both - aloneA], [both - aloneB, aloneB - empty]];
  const mean = credits[0].map((value, index) => (value + credits[1][index]) / 2);
  assert.equal(mean.reduce((sum, value) => sum + value, 0), both - empty);
  let body = text(20, 27, 'Average the marginal credit from both orders', { width: 320, weight: 700 });
  for (const [stage, gains] of [...credits, mean].entries()) {
    const vertical = 129 + stage * 151;
    body += text(20, vertical - 36, ['A arrives first: A gets 2; B gets 4', 'B arrives first: B gets 1; A gets 5', 'Average: A gets 3.5; B gets 2.5'][stage], { width: 320, weight: 700 });
    let position = 20;
    for (const [feature, gain] of gains.entries()) {
      const tone = feature ? 'output' : 'input';
      body += `<rect x="${position}" y="${vertical}" width="${320 * gain / both}" height="44" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`;
      position += 320 * gain / both;
    }
  }
  body += text(20, 547, 'Same total payoff; different entry order.', { width: 320, tone: 'soft' });
  return { body, height: 595 };
}

function massCutoffs(picture, model, tools) {
  const { text, circle, color } = tools;
  assert.equal(picture.counts.reduce((sum, value) => sum + value, 0), 100);
  let body = text(20, 27, 'Keep token identities; change the selection rule', { width: 320, weight: 700 });
  for (const [stage, selected] of picture.selected.entries()) {
    const vertical = 133 + stage * 229;
    const kept = picture.counts.slice(0, selected).reduce((sum, value) => sum + value, 0);
    body += text(20, vertical - 43, stage ? 'Top-p 0.8 needs the third token' : 'Top-k two stops after the second token', { width: 320, weight: 700 });
    let group = 0;
    let boundary = picture.counts[0];
    for (let member = 0; member < 100; member++) {
      while (member >= boundary) { group++; boundary += picture.counts[group]; }
      body += `<g data-picture-mass-member="${member}" data-picture-stage="${stage}" data-picture-kept="${group < selected}">${circle(37 + member % 20 * 15, vertical + Math.floor(member / 20) * 15, 3, group < selected ? 'input' : 'neutral', true)}</g>`;
    }
    body += text(20, vertical + 117, `Retained mass ${kept / 100}; ${selected} tokens`, { width: 320, weight: 700, tone: 'input' });
  }
  assert.equal(picture.counts.slice(0, 2).reduce((sum, value) => sum + value, 0), 70);
  assert.equal(picture.counts.slice(0, 3).reduce((sum, value) => sum + value, 0), 85);
  return { body, height: 530 };
}

function jointPaths(picture, model, tools) {
  const { text, color, fill, number } = tools;
  assert.equal(picture.masses.reduce((sum, value) => sum + value, 0), 1);
  const rowTotals = [picture.masses[0] + picture.masses[1], picture.masses[2] + picture.masses[3]];
  const columnTotals = [picture.masses[0] + picture.masses[2], picture.masses[1] + picture.masses[3]];
  assert(rowTotals[0] > rowTotals[1] && columnTotals[1] > columnTotals[0]);
  assert.equal(Math.max(...picture.masses), picture.masses[0]);
  let body = text(20, 27, 'Pool path masses along each time axis', { width: 320, weight: 700 });
  for (const [index, mass] of picture.masses.entries()) {
    const horizontal = 30 + index % 2 * 170;
    const bottom = 278 + Math.floor(index / 2) * 169;
    const tone = index === 0 ? 'output' : index === 1 ? 'warning' : 'input';
    body += `<rect data-picture-path-mass="${mass}" x="${horizontal}" y="${bottom - mass * 400}" width="130" height="${mass * 400}" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`
      + text(horizontal + 65, bottom + 37, `${['AA', 'AB', 'BA', 'BB'][index]} ${mass}`, { width: 155, anchor: 'middle', weight: 700, tone });
  }
  body += text(20, 539, `Marginals choose A ${number(rowTotals[0])}, then B ${number(columnTotals[1])}`, { width: 320, weight: 700, tone: 'warning' })
    + text(20, 604, 'Largest whole path: AA 0.35, not AB 0.30', { width: 320, weight: 700, tone: 'output' });
  return { body, height: 657 };
}

function iterativeStep(picture, model, tools) {
  const { text, circle, line, number } = tools;
  const horizontal = value => 180 + value * 76;
  let body = text(20, 27, 'Same quadratic; different update distances', { width: 320, weight: 700 });
  for (const [lane, rate] of picture.rates.entries()) {
    const vertical = 171 + lane * 211;
    const gain = 1 - 2 * rate;
    const tone = lane === 2 ? 'warning' : lane === 1 ? 'output' : 'input';
    body += text(20, vertical - 82, `Rate ${rate}: multiply the coordinate by ${number(gain)}`, { width: 320, weight: 700, tone })
      + line(35, vertical, 325, vertical) + line(180, vertical - 50, 180, vertical + 51, 'neutral', true);
    for (let step = 0; step <= picture.steps; step++) {
      const value = gain ** step;
      body += `<g data-picture-update-step="${lane},${step}" data-picture-coordinate="${value}">${circle(horizontal(value), vertical - 30 + step * 20, 5, tone, true)}</g>`;
      if (step) body += arrowBetween([horizontal(gain ** (step - 1)), vertical - 50 + step * 20], [horizontal(value), vertical - 30 + step * 20], tone, tools);
    }
  }
  assert(Math.abs(1 - 2 * picture.rates[0]) < 1 && Math.abs(1 - 2 * picture.rates[1]) < 1 && Math.abs(1 - 2 * picture.rates[2]) > 1);
  body += text(20, 700, 'Oscillation can contract; oversized steps grow.', { width: 320, tone: 'soft' });
  return { body, height: 750 };
}

function constraintContact(picture, model, tools) {
  const { text, circle, line, color, fill, number } = tools;
  const position = point => [140 + point[0] * 75, 235 - point[1] * 75];
  const l1 = [1, 0];
  const l2 = picture.target.map(value => value / Math.hypot(...picture.target));
  assert(Math.abs(l1[0]) + Math.abs(l1[1]) === 1);
  assert(Math.abs(Math.hypot(...l2) - 1) < 1e-12 && l2[1] > 0);
  assert(picture.target[1] <= picture.target[0] - 1);
  let body = text(20, 27, 'The nearest feasible point meets a boundary', { width: 320, weight: 700 })
    + line(...position([-1.2, 0]), ...position([2.5, 0]))
    + line(...position([0, -1.2]), ...position([0, 1.2]))
    + `<path d="${[[1, 0], [0, 1], [-1, 0], [0, -1]].map((point, index) => `${index ? 'L' : 'M'}${position(point).join(' ')}`).join('')}Z" style="fill:${fill('input')};stroke:${color('input')};stroke-width:2"></path>`
    + `<circle cx="140" cy="235" r="75" style="fill:none;stroke:${color('output')};stroke-width:2"></circle>`
    + circle(...position(picture.target), 8, 'focus', true)
    + text(340, 136, 'target (2, 0.4)', { width: 160, anchor: 'end', weight: 700, tone: 'focus' });
  for (const [index, optimum] of [l1, l2].entries()) {
    const tone = index ? 'output' : 'input';
    body += line(...position(picture.target), ...position(optimum), tone, true) + circle(...position(optimum), 5, tone, true);
  }
  body += text(20, 407, 'Diamond contact: (1, 0)', { width: 320, weight: 700, tone: 'input' })
    + text(20, 464, `Circle contact: (${number(l2[0])}, ${number(l2[1])})`, { width: 320, weight: 700, tone: 'output' })
    + text(20, 523, 'Only the corner lies exactly on the coordinate axis.', { width: 320, tone: 'soft' });
  return { body, height: 574 };
}

function fixedNoise(picture, model, tools) {
  const { text, circle, line, number } = tools;
  const horizontal = value => 35 + value * 72;
  let body = text(20, 27, 'Pin the noise draw; move its transformation', { width: 320, weight: 700 })
    + circle(180, 101, 24, 'focus') + text(180, 107, 'e=-1', { width: 45, anchor: 'middle', weight: 700 });
  for (const [stage, mean] of picture.means.entries()) {
    const vertical = 249 + stage * 207;
    const sample = mean + picture.sigma * picture.epsilon;
    body += text(20, vertical - 68, `Mean ${mean}; same scale 0.5 and noise -1`, { width: 320, weight: 700 })
      + line(35, vertical, 323, vertical)
      + circle(horizontal(mean), vertical, 7, 'input', true)
      + arrowBetween([horizontal(mean), vertical - 35], [horizontal(sample), vertical - 35], 'output', tools)
      + circle(horizontal(sample), vertical, 7, 'output', true)
      + text(20, vertical + 64, `Sample ${number(sample)}`, { width: 320, weight: 700, tone: 'output' });
  }
  return { body, height: 568 };
}

function scaleBeforeRound(picture, model, tools) {
  const { text, circle, line, color } = tools;
  const original = 2 ** picture.exponent;
  const scale = 2 ** picture.scaleExponent;
  const floor = 2 ** -24;
  assert(original < floor / 2);
  assert.equal(original * scale, 2 ** -10);
  assert.equal((original * scale) / scale, original);
  let body = text(20, 27, 'Scale before crossing the representation floor', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 172 + stage * 239;
    body += text(20, vertical - 73, stage ? 'Scale first: the intermediate survives' : 'Round first: the small gradient becomes zero', { width: 320, weight: 700 })
      + circle(65, vertical, 23, stage ? 'input' : 'warning')
      + text(65, vertical + 6, stage ? '2^-10' : '2^-30', { width: 54, anchor: 'middle', size: 14, weight: 700 })
      + line(180, vertical - 47, 180, vertical + 47, 'warning', true)
      + text(180, vertical + 77, 'FP16 rounding', { width: 180, anchor: 'middle', tone: 'soft', size: 15 })
      + arrowBetween([94, vertical], [250, vertical], stage ? 'input' : 'warning', tools)
      + circle(281, vertical, 23, stage ? 'output' : 'neutral')
      + text(281, vertical + 6, stage ? '2^-10' : '0', { width: 54, anchor: 'middle', size: stage ? 14 : 23, weight: 700 });
  }
  body += text(20, 549, 'Divide by 2^20 before the step: recover 2^-30.', { width: 320, weight: 700, tone: 'output' });
  return { body, height: 606 };
}

function rankCredit(picture, model, tools) {
  const { text, circle, line, number } = tools;
  let hits = 0;
  let rr = 0;
  let average = 0;
  let body = text(20, 27, 'Same hits; different credit collection rules', { width: 320, weight: 700 });
  for (const [index, relevant] of picture.relevant.entries()) {
    const horizontal = 55 + index * 83;
    const rank = index + 1;
    if (relevant) { hits++; if (!rr) rr = 1 / rank; average += hits / rank; }
    body += circle(horizontal, 130, 18, relevant ? 'output' : 'neutral', relevant) + text(horizontal, 185, `rank ${rank}`, { width: 77, anchor: 'middle', size: 15, weight: 700 });
    if (relevant) body += arrowBetween([horizontal, 217], [horizontal, 307], 'output', tools) + text(horizontal, 352, `${hits}/${rank}`, { width: 75, anchor: 'middle', weight: 700, tone: 'output' });
    if (relevant && hits === 1) body += text(horizontal, 436, 'RR stops here', { width: 133, anchor: 'middle', weight: 700, tone: 'input' });
  }
  average /= hits;
  assert.equal(rr, .5);
  assert.equal(average, .5);
  body += text(20, 530, 'AP gathers both hit precisions, then averages.', { width: 320, weight: 700, tone: 'output' })
    + text(20, 595, 'Both equal 0.5 here, but for different reasons.', { width: 320, tone: 'soft' });
  return { body, height: 645 };
}

function rankDisplacement(picture, model, tools) {
  const { text, circle, line, number } = tools;
  let body = text(20, 27, 'Equal rank moves cross unequal discount gaps', { width: 320, weight: 700 });
  const discount = rank => 1 / Math.log2(rank + 1);
  for (const [stage, move] of picture.moves.entries()) {
    const vertical = 162 + stage * 229;
    body += text(20, vertical - 78, `Move the relevant item: ${move[0]} to ${move[1]}`, { width: 320, weight: 700 });
    for (let rank = 1; rank <= 10; rank++) body += circle(34 + (rank - 1) * 32, vertical, 10 * Math.sqrt(discount(rank)), move.includes(rank) ? 'warning' : 'neutral', true);
    body += arrowBetween([34 + (move[0] - 1) * 32, vertical + 41], [34 + (move[1] - 1) * 32, vertical + 41], 'warning', tools, 3)
      + text(20, vertical + 112, `Lost discount ${number(discount(move[0]) - discount(move[1]))}`, { width: 320, weight: 700, tone: 'warning' });
  }
  return { body, height: 550 };
}

function quantizationRuler(picture, model, tools) {
  const { text, circle, line, number } = tools;
  const horizontal = value => 180 + value * 120;
  let body = text(20, 27, 'Same small signs; different snapping intervals', { width: 320, weight: 700 })
    + text(20, 82, 'Zoomed small-channel ruler: minus one to one', { width: 320, tone: 'soft' });
  for (const [stage, scale] of picture.scales.entries()) {
    const vertical = 230 + stage * 237;
    body += text(20, vertical - 98, stage ? 'Own scale preserves the two signs' : 'Outlier 254 sets shared scale two', { width: 320, weight: 700 });
    for (const [index, value] of picture.values.entries()) {
      const rounded = Math.round(value / scale);
      assert(rounded >= -127 && rounded <= 127);
      const quantized = rounded === 0 ? 0 : rounded;
      const reconstructed = quantized * scale;
      const row = vertical - 30 + index * 60;
      body += line(60, row, 300, row) + circle(horizontal(value), row, 6, 'input', true)
        + arrowBetween([horizontal(value), row - 20], [horizontal(reconstructed), row - 20], stage ? 'output' : 'warning', tools)
        + circle(horizontal(reconstructed), row, 7, stage ? 'output' : 'warning')
        + text(340, row + 39, `code ${Object.is(quantized, -0) ? 0 : quantized}`, { width: 103, anchor: 'end', size: 15, weight: 700, tone: stage ? 'output' : 'warning' });
      assert.equal(reconstructed, stage ? value : 0);
    }
  }
  return { body, height: 597 };
}

function oneSidedPlateau(picture, model, tools) {
  const { text, circle, line, color } = tools;
  const horizontal = ratio => 35 + ratio * 145;
  let body = text(20, 27, 'The objective flattens; the ratio can continue', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 250 + stage * 304;
    const objective = ratio => stage ? Math.min(-ratio, -.8) : Math.min(ratio, 1.2);
    const ordinate = value => vertical - value * 65;
    const tone = stage ? 'warning' : 'output';
    body += text(20, vertical - 154, stage ? 'Negative advantage: plateau below ratio 0.8' : 'Positive advantage: plateau above ratio 1.2', { width: 320, weight: 700 })
      + `<path d="${Array.from({ length: 41 }, (_, index) => { const ratio = index / 20; return `${index ? 'L' : 'M'}${horizontal(ratio)} ${ordinate(objective(ratio))}`; }).join('')}" style="fill:none;stroke:${color(tone)};stroke-width:3"></path>`;
    const probe = stage ? picture.probes[0] : picture.probes[1];
    body += circle(horizontal(probe), ordinate(objective(probe)), 7, tone, true)
      + text(20, vertical + (stage ? 169 : 71), `Ratio ${probe} is still outside the clipping interval.`, { width: 320, weight: 700, tone });
  }
  return { body, height: 800 };
}

function conditioningPaths(picture, model, tools) {
  const { text, circle, line, color } = tools;
  let body = text(20, 27, 'Conditioning depends on the path motif', { width: 320, weight: 700 });
  for (let lane = 0; lane < 4; lane++) {
    const collider = lane >= 2;
    const conditioned = lane % 2 === 1;
    const open = collider ? conditioned : !conditioned;
    const vertical = 128 + lane * 148;
    body += text(20, vertical - 43, `${collider ? 'Collider' : 'Chain'}: ${conditioned ? 'middle observed' : 'middle unobserved'}`, { width: 320, weight: 700 });
    for (const [index, horizontal] of [53, 148, 242].entries()) body += circle(horizontal, vertical, 19, index === 1 && conditioned ? 'focus' : 'input') + text(horizontal, vertical + 5, ['A', 'M', 'B'][index], { width: 32, anchor: 'middle', weight: 700 });
    body += arrowBetween([75, vertical], [124, vertical], 'input', tools)
      + (collider ? arrowBetween([220, vertical], [172, vertical], 'input', tools) : arrowBetween([171, vertical], [219, vertical], 'input', tools))
      + text(340, vertical + 6, open ? 'open' : 'closed', { width: 73, anchor: 'end', size: 15, weight: 700, tone: open ? 'output' : 'warning' });
    if (!open) body += line(140, vertical + 34, 156, vertical + 50, 'warning') + line(156, vertical + 34, 140, vertical + 50, 'warning');
    else body += line(53, vertical + 42, 242, vertical + 42, 'output', true);
  }
  return { body, height: 667 };
}

function confoundingIntervention(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Assignment and intervention have different arrows', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const top = 134 + stage * 262;
    body += text(20, top - 57, stage ? 'Intervene: cut the assignment edge into X' : 'Observe: common cause U assigns both', { width: 320, weight: 700 });
    const positions = [[180, top], [63, top + 112], [297, top + 112]];
    for (const [index, position] of positions.entries()) body += circle(...position, 22, index === 0 ? 'state' : index === 1 ? 'input' : 'output') + text(position[0], position[1] + 6, ['U', 'X', 'Y'][index], { width: 37, anchor: 'middle', weight: 700 });
    body += arrowBetween([195, top + 19], [279, top + 93], 'output', tools);
    if (!stage) body += arrowBetween([165, top + 19], [81, top + 93], 'input', tools);
    else body += line(165, top + 19, 145, top + 36, 'warning', true) + line(126, top + 63, 81, top + 93, 'warning', true) + line(126, top + 40, 145, top + 60, 'warning') + line(145, top + 40, 126, top + 60, 'warning');
  }
  body += text(20, 582, 'U: engagement. X: messages. Y: churn.', { width: 320, tone: 'soft' });
  return { body, height: 633 };
}

function actorCriticRoutes(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Both heads update the shared trunk', { width: 320, weight: 700 });
  const nodes = [{ point: [74, 137], name: 'Actor', tone: 'input' }, { point: [284, 137], name: 'Critic', tone: 'output' }, { point: [180, 332], name: 'Shared', tone: 'state' }, { point: [284, 470], name: 'Adv.', tone: 'warning' }];
  for (const node of nodes) body += circle(...node.point, 31, node.tone) + text(node.point[0], node.point[1] + 6, node.name, { width: 59, anchor: 'middle', size: 15, weight: 700 });
  body += arrowBetween([90, 166], [163, 302], 'input', tools, 3)
    + arrowBetween([269, 166], [198, 302], 'output', tools, 3)
    + line(104, 158, 316, 213, 'warning', true) + line(316, 213, 316, 361, 'warning', true)
    + line(316, 399, 298, 439, 'warning', true)
    + line(305, 366, 327, 388, 'warning') + line(327, 366, 305, 388, 'warning')
    + text(20, 469, 'Actor-to-advantage gradient stops', { width: 217, weight: 700, tone: 'warning' })
    + text(20, 555, 'Shared-feature gradients remain active.', { width: 320, tone: 'soft' });
  return { body, height: 606 };
}

function reverseFanout(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'One loss sends one accumulated reverse flow', { width: 320, weight: 700 })
    + circle(180, 115, 27, 'focus') + text(180, 121, 'Loss', { width: 48, anchor: 'middle', weight: 700 })
    + arrowBetween([180, 146], [180, 233], 'state', tools, 3)
    + text(340, 202, 'dL/du = 2', { width: 119, anchor: 'end', weight: 700, tone: 'state' })
    + circle(180, 265, 27, 'state') + text(180, 271, 'u', { width: 48, anchor: 'middle', weight: 700 });
  for (let branch = 0; branch < 2; branch++) {
    const horizontal = branch ? 282 : 78;
    const gradient = 2 * (branch ? 3 : 1);
    body += arrowBetween([180, 297], [horizontal, 390], branch ? 'output' : 'input', tools, 3)
      + circle(horizontal, 423, 28, branch ? 'output' : 'input')
      + text(horizontal, 429, branch ? 'b' : 'a', { width: 45, anchor: 'middle', weight: 700 })
      + text(horizontal, 503, `gradient ${gradient}`, { width: 142, anchor: 'middle', weight: 700, tone: branch ? 'output' : 'input' });
  }
  return { body, height: 552 };
}

function forwardBypass(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'Keep the raw stream; add a learned correction', { width: 320, weight: 700 })
    + circle(180, 104, 26, 'input') + text(180, 110, 'x', { width: 44, anchor: 'middle', weight: 700 });
  body += arrowBetween([159, 129], [75, 228], 'input', tools, 3)
    + arrowBetween([201, 129], [285, 228], 'state', tools)
    + text(74, 280, 'Unchanged x', { width: 126, anchor: 'middle', weight: 700, tone: 'input' })
    + text(285, 280, 'Norm, then F', { width: 120, anchor: 'middle', weight: 700, tone: 'state' })
    + arrowBetween([75, 319], [162, 416], 'input', tools, 3)
    + arrowBetween([285, 319], [199, 416], 'output', tools)
    + circle(180, 449, 27, 'output') + text(180, 455, '+', { width: 48, anchor: 'middle', size: 28, weight: 700 })
    + text(20, 534, 'x + F(norm(x)); not F(norm(x)) alone', { width: 320, weight: 700 })
    + text(20, 591, 'A zero correction leaves the original x intact.', { width: 320, tone: 'soft' });
  return { body, height: 644 };
}

function dependentGeneration(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'Training samples a time; generation follows time', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 157 + stage * 239;
    const tokens = stage ? ['z3', 'z2', 'z1', 'x0'] : ['x0', 't', 'xt', 'loss'];
    body += text(20, vertical - 58, stage ? 'Generation: each output feeds the next step' : 'Training: choose one noise time for this example', { width: 320, weight: 700 });
    for (const [index, token] of tokens.entries()) {
      const horizontal = 50 + index * 84;
      body += circle(horizontal, vertical, 24, stage ? 'output' : 'input') + text(horizontal, vertical + 6, token, { width: 44, anchor: 'middle', size: 16, weight: 700 });
      if (index) body += arrowBetween([horizontal - 56, vertical], [horizontal - 28, vertical], stage ? 'output' : 'input', tools);
    }
  }
  body += text(20, 522, 'The reverse chain retains predecessor dependencies.', { width: 320, tone: 'soft' });
  return { body, height: 575 };
}

function hiddenEmissions(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'Dependence travels through the hidden lane', { width: 320, weight: 700 });
  for (let time = 0; time < 3; time++) {
    const horizontal = 58 + time * 122;
    body += circle(horizontal, 143, 25, 'state') + text(horizontal, 149, `H${time + 1}`, { width: 45, anchor: 'middle', weight: 700 })
      + arrowBetween([horizontal, 172], [horizontal, 306], 'state', tools)
      + circle(horizontal, 339, 25, 'input') + text(horizontal, 345, `O${time + 1}`, { width: 45, anchor: 'middle', weight: 700 });
    if (time) body += arrowBetween([horizontal - 93, 143], [horizontal - 29, 143], 'state', tools, 3);
  }
  body += text(20, 434, 'Each observation is a local child of its hidden state.', { width: 320, weight: 700 })
    + text(20, 501, 'There is no direct observation-to-observation edge.', { width: 320, tone: 'soft' });
  return { body, height: 553 };
}

function boundClimb(picture, model, tools) {
  const { text, line, circle, color, number } = tools;
  const position = (parameter, objective) => [35 + (parameter + 1) * 145, 325 - (objective + 4) * 60];
  let body = text(20, 27, 'Climb a supporting bound that touches now', { width: 320, weight: 700 });
  for (const [index, series] of model.series.entries()) body += `<path d="${series[1].map(([parameter, objective], vertex) => `${vertex ? 'L' : 'M'}${position(parameter, objective).join(' ')}`).join('')}" style="fill:none;stroke:${color(index ? 'warning' : 'output')};stroke-width:2.5"></path>`;
  const objective = parameter => -(parameter ** 2);
  const bound = parameter => objective(parameter) - (parameter - picture.current) ** 2;
  assert.equal(objective(picture.current), bound(picture.current));
  assert(objective(picture.next) >= bound(picture.next) && bound(picture.next) >= objective(picture.current));
  body += circle(...position(picture.current, objective(picture.current)), 7, 'focus', true)
    + arrowBetween(position(picture.current, bound(picture.current) - .25), position(picture.next, bound(picture.next) - .25), 'warning', tools, 3)
    + line(...position(picture.next, bound(picture.next)), ...position(picture.next, objective(picture.next)), 'output', true)
    + circle(...position(picture.next, objective(picture.next)), 6, 'output', true)
    + text(20, 395, 'Contact at 0.5; bound maximum at 0.25', { width: 320, weight: 700 })
    + text(20, 458, `Objective rises: -0.25 to ${number(objective(picture.next))}`, { width: 320, weight: 700, tone: 'output' });
  return { body, height: 511 };
}

function complexTurn(picture, model, tools) {
  const { text, circle, line, color, number } = tools;
  let body = text(20, 27, 'Complex multiplication turns the real-axis endpoint', { width: 320, weight: 700 });
  for (const [stage, factors] of picture.factors.entries()) {
    const center = [180, 180 + stage * 257];
    const exponent = factors.reduce((sum, turn) => sum + turn, 0);
    const score = Math.round(Math.cos(exponent * Math.PI / 2));
    assert.equal(score, stage ? 1 : -1);
    body += text(20, center[1] - 104, stage ? 'Swap: 1 times i times -i' : 'Forward: i times i times 1', { width: 320, weight: 700 })
      + circle(...center, 61, 'neutral') + line(center[0] - 78, center[1], center[0] + 78, center[1]) + line(center[0], center[1] - 78, center[0], center[1] + 78)
      + arrowBetween(center, [center[0] + score * 61, center[1]], stage ? 'output' : 'warning', tools, 3)
      + text(20, center[1] + 119, `Real part ${score}`, { width: 320, weight: 700, tone: stage ? 'output' : 'warning' });
  }
  return { body, height: 608 };
}

function uncertaintyChoice(picture, model, tools) {
  const { text, circle, line, number } = tools;
  const horizontal = value => 35 + (value - .5) / .3 * 290;
  let body = text(20, 27, 'One pair of arms; three choice rules', { width: 320, weight: 700 });
  for (const [stage, values] of picture.values.entries()) {
    const vertical = 142 + stage * 177;
    body += text(20, vertical - 57, ['Estimated mean: choose A', 'Mean plus bonus: choose B', 'One posterior draw: choose B'][stage], { width: 320, weight: 700 }) + line(35, vertical, 325, vertical);
    const winner = values[0] > values[1] ? 0 : 1;
    for (const [arm, value] of values.entries()) body += circle(horizontal(value), vertical, 7, arm ? 'output' : 'input', true)
      + text(horizontal(value), vertical + (arm === winner ? -18 : 40), `${arm ? 'B' : 'A'} ${value}`, { width: 107, anchor: 'middle', weight: 700, tone: arm ? 'output' : 'input' });
  }
  return { body, height: 582 };
}

function pathScoreMovement(picture, model, tools) {
  const { text, circle, line, number } = tools;
  const horizontal = value => 70 + (value + 1) * 51;
  let body = text(20, 27, 'Transition cost can undo local emission gains', { width: 320, weight: 700 });
  for (const [path, terms] of picture.paths.entries()) {
    const vertical = 142 + path * 171;
    const score = terms[0] + terms[1];
    assert.equal(score, model.rows[path][1]);
    body += text(20, vertical - 50, `${picture.labels[path]}: emissions ${terms[0]}, transition ${terms[1]}`, { width: 320, weight: 700 })
      + line(35, vertical, 325, vertical)
      + circle(horizontal(terms[0]), vertical, 7, 'input', true);
    if (terms[1]) body += arrowBetween([horizontal(terms[0]), vertical - 26], [horizontal(score), vertical - 26], 'warning', tools, 3);
    body += circle(horizontal(score), vertical, 8, score < 0 ? 'warning' : 'output')
      + text(20, vertical + 64, `Complete-path score ${score}`, { width: 320, weight: 700, tone: score < 0 ? 'warning' : 'output' });
  }
  return { body, height: 599 };
}

function opposedDirections(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Equal-length gradients can point opposite ways', { width: 320, weight: 700 })
    + circle(65, 178, 7, 'focus', true)
    + arrowBetween([65, 178], [285, 178], 'input', tools, 4)
    + arrowBetween([285, 276], [65, 276], 'warning', tools, 4)
    + line(285, 178, 285, 276, 'neutral', true)
    + text(20, 123, 'Task A: one unit to the right', { width: 320, weight: 700, tone: 'input' })
    + text(20, 341, 'Task B: one unit back to the origin', { width: 320, weight: 700, tone: 'warning' })
    + text(20, 422, 'Sum zero; both norms still equal one.', { width: 320, weight: 700 });
  assert.deepEqual(model.vectors[0][1].map((value, index) => value + model.vectors[1][1][index]), [0, 0]);
  return { body, height: 476 };
}

function varianceSquare(picture, model, tools) {
  const { text, color, fill, line } = tools;
  let body = text(20, 27, 'Variance is the square of the signal scale', { width: 320, weight: 700 });
  for (const [stage, deviation] of picture.deviations.entries()) {
    const horizontal = stage ? 282 : 28;
    const unit = 18;
    const tone = stage ? 'output' : 'input';
    for (let row = 0; row < deviation; row++) for (let column = 0; column < deviation; column++) body += `<rect data-picture-variance-tile="${stage},${row},${column}" x="${horizontal + column * unit}" y="${286 - (row + 1) * unit}" width="${unit}" height="${unit}" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1"></rect>`;
    body += text(horizontal, 328, `SD ${deviation}`, { width: 85, weight: 700, tone }) + text(stage ? 340 : 20, 384, `Variance ${deviation ** 2}`, { width: 150, anchor: stage ? 'end' : 'start', weight: 700, tone });
    assert.equal(deviation ** 2, model.rows[stage][1]);
  }
  body += text(20, 452, 'Divide SD by eight; divide area by sixty-four.', { width: 320, weight: 700 })
    + text(20, 519, 'This calculation requires independent unit-variance products.', { width: 320, tone: 'soft' });
  return { body, height: 578 };
}

function saturatedGradient(picture, model, tools) {
  const { text, circle, line, number } = tools;
  const probability = picture.probability;
  const gradients = [1 - probability, 2 * probability * (1 - probability) ** 2];
  assert.equal(gradients[0], .99);
  assert(Math.abs(gradients[1] - .019602) < 1e-12);
  let body = text(20, 27, 'Wrong confidence passes through different loss routes', { width: 320, weight: 700 });
  for (const [stage, gradient] of gradients.entries()) {
    const vertical = 153 + stage * 215;
    body += text(20, vertical - 59, stage ? 'Squared error includes the small sigmoid gate' : 'Cross-entropy cancellation preserves sensitivity', { width: 320, weight: 700 });
    const size = Math.max(2, gradient * 22);
    body += `<path d="M35 ${vertical - size}H325V${vertical + size}H35Z" style="fill:${tools.fill(stage ? 'warning' : 'output')};stroke:${tools.color(stage ? 'warning' : 'output')};stroke-width:1.5"></path>`
      + text(20, vertical + 91, `Absolute logit gradient ${number(gradient)}`, { width: 320, weight: 700, tone: stage ? 'warning' : 'output' });
  }
  return { body, height: 508 };
}

function sharedFactorArea(picture, model, tools) {
  const { text, color, fill, circle } = tools;
  let body = text(20, 27, 'One shared factor contributes to all asset pairs', { width: 320, weight: 700 });
  for (let row = 0; row < 2; row++) for (let column = 0; column < 2; column++) {
    const horizontal = 55 + column * 170;
    const vertical = 143 + row * 154;
    const product = picture.loadings[row] * picture.loadings[column];
    assert.equal(product, model.values[row][column]);
    for (let across = 0; across < picture.loadings[column]; across++) for (let down = 0; down < picture.loadings[row]; down++) body += `<rect x="${horizontal + across * 32}" y="${vertical + down * 32}" width="32" height="32" style="fill:${fill('input')};stroke:${color('input')};stroke-width:1.5"></rect>`;
    body += text(horizontal, vertical + 103, `Shared ${product}`, { width: 123, weight: 700, tone: 'input' });
    if (row === column) body += circle(horizontal + 83, vertical + 22, 7, 'warning', true);
  }
  body += text(20, 501, 'Independent noise adds only to diagonal entries.', { width: 320, weight: 700, tone: 'warning' })
    + text(20, 566, 'Blue areas: shared covariance. Red marks: private noise.', { width: 320, tone: 'soft' });
  return { body, height: 621 };
}

function parameterVersion(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Accumulate first; advance the parameter version once', { width: 320, weight: 700 });
  for (const [index, gradient] of picture.gradients.entries()) {
    const horizontal = 55 + index * 83;
    body += circle(horizontal, 142, 22, 'input') + text(horizontal, 148, gradient, { width: 36, anchor: 'middle', weight: 700 })
      + text(horizontal, 214, 'v0', { width: 55, anchor: 'middle', weight: 700, tone: 'state' })
      + arrowBetween([horizontal, 246], [180, 340], 'input', tools);
  }
  const total = picture.gradients.reduce((sum, value) => sum + value, 0);
  body += text(180, 386, `Accumulated gradient ${total}`, { width: 320, anchor: 'middle', weight: 700 })
    + line(35, 432, 325, 432, 'warning', true) + text(20, 476, 'One optimizer step crosses the boundary', { width: 320, weight: 700 })
    + circle(180, 547, 24, 'output') + text(180, 553, 'v1', { width: 45, anchor: 'middle', weight: 700, tone: 'output' });
  assert.equal(total, 10);
  return { body, height: 594 };
}

function frozenFactor(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  let body = text(20, 27, 'Freeze the item factors; solve user rows independently', { width: 320, weight: 700 })
    + `<rect x="82" y="94" width="196" height="62" rx="4" style="fill:${fill('state')};stroke:${color('state')};stroke-width:2"></rect>`
    + text(180, 132, 'Fixed V = (1, 2)', { width: 178, anchor: 'middle', weight: 700, tone: 'state' });
  for (const [user, row] of picture.ratings.entries()) {
    const horizontal = user ? 278 : 82;
    const optimum = row.reduce((sum, value, index) => sum + value * picture.fixed[index], 0) / picture.fixed.reduce((sum, value) => sum + value ** 2, 0);
    assert.deepEqual(picture.fixed.map(value => value * optimum), row);
    body += arrowBetween([180, 168], [horizontal, 270], 'state', tools)
      + circle(horizontal, 306, 27, 'output') + text(horizontal, 312, optimum, { width: 48, anchor: 'middle', size: 22, weight: 700 })
      + text(horizontal, 386, `User ${user + 1}: (${row.join(', ')})`, { width: 150, anchor: 'middle', weight: 700 });
  }
  body += text(20, 475, 'No user-row solve changes the frozen item vector.', { width: 320, tone: 'soft' });
  return { body, height: 530 };
}

function coordinateRelabel(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'The same prediction has different coordinate labels', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 152 + stage * 229;
    body += text(20, vertical - 56, stage ? 'Coordinate b: a = 2b' : 'Coordinate a: prediction equals a', { width: 320, weight: 700 }) + line(35, vertical, 325, vertical);
    for (let tick = 0; tick < 5; tick++) body += circle(35 + tick * 72.5, vertical, tick === 2 ? 7 : 3, stage ? 'warning' : 'input', true)
      + text(35 + tick * 72.5, vertical + 45, stage ? tick / picture.factor : tick, { width: 58, anchor: tick === 0 ? 'start' : tick === 4 ? 'end' : 'middle', weight: 700 });
  }
  body += text(20, 520, 'Same prediction a=2, b=1; same loss four.', { width: 320, weight: 700 })
    + text(20, 584, 'Coordinate curvatures differ: two versus eight.', { width: 320, tone: 'soft' });
  return { body, height: 639 };
}

function hardForwardSurrogate(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'A hard forward gate can use a declared reverse surrogate', { width: 320, weight: 700 });
  body += circle(65, 175, 27, 'input') + text(65, 181, picture.input, { width: 48, anchor: 'middle', weight: 700 })
    + arrowBetween([97, 175], [252, 175], 'input', tools, 3)
    + circle(283, 175, 27, 'output') + text(283, 181, '1', { width: 48, anchor: 'middle', weight: 700 })
    + text(20, 264, 'Forward: threshold 0.2 to one', { width: 320, weight: 700 })
    + arrowBetween([283, 375], [65, 375], 'warning', tools, 3)
    + text(20, 333, 'Chosen straight-through multiplier: one', { width: 320, weight: 700, tone: 'warning' })
    + text(20, 451, 'True local hard-threshold derivative: zero', { width: 320, weight: 700 })
    + text(20, 522, 'The surrogate is not the derivative of the hard gate.', { width: 320, tone: 'soft' });
  assert(picture.input > 0);
  return { body, height: 581 };
}

function balancedReservoirs(picture, model, tools) {
  const { text, circle, line, number } = tools;
  const flows = picture.masses.map((mass, index) => mass * picture.rates[index]);
  assert.equal(flows[0], flows[1]);
  let body = text(20, 27, 'Unequal reservoirs exchange equal mass flows', { width: 320, weight: 700 });
  for (let state = 0; state < 2; state++) {
    const horizontal = state ? 280 : 82;
    const tone = state ? 'output' : 'input';
    body += circle(horizontal, 263, 55 * Math.sqrt(picture.masses[state]), tone)
      + text(horizontal, 269, state ? 'B' : 'A', { width: 48, anchor: 'middle', size: 25, weight: 700 })
      + text(horizontal, 369, state ? 'mass 1/3' : 'mass 2/3', { width: 136, anchor: 'middle', weight: 700, tone });
  }
  body += arrowBetween([126, 208], [249, 208], 'input', tools, 8) + arrowBetween([249, 315], [126, 315], 'output', tools, 8)
    + text(20, 144, 'A sends 0.2 of its mass; B sends 0.4', { width: 320, weight: 700 })
    + text(20, 451, 'Each direction transfers 2/15 per step.', { width: 320, weight: 700 })
    + text(20, 519, 'The distribution stays; individual states still move.', { width: 320, tone: 'soft' });
  return { body, height: 577 };
}

function weightedJacobian(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Gather every output contribution into each input', { width: 320, weight: 700 });
  for (let output = 0; output < 2; output++) for (let input = 0; input < 2; input++) {
    const horizontal = 30 + input * 177;
    const vertical = 142 + output * 174;
    const tone = output ? 'output' : 'input';
    const derivative = picture.jacobian[output][input];
    const upstream = picture.upstream[output];
    for (let across = 0; across < upstream; across++) for (let down = 0; down < derivative; down++) body += `<rect x="${horizontal + across * 23}" y="${vertical + down * 23}" width="23" height="23" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>`;
    body += text(horizontal, vertical + 128, `${upstream} x ${derivative} = ${upstream * derivative}`, { width: 130, weight: 700, tone });
  }
  const sums = picture.jacobian[0].map((value, input) => value * picture.upstream[0] + picture.jacobian[1][input] * picture.upstream[1]);
  assert.deepEqual(sums, model.values.at(-1));
  body += text(20, 528, 'Input 1 receives 5; input 2 receives 8.', { width: 320, weight: 700 })
    + text(20, 593, 'Sum down each input column, not across output rows.', { width: 320, tone: 'soft' });
  return { body, height: 648 };
}

function tokenBuckets(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  let body = text(20, 27, 'Same text, same code length, different token buckets', { width: 320, weight: 700 });
  for (const [stage, group] of picture.groups.entries()) {
    const vertical = 142 + stage * 241;
    body += text(20, vertical - 49, stage ? 'Two tokens: ab, cd' : 'Four tokens: a, b, c, d', { width: 320, weight: 700 });
    for (let start = 0; start < picture.letters.length; start += group) body += `<rect x="${25 + start * 80}" y="${vertical}" width="${group * 80 - 10}" height="62" rx="4" style="fill:${fill(stage ? 'output' : 'input')};stroke:${color(stage ? 'output' : 'input')};stroke-width:2"></rect>`;
    for (const [index, letter] of picture.letters.entries()) body += text(60 + index * 80, vertical + 39, letter, { width: 60, anchor: 'middle', size: 25, weight: 700 });
    const tokens = picture.letters.length / group;
    body += text(20, vertical + 123, `4 bits / ${tokens} tokens; perplexity ${2 ** (picture.bits / tokens)}`, { width: 320, weight: 700 });
  }
  body += text(20, 572, 'Shared unit: four bits / four characters = one.', { width: 320, tone: 'soft' });
  return { body, height: 628 };
}

function unboundedValley(picture, model, tools) {
  const { text, line, color, fill } = tools;
  let body = text(20, 27, 'The zero-loss valley has no end in y', { width: 320, weight: 700 })
    + `<rect x="93" y="104" width="174" height="306" style="fill:${fill('input')};stroke:${color('input')};stroke-width:1.5;stroke-dasharray:5 4"></rect>`
    + line(35, 256, 325, 256)
    + arrowBetween([180, 256], [180, 90], 'output', tools, 4)
    + arrowBetween([180, 256], [180, 425], 'output', tools, 4)
    + text(20, 473, 'Inside the strip: x squared is at most one.', { width: 320, weight: 700 })
    + text(20, 541, 'At x=0, any y still gives zero loss.', { width: 320, weight: 700, tone: 'output' });
  assert.equal(0 ** 2, 0);
  return { body, height: 595 };
}

function reciprocalRates(picture, model, tools) {
  const { text, color, fill, number } = tools;
  const reciprocals = picture.rates.map(rate => 1 / rate);
  const result = 2 / reciprocals.reduce((sum, value) => sum + value, 0);
  assert(Math.abs(result - model.rows.at(-1)[1]) < 1e-12);
  let body = text(20, 27, 'The weaker rate dominates reciprocal cost', { width: 320, weight: 700 });
  for (const [index, value] of reciprocals.entries()) {
    const vertical = 148 + index * 182;
    const tone = index ? 'warning' : 'input';
    body += text(20, vertical - 39, `${index ? 'Recall' : 'Precision'} ${picture.rates[index]}: reciprocal ${number(value)}`, { width: 320, weight: 700 })
      + `<rect x="20" y="${vertical}" width="${value * 30}" height="44" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>`;
  }
  body += text(20, 463, 'F1 = 2 / (1/0.9 + 1/0.1) = 0.18', { width: 320, weight: 700 })
    + text(20, 532, 'Not the arithmetic mean 0.5.', { width: 320, tone: 'soft' });
  return { body, height: 586 };
}

function eventCountWait(picture, model, tools) {
  const { text, line, circle, number } = tools;
  let elapsed = 0;
  const events = picture.waits.map(wait => { elapsed += wait; return elapsed; });
  const horizontal = value => 35 + value / 5 * 290;
  let body = text(20, 27, 'Count marks, or measure the spaces between them', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 177 + stage * 230;
    body += text(20, vertical - 62, stage ? 'A waiting time: the gap to the next event' : 'A count: three events in this five-second window', { width: 320, weight: 700 }) + line(35, vertical, 325, vertical);
    for (const event of events) body += circle(horizontal(event), vertical, 6, 'input', true);
    if (stage) body += arrowBetween([horizontal(events[0]), vertical + 42], [horizontal(events[1]), vertical + 42], 'output', tools, 3)
      + text(20, vertical + 101, 'Gap 1.42 seconds, not 1.42 events', { width: 320, weight: 700, tone: 'output' });
    else body += text(20, vertical + 64, 'Count = 3, an integer-valued observation', { width: 320, weight: 700 });
  }
  return { body, height: 565 };
}

function sharedNoiseFloor(picture, model, tools) {
  const { text, color, fill, number } = tools;
  const unit = .05;
  let body = text(20, 27, 'Averaging removes private error, not shared error', { width: 320, weight: 700 });
  for (const [stage, trees] of picture.trees.entries()) {
    const vertical = 146 + stage * 161;
    const privateVariance = picture.private / trees;
    body += text(20, vertical - 40, `${trees} trees: shared ${picture.common}, private ${number(privateVariance)}`, { width: 320, weight: 700 });
    const sharedUnits = Math.round(picture.common / unit);
    const privateUnits = Math.round(privateVariance / unit);
    for (let tile = 0; tile < sharedUnits + privateUnits; tile++) {
      const tone = tile < sharedUnits ? 'warning' : 'input';
      body += `<rect data-picture-noise-tile="${tile < sharedUnits ? 'shared' : 'private'}" data-picture-stage="${stage}" x="${25 + tile * 15}" y="${vertical}" width="15" height="44" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>`;
    }
  }
  body += text(20, 558, 'The same four red tiles remain at every size.', { width: 320, weight: 700, tone: 'warning' });
  return { body, height: 616 };
}

function documentMatches(picture, model, tools) {
  const { text, color, fill, number } = tools;
  let body = text(20, 27, 'Count the matches; keep document length explicit', { width: 320, weight: 700 });
  for (const [index, [frequency, length]] of picture.documents.entries()) {
    const horizontal = 20 + index * 112;
    const score = frequency * 2.2 / (frequency + 1.2 * (.25 + .75 * length));
    body += text(horizontal, 100, `${frequency} matches`, { width: 104, weight: 700 }) + text(horizontal, 143, `length ${length}x`, { width: 104, tone: 'soft' });
    for (let word = 0; word < 10 * length; word++) {
      const tone = word < frequency ? 'input' : 'neutral';
      body += `<rect x="${horizontal + word % 5 * 17}" y="${193 + Math.floor(word / 5) * 29}" width="15" height="22" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>`;
    }
    body += text(horizontal, 383, `score ${number(score)}`, { width: 104, weight: 700, tone: 'output' });
  }
  body += text(20, 475, 'More matches help less. Extra length dilutes the same matches.', { width: 320, weight: 700 });
  return { body, height: 540 };
}

function fixedFloor(picture, model, tools) {
  const { text, color, fill, number } = tools;
  let body = text(20, 27, 'Keep the floor; decay only the remaining headroom', { width: 320, weight: 700 });
  for (const [stage, phase] of picture.phases.entries()) {
    const horizontal = 34 + stage * 109;
    const headroom = (1 - picture.floor) * (1 + Math.cos(Math.PI * phase)) / 2;
    const total = picture.floor + headroom;
    const floorHeight = picture.floor * 210;
    body += `<rect x="${horizontal}" y="${362 - floorHeight}" width="64" height="${floorHeight}" style="fill:${fill('warning')};stroke:${color('warning')};stroke-width:2"></rect><rect x="${horizontal}" y="${362 - total * 210}" width="64" height="${headroom * 210}" style="fill:${fill('input')};stroke:${color('input')};stroke-width:2"></rect>`
      + text(horizontal + 32, 411, number(total), { width: 90, anchor: 'middle', weight: 700, size: 23 });
  }
  body += text(20, 489, 'Midpoint: floor 0.2 + half of 0.8 = 0.6', { width: 320, weight: 700 })
    + text(20, 555, 'A chosen schedule, not an observed learning curve.', { width: 320, tone: 'soft' });
  return { body, height: 613 };
}

function savedRebuiltStates(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  let body = text(20, 27, 'Saved boundaries and rebuilt segments compete', { width: 320, weight: 700 });
  for (const [stage, segment] of picture.segments.entries()) {
    const vertical = 132 + stage * 196;
    const saved = picture.layers / segment;
    body += text(20, vertical - 43, `Segment ${segment}: ${saved} saved boundaries`, { width: 320, weight: 700 });
    for (let layer = 0; layer < picture.layers; layer++) body += circle(25 + layer * 20, vertical, layer % segment === 0 ? 5 : 2.5, layer % segment === 0 ? 'input' : 'neutral', true);
    for (let state = 0; state < segment; state++) body += `<rect x="${25 + state * 20}" y="${vertical + 39}" width="17" height="20" style="fill:${fill('warning')};stroke:${color('warning')};stroke-width:1.5"></rect>`;
    body += text(20, vertical + 108, `${saved} saved + ${segment} rebuilt = ${saved + segment}`, { width: 320, weight: 700 });
  }
  assert.equal(picture.layers / 4 + 4, 8);
  return { body, height: 682 };
}

function repeatOrCondition(picture, model, tools) {
  const { text, line, circle, color } = tools;
  const horizontal = value => 35 + value * 290;
  let body = text(20, 27, 'Repeat samples, or condition on this sample', { width: 320, weight: 700 })
    + text(20, 83, 'Repeated intervals move around fixed truth', { width: 320, weight: 700 });
  for (const [index, interval] of model.intervals.entries()) body += line(horizontal(interval[0]), 140 + index * 21, horizontal(interval[1]), 140 + index * 21, 'input');
  body += line(horizontal(.5), 125, horizontal(.5), 346, 'focus', true)
    + text(20, 392, 'Illustrative intervals, not a measured coverage rate', { width: 320, tone: 'soft' })
    + text(20, 463, 'One sample: three successes, two failures', { width: 320, weight: 700 });
  const points = Array.from({ length: 81 }, (_, index) => { const probability = index / 80; return [probability, 60 * probability ** 3 * (1 - probability) ** 2]; });
  assert(Math.abs(60 * (1 / 4 - 2 / 5 + 1 / 6) - 1) < 1e-12);
  body += densityArea(points, 660, horizontal, 60, 'output', tools)
    + text(20, 708, 'Posterior Beta(4,3): uncertainty over probability', { width: 320, weight: 700, tone: 'output' });
  return { body, height: 769 };
}

function equalKernelArea(picture, model, tools) {
  const { text, color, fill } = tools;
  const total = picture.products.reduce((sum, value) => sum + value, 0);
  assert.equal(total, picture.base ** 2);
  assert.equal(total, 36);
  let body = text(20, 27, 'Two exact routes gather the same thirty-six units', { width: 320, weight: 700 })
    + text(20, 96, 'Explicit feature products', { width: 320, weight: 700, tone: 'input' });
  for (const [product, contribution] of picture.products.entries()) {
    for (let tile = 0; tile < contribution; tile++) body += `<rect data-picture-kernel-unit="explicit" x="${25 + product * 53 + tile % 2 * 16}" y="${211 - Math.floor(tile / 2) * 16}" width="16" height="16" style="fill:${fill('input')};stroke:${color('input')};stroke-width:1"></rect>`;
    body += text(41 + product * 53, 268, contribution, { width: 47, anchor: 'middle', weight: 700 });
  }
  body += text(20, 341, 'Compact kernel: (coordinate dot 4 + 2) squared', { width: 320, weight: 700, tone: 'output' });
  for (let row = 0; row < picture.base; row++) for (let column = 0; column < picture.base; column++) body += `<rect data-picture-kernel-unit="compact" x="${130 + column * 16}" y="${406 + row * 16}" width="16" height="16" style="fill:${fill('output')};stroke:${color('output')};stroke-width:1"></rect>`;
  body += text(20, 572, '36 = 36. No feature-expansion approximation.', { width: 320, weight: 700 });
  return { body, height: 633 };
}

function averagingPopulation(picture, model, tools) {
  const { text, circle, color, fill, number } = tools;
  const divergence = (left, right) => left.reduce((sum, value, index) => sum + value * Math.log(value / right[index]), 0);
  const values = [divergence(picture.target, picture.candidates[0]), divergence(picture.target, picture.candidates[1]), divergence(picture.candidates[0], picture.target), divergence(picture.candidates[1], picture.target)];
  assert(values[0] < values[1] && values[3] < values[2]);
  let body = text(20, 27, 'Changing the averaging population can reverse rank', { width: 320, weight: 700 });
  for (let row = 0; row < 4; row++) {
    const vertical = 135 + row * 170;
    const weights = row < 2 ? picture.target : picture.candidates[row - 2];
    const preferred = row === 0 || row === 3;
    const tone = preferred ? 'output' : 'warning';
    body += text(20, vertical - 40, `${row < 2 ? 'Average under P' : 'Average under Q' + (row - 1)}: candidate Q${row % 2 + 1}`, { width: 320, weight: 700 });
    for (let member = 0; member < 100; member++) body += circle(25 + member % 10 * 8, vertical + Math.floor(member / 10) * 8, 2.3, member < weights[0] * 100 ? 'input' : 'state', true);
    body += `<rect data-picture-divergence="${values[row]}" x="175" y="${vertical + 18}" width="${values[row] / .16 * 150}" height="28" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>`
      + text(175, vertical + 90, `KL ${number(values[row])}`, { width: 161, weight: 700, tone });
  }
  body += text(20, 790, 'Forward keeps P fixed. Reverse averages each Q.', { width: 320, tone: 'soft' });
  return { body, height: 847 };
}

function neighborhoodMessages(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Only the declared neighborhood reaches B', { width: 320, weight: 700 });
  const positions = [[65, 153], [180, 318], [295, 153], [65, 459]];
  for (const [index, point] of positions.entries()) body += circle(...point, 25, index === 3 ? 'neutral' : 'input') + text(point[0], point[1] + 6, `${String.fromCharCode(65 + index)}:${picture.features[index]}`, { width: 53, anchor: 'middle', size: index === 3 ? 14 : 16, weight: 700 });
  body += arrowBetween([80, 177], [164, 290], 'input', tools) + arrowBetween([280, 177], [197, 290], 'input', tools)
    + line(85, 440, 111, 411, 'warning', true) + line(132, 385, 161, 342, 'warning', true)
    + line(111, 385, 131, 405, 'warning') + line(131, 385, 111, 405, 'warning')
    + text(20, 551, 'Mean of A, B, C = (1+2+3)/3 = 2', { width: 320, weight: 700 })
    + text(20, 615, 'D=100 has no edge and contributes nothing.', { width: 320, tone: 'soft' });
  assert.deepEqual(model.values[1], [1, 1, 1]);
  return { body, height: 670 };
}

function bandwidthCeiling(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  let body = text(20, 27, 'Same byte bandwidth; more work per byte', { width: 320, weight: 700 });
  for (const [stage, intensity] of picture.intensities.entries()) {
    const vertical = 150 + stage * 225;
    body += text(20, vertical - 49, `Arithmetic intensity ${intensity} FLOP per byte`, { width: 320, weight: 700 })
      + circle(60, vertical, 25, 'input') + text(60, vertical + 6, 'byte', { width: 46, anchor: 'middle', size: 14, weight: 700 })
      + arrowBetween([90, vertical], [212, vertical], 'input', tools, 3);
    for (let work = 0; work < intensity; work++) body += `<rect x="${246 + work * 36}" y="${vertical - 18}" width="33" height="36" style="fill:${fill('output')};stroke:${color('output')};stroke-width:2"></rect>`;
    body += text(20, vertical + 91, `At 1 TB/s: memory ceiling ${intensity} TFLOP/s`, { width: 320, weight: 700, tone: 'output' });
  }
  body += text(20, 570, 'A ceiling is not a measured achieved speed.', { width: 320, tone: 'soft' });
  return { body, height: 626 };
}

function normalizerShift(picture, model, tools) {
  const { text, color, fill, number } = tools;
  let body = text(20, 27, 'Raw mass changes; normalized shares repeat', { width: 320, weight: 700 });
  const originals = picture.logits.map(value => Math.exp(value));
  const denominator = originals.reduce((sum, value) => sum + value, 0);
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 125 + stage * 192;
    const masses = originals.map(value => value * Math.exp(stage * picture.shift));
    const total = masses.reduce((sum, value) => sum + value, 0);
    body += text(20, vertical - 38, stage ? 'Add two to both logits: multiply raw mass by exp(2)' : 'Original logits zero and one', { width: 320, weight: 700 });
    let position = 20;
    for (const [index, mass] of masses.entries()) { body += `<rect x="${position}" y="${vertical}" width="${mass * 10}" height="38" style="fill:${fill(index ? 'output' : 'input')};stroke:${color(index ? 'output' : 'input')};stroke-width:1.5"></rect>`; position += mass * 10; }
    body += text(20, vertical + 99, `Log normalizer ${number(Math.log(total))}`, { width: 320, weight: 700 });
  }
  for (let stage = 0; stage < 2; stage++) {
    let position = 20;
    for (const [index, mass] of originals.entries()) { const share = mass / denominator; body += `<rect x="${position}" y="${519 + stage * 65}" width="${share * 320}" height="31" style="fill:${fill(index ? 'output' : 'input')};stroke:${color(index ? 'output' : 'input')};stroke-width:1.5"></rect>`; position += share * 320; }
  }
  body += text(20, 671, 'Both normalize to 0.269 and 0.731.', { width: 320, weight: 700 });
  return { body, height: 728 };
}

function retainedFrontier(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  let body = text(20, 27, 'The retained frontier excludes weaker identities', { width: 320, weight: 700 });
  for (const [stage, retained] of picture.retained.entries()) {
    const vertical = 151 + stage * 241;
    body += text(20, vertical - 63, stage ? 'New eight replaces seven; retain nine and eight' : 'From the first two batches: retain nine and seven', { width: 320, weight: 700 });
    for (const [index, value] of picture.values.entries()) {
      const horizontal = 40 + index * 70;
      const kept = retained.includes(value);
      body += circle(horizontal, vertical, kept ? 22 : 14, kept ? 'output' : 'neutral') + text(horizontal, vertical + 6, value, { width: 38, anchor: 'middle', size: 20, weight: 700, tone: kept ? 'output' : 'soft' });
    }
    body += text(20, vertical + 102, `Retained minimum ${Math.min(...retained)}; discarded three stays below it`, { width: 320, weight: 700 });
  }
  return { body, height: 565 };
}

function shapeContract(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Addition requires all tensor axes to match', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 141 + stage * 256;
    const shortcut = stage ? picture.output : picture.input;
    body += text(20, vertical - 55, stage ? 'Project the shortcut to the branch shape' : 'Unchanged shortcut cannot add to this branch', { width: 320, weight: 700 });
    for (const [axis, value] of shortcut.entries()) {
      const horizontal = 20 + axis * 112;
      const target = picture.output[axis];
      const match = value === target;
      body += `<rect x="${horizontal}" y="${vertical}" width="98" height="55" style="fill:${fill(match ? 'output' : 'warning')};stroke:${color(match ? 'output' : 'warning')};stroke-width:2"></rect>`
        + text(horizontal + 49, vertical + 33, `${value}:${target}`, { width: 89, anchor: 'middle', size: 20, weight: 700 })
        + text(horizontal + 49, vertical + 98, ['height', 'width', 'channels'][axis], { width: 101, anchor: 'middle', size: 15 });
    }
  }
  body += text(20, 590, 'Every axis matches only after the projection.', { width: 320, tone: 'soft' });
  return { body, height: 648 };
}

function featureSummary(picture, model, tools) {
  const { text, circle, number } = tools;
  const sums = picture.query.map((value, feature) => picture.features.reduce((sum, row, index) => sum + row[feature] * picture.values[index], 0));
  const normalizers = picture.query.map((value, feature) => picture.features.reduce((sum, row) => sum + row[feature], 0));
  assert.deepEqual(sums, [8, 10]);
  assert.deepEqual(normalizers, [2, 2]);
  const numerator = sums.reduce((sum, value, feature) => sum + value * picture.query[feature], 0);
  const denominator = normalizers.reduce((sum, value, feature) => sum + value * picture.query[feature], 0);
  let body = text(20, 27, 'Store feature sums; let each query read them', { width: 320, weight: 700 });
  for (const [index, value] of picture.values.entries()) body += circle(57 + index * 123, 142, 22, 'input') + text(57 + index * 123, 148, value, { width: 38, anchor: 'middle', weight: 700 });
  body += text(20, 222, 'Key features: (1,0), (0,1), (1,1)', { width: 320, weight: 700 })
    + arrowBetween([75, 250], [75, 335], 'input', tools) + arrowBetween([285, 250], [285, 335], 'input', tools)
    + text(20, 383, 'S=(8,10)', { width: 143, weight: 700, tone: 'output' }) + text(340, 383, 'z=(2,2)', { width: 143, anchor: 'end', weight: 700, tone: 'output' })
    + text(20, 462, 'Query features (1,2) weight both sums.', { width: 320, weight: 700 })
    + text(20, 539, `${numerator} / ${denominator} = ${number(numerator / denominator)}`, { width: 320, size: 23, weight: 700, tone: 'output' });
  return { body, height: 595 };
}

function positionEntryPoint(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Position enters different stages of the calculation', { width: 320, weight: 700 })
    + text(20, 91, 'RoPE: rotate coordinates before the dot product', { width: 320, weight: 700 });
  for (const [index, label] of ['q', 'k'].entries()) {
    const horizontal = 80 + index * 200;
    body += circle(horizontal, 172, 27, 'input') + text(horizontal, 178, label, { width: 45, anchor: 'middle', weight: 700 })
      + arrowBetween([horizontal, 205], [horizontal, 278], 'input', tools)
      + text(horizontal, 325, 'rotate', { width: 110, anchor: 'middle', weight: 700, tone: 'input' });
  }
  body += text(20, 411, 'ALiBi: subtract distance cost from the score', { width: 320, weight: 700 });
  for (const [index, distance] of picture.distances.entries()) body += text(55 + index * 125, 499, `d=${distance}`, { width: 89, anchor: 'middle', weight: 700 })
    + text(55 + index * 125, 561, distance ? -picture.slope * distance : 0, { width: 89, anchor: 'middle', size: 23, weight: 700, tone: 'warning' });
  return { body, height: 614 };
}

function alignmentAxes(picture, model, tools) {
  const { text, circle, line } = tools;
  const minimumTime = Math.min(...picture.path.map(point => point[0]));
  const minimumOutput = Math.min(...picture.path.map(point => point[1]));
  const point = state => [65 + (state[1] - minimumOutput) * 112, 153 + (state[0] - minimumTime) * 112];
  let body = text(20, 27, 'Token edges move output; blank edges move time', { width: 320, weight: 700 });
  for (let time = 0; time < 3; time++) for (let output = 0; output < 3; output++) body += circle(65 + output * 112, 153 + time * 112, 4, 'neutral', true);
  for (const [index, state] of picture.path.entries()) {
    body += circle(...point(state), 9, 'input', true);
    if (index) {
      const previous = picture.path[index - 1];
      const token = state[0] === previous[0];
      assert(token ? state[1] === previous[1] + 1 : state[0] === previous[0] + 1 && state[1] === previous[1]);
      body += arrowBetween(point(previous), point(state), token ? 'input' : 'output', tools, 3);
    }
  }
  body += text(20, 455, 'Horizontal: emit token, keep audio time', { width: 320, weight: 700, tone: 'input' })
    + text(20, 521, 'Vertical: emit blank, advance audio time', { width: 320, weight: 700, tone: 'output' });
  return { body, height: 579 };
}

function denominatorMembers(picture, model, tools) {
  const { text, circle, number } = tools;
  const hits = model.values[0][0];
  const denominators = [hits + model.values[0][1], hits + model.values[1][0]];
  let body = text(20, 27, 'The same hit identities occupy different wholes', { width: 320, weight: 700 });
  for (const [stage, count] of denominators.entries()) {
    const vertical = 142 + stage * 225;
    body += text(20, vertical - 45, `${stage ? 'Predicted positives' : 'Actual positives'}: ${count}`, { width: 320, weight: 700 });
    for (let member = 0; member < count; member++) body += `<g data-picture-denominator-member="${member < hits ? member : stage ? member + 1 : member}" data-picture-stage="${stage}">${circle(30 + member % 20 * 15, vertical + Math.floor(member / 20) * 15, 3, member < hits ? 'output' : 'neutral', true)}</g>`;
    body += text(20, vertical + 125, `${stage ? 'Precision' : 'Recall'} ${hits}/${count} = ${number(hits / count)}`, { width: 320, weight: 700, tone: 'output' });
  }
  return { body, height: 550 };
}

function commonRotation(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Turn both directions together; keep their relative angle', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const center = [180, 190 + stage * 266];
    const angle = stage * picture.angle;
    const query = [Math.cos(angle), Math.sin(angle)];
    const key = [-Math.sin(angle), Math.cos(angle)];
    assert(Math.abs(query.reduce((sum, value, index) => sum + value * key[index], 0)) < 1e-12);
    body += text(20, center[1] - 107, stage ? 'Both turned forty-five degrees' : 'Original: a ninety-degree angle', { width: 320, weight: 700 });
    for (const [index, direction] of [query, key].entries()) body += arrowBetween(center, [center[0] + direction[0] * 81, center[1] - direction[1] * 81], index ? 'output' : 'input', tools, 3);
    body += circle(...center, 3, 'focus', true);
  }
  body += text(20, 611, 'Dot product stays zero under the common rotation.', { width: 320, tone: 'soft' });
  return { body, height: 669 };
}

function queryRows(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'The query destinations determine output rows', { width: 320, weight: 700 });
  for (let source = 0; source < picture.sources; source++) body += circle(58 + source * 122, 134, 24, 'state') + text(58 + source * 122, 140, `K${source}`, { width: 43, anchor: 'middle', weight: 700 });
  for (let query = 0; query < picture.queries; query++) {
    const horizontal = 90 + query * 180;
    for (let source = 0; source < picture.sources; source++) body += arrowBetween([58 + source * 122, 163], [horizontal, 303], 'state', tools);
    body += circle(horizontal, 337, 27, 'input') + text(horizontal, 343, `Q${query}`, { width: 49, anchor: 'middle', weight: 700 })
      + arrowBetween([horizontal, 369], [horizontal, 449], 'input', tools)
      + text(horizontal, 501, `Output ${query}`, { width: 150, anchor: 'middle', weight: 700, tone: 'output' });
  }
  assert.equal(picture.queries, model.values.length);
  return { body, height: 554 };
}

function lostBoundary(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Different boundaries collapse to the same coarse cell', { width: 320, weight: 700 });
  for (const [pattern, values] of picture.patterns.entries()) {
    const vertical = 142 + pattern * 235;
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    assert.equal(mean, .5);
    for (const [index, value] of values.entries()) body += `<rect x="${23 + index * 49}" y="${vertical}" width="49" height="57" style="fill:${fill(value ? 'output' : 'neutral')};stroke:${color(value ? 'output' : 'neutral')};stroke-width:1.5"></rect>`;
    body += arrowBetween([231, vertical + 29], [267, vertical + 29], 'warning', tools)
      + `<rect x="283" y="${vertical}" width="55" height="57" style="fill:${fill('warning')};stroke:${color('warning')};stroke-width:2"></rect>`
      + text(310, vertical + 34, mean, { width: 49, anchor: 'middle', weight: 700 });
  }
  body += text(20, 524, 'Enlarging 0.5 cannot reveal which pattern was pooled.', { width: 320, weight: 700 })
    + text(20, 592, 'Fine-resolution evidence supplies the missing distinction.', { width: 320, tone: 'soft' });
  return { body, height: 649 };
}

function packedIsolation(picture, model, tools) {
  const { text, color, fill, line } = tools;
  let body = text(20, 27, 'Causal order stays inside each packed example', { width: 320, weight: 700 });
  for (let row = 0; row < 4; row++) for (let column = 0; column < 4; column++) {
    const allowed = model.values[row][column] === 1;
    const tone = allowed ? row < 2 ? 'input' : 'output' : 'neutral';
    body += `<rect x="${61 + column * 61}" y="${147 + row * 61}" width="61" height="61" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>`;
    if (allowed) body += text(91 + column * 61, 185 + row * 61, '1', { width: 45, anchor: 'middle', weight: 700, tone });
  }
  body += line(61, 269, 305, 269, 'warning') + line(183, 147, 183, 391, 'warning')
    + text(20, 457, 'Rows and columns: A0, A1, B0, B1', { width: 320, weight: 700 })
    + text(20, 528, 'B reads its own earlier tokens, not earlier example A.', { width: 320, tone: 'soft' });
  assert.equal(model.values[3][1], 0);
  assert.equal(model.values[3][2], 1);
  return { body, height: 587 };
}

function velocityFold(picture, model, tools) {
  const { text, circle, line, number } = tools;
  let body = text(20, 27, 'Retain velocity, then add the new direction', { width: 320, weight: 700 });
  for (const [lane, gradients] of picture.gradients.entries()) {
    const vertical = 158 + lane * 280;
    const baseline = vertical + 70;
    let velocity = 0;
    body += text(20, vertical - 56, lane ? 'Alternating gradients cancel retained direction' : 'Persistent gradients reinforce retained direction', { width: 320, weight: 700 });
    for (const [step, gradient] of gradients.entries()) {
      const previous = velocity;
      velocity = picture.beta * previous + gradient;
      const horizontal = 45 + step * 83;
      body += arrowBetween([horizontal, baseline], [horizontal, baseline - velocity * 44], lane ? 'warning' : 'input', tools, 3)
        + text(horizontal, baseline + 66, number(velocity), { width: 78, anchor: 'middle', weight: 700, tone: lane ? 'warning' : 'input' });
      assert.equal(velocity, model.series[lane][1][step + 1][1]);
    }
  }
  return { body, height: 642 };
}

function missingCrossProducts(picture, model, tools) {
  const { text, color, fill, line } = tools;
  let body = text(20, 27, 'Four block slots require four computed products', { width: 320, weight: 700 });
  for (let row = 0; row < 2; row++) for (let column = 0; column < 2; column++) {
    const horizontal = 27 + column * 165;
    const vertical = 132 + row * 151;
    const diagonal = row === column;
    body += `<rect x="${horizontal}" y="${vertical}" width="145" height="105" style="fill:${fill(diagonal ? 'input' : 'neutral')};stroke:${color(diagonal ? 'input' : 'warning')};stroke-width:2${diagonal ? '' : ';stroke-dasharray:5 4'}"></rect>`
      + text(horizontal + 72, vertical + 47, `A${row}B${column}`, { width: 129, anchor: 'middle', weight: 700, tone: diagonal ? 'input' : 'warning' })
      + text(horizontal + 72, vertical + 82, diagonal ? 'computed' : 'missing', { width: 129, anchor: 'middle', size: 15 });
  }
  body += text(20, 487, 'Adding diagonal results does not fill the empty slots.', { width: 320, weight: 700 });
  return { body, height: 551 };
}

function subtractGradient(picture, model, tools) {
  const { text, line, circle, number } = tools;
  const position = value => 180 + value * 500;
  let body = text(20, 27, 'Descent subtracts the gradient from each logit', { width: 320, weight: 700 });
  for (const [index, gradient] of picture.gradients.entries()) {
    const vertical = 149 + index * 170;
    const next = -picture.rate * gradient;
    const tone = index ? 'warning' : 'output';
    body += text(20, vertical - 47, `Class ${String.fromCharCode(65 + index)}: gradient ${number(gradient)}`, { width: 320, weight: 700 })
      + line(35, vertical, 325, vertical) + circle(position(0), vertical, 5, 'neutral', true)
      + arrowBetween([position(0), vertical - 22], [position(next), vertical - 22], tone, tools, 3)
      + circle(position(next), vertical, 6, tone, true)
      + text(20, vertical + 73, `New logit ${number(next)}`, { width: 320, weight: 700, tone });
    assert(Math.abs(next - (index ? -.1 : .2)) < 1e-12);
  }
  return { body, height: 619 };
}

function twoHopRelay(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'A relay needs one layer per message hop', { width: 320, weight: 700 });
  for (let layer = 0; layer < 2; layer++) {
    const vertical = 177 + layer * 251;
    body += text(20, vertical - 77, layer ? 'Second layer: G passes B information to A' : 'First layer: B information reaches G, not A', { width: 320, weight: 700 });
    for (const [index, label] of ['A', 'G', 'B'].entries()) body += circle(59 + index * 123, vertical, 25, index === 2 || index === 1 || layer ? 'input' : 'neutral') + text(59 + index * 123, vertical + 6, label, { width: 44, anchor: 'middle', weight: 700 });
    body += layer ? arrowBetween([153, vertical], [89, vertical], 'input', tools, 4) : arrowBetween([275, vertical], [212, vertical], 'input', tools, 4);
    if (!layer) body += line(180, vertical + 36, 180, vertical + 107, 'warning', true);
  }
  body += text(20, 575, 'One-layer adjacency is not the two-layer receptive field.', { width: 320, tone: 'soft' });
  return { body, height: 638 };
}

function thresholdMembers(picture, model, tools) {
  const { text, circle, line } = tools;
  const horizontal = value => 35 + (value + 2) / 4 * 290;
  let body = text(20, 27, 'Raise the threshold across the same score identities', { width: 320, weight: 700 });
  for (const [stage, threshold] of picture.thresholds.entries()) {
    const vertical = 155 + stage * 255;
    body += text(20, vertical - 66, `Accept scores at least ${threshold}`, { width: 320, weight: 700 })
      + line(horizontal(threshold), vertical - 24, horizontal(threshold), vertical + 105, 'focus', true);
    for (const [group, scores] of [picture.genuine, picture.impostors].entries()) for (const [member, score] of scores.entries()) body += `<g data-picture-speaker-member="${group},${member}" data-picture-stage="${stage}" data-picture-accepted="${score >= threshold}">${circle(horizontal(score), vertical + group * 73, 7, group ? 'warning' : 'output', score >= threshold)}</g>`;
  }
  body += text(20, 590, 'Green: genuine. Red: impostor. Filled: accepted.', { width: 320, tone: 'soft' });
  return { body, height: 649 };
}

function nonzeroContributions(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'All examples remain; only nonzero weights contribute', { width: 320, weight: 700 });
  for (const [index, weight] of picture.weights.entries()) {
    const horizontal = 55 + index * 83;
    body += circle(horizontal, 140, 22, weight ? 'input' : 'neutral') + text(horizontal, 146, index + 1, { width: 40, anchor: 'middle', weight: 700 })
      + text(horizontal, 218, weight, { width: 70, anchor: 'middle', weight: 700, tone: weight ? 'input' : 'soft' });
    if (weight) body += arrowBetween([horizontal, 251], [180, 371], 'input', tools, 2 + weight * 5);
    else body += line(horizontal, 251, horizontal, 314, 'warning', true);
  }
  body += circle(180, 408, 25, 'output') + text(180, 414, 'sum', { width: 45, anchor: 'middle', weight: 700 })
    + text(20, 497, 'Examples two and four add zero, not missing data.', { width: 320, weight: 700 });
  return { body, height: 559 };
}

function fixedMergeOrder(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'A fixed merge order consumes the same characters', { width: 320, weight: 700 });
  for (const [stage, tokens] of picture.stages.entries()) {
    const vertical = 120 + stage * 153;
    assert.equal(tokens.join(''), 'lower');
    body += text(20, vertical - 38, ['Original word characters', 'Apply learned l + o merge', 'Apply learned lo + w merge', 'Apply learned e + r merge'][stage], { width: 320, weight: 700 });
    let position = 25;
    for (const token of tokens) {
      const width = token.length * 62;
      body += `<rect x="${position}" y="${vertical}" width="${width}" height="52" style="fill:${fill('input')};stroke:${color('input')};stroke-width:1.5"></rect>`
        + text(position + width / 2, vertical + 33, token, { width: width - 6, anchor: 'middle', weight: 700 });
      position += width;
    }
  }
  return { body, height: 650 };
}

function mixingAxes(picture, model, tools) {
  const { text, color, fill, line } = tools;
  let body = text(20, 27, 'Same tensor; two different mixing directions', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 136 + stage * 274;
    body += text(20, vertical - 50, stage ? 'Feed-forward: features inside each token row' : 'Attention: information across token positions', { width: 320, weight: 700 });
    for (let row = 0; row < 3; row++) for (let column = 0; column < 3; column++) body += `<rect x="${85 + column * 62}" y="${vertical + row * 47}" width="62" height="47" style="fill:${fill(stage ? 'output' : 'input')};stroke:${color(stage ? 'output' : 'input')};stroke-width:1.5"></rect>`;
    if (stage) for (let row = 0; row < 3; row++) body += arrowBetween([93, vertical + row * 47 + 24], [260, vertical + row * 47 + 24], 'output', tools, 3);
    else body += arrowBetween([177, vertical + 15], [177, vertical + 126], 'input', tools, 3);
  }
  return { body, height: 612 };
}

function localGap(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Keep local pairs; change the displayed global gap', { width: 320, weight: 700 });
  for (const [stage, positions] of picture.layouts.entries()) {
    const vertical = 167 + stage * 233;
    assert.equal(positions[1] - positions[0], 30);
    assert.equal(positions[3] - positions[2], 30);
    body += text(20, vertical - 53, `Layout ${stage + 1}: A-B and C-D remain local pairs`, { width: 320, weight: 700 })
      + line(positions[0], vertical, positions[1], vertical, 'input') + line(positions[2], vertical, positions[3], vertical, 'output');
    for (const [index, horizontal] of positions.entries()) body += circle(horizontal, vertical, 7, index < 2 ? 'input' : 'output', true) + text(horizontal, vertical + 43, String.fromCharCode(65 + index), { width: 31, anchor: 'middle', weight: 700 });
  }
  body += text(20, 533, 'The page gap is not certified original-space distance.', { width: 320, tone: 'soft' });
  return { body, height: 596 };
}

function piecewiseFit(picture, model, tools) {
  const { text, line, circle, color } = tools;
  const horizontal = value => 35 + value * 290;
  const truth = value => value ** 2;
  const approximation = value => { const left = Math.min(3, Math.floor(value * picture.segments)) / picture.segments; const right = left + 1 / picture.segments; return left ** 2 + (value - left) * (right + left); };
  let body = text(20, 27, 'Four line segments approximate the curved target', { width: 320, weight: 700 });
  for (const [index, operation] of [truth, approximation].entries()) body += `<path d="${Array.from({ length: 81 }, (_, point) => { const value = point / 80; return `${point ? 'L' : 'M'}${horizontal(value)} ${353 - operation(value) * 245}`; }).join('')}" style="fill:none;stroke:${color(index ? 'input' : 'output')};stroke-width:2"></path>`;
  for (let point = 0; point <= picture.segments; point++) body += circle(horizontal(point / picture.segments), 353 - truth(point / picture.segments) * 245, 4, 'input', true);
  body += text(20, 424, 'Magnified error: its peak stays below 0.02', { width: 320, weight: 700 });
  const baseline = 686;
  body += line(35, baseline - picture.tolerance * 6000, 325, baseline - picture.tolerance * 6000, 'warning', true)
    + `<path d="${Array.from({ length: 81 }, (_, point) => { const value = point / 80; const error = approximation(value) - truth(value); assert(error <= 1 / 64 + 1e-12); return `${point ? 'L' : 'M'}${horizontal(value)} ${baseline - error * 6000}`; }).join('')}" style="fill:none;stroke:${color('input')};stroke-width:2"></path>`
    + text(20, 739, 'Maximum error 1/64; this demonstrates existence.', { width: 320, weight: 700 });
  return { body, height: 800 };
}

function equalValuePolicies(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Equal values still need a declared selection rule', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 161 + stage * 233;
    body += text(20, vertical - 63, stage ? 'Greedy tie-break: choose A by declaration' : 'Equal-value softmax: equal probability', { width: 320, weight: 700 });
    for (let action = 0; action < 2; action++) body += circle(88 + action * 184, vertical, 29, action ? 'output' : 'input') + text(88 + action * 184, vertical + 6, action ? 'B' : 'A', { width: 49, anchor: 'middle', weight: 700 })
      + text(88 + action * 184, vertical + 82, stage ? action ? '0' : '1' : '0.5', { width: 100, anchor: 'middle', size: 23, weight: 700 });
  }
  body += text(20, 548, 'The same values do not specify a unique action.', { width: 320, tone: 'soft' });
  return { body, height: 609 };
}

function patchTokenInventory(picture, model, tools) {
  const { text, color, fill } = tools;
  const labels = ['P00', 'P01', 'P10', 'P11'];
  let body = text(20, 27, 'Four spatial patches become five sequence positions', { width: 320, weight: 700 });
  for (let patch = 0; patch < 4; patch++) body += `<rect x="${90 + patch % 2 * 91}" y="${108 + Math.floor(patch / 2) * 75}" width="91" height="75" style="fill:${fill('input')};stroke:${color('input')};stroke-width:2"></rect>`
    + text(135 + patch % 2 * 91, 152 + Math.floor(patch / 2) * 75, labels[patch], { width: 80, anchor: 'middle', weight: 700 });
  body += text(20, 330, 'Prepend CLS, then assign five positions', { width: 320, weight: 700 });
  for (const [position, label] of ['CLS', ...labels].entries()) body += `<rect x="${21 + position * 64}" y="417" width="60" height="57" style="fill:${fill(position ? 'input' : 'output')};stroke:${color(position ? 'input' : 'output')};stroke-width:2"></rect>`
    + text(51 + position * 64, 451, label, { width: 53, anchor: 'middle', size: 14, weight: 700 }) + text(51 + position * 64, 525, position, { width: 54, anchor: 'middle', weight: 700, tone: 'soft' });
  return { body, height: 578 };
}

function sharedMaskWeights(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Masks open paths through the same weight identities', { width: 320, weight: 700 });
  for (const [stage, mask] of picture.masks.entries()) {
    const vertical = 157 + stage * 258;
    body += text(20, vertical - 57, `Mask ${stage ? 'B' : 'A'}: weights remain w1, w2, w3`, { width: 320, weight: 700 });
    for (const [weight, active] of mask.entries()) {
      const horizontal = 65 + weight * 116;
      body += `<g data-picture-shared-weight="${weight}" data-picture-stage="${stage}">${circle(horizontal, vertical, 26, active ? 'input' : 'neutral')}${text(horizontal, vertical + 6, `w${weight + 1}`, { width: 46, anchor: 'middle', weight: 700 })}</g>`;
      if (active) body += arrowBetween([horizontal, vertical + 31], [horizontal, vertical + 115], 'input', tools, 3);
      else body += line(horizontal, vertical + 31, horizontal, vertical + 92, 'warning', true);
    }
  }
  body += text(20, 599, 'Changing masks does not create independent trained copies.', { width: 320, tone: 'soft' });
  return { body, height: 662 };
}

function translatedDisplacement(picture, model, tools) {
  const { text, circle, line } = tools;
  const position = point => [57 + point[0] * 75, 326 - point[1] * 91];
  let body = text(20, 27, 'Translate the displacement, then search vocabulary', { width: 320, weight: 700 });
  for (const [index, point] of picture.points.entries()) body += circle(...position(point), 6, index === 3 ? 'output' : 'input', true) + text(position(point)[0], position(point)[1] + (index === 3 ? -25 : 35), ['A', 'B', 'C', 'query'][index], { width: 86, anchor: 'middle', weight: 700 });
  body += arrowBetween(position(picture.points[0]), position(picture.points[1]), 'input', tools, 3) + arrowBetween(position(picture.points[2]), position(picture.points[3]), 'output', tools, 3)
    + text(20, 439, 'Same displacement (1,1); new endpoint (3,1)', { width: 320, weight: 700 })
    + text(20, 513, 'The endpoint is a coordinate, not yet a chosen word.', { width: 320, tone: 'soft' });
  assert.deepEqual(picture.points[1].map((value, index) => value - picture.points[0][index]), picture.points[3].map((value, index) => value - picture.points[2][index]));
  return { body, height: 575 };
}

function serialGates(picture, model, tools) {
  const { text, circle, line, color } = tools;
  let body = text(20, 27, 'Every required gate must pass on the same route', { width: 320, weight: 700 });
  for (const [lane, path] of picture.lanes.entries()) {
    const vertical = 164 + lane * 270;
    const spacing = 224 / (path.labels.length - 1);
    let alive = true;
    for (const [gate, accepted] of path.states.entries()) {
      const horizontal = 68 + gate * spacing;
      const tone = accepted ? 'output' : 'warning';
      if (gate) {
        if (alive) body += arrowBetween([horizontal - spacing + 27, vertical], [horizontal - 27, vertical], 'output', tools, 3);
        else body += line(horizontal - spacing + 27, vertical, horizontal - 27, vertical, 'neutral', true);
      }
      body += circle(horizontal, vertical, 22, tone) + text(horizontal, vertical + 6, accepted ? '+' : 'x', { width: 37, anchor: 'middle', size: 23, weight: 700, tone })
        + text(horizontal, vertical + 66, path.labels[gate], { width: Math.min(104, spacing - 8), anchor: 'middle', size: 15, weight: 700 });
      alive &&= accepted;
    }
    assert.equal(alive, path.states.every(Boolean));
    body += text(20, vertical + 145, path.detail, { width: 320, weight: 700, tone: alive ? 'output' : 'warning' });
  }
  return { body, height: 164 + picture.lanes.length * 270 };
}

function evidenceChain(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Known evidence stops where observation stops', { width: 320, weight: 700 });
  for (const [stage, state] of picture.states.entries()) {
    const vertical = 135 + stage * 133;
    const verified = state === 'verified';
    body += circle(58, vertical, 21, verified ? 'output' : 'neutral') + text(58, vertical + 6, verified ? '+' : '?', { width: 38, anchor: 'middle', size: 22, weight: 700 })
      + text(113, vertical + 6, picture.labels[stage], { width: 226, weight: 700, tone: verified ? 'output' : 'soft' });
    if (stage) body += line(58, vertical - 109, 58, vertical - 27, verified ? 'output' : 'warning', !verified);
  }
  body += text(20, 135 + picture.states.length * 133, 'Unknown is not a pass, a zero, or a completed proof.', { width: 320, tone: 'soft' });
  return { body, height: 212 + picture.states.length * 133 };
}

function observedLabels(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Only observed outcomes cross into labeled data', { width: 320, weight: 700 });
  for (const [row, label] of picture.labels.entries()) {
    const vertical = 141 + row * 140;
    const known = picture.values[row] !== '?';
    body += text(20, vertical + 6, label, { width: 133, weight: 700 })
      + (known ? arrowBetween([152, vertical], [266, vertical], 'output', tools, 3) : line(152, vertical, 232, vertical, 'warning', true))
      + circle(294, vertical, 24, known ? 'output' : 'neutral')
      + text(294, vertical + 6, picture.values[row], { width: 47, anchor: 'middle', size: known && picture.values[row].length > 1 ? 13 : 23, weight: 700 });
  }
  return { body, height: 208 + picture.labels.length * 140 };
}

function separateStateRails(picture, model, tools) {
  const { text, circle, color, fill, line } = tools;
  let body = text(20, 27, 'Each action changes only its own persistent state', { width: 320, weight: 700 });
  for (const [step, values] of picture.states.entries()) {
    const vertical = 131 + step * 132;
    body += text(20, vertical + 7, picture.actions[step], { width: 88, size: 15, weight: 700 });
    for (const [rail, value] of values.entries()) {
      const horizontal = 166 + rail * 116;
      const ready = picture.good ? picture.good[rail].includes(value) : rail ? value === 'zero' : value === 'train';
      body += circle(horizontal, vertical, 37, ready ? 'output' : 'warning') + text(horizontal, vertical + 6, value, { width: 80, anchor: 'middle', size: 14, weight: 700 });
      if (step) body += line(horizontal, vertical - 89, horizontal, vertical - 43, ready ? 'output' : 'warning', values[rail] === picture.states[step - 1][rail]);
    }
  }
  const footer = 131 + picture.states.length * 132;
  body += text(20, footer, picture.footer ?? 'Both rails must be ready: training mode and clean gradients.', { width: 320, weight: 700 });
  return { body, height: footer + 67 };
}

function brokenUpdate(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'A computed gradient is not yet an applied update', { width: 320, weight: 700 })
    + circle(70, 161, 27, 'input') + text(70, 167, picture.gradient, { width: 48, anchor: 'middle', weight: 700 })
    + text(20, 227, 'nonzero gradient', { width: 130, weight: 700, tone: 'input' })
    + line(105, 161, 154, 161, 'warning', true) + line(188, 161, 246, 161, 'warning', true)
    + line(155, 149, 181, 174, 'warning') + line(181, 149, 155, 174, 'warning')
    + circle(279, 161, 27, 'warning') + text(279, 167, picture.change, { width: 47, anchor: 'middle', weight: 700 })
    + text(211, 227, 'delta weights', { width: 129, weight: 700, tone: 'warning' })
    + text(20, 374, 'Tiny-batch loss stays flat at 1.10.', { width: 320, weight: 700 })
    + text(20, 455, 'Diagnose the step before full-data generalization.', { width: 320, tone: 'soft' });
  assert.equal(picture.change, 0);
  return { body, height: 518 };
}

function compareSwapOwner(picture, model, tools) {
  const { text, circle, line } = tools;
  let version = 0;
  let body = text(20, 27, 'The matching version grants one write owner', { width: 320, weight: 700 });
  for (let attempt = 0; attempt < 2; attempt++) {
    const vertical = 154 + attempt * 245;
    const expected = 0;
    const accepted = expected === version;
    body += text(20, vertical - 59, `Attempt ${attempt ? 'B' : 'A'}: expect zero, actual ${version}`, { width: 320, weight: 700 })
      + circle(67, vertical, 25, 'input') + text(67, vertical + 6, expected, { width: 45, anchor: 'middle', weight: 700 })
      + (accepted ? arrowBetween([97, vertical], [250, vertical], 'output', tools, 3) : line(97, vertical, 232, vertical, 'warning', true));
    if (accepted) version++;
    body += circle(283, vertical, 25, accepted ? 'output' : 'warning') + text(283, vertical + 6, version, { width: 45, anchor: 'middle', weight: 700 })
      + text(20, vertical + 100, accepted ? 'A commits v1 and owns the accepted result.' : 'B is rejected; A’s v1 stays untouched.', { width: 320, weight: 700, tone: accepted ? 'output' : 'warning' });
  }
  assert.equal(version, 1);
  return { body, height: 561 };
}

function idempotentReceipt(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Same key, one account effect, one retained receipt', { width: 320, weight: 700 });
  for (let attempt = 0; attempt < 2; attempt++) {
    const vertical = 163 + attempt * 248;
    body += text(20, vertical - 65, attempt ? 'Reconcile after losing the reply' : 'Authorized dispatch commits once', { width: 320, weight: 700 })
      + circle(68, vertical, 28, 'input') + text(68, vertical + 6, 'key K', { width: 50, anchor: 'middle', size: 14, weight: 700 })
      + arrowBetween([102, vertical], [248, vertical], attempt ? 'output' : 'input', tools, 3)
      + circle(284, vertical, 28, 'output') + text(284, vertical + 6, 'R1', { width: 50, anchor: 'middle', weight: 700 })
      + text(20, vertical + 96, attempt ? 'Return original receipt R1; no new payment.' : 'Account effect one; receipt R1 persists.', { width: 320, weight: 700, tone: 'output' });
  }
  return { body, height: 570 };
}

function retainedEvidence(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'A downstream stage cannot restore removed evidence', { width: 320, weight: 700 });
  for (const [stage, retained] of picture.stages.entries()) {
    const vertical = 142 + stage * 181;
    body += text(20, vertical - 47, picture.labels[stage], { width: 320, weight: 700 });
    for (const [passage, name] of ['A', 'B'].entries()) {
      const horizontal = 86 + passage * 184;
      const present = retained.includes(name);
      body += circle(horizontal, vertical, 27, present ? 'input' : 'neutral') + text(horizontal, vertical + 6, present ? name : '-', { width: 48, anchor: 'middle', weight: 700 });
      if (stage) body += line(horizontal, vertical - 151, horizontal, vertical - 68, present ? 'input' : 'warning', !present)
        + line(horizontal, vertical - 38, horizontal, vertical - 31, present ? 'input' : 'warning', !present);
    }
  }
  body += text(20, 633, 'Final context still lacks required passage B.', { width: 320, weight: 700, tone: 'warning' });
  return { body, height: 691 };
}

function persistentRefreshing(picture, model, tools) {
  const { text, color, fill, line } = tools;
  let body = text(20, 27, 'Page ownership persists while work slots refresh', { width: 320, weight: 700 })
    + text(20, 95, 'Twelve pages: nine reserved, three free', { width: 320, weight: 700 });
  for (const [page, owner] of picture.pages.entries()) {
    const horizontal = 24 + page % 6 * 52;
    const vertical = 148 + Math.floor(page / 6) * 60;
    const tone = owner === '-' ? 'neutral' : owner === 'A' ? 'input' : 'output';
    body += `<rect x="${horizontal}" y="${vertical}" width="48" height="54" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>` + text(horizontal + 24, vertical + 33, owner, { width: 42, anchor: 'middle', weight: 700 });
  }
  for (let iteration = 0; iteration < 2; iteration++) {
    const vertical = 378 + iteration * 154;
    body += text(20, vertical - 35, `Iteration ${iteration + 1}: eight work slots`, { width: 320, weight: 700 });
    for (const [slot, task] of picture.work.entries()) body += `<rect x="${24 + slot * 39}" y="${vertical}" width="36" height="46" style="fill:${fill('state')};stroke:${color('state')};stroke-width:1.5"></rect>` + text(42 + slot * 39, vertical + 29, task, { width: 29, anchor: 'middle', weight: 700 });
  }
  body += text(20, 668, 'Refreshing work does not free A or B’s reserved pages.', { width: 320, tone: 'soft' });
  assert.equal(picture.pages.filter(owner => owner !== '-').length, 9);
  return { body, height: 731 };
}

function nodeReassignment(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Reassign the same eight instantaneous nodes', { width: 320, weight: 700 });
  for (const [stage, owners] of [picture.before, picture.after].entries()) {
    const vertical = 143 + stage * 261;
    body += text(20, vertical - 51, stage ? 'Serving pressure: checkpoint, pause, reclaim' : 'Six serving nodes and two offline nodes', { width: 320, weight: 700 });
    for (const [node, owner] of owners.entries()) {
      const horizontal = 56 + node % 4 * 82;
      const row = vertical + Math.floor(node / 4) * 83;
      body += `<g data-picture-node-member="${node}" data-picture-stage="${stage}">${circle(horizontal, row, 26, owner === 'S' ? 'input' : 'warning')}${text(horizontal, row + 6, owner, { width: 44, anchor: 'middle', weight: 700 })}</g>`;
    }
  }
  assert.equal(picture.before.length, picture.after.length);
  body += text(20, 600, 'No ninth node was created; two owners changed.', { width: 320, tone: 'soft' });
  return { body, height: 662 };
}

function assignmentMembership(picture, model, tools) {
  const { text, circle, color, line } = tools;
  let body = text(20, 27, 'Delivery changes; the original membership remains', { width: 320, weight: 700 })
    + text(20, 91, `${picture.total} original members; ${picture.served} served or shown`, { width: 320, weight: 700 });
  assert.equal(picture.total % picture.unit, 0);
  assert.equal(picture.served % picture.unit, 0);
  const members = picture.total / picture.unit;
  for (let member = 0; member < members; member++) {
    const selected = member < picture.served / picture.unit;
    const horizontal = 29 + member % 20 * 15;
    const vertical = 174 + Math.floor(member / 20) * 29;
    body += `<g data-picture-original-member="${member}" data-picture-selected="${selected}">${circle(horizontal, vertical, 4, selected ? 'output' : 'neutral', selected)}</g>`;
  }
  body += text(20, 399, `${picture.total - picture.served} remain in the original cohort.`, { width: 320, weight: 700 })
    + text(20, 477, `One mark represents ${picture.unit}. Missing exposure is not a negative label.`, { width: 320, tone: 'soft' });
  return { body, height: 552 };
}

function conversationAssignment(picture, model, tools) {
  const { text, circle, line, color } = tools;
  let body = text(20, 27, 'One conversation encloses one measured outcome', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 157 + stage * 246;
    body += text(20, vertical - 59, stage ? 'Turn switching: A, B, A contaminate one outcome' : 'Conversation assignment: A, A, A remain together', { width: 320, weight: 700 });
    for (let turn = 0; turn < 3; turn++) body += circle(63 + turn * 115, vertical, 25, stage && turn === 1 ? 'warning' : 'input') + text(63 + turn * 115, vertical + 6, stage && turn === 1 ? 'B' : 'A', { width: 45, anchor: 'middle', weight: 700 });
    body += line(38, vertical + 41, 38, vertical + 58) + line(38, vertical + 58, 318, vertical + 58) + line(318, vertical + 58, 318, vertical + 41)
      + text(180, vertical + 113, 'one outcome', { width: 230, anchor: 'middle', weight: 700, tone: stage ? 'warning' : 'output' });
  }
  return { body, height: 576 };
}

function hierarchicalSum(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'Combine local partials once, then distribute the total', { width: 320, weight: 700 });
  const local = [picture.inputs[0] + picture.inputs[1], picture.inputs[2] + picture.inputs[3]];
  const total = local[0] + local[1];
  for (const [rank, value] of picture.inputs.entries()) {
    const horizontal = 53 + rank * 85;
    const parent = 95 + Math.floor(rank / 2) * 170;
    body += circle(horizontal, 136, 21, 'input') + text(horizontal, 142, value, { width: 36, anchor: 'middle', weight: 700 }) + arrowBetween([horizontal, 163], [parent, 248], 'input', tools);
  }
  for (const [group, value] of local.entries()) body += circle(95 + group * 170, 277, 26, 'state') + text(95 + group * 170, 283, value, { width: 45, anchor: 'middle', weight: 700 }) + arrowBetween([95 + group * 170, 309], [180, 391], 'state', tools);
  body += circle(180, 421, 27, 'output') + text(180, 427, total, { width: 46, anchor: 'middle', weight: 700 });
  for (let rank = 0; rank < 4; rank++) body += arrowBetween([180, 453], [53 + rank * 85, 539], 'output', tools) + circle(53 + rank * 85, 569, 22, 'output') + text(53 + rank * 85, 575, total, { width: 39, anchor: 'middle', weight: 700 });
  assert.equal(total, 10);
  return { body, height: 628 };
}

function ringReduction(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Reduce the partial; circulate the completed chunk', { width: 320, weight: 700 });
  let partial = 0;
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 165 + stage * 242;
    body += text(20, vertical - 57, stage ? 'All-gather copies the completed ten' : 'Reduction adds each local contribution once', { width: 320, weight: 700 });
    for (const [rank, value] of picture.inputs.entries()) {
      if (!stage) partial += value;
      const horizontal = 53 + rank * 85;
      if (rank) body += arrowBetween([horizontal - 62, vertical], [horizontal - 26, vertical], stage ? 'output' : 'input', tools);
      body += circle(horizontal, vertical, 22, stage ? 'output' : 'input') + text(horizontal, vertical + 6, partial, { width: 39, anchor: 'middle', weight: 700 });
      if (!stage) body += text(horizontal, vertical + 61, `+${value}`, { width: 56, anchor: 'middle', tone: 'soft' });
    }
  }
  assert.equal(partial, 10);
  return { body, height: 498 };
}

function strictBoundary(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Equality belongs to the rejected region', { width: 320, weight: 700 }) + line(45, 203, 315, 203);
  for (const [index, value] of picture.positions.entries()) {
    const horizontal = 61 + index * 120;
    const accepted = picture.accepted[index];
    assert.equal(accepted, value > 0);
    body += circle(horizontal, 203, 7, accepted ? 'output' : 'warning', true)
      + text(horizontal, 163, value, { width: 71, anchor: 'middle', weight: 700 })
      + text(horizontal, 277, picture.labels[index], { width: 97, anchor: 'middle', size: 16, weight: 700 })
      + text(horizontal, 348, accepted ? 'allow' : 'reject', { width: 96, anchor: 'middle', weight: 700, tone: accepted ? 'output' : 'warning' });
  }
  body += line(181, 182, 181, 234, 'focus') + text(20, 462, 'Expiry minus now must be strictly positive.', { width: 320, tone: 'soft' });
  return { body, height: 523 };
}

function suppressionRoute(picture, model, tools) {
  const { text, color, fill, line } = tools;
  let body = text(20, 27, 'A suppresses B; removed B cannot suppress C', { width: 320, weight: 700 });
  for (const [index, box] of model.boxes.entries()) body += `<rect x="${box.left}" y="121" width="${box.width}" height="140" style="fill:${fill(index === 1 ? 'warning' : 'input')};fill-opacity:.22;stroke:${color(index === 1 ? 'warning' : 'input')};stroke-width:2${index === 1 ? ';stroke-dasharray:5 4' : ''}"></rect>` + text(box.left + box.width / 2, 199, box.name, { width: 66, anchor: 'middle', weight: 700 });
  for (let index = 0; index < model.boxes.length - 1; index++) {
    const left = model.boxes[index];
    const right = model.boxes[index + 1];
    const overlap = Math.max(0, Math.min(left.left + left.width, right.left + right.width) - Math.max(left.left, right.left));
    assert(Math.abs(overlap / (left.width + right.width - overlap) - 1 / 3) < 1e-12);
  }
  body += arrowBetween([74, 378], [175, 378], 'warning', tools, 3) + text(68, 423, 'A', { width: 60, anchor: 'middle', weight: 700 }) + text(183, 423, 'B out', { width: 80, anchor: 'middle', weight: 700, tone: 'warning' })
    + line(216, 378, 279, 378, 'warning', true) + text(303, 423, 'C', { width: 55, anchor: 'middle', weight: 700, tone: 'output' })
    + text(20, 511, 'Adjacent IoU 1/3 > 0.30. A and C survive.', { width: 320, weight: 700 });
  return { body, height: 574 };
}

function clockAfterBoundary(picture, model, tools) {
  const { text, circle, line } = tools;
  const first = Math.min(...picture.events.map(event => event.value));
  const last = Math.max(...picture.events.map(event => event.value));
  const horizontal = value => 60 + (value - first) / (last - first) * 240;
  let body = text(20, 27, 'Later evidence cannot cross the earlier readout boundary', { width: 320, weight: 700 });
  for (const [index, event] of picture.events.entries()) {
    const vertical = 143 + index * 120;
    const later = event.value > picture.boundary;
    body += line(45, vertical, 315, vertical) + line(horizontal(picture.boundary), vertical - 27, horizontal(picture.boundary), vertical + 27, 'focus', true)
      + circle(horizontal(event.value), vertical, 7, later ? 'warning' : 'input', true)
      + text(20, vertical + 61, event.label, { width: 320, weight: 700, tone: later ? 'warning' : 'input' });
  }
  return { body, height: 216 + picture.events.length * 120 };
}

function unscannedPartition(picture, model, tools) {
  const { text, circle, line, color, fill } = tools;
  let body = text(20, 27, 'A scan route reaches members, not excluded partitions', { width: 320, weight: 700 });
  for (let partition = 0; partition < 3; partition++) {
    const horizontal = 24 + partition * 112;
    body += `<rect x="${horizontal}" y="147" width="91" height="151" style="fill:${fill(partition === 2 ? 'neutral' : 'input')};stroke:${color(partition === 2 ? 'warning' : 'input')};stroke-width:2"></rect>`
      + text(horizontal + 45, 184, String.fromCharCode(65 + partition), { width: 76, anchor: 'middle', weight: 700 })
      + circle(horizontal + 45, 235, 9, partition === 2 ? 'output' : 'input', true);
    body += partition === 2 ? line(horizontal + 45, 320, horizontal + 45, 378, 'warning', true) : arrowBetween([horizontal + 45, 376], [horizontal + 45, 308], 'input', tools, 3);
  }
  body += text(20, 435, 'C contains the true neighbor but is not scanned.', { width: 320, weight: 700, tone: 'warning' }) + text(20, 513, 'PQ changes representation, not the missing scan route.', { width: 320, tone: 'soft' });
  return { body, height: 578 };
}

function roleBudget(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Each fictional role spends the same hundred units', { width: 320, weight: 700 });
  for (const [role, values] of picture.values.entries()) {
    const vertical = 144 + role * 169;
    assert.equal(values.reduce((sum, value) => sum + value, 0), 100);
    body += text(20, vertical - 38, picture.labels[role], { width: 320, weight: 700 });
    let offset = 25;
    for (const [activity, value] of values.entries()) {
      const tone = ['input', 'output', 'warning'][activity];
      body += `<rect x="${offset}" y="${vertical}" width="${value * 3.1}" height="38" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>`;
      offset += value * 3.1;
    }
    body += text(20, vertical + 81, values.join(' / ') + ' research / build / coordinate', { width: 320, size: 15, tone: 'soft' });
  }
  return { body, height: 605 };
}

function committedProvisional(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  let body = text(20, 27, 'Append committed positions or revise provisional ones', { width: 320, weight: 700 });
  for (let lane = 0; lane < 2; lane++) {
    const vertical = 132 + lane * 349;
    body += text(20, vertical - 70, lane ? 'Diffusion: every intermediate position remains provisional' : 'Autoregression: committed prefix grows', { width: 320, weight: 700 });
    for (let step = 0; step < 3; step++) for (let position = 0; position < 3; position++) {
      const present = lane || position <= step;
      const committed = present && (!lane || step === 2);
      const horizontal = 77 + position * 95;
      const row = vertical + step * 79;
      body += circle(horizontal, row, 25, committed ? 'output' : 'neutral') + text(horizontal, row + 6, present ? committed ? String.fromCharCode(65 + position) : '?' : '-', { width: 43, anchor: 'middle', weight: 700 });
    }
  }
  return { body, height: 711 };
}

function versionedReply(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Keep the request version attached through the reply', { width: 320, weight: 700 });
  for (const [step, label] of ['request v7', 'edit to v8', 'reply still v7', 'revalidate v8'].entries()) {
    const vertical = 134 + step * 126;
    body += circle(60, vertical, 25, step === 2 ? 'warning' : step === 3 ? 'output' : 'input') + text(60, vertical + 6, step === 1 || step === 3 ? '8' : '7', { width: 43, anchor: 'middle', weight: 700 }) + text(118, vertical + 6, label, { width: 219, weight: 700 });
    if (step) body += line(60, vertical - 98, 60, vertical - 29, step === 2 ? 'warning' : 'input', step === 2);
  }
  body += text(20, 641, 'Version mismatch blocks applying the old reply.', { width: 320, tone: 'soft' });
  return { body, height: 704 };
}

function memoryRoundtrip(picture, model, tools) {
  const { text, circle, line, color } = tools;
  let body = text(20, 27, 'Keep the fitting intermediate inside the SRAM boundary', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 172 + stage * 276;
    body += text(20, vertical - 71, stage ? 'Fused: on-chip intermediate reuse' : 'Unfused: intermediate write then reread', { width: 320, weight: 700 })
      + line(201, vertical - 44, 201, vertical + 117, 'focus', true)
      + circle(78, vertical, 26, 'input') + text(78, vertical + 6, 'SRAM', { width: 48, anchor: 'middle', size: 13, weight: 700 });
    if (!stage) body += circle(284, vertical, 26, 'warning') + text(284, vertical + 6, 'HBM', { width: 48, anchor: 'middle', size: 14, weight: 700 }) + arrowBetween([112, vertical - 17], [252, vertical - 17], 'warning', tools, 3) + arrowBetween([252, vertical + 36], [112, vertical + 36], 'warning', tools, 3);
    else body += arrowBetween([75, vertical + 34], [75, vertical + 91], 'output', tools, 3) + text(20, vertical + 145, 'Input reads and output writes still remain.', { width: 320, weight: 700 });
  }
  return { body, height: 656 };
}

function complementBeforeStore(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Query the earlier complement before storing this index', { width: 320, weight: 700 })
    + text(20, 113, 'Target six; current four at index two', { width: 320, weight: 700 })
    + circle(65, 210, 26, 'input') + text(65, 216, '4', { width: 45, anchor: 'middle', weight: 700 })
    + arrowBetween([100, 210], [247, 210], 'input', tools, 3)
    + circle(286, 210, 26, 'state') + text(286, 216, '2', { width: 45, anchor: 'middle', weight: 700 })
    + text(20, 322, 'Saved two belongs to earlier index one.', { width: 320, weight: 700 })
    + text(20, 420, 'Return distinct original indices (1,2).', { width: 320, weight: 700, tone: 'output' })
    + text(20, 508, 'The current position cannot serve as its own earlier match.', { width: 320, tone: 'soft' });
  return { body, height: 572 };
}

function policyRegion(picture, model, tools) {
  const { text, line, circle, color, fill } = tools;
  const horizontal = value => 25 + value * 310;
  let body = text(20, 27, 'Policy P7 maps a score coordinate to an action region', { width: 320, weight: 700 });
  const ends = [0, ...picture.boundaries, 1];
  for (const [region, action] of picture.actions.entries()) {
    const tone = ['output', 'state', 'warning'][region];
    body += `<rect x="${horizontal(ends[region])}" y="160" width="${(ends[region + 1] - ends[region]) * 310}" height="50" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>`
      + text(horizontal((ends[region] + ends[region + 1]) / 2), 271, action, { width: Math.max(54, (ends[region + 1] - ends[region]) * 310 - 10), anchor: 'middle', size: 15, weight: 700 });
  }
  body += circle(horizontal(picture.score), 185, 7, 'focus', true) + text(horizontal(picture.score), 134, picture.score, { width: 86, anchor: 'middle', weight: 700 }) + text(20, 385, '0.62 lies inside the declared 0.4–0.8 review region.', { width: 320, weight: 700 });
  return { body, height: 456 };
}

function matchedComparison(picture, model, tools) {
  const { text, circle, line, number } = tools;
  let body = text(20, 27, 'Compare methods only across the matched boundary', { width: 320, weight: 700 });
  const known = picture.values.filter(value => value !== null);
  const minimum = Math.min(...known);
  const maximum = Math.max(...known);
  const horizontal = value => 45 + (value - minimum) / (maximum - minimum) * 270;
  for (const [row, value] of picture.values.entries()) {
    const vertical = 144 + row * 146;
    body += text(20, vertical - 42, picture.labels[row], { width: 320, weight: 700 });
    if (value === null) body += line(35, vertical, 292, vertical, 'warning', true) + circle(308, vertical, 20, 'neutral') + text(308, vertical + 6, '?', { width: 34, anchor: 'middle', weight: 700 });
    else body += line(45, vertical, 315, vertical, 'neutral') + circle(horizontal(value), vertical, 6, row === 2 ? 'output' : 'input', true) + text(20, vertical + 54, `${number(value)} ${picture.suffix}; shared ruler ${number(minimum)} to ${number(maximum)}`, { width: 320, weight: 700, tone: row === 2 ? 'output' : 'soft' });
  }
  body += text(20, 651, picture.values.includes(null) ? 'The matched baseline result is still missing.' : 'Matched method gap: 83 - 82.8 = 0.2 points.', { width: 320, weight: 700, tone: picture.values.includes(null) ? 'warning' : 'output' });
  return { body, height: 713 };
}

function groupCut(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'The held-out boundary must respect user identity', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 163 + stage * 242;
    body += text(20, vertical - 65, stage ? 'Group split: all three A rows stay together' : 'Random rows: A crosses train and test', { width: 320, weight: 700 });
    for (let row = 0; row < 3; row++) body += circle(64 + row * 116, vertical, 25, !stage && row === 1 ? 'warning' : 'input') + text(64 + row * 116, vertical + 6, 'A', { width: 44, anchor: 'middle', weight: 700 }) + text(64 + row * 116, vertical + 82, !stage && row === 1 ? 'test' : 'train', { width: 94, anchor: 'middle', size: 17, weight: 700 });
  }
  return { body, height: 553 };
}

function coreBorderSource(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Joining a cluster does not grant expansion authority', { width: 320, weight: 700 });
  const centers = [[75, 213], [210, 213], [298, 420]];
  body += circle(...centers[0], 66, 'input') + circle(...centers[1], 30, 'state') + circle(...centers[2], 22, 'warning');
  for (const [index, label] of ['D: five', 'E: two', 'F: one'].entries()) body += text(centers[index][0], centers[index][1] + 7, label, { width: index ? 55 : 115, anchor: 'middle', size: 13, weight: 700 });
  body += arrowBetween([147, 213], [176, 213], 'input', tools, 3) + line(224, 251, 274, 374, 'warning', true)
    + text(20, 512, 'minPts four includes self. Only D is an expansion source.', { width: 320, weight: 700 });
  return { body, height: 578 };
}

function firstFailure(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Pin the first rank-time failure before the later symptoms', { width: 320, weight: 700 });
  for (let rank = 0; rank < 3; rank++) for (let step = 0; step < 3; step++) {
    const state = model.values[rank][step];
    const tone = state === 'ok' ? 'input' : 'warning';
    const horizontal = 50 + step * 91;
    const vertical = 149 + rank * 90;
    body += `<rect x="${horizontal}" y="${vertical}" width="87" height="73" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:${state === 'FIRST' ? 4 : 1.5}"></rect>` + text(horizontal + 43, vertical + 44, state, { width: 76, anchor: 'middle', size: 16, weight: 700 });
  }
  body += text(20, 483, 'Rows rank 0–2; columns step 1–3. First: rank one, step two.', { width: 320, weight: 700 });
  return { body, height: 558 };
}

function alternativeWhole(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Two feasible portfolios compete for the same capacity', { width: 320, weight: 700 });
  for (const [choice, value] of picture.values.entries()) {
    const vertical = 161 + choice * 206;
    body += text(20, vertical - 45, picture.labels[choice], { width: 320, weight: 700 }) + `<rect x="25" y="${vertical}" width="${value * 27}" height="42" style="fill:${fill(choice ? 'output' : 'input')};stroke:${color(choice ? 'output' : 'input')};stroke-width:2"></rect>` + text(20, vertical + 101, `${value} assumed forward-value units`, { width: 320, weight: 700 });
  }
  return { body, height: 537 };
}

function conditionalSpatialCut(picture, model, tools) {
  const { text, color, fill, line } = tools;
  let body = text(20, 27, 'The second split exists only inside the right root branch', { width: 320, weight: 700 });
  body += `<rect x="45" y="125" width="130" height="272" style="fill:${fill('input')};stroke:${color('input')};stroke-width:2"></rect>`
    + `<rect x="175" y="125" width="140" height="136" style="fill:${fill('state')};stroke:${color('state')};stroke-width:2"></rect>`
    + `<rect x="175" y="261" width="140" height="136" style="fill:${fill('output')};stroke:${color('output')};stroke-width:2"></rect>`
    + text(110, 267, 'A', { width: 111, anchor: 'middle', weight: 700 }) + text(245, 199, 'C', { width: 115, anchor: 'middle', weight: 700 }) + text(245, 335, 'B', { width: 115, anchor: 'middle', weight: 700 })
    + text(20, 464, 'Root cut x=0. Right branch alone splits at y=1.', { width: 320, weight: 700 }) + text(20, 542, 'Every left-region point still belongs to A.', { width: 320, tone: 'soft' });
  return { body, height: 604 };
}

function authorityRails(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Context access and tool authority are independent routes', { width: 320, weight: 700 });
  for (const [mode, label] of picture.labels.entries()) {
    const vertical = 143 + mode * 138;
    body += text(20, vertical + 7, label, { width: 99, weight: 700 }) + circle(161, vertical, 23, 'input') + text(161, vertical + 6, 'repo', { width: 40, anchor: 'middle', size: 14, weight: 700 });
    const allowed = picture.allowed[mode];
    body += allowed ? arrowBetween([190, vertical], [270, vertical], 'output', tools, 3) : line(190, vertical, 253, vertical, 'warning', true);
    body += circle(300, vertical, 22, allowed ? 'output' : 'neutral') + text(300, vertical + 6, allowed ? '+' : 'x', { width: 37, anchor: 'middle', weight: 700 });
  }
  return { body, height: 209 + picture.labels.length * 138 };
}

function availabilityBefore(picture, model, tools) {
  const { text, circle, line } = tools;
  const horizontal = hour => 40 + (hour - 11) / 1.25 * 280;
  let body = text(20, 27, 'Only available versions cross into the noon lookup', { width: 320, weight: 700 });
  const events = [[11 + 5 / 60, 'A available 11:05', true], [11 + 50 / 60, 'B event 11:50; not yet available', false], [12 + 10 / 60, 'B available 12:10; after the lookup', false]];
  for (const [row, [time, label, available]] of events.entries()) {
    const vertical = 145 + row * 149;
    body += line(40, vertical, 320, vertical) + line(horizontal(12), vertical - 25, horizontal(12), vertical + 28, 'focus', true) + circle(horizontal(time), vertical, 7, available ? 'output' : 'warning', true) + text(20, vertical + 68, label, { width: 320, weight: 700 });
  }
  body += text(20, 636, 'Noon returns A. Event time alone cannot admit B.', { width: 320, weight: 700, tone: 'output' });
  return { body, height: 704 };
}

function sharedSupplyReturn(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'One shared supply response returns to both cells', { width: 320, weight: 700 });
  for (let cell = 0; cell < 2; cell++) body += circle(90 + cell * 180, 147, 27, cell ? 'state' : 'input') + text(90 + cell * 180, 153, cell ? 'C' : 'T', { width: 46, anchor: 'middle', weight: 700 }) + arrowBetween([90 + cell * 180, 180], [180, 280], cell ? 'state' : 'input', tools, 3);
  body += circle(180, 323, 43, 'focus') + text(180, 329, 'creators', { width: 76, anchor: 'middle', size: 14, weight: 700 });
  for (let cell = 0; cell < 2; cell++) body += arrowBetween([180, 372], [90 + cell * 180, 473], cell ? 'warning' : 'input', tools, 3) + circle(90 + cell * 180, 509, 29, cell ? 'warning' : 'input') + text(90 + cell * 180, 515, cell ? 'C' : 'T', { width: 49, anchor: 'middle', weight: 700 });
  body += text(20, 603, 'Changed creator supply also reaches future control C.', { width: 320, weight: 700, tone: 'warning' });
  return { body, height: 667 };
}

function duplicatePenalty(picture, model, tools) {
  const { text, circle, color, fill, line } = tools;
  let body = text(20, 27, 'Repeated song identity reduces whole-page utility', { width: 320, weight: 700 });
  for (let choice = 0; choice < 2; choice++) {
    const vertical = 155 + choice * 249;
    body += text(20, vertical - 56, choice ? 'A six + C four; no duplicate penalty' : 'A six + B five; same song repeats', { width: 320, weight: 700 });
    for (let shelf = 0; shelf < 2; shelf++) body += circle(89 + shelf * 180, vertical, 26, choice || !shelf ? 'input' : 'warning') + text(89 + shelf * 180, vertical + 6, choice && shelf ? 'new' : 'song X', { width: 48, anchor: 'middle', size: 13, weight: 700 });
    body += text(20, vertical + 109, choice ? '6 + 4 = 10' : '6 + 5 - 4 = 7', { width: 320, weight: 700, tone: choice ? 'output' : 'warning' });
  }
  return { body, height: 581 };
}

function fixedSourceGrowingPrefix(picture, model, tools) {
  const { text, circle, color, fill, line } = tools;
  let body = text(20, 27, 'Keep the source memory; grow the decoder prefix', { width: 320, weight: 700 });
  for (let step = 0; step < 3; step++) {
    const vertical = 144 + step * 175;
    body += circle(64, vertical, 27, 'input') + text(64, vertical + 6, 'KV', { width: 48, anchor: 'middle', weight: 700 });
    for (let token = 0; token <= step; token++) body += `<rect x="${130 + token * 62}" y="${vertical - 24}" width="56" height="48" style="fill:${fill('output')};stroke:${color('output')};stroke-width:2"></rect>` + text(158 + token * 62, vertical + 6, token ? String.fromCharCode(64 + token) : 'BOS', { width: 48, anchor: 'middle', size: 13, weight: 700 });
    if (step) body += line(64, vertical - 144, 64, vertical - 31, 'input');
  }
  body += text(20, 633, 'Same source identity; prefixes BOS, BOS A, BOS A B.', { width: 320, weight: 700 });
  return { body, height: 698 };
}

function weightedTail(picture, model, tools) {
  const { text, color, fill, line, number } = tools;
  let body = text(20, 27, 'Keep traffic weights and the independent tail floor', { width: 320, weight: 700 });
  for (const [candidate, values] of picture.values.entries()) {
    const vertical = 165 + candidate * 251;
    const average = values.reduce((sum, value, index) => sum + value * picture.weights[index], 0);
    body += text(20, vertical - 55, `Candidate ${candidate ? 'B' : 'A'}: weighted ${number(average)}`, { width: 320, weight: 700 });
    let horizontal = 25;
    for (const [slice, value] of values.entries()) {
      const width = picture.weights[slice] * 310;
      const tone = slice && value < picture.floor ? 'warning' : 'input';
      body += `<rect x="${horizontal}" y="${vertical + (1 - value) * 120}" width="${width}" height="${value * 120}" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>`;
      horizontal += width;
    }
    body += line(25, vertical + (1 - picture.floor) * 120, 335, vertical + (1 - picture.floor) * 120, 'warning', true)
      + text(20, vertical + 167, `Head ${values[0]}; tail ${values[1]}; tail floor ${picture.floor}`, { width: 320, weight: 700 });
  }
  return { body, height: 649 };
}

function restoreCommon(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Restore both ranks from the same committed checkpoint', { width: 320, weight: 700 });
  for (let rank = 0; rank < 2; rank++) body += circle(90 + rank * 180, 147, 27, 'warning') + text(90 + rank * 180, 153, rank ? 'fail' : 'v5', { width: 48, anchor: 'middle', size: 16, weight: 700 }) + arrowBetween([90 + rank * 180, 179], [180, 287], 'warning', tools, 3);
  body += circle(180, 327, 34, 'input') + text(180, 333, 'v4', { width: 60, anchor: 'middle', weight: 700 });
  for (let rank = 0; rank < 2; rank++) body += arrowBetween([180, 369], [90 + rank * 180, 472], 'output', tools, 3) + circle(90 + rank * 180, 510, 29, 'output') + text(90 + rank * 180, 516, 'v4', { width: 50, anchor: 'middle', weight: 700 });
  body += text(20, 603, 'Both restored ranks retry from v4, not mixed local progress.', { width: 320, weight: 700 });
  return { body, height: 669 };
}

function independentChoices(picture, model, tools) {
  const { text, color, fill, line } = tools;
  let body = text(20, 27, 'Choose supervision and update scope independently', { width: 320, weight: 700 });
  for (let signal = 0; signal < 2; signal++) for (let scope = 0; scope < 2; scope++) {
    const horizontal = 26 + scope * 166;
    const vertical = 139 + signal * 156;
    body += `<rect x="${horizontal}" y="${vertical}" width="141" height="111" style="fill:${fill(scope ? 'output' : 'input')};stroke:${color(scope ? 'output' : 'input')};stroke-width:2"></rect>` + text(horizontal + 70, vertical + 44, signal ? 'pairs' : 'labels', { width: 120, anchor: 'middle', weight: 700 }) + text(horizontal + 70, vertical + 85, scope ? 'LoRA' : 'full', { width: 120, anchor: 'middle', weight: 700 });
  }
  body += text(20, 492, 'Each supervision choice has both scope choices.', { width: 320, tone: 'soft' });
  return { body, height: 554 };
}

function independentEvidence(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, picture.title ?? 'Each claim needs its own observed evidence', { width: 320, weight: 700 });
  for (const [claim, label] of picture.labels.entries()) {
    const vertical = 129 + claim * 123;
    const verified = picture.states[claim] === 'verified';
    body += text(20, vertical + 7, label, { width: 230, weight: 700 }) + circle(302, vertical, 22, verified ? 'output' : 'neutral') + text(302, vertical + 6, verified ? '+' : '?', { width: 38, anchor: 'middle', weight: 700, size: 23 });
  }
  return { body, height: 193 + picture.labels.length * 123 };
}

function stableTileMerge(picture, model, tools) {
  const { text, circle, number } = tools;
  const tiles = picture.logits.map((logits, tile) => { const maximum = Math.max(...logits); return { maximum, denominator: logits.reduce((sum, value) => sum + Math.exp(value - maximum), 0), numerator: logits.reduce((sum, value, index) => sum + Math.exp(value - maximum) * picture.values[tile][index], 0) }; });
  const maximum = Math.max(...tiles.map(tile => tile.maximum));
  const denominator = tiles.reduce((sum, tile) => sum + tile.denominator * Math.exp(tile.maximum - maximum), 0);
  const numerator = tiles.reduce((sum, tile) => sum + tile.numerator * Math.exp(tile.maximum - maximum), 0);
  const completeDenominator = picture.logits.flat().reduce((sum, logit) => sum + Math.exp(logit), 0);
  const completeNumerator = picture.logits.flat().reduce((sum, logit, index) => sum + Math.exp(logit) * picture.values.flat()[index], 0);
  assert(Math.abs(numerator / denominator - completeNumerator / completeDenominator) < 1e-12);
  let body = text(20, 27, 'Merge retained statistics, then discard temporary scores', { width: 320, weight: 700 });
  for (const [tile, state] of tiles.entries()) {
    const vertical = 149 + tile * 164;
    body += text(20, vertical - 42, `Tile ${tile + 1}: retain M, L, O`, { width: 320, weight: 700 });
    for (const [component, value] of [state.maximum, state.denominator, state.numerator].entries()) body += circle(61 + component * 119, vertical, 28, 'input') + text(61 + component * 119, vertical + 6, number(value), { width: 50, anchor: 'middle', size: 13, weight: 700 });
  }
  body += text(20, 454, `Merge at M=${maximum}: L=${number(denominator)}, O=${number(numerator)}`, { width: 320, weight: 700 }) + text(20, 554, `Output O/L = ${number(numerator / denominator)}`, { width: 320, weight: 700, tone: 'output' });
  return { body, height: 619 };
}

function transientGather(picture, model, tools) {
  const { text, color, fill, line } = tools;
  let body = text(20, 27, 'One rank’s shard grows temporarily into one active unit', { width: 320, weight: 700 });
  for (let stage = 0; stage < 3; stage++) {
    const vertical = 137 + stage * 169;
    body += text(20, vertical - 37, ['Rest: retain one quarter', 'Compute: gather this wrapped unit', 'Afterward: retain the same quarter'][stage], { width: 320, weight: 700 });
    for (let shard = 0; shard < 4; shard++) body += `<rect x="${27 + shard * 77}" y="${vertical}" width="73" height="57" style="fill:${fill(!shard || stage === 1 ? 'input' : 'neutral')};stroke:${color(!shard || stage === 1 ? 'input' : 'neutral')};stroke-width:2"></rect>` + text(63 + shard * 77, vertical + 35, !shard || stage === 1 ? shard + 1 : '-', { width: 62, anchor: 'middle', weight: 700 });
  }
  return { body, height: 558 };
}

function gradientOwnership(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Freeze parameter updates without freezing every gradient path', { width: 320, weight: 700 });
  for (let step = 0; step < 2; step++) {
    const vertical = 173 + step * 284;
    body += text(20, vertical - 69, step ? 'G step: gradient passes through frozen D' : 'D step: stop the fake-sample path into G', { width: 320, weight: 700 });
    for (const [node, label] of ['G', 'fake', 'D', 'loss'].entries()) body += circle(49 + node * 86, vertical, 22, node === 2 ? 'state' : 'input') + text(49 + node * 86, vertical + 6, label, { width: 38, anchor: 'middle', size: 14, weight: 700 });
    for (let edge = 0; edge < 3; edge++) body += !step && edge === 0 ? line(112, vertical + 46, 72, vertical + 46, 'warning', true) : arrowBetween([49 + (edge + 1) * 86 - 24, vertical + 46], [49 + edge * 86 + 24, vertical + 46], 'output', tools, 3);
    body += text(20, vertical + 134, step ? 'D weights stay fixed; G owns this update.' : 'D owns the update; G receives no fake-sample gradient.', { width: 320, weight: 700 });
  }
  return { body, height: 658 };
}

function countFloor(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'The joint-pass floor counts actual trial identities', { width: 320, weight: 700 });
  for (let trial = 0; trial < picture.total; trial++) {
    const horizontal = 43 + trial % 10 * 30;
    const vertical = 145 + Math.floor(trial / 10) * 62;
    body += circle(horizontal, vertical, 7, trial < picture.passed ? 'output' : 'warning', true);
    if (trial === picture.required - 1) body += circle(horizontal, vertical, 14, 'focus');
  }
  body += text(20, 344, `${picture.passed}/${picture.total} = ${picture.passed / picture.total * 100}%; required ${picture.required}/${picture.total}`, { width: 320, weight: 700 }) + text(20, 438, 'Outlined trial fourteen must also become a joint pass.', { width: 320, tone: 'soft' });
  assert.equal(picture.required / picture.total, .7);
  return { body, height: 503 };
}

function replaceOneState(picture, model, tools) {
  const { text, color, fill, line } = tools;
  let body = text(20, 27, 'Replace the chosen component; preserve the other identity', { width: 320, weight: 700 });
  for (const [stage, values] of picture.states.entries()) {
    const vertical = 139 + stage * 178;
    for (const [component, value] of values.entries()) {
      const horizontal = 23 + component * 169;
      const changed = stage && values[component] !== picture.states[stage - 1][component];
      body += `<rect x="${horizontal}" y="${vertical}" width="145" height="99" style="fill:${fill(changed ? 'output' : 'input')};stroke:${color(changed ? 'output' : 'input')};stroke-width:${changed ? 3 : 1.5}"></rect>` + text(horizontal + 72, vertical + 44, value, { width: 124, anchor: 'middle', size: 16, weight: 700 });
    }
  }
  if (picture.states.length === 2) assert.equal(picture.states[0][1 - picture.changed], picture.states[1][1 - picture.changed]);
  else assert.equal(picture.states[1][1 - picture.changed], picture.states[2][1 - picture.changed]);
  return { body, height: 193 + picture.states.length * 178 };
}

function gainFloor(picture, model, tools) {
  const { text, line, circle } = tools;
  let body = text(20, 27, 'A positive gain can remain below the worthwhile floor', { width: 320, weight: 700 })
    + line(45, 199, 315, 199) + line(288, 166, 288, 232, 'warning', true) + circle(45 + picture.estimate / picture.required * 243, 199, 8, 'input', true)
    + text(20, 305, `Estimate ${picture.estimate} point; practical floor ${picture.required} point`, { width: 320, weight: 700 }) + text(20, 409, 'Nonzero and worthwhile are separate questions.', { width: 320, tone: 'soft' });
  assert(picture.estimate < picture.required);
  return { body, height: 474 };
}

function normalizedRanking(picture, model, tools) {
  const { text, line, circle, number } = tools;
  let body = text(20, 27, 'Keep finished beam identities; change the declared score', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 160 + stage * 267;
    const values = picture.probabilities.map((probability, index) => stage ? Math.log(probability) / picture.lengths[index] : probability);
    body += text(20, vertical - 56, stage ? 'Final score: log probability divided by length' : 'Raw complete-sequence probability', { width: 320, weight: 700 });
    for (const [beam, value] of values.entries()) body += text(20, vertical + beam * 76, `${picture.labels[beam]}: ${number(value)}`, { width: 320, weight: 700, tone: value === Math.max(...values) ? 'output' : 'soft' });
    body += text(20, vertical + 174, `Winner ${picture.labels[values.indexOf(Math.max(...values))]}`, { width: 320, weight: 700, tone: 'output' });
  }
  assert(picture.probabilities[0] > picture.probabilities[1]);
  assert(Math.log(picture.probabilities[0]) / picture.lengths[0] < Math.log(picture.probabilities[1]) / picture.lengths[1]);
  return { body, height: 665 };
}

function countVersusSize(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Read count and bytes per vector are different dimensions', { width: 320, weight: 700 });
  for (const [caseIndex, [count, size]] of [[4, 4], [2, 4], [4, 2], [2, 2]].entries()) {
    const vertical = 129 + caseIndex * 159;
    body += text(20, vertical - 38, ['Baseline', 'IVF: fewer read vectors', 'PQ: smaller vector representation', 'Both dimensions reduced'][caseIndex], { width: 320, weight: 700 });
    for (let vector = 0; vector < count; vector++) body += `<rect x="${26 + vector * 78}" y="${vertical}" width="71" height="${size * 15}" style="fill:${fill('input')};stroke:${color('input')};stroke-width:2"></rect>`;
  }
  return { body, height: 685 };
}

function hypothesisSeparation(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Choose the probe where predicted outcomes separate', { width: 320, weight: 700 });
  for (let probe = 0; probe < 2; probe++) {
    const vertical = 166 + probe * 252;
    body += text(20, vertical - 61, probe ? 'Strong probe: different predicted results' : 'Weak probe: both predict the same result', { width: 320, weight: 700 });
    for (let hypothesis = 0; hypothesis < 2; hypothesis++) body += circle(75 + hypothesis * 207, vertical, 26, 'input') + text(75 + hypothesis * 207, vertical + 6, `H${hypothesis + 1}`, { width: 46, anchor: 'middle', weight: 700 }) + arrowBetween([75 + hypothesis * 207, vertical + 32], [probe ? 75 + hypothesis * 207 : 180, vertical + 115], probe ? 'output' : 'neutral', tools, 3);
    if (probe) for (let hypothesis = 0; hypothesis < 2; hypothesis++) body += text(75 + hypothesis * 207, vertical + 157, hypothesis ? 'B' : 'A', { width: 80, anchor: 'middle', weight: 700, tone: 'output' });
    else body += text(180, vertical + 157, 'same', { width: 170, anchor: 'middle', weight: 700 });
  }
  return { body, height: 640 };
}

function futureNet(picture, model, tools) {
  const { text, line, circle } = tools;
  const horizontal = value => 156 + value * 43;
  const net = picture.value - picture.remaining;
  let body = text(20, 27, 'Only future value and remaining cost enter the next choice', { width: 320, weight: 700 })
    + line(34, 200, 320, 200) + circle(horizontal(0), 200, 4, 'neutral', true)
    + arrowBetween([horizontal(0), 168], [horizontal(picture.value), 168], 'input', tools, 3)
    + arrowBetween([horizontal(picture.value), 255], [horizontal(net), 255], 'warning', tools, 3)
    + text(20, 349, `Future ${picture.value} minus remaining ${picture.remaining} = ${net}`, { width: 320, weight: 700, tone: 'warning' })
    + text(20, 445, `Already spent ${picture.sunk} stays outside this forward calculation.`, { width: 320, tone: 'soft' });
  assert.equal(net, -2);
  return { body, height: 512 };
}

function scopeJoin(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Shared interface changes cross more than one owner’s boundary', { width: 320, weight: 700 })
    + circle(180, 152, 28, 'input') + text(180, 158, 'local', { width: 49, anchor: 'middle', size: 14, weight: 700 });
  for (let owner = 0; owner < 2; owner++) body += arrowBetween([180, 187], [90 + owner * 180, 293], 'state', tools, 3) + circle(90 + owner * 180, 329, 29, 'state') + text(90 + owner * 180, 335, `O${owner + 1}`, { width: 49, anchor: 'middle', weight: 700 });
  body += text(20, 438, 'Internal choice stays local; shared change requires alignment.', { width: 320, weight: 700 });
  return { body, height: 509 };
}

function recoverableFork(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Fork a bounded probe without erasing the committed mainline', { width: 320, weight: 700 }) + circle(180, 144, 29, 'input') + text(180, 150, 'saved', { width: 51, anchor: 'middle', size: 14, weight: 700 });
  for (let branch = 0; branch < 2; branch++) body += arrowBetween([180, 180], [88 + branch * 184, 290], branch ? 'state' : 'input', tools, 3) + circle(88 + branch * 184, 330, 30, branch ? 'state' : 'input') + text(88 + branch * 184, 402, picture.labels[branch], { width: 136, anchor: 'middle', size: 16, weight: 700 });
  body += line(272, 430, 272, 490, 'warning', true) + text(20, 565, 'Comparative evidence is required before probe promotion.', { width: 320, tone: 'soft' });
  return { body, height: 631 };
}

function cachedAndOutput(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Each call retains input cost and adds fresh generated output', { width: 320, weight: 700 });
  for (let call = 0; call < 3; call++) {
    const vertical = 146 + call * 160;
    body += text(20, vertical - 36, `Call ${call + 1}`, { width: 320, weight: 700 });
    for (let component = 0; component < 2; component++) body += `<rect x="${25 + component * 165}" y="${vertical}" width="145" height="70" style="fill:${fill(component ? 'warning' : 'input')};stroke:${color(component ? 'warning' : 'input')};stroke-width:2"></rect>` + text(97 + component * 165, vertical + 32, component ? 'fresh output' : 'cached input', { width: 124, anchor: 'middle', size: 15, weight: 700 });
  }
  body += text(20, 613, 'Caching discounts one component; it does not zero the total.', { width: 320, tone: 'soft' });
  return { body, height: 681 };
}

function logicalGrowth(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Packing gaps does not remove logical context identities', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 143 + stage * 263;
    body += text(20, vertical - 50, stage ? 'Longer context: eight tightly allocated identities' : 'Short context: four identities plus allocation gaps', { width: 320, weight: 700 });
    for (let slot = 0; slot < 8; slot++) {
      const present = stage || slot % 2 === 0;
      body += `<rect x="${24 + slot % 4 * 79}" y="${vertical + Math.floor(slot / 4) * 69}" width="72" height="58" style="fill:${fill(present ? 'input' : 'neutral')};stroke:${color(present ? 'input' : 'neutral')};stroke-width:2"></rect>` + text(60 + slot % 4 * 79, vertical + Math.floor(slot / 4) * 69 + 35, present ? stage ? slot + 1 : slot / 2 + 1 : '-', { width: 63, anchor: 'middle', weight: 700 });
    }
  }
  body += text(20, 614, 'Less waste can coexist with more logical KV and reads.', { width: 320, tone: 'soft' });
  return { body, height: 680 };
}

function differentTraces(picture, model, tools) {
  const { text, line, circle, color } = tools;
  let body = text(20, 27, 'Training and evaluation are separate measurement traces', { width: 320, weight: 700 });
  for (let trace = 0; trace < 2; trace++) {
    const vertical = 202 + trace * 224;
    body += text(20, vertical - 78, trace ? 'Evaluation alone spikes' : 'Training remains stable', { width: 320, weight: 700 });
    const coordinates = [[45, vertical], [135, vertical], [225, trace ? vertical - 49 : vertical], [315, vertical]];
    for (let index = 1; index < coordinates.length; index++) body += line(...coordinates[index - 1], ...coordinates[index], trace ? 'warning' : 'input');
  }
  body += text(20, 554, 'This localizes a discrepancy, not its complete cause.', { width: 320, tone: 'soft' });
  return { body, height: 619 };
}

function storedExecuted(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Retain all stored experts; execute only this token’s routes', { width: 320, weight: 700 });
  for (const [expert, active] of picture.active.entries()) {
    const horizontal = 52 + expert * 85;
    body += circle(horizontal, 152, 24, 'input') + text(horizontal, 158, `E${expert + 1}`, { width: 41, anchor: 'middle', weight: 700 });
    body += active ? arrowBetween([horizontal, 183], [horizontal, 328], 'output', tools, 3) : line(horizontal, 183, horizontal, 298, 'warning', true);
    body += circle(horizontal, 363, 24, active ? 'output' : 'neutral') + text(horizontal, 369, active ? '+' : '-', { width: 41, anchor: 'middle', weight: 700 });
  }
  body += text(20, 467, 'Four stored identities; two active experts for this token.', { width: 320, weight: 700 });
  return { body, height: 536 };
}

function contributionBoundaries(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Different contribution scopes require different attribution', { width: 320, weight: 700 });
  for (const [row, [claim, owner]] of [['design decision', 'owned'], ['execution', 'shared'], ['outcome', 'team']].entries()) {
    const vertical = 141 + row * 151;
    body += text(20, vertical + 7, claim, { width: 177, weight: 700 }) + circle(280, vertical, 29, row ? 'state' : 'input') + text(280, vertical + 6, owner, { width: 54, anchor: 'middle', size: 13, weight: 700 });
  }
  return { body, height: 519 };
}

function trainingInformation(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Joint training information is absent from the deployed actor', { width: 320, weight: 700 });
  for (let phase = 0; phase < 2; phase++) {
    const vertical = 168 + phase * 253;
    body += text(20, vertical - 59, phase ? 'Execution: actor has local observation only' : 'Training: centralized critic has joint state', { width: 320, weight: 700 }) + circle(81, vertical, 29, phase ? 'neutral' : 'state') + text(81, vertical + 6, phase ? '-' : 'joint', { width: 50, anchor: 'middle', size: 14, weight: 700 }) + circle(280, vertical, 29, 'input') + text(280, vertical + 6, 'local', { width: 50, anchor: 'middle', size: 14, weight: 700 });
    body += phase ? line(118, vertical, 244, vertical, 'warning', true) : arrowBetween([118, vertical], [244, vertical], 'state', tools, 3);
  }
  return { body, height: 510 };
}

function differentMeetingPoints(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'The join determines when image and text can interact', { width: 320, weight: 700 });
  for (let architecture = 0; architecture < 2; architecture++) {
    const vertical = 160 + architecture * 322;
    body += text(20, vertical - 57, architecture ? 'Late fusion: process separately, then join' : 'Early fusion: join before shared processing', { width: 320, weight: 700 });
    for (let modality = 0; modality < 2; modality++) {
      const horizontal = 83 + modality * 194;
      body += circle(horizontal, vertical, 24, modality ? 'state' : 'input') + text(horizontal, vertical + 6, modality ? 'text' : 'image', { width: 46, anchor: 'middle', size: 13, weight: 700 });
      if (architecture) body += arrowBetween([horizontal, vertical + 29], [horizontal, vertical + 86], modality ? 'state' : 'input', tools, 3) + circle(horizontal, vertical + 116, 24, modality ? 'state' : 'input');
      body += arrowBetween([horizontal, vertical + (architecture ? 147 : 30)], [180, vertical + (architecture ? 202 : 118)], modality ? 'state' : 'input', tools, 3);
    }
    body += circle(180, vertical + (architecture ? 236 : 155), 29, 'output') + text(180, vertical + (architecture ? 242 : 161), 'join', { width: 49, anchor: 'middle', size: 14, weight: 700 });
  }
  return { body, height: 785 };
}

function truthVersusPredictions(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Assignment consumes truth; inference suppression consumes predictions', { width: 320, weight: 700 });
  for (let phase = 0; phase < 2; phase++) {
    const vertical = 179 + phase * 253;
    body += text(20, vertical - 71, phase ? 'Inference: predictions only; suppress redundancy' : 'Training: match predictions to truth supervision', { width: 320, weight: 700 }) + circle(79, vertical, 29, phase ? 'neutral' : 'input') + text(79, vertical + 6, phase ? '-' : 'truth', { width: 50, anchor: 'middle', size: 14, weight: 700 }) + circle(280, vertical, 29, 'state') + text(280, vertical + 6, 'pred.', { width: 50, anchor: 'middle', size: 14, weight: 700 });
    body += phase ? line(116, vertical, 243, vertical, 'warning', true) : arrowBetween([116, vertical], [243, vertical], 'input', tools, 3);
  }
  return { body, height: 515 };
}

function parallelFinish(picture, model, tools) {
  const { text, color, fill, line } = tools;
  let body = text(20, 27, 'Parallel dependencies finish at their maximum duration', { width: 320, weight: 700 });
  for (const [caseIndex, values] of picture.cases.entries()) {
    const vertical = 138 + caseIndex * 210;
    const finish = Math.max(...values);
    body += text(20, vertical - 38, `Case ${caseIndex + 1}: max(${values.join(',')}) = ${finish}`, { width: 320, weight: 700 });
    for (const [task, value] of values.entries()) body += `<rect x="30" y="${vertical + task * 51}" width="${value * 2.9}" height="34" style="fill:${fill(task ? 'state' : 'input')};stroke:${color(task ? 'state' : 'input')};stroke-width:1.5"></rect>`;
    body += line(30 + finish * 2.9, vertical - 11, 30 + finish * 2.9, vertical + 105, 'warning', true);
  }
  return { body, height: 157 + picture.cases.length * 210 };
}

function logicalPhysicalPages(picture, model, tools) {
  const { text, circle, color, fill, line } = tools;
  let body = text(20, 27, 'The block table restores logical order from scattered pages', { width: 320, weight: 700 });
  for (const [block, page] of picture.pages.entries()) {
    const horizontal = 61 + block * 119;
    body += circle(horizontal, 151, 25, 'input') + text(horizontal, 157, block, { width: 43, anchor: 'middle', weight: 700 }) + text(horizontal, 232, `page ${page}`, { width: 98, anchor: 'middle', weight: 700 });
  }
  for (let page = 0; page < 10; page++) {
    const present = picture.pages.includes(page);
    const horizontal = 26 + page % 5 * 63;
    const vertical = 333 + Math.floor(page / 5) * 84;
    body += `<rect x="${horizontal}" y="${vertical}" width="56" height="60" style="fill:${fill(present ? 'output' : 'neutral')};stroke:${color(present ? 'output' : 'neutral')};stroke-width:2"></rect>` + text(horizontal + 28, vertical + 36, page, { width: 47, anchor: 'middle', weight: 700 });
  }
  body += text(20, 563, 'Logical order is page 7, then page 2, then page 9.', { width: 320, weight: 700 });
  assert.equal(new Set(picture.pages).size, 3);
  return { body, height: 628 };
}

function pipelineWave(picture, model, tools) {
  const { text, color, fill, line } = tools;
  let body = text(20, 27, 'Follow each microbatch through the diagonal stage wave', { width: 320, weight: 700 });
  let occupied = 0;
  for (let stage = 0; stage < 3; stage++) for (let time = 0; time < 5; time++) {
    const label = model.values[stage][time];
    const active = label !== '-';
    occupied += Number(active);
    const tone = active ? ['input', 'state', 'output'][label.charCodeAt(0) - 65] : 'neutral';
    const horizontal = 29 + time * 61;
    const vertical = 145 + stage * 81;
    body += `<rect x="${horizontal}" y="${vertical}" width="56" height="66" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>` + text(horizontal + 28, vertical + 41, label, { width: 48, anchor: 'middle', weight: 700 });
  }
  assert.equal(occupied, 9);
  body += text(20, 474, 'Five elapsed slots; nine busy and six idle stage-slots.', { width: 320, weight: 700 }) + text(20, 551, 'Idle stage capacity is already inside the same wall clock.', { width: 320, tone: 'soft' });
  return { body, height: 617 };
}

function distinctExamples(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'Count each distinct example once across the data replicas', { width: 320, weight: 700 });
  let count = 0;
  for (let replica = 0; replica < picture.replicas; replica++) {
    const vertical = 144 + replica * 243;
    body += text(20, vertical - 42, `Data replica ${replica + 1}: four batches of sixteen`, { width: 320, weight: 700 });
    for (let batch = 0; batch < picture.batches; batch++) for (let example = 0; example < picture.batchSize; example++) {
      body += `<g data-picture-example-member="${count++}">${circle(29 + example * 20, vertical + batch * 36, 3, replica ? 'output' : 'input', true)}</g>`;
    }
  }
  assert.equal(count, 128);
  body += text(20, 606, '2 x 4 x 16 = 128. TP and PP reuse these same identities.', { width: 320, weight: 700 });
  return { body, height: 673 };
}

function oneLoadManyComputes(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'One loaded weight tile serves four token computations', { width: 320, weight: 700 }) + circle(180, 154, 32, 'input') + text(180, 160, 'W', { width: 54, anchor: 'middle', weight: 700 });
  for (let token = 0; token < 4; token++) body += arrowBetween([180, 193], [52 + token * 85, 322], 'input', tools, 3) + circle(52 + token * 85, 361, 25, 'output') + text(52 + token * 85, 367, token + 1, { width: 44, anchor: 'middle', weight: 700 });
  body += text(20, 477, 'Each token still needs its math and cache reads.', { width: 320, weight: 700 });
  return { body, height: 546 };
}

function overlapTail(picture, model, tools) {
  const { text, color, fill, line } = tools;
  const scale = 7.2;
  let body = text(20, 27, 'Only communication beyond useful overlap extends the step', { width: 320, weight: 700 })
    + text(20, 107, 'Communication: forty shared-clock units', { width: 320, weight: 700 })
    + `<rect x="30" y="165" width="${picture.communication * scale}" height="47" style="fill:${fill('input')};stroke:${color('input')};stroke-width:2"></rect>`
    + text(20, 284, 'Useful computation covers the first thirty', { width: 320, weight: 700 })
    + `<rect x="30" y="342" width="${picture.covered * scale}" height="47" style="fill:${fill('output')};stroke:${color('output')};stroke-width:2"></rect>`
    + line(30 + picture.covered * scale, 150, 30 + picture.covered * scale, 258, 'focus', true)
    + line(30 + picture.covered * scale, 318, 30 + picture.covered * scale, 404, 'focus', true)
    + text(20, 485, `Exposed tail ${picture.communication} - ${picture.covered} = ${picture.communication - picture.covered}`, { width: 320, weight: 700, tone: 'warning' });
  return { body, height: 552 };
}

function versionedChunk(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Carry one chunk identity and version through the answer lineage', { width: 320, weight: 700 });
  for (const [stage, label] of ['doc D v3; indexed C7', 'request retrieves C7 v3', 'claim cites actual C7 v3'].entries()) {
    const vertical = 149 + stage * 170;
    body += circle(63, vertical, 27, 'input') + text(63, vertical + 6, 'C7', { width: 47, anchor: 'middle', weight: 700 }) + text(124, vertical + 6, label, { width: 211, weight: 700 });
    if (stage) body += line(63, vertical - 138, 63, vertical - 31, 'input');
  }
  return { body, height: 569 };
}

function ageVersusLatency(picture, model, tools) {
  const { text, line, circle } = tools;
  const horizontal = value => 40 + value * 28;
  let body = text(20, 27, 'The two durations begin at different clock events', { width: 320, weight: 700 });
  for (let lane = 0; lane < 2; lane++) {
    const vertical = 174 + lane * 232;
    const start = lane ? 9.9 : 0;
    body += line(40, vertical, 320, vertical) + circle(horizontal(start), vertical, 5, lane ? 'input' : 'warning', true) + circle(horizontal(10), vertical, 5, 'output', true);
    body += text(20, vertical - 55, lane ? 'Request starts 9.9; response returns 10' : 'Snapshot time zero; response time ten', { width: 320, weight: 700 }) + text(20, vertical + 81, lane ? 'Request latency 10 - 9.9 = 0.1' : 'Feature age 10 - 0 = 10', { width: 320, weight: 700, tone: lane ? 'input' : 'warning' });
  }
  assert(Math.abs(10 - 9.9 - .1) < 1e-12);
  return { body, height: 554 };
}

function acceptedBaseline(picture, model, tools) {
  const { text, circle, line } = tools;
  let baseline;
  let body = text(20, 27, 'Only accepted quality-qualified changes replace the baseline', { width: 320, weight: 700 });
  for (const [step, value] of picture.values.entries()) {
    const vertical = 147 + step * 167;
    if (picture.accepted[step]) baseline = value;
    body += circle(66, vertical, 25, picture.accepted[step] ? 'input' : 'warning') + text(66, vertical + 6, value, { width: 44, anchor: 'middle', weight: 700 })
      + (picture.accepted[step] ? arrowBetween([97, vertical], [251, vertical], 'output', tools, 3) : line(97, vertical, 230, vertical, 'warning', true))
      + circle(286, vertical, 25, 'output') + text(286, vertical + 6, baseline, { width: 44, anchor: 'middle', weight: 700 })
      + text(20, vertical + 83, picture.accepted[step] ? 'Quality passes; baseline accepts this cost.' : 'Tail quality fails; baseline stays eight.', { width: 320, weight: 700 });
  }
  assert.equal(baseline, 8);
  return { body, height: 633 };
}

function opportunityCount(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'The downstream decision sees only its available identities', { width: 320, weight: 700 });
  for (const [row, count] of picture.counts.entries()) {
    const vertical = 140 + row * 252;
    const unit = picture.units?.[row] ?? 1;
    assert.equal(count % unit, 0);
    body += text(20, vertical - 45, picture.labels[row], { width: 320, weight: 700 });
    for (let item = 0; item < count / unit; item++) body += circle(29 + item % 20 * 15, vertical + Math.floor(item / 20) * 27, 3, row ? 'output' : 'input', true);
    body += text(20, vertical + 172, `${count} available identities; one mark ${unit}`, { width: 320, weight: 700 });
  }
  return { body, height: 636 };
}

function preferenceRoutes(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Preference pairs feed two different optimization routes', { width: 320, weight: 700 });
  for (let route = 0; route < 2; route++) {
    const vertical = 161 + route * 258;
    body += text(20, vertical - 75, route ? 'DPO: direct pair objective' : 'Illustrated RLHF: reward fitting then rollouts', { width: 320, weight: 700 });
    const labels = route ? ['pairs', 'direct loss'] : ['pairs', 'reward', 'rollouts'];
    for (const [stage, label] of labels.entries()) {
      const horizontal = 63 + stage * (234 / (labels.length - 1));
      body += circle(horizontal, vertical, 34, stage ? 'output' : 'input') + text(horizontal, vertical + 6, label, { width: 72, anchor: 'middle', size: 14, weight: 700 });
      if (stage) body += arrowBetween([horizontal - 234 / (labels.length - 1) + 42, vertical], [horizontal - 42, vertical], 'input', tools, 3);
    }
  }
  return { body, height: 508 };
}

function labelVisitedState(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'The corrective label belongs to the state the learner visited', { width: 320, weight: 700 })
    + circle(70, 159, 27, 'input') + text(70, 165, 'demo', { width: 47, anchor: 'middle', size: 14, weight: 700 })
    + circle(280, 159, 27, 'warning') + text(280, 165, 'drift', { width: 47, anchor: 'middle', size: 14, weight: 700 })
    + arrowBetween([280, 192], [280, 306], 'warning', tools, 3) + circle(280, 345, 30, 'state') + text(280, 351, 'expert', { width: 53, anchor: 'middle', size: 14, weight: 700 })
    + arrowBetween([280, 384], [180, 482], 'output', tools, 3) + circle(180, 521, 30, 'output') + text(180, 527, 'data', { width: 52, anchor: 'middle', size: 14, weight: 700 })
    + text(20, 625, 'Expert action labels the visited drift state in aggregated data.', { width: 320, weight: 700 });
  return { body, height: 692 };
}

function prefixCorrection(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'First rejection cuts the dependent draft suffix', { width: 320, weight: 700 });
  for (let token = 0; token < 3; token++) body += circle(61 + token * 119, 148, 25, token ? 'warning' : 'input') + text(61 + token * 119, 154, String.fromCharCode(65 + token), { width: 44, anchor: 'middle', weight: 700 }) + text(61 + token * 119, 230, ['accept', 'reject', 'discard'][token], { width: 100, anchor: 'middle', size: 16, weight: 700 });
  for (const horizontal of [180, 299]) body += line(horizontal, 183, horizontal, 207, 'warning', true) + line(horizontal, 246, horizontal, 324, 'warning', true);
  for (const [token, label] of ['A', 'replace'].entries()) body += circle(93 + token * 174, 405, 31, 'output') + text(93 + token * 174, 411, label, { width: 64, anchor: 'middle', size: 13, weight: 700 });
  body += text(20, 519, 'Continue from accepted A plus the corrected replacement.', { width: 320, weight: 700 });
  return { body, height: 587 };
}

function revisableWords(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Visible hypotheses remain revisable before finalization', { width: 320, weight: 700 });
  for (const [stage, words] of [['I', 'scream'], ['ice', 'cream'], ['ice', 'cream']].entries()) {
    const vertical = 145 + stage * 174;
    body += text(20, vertical - 41, stage === 2 ? 'Finalization receipt: committed' : 'Provisional hypothesis', { width: 320, weight: 700 });
    for (const [word, value] of words.entries()) body += circle(87 + word * 182, vertical, 32, stage === 2 ? 'output' : 'neutral') + text(87 + word * 182, vertical + 6, value, { width: 57, anchor: 'middle', size: 15, weight: 700 });
  }
  return { body, height: 572 };
}

function deviceTimeArea(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Device count times elapsed hours gives the resource area', { width: 320, weight: 700 });
  for (const [run, [devices, hours]] of picture.cases.entries()) {
    const vertical = 143 + run * 271;
    body += text(20, vertical - 43, `${devices} devices x ${hours} hours = ${devices * hours}`, { width: 320, weight: 700 });
    for (let device = 0; device < devices; device++) body += `<rect x="${26 + device * 18}" y="${vertical}" width="18" height="${hours * 70}" style="fill:${fill(run ? 'warning' : 'input')};stroke:${color(run ? 'warning' : 'input')};stroke-width:1.2"></rect>`;
  }
  assert.equal(picture.cases[0][0] * picture.cases[0][1], 16);
  assert.equal(picture.cases[1][0] * picture.cases[1][1], 24);
  return { body, height: 583 };
}

function partialCoordinateSum(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'Channel shards contribute to one output coordinate', { width: 320, weight: 700 });
  for (let shard = 0; shard < 2; shard++) body += circle(90 + shard * 180, 155, 29, 'input') + text(90 + shard * 180, 161, shard ? '6' : '4', { width: 50, anchor: 'middle', weight: 700 }) + arrowBetween([90 + shard * 180, 191], [180, 308], 'input', tools, 3);
  body += circle(180, 350, 32, 'output') + text(180, 356, '10', { width: 55, anchor: 'middle', weight: 700 }) + text(20, 477, '4 + 6 reconstructs one coordinate, not two concatenated ones.', { width: 320, weight: 700 });
  return { body, height: 547 };
}

function partitionedObject(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Each parallel axis partitions a different object', { width: 320, weight: 700 });
  for (const [axis, [name, object]] of [['DP', 'examples'], ['PP', 'layers'], ['TP', 'tensor'], ['ZeRO', 'state']].entries()) {
    const vertical = 139 + axis * 157;
    body += text(20, vertical - 38, `${name}: partition ${object}`, { width: 320, weight: 700 });
    for (let shard = 0; shard < 2; shard++) body += `<rect x="${26 + shard * 155}" y="${vertical}" width="150" height="67" style="fill:${fill(shard ? 'output' : 'input')};stroke:${color(shard ? 'output' : 'input')};stroke-width:2"></rect>` + text(101 + shard * 155, vertical + 36, `${object} ${shard}`, { width: 131, anchor: 'middle', size: 15, weight: 700 });
  }
  return { body, height: 673 };
}

function offlineIndexLiveQuery(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Encode items before the request; search their retained index', { width: 320, weight: 700 })
    + circle(89, 152, 30, 'input') + text(89, 158, 'items', { width: 53, anchor: 'middle', size: 14, weight: 700 })
    + arrowBetween([126, 152], [235, 152], 'input', tools, 3) + circle(278, 152, 30, 'state') + text(278, 158, 'index', { width: 53, anchor: 'middle', size: 14, weight: 700 })
    + line(25, 285, 335, 285, 'focus', true) + text(20, 346, 'Live request boundary', { width: 320, weight: 700 })
    + circle(89, 445, 30, 'output') + text(89, 451, 'query', { width: 53, anchor: 'middle', size: 14, weight: 700 })
    + arrowBetween([124, 421], [259, 188], 'output', tools, 3) + text(20, 572, 'Query encoding happens now; item vectors already exist.', { width: 320, weight: 700 });
  return { body, height: 640 };
}

function differentLatentSource(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Generation reaches the decoder from the prior, not an encoder', { width: 320, weight: 700 });
  for (let phase = 0; phase < 2; phase++) {
    const vertical = 169 + phase * 260;
    body += text(20, vertical - 66, phase ? 'Generate: no observed input; sample prior z' : 'Train: observed x is encoded into q(z|x)', { width: 320, weight: 700 });
    const labels = phase ? ['prior', 'z', 'decode'] : ['x', 'encode', 'decode'];
    for (const [stage, label] of labels.entries()) {
      const horizontal = 61 + stage * 119;
      body += circle(horizontal, vertical, 34, stage ? 'output' : 'input') + text(horizontal, vertical + 6, label, { width: 72, anchor: 'middle', size: 14, weight: 700 });
      if (stage) body += arrowBetween([horizontal - 77, vertical], [horizontal - 42, vertical], 'input', tools, 3);
    }
  }
  return { body, height: 519 };
}

function costPerOutput(picture, model, tools) {
  const { text, color, fill, line, number } = tools;
  let body = text(20, 27, 'The same total work is divided by accepted output count', { width: 320, weight: 700 });
  for (const [caseIndex, outputs] of picture.outputs.entries()) {
    const vertical = 147 + caseIndex * 258;
    body += text(20, vertical - 48, `${picture.total} total work / ${outputs} accepted`, { width: 320, weight: 700 });
    for (let output = 0; output < outputs; output++) body += `<rect x="${27 + output * 286 / outputs}" y="${vertical}" width="${286 / outputs}" height="62" style="fill:${fill(outputs > 1 ? 'output' : 'warning')};stroke:${color(outputs > 1 ? 'output' : 'warning')};stroke-width:2"></rect>`;
    body += text(20, vertical + 130, `${number(picture.total / outputs)} per token; baseline ${picture.baseline}`, { width: 320, weight: 700, tone: outputs > 1 ? 'output' : 'warning' });
  }
  assert.equal(picture.total / picture.outputs[0], .65);
  return { body, height: 603 };
}

function splitDependencies(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'New-user and future-time claims need different boundaries', { width: 320, weight: 700 });
  for (let split = 0; split < 2; split++) {
    const vertical = 163 + split * 255;
    body += text(20, vertical - 59, split ? 'Temporal split: past train, future test' : 'Group split: A trains, new user B tests', { width: 320, weight: 700 }) + line(180, vertical - 35, 180, vertical + 113, 'focus', true);
    for (let position = 0; position < 4; position++) body += circle(53 + position * 85, vertical, 23, position < 2 ? 'input' : 'output') + text(53 + position * 85, vertical + 6, split ? position + 1 : position < 2 ? 'A' : 'B', { width: 40, anchor: 'middle', weight: 700 }) + text(53 + position * 85, vertical + 80, position < 2 ? 'train' : 'test', { width: 70, anchor: 'middle', size: 16, weight: 700 });
  }
  return { body, height: 568 };
}

function epochMembers(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, picture.title ?? 'Late packets cannot reopen the retired epoch', { width: 320, weight: 700 });
  for (const [row, [active, packet]] of [[7, 7], [8, 7], [8, 8]].entries()) {
    const vertical = 148 + row * 174;
    const accepted = active === packet;
    body += text(20, vertical - 42, `Active ${active}; packet ${packet}`, { width: 320, weight: 700 }) + circle(69, vertical, 25, 'input') + text(69, vertical + 6, packet, { width: 44, anchor: 'middle', weight: 700 })
      + (accepted ? arrowBetween([100, vertical], [250, vertical], 'output', tools, 3) : line(100, vertical, 232, vertical, 'warning', true)) + circle(284, vertical, 25, accepted ? 'output' : 'warning') + text(284, vertical + 6, accepted ? '+' : 'x', { width: 44, anchor: 'middle', weight: 700 });
  }
  body += text(20, 634, picture.footer ?? 'Rejecting packet seven leaves active epoch eight intact.', { width: 320, weight: 700 });
  return { body, height: 701 };
}

function positiveInWhole(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'The declared positive belongs to numerator and denominator', { width: 320, weight: 700 }) + text(20, 105, 'Numerator: declared positive only', { width: 320, weight: 700 }) + circle(62, 177, 28, 'output') + text(62, 183, '+', { width: 49, anchor: 'middle', weight: 700 })
    + text(112, 305, 'Denominator: the whole declared candidate set', { width: 228, weight: 700 });
  for (let candidate = 0; candidate < 3; candidate++) body += circle(62 + candidate * 118, 401, 28, candidate ? 'input' : 'output') + text(62 + candidate * 118, 407, candidate ? `N${candidate}` : '+', { width: 49, anchor: 'middle', weight: 700 });
  body += line(62, 213, 62, 367, 'output') + text(20, 517, 'Same positive identity; negatives remain declared negatives.', { width: 320, weight: 700 });
  return { body, height: 584 };
}

function scaledClippingOrder(picture, model, tools) {
  const { text, circle, line, number } = tools;
  const scaled = picture.gradient * picture.scale;
  const correct = Math.min(picture.limit, scaled / picture.scale);
  const incorrect = Math.min(picture.limit, scaled) / picture.scale;
  let body = text(20, 27, 'Unscale before clipping to preserve the intended threshold', { width: 320, weight: 700 });
  for (let order = 0; order < 2; order++) {
    const vertical = 162 + order * 239;
    const values = order ? [scaled, picture.limit, incorrect] : [scaled, picture.gradient, correct];
    body += text(20, vertical - 61, order ? 'Wrong: clip, then unscale' : 'Correct: unscale, then clip', { width: 320, weight: 700 });
    for (const [stage, value] of values.entries()) {
      const horizontal = 62 + stage * 118;
      body += circle(horizontal, vertical, 28, order ? 'warning' : 'input') + text(horizontal, vertical + 6, number(value), { width: 49, anchor: 'middle', size: 14, weight: 700 });
      if (stage) body += arrowBetween([horizontal - 83, vertical], [horizontal - 35, vertical], order ? 'warning' : 'input', tools, 3);
    }
  }
  body += text(20, 558, 'Nonfinite values stop before the parameter step.', { width: 320, weight: 700, tone: 'warning' });
  assert.equal(correct, 1);
  assert.equal(incorrect, .01);
  return { body, height: 625 };
}

function lastStepPaths(picture, model, tools) {
  const { text, color, fill } = tools;
  const paths = [];
  const visit = (remaining, path) => { if (!remaining) paths.push(path); else for (const step of [1, 2]) if (step <= remaining) visit(remaining - step, [...path, step]); };
  visit(picture.target, []);
  paths.sort((left, right) => left.at(-1) - right.at(-1));
  const counts = [1, 2].map(step => paths.filter(path => path.at(-1) === step).length);
  assert.deepEqual(counts, picture.expected ?? [5, 3]);
  assert.equal(paths.length, counts[0] + counts[1]);
  let body = text(20, 27, `Sort all ${paths.length} paths by their disjoint final move`, { width: 320, weight: 700 });
  for (const [row, path] of paths.entries()) {
    let horizontal = 27;
    const vertical = 117 + row * 51;
    for (const [index, step] of path.entries()) {
      const tone = index === path.length - 1 ? step === 1 ? 'input' : 'output' : 'neutral';
      body += `<rect x="${horizontal}" y="${vertical}" width="${step * 33}" height="28" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>` + text(horizontal + step * 16.5, vertical + 20, step, { width: step * 33 - 6, anchor: 'middle', size: 14, weight: 700 });
      horizontal += step * 33;
    }
    body += text(225, vertical + 21, `last ${path.at(-1)}`, { width: 109, weight: 700, tone: path.at(-1) === 1 ? 'input' : 'output' });
  }
  body += text(20, 609, `${counts[0]} end in one; ${counts[1]} end in two. Every path sums to ${picture.target}.`, { width: 320, weight: 700 });
  return { body, height: 679 };
}

function prerequisiteCounter(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Each emitted parent removes one unmet prerequisite', { width: 320, weight: 700 });
  for (let emitted = 0; emitted <= 2; emitted++) {
    const vertical = 137 + emitted * 188;
    const unmet = 2 - emitted;
    for (let parent = 0; parent < 2; parent++) body += circle(60 + parent * 91, vertical, 23, parent < emitted ? 'output' : 'neutral') + text(60 + parent * 91, vertical + 6, parent + 1, { width: 40, anchor: 'middle', weight: 700 });
    body += arrowBetween([183, vertical], [257, vertical], unmet ? 'warning' : 'output', tools, 3) + circle(295, vertical, 26, unmet ? 'warning' : 'output') + text(295, vertical + 6, unmet, { width: 46, anchor: 'middle', weight: 700 }) + text(20, vertical + 93, unmet ? `${unmet} unmet: course three remains blocked.` : 'Zero unmet: course three becomes ready.', { width: 320, weight: 700 });
  }
  return { body, height: 674 };
}

function nonadjacentChoice(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Taking the last house excludes its immediate neighbor', { width: 320, weight: 700 });
  for (let choice = 0; choice < 2; choice++) {
    const vertical = 163 + choice * 240;
    const selected = choice ? [1, 3] : [0, 2];
    body += text(20, vertical - 55, choice ? 'Take house three: earlier safe prefix seven + three' : 'Skip house three: keep houses zero and two', { width: 320, weight: 700 });
    for (const [house, value] of picture.values.entries()) body += circle(54 + house * 84, vertical, 24, selected.includes(house) ? 'input' : 'neutral') + text(54 + house * 84, vertical + 6, value, { width: 42, anchor: 'middle', weight: 700 });
    const total = selected.reduce((sum, house) => sum + picture.values[house], 0);
    body += text(20, vertical + 112, `Total ${total}`, { width: 320, weight: 700, tone: choice ? 'input' : 'output' });
    assert.equal(total, choice ? 10 : 11);
  }
  return { body, height: 583 };
}

function conditionalDiagonal(picture, model, tools) {
  const { text, color, fill, circle, line } = tools;
  let body = text(20, 27, 'A matching character extends the diagonal predecessor', { width: 320, weight: 700 });
  for (let row = 0; row < 3; row++) for (let column = 0; column < 4; column++) {
    const horizontal = 44 + column * 69;
    const vertical = 147 + row * 75;
    const selected = row === 2 && column === 3 || row === 1 && column === 2;
    body += `<rect x="${horizontal}" y="${vertical}" width="64" height="67" style="fill:${fill(selected ? 'input' : 'neutral')};stroke:${color(selected ? 'input' : 'neutral')};stroke-width:2"></rect>` + text(horizontal + 32, vertical + 41, model.values[row][column], { width: 55, anchor: 'middle', weight: 700 });
  }
  body += arrowBetween([251, 277], [266, 292], 'output', tools, 3) + text(20, 453, 'AB versus ACB: matching B uses diagonal one + one = two.', { width: 320, weight: 700 });
  assert.equal(model.values[2][3], model.values[1][2] + 1);
  return { body, height: 523 };
}

function oldSnapshotExtension(picture, model, tools) {
  const { text, circle, line } = tools;
  const created = picture.old.map(value => value + picture.item);
  const union = [...new Set([...picture.old, ...created])];
  let body = text(20, 27, 'Extend only the frozen old snapshot with the current item', { width: 320, weight: 700 });
  for (const [member, value] of picture.old.entries()) {
    const horizontal = picture.old.length === 1 ? 180 : 87 + member * 182;
    body += circle(horizontal, 158, 28, 'input') + text(horizontal, 164, value, { width: 49, anchor: 'middle', weight: 700 })
      + line(horizontal, 193, horizontal, 238, 'input', false, 3) + arrowBetween([horizontal, 282], [horizontal, 312], 'input', tools, 3)
      + circle(horizontal, 350, 28, 'output') + text(horizontal, 356, created[member], { width: 49, anchor: 'middle', weight: 700 });
  }
  body += text(20, 264, `Add item ${picture.item} once to each old value`, { width: 320, weight: 700 }) + text(20, 471, `Publish union ${union.join(', ')} only afterward.`, { width: 320, weight: 700 }) + text(20, 559, 'New sums cannot feed back through this same item.', { width: 320, tone: 'soft' });
  assert(!union.includes(2 * picture.item));
  return { body, height: 625 };
}

function graphVersusParents(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Graph edges remain fixed while parent bookkeeping changes', { width: 320, weight: 700 });
  for (let node = 0; node < 4; node++) {
    const horizontal = 57 + node * 82;
    body += circle(horizontal, 144, 22, 'input') + text(horizontal, 150, node + 1, { width: 39, anchor: 'middle', weight: 700 });
    if (node) body += line(horizontal - 57, 144, horizontal - 25, 144, 'input');
  }
  for (let stage = 0; stage < 2; stage++) {
    const vertical = 322 + stage * 185;
    body += text(20, vertical - 46, stage ? 'After compression: parent of four is one' : 'Before compression: parent of four is three', { width: 320, weight: 700 });
    for (const [index, label] of ['2->1', '3->1', stage ? '4->1' : '4->3'].entries()) body += text(58 + index * 119, vertical + 6, label, { width: 104, anchor: 'middle', weight: 700, tone: stage && index === 2 ? 'output' : 'input' });
  }
  body += text(20, 620, 'No new four-one problem edge was added.', { width: 320, tone: 'soft' });
  return { body, height: 686 };
}

function orderedChoicePath(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'A permitted choice order emits each membership once', { width: 320, weight: 700 });
  for (const [row, path] of picture.paths.entries()) {
    const vertical = 157 + row * 244;
    const valid = path.every((value, index) => !index || (picture.strict ? value > path[index - 1] : value >= path[index - 1]));
    for (const [step, value] of path.entries()) {
      const horizontal = 62 + step * 236 / (path.length - 1);
      body += circle(horizontal, vertical, 25, valid ? 'input' : 'warning') + text(horizontal, vertical + 6, value, { width: 44, anchor: 'middle', weight: 700 });
      if (step) body += arrowBetween([horizontal - 236 / (path.length - 1) + 32, vertical], [horizontal - 32, vertical], valid ? 'input' : 'warning', tools, 3);
    }
    body += text(20, vertical + 108, valid ? 'Allowed order: emit this membership.' : 'Backward order: duplicate membership; do not emit.', { width: 320, weight: 700, tone: valid ? 'output' : 'warning' });
  }
  return { body, height: 578 };
}

function monotonePairBound(picture, model, tools) {
  const { text, circle, line } = tools;
  const sums = picture.partners.map(partner => picture.fixed + picture.left + partner);
  const maximum = Math.max(...sums);
  assert(maximum < picture.target);
  assert(picture.partners.every((value, index) => !index || value >= picture.partners[index - 1]));
  let body = text(20, 27, `Fixed ${picture.fixed} and left ${picture.left}: even the largest partner fails`, { width: 320, weight: 700 });
  for (const [index, partner] of picture.partners.entries()) {
    const horizontal = 91 + index * 178;
    body += circle(horizontal, 169, 28, 'input') + text(horizontal, 175, partner, { width: 49, anchor: 'middle', weight: 700 })
      + arrowBetween([horizontal, 204], [horizontal, 303], 'warning', tools, 3) + circle(horizontal, 342, 28, 'warning') + text(horizontal, 348, sums[index], { width: 49, anchor: 'middle', weight: 700 });
  }
  body += text(20, 110, `Add ${picture.fixed + picture.left} to each partner`, { width: 320, weight: 700 }) + text(20, 452, `Largest possible sum ${maximum} is below target ${picture.target}.`, { width: 320, weight: 700, tone: 'warning' })
    + line(30, 545, 330, 545, 'neutral');
  for (const value of [...sums, picture.target]) {
    const horizontal = 56 + (value - sums[0]) * 274 / (picture.target - sums[0]);
    body += circle(horizontal, 545, 8, value === picture.target ? 'output' : 'warning') + text(horizontal, 581, value, { width: 40, anchor: 'middle' });
  }
  assert.equal(maximum, -4);
  return { body, height: 629 };
}

function candidateCut(picture, model, tools) {
  const { text, circle, line } = tools;
  const answer = picture.minimum ? Math.min(...picture.values) : picture.target;
  let previous = picture.values.map((value, index) => index);
  let body = text(20, 27, 'One comparison removes a region, not just one item', { width: 320, weight: 700 });
  for (const [stage, state] of picture.stages.entries()) {
    const vertical = 160 + stage * 190;
    assert(state.keep.every(index => previous.includes(index)));
    assert(state.keep.some(index => picture.values[index] === answer));
    body += text(20, vertical - 68, state.label, { width: 320, weight: 700 });
    for (const [index, value] of picture.values.entries()) {
      const horizontal = 43 + index * 274 / (picture.values.length - 1);
      const retained = state.keep.includes(index);
      body += circle(horizontal, vertical, 20, retained ? 'input' : 'neutral') + text(horizontal, vertical + 5, value, { width: 36, anchor: 'middle', size: 14, weight: 700 })
        + text(horizontal, vertical + 53, index, { width: 36, anchor: 'middle', size: 14, tone: 'soft' });
      if (!retained) body += line(horizontal - 14, vertical - 14, horizontal + 14, vertical + 14, 'warning');
    }
    previous = state.keep;
  }
  const footer = 160 + (picture.stages.length - 1) * 190 + 117;
  body += text(20, footer, 'Original positions stay fixed; crossed items cannot contain the answer.', { width: 320, weight: 700 });
  return { body, height: footer + 72 };
}

function shorterWallBound(picture, model, tools) {
  const { text, line, color, fill } = tools;
  let body = text(20, 27, 'Keeping the short wall leaves the same height ceiling', { width: 320, weight: 700 });
  for (const [stage, width] of picture.widths.entries()) {
    const vertical = 271 + stage * 260;
    const right = 48 + width * 84;
    const ceiling = vertical - picture.lower * 36;
    body += text(20, vertical - 164, stage ? 'Move only the taller endpoint inward' : 'Current width three; shorter wall two', { width: 320, weight: 700 })
      + `<rect x="48" y="${ceiling}" width="${width * 84}" height="${picture.lower * 36}" style="fill:${fill('input')};stroke:${color('input')};stroke-width:1.5"></rect>`
      + line(48, vertical, 48, ceiling, 'output') + line(right, vertical, right, vertical - 126, 'neutral', !!stage) + line(32, ceiling, 331, ceiling, 'warning', true)
      + text(32, ceiling - 18, `height <= ${picture.lower}`, { width: 170, weight: 700, tone: 'warning' }) + text(20, vertical + 62, `${stage ? 'Area at most' : 'Area'} ${picture.lower * width}; width ${width}`, { width: 320, weight: 700 });
  }
  assert.deepEqual(picture.widths.map(width => width * picture.lower), [6, 4]);
  return { body, height: 648 };
}

function ceilingHours(picture, model, tools) {
  const { text, color, fill } = tools;
  const totals = [];
  let body = text(20, 27, 'A partial final hour still occupies one hour box', { width: 320, weight: 700 });
  for (const [stage, speed] of picture.speeds.entries()) {
    const vertical = 164 + stage * 310;
    const hours = picture.piles.map(pile => Math.ceil(pile / speed));
    const total = hours.reduce((sum, count) => sum + count, 0);
    totals.push(total);
    body += text(20, vertical - 67, `Speed ${speed}; piles keep their separate hour counts`, { width: 320, weight: 700 });
    for (const [index, pile] of picture.piles.entries()) {
      const horizontal = 28 + index * 80;
      body += text(horizontal + 27, vertical - 17, pile, { width: 54, anchor: 'middle', weight: 700 });
      for (let hour = 0; hour < hours[index]; hour++) {
        const amount = Math.min(speed, pile - hour * speed);
        body += `<rect x="${horizontal}" y="${vertical + hour * 29}" width="54" height="26" style="fill:none;stroke:${color('neutral')};stroke-width:1.5"></rect>`
          + `<rect x="${horizontal}" y="${vertical + hour * 29}" width="${54 * amount / speed}" height="26" style="fill:${fill('input')}"></rect>` + text(horizontal + 27, vertical + hour * 29 + 19, amount, { width: 48, anchor: 'middle', size: 14, weight: 700 });
      }
    }
    body += text(20, vertical + 170, `${hours.join(' + ')} = ${total} hours; deadline ${picture.hours}`, { width: 320, weight: 700, tone: total <= picture.hours ? 'output' : 'warning' });
  }
  assert.deepEqual(totals, picture.expected);
  return { body, height: 719 };
}

function greedyCapacity(picture, model, tools) {
  const { text, color, fill, line } = tools;
  const maximum = Math.max(...picture.capacities);
  const counts = [];
  let body = text(20, 27, 'Keep input order; open a row only when its capacity would overflow', { width: 320, weight: 700 });
  for (const [stage, capacity] of picture.capacities.entries()) {
    const groups = [[]];
    for (const value of picture.values) {
      assert(value > 0 && value <= capacity);
      if (groups.at(-1).reduce((sum, member) => sum + member, 0) + value > capacity) groups.push([]);
      groups.at(-1).push(value);
    }
    counts.push(groups.length);
    const vertical = 139 + stage * 320;
    body += text(20, vertical - 48, `Capacity ${capacity}`, { width: 320, weight: 700 });
    for (const [row, group] of groups.entries()) {
      let horizontal = 28;
      const top = vertical + row * 63;
      body += `<rect x="28" y="${top}" width="290" height="31" style="fill:none;stroke:${color('neutral')};stroke-width:1.5"></rect>`;
      for (const value of group) {
        const width = 290 * value / maximum;
        body += `<rect x="${horizontal}" y="${top}" width="${width}" height="31" style="fill:${fill(row % 2 ? 'state' : 'input')};stroke:${color('input')};stroke-width:1.5"></rect>` + text(horizontal + width / 2, top + 22, value, { width: width - 4, anchor: 'middle', size: 14, weight: 700 });
        horizontal += width;
      }
      body += line(28 + capacity * 290 / maximum, top - 5, 28 + capacity * 290 / maximum, top + 35, 'warning', true) + text(28, top + 55, `Sum ${group.reduce((sum, value) => sum + value, 0)}`, { width: 290, size: 14 });
    }
    body += text(20, vertical + 236, `${groups.length} groups; at most ${picture.allowed} allowed`, { width: 320, weight: 700, tone: groups.length <= picture.allowed ? 'output' : 'warning' });
  }
  assert.deepEqual(counts, picture.expected);
  return { body, height: 749 };
}

function retainedMinHeap(picture, model, tools) {
  const { text, circle, line, color } = tools;
  const sorted = [...picture.entries].sort((left, right) => right[1] - left[1]);
  const retained = sorted.slice(0, picture.count).reverse();
  const discarded = sorted.slice(picture.count);
  assert(retained.every(entry => discarded.every(other => entry[1] >= other[1])));
  let body = text(20, 27, 'The heap root is the weakest retained entry, not the weakest input', { width: 320, weight: 700 }) + text(20, 117, `Retain ${picture.count}; compare by ${picture.rootAnswer ? 'value' : 'frequency'}`, { width: 320, weight: 700 })
    + `<rect x="22" y="145" width="217" height="240" style="fill:none;stroke:${color('input')};stroke-width:2"></rect>`;
  for (const [index, [value, priority]] of retained.entries()) {
    const horizontal = index ? 70 + (index - 1) * 118 : 129;
    const vertical = index ? 310 : 192;
    if (index) body += line(129, 223, horizontal, 279, 'input');
    body += circle(horizontal, vertical, 29, index ? 'input' : 'output') + text(horizontal, vertical + 6, value, { width: 50, anchor: 'middle', weight: 700 });
    if (!picture.rootAnswer) body += text(horizontal, vertical + 55, `freq ${priority}`, { width: 100, anchor: 'middle', size: 14, weight: 700 });
  }
  for (const [index, [value, priority]] of discarded.entries()) {
    body += circle(296, 204 + index * 95, 26, 'warning') + text(296, 210 + index * 95, value, { width: 46, anchor: 'middle', weight: 700 });
    if (!picture.rootAnswer) body += text(296, 260 + index * 95, `freq ${priority}`, { width: 89, anchor: 'middle', size: 14, weight: 700 });
  }
  body += text(296, 343, 'out', { width: 89, anchor: 'middle', weight: 700, tone: 'warning' }) + text(20, 472, picture.rootAnswer ? `Root ${retained[0][0]} is the third largest; discarded two is not.` : `Keep values ${retained.map(entry => entry[0]).join(' and ')}; value nine loses on frequency.`, { width: 320, weight: 700 });
  return { body, height: 547 };
}

function tailsVersusWitness(picture, model, tools) {
  const { text, circle, line } = tools;
  const tails = [];
  const indices = [];
  for (const [index, value] of picture.values.entries()) {
    const found = tails.findIndex(tail => tail >= value);
    const position = found < 0 ? tails.length : found;
    tails[position] = value;
    indices[position] = index;
  }
  assert.deepEqual(tails, [2, 5, 7]);
  assert.deepEqual(indices, [3, 1, 2]);
  let body = text(20, 27, 'Extension bounds can point backward in source time', { width: 320, weight: 700 });
  for (const [index, value] of picture.values.entries()) {
    const horizontal = 53 + index * 84;
    if (index && index < 3) body += arrowBetween([horizontal - 58, 155], [horizontal - 27, 155], 'output', tools, 3);
    body += circle(horizontal, 155, 23, index < 3 ? 'output' : 'input') + text(horizontal, 161, value, { width: 40, anchor: 'middle', weight: 700 }) + text(horizontal, 214, `index ${index}`, { width: 77, anchor: 'middle', size: 14 });
  }
  body += text(20, 298, 'Updated tails: one bound for each achievable length', { width: 320, weight: 700 });
  for (const [position, value] of tails.entries()) {
    const horizontal = 62 + position * 118;
    body += circle(horizontal, 401, 27, 'input') + text(horizontal, 407, value, { width: 48, anchor: 'middle', weight: 700 }) + text(horizontal, 458, `index ${indices[position]}`, { width: 105, anchor: 'middle', size: 14 });
  }
  body += text(20, 549, 'Two at index three cannot come before five at index one. The actual witness is still 3,5,7.', { width: 320, weight: 700 });
  return { body, height: 645 };
}

function heapOrderGate(picture, model, tools) {
  const { text, circle, line } = tools;
  let body = text(20, 27, 'Equal sizes do not establish lower-half / upper-half order', { width: 320, weight: 700 });
  for (const [stage, [left, right]] of picture.cases.entries()) {
    const vertical = 188 + stage * 256;
    const ordered = left <= right;
    body += text(20, vertical - 73, `Both sizes one; left maximum ${left}, right minimum ${right}`, { width: 320, weight: 700 }) + line(89, vertical, 271, vertical, ordered ? 'output' : 'warning');
    for (const [position, value] of [left, right].entries()) {
      const horizontal = 89 + position * 182;
      body += `<g data-picture-heap-half="${position ? 'upper' : 'lower'}" data-picture-heap-stage="${stage}">${circle(horizontal, vertical, 23, position ? 'state' : 'input')}${text(horizontal, vertical + 6, value, { width: 40, anchor: 'middle', weight: 700 })}${text(horizontal, vertical + 68, position ? 'upper' : 'lower', { width: 79, anchor: 'middle', size: 14, weight: 700 })}</g>`;
    }
    body += text(20, vertical + 119, ordered ? 'Lower precedes upper: ordering holds.' : 'Lower lies above upper: ordering fails.', { width: 320, weight: 700, tone: ordered ? 'output' : 'warning' });
  }
  assert(picture.cases[0][0] > picture.cases[0][1]);
  assert(picture.cases[1][0] <= picture.cases[1][1]);
  return { body, height: 638 };
}

function subsetRankOrder(picture, model, tools) {
  const { text, circle } = tools;
  const ordered = [...picture.selected].sort((left, right) => picture.scores[right] - picture.scores[left]);
  const best = picture.scores.map((score, index) => index).sort((left, right) => picture.scores[right] - picture.scores[left]).slice(0, picture.selected.length);
  assert.deepEqual(ordered, best);
  assert.notDeepEqual(picture.selected, ordered);
  let body = text(20, 27, 'Correct membership still needs a separate ordering step', { width: 320, weight: 700 });
  for (const [stage, members] of [picture.selected, ordered].entries()) {
    const vertical = 169 + stage * 342;
    body += text(20, vertical - (stage ? 232 : 70), stage ? 'Sort only these selected candidates by score' : 'One permitted partition result; not score sorted', { width: 320, weight: 700 });
    for (const [position, index] of members.entries()) {
      const horizontal = 87 + position * 182;
      body += circle(horizontal, vertical, 29, stage ? 'output' : 'input') + text(horizontal, vertical + 6, index, { width: 50, anchor: 'middle', weight: 700 }) + text(horizontal, vertical + 70, `score ${picture.scores[index]}`, { width: 142, anchor: 'middle', weight: 700 });
      if (stage) body += arrowBetween([87 + picture.selected.indexOf(index) * 182, 313], [horizontal, vertical - 37], 'input', tools, 3);
    }
  }
  body += text(20, 669, 'Indices remain one and three; only their output order changes.', { width: 320, weight: 700 });
  return { body, height: 745 };
}

function frequencyBuckets(picture, model, tools) {
  const { text, circle, line, color } = tools;
  const entries = [...picture.entries].sort((left, right) => right[1] - left[1]);
  const answer = entries.slice(0, picture.count).map(entry => entry[0]);
  assert.deepEqual(answer, [1, 8]);
  let body = text(20, 27, 'Scan descending frequencies, not descending numeric values', { width: 320, weight: 700 });
  for (const [row, [value, frequency]] of entries.entries()) {
    const vertical = 171 + row * 177;
    const collected = row < picture.count;
    body += text(20, vertical - 8, `bucket ${frequency}`, { width: 120, weight: 700, tone: collected ? 'output' : 'soft' })
      + `<rect x="152" y="${vertical - 34}" width="164" height="68" style="fill:none;stroke:${color(collected ? 'output' : 'neutral')};stroke-width:2"></rect>`
      + circle(232, vertical, 25, collected ? 'output' : 'neutral') + text(232, vertical + 6, value, { width: 44, anchor: 'middle', weight: 700 });
    if (row < picture.count - 1) body += arrowBetween([81, vertical + 32], [81, vertical + 129], 'output', tools, 3);
    if (row === picture.count - 1) body += line(81, vertical + 32, 81, vertical + 118, 'warning', true) + line(65, vertical + 76, 97, vertical + 76, 'warning') + text(118, vertical + 83, 'two collected: stop', { width: 210, size: 16, weight: 700, tone: 'warning' });
  }
  body += text(20, 626, 'Emit one and eight. Nine remains in an unvisited lower-frequency bucket.', { width: 320, weight: 700 });
  return { body, height: 724 };
}

function firstUnequalCharacter(picture, model, tools) {
  const { text, circle, line } = tools;
  const words = picture.words.map(word => [...word]);
  const mismatch = words[0].findIndex((letter, index) => letter !== words[1][index]);
  assert.equal(mismatch, 2);
  let body = text(20, 27, 'Shared prefix columns supply no ordering edge', { width: 320, weight: 700 });
  for (const [row, letters] of words.entries()) for (const [column, letter] of letters.entries()) {
    const horizontal = 62 + column * 118;
    const vertical = 151 + row * 157;
    body += circle(horizontal, vertical, 27, column === mismatch ? 'input' : 'neutral') + text(horizontal, vertical + 6, letter, { width: 48, anchor: 'middle', weight: 700 });
    if (!row && column < mismatch) body += line(horizontal, vertical + 35, horizontal, vertical + 122, 'neutral', true);
  }
  body += text(20, 420, 'Only the first unequal pair becomes t before f', { width: 320, weight: 700 }) + circle(92, 527, 27, 'input') + text(92, 533, words[0][mismatch], { width: 48, anchor: 'middle', weight: 700 })
    + arrowBetween([127, 527], [233, 527], 'output', tools, 3) + circle(268, 527, 27, 'output') + text(268, 533, words[1][mismatch], { width: 48, anchor: 'middle', weight: 700 });
  return { body, height: 594 };
}

function childReturnJoin(picture, model, tools) {
  const { text, circle } = tools;
  const failed = picture.children.includes('FAIL');
  const joined = failed ? 'FAIL' : Math.max(...picture.children);
  const result = failed ? 'FAIL' : joined + 1;
  assert.equal(result, picture.expected);
  let body = text(20, 27, failed ? 'A failed child is not a usable height' : 'The parent extends only the larger child depth', { width: 320, weight: 700 });
  for (const [child, value] of picture.children.entries()) {
    const horizontal = 75 + child * 210;
    body += text(horizontal, 111, child ? 'right return' : 'left return', { width: 135, anchor: 'middle', size: 16, weight: 700 }) + circle(horizontal, 170, 29, value === 'FAIL' ? 'warning' : 'input')
      + text(horizontal, 176, value, { width: 50, anchor: 'middle', size: 14, weight: 700 }) + arrowBetween([horizontal, 207], [child ? 205 : 155, 317], failed ? 'warning' : 'input', tools, 3);
  }
  body += circle(180, 350, 30, failed ? 'warning' : 'input') + text(180, 356, failed ? 'gate' : 'max', { width: 52, anchor: 'middle', size: 16, weight: 700 })
    + arrowBetween([180, 388], [180, 523], failed ? 'warning' : 'output', tools, 3) + text(213, 460, failed ? 'propagate' : '+1 node', { width: 122, size: 16, weight: 700 })
    + circle(180, 560, 29, failed ? 'warning' : 'output') + text(180, 566, result, { width: 50, anchor: 'middle', size: 14, weight: 700 })
    + text(20, 657, failed ? 'No valid sibling height repairs FAIL.' : `One parent + max(${picture.children.join(',')}) = ${result}.`, { width: 320, weight: 700 });
  return { body, height: 735 };
}

function frozenQueueLevel(picture, model, tools) {
  const { text, circle } = tools;
  const frames = [{ label: 'Freeze two entries for this level', queue: ['A', 'B'], remaining: 2 }, { label: 'Process A; append next-level children', queue: ['B', 'A1', 'A2'], remaining: 1 }, { label: 'Process leaf B; stop this level', queue: ['A1', 'A2'], remaining: 0 }];
  let body = text(20, 27, 'New queue entries do not renew the current level budget', { width: 320, weight: 700 });
  for (const [stage, frame] of frames.entries()) {
    const vertical = 161 + stage * 230;
    body += text(20, vertical - 63, frame.label, { width: 320, weight: 700 });
    for (const [position, member] of frame.queue.entries()) {
      const horizontal = frame.queue.length === 3 ? 60 + position * 120 : 89 + position * 182;
      body += circle(horizontal, vertical, 27, member.length === 1 ? 'input' : 'output') + text(horizontal, vertical + 6, member, { width: 48, anchor: 'middle', weight: 700 });
    }
    body += text(20, vertical + 90, `${frame.remaining} current-level entries remain`, { width: 320, weight: 700 });
    assert.equal(frame.remaining + stage, 2);
  }
  assert.equal(frames[1].queue.length, 3);
  assert.equal(frames[1].remaining, 1);
  return { body, height: 782 };
}

function branchingReturn(picture, model, tools) {
  const { text, circle, line } = tools;
  const complete = picture.root + picture.gains.reduce((sum, gain) => sum + Math.max(0, gain), 0);
  const returned = picture.root + Math.max(0, ...picture.gains);
  assert.equal(complete, 20);
  assert.equal(returned, 16);
  let body = text(20, 27, 'A complete path can branch here; the returned route cannot', { width: 320, weight: 700 });
  for (let stage = 0; stage < 2; stage++) {
    const vertical = stage ? 561 : 160;
    body += circle(180, vertical, 28, 'input') + text(180, vertical + 6, picture.root, { width: 49, anchor: 'middle', weight: 700 });
    for (const [child, gain] of picture.gains.entries()) {
      const horizontal = 70 + child * 220;
      const included = !stage || gain === Math.max(...picture.gains);
      body += line(180, vertical + 33, horizontal, vertical + 77, included ? 'output' : 'neutral') + circle(horizontal, vertical + 110, 27, included ? 'output' : 'neutral')
        + text(horizontal, vertical + 116, gain, { width: 48, anchor: 'middle', weight: 700 });
    }
    if (!stage) body += text(20, 351, `Complete: ${picture.gains[0]} + ${picture.root} + ${picture.gains[1]} = ${complete}`, { width: 320, weight: 700 });
    else body += text(20, 415, 'Choose one child for the route returned to the caller', { width: 320, weight: 700 }) + arrowBetween([180, vertical - 35], [180, 491], 'output', tools, 3)
      + circle(180, 467, 17, 'output') + text(180, 472, returned, { width: 30, anchor: 'middle', size: 14, weight: 700 }) + text(212, 473, 'to parent', { width: 126, size: 16 })
      + text(20, 752, `Return ${picture.root} + max(${picture.gains.join(',')}) = ${returned}, not ${complete}.`, { width: 320, weight: 700 });
  }
  return { body, height: 831 };
}

function cloneIdentityBoundary(picture, model, tools) {
  const { text, circle, color } = tools;
  const copies = picture.labels.map(label => `${label}'`);
  assert(copies.every(copy => !picture.labels.includes(copy)));
  let body = text(20, 27, 'The copied back edge stays inside the copy enclosure', { width: 320, weight: 700 });
  for (let frame = 0; frame < 2; frame++) {
    const vertical = 186 + frame * 270;
    const labels = frame ? copies : picture.labels;
    body += text(frame ? 113 : 20, vertical - 77, frame ? 'Reuse A-prime' : 'Original cycle: A and B', { width: frame ? 135 : 320, weight: 700 })
      + `<rect x="24" y="${vertical - 47}" width="312" height="99" style="fill:none;stroke:${color(frame ? 'output' : 'input')};stroke-width:2"></rect>`;
    for (const [position, label] of labels.entries()) body += circle(71 + position * 218, vertical, 26, frame ? 'output' : 'input') + text(71 + position * 218, vertical + 6, label, { width: 46, anchor: 'middle', weight: 700 });
    body += arrowBetween([105, vertical - 12], [255, vertical - 12], frame ? 'output' : 'input', tools, 2) + arrowBetween([255, vertical + 13], [105, vertical + 13], frame ? 'output' : 'input', tools, 2);
  }
  for (const horizontal of [71, 289]) body += arrowBetween([horizontal, 248], [horizontal, 397], 'neutral', tools, 2);
  body += text(113, 328, 'copy-map lookup', { width: 135, anchor: 'start', size: 16, weight: 700 }) + text(20, 588, 'B-prime links to A-prime, never to original A.', { width: 320, weight: 700 });
  return { body, height: 669 };
}

function rootMembershipSplit(picture, model, tools) {
  const { text, circle, color, line } = tools;
  const root = picture.preorder[0];
  const split = picture.inorder.indexOf(root);
  assert.equal(split, 1);
  assert.equal(picture.inorder.length - split - 1, 3);
  let body = text(20, 27, 'Preorder names the root; inorder determines its memberships', { width: 320, weight: 700 });
  for (const [row, values] of [picture.preorder, picture.inorder].entries()) {
    const vertical = 163 + row * 221;
    body += text(row ? 143 : 20, vertical - 64, row ? 'Inorder splits this root' : 'First preorder item chooses root three', { width: row ? 195 : 320, weight: 700 });
    for (const [position, value] of values.entries()) {
      const horizontal = 48 + position * 66;
      body += circle(horizontal, vertical, 23, value === root ? 'output' : 'input') + text(horizontal, vertical + 6, value, { width: 41, anchor: 'middle', weight: 700 });
    }
  }
  const rootHorizontal = 48 + split * 66;
  body += arrowBetween([48, 193], [rootHorizontal, 353], 'output', tools, 3)
    + `<rect x="21" y="347" width="53" height="75" style="fill:none;stroke:${color('input')};stroke-width:2"></rect>`
    + `<rect x="151" y="347" width="189" height="75" style="fill:none;stroke:${color('input')};stroke-width:2"></rect>` + line(rootHorizontal, 427, rootHorizontal, 484, 'output', true)
    + text(20, 518, 'Left: one node, 9. Right: three nodes, 15,20,7.', { width: 320, weight: 700 });
  return { body, height: 593 };
}

function emptyReadyCounterexample(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'The ready queue can empty before every course has emitted', { width: 320, weight: 700 });
  for (let frame = 0; frame < 2; frame++) {
    const vertical = 168 + frame * 281;
    body += text(20, vertical - 65, frame ? 'A emitted; B and C still wait on each other' : 'Completed chain: all three emitted', { width: 320, weight: 700 });
    for (const [position, label] of ['A', 'B', 'C'].entries()) {
      const horizontal = 59 + position * 121;
      body += circle(horizontal, vertical, 24, !frame || !position ? 'output' : 'warning') + text(horizontal, vertical + 6, label, { width: 42, anchor: 'middle', weight: 700 });
    }
    if (!frame) for (let edge = 0; edge < 2; edge++) body += arrowBetween([90 + edge * 121, vertical], [149 + edge * 121, vertical], 'output', tools, 2);
    else body += arrowBetween([212, vertical - 11], [269, vertical - 11], 'warning', tools, 2) + arrowBetween([269, vertical + 12], [212, vertical + 12], 'warning', tools, 2);
    body += text(20, vertical + 85, 'Ready queue: empty', { width: 320, weight: 700 }) + text(20, vertical + 135, frame ? 'Emitted 1/3: cycle, not success' : 'Emitted 3/3: success', { width: 320, weight: 700, tone: frame ? 'warning' : 'output' });
  }
  return { body, height: 661 };
}

function componentMembership(picture, model, tools) {
  const { text, circle, line, color } = tools;
  let groups = picture.labels.map((label, index) => index);
  const merge = (start, end) => {
    const source = groups[start];
    const target = groups[end];
    if (source === target) return false;
    groups = groups.map(group => group === target ? source : group);
    return true;
  };
  for (const [start, end] of picture.edges.slice(0, -1)) assert(merge(start, end));
  const before = [...groups];
  assert.equal(merge(...picture.edges.at(-1)), false);
  assert.equal(new Set(groups).size, picture.expected);
  let body = text(20, 27, 'An already-connected pair does not remove another component', { width: 320, weight: 700 });
  for (const [frame, memberships] of [before, groups].entries()) {
    const vertical = 182 + frame * 248;
    const positions = picture.labels.map((label, index) => 57 + index * 246 / (picture.labels.length - 1));
    const finalEdge = picture.edges.at(-1).map(index => picture.labels[index]).join('-');
    body += text(20, vertical - 76, frame ? `Test ${finalEdge}: already in one group` : 'After two earlier successful unions', { width: 320, weight: 700 });
    for (const group of new Set(memberships)) {
      const members = positions.filter((horizontal, index) => memberships[index] === group);
      const left = Math.min(...members) - 28;
      const width = Math.max(...members) - Math.min(...members) + 56;
      body += `<rect x="${left}" y="${vertical - 42}" width="${width}" height="84" style="fill:none;stroke:${color('input')};stroke-width:2"></rect>`;
    }
    for (const [edge, [start, end]] of picture.edges.slice(0, picture.edges.length - 1 + frame).entries()) {
      const tone = frame && edge === picture.edges.length - 1 ? 'warning' : 'input';
      const source = positions[start];
      const target = positions[end];
      if (Math.abs(start - end) === 1) body += line(Math.min(source, target) + 25, vertical, Math.max(source, target) - 25, vertical, tone);
      else body += `<path d="M ${source} ${vertical - 16} Q ${(source + target) / 2} ${vertical - 52} ${target} ${vertical - 16}" style="fill:none;stroke:${color(tone)};stroke-width:3"></path>`;
    }
    for (const [index, label] of picture.labels.entries()) body += circle(positions[index], vertical, 22, 'input') + text(positions[index], vertical + 6, label, { width: 38, anchor: 'middle', weight: 700 });
    body += text(20, vertical + 82, `${new Set(memberships).size} components${frame ? ': count unchanged' : ''}`, { width: 320, weight: 700, tone: frame ? 'warning' : 'input' });
  }
  body += text(20, 594, picture.footer, { width: 320, weight: 700 });
  return { body, height: 686 };
}

function binaryTreeNodes(tree, top, tools, highlight = [], tone = 'output') {
  const { text, circle, line } = tools;
  const position = path => {
    let horizontal = 180;
    let offset = 90;
    for (const direction of path) { horizontal += direction === 'L' ? -offset : offset; offset /= 2; }
    return [horizontal, top + path.length * 76];
  };
  let body = '';
  for (const [path, value] of Object.entries(tree)) {
    const [horizontal, vertical] = position(path);
    if (path) {
      const parent = path.slice(0, -1);
      assert(Object.hasOwn(tree, parent) && tree[parent] !== null);
      const [parentHorizontal, parentVertical] = position(parent);
      body += line(parentHorizontal, parentVertical + 23, horizontal, vertical - 23, highlight.includes(path) ? tone : 'input');
    }
    body += circle(horizontal, vertical, value === null ? 15 : 21, highlight.includes(path) ? tone : value === null ? 'neutral' : 'input')
      + text(horizontal, vertical + 5, value === null ? '#' : value, { width: value === null ? 26 : 37, anchor: 'middle', size: 14, weight: 700 });
  }
  return body;
}

function treePreorder(tree, path = '') {
  if (!Object.hasOwn(tree, path) || tree[path] === null) return ['#'];
  return [tree[path], ...treePreorder(tree, `${path}L`), ...treePreorder(tree, `${path}R`)];
}

function treeShapeComparison(picture, model, tools) {
  const { text } = tools;
  const encodings = picture.trees.map(tree => treePreorder(tree));
  assert.notDeepEqual(encodings[0], encodings[1]);
  if (picture.invert) {
    const mirrored = Object.fromEntries(Object.entries(picture.trees[0]).map(([path, value]) => [[...path].map(direction => direction === 'L' ? 'R' : 'L').join(''), value]));
    assert.deepEqual(picture.trees[1], mirrored);
  }
  else if (!picture.subtree) assert.deepEqual(encodings[0].filter(value => value !== '#'), encodings[1].filter(value => value !== '#'));
  let body = text(20, 27, picture.invert ? 'Mirror child links, not merely stored values' : 'Child positions and missing children belong to the structure', { width: 320, weight: 700 });
  for (const [frame, tree] of picture.trees.entries()) {
    const vertical = 180 + frame * 330;
    const highlight = picture.invert ? Object.keys(tree).filter(path => [2, 4].includes(tree[path])) : picture.subtree ? ['R'] : ['L', 'R'];
    const label = picture.invert ? frame ? 'After: 2 and 4 in the right subtree' : 'Before: 2 and 4 in the left subtree' : picture.subtree ? frame ? 'Candidate has an extra right child seven' : 'Wanted subtree has no right child' : frame ? 'Tree B: two is on the right' : 'Tree A: two is on the left';
    body += text(20, vertical - 83, label, { width: 320, weight: 700 }) + binaryTreeNodes(tree, vertical, tools, highlight, picture.invert ? 'output' : 'warning');
    body += text(20, vertical + 214, picture.serialized ? `Encode: ${encodings[frame].join(', ')}` : picture.invert ? 'Node values unchanged.' : picture.subtree ? 'Missing children count too.' : 'Value preorder: 1,2.', { width: 320, size: 16, weight: 700 });
  }
  return { body, height: 795 };
}

function inheritedAncestorBound(picture, model, tools) {
  const { text, circle, line, color, fill } = tools;
  assert(picture.value > picture.lower && picture.value >= picture.upper);
  const horizontal = value => 32 + (value - picture.lower) * 296 / (picture.value - picture.lower);
  let body = text(20, 27, 'The descendant inherits every ancestor boundary', { width: 320, weight: 700 }) + binaryTreeNodes(picture.tree, 145, tools, ['LR'], 'warning')
    + `<rect x="42" y="185" width="123" height="146" style="fill:none;stroke:${color('warning')};stroke-width:2"></rect>` + text(20, 388, 'Everything in ancestor ten\'s left subtree must remain below ten.', { width: 320, weight: 700 })
    + `<rect x="32" y="493" width="${horizontal(picture.upper) - 32}" height="30" style="fill:${fill('output')}"></rect>` + line(32, 508, 328, 508, 'neutral') + line(horizontal(picture.upper), 476, horizontal(picture.upper), 533, 'warning', true);
  for (const value of [picture.lower, picture.upper, picture.value]) body += circle(horizontal(value), 508, 7, value === picture.value ? 'warning' : 'neutral') + text(horizontal(value), 553, value, { width: 44, anchor: 'middle', weight: 700 });
  body += text(20, 622, `Required ${picture.lower} < node < ${picture.upper}; ${picture.value} lies outside.`, { width: 320, weight: 700, tone: 'warning' });
  return { body, height: 697 };
}

function bstStopBoundary(picture, model, tools) {
  const { text } = tools;
  let body = text(20, 27, 'Stop at a split or at a target that equals this node', { width: 320, weight: 700 });
  for (const [frame, targets] of picture.targets.entries()) {
    const vertical = 169 + frame * 285;
    let path = '';
    while (targets.every(target => target < picture.tree[path]) || targets.every(target => target > picture.tree[path])) {
      path += targets.every(target => target < picture.tree[path]) ? 'L' : 'R';
      assert(Object.hasOwn(picture.tree, path));
    }
    assert.equal(picture.tree[path], 6);
    body += text(20, vertical - 75, frame ? 'Targets six and eight: six already occupies the root' : 'Targets two and eight: their child directions split', { width: 320, weight: 700 })
      + binaryTreeNodes(picture.tree, vertical, tools, [path]) + text(20, vertical + 167, 'Stop at six in both cases.', { width: 320, weight: 700 });
  }
  return { body, height: 697 };
}

function inorderVersusPush(picture, model, tools) {
  const { text, circle } = tools;
  const visit = path => !Object.hasOwn(picture.tree, path) || picture.tree[path] === null ? [] : [...visit(`${path}L`), picture.tree[path], ...visit(`${path}R`)];
  const order = visit('');
  assert.deepEqual(order, [1, 2, 3]);
  assert.notEqual(picture.stack[0], order[picture.rank - 1]);
  let body = text(20, 27, 'Pushing discovers a node; inorder visiting emits its rank', { width: 320, weight: 700 }) + text(20, 112, 'Stack push order: two, then its left child one', { width: 320, weight: 700 });
  for (const [position, value] of picture.stack.entries()) body += circle(89 + position * 182, 185, 26, 'input') + text(89 + position * 182, 191, value, { width: 46, anchor: 'middle', weight: 700 });
  body += binaryTreeNodes(picture.tree, 326, tools) + text(20, 486, 'Actual inorder emissions: one, two, three', { width: 320, weight: 700 });
  for (const [position, value] of order.entries()) body += circle(62 + position * 118, 571, 26, position === picture.rank - 1 ? 'output' : 'input') + text(62 + position * 118, 577, value, { width: 46, anchor: 'middle', weight: 700 });
  body += text(20, 671, `Rank ${picture.rank} is visited value ${order[picture.rank - 1]}, not the first pushed value ${picture.stack[0]}.`, { width: 320, weight: 700 });
  return { body, height: 756 };
}

function pointerIdentityMeeting(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'Equal values do not merge two distinct objects', { width: 320, weight: 700 });
  for (const [position, label] of ['A', 'B'].entries()) {
    const horizontal = 90 + position * 180;
    body += circle(horizontal, 164, 27, 'input') + text(horizontal, 170, picture.value, { width: 48, anchor: 'middle', weight: 700 })
      + text(horizontal, 227, `node ${label}`, { width: 110, anchor: 'middle', weight: 700 }) + text(horizontal, 113, position ? 'fast' : 'slow', { width: 110, anchor: 'middle', size: 16 });
  }
  body += text(20, 315, 'After valid moves, both pointers can reach one shared C', { width: 320, weight: 700 }) + circle(180, 448, 27, 'output') + text(180, 454, picture.value, { width: 48, anchor: 'middle', weight: 700 })
    + arrowBetween([58, 448], [145, 448], 'output', tools, 3) + arrowBetween([302, 448], [215, 448], 'output', tools, 3)
    + text(58, 407, 'slow', { width: 86, anchor: 'middle', size: 16 }) + text(302, 407, 'fast', { width: 86, anchor: 'middle', size: 16 }) + text(180, 516, 'one node C', { width: 150, anchor: 'middle', weight: 700 });
  return { body, height: 587 };
}

function linkedChainFrames(picture, model, tools) {
  const { text, circle, color } = tools;
  let body = text(20, 27, picture.title, { width: 320, weight: 700 });
  for (const [row, { label, chain }] of picture.rows.entries()) {
    const vertical = 169 + row * 200;
    assert.equal(chain.at(-1), 'null');
    assert.equal(new Set(chain).size, chain.length);
    body += text(20, vertical - 73, label, { width: 320, weight: 700 });
    const radius = chain.length >= 5 ? 17 : 22;
    for (const [position, value] of chain.entries()) {
      const horizontal = chain.length === 1 ? 180 : 44 + position * 272 / (chain.length - 1);
      if (position) {
        const previous = 44 + (position - 1) * 272 / (chain.length - 1);
        body += arrowBetween([previous + radius + 4, vertical], [horizontal - radius - 4, vertical], row ? 'output' : 'input', tools, 2);
      }
      body += value === 'null' ? `<rect x="${horizontal - 19}" y="${vertical - 16}" width="38" height="32" style="fill:none;stroke:${color('neutral')};stroke-width:1.5"></rect>` : circle(horizontal, vertical, radius, row ? 'output' : 'input');
      body += text(horizontal, vertical + 5, value, { width: value === 'null' ? 36 : Math.max(36, radius * 2 - 4), anchor: 'middle', size: value === 'null' ? 12 : 14, weight: 700 });
    }
  }
  return { body, height: 169 + (picture.rows.length - 1) * 200 + 87 };
}

function minimumHistoryPop(picture, model, tools) {
  const { text, color, fill } = tools;
  const minima = picture.values.map((value, index) => Math.min(...picture.values.slice(0, index + 1)));
  assert.deepEqual(minima, [5, 2, 2]);
  assert.equal(minima.at(-2), 2);
  let body = text(20, 27, 'Each value owns its own minimum-so-far snapshot', { width: 320, weight: 700 });
  for (let frame = 0; frame < 2; frame++) {
    const horizontal = 28 + frame * 192;
    body += text(horizontal, 114, frame ? 'After one pop' : 'Before pop', { width: 112, size: 16, weight: 700 }) + text(horizontal, 153, 'value | min', { width: 112, size: 14 });
    for (let index = 0; index < picture.values.length - frame; index++) {
      const vertical = 339 - index * 66;
      body += `<rect x="${horizontal}" y="${vertical}" width="112" height="52" style="fill:${fill('input')};stroke:${color('input')};stroke-width:2"></rect>`
        + text(horizontal + 28, vertical + 33, picture.values[index], { width: 47, anchor: 'middle', weight: 700 }) + text(horizontal + 84, vertical + 33, minima[index], { width: 47, anchor: 'middle', weight: 700 });
    }
  }
  body += text(20, 490, 'The surviving top pair is still value two / minimum two.', { width: 320, weight: 700 });
  return { body, height: 568 };
}

function histogramSentinelRetained(picture, model, tools) {
  const { text, circle, color, fill, line } = tools;
  const stack = [];
  let best = 0;
  for (const [index, height] of [...picture.heights, 0].entries()) {
    let start = index;
    while (stack.length && stack.at(-1)[1] > height) {
      const [earlier, prior] = stack.pop();
      best = Math.max(best, prior * (index - earlier));
      start = earlier;
    }
    stack.push([start, height]);
  }
  assert.equal(best, 3);
  assert.equal(stack.length, 1);
  assert.equal(stack[0][1], 0);
  let body = text(20, 27, 'The final zero resolves positive spans but remains on the stack', { width: 320, weight: 700 });
  for (const [index, height] of picture.heights.entries()) body += `<rect x="${32 + index * 70}" y="${247 - height * 43}" width="70" height="${height * 43}" style="fill:${fill('input')};stroke:${color('input')};stroke-width:2"></rect>` + text(67 + index * 70, 289, height, { width: 60, anchor: 'middle', weight: 700 });
  body += `<rect x="32" y="204" width="210" height="43" style="fill:${fill('output')};stroke:${color('output')};stroke-width:2"></rect>` + line(242, 247, 312, 247, 'warning', true)
    + text(277, 289, '0', { width: 60, anchor: 'middle', weight: 700, tone: 'warning' }) + text(20, 368, 'Best area three; no unresolved positive heights.', { width: 320, weight: 700 })
    + circle(180, 477, 29, 'output') + text(180, 483, stack[0][1], { width: 50, anchor: 'middle', weight: 700 }) + text(20, 574, 'Actual stack: one retained zero-sentinel entry.', { width: 320, weight: 700 });
  return { body, height: 653 };
}

function recencyFrontier(picture, model, tools) {
  const { text, circle, color, line } = tools;
  assert.deepEqual(picture.states[1], [picture.states[0][1], picture.states[0][0]]);
  assert.deepEqual(picture.states[2], [picture.states[0][0], 3]);
  let body = text(20, 27, 'A read changes age, so the next insertion changes its victim', { width: 320, weight: 700 });
  for (const [stage, keys] of picture.states.entries()) {
    const vertical = 161 + stage * 230;
    body += text(20, vertical - 67, ['Before: one is oldest', 'get(1): two becomes oldest', 'Insert three: evict the now-oldest two'][stage], { width: 320, weight: 700 })
      + `<rect x="42" y="${vertical - 37}" width="204" height="74" style="fill:none;stroke:${color('input')};stroke-width:2"></rect>`;
    for (const [position, key] of keys.entries()) body += circle(82 + position * 122, vertical, 24, 'input') + text(82 + position * 122, vertical + 6, key, { width: 42, anchor: 'middle', weight: 700 });
    body += text(42, vertical + 83, 'oldest', { width: 99, size: 14 }) + text(246, vertical + 83, 'newest', { width: 99, anchor: 'end', size: 14 });
    if (stage === 2) body += circle(306, vertical, 23, 'warning') + text(306, vertical + 6, 2, { width: 40, anchor: 'middle', weight: 700 }) + line(291, vertical - 16, 321, vertical + 16, 'warning') + text(306, vertical + 83, 'out', { width: 77, anchor: 'middle', size: 14 });
  }
  return { body, height: 781 };
}

function coinPredecessor(picture, model, tools) {
  const { text, circle } = tools;
  const sentinel = picture.amount + 1;
  const candidates = [];
  let body = text(20, 27, 'A candidate needs a reachable smaller-amount predecessor', { width: 320, weight: 700 });
  for (const [row, coin] of picture.coins.entries()) {
    const predecessor = picture.amount - coin;
    const cost = picture.costs[predecessor];
    const valid = cost < sentinel;
    if (valid) candidates.push(cost + 1);
    const vertical = 177 + row * 260;
    body += text(20, vertical - 65, `Use coin ${coin}; subtract it from amount ${picture.amount}`, { width: 320, weight: 700 }) + circle(62, vertical, 28, 'input') + text(62, vertical + 6, picture.amount, { width: 49, anchor: 'middle', weight: 700 })
      + arrowBetween([99, vertical], [261, vertical], valid ? 'output' : 'warning', tools, 3) + circle(298, vertical, 28, valid ? 'output' : 'warning') + text(298, vertical + 6, predecessor, { width: 49, anchor: 'middle', weight: 700 })
      + text(20, vertical + 81, valid ? `Base cost ${cost} + one coin = ${cost + 1}: keep` : `Unreachable sentinel ${cost} + one = ${cost + 1}: reject`, { width: 320, weight: 700, tone: valid ? 'output' : 'warning' });
  }
  assert.equal(Math.min(...candidates), 1);
  body += text(20, 642, 'The valid amount-zero route supplies the one-coin answer.', { width: 320, weight: 700 });
  return { body, height: 723 };
}

function decimalPairGate(picture, model, tools) {
  const { text, circle } = tools;
  let body = text(20, 27, 'Keep digit order intact when checking the range ten through twenty-six', { width: 320, weight: 700 });
  const accepted = [];
  for (const [row, pair] of picture.pairs.entries()) {
    const value = Number(pair);
    const valid = pair.length === 2 && pair[0] !== '0' && value >= 10 && value <= 26;
    accepted.push(valid);
    const vertical = 174 + row * 193;
    for (const [position, digit] of [...pair].entries()) body += circle(54 + position * 75, vertical, 24, 'input') + text(54 + position * 75, vertical + 6, digit, { width: 42, anchor: 'middle', weight: 700 });
    body += arrowBetween([166, vertical], [260, vertical], valid ? 'output' : 'warning', tools, 3) + circle(298, vertical, 27, valid ? 'output' : 'warning') + text(298, vertical + 6, valid ? '+' : 'x', { width: 48, anchor: 'middle', weight: 700 })
      + text(20, vertical + 88, valid ? 'Ten is an allowed two-digit code.' : pair[0] === '0' ? 'Leading zero: zero-one is not ten.' : 'Twenty-seven exceeds the allowed range.', { width: 320, weight: 700, tone: valid ? 'output' : 'warning' });
  }
  assert.deepEqual(accepted, [true, false, false]);
  return { body, height: 726 };
}

function editDiagonalCost(picture, model, tools) {
  const { text, circle } = tools;
  const results = [];
  let body = text(20, 27, 'The mismatch toll is one; the matching diagonal toll is zero', { width: 320, weight: 700 });
  for (const [row, [left, right]] of picture.pairs.entries()) {
    const cost = left === right ? 0 : 1;
    results.push(cost);
    const vertical = 174 + row * 259;
    body += text(20, vertical - 66, `Compare ${left} with ${right}; empty-prefix diagonal is zero`, { width: 320, weight: 700 }) + circle(62, vertical, 28, 'input') + text(62, vertical + 6, 0, { width: 49, anchor: 'middle', weight: 700 })
      + arrowBetween([99, vertical], [261, vertical], cost ? 'warning' : 'output', tools, 3) + text(180, vertical - 25, `+${cost}`, { width: 70, anchor: 'middle', weight: 700 })
      + circle(298, vertical, 28, cost ? 'warning' : 'output') + text(298, vertical + 6, cost, { width: 49, anchor: 'middle', weight: 700 })
      + text(20, vertical + 98, cost ? 'Replace a by b: one edit.' : 'Match a with a: inherit the diagonal unchanged.', { width: 320, weight: 700 });
  }
  assert.deepEqual(results, [1, 0]);
  return { body, height: 612 };
}

function circularEndpointCases(picture, model, tools) {
  const { text, circle, line } = tools;
  const positions = [[81, 182], [279, 182], [180, 311]];
  let body = text(20, 27, 'The circular edge forbids taking both linear endpoints', { width: 320, weight: 700 });
  for (const [index, value] of picture.values.entries()) {
    const [horizontal, vertical] = positions[index];
    const next = positions[(index + 1) % positions.length];
    body += line(horizontal, vertical, next[0], next[1], index === positions.length - 1 ? 'warning' : 'input');
  }
  for (const [index, value] of picture.values.entries()) body += circle(...positions[index], 24, index === 1 ? 'input' : 'warning') + text(positions[index][0], positions[index][1] + 6, value, { width: 42, anchor: 'middle', weight: 700 });
  const optima = [];
  for (let frame = 0; frame < 2; frame++) {
    const excluded = frame ? 0 : picture.values.length - 1;
    const vertical = 511 + frame * 210;
    const best = Math.max(...picture.values.filter((value, index) => index !== excluded));
    optima.push(best);
    body += text(20, vertical - 66, frame ? 'Case B: exclude the first endpoint' : 'Case A: exclude the last endpoint', { width: 320, weight: 700 });
    for (const [index, value] of picture.values.entries()) {
      const horizontal = 62 + index * 118;
      body += circle(horizontal, vertical, 26, index === excluded ? 'neutral' : 'input') + text(horizontal, vertical + 6, value, { width: 46, anchor: 'middle', weight: 700 });
      if (index === excluded) body += line(horizontal - 17, vertical - 17, horizontal + 17, vertical + 17, 'warning');
    }
    body += text(20, vertical + 85, `Linear case optimum ${best}`, { width: 320, weight: 700 });
  }
  assert.deepEqual(optima, [3, 3]);
  body += text(20, 871, 'Take max(3,3) = 3, not sum six.', { width: 320, weight: 700 });
  return { body, height: 947 };
}

function sharedIncreasingSuffix(picture, model, tools) {
  const { text, circle } = tools;
  const position = ([row, column]) => [62 + column * 118, 170 + row * 126];
  let body = text(20, 27, 'Two starts enter the same adjacent increasing suffix', { width: 320, weight: 700 });
  for (const start of picture.starts) {
    const path = [start, ...picture.suffix];
    for (let step = 1; step < path.length; step++) {
      const earlier = path[step - 1];
      const current = path[step];
      assert.equal(Math.abs(earlier[0] - current[0]) + Math.abs(earlier[1] - current[1]), 1);
      assert(picture.grid[earlier[0]][earlier[1]] < picture.grid[current[0]][current[1]]);
      const source = position(earlier);
      const target = position(current);
      const gap = 33;
      const length = Math.hypot(target[0] - source[0], target[1] - source[1]);
      body += arrowBetween([source[0] + (target[0] - source[0]) * gap / length, source[1] + (target[1] - source[1]) * gap / length], [target[0] - (target[0] - source[0]) * gap / length, target[1] - (target[1] - source[1]) * gap / length], step === 2 ? 'output' : 'input', tools, 3);
    }
  }
  for (const [row, values] of picture.grid.entries()) for (const [column, value] of values.entries()) {
    const [horizontal, vertical] = position([row, column]);
    body += circle(horizontal, vertical, 25, picture.suffix.some(cell => cell[0] === row && cell[1] === column) ? 'output' : value ? 'input' : 'neutral') + text(horizontal, vertical + 6, value, { width: 44, anchor: 'middle', weight: 700 });
  }
  body += text(20, 427, 'Cache at cell three: suffix length two for route 3,4.', { width: 320, weight: 700 }) + text(20, 537, 'Reuse that cell-owned answer from either start. The two copies of the route are not separate subproblems.', { width: 320, weight: 700 });
  return { body, height: 651 };
}

function palindromeOccurrenceSpans(picture, model, tools) {
  const { text, color, fill } = tools;
  const spans = [];
  for (let start = 0; start < picture.string.length; start++) for (let end = start + 1; end <= picture.string.length; end++) {
    const value = picture.string.slice(start, end);
    if (value === [...value].reverse().join('')) spans.push([start, end, value]);
  }
  spans.sort((left, right) => left[1] - left[0] - (right[1] - right[0]) || left[0] - right[0]);
  assert.equal(spans.length, 6);
  assert.equal(new Set(spans.map(span => span[2])).size, 3);
  let body = text(20, 27, 'The same text at a different interval is another occurrence', { width: 320, weight: 700 });
  for (let position = 0; position < picture.string.length; position++) body += text(45 + position * 95, 128, `${picture.string[position]} at ${position}`, { width: 88, size: 14, weight: 700 });
  for (const [row, [start, end]] of spans.entries()) {
    const vertical = 183 + row * 67;
    body += `<rect x="${36 + start * 95}" y="${vertical}" width="${(end - start) * 95}" height="28" style="fill:${fill(end - start === 1 ? 'input' : end - start === 2 ? 'state' : 'output')};stroke:${color('input')};stroke-width:1.5"></rect>`
      + text(36, vertical + 52, `[${start},${end})`, { width: 285, size: 14 });
  }
  body += text(20, 659, 'Six start-end intervals; only three distinct palindrome texts.', { width: 320, weight: 700 });
  return { body, height: 741 };
}

function unusedPositionAvailability(picture, model, tools) {
  const { text, circle, line } = tools;
  const remaining = picture.values.map((value, index) => index).filter(index => !picture.used.includes(index));
  assert.deepEqual(remaining, [1]);
  let body = text(20, 27, 'Used positions are blocked; unused earlier positions are not', { width: 320, weight: 700 });
  for (const [index, value] of picture.values.entries()) {
    const horizontal = 62 + index * 118;
    body += circle(horizontal, 169, 26, picture.used.includes(index) ? 'neutral' : 'output') + text(horizontal, 175, value, { width: 46, anchor: 'middle', weight: 700 }) + text(horizontal, 231, `index ${index}`, { width: 104, anchor: 'middle', size: 14 });
    if (picture.used.includes(index)) body += line(horizontal - 17, 152, horizontal + 17, 186, 'warning');
  }
  body += text(20, 325, 'Partial permutation chose values one, then three', { width: 320, weight: 700 });
  for (const [position, index] of picture.used.entries()) body += circle(89 + position * 182, 413, 26, 'input') + text(89 + position * 182, 419, picture.values[index], { width: 46, anchor: 'middle', weight: 700 });
  body += arrowBetween([271, 447], [203, 533], 'output', tools, 3) + circle(180, 560, 26, 'output') + text(180, 566, picture.values[remaining[0]], { width: 46, anchor: 'middle', weight: 700 })
    + text(20, 650, 'Next value two is legal: its source position is still unused.', { width: 320, weight: 700 });
  return { body, height: 737 };
}

function binaryChoiceLeaves(picture, model, tools) {
  const { text, circle } = tools;
  assert.equal(picture.values.length, 2);
  let body = text(20, 27, 'Every include-or-exclude decision path emits one membership', { width: 320, weight: 700 });
  for (const [position, label] of ['take 1?', 'take 2?', 'result'].entries()) body += text([60, 170, 295][position], 111, label, { width: position === 2 ? 82 : 99, anchor: 'middle', size: 14 });
  const memberships = [];
  for (let path = 0; path < 4; path++) {
    const choices = [!!(path & 2), !!(path & 1)];
    const subset = picture.values.filter((value, index) => choices[index]);
    memberships.push(subset.join(','));
    const vertical = 177 + path * 127;
    for (const [index, included] of choices.entries()) body += circle(60 + index * 110, vertical, 25, included ? 'output' : 'neutral') + text(60 + index * 110, vertical + 6, included ? '+' : '-', { width: 44, anchor: 'middle', weight: 700 });
    body += arrowBetween([204, vertical], [254, vertical], 'input', tools, 2) + text(295, vertical + 6, `{${subset.join(',')}}`, { width: 82, anchor: 'middle', size: 16, weight: 700 });
  }
  assert.equal(new Set(memberships).size, 4);
  body += text(20, 668, 'Two explicit exclusions emit the empty subset.', { width: 320, weight: 700 });
  return { body, height: 747 };
}

function gridFinalMove(picture, model, tools) {
  const { text, circle, line } = tools;
  const paths = [];
  const visit = path => {
    const [row, column] = path.at(-1);
    if (row === picture.rows - 1 && column === picture.columns - 1) paths.push(path);
    else { if (row + 1 < picture.rows) visit([...path, [row + 1, column]]); if (column + 1 < picture.columns) visit([...path, [row, column + 1]]); }
  };
  visit([[0, 0]]);
  assert.equal(paths.length, 6);
  assert.equal(paths.filter(path => path.at(-2)[0] === picture.rows - 2).length, 3);
  let body = text(20, 27, 'Each route has exactly one final incoming edge', { width: 320, weight: 700 });
  for (const [index, path] of paths.entries()) {
    const left = 39 + index % 2 * 183;
    const top = 164 + Math.floor(index / 2) * 177;
    for (let row = 0; row < picture.rows; row++) for (let column = 0; column < picture.columns; column++) body += circle(left + column * 37, top + row * 37, 3, 'neutral');
    for (let step = 1; step < path.length; step++) {
      const source = path[step - 1];
      const target = path[step];
      body += line(left + source[1] * 37, top + source[0] * 37, left + target[1] * 37, top + target[0] * 37, step === path.length - 1 ? 'output' : 'input');
    }
    body += circle(left, top, 5, 'input') + circle(left + (picture.columns - 1) * 37, top + (picture.rows - 1) * 37, 5, 'output');
    body += text(left - 12, top + 117, path.at(-2)[0] === picture.rows - 2 ? 'last: from above' : 'last: from left', { width: 132, size: 14, weight: 700 });
  }
  body += text(20, 734, 'Three enter from above; three from the left. The groups sum to six.', { width: 320, weight: 700 });
  return { body, height: 828 };
}

function wordBoundaryAlternatives(picture, model, tools) {
  const { text, color, fill, line } = tools;
  const validCases = [];
  let body = text(20, 27, 'A longer reachable prefix can leave an impossible suffix', { width: 320, weight: 700 }) + text(20, 112, `Dictionary: ${picture.words.join(', ')}`, { width: 320, size: 16, weight: 700 });
  for (const [row, split] of picture.splits.entries()) {
    const prefix = picture.string.slice(0, split);
    const suffix = picture.string.slice(split);
    const valid = picture.words.includes(prefix) && picture.words.includes(suffix);
    validCases.push(valid);
    const vertical = 210 + row * 223;
    body += text(20, vertical - 54, valid ? `${prefix} then ${suffix}: both words exist` : `${prefix} then ${suffix}: suffix absent`, { width: 320, weight: 700 });
    for (const [position, letter] of [...picture.string].entries()) {
      const horizontal = 40 + position * 70;
      const tone = position < split ? 'input' : valid ? 'output' : 'warning';
      body += `<rect x="${horizontal}" y="${vertical}" width="70" height="46" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>` + text(horizontal + 35, vertical + 30, letter, { width: 60, anchor: 'middle', weight: 700 });
    }
    body += line(40 + split * 70, vertical - 8, 40 + split * 70, vertical + 67, valid ? 'output' : 'warning', true);
  }
  assert.deepEqual(validCases, [false, true]);
  body += text(20, 612, 'Reachable boundaries, not the longest greedy word, establish the full split.', { width: 320, weight: 700 });
  return { body, height: 706 };
}

function savedWitnessTrace(picture, model, tools) {
  const { text, circle, color, line, number } = tools;
  let current;
  if (picture.mode === 'profit') {
    let minimum = Infinity;
    current = picture.values.map(value => { minimum = Math.min(minimum, value); return value - minimum; });
  }
  else if (picture.mode === 'sum') {
    let suffix = 0;
    current = picture.values.map(value => { suffix = Math.max(value, suffix + value); return suffix; });
  }
  else if (picture.mode === 'palindrome') current = picture.values.map(word => { assert.equal(word, [...word].reverse().join('')); return word.length; });
  else { assert.equal(picture.mode, 'rooms'); current = picture.values.map(time => picture.intervals.filter(([start, end]) => start <= time && time < end).length); }
  let best = -Infinity;
  const saved = current.map(value => { best = Math.max(best, value); return best; });
  assert.deepEqual([current, saved], picture.expected);
  const minimum = Math.min(0, ...current);
  const maximum = Math.max(...saved);
  const horizontal = index => 70 + index * 246 / (current.length - 1);
  const vertical = value => 436 - (value - minimum) * 240 / (maximum - minimum);
  let body = text(20, 27, picture.title, { width: 320, weight: 700 }) + text(20, 114, picture.mode === 'rooms' ? 'Three ordered occupancy checkpoints' : `Input order: ${picture.values.join(', ')}`, { width: 320, size: 16, weight: 700 })
    + text(20, 164, 'Blue: current; green: saved best', { width: 320, size: 16, weight: 700 });
  for (const [series, tone] of [[saved, 'output'], [current, 'input']]) {
    const commands = series.map((value, index) => index ? `H ${horizontal(index)} V ${vertical(value)}` : `M ${horizontal(index)} ${vertical(value)}`).join(' ');
    body += `<path d="${commands}" style="fill:none;stroke:${color(tone)};stroke-width:3${tone === 'input' ? ';stroke-dasharray:5 3' : ''}"></path>`;
  }
  for (const [index, value] of current.entries()) body += circle(horizontal(index), vertical(value), 5, 'input') + `<circle cx="${horizontal(index)}" cy="${vertical(saved[index])}" r="9" style="fill:none;stroke:${color('output')};stroke-width:2"></circle>` + text(horizontal(index), 490, index + 1, { width: 48, anchor: 'middle', size: 14 });
  body += line(60, 436, 330, 436, 'neutral') + text(44, 202, number(maximum), { width: 38, anchor: 'end', size: 14 }) + text(44, 442, number(minimum), { width: 38, anchor: 'end', size: 14 })
    + text(20, 539, 'Checkpoint number', { width: 320, size: 16 }) + text(20, 606, `Current: ${current.join(', ')}`, { width: 320, weight: 700, tone: 'input' }) + text(20, 674, `Saved: ${saved.join(', ')}`, { width: 320, weight: 700, tone: 'output' });
  return { body, height: 750 };
}

function replacementWindowWitness(picture, model, tools) {
  const { text, circle, line } = tools;
  const demands = [];
  let body = text(20, 27, 'A later invalid window does not erase the earlier valid witness', { width: 320, weight: 700 });
  for (const [row, window] of picture.windows.entries()) {
    const counts = new Map();
    for (const character of window) counts.set(character, (counts.get(character) ?? 0) + 1);
    const majority = [...counts].sort((left, right) => right[1] - left[1])[0][0];
    const needed = window.length - counts.get(majority);
    demands.push(needed);
    const vertical = 170 + row * 255;
    body += text(20, vertical - 64, row ? 'Later window: ABBA' : 'Saved witness: AABA', { width: 320, weight: 700 });
    for (const [position, character] of [...window].entries()) {
      const horizontal = 53 + position * 84;
      const change = character !== majority;
      body += circle(horizontal, vertical, 24, change ? 'warning' : 'input') + text(horizontal, vertical + 6, character, { width: 42, anchor: 'middle', weight: 700 });
      if (change) body += line(horizontal - 15, vertical - 15, horizontal + 15, vertical + 15, 'warning');
    }
    body += text(20, vertical + 97, `${needed} edits needed; budget ${picture.budget}: ${needed <= picture.budget ? 'valid' : 'invalid'}`, { width: 320, weight: 700, tone: needed <= picture.budget ? 'output' : 'warning' });
  }
  assert.deepEqual(demands, [1, 2]);
  body += text(20, 635, 'Saved length four still has the valid AABA witness.', { width: 320, weight: 700 });
  return { body, height: 721 };
}

function negativeExtremaFlip(picture, model, tools) {
  const { text, circle } = tools;
  const products = picture.previous.map(value => value * picture.multiplier);
  assert.deepEqual(products, [-12, 24]);
  const result = Math.max(picture.multiplier, ...products);
  assert.equal(result, 24);
  let body = text(20, 27, `The same negative multiplier ${picture.multiplier} reverses extrema order`, { width: 320, weight: 700 });
  for (const [index, value] of picture.previous.entries()) {
    const source = index ? 89 : 271;
    const destination = index ? 271 : 89;
    body += text(source, 117, index ? 'previous min' : 'previous max', { width: 150, anchor: 'middle', size: 16, weight: 700 }) + circle(source, 178, 28, 'input') + text(source, 184, value, { width: 49, anchor: 'middle', weight: 700 })
      + arrowBetween([source, 215], [destination, 353], index ? 'output' : 'input', tools, 3) + circle(destination, 390, 28, index ? 'output' : 'neutral') + text(destination, 396, products[index], { width: 49, anchor: 'middle', weight: 700 })
      + text(destination, 453, index ? 'new max' : 'new min', { width: 150, anchor: 'middle', size: 16, weight: 700 });
  }
  body += text(20, 502, 'All candidates: standalone -4, old-max product -12, old-min product 24.', { width: 320, weight: 700 }) + text(20, 633, 'Keep both prior extrema; the old minimum supplies the winner.', { width: 320, weight: 700 });
  return { body, height: 724 };
}

function nondecreasingWindowBoundary(picture, model, tools) {
  const { text, circle, line } = tools;
  const candidate = picture.previousOccurrence + 1;
  const retained = Math.max(picture.previousLeft, candidate);
  assert.equal(candidate, 1);
  assert.equal(retained, 2);
  assert.equal(new Set(picture.string.slice(retained)).size, picture.string.slice(retained).length);
  let body = text(20, 27, 'An old occurrence cannot reopen a discarded prefix', { width: 320, weight: 700 });
  for (const [row, left] of [candidate, retained].entries()) {
    const vertical = 182 + row * 260;
    body += text(20, vertical - 68, row ? 'Keep left = max(2,1) = 2' : 'Wrong left = old-a index + one = 1', { width: 320, weight: 700 });
    for (const [index, letter] of [...picture.string].entries()) {
      const horizontal = 53 + index * 84;
      body += circle(horizontal, vertical, 23, index < left ? 'neutral' : row ? 'output' : 'warning') + text(horizontal, vertical + 6, letter, { width: 40, anchor: 'middle', weight: 700 }) + text(horizontal, vertical + 63, index, { width: 40, anchor: 'middle', size: 14 });
    }
    body += line(53 + left * 84 - 31, vertical - 43, 53 + left * 84 - 31, vertical + 85, row ? 'output' : 'warning', true);
  }
  body += text(20, 631, 'Left one reopens both b copies. Left two keeps the unique suffix ba.', { width: 320, weight: 700 });
  return { body, height: 724 };
}

function reachableJumpFrontier(picture, model, tools) {
  const { text, circle, color, line } = tools;
  let frontier = 0;
  const visited = [];
  for (const [index, value] of picture.values.entries()) {
    if (index > frontier) break;
    visited.push(index);
    frontier = Math.max(frontier, index + value);
  }
  assert.equal(frontier, 1);
  assert.deepEqual(visited, [0, 1]);
  let body = text(20, 27, 'Only a reachable position may extend the reachable frontier', { width: 320, weight: 700 }) + `<rect x="25" y="135" width="205" height="83" style="fill:none;stroke:${color('output')};stroke-width:2"></rect>`;
  for (const [index, value] of picture.values.entries()) {
    const horizontal = 62 + index * 118;
    body += text(horizontal, 113, `index ${index}`, { width: 103, anchor: 'middle', size: 14 }) + circle(horizontal, 177, 26, visited.includes(index) ? 'output' : 'warning') + text(horizontal, 183, value, { width: 46, anchor: 'middle', weight: 700 });
  }
  body += arrowBetween([96, 177], [146, 177], 'output', tools, 3) + line(242, 120, 242, 291, 'warning', true) + text(20, 332, 'Index one is reachable but has jump zero. Index two remains outside.', { width: 320, weight: 700 })
    + text(20, 470, 'The large nine is never read as a valid frontier contribution.', { width: 320, weight: 700, tone: 'warning' });
  return { body, height: 569 };
}

function runStartMembership(picture, model, tools) {
  const { text, circle, line } = tools;
  const values = new Set(picture.values);
  const starts = picture.values.filter(value => !values.has(value - 1));
  assert.deepEqual(starts, [1, 8]);
  const horizontal = value => 40 + (value - 1) * 40;
  let body = text(20, 27, 'A value without a predecessor starts a whole maximal run', { width: 320, weight: 700 });
  for (const value of picture.values) {
    if (values.has(value + 1)) body += line(horizontal(value) + 18, 194, horizontal(value + 1) - 18, 194, 'input');
    body += circle(horizontal(value), 194, 17, starts.includes(value) ? 'output' : 'input') + text(horizontal(value), 200, value, { width: 29, anchor: 'middle', size: 14, weight: 700 });
    if (starts.includes(value)) body += text(horizontal(value), 264, 'start', { width: 64, anchor: 'middle', size: 14, weight: 700 });
  }
  body += text(20, 365, 'Starting at one covers 1,2,3 once. Eight begins its separate run.', { width: 320, weight: 700 }) + text(20, 501, 'Two and three stay in the set; their existing predecessors rule out redundant starts.', { width: 320, weight: 700 });
  return { body, height: 611 };
}

function temperatureVersusIndex(picture, model, tools) {
  const { text, color, fill, line } = tools;
  const warmer = picture.temperatures.findIndex((temperature, index) => index > 0 && temperature > picture.temperatures[0]);
  assert.equal(warmer, 3);
  assert.equal(picture.temperatures[warmer] - picture.temperatures[0], 10);
  let body = text(20, 27, 'Temperature change is vertical; waiting follows daily indices', { width: 320, weight: 700 });
  for (const [index, temperature] of picture.temperatures.entries()) {
    const horizontal = 48 + index * 88;
    const top = 350 - (temperature - 60) * 7;
    body += `<rect x="${horizontal - 18}" y="${top}" width="36" height="${350 - top}" style="fill:${fill(index === warmer ? 'output' : 'input')};stroke:${color('input')};stroke-width:1.5"></rect>`
      + text(horizontal, top - 20, temperature, { width: 43, anchor: 'middle', size: 14, weight: 700 }) + text(horizontal, 391, index, { width: 43, anchor: 'middle', size: 14 });
  }
  body += line(22, 280, 331, 280, 'neutral', true) + line(24, 350, 333, 350, 'neutral') + text(22, 356, 60, { width: 20, anchor: 'end', size: 14 }) + text(20, 453, 'Horizontal index difference: 3 - 0 = 3 days', { width: 320, weight: 700 })
    + arrowBetween([48, 512], [312, 512], 'output', tools, 3) + text(20, 604, 'Vertical temperature difference: 80 - 70 = 10 degrees, not the waiting time.', { width: 320, weight: 700 });
  return { body, height: 704 };
}

function nestedStartMembership(picture, model, tools) {
  const { text, circle } = tools;
  const odds = picture.values.map((value, start) => picture.values.slice(start).reduce((count, member) => count + member % 2, 0));
  const selected = [odds.map(count => count <= picture.exact), odds.map(count => count <= picture.exact - 1), odds.map(count => count === picture.exact)];
  assert.deepEqual(selected.map(row => row.filter(Boolean).length), [6, 2, 4]);
  let body = text(20, 27, 'Keep start identities fixed at final endpoint five', { width: 320, weight: 700 }) + text(20, 105, `Input: ${picture.values.join(', ')}`, { width: 320, size: 16 });
  for (const [row, membership] of selected.entries()) {
    const vertical = 198 + row * 191;
    body += text(20, vertical - 57, ['At most three odds: all six starts', 'At most two: only starts four and five', 'Exactly three: starts zero through three'][row], { width: 320, weight: 700 });
    for (const [start, included] of membership.entries()) body += circle(43 + start * 54, vertical, 19, included ? 'output' : 'neutral') + text(43 + start * 54, vertical + 6, start, { width: 33, anchor: 'middle', size: 14, weight: 700 });
  }
  body += text(20, 711, 'Six minus the same two starts leaves four exact-three subarrays.', { width: 320, weight: 700 });
  return { body, height: 803 };
}

function characterCopyCounts(picture, model, tools) {
  const { text, circle, color } = tools;
  const characters = [...new Set(picture.words.join(''))].sort();
  const profiles = picture.words.map(word => characters.map(character => [...word].filter(letter => letter === character).length));
  if (picture.mode === 'groups') assert.equal(new Set(profiles.map(profile => JSON.stringify(profile))).size, picture.expectedGroups);
  else assert.equal(profiles[1].every((count, index) => picture.mode === 'minimum' ? count >= profiles[0][index] : count === profiles[0][index]), false);
  let body = text(20, 27, 'Keep every character copy in its shared character bin', { width: 320, weight: 700 });
  for (const [row, word] of picture.words.entries()) {
    const vertical = 190 + row * 270;
    body += text(20, vertical - 97, `${row ? 'Compare' : picture.mode === 'minimum' ? 'Required' : 'Reference'}: ${word}`, { width: 320, weight: 700 });
    for (const [column, character] of characters.entries()) {
      const horizontal = characters.length === 3 ? 53 + column * 127 : 90 + column * 180;
      const count = profiles[row][column];
      const reference = profiles[0][column];
      body += text(horizontal, vertical - 43, character, { width: 89, anchor: 'middle', weight: 700 });
      for (let copy = 0; copy < Math.max(count, reference); copy++) {
        const present = copy < count;
        const extra = row && picture.mode !== 'minimum' && copy >= reference;
        const tone = !present || extra ? 'warning' : row && count === reference ? 'output' : 'input';
        body += circle(horizontal, vertical + copy * 55, 20, tone) + text(horizontal, vertical + copy * 55 + 6, present ? character : '-', { width: 34, anchor: 'middle', size: 14, weight: 700 });
      }
      body += text(horizontal, vertical + 117, `count ${count}`, { width: 98, anchor: 'middle', size: 14, weight: 700 });
    }
  }
  const footer = 190 + (picture.words.length - 1) * 270 + 195;
  body += text(20, footer, picture.footer, { width: 320, weight: 700 });
  return { body, height: footer + 100 };
}

function setEntryCollapse(picture, model, tools) {
  const { text, circle } = tools;
  const distinct = [...new Set(picture.values)];
  assert.equal(picture.values.length, 4);
  assert.equal(distinct.length, 3);
  let body = text(20, 27, 'Different positions with equal values share one set entry', { width: 320, weight: 700 });
  for (const [position, value] of picture.values.entries()) {
    const horizontal = 53 + position * 84;
    const destination = 62 + distinct.indexOf(value) * 118;
    body += text(horizontal, 115, `index ${position}`, { width: 77, anchor: 'middle', size: 14 }) + circle(horizontal, 171, 24, 'input') + text(horizontal, 177, value, { width: 42, anchor: 'middle', weight: 700 })
      + arrowBetween([horizontal, 204], [destination, 363], 'input', tools, 2);
  }
  for (const [entry, value] of distinct.entries()) body += circle(62 + entry * 118, 401, 26, 'output') + text(62 + entry * 118, 407, value, { width: 46, anchor: 'middle', weight: 700 });
  body += text(20, 514, 'Four input positions collapse to three distinct entries. The two ones share their destination.', { width: 320, weight: 700 });
  return { body, height: 622 };
}

function nextIntervalFit(picture, model, tools) {
  const { text, color, fill, line } = tools;
  const minimum = Math.min(...picture.cases.flatMap(({ kept, next }) => [...kept, ...next]));
  const maximum = Math.max(...picture.cases.flatMap(({ kept, next }) => [...kept, ...next]));
  const horizontal = value => 30 + (value - minimum) * 300 / (maximum - minimum);
  const results = [];
  let body = text(20, 27, picture.title, { width: 320, weight: 700 });
  for (const [frame, { kept, next }] of picture.cases.entries()) {
    const top = 175 + frame * 280;
    const fits = next[0] >= kept[1];
    results.push(fits);
    const overlap = [Math.max(kept[0], next[0]), Math.min(kept[1], next[1])];
    body += text(20, top - 68, `Kept [${kept.join(',')}); next [${next.join(',')})`, { width: 320, weight: 700 });
    for (const [lane, span] of [kept, next].entries()) body += `<rect x="${horizontal(span[0])}" y="${top + lane * 45}" width="${horizontal(span[1]) - horizontal(span[0])}" height="23" style="fill:${fill(lane ? fits ? 'output' : 'warning' : 'input')};stroke:${color(lane ? fits ? 'output' : 'warning' : 'input')};stroke-width:1.5"></rect>`;
    if (overlap[1] > overlap[0]) body += `<rect x="${horizontal(overlap[0])}" y="${top - 6}" width="${horizontal(overlap[1]) - horizontal(overlap[0])}" height="81" style="fill:none;stroke:${color('warning')};stroke-width:2;stroke-dasharray:5 4"></rect>`;
    for (const value of [...new Set([...kept, ...next])]) body += line(horizontal(value), top + 82, horizontal(value), top + 88, 'neutral') + text(horizontal(value), top + 117, value, { width: 43, anchor: 'middle', size: 14 });
    body += text(20, top + 167, fits ? `Start ${next[0]} >= kept end ${kept[1]}: next fits` : `Start ${next[0]} < kept end ${kept[1]}: overlap`, { width: 320, weight: 700, tone: fits ? 'output' : 'warning' });
  }
  assert.deepEqual(results, [true, false]);
  return { body, height: 714 };
}

function intervalStageUnion(picture, model, tools) {
  const { text, color, fill } = tools;
  const normalize = spans => {
    const result = [];
    for (const [start, end] of [...spans].sort((left, right) => left[0] - right[0])) {
      if (result.length && start <= result.at(-1)[1]) result.at(-1)[1] = Math.max(result.at(-1)[1], end);
      else result.push([start, end]);
    }
    return result;
  };
  const spans = picture.stages.flatMap(stage => stage.spans);
  const minimum = Math.min(...spans.map(span => span[0]));
  const maximum = Math.max(...spans.map(span => span[1]));
  const horizontal = value => 28 + (value - minimum) * 304 / (maximum - minimum);
  let body = text(20, 27, 'Overlap extends the active endpoint; separated spans survive', { width: 320, weight: 700 });
  for (const [frame, stage] of picture.stages.entries()) {
    assert.deepEqual(normalize(stage.spans), picture.expected);
    const vertical = 185 + frame * 270;
    body += text(20, vertical - 76, stage.label, { width: 320, weight: 700 });
    for (const [row, [start, end]] of stage.spans.entries()) {
      const top = vertical + row * 64;
      const active = frame && (!row || frame === picture.stages.length - 1);
      body += text(28, top - 15, `[${start},${end}]`, { width: 304, size: 14, weight: 700 }) + `<rect x="${horizontal(start)}" y="${top}" width="${horizontal(end) - horizontal(start)}" height="22" style="fill:${fill(active ? 'output' : 'input')};stroke:${color(active ? 'output' : 'input')};stroke-width:2"></rect>`;
    }
  }
  assert.deepEqual(picture.stages.at(-1).spans, picture.expected);
  const footer = 185 + (picture.stages.length - 1) * 270 + picture.stages.at(-1).spans.length * 64 + 58;
  body += text(20, footer, `Final union: ${picture.expected.map(span => `[${span.join(',')}]`).join(', ')}`, { width: 320, weight: 700 });
  return { body, height: footer + 94 };
}

function advanceWinningHead(picture, model, tools) {
  const { text, circle } = tools;
  const firstHeads = picture.lists.map(list => list[0]);
  const firstWinner = firstHeads.indexOf(Math.min(...firstHeads));
  const secondHeads = firstHeads.map((value, index) => index === firstWinner ? picture.lists[index][1] : value);
  assert.deepEqual(firstHeads, [1, 4, 2]);
  assert.deepEqual(secondHeads, [9, 4, 2]);
  let body = text(20, 27, 'Advance one source head, then compare all exposed heads again', { width: 320, weight: 700 });
  for (const [frame, heads] of [firstHeads, secondHeads].entries()) {
    const vertical = 181 + frame * 290;
    const winner = heads.indexOf(Math.min(...heads));
    body += text(20, vertical - 70, frame ? 'One was emitted; its source now exposes nine' : 'Initial heads: one is the minimum', { width: 320, weight: 700 });
    for (const [index, value] of heads.entries()) {
      const horizontal = 62 + index * 118;
      body += circle(horizontal, vertical, 26, index === winner ? 'output' : 'input') + text(horizontal, vertical + 6, value, { width: 46, anchor: 'middle', weight: 700 });
      if (!frame && picture.lists[index].length > 1) body += arrowBetween([horizontal, vertical + 35], [horizontal, vertical + 85], 'input', tools, 2) + circle(horizontal, vertical + 119, 24, 'input') + text(horizontal, vertical + 125, picture.lists[index][1], { width: 42, anchor: 'middle', weight: 700 });
    }
    if (frame) body += text(20, vertical + 110, `Next minimum is ${heads[winner]}, not nine.`, { width: 320, weight: 700 });
  }
  return { body, height: 669 };
}

function staleDistanceEntry(picture, model, tools) {
  const { text, circle, line, color } = tools;
  assert.deepEqual(picture.queued.map(distance => distance === picture.best), [true, false]);
  let body = text(20, 27, 'Both queued estimates name one node and one best-distance record', { width: 320, weight: 700 }) + text(20, 109, 'Queued distances for the same node', { width: 320, size: 16, weight: 700 });
  for (const [row, distance] of picture.queued.entries()) {
    const vertical = 190 + row * 281;
    body += circle(74, vertical, 28, row ? 'warning' : 'input') + text(74, vertical + 6, distance, { width: 49, anchor: 'middle', weight: 700 });
  }
  body += `<rect x="218" y="277" width="114" height="109" style="fill:none;stroke:${color('output')};stroke-width:2"></rect>` + text(219, 238, 'best distance', { width: 113, size: 14, weight: 700 })
    + circle(275, 330, 27, 'output') + text(275, 336, picture.best, { width: 48, anchor: 'middle', weight: 700 }) + arrowBetween([111, 207], [241, 303], 'output', tools, 3)
    + line(111, 454, 178, 409, 'warning', true) + line(167, 399, 189, 420, 'warning') + text(122, 508, 'stale: skip', { width: 196, weight: 700, tone: 'warning' })
    + text(20, 607, 'Entry two may expand. Entry six must not expand or replace the known shorter distance.', { width: 320, weight: 700 });
  return { body, height: 718 };
}

function nestedRepeatPayload(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  const inner = picture.inner.repeat(picture.innerCount);
  const payload = picture.prefix + inner;
  const decoded = payload.repeat(picture.outerCount);
  assert.equal(inner, 'bbb');
  assert.equal(payload, 'abbb');
  assert.equal(decoded, 'abbbabbb');
  let body = text(20, 27, 'Resolve the inner repeat before copying the completed outer payload', { width: 320, weight: 700 }) + text(35, 136, 'Outer repeat: two whole payloads', { width: 290, weight: 700 })
    + `<rect x="25" y="152" width="310" height="210" style="fill:none;stroke:${color('input')};stroke-width:2"></rect>` + circle(69, 252, 23, 'input') + text(69, 258, picture.prefix, { width: 40, anchor: 'middle', weight: 700 })
    + text(127, 185, 'Inner: three b copies', { width: 193, size: 14, weight: 700 }) + `<rect x="117" y="196" width="200" height="96" style="fill:none;stroke:${color('output')};stroke-width:2"></rect>`;
  for (let copy = 0; copy < picture.innerCount; copy++) body += `<rect x="${133 + copy * 58}" y="221" width="48" height="48" style="fill:${fill('output')};stroke:${color('output')};stroke-width:1.5"></rect>` + text(157 + copy * 58, 252, picture.inner, { width: 41, anchor: 'middle', weight: 700 });
  body += text(20, 421, `Completed payload: ${picture.prefix} + ${inner} = ${payload}`, { width: 320, weight: 700 }) + text(20, 504, 'Repeat that whole completed payload twice', { width: 320, weight: 700 });
  for (let copy = 0; copy < picture.outerCount; copy++) body += `<rect x="${34 + copy * 164}" y="580" width="128" height="58" style="fill:${fill('output')};stroke:${color('output')};stroke-width:2"></rect>` + text(98 + copy * 164, 617, payload, { width: 117, anchor: 'middle', size: 20, weight: 700 });
  return { body, height: 709 };
}

function lengthDelimitedPayload(picture, model, tools) {
  const { text, color, fill, line } = tools;
  const encoded = `${picture.payload.length}${picture.separator}${picture.payload}`;
  assert.equal(encoded, '3#a#b');
  assert.equal(encoded.slice(encoded.indexOf(picture.separator) + 1, encoded.indexOf(picture.separator) + 1 + picture.payload.length), picture.payload);
  let body = text(20, 27, 'Length, not the next separator-looking symbol, ends the payload', { width: 320, weight: 700 });
  for (const [position, label] of ['length', 'sep', 'three payload chars'].entries()) body += text([60, 120, 242][position], 124, label, { width: [70, 54, 188][position], anchor: 'middle', size: 14, weight: 700 });
  for (const [position, character] of [...encoded].entries()) {
    const horizontal = 32 + position * 60;
    const tone = position > 1 ? 'output' : position ? 'neutral' : 'input';
    body += `<rect x="${horizontal}" y="181" width="56" height="50" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>` + text(horizontal + 28, 215, character, { width: 48, anchor: 'middle', size: 20, weight: 700 });
  }
  body += line(152, 253, 328, 253, 'output') + line(152, 242, 152, 260, 'output') + line(328, 242, 328, 260, 'output')
    + text(20, 332, 'The interior hash is counted as payload, not another boundary.', { width: 320, weight: 700 }) + text(180, 463, picture.payload, { width: 280, anchor: 'middle', size: 28, weight: 700, tone: 'output' })
    + text(20, 563, 'Consume exactly three payload characters, including the interior hash.', { width: 320, weight: 700 });
  return { body, height: 662 };
}

function triePrefixTerminal(picture, model, tools) {
  const { text, circle, line, color } = tools;
  assert(picture.word.startsWith(picture.prefix));
  const prefixEnd = picture.prefix.length - 1;
  let body = text(20, 27, 'A reachable prefix path and a stored-word flag are separate state', { width: 320, weight: 700 });
  for (let frame = 0; frame < 2; frame++) {
    const vertical = 182 + frame * 292;
    body += text(20, vertical - 76, frame ? 'Insert app: keep the path; add its word flag' : 'After apple only: app is not yet a word', { width: 320, weight: 700 });
    for (const [position, letter] of [...picture.word].entries()) {
      const horizontal = 42 + position * 68;
      const terminal = position === picture.word.length - 1 || frame && position === prefixEnd;
      if (position) body += line(horizontal - 46, vertical, horizontal - 23, vertical, 'input');
      body += circle(horizontal, vertical, 21, terminal ? 'output' : 'input') + text(horizontal, vertical + 6, letter, { width: 37, anchor: 'middle', size: 16, weight: 700 });
      if (terminal) body += `<circle cx="${horizontal}" cy="${vertical}" r="27" style="fill:none;stroke:${color('output')};stroke-width:1.5"></circle>`;
    }
    body += text(178, vertical + 85, frame ? 'app: word' : 'app: prefix', { width: 139, anchor: 'middle', size: 14, weight: 700 }) + text(314, vertical + 85, 'apple: word', { width: 88, anchor: 'middle', size: 14, weight: 700 });
  }
  return { body, height: 650 };
}

function wildcardTerminalSearch(picture, model, tools) {
  const { text, circle, line, color } = tools;
  assert.equal(picture.word, 'bad');
  let body = text(20, 27, 'A complete matching path still needs a terminal word node', { width: 320, weight: 700 });
  for (let frame = 0; frame < 2; frame++) {
    const vertical = 177 + frame * 274;
    body += text(20, vertical - 76, frame ? 'b.d follows the stored a child to terminal d' : 'ba stops at a: reachable prefix, not a word', { width: 320, weight: 700 });
    for (const [position, letter] of [...picture.word].entries()) {
      const horizontal = 62 + position * 118;
      if (position) body += line(horizontal - 86, vertical, horizontal - 32, vertical, !frame && position === 2 ? 'neutral' : 'input');
      body += circle(horizontal, vertical, 25, position === 2 ? 'output' : 'input') + text(horizontal, vertical + 6, letter, { width: 44, anchor: 'middle', weight: 700 });
      if (position === 2) body += `<circle cx="${horizontal}" cy="${vertical}" r="30" style="fill:none;stroke:${color('output')};stroke-width:1.5"></circle>`;
    }
    body += text(20, vertical + 112, frame ? 'Wildcard matches a; terminal d completes bad: true.' : 'The a node has no word flag: ba is false.', { width: 320, weight: 700, tone: frame ? 'output' : 'warning' });
  }
  return { body, height: 676 };
}

function emitAndContinueTrie(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  assert.deepEqual(picture.words, ['a', 'ab']);
  assert.equal(picture.board.join(''), 'ab');
  let body = text(20, 27, 'Terminal means emit; it does not erase the live child edge', { width: 320, weight: 700 }) + text(20, 115, 'Both a and ab have terminal word flags', { width: 320, weight: 700 });
  for (const [position, letter] of picture.board.entries()) body += circle(89 + position * 182, 201, 27, 'output') + text(89 + position * 182, 207, letter, { width: 48, anchor: 'middle', weight: 700 });
  body += arrowBetween([125, 201], [235, 201], 'output', tools, 3) + text(89, 282, 'emit a', { width: 137, anchor: 'middle', weight: 700 }) + text(271, 282, 'emit ab', { width: 137, anchor: 'middle', weight: 700 })
    + text(20, 384, 'Adjacent board cells permit continuing from a to b', { width: 320, weight: 700 });
  for (const [position, letter] of picture.board.entries()) body += `<rect x="${45 + position * 183}" y="465" width="87" height="87" style="fill:${fill('input')};stroke:${color('input')};stroke-width:2"></rect>` + text(88.5 + position * 183, 518, letter, { width: 77, anchor: 'middle', size: 24, weight: 700 });
  body += arrowBetween([140, 508], [220, 508], 'output', tools, 3) + text(20, 637, 'Returning immediately after a would lose the valid longer word ab.', { width: 320, weight: 700 });
  return { body, height: 733 };
}

function usedBoardCellRefusal(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  const visited = new Set();
  const allowed = picture.requested.map(cell => { const valid = !visited.has(cell); visited.add(cell); return valid; });
  assert.deepEqual(allowed, [true, true, false]);
  let body = text(20, 27, 'The final A matches in value but asks for an already-used cell', { width: 320, weight: 700 }) + text(20, 123, 'Board: only two adjacent cells exist', { width: 320, weight: 700 });
  for (const [cell, letter] of picture.board.entries()) body += `<rect x="${49 + cell * 181}" y="183" width="82" height="78" style="fill:${fill('input')};stroke:${color('input')};stroke-width:2"></rect>` + text(90 + cell * 181, 229, letter, { width: 72, anchor: 'middle', size: 24, weight: 700 }) + text(90 + cell * 181, 305, `cell ${cell}`, { width: 127, anchor: 'middle', size: 16 });
  body += text(20, 395, 'Requested visits for ABA', { width: 320, weight: 700 });
  for (const [step, cell] of picture.requested.entries()) {
    const horizontal = 62 + step * 118;
    body += circle(horizontal, 484, 26, allowed[step] ? 'input' : 'warning') + text(horizontal, 490, picture.board[cell], { width: 46, anchor: 'middle', weight: 700 }) + text(horizontal, 554, `cell ${cell}`, { width: 107, anchor: 'middle', size: 16, tone: allowed[step] ? 'input' : 'warning' });
    if (step) body += arrowBetween([horizontal - 85, 484], [horizontal - 33, 484], allowed[step] ? 'input' : 'warning', tools, 2);
  }
  body += text(20, 648, 'Cell zero is already on this path. The third request is refused.', { width: 320, weight: 700, tone: 'warning' });
  return { body, height: 746 };
}

function normalizedEndComparison(picture, model, tools) {
  const { text, circle, color } = tools;
  const retained = [...picture.input].map((character, index) => [character, index]).filter(([character]) => /^[a-z0-9]$/i.test(character));
  const normalized = retained.map(([character]) => character.toLowerCase()).join('');
  assert.equal(normalized, 'aba');
  assert.equal(normalized, [...normalized].reverse().join(''));
  let body = text(20, 27, 'Skip punctuation, normalize case, then compare the retained ends', { width: 320, weight: 700 });
  for (const [index, character] of [...picture.input].entries()) {
    const horizontal = 53 + index * 63;
    const destination = retained.findIndex(entry => entry[1] === index);
    body += circle(horizontal, 169, 19, destination >= 0 ? 'input' : 'neutral') + text(horizontal, 175, character, { width: 33, anchor: 'middle', size: 14, weight: 700 });
    if (destination >= 0) body += arrowBetween([horizontal, 197], [62 + destination * 118, 345], 'input', tools, 2);
  }
  for (const [index, character] of [...normalized].entries()) body += circle(62 + index * 118, 379, 25, index === 1 ? 'input' : 'output') + text(62 + index * 118, 385, character, { width: 44, anchor: 'middle', weight: 700 });
  body += `<path d="M 62 413 Q 180 499 298 413" style="fill:none;stroke:${color('output')};stroke-width:3"></path>` + text(20, 520, 'Retained lowercase sequence: aba', { width: 320, weight: 700 })
    + text(20, 605, 'Compare the two a ends; punctuation never enters the comparison.', { width: 320, weight: 700 });
  return { body, height: 705 };
}

function topOpenerRefusal(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  const required = { ')': '(', ']': '[', '}': '{' }[picture.incoming];
  assert(picture.stack.includes(required));
  assert.notEqual(picture.stack.at(-1), required);
  let body = text(20, 27, 'A closer tests the top opener, not a matching deeper one', { width: 320, weight: 700 }) + text(20, 111, 'Input so far: ([ ; incoming )', { width: 320, size: 16, weight: 700 })
    + circle(62, 209, 27, 'warning') + text(62, 215, picture.incoming, { width: 48, anchor: 'middle', weight: 700 });
  for (const [index, opener] of picture.stack.entries()) {
    const vertical = 250 - index * 72;
    body += `<rect x="157" y="${vertical}" width="80" height="60" style="fill:${fill(index ? 'warning' : 'neutral')};stroke:${color(index ? 'warning' : 'neutral')};stroke-width:2"></rect>` + text(197, vertical + 39, opener, { width: 69, anchor: 'middle', size: 24, weight: 700 });
  }
  body += arrowBetween([99, 209], [146, 209], 'warning', tools, 3) + text(259, 211, 'top', { width: 79, size: 16, weight: 700 }) + text(252, 290, 'deeper', { width: 86, size: 16 })
    + text(20, 410, 'The required round opener is deeper. The actual top is square, so reject.', { width: 320, weight: 700 }) + text(20, 557, 'A matching opener somewhere in the stack cannot bypass last-in-first-out order.', { width: 320, weight: 700 });
  return { body, height: 670 };
}

function causalWeightProducts(picture, model, tools) {
  const { text, circle } = tools;
  const weights = model.values[picture.query];
  const products = weights.map((weight, index) => weight * picture.values[index]);
  assert(weights.every((weight, index) => index <= picture.query || weight === 0));
  assert.deepEqual(products, [.4, 1.2, 0]);
  assert(Math.abs(products.reduce((sum, value) => sum + value, 0) - 1.6) < 1e-12);
  let body = text(20, 27, 'A zero future weight blocks a contribution, not the input\'s existence', { width: 320, weight: 700 });
  for (const [index, value] of picture.values.entries()) {
    const horizontal = 62 + index * 118;
    const allowed = index <= picture.query;
    body += text(horizontal, 117, `key ${index}`, { width: 102, anchor: 'middle', size: 16, weight: 700 }) + circle(horizontal, 178, 27, allowed ? 'input' : 'warning') + text(horizontal, 184, weights[index], { width: 48, anchor: 'middle', size: 16, weight: 700 })
      + text(horizontal, 280, 'times', { width: 103, anchor: 'middle', size: 16 }) + circle(horizontal, 380, 27, 'input') + text(horizontal, 386, value, { width: 48, anchor: 'middle', size: 16, weight: 700 })
      + text(horizontal, 486, 'equals', { width: 103, anchor: 'middle', size: 16 }) + circle(horizontal, 576, 27, allowed ? 'output' : 'warning') + text(horizontal, 582, products[index], { width: 48, anchor: 'middle', size: 16, weight: 700 });
  }
  body += text(20, 708, 'Output .4 + 1.2 + 0 = 1.6. The large future scalar contributes exactly zero.', { width: 320, weight: 700 });
  return { body, height: 813 };
}

function halvedPopcountReuse(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  const shifted = picture.value >> 1;
  const bits = picture.value.toString(2);
  const reused = shifted.toString(2);
  const priorCount = [...reused].filter(bit => bit === '1').length;
  const low = picture.value & 1;
  assert.equal(bits, '1101');
  assert.equal(shifted, 6);
  assert.equal(priorCount + low, 3);
  let body = text(20, 27, 'Dropping the low bit reuses the popcount of the remaining high bits', { width: 320, weight: 700 }) + text(20, 112, 'Thirteen: binary 1101', { width: 320, weight: 700 });
  for (const [position, bit] of [...bits].entries()) {
    const horizontal = 40 + position * 71;
    body += `<rect x="${horizontal}" y="154" width="66" height="46" style="fill:${fill(position === bits.length - 1 ? 'state' : 'input')};stroke:${color('input')};stroke-width:1.5"></rect>` + text(horizontal + 33, 185, bit, { width: 56, anchor: 'middle', weight: 700 });
    if (position < reused.length) body += arrowBetween([horizontal + 33, 208], [horizontal + 33, 337], 'input', tools, 2);
  }
  for (const [position, bit] of [...reused].entries()) body += `<rect x="${40 + position * 71}" y="349" width="66" height="46" style="fill:${fill('input')};stroke:${color('input')};stroke-width:1.5"></rect>` + text(73 + position * 71, 380, bit, { width: 56, anchor: 'middle', weight: 700 });
  body += text(20, 441, 'Six: binary 110; count two', { width: 218, size: 16, weight: 700 }) + circle(91, 496, 25, 'input') + text(91, 502, priorCount, { width: 44, anchor: 'middle', weight: 700 })
    + arrowBetween([286, 208], [297, 461], 'state', tools, 2) + circle(297, 496, 25, 'state') + text(297, 502, low, { width: 44, anchor: 'middle', weight: 700 })
    + text(20, 557, 'count(6)', { width: 100, size: 16 }) + text(340, 557, 'low bit', { width: 84, anchor: 'end', size: 16 })
    + arrowBetween([91, 533], [155, 622], 'input', tools, 3) + arrowBetween([297, 533], [205, 622], 'state', tools, 3) + circle(180, 658, 27, 'output') + text(180, 664, priorCount + low, { width: 48, anchor: 'middle', weight: 700 })
    + text(20, 756, 'Two reused set bits + low one = three.', { width: 320, weight: 700 });
  return { body, height: 844 };
}

function sharedNormalizerTarget(picture, model, tools) {
  const { text, circle } = tools;
  const normalizer = Math.log(picture.logits.reduce((sum, logit) => sum + Math.exp(logit), 0));
  const losses = picture.logits.map(logit => normalizer - logit);
  assert(Math.abs(losses[0] - .1269280110429726) < 1e-12);
  assert(Math.abs(losses[1] - losses[0] - 2) < 1e-12);
  let body = text(20, 27, 'The same normalizer forks into target-specific subtractions', { width: 320, weight: 700 }) + text(20, 114, `Logits 2,0; shared logZ = ${normalizer.toFixed(3)}`, { width: 320, size: 16, weight: 700 })
    + circle(180, 210, 28, 'input') + text(180, 216, 'Z', { width: 49, anchor: 'middle', weight: 700 });
  for (const [target, logit] of picture.logits.entries()) {
    const horizontal = 89 + target * 182;
    body += arrowBetween([target ? 200 : 160, 242], [horizontal, 338], target ? 'state' : 'input', tools, 3) + circle(horizontal, 374, 27, target ? 'state' : 'input') + text(horizontal, 380, logit, { width: 48, anchor: 'middle', weight: 700 })
      + text(horizontal, 448, `target ${target} logit`, { width: 155, anchor: 'middle', size: 16 }) + text(horizontal, 513, `Z - ${logit}`, { width: 155, anchor: 'middle', weight: 700 })
      + circle(horizontal, 579, 31, 'output') + text(horizontal, 585, losses[target].toFixed(3), { width: 54, anchor: 'middle', size: 14, weight: 700 });
  }
  body += text(20, 697, 'Changing target changes the selected logit and loss, not this shared normalizer.', { width: 320, weight: 700 });
  return { body, height: 798 };
}

function boundedBatchMembership(picture, model, tools) {
  const { text, circle, color } = tools;
  const identities = Array.from({ length: picture.count }, (value, index) => index);
  const batches = [];
  for (let start = 0; start < identities.length; start += picture.size) batches.push(identities.slice(start, start + picture.size));
  assert.deepEqual(batches.map(batch => batch.length), [3, 3, 1]);
  assert.deepEqual(batches.flat(), identities);
  let body = text(20, 27, 'Seven distinct positions divide into three, three, and one', { width: 320, weight: 700 }) + text(20, 119, 'Every input identity appears exactly once', { width: 320, size: 16, weight: 700 });
  for (const [batchIndex, batch] of batches.entries()) {
    const left = 15 + batch[0] * 47;
    const width = (batch.length - 1) * 47 + 42;
    body += `<rect x="${left}" y="154" width="${width}" height="71" style="fill:none;stroke:${color(batchIndex === 2 ? 'output' : 'input')};stroke-width:2"></rect>`;
    for (const identity of batch) body += circle(36 + identity * 47, 188, 17, batchIndex === 2 ? 'output' : 'input') + text(36 + identity * 47, 194, identity, { width: 29, anchor: 'middle', size: 14, weight: 700 });
    body += text(left + width / 2, 271, batch.length, { width: Math.max(38, width), anchor: 'middle', size: 16, weight: 700 });
  }
  for (const [batchIndex, batch] of batches.entries()) body += text(20, 380 + batchIndex * 78, `Slice [${batchIndex * picture.size}:${(batchIndex + 1) * picture.size}] keeps ${batch.join(', ')}`, { width: 320, weight: 700 });
  body += text(20, 670, 'The final request ends at nine, but only real position six remains.', { width: 320, weight: 700 });
  return { body, height: 774 };
}

function xorRangePairing(picture, model, tools) {
  const { text, circle, line } = tools;
  const range = Array.from({ length: picture.count + 1 }, (value, index) => index);
  const result = [...range, ...picture.observed].reduce((total, value) => total ^ value, 0);
  assert.equal(result, 2);
  let body = text(20, 27, 'Equal XOR terms cancel; the missing range value stays unpaired', { width: 320, weight: 700 }) + text(20, 111, 'Range above; observations below', { width: 320, size: 16, weight: 700 });
  for (const [position, value] of range.entries()) {
    const horizontal = 53 + position * 84;
    const paired = picture.observed.includes(value);
    body += circle(horizontal, 177, 24, paired ? 'input' : 'output') + text(horizontal, 183, value, { width: 42, anchor: 'middle', weight: 700 });
    if (paired) body += line(horizontal, 209, horizontal, 353, 'input') + line(horizontal - 12, 269, horizontal + 12, 293, 'neutral') + circle(horizontal, 385, 24, 'input') + text(horizontal, 391, value, { width: 42, anchor: 'middle', weight: 700 });
    else body += arrowBetween([horizontal, 211], [180, 548], 'output', tools, 3);
  }
  body += circle(180, 581, 27, 'output') + text(180, 587, result, { width: 48, anchor: 'middle', weight: 700 }) + text(20, 686, 'All equal pairs cancel. The absent expected value two remains.', { width: 320, weight: 700 });
  return { body, height: 787 };
}

function clearLowestOne(picture, model, tools) {
  const { text, color, fill } = tools;
  const states = [picture.value];
  while (states.at(-1)) states.push(states.at(-1) & (states.at(-1) - 1));
  assert.deepEqual(states, [12, 8, 0]);
  let body = text(20, 27, 'One pass clears one remaining low one, not one arbitrary position', { width: 320, weight: 700 });
  for (const [row, value] of states.entries()) {
    const bits = value.toString(2).padStart(4, '0');
    const next = states[row + 1];
    const nextBits = next === undefined ? bits : next.toString(2).padStart(4, '0');
    if (next !== undefined) assert.equal([...bits].filter(bit => bit === '1').length - [...nextBits].filter(bit => bit === '1').length, 1);
    const vertical = 175 + row * 190;
    body += text(20, vertical - 61, row ? `${states[row - 1]} & ${states[row - 1] - 1} = ${value}` : 'Initial twelve: binary 1100', { width: 320, weight: 700 });
    for (const [position, bit] of [...bits].entries()) {
      const horizontal = 38 + position * 71;
      const tone = bit === '1' && nextBits[position] === '0' ? 'warning' : bit === '1' ? 'input' : 'neutral';
      body += `<rect x="${horizontal}" y="${vertical}" width="65" height="47" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>` + text(horizontal + 32.5, vertical + 31, bit, { width: 56, anchor: 'middle', weight: 700 });
    }
  }
  body += text(20, 698, 'Two passes remove two set bits. Empty positions need no clearing passes.', { width: 320, weight: 700 });
  return { body, height: 802 };
}

function lengthOwnedMask(picture, model, tools) {
  const { text, color, fill } = tools;
  const width = Math.max(...picture.sequences.map(sequence => sequence.length));
  const masks = picture.sequences.map(sequence => Array.from({ length: width }, (value, index) => index < sequence.length));
  assert.deepEqual(masks, [[true, true], [true, false]]);
  let body = text(20, 27, 'Original sequence length owns validity; zero value does not', { width: 320, weight: 700 });
  for (const [row, sequence] of picture.sequences.entries()) {
    const vertical = 185 + row * 262;
    body += text(20, vertical - 79, `Original sequence length ${sequence.length}`, { width: 320, weight: 700 });
    for (let column = 0; column < width; column++) {
      const valid = masks[row][column];
      const value = valid ? sequence[column] : picture.padding;
      const horizontal = 48 + column * 183;
      body += `<rect x="${horizontal}" y="${vertical}" width="82" height="86" style="fill:${fill(valid ? 'input' : 'neutral')};stroke:${color(valid ? 'input' : 'warning')};stroke-width:2${valid ? '' : ';stroke-dasharray:5 3'}"></rect>`
        + text(horizontal + 41, vertical + 37, value, { width: 72, anchor: 'middle', size: 24, weight: 700 }) + text(horizontal + 41, vertical + 68, valid ? 'valid' : 'padding', { width: 72, anchor: 'middle', size: 14, weight: 700 });
    }
  }
  body += text(20, 642, 'Both cells can store zero. Only the first zero belongs to its original sequence.', { width: 320, weight: 700 });
  return { body, height: 745 };
}

function distanceDotIdentity(picture, model, tools) {
  const { text, circle, line } = tools;
  const [first, second] = picture.points;
  const norms = picture.points.map(point => point.reduce((sum, value) => sum + value * value, 0));
  const dot = first.reduce((sum, value, index) => sum + value * second[index], 0);
  const squared = first.reduce((sum, value, index) => sum + (value - second[index]) ** 2, 0);
  assert.deepEqual(norms, [5, 13]);
  assert.equal(dot, 7);
  assert.equal(norms[0] + norms[1] - 2 * dot, squared);
  assert.equal(squared, 4);
  let body = text(20, 27, 'The norm identity equals the actual squared displacement', { width: 320, weight: 700 }) + line(41, 320, 326, 320, 'neutral') + line(41, 320, 41, 105, 'neutral');
  for (const [index, point] of picture.points.entries()) {
    const horizontal = 41 + point[0] * 82;
    const vertical = 320 - point[1] * 82;
    body += line(41, 320, horizontal, vertical, index ? 'state' : 'input') + circle(horizontal, vertical, 20, index ? 'state' : 'input') + text(horizontal, vertical - 38, `(${point.join(',')})`, { width: 99, anchor: 'middle', size: 16, weight: 700 });
  }
  const sharedVertical = 320 - first[1] * 82;
  assert.equal(first[1], second[1]);
  body += arrowBetween([first[0] * 82 + 67, sharedVertical], [second[0] * 82 + 15, sharedVertical], 'output', tools, 3) + text(20, 387, 'Horizontal gap two; squared displacement four.', { width: 320, weight: 700 })
    + text(20, 501, 'Row norms: five and thirteen. Dot product: seven.', { width: 320, weight: 700 }) + text(20, 627, '5 + 13 - 2 x 7 = 4', { width: 320, weight: 700, tone: 'output' });
  return { body, height: 713 };
}

function excludedZeroProduct(picture, model, tools) {
  const { text, circle, line } = tools;
  const left = picture.values.map((value, index) => picture.values.slice(0, index).reduce((product, member) => product * member, 1));
  const right = picture.values.map((value, index) => picture.values.slice(index + 1).reduce((product, member) => product * member, 1));
  const output = left.map((product, index) => product * right[index]);
  assert.deepEqual(left, [1, 2, 0]);
  assert.deepEqual(right, [0, 4, 1]);
  assert.deepEqual(output, [0, 8, 0]);
  let body = text(20, 27, 'For the middle output, exclude its own zero rather than every zero', { width: 320, weight: 700 });
  for (const [index, value] of picture.values.entries()) {
    const horizontal = 62 + index * 118;
    body += text(horizontal, 115, `index ${index}`, { width: 103, anchor: 'middle', size: 14 }) + circle(horizontal, 178, 26, index === picture.index ? 'neutral' : index ? 'state' : 'input') + text(horizontal, 184, value, { width: 46, anchor: 'middle', weight: 700 });
    if (index === picture.index) body += line(horizontal - 17, 161, horizontal + 17, 195, 'warning');
    else body += arrowBetween([horizontal, 213], [index ? 205 : 155, 411], index ? 'state' : 'input', tools, 3);
  }
  body += text(180, 278, 'exclude current', { width: 137, anchor: 'middle', size: 16, weight: 700 }) + circle(180, 449, 28, 'output') + text(180, 455, output[picture.index], { width: 49, anchor: 'middle', weight: 700 })
    + text(20, 543, 'Strictly left two x strictly right four = eight.', { width: 320, weight: 700 }) + text(20, 626, 'All outputs: zero, eight, zero', { width: 320, weight: 700 });
  return { body, height: 721 };
}

function fixedWidthBitMirror(picture, model, tools) {
  const { text, color, fill } = tools;
  const input = BigInt(picture.value);
  let reversed = 0n;
  for (let bit = 0; bit < picture.width; bit++) if (input & (1n << BigInt(bit))) reversed |= 1n << BigInt(picture.width - bit - 1);
  assert.equal(reversed, 2147483648n);
  let body = text(20, 27, 'The fixed thirty-two positions remain even when most bits are zero', { width: 320, weight: 700 });
  for (const [frame, word] of [input, reversed].entries()) {
    const vertical = 182 + frame * 272;
    body += text(28, vertical - 51, 'bit 31', { width: 88, size: 16, weight: 700 }) + text(324, vertical - 51, 'bit 0', { width: 88, anchor: 'end', size: 16, weight: 700 });
    for (let position = 0; position < picture.width; position++) {
      const set = !!(word & (1n << BigInt(picture.width - position - 1)));
      body += `<rect x="${28 + position * 9}" y="${vertical}" width="8" height="40" style="fill:${fill(set ? 'output' : 'neutral')};stroke:${color(set ? 'output' : 'neutral')};stroke-width:1"></rect>`;
    }
  }
  body += arrowBetween([311, 230], [32, 442], 'output', tools, 3) + text(20, 613, 'Low bit zero maps to high bit thirty-one. Unsigned result: 2147483648.', { width: 320, weight: 700 });
  return { body, height: 722 };
}

function stableShiftInvariance(picture, model, tools) {
  const { text, circle, color, fill } = tools;
  const maximum = Math.max(...picture.logits);
  const shifted = picture.logits.map(logit => logit - maximum);
  const exponents = shifted.map(logit => Math.exp(logit));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  const probabilities = exponents.map(value => value / total);
  assert.deepEqual(shifted, [-1, 0]);
  assert(Math.abs(probabilities[0] - 1 / (1 + Math.E)) < 1e-12);
  assert(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  let body = text(20, 27, 'A common logit shift changes exponential scale, not probability shares', { width: 320, weight: 700 });
  for (const [index, logit] of picture.logits.entries()) {
    const horizontal = 89 + index * 182;
    body += circle(horizontal, 178, 28, 'input') + text(horizontal, 184, logit, { width: 49, anchor: 'middle', size: 14, weight: 700 }) + arrowBetween([horizontal, 215], [horizontal, 362], 'input', tools, 3)
      + circle(horizontal, 399, 28, 'output') + text(horizontal, 405, shifted[index], { width: 49, anchor: 'middle', weight: 700 });
  }
  body += text(128, 292, `minus ${maximum}`, { width: 105, size: 16, weight: 700 }) + text(20, 491, 'Stable exponents: exp(-1) and exp(0)', { width: 320, weight: 700 });
  let horizontal = 20;
  for (const [index, probability] of probabilities.entries()) {
    const width = 320 * probability;
    body += `<rect x="${horizontal}" y="566" width="${width}" height="43" style="fill:${fill(index ? 'output' : 'input')};stroke:${color(index ? 'output' : 'input')};stroke-width:1.5"></rect>` + text(horizontal + width / 2, 594, probability.toFixed(3), { width: width - 6, anchor: 'middle', size: 14, weight: 700 });
    horizontal += width;
  }
  body += text(20, 699, 'The common exp(1001) factor cancels before numeric exponentiation.', { width: 320, weight: 700 });
  return { body, height: 800 };
}

function queryBeforeInsert(picture, model, tools) {
  const { text, circle, color } = tools;
  const memory = new Map(picture.mode === 'prefix' ? [[0, 1]] : []);
  const frames = [];
  if (picture.mode === 'complement') {
    for (const [index, value] of picture.values.entries()) {
      const key = picture.target - value;
      const found = memory.get(key);
      frames.push({ label: `Index ${index}: query complement ${key} first`, current: value, stored: memory.size ? '3:0' : 'empty', result: found === undefined ? 'No earlier partner; only then store 3 -> index zero.' : `Found earlier index ${found}; return ${found},${index}.` });
      if (found !== undefined) { assert.equal(found, 0); assert.equal(index, 1); break; }
      memory.set(value, index);
    }
  }
  else {
    assert.equal(picture.mode, 'prefix');
    const prefix = picture.values[0];
    const matches = memory.get(prefix - picture.target) ?? 0;
    assert.equal(matches, 1);
    frames.push({ label: 'Query seeded prefix zero before inserting this zero', current: prefix, stored: '0:1', result: 'One seeded match contributes exactly one subarray.' });
    memory.set(prefix, (memory.get(prefix) ?? 0) + 1);
    frames.push({ label: 'Then insert this prefix; do not repeat its query', current: prefix, stored: '0:2', result: 'Stored frequency becomes two; the answer remains one.' });
  }
  assert.equal(frames.length, 2);
  let body = text(20, 27, 'The query sees only the earlier state; insertion happens afterward', { width: 320, weight: 700 });
  for (const [row, frame] of frames.entries()) {
    const vertical = 182 + row * 285;
    body += text(20, vertical - 76, frame.label, { width: 320, weight: 700 }) + circle(62, vertical, 27, 'input') + text(62, vertical + 6, frame.current, { width: 48, anchor: 'middle', weight: 700 })
      + arrowBetween([98, vertical], [226, vertical], row && picture.mode === 'prefix' ? 'state' : 'input', tools, 3) + `<rect x="239" y="${vertical - 37}" width="98" height="74" style="fill:none;stroke:${color('input')};stroke-width:2"></rect>`
      + text(288, vertical + 6, frame.stored, { width: 88, anchor: 'middle', size: 14, weight: 700 }) + text(20, vertical + 114, frame.result, { width: 320, weight: 700, tone: 'output' });
  }
  return { body, height: 707 };
}

function xorCarryTrace(picture, model, tools) {
  const { text, color, fill } = tools;
  let body = text(20, 27, 'Provisional sum and shifted carry preserve the same total separately', { width: 320, weight: 700 }) + text(20, 108, 'Shown: low four bits; higher positions are zero', { width: 320, size: 14, weight: 700 });
  for (const [row, [sum, carry]] of picture.states.entries()) {
    assert.equal(sum + carry, 8);
    if (row < picture.states.length - 1) assert.deepEqual(picture.states[row + 1], [sum ^ carry, (sum & carry) << 1]);
    const vertical = 188 + row * 123;
    body += text(20, vertical - 31, row ? `Pass ${row}: sum ${sum}; carry ${carry}` : `Input: sum ${sum}; carry ${carry}`, { width: 320, size: 14, weight: 700 });
    for (const [operand, value] of [sum, carry].entries()) for (const [position, bit] of [...value.toString(2).padStart(4, '0')].entries()) {
      const horizontal = 29 + operand * 180 + position * 28;
      const tone = bit === '0' ? 'neutral' : operand ? 'state' : row === picture.states.length - 1 ? 'output' : 'input';
      body += `<rect x="${horizontal}" y="${vertical}" width="24" height="30" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>` + text(horizontal + 12, vertical + 21, bit, { width: 20, anchor: 'middle', size: 14, weight: 700 });
    }
  }
  assert.equal(picture.states.at(-1)[1], 0);
  body += text(20, 841, 'Every row totals eight. Stop only when the shifted carry is zero.', { width: 320, weight: 700 });
  return { body, height: 942 };
}

function markBeforeEnqueue(picture, model, tools) {
  const { text, circle, line, color } = tools;
  const queue = [[0, 0]];
  const discovered = new Set(['0,0']);
  const insertions = new Map([['0,0', 1]]);
  while (queue.length) {
    const [row, column] = queue.shift();
    for (const [rowChange, columnChange] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
      const nextRow = row + rowChange;
      const nextColumn = column + columnChange;
      const key = `${nextRow},${nextColumn}`;
      if (picture.grid[nextRow]?.[nextColumn] === 1 && !discovered.has(key)) {
        discovered.add(key);
        insertions.set(key, (insertions.get(key) ?? 0) + 1);
        queue.push([nextRow, nextColumn]);
      }
    }
  }
  assert.equal(discovered.size, 4);
  assert.equal(insertions.get('1,1'), 1);
  let body = text(20, 27, 'Both neighbors reach one shared cell, not two copies of it', { width: 320, weight: 700 });
  for (const [horizontal, vertical] of [[89, 177], [269, 177], [89, 323], [269, 323]]) body += circle(horizontal, vertical, 25, horizontal === 269 && vertical === 323 ? 'output' : 'input') + text(horizontal, vertical + 6, 1, { width: 44, anchor: 'middle', weight: 700 });
  body += arrowBetween([122, 177], [236, 177], 'input', tools, 2) + arrowBetween([89, 211], [89, 289], 'input', tools, 2) + arrowBetween([269, 211], [269, 289], 'output', tools, 2)
    + line(122, 323, 213, 323, 'warning', true) + line(207, 310, 207, 336, 'warning') + text(269, 392, 'marked (1,1)', { width: 106, anchor: 'middle', size: 14, weight: 700 });
  const queues = [['B', '1,1'], ['1,1']];
  for (const [frame, members] of queues.entries()) {
    const vertical = 520 + frame * 186;
    body += text(20, vertical - 69, frame ? 'After B: shared cell was already marked' : 'After A discovers the shared cell', { width: 320, weight: 700 }) + `<rect x="25" y="${vertical - 34}" width="310" height="68" style="fill:none;stroke:${color('input')};stroke-width:2"></rect>`;
    for (const [position, member] of members.entries()) {
      const horizontal = members.length === 1 ? 180 : 89 + position * 182;
      body += circle(horizontal, vertical, 24, member === 'B' ? 'input' : 'output') + text(horizontal, vertical + 6, member, { width: 42, anchor: 'middle', size: 14, weight: 700 });
    }
  }
  body += text(20, 828, 'The shared cell enters the queue once because its mark precedes the second discovery.', { width: 320, weight: 700 });
  return { body, height: 935 };
}

function reversedHeightAdmission(picture, model, tools) {
  const { text, circle, line } = tools;
  const allowed = (current, neighbor) => neighbor >= current;
  assert.deepEqual(picture.cases.map(([current, neighbor]) => allowed(current, neighbor)), [true, false]);
  assert(allowed(2, 2));
  let body = text(20, 27, 'Reverse source search moves to equal or higher heights', { width: 320, weight: 700 });
  for (const [frame, [current, neighbor]] of picture.cases.entries()) {
    const baseline = 355 + frame * 301;
    const source = [62, baseline - current * 36];
    const target = [298, baseline - neighbor * 36];
    const accepted = allowed(current, neighbor);
    body += text(20, baseline - 244, accepted ? 'Ocean-side two can reverse toward five' : 'Lower one cannot reverse toward ocean-side two', { width: 320, weight: 700 })
      + line(62, baseline, source[0], source[1], 'input') + line(298, baseline, target[0], target[1], accepted ? 'output' : 'warning')
      + circle(...source, 23, 'input') + text(source[0], source[1] + 6, current, { width: 40, anchor: 'middle', weight: 700 }) + circle(...target, 23, accepted ? 'output' : 'warning') + text(target[0], target[1] + 6, neighbor, { width: 40, anchor: 'middle', weight: 700 });
    if (accepted) body += arrowBetween([95, source[1] - 11], [265, target[1] + 15], 'output', tools, 3) + arrowBetween([265, target[1] + 29], [95, source[1] + 3], 'input', tools, 2);
    else body += line(95, source[1] + 5, 211, target[1] - 8, 'warning', true) + line(204, target[1] - 20, 204, target[1] + 6, 'warning');
  }
  body += text(20, 778, 'The accepted reverse uphill edge corresponds to water flowing downhill in the original direction.', { width: 320, weight: 700 });
  return { body, height: 884 };
}

function transposeThenRowReverse(picture, model, tools) {
  const { text, color, fill } = tools;
  const transposed = picture.grid[0].map((value, column) => picture.grid.map(row => row[column]));
  const rotated = transposed.map(row => [...row].reverse());
  assert.deepEqual(transposed, [[1, 3], [2, 4]]);
  assert.deepEqual(rotated, [[3, 1], [4, 2]]);
  let body = text(20, 27, 'Two coordinate changes form the clockwise quarter-turn', { width: 320, weight: 700 });
  for (const [frame, grid] of [picture.grid, transposed, rotated].entries()) {
    const vertical = 175 + frame * 260;
    body += text(20, vertical - 70, ['Original coordinates', 'Transpose: exchange row and column', 'Reverse each row horizontally'][frame], { width: 320, weight: 700 });
    for (const [row, values] of grid.entries()) for (const [column, value] of values.entries()) {
      const horizontal = 90 + column * 110;
      const top = vertical + row * 66;
      const tone = value === 1 ? 'output' : 'input';
      body += `<rect x="${horizontal}" y="${top}" width="66" height="58" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:2"></rect>` + text(horizontal + 33, top + 37, value, { width: 56, anchor: 'middle', size: 24, weight: 700 });
    }
  }
  body += text(20, 894, 'Original one stays at top-left after transpose, then moves to top-right after row reversal.', { width: 320, weight: 700 });
  return { body, height: 1004 };
}

function simultaneousInfectionWaves(picture, model, tools) {
  const { text, color, fill } = tools;
  const arrivals = picture.arrivals.map(row => row.map(value => value === 'wall' ? 'wall' : -1));
  arrivals[0][0] = 0;
  const queue = [[0, 0]];
  while (queue.length) {
    const [row, column] = queue.shift();
    for (const [rowChange, columnChange] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
      const nextRow = row + rowChange;
      const nextColumn = column + columnChange;
      if (arrivals[nextRow]?.[nextColumn] === -1) {
        arrivals[nextRow][nextColumn] = arrivals[row][column] + 1;
        queue.push([nextRow, nextColumn]);
      }
    }
  }
  assert.deepEqual(arrivals, picture.arrivals);
  const waves = Array.from({ length: 5 }, (value, minute) => arrivals.flatMap((row, rowIndex) => row.flatMap((arrival, column) => arrival === minute ? [`${rowIndex},${column}`] : [])));
  assert.deepEqual(waves.map(wave => wave.length), [1, 2, 2, 1, 1]);
  let body = text(20, 27, 'Equal arrival times form one simultaneous infection wave', { width: 320, weight: 700 });
  for (const [row, values] of arrivals.entries()) for (const [column, value] of values.entries()) {
    const horizontal = 53 + column * 85;
    const vertical = 167 + row * 77;
    const tone = value === 'wall' ? 'neutral' : value === 0 ? 'input' : 'output';
    body += `<rect x="${horizontal}" y="${vertical}" width="76" height="64" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>` + text(horizontal + 38, vertical + 39, value, { width: 65, anchor: 'middle', size: value === 'wall' ? 14 : 20, weight: 700 });
  }
  body += text(20, 470, 'Minute buckets: zero contains only the initial seed', { width: 320, weight: 700 });
  for (const [minute, members] of waves.entries()) {
    const horizontal = 43 + minute * 67;
    body += text(horizontal, 552, `t${minute}`, { width: 57, anchor: 'middle', size: 16, weight: 700 });
    for (const [position, member] of members.entries()) body += `<rect x="${horizontal - 28}" y="${582 + position * 43}" width="56" height="31" style="fill:${fill(minute ? 'output' : 'input')};stroke:${color(minute ? 'output' : 'input')};stroke-width:1.5"></rect>` + text(horizontal, 604 + position * 43, member, { width: 48, anchor: 'middle', size: 14, weight: 700 });
  }
  body += text(20, 751, 'Six new infections fit into four waves. The last cell arrives at minute four.', { width: 320, weight: 700 });
  return { body, height: 858 };
}

function frozenZeroObligations(picture, model, tools) {
  const { text, color, fill } = tools;
  const rows = new Set();
  const columns = new Set();
  for (const [row, values] of picture.grid.entries()) for (const [column, value] of values.entries()) if (!value) { rows.add(row); columns.add(column); }
  const output = picture.grid.map((values, row) => values.map((value, column) => rows.has(row) || columns.has(column) ? 0 : value));
  assert.deepEqual([...rows], [0]);
  assert.deepEqual([...columns], [1]);
  assert.deepEqual(output, [[0, 0, 0], [1, 0, 1]]);
  let body = text(20, 27, 'Freeze original zero obligations; new zeros cannot add new ones', { width: 320, weight: 700 });
  for (const [frame, grid] of [picture.grid, output].entries()) {
    const vertical = 181 + frame * 329;
    body += text(20, vertical - 76, frame ? 'Apply only frozen row zero and column one' : 'The only original zero is row zero, column one', { width: 320, weight: 700 });
    for (const [row, values] of grid.entries()) for (const [column, value] of values.entries()) {
      const horizontal = 48 + column * 88;
      const top = vertical + row * 78;
      const tone = !frame && !value ? 'warning' : frame && !value ? 'output' : 'input';
      body += `<rect x="${horizontal}" y="${top}" width="76" height="56" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.5"></rect>` + text(horizontal + 38, top + 35, value, { width: 66, anchor: 'middle', size: 24, weight: 700 });
    }
    if (!frame) body += `<rect x="42" y="${vertical - 6}" width="264" height="68" style="fill:none;stroke:${color('output')};stroke-width:2"></rect><rect x="130" y="${vertical - 6}" width="88" height="146" style="fill:none;stroke:${color('state')};stroke-width:2"></rect>`;
  }
  body += text(20, 753, 'The two lower corner ones remain. Newly written top-row zeros cannot invent extra column obligations.', { width: 320, weight: 700 });
  return { body, height: 864 };
}

function crossedSpiralBoundary(picture, model, tools) {
  const { text, color, fill, line } = tools;
  let top = 0;
  const bottom = picture.grid.length - 1;
  const emitted = [...picture.grid[top]];
  top++;
  const bottomPass = top <= bottom ? picture.grid[bottom] : [];
  assert.deepEqual(emitted, [1, 2, 3]);
  assert.equal(top, 1);
  assert.equal(bottom, 0);
  assert.equal(bottomPass.length, 0);
  let body = text(20, 27, 'The top sweep already consumes the one-row remainder', { width: 320, weight: 700 }) + text(20, 115, 'Emit one, two, three across the top', { width: 320, weight: 700 });
  for (const [position, value] of emitted.entries()) body += `<rect x="${52 + position * 86}" y="177" width="76" height="60" style="fill:${fill('output')};stroke:${color('output')};stroke-width:2"></rect>` + text(90 + position * 86, 215, value, { width: 66, anchor: 'middle', size: 24, weight: 700 });
  body += line(30, 237, 330, 237, 'warning', true) + line(30, 257, 330, 257, 'input', true) + text(20, 332, 'Bottom stays at row zero.', { width: 320, weight: 700 }) + text(20, 419, 'Top advances to row one, beyond that bottom.', { width: 320, weight: 700 })
    + text(20, 563, 'Guard 1 <= 0 fails. Skip the bottom sweep; all three cells were already emitted.', { width: 320, weight: 700, tone: 'warning' });
  return { body, height: 677 };
}

const drawers = {
  'mark-before-enqueue': markBeforeEnqueue,
  'reversed-height-admission': reversedHeightAdmission,
  'transpose-then-row-reverse': transposeThenRowReverse,
  'simultaneous-infection-waves': simultaneousInfectionWaves,
  'frozen-zero-obligations': frozenZeroObligations,
  'crossed-spiral-boundary': crossedSpiralBoundary,
  'causal-weight-products': causalWeightProducts,
  'halved-popcount-reuse': halvedPopcountReuse,
  'shared-normalizer-target': sharedNormalizerTarget,
  'bounded-batch-membership': boundedBatchMembership,
  'xor-range-pairing': xorRangePairing,
  'clear-lowest-one': clearLowestOne,
  'length-owned-mask': lengthOwnedMask,
  'distance-dot-identity': distanceDotIdentity,
  'excluded-zero-product': excludedZeroProduct,
  'fixed-width-bit-mirror': fixedWidthBitMirror,
  'stable-shift-invariance': stableShiftInvariance,
  'query-before-insert': queryBeforeInsert,
  'xor-carry-trace': xorCarryTrace,
  'nested-repeat-payload': nestedRepeatPayload,
  'length-delimited-payload': lengthDelimitedPayload,
  'trie-prefix-terminal': triePrefixTerminal,
  'wildcard-terminal-search': wildcardTerminalSearch,
  'emit-and-continue-trie': emitAndContinueTrie,
  'used-board-cell-refusal': usedBoardCellRefusal,
  'normalized-end-comparison': normalizedEndComparison,
  'top-opener-refusal': topOpenerRefusal,
  'next-interval-fit': nextIntervalFit,
  'interval-stage-union': intervalStageUnion,
  'advance-winning-head': advanceWinningHead,
  'stale-distance-entry': staleDistanceEntry,
  'saved-witness-trace': savedWitnessTrace,
  'replacement-window-witness': replacementWindowWitness,
  'negative-extrema-flip': negativeExtremaFlip,
  'nondecreasing-window-boundary': nondecreasingWindowBoundary,
  'reachable-jump-frontier': reachableJumpFrontier,
  'run-start-membership': runStartMembership,
  'temperature-versus-index': temperatureVersusIndex,
  'nested-start-membership': nestedStartMembership,
  'character-copy-counts': characterCopyCounts,
  'set-entry-collapse': setEntryCollapse,
  'coin-predecessor': coinPredecessor,
  'decimal-pair-gate': decimalPairGate,
  'edit-diagonal-cost': editDiagonalCost,
  'circular-endpoint-cases': circularEndpointCases,
  'shared-increasing-suffix': sharedIncreasingSuffix,
  'palindrome-occurrence-spans': palindromeOccurrenceSpans,
  'unused-position-availability': unusedPositionAvailability,
  'binary-choice-leaves': binaryChoiceLeaves,
  'grid-final-move': gridFinalMove,
  'word-boundary-alternatives': wordBoundaryAlternatives,
  'tree-shape-comparison': treeShapeComparison,
  'inherited-ancestor-bound': inheritedAncestorBound,
  'bst-stop-boundary': bstStopBoundary,
  'inorder-versus-push': inorderVersusPush,
  'pointer-identity-meeting': pointerIdentityMeeting,
  'linked-chain-frames': linkedChainFrames,
  'minimum-history-pop': minimumHistoryPop,
  'histogram-sentinel-retained': histogramSentinelRetained,
  'recency-frontier': recencyFrontier,
  'first-unequal-character': firstUnequalCharacter,
  'child-return-join': childReturnJoin,
  'frozen-queue-level': frozenQueueLevel,
  'branching-return': branchingReturn,
  'clone-identity-boundary': cloneIdentityBoundary,
  'root-membership-split': rootMembershipSplit,
  'empty-ready-counterexample': emptyReadyCounterexample,
  'component-membership': componentMembership,
  'frequency-buckets': frequencyBuckets,
  'monotone-pair-bound': monotonePairBound,
  'candidate-cut': candidateCut,
  'shorter-wall-bound': shorterWallBound,
  'ceiling-hours': ceilingHours,
  'greedy-capacity': greedyCapacity,
  'retained-min-heap': retainedMinHeap,
  'tails-versus-witness': tailsVersusWitness,
  'heap-order-gate': heapOrderGate,
  'subset-rank-order': subsetRankOrder,
  'condition-population': conditionPopulation,
  'square-spread': squareSpread,
  'weighted-values': weightedValues,
  'biased-sample': biasedSample,
  'select-denominator': selectDenominator,
  'forecast-count': forecastCount,
  'logged-support': loggedSupport,
  'replace-slot': replaceSlot,
  'nested-cohorts': nestedCohorts,
  'bounded-capacity': boundedCapacity,
  'arithmetic-paths': arithmeticPaths,
  'gain-cascade': gainCascade,
  'residual-step': residualStep,
  'linked-scales': linkedScales,
  'geometric-action': geometricAction,
  'event-clock': eventClock,
  'policy-bands': policyBands,
  'budget-transfer': budgetTransfer,
  'decay-bypass': decayBypass,
  'density-update': densityUpdate,
  'random-branch': randomBranch,
  'split-observation': splitObservation,
  'binary-channel': binaryChannel,
  'average-distribution': averageDistribution,
  'channel-pooling': channelPooling,
  'sufficient-compression': sufficientCompression,
  'bit-reassignment': bitReassignment,
  'target-redistribution': targetRedistribution,
  'interpolation-freedom': interpolationFreedom,
  'likelihood-candidates': likelihoodCandidates,
  'local-slope': localSlope,
  'weighted-draws': weightedDraws,
  'mass-expansion': massExpansion,
  'baseline-displacement': baselineDisplacement,
  'odds-update': oddsUpdate,
  'rectangle-exchange': rectangleExchange,
  'copied-gradient': copiedGradient,
  'potential-loop': potentialLoop,
  'backward-fold': backwardFold,
  'target-corruption': targetCorruption,
  'cluster-resample': clusterResample,
  'gradient-boundary': gradientBoundary,
  'alignment-collapse': alignmentCollapse,
  'patch-columns': patchColumns,
  'shared-memory': sharedMemory,
  'feature-partition': featurePartition,
  'cache-frontier': cacheFrontier,
  'causal-frontier': causalFrontier,
  'clipped-image-mass': clippedImageMass,
  'remove-structure': removeStructure,
  'relabel-population': relabelPopulation,
  'centered-group': centeredGroup,
  'axis-pooling': axisPooling,
  'count-pooling': countPooling,
  'assigned-centers': assignedCenters,
  'uncertainty-pin': uncertaintyPin,
  'soften-mass': softenMass,
  'ordering-credit': orderingCredit,
  'mass-cutoffs': massCutoffs,
  'joint-paths': jointPaths,
  'iterative-step': iterativeStep,
  'constraint-contact': constraintContact,
  'fixed-noise': fixedNoise,
  'scale-before-round': scaleBeforeRound,
  'rank-credit': rankCredit,
  'rank-displacement': rankDisplacement,
  'quantization-ruler': quantizationRuler,
  'one-sided-plateau': oneSidedPlateau,
  'conditioning-paths': conditioningPaths,
  'confounding-intervention': confoundingIntervention,
  'actor-critic-routes': actorCriticRoutes,
  'reverse-fanout': reverseFanout,
  'forward-bypass': forwardBypass,
  'dependent-generation': dependentGeneration,
  'hidden-emissions': hiddenEmissions,
  'bound-climb': boundClimb,
  'complex-turn': complexTurn,
  'uncertainty-choice': uncertaintyChoice,
  'path-score-movement': pathScoreMovement,
  'opposed-directions': opposedDirections,
  'variance-square': varianceSquare,
  'saturated-gradient': saturatedGradient,
  'shared-factor-area': sharedFactorArea,
  'parameter-version': parameterVersion,
  'frozen-factor': frozenFactor,
  'coordinate-relabel': coordinateRelabel,
  'hard-forward-surrogate': hardForwardSurrogate,
  'balanced-reservoirs': balancedReservoirs,
  'weighted-jacobian': weightedJacobian,
  'token-buckets': tokenBuckets,
  'unbounded-valley': unboundedValley,
  'reciprocal-rates': reciprocalRates,
  'event-count-wait': eventCountWait,
  'shared-noise-floor': sharedNoiseFloor,
  'document-matches': documentMatches,
  'fixed-floor': fixedFloor,
  'saved-rebuilt-states': savedRebuiltStates,
  'repeat-or-condition': repeatOrCondition,
  'equal-kernel-area': equalKernelArea,
  'averaging-population': averagingPopulation,
  'neighborhood-messages': neighborhoodMessages,
  'bandwidth-ceiling': bandwidthCeiling,
  'normalizer-shift': normalizerShift,
  'retained-frontier': retainedFrontier,
  'shape-contract': shapeContract,
  'feature-summary': featureSummary,
  'position-entry-point': positionEntryPoint,
  'alignment-axes': alignmentAxes,
  'denominator-members': denominatorMembers,
  'common-rotation': commonRotation,
  'query-rows': queryRows,
  'lost-boundary': lostBoundary,
  'packed-isolation': packedIsolation,
  'velocity-fold': velocityFold,
  'missing-cross-products': missingCrossProducts,
  'subtract-gradient': subtractGradient,
  'two-hop-relay': twoHopRelay,
  'threshold-members': thresholdMembers,
  'nonzero-contributions': nonzeroContributions,
  'fixed-merge-order': fixedMergeOrder,
  'mixing-axes': mixingAxes,
  'local-gap': localGap,
  'piecewise-fit': piecewiseFit,
  'equal-value-policies': equalValuePolicies,
  'patch-token-inventory': patchTokenInventory,
  'shared-mask-weights': sharedMaskWeights,
  'translated-displacement': translatedDisplacement,
  'serial-gates': serialGates,
  'evidence-chain': evidenceChain,
  'observed-labels': observedLabels,
  'separate-state-rails': separateStateRails,
  'broken-update': brokenUpdate,
  'compare-swap-owner': compareSwapOwner,
  'idempotent-receipt': idempotentReceipt,
  'retained-evidence': retainedEvidence,
  'persistent-refreshing': persistentRefreshing,
  'node-reassignment': nodeReassignment,
  'assignment-membership': assignmentMembership,
  'conversation-assignment': conversationAssignment,
  'hierarchical-sum': hierarchicalSum,
  'ring-reduction': ringReduction,
  'strict-boundary': strictBoundary,
  'suppression-route': suppressionRoute,
  'clock-after-boundary': clockAfterBoundary,
  'unscanned-partition': unscannedPartition,
  'role-budget': roleBudget,
  'committed-provisional': committedProvisional,
  'versioned-reply': versionedReply,
  'memory-roundtrip': memoryRoundtrip,
  'complement-before-store': complementBeforeStore,
  'policy-region': policyRegion,
  'matched-comparison': matchedComparison,
  'group-cut': groupCut,
  'core-border-source': coreBorderSource,
  'first-failure': firstFailure,
  'alternative-whole': alternativeWhole,
  'conditional-spatial-cut': conditionalSpatialCut,
  'authority-rails': authorityRails,
  'availability-before': availabilityBefore,
  'shared-supply-return': sharedSupplyReturn,
  'duplicate-penalty': duplicatePenalty,
  'fixed-source-growing-prefix': fixedSourceGrowingPrefix,
  'weighted-tail': weightedTail,
  'restore-common': restoreCommon,
  'independent-choices': independentChoices,
  'independent-evidence': independentEvidence,
  'stable-tile-merge': stableTileMerge,
  'transient-gather': transientGather,
  'gradient-ownership': gradientOwnership,
  'count-floor': countFloor,
  'replace-one-state': replaceOneState,
  'gain-floor': gainFloor,
  'normalized-ranking': normalizedRanking,
  'count-versus-size': countVersusSize,
  'hypothesis-separation': hypothesisSeparation,
  'future-net': futureNet,
  'scope-join': scopeJoin,
  'recoverable-fork': recoverableFork,
  'cached-and-output': cachedAndOutput,
  'logical-growth': logicalGrowth,
  'different-traces': differentTraces,
  'stored-executed': storedExecuted,
  'contribution-boundaries': contributionBoundaries,
  'training-information': trainingInformation,
  'different-meeting-points': differentMeetingPoints,
  'truth-versus-predictions': truthVersusPredictions,
  'parallel-finish': parallelFinish,
  'logical-physical-pages': logicalPhysicalPages,
  'pipeline-wave': pipelineWave,
  'distinct-examples': distinctExamples,
  'one-load-many-computes': oneLoadManyComputes,
  'overlap-tail': overlapTail,
  'versioned-chunk': versionedChunk,
  'age-versus-latency': ageVersusLatency,
  'accepted-baseline': acceptedBaseline,
  'opportunity-count': opportunityCount,
  'preference-routes': preferenceRoutes,
  'label-visited-state': labelVisitedState,
  'prefix-correction': prefixCorrection,
  'revisable-words': revisableWords,
  'device-time-area': deviceTimeArea,
  'partial-coordinate-sum': partialCoordinateSum,
  'partitioned-object': partitionedObject,
  'offline-index-live-query': offlineIndexLiveQuery,
  'different-latent-source': differentLatentSource,
  'cost-per-output': costPerOutput,
  'split-dependencies': splitDependencies,
  'epoch-members': epochMembers,
  'positive-in-whole': positiveInWhole,
  'scaled-clipping-order': scaledClippingOrder,
  'last-step-paths': lastStepPaths,
  'prerequisite-counter': prerequisiteCounter,
  'nonadjacent-choice': nonadjacentChoice,
  'conditional-diagonal': conditionalDiagonal,
  'old-snapshot-extension': oldSnapshotExtension,
  'graph-versus-parents': graphVersusParents,
  'ordered-choice-path': orderedChoicePath,
};

export function renderMentalPicture(identity, model, tools) {
  const picture = mentalPictures[identity];
  if (!picture) return null;
  assert(picture.review && picture.prediction, `${identity}: individual editorial decision and transfer prediction required`);
  if (picture.operation === 'coefficient-restraint') {
    assert.equal(model.kind, 'shrinkage');
    return null;
  }
  assert(drawers[picture.operation], `${identity}: supported physical mechanism required`);
  const drawing = drawers[picture.operation](picture, model, tools);
  assert(drawing.height > 0 && !/NaN|undefined|Infinity/.test(drawing.body), identity);
  return { ...drawing, composition: picture.operation };
}
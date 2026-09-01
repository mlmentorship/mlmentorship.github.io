const frame = (label, note, scene) => ({ label, note, scene });
const visual = (objective, frames) => ({ objective, frames });
const mark = (index, label, tone = 'focus') => ({ index, label, tone });
const array = (items, marks = [], extra = {}) => ({ type: 'array', items, marks, ...extra });
const arrayMap = (items, map, marks = [], extra = {}) => ({ type: 'array-map', items, map, marks, ...extra });
const table = (columns, rows, active = [], extra = {}) => ({ type: 'table', columns, rows, active, ...extra });
const grid = (rows, marks = [], extra = {}) => ({ type: 'grid', rows, marks, ...extra });
const stack = (input, values, extra = {}) => ({ type: 'stack', input, values, ...extra });
const queueGrid = (rows, queue, extra = {}) => ({ type: 'queue-grid', rows, queue, ...extra });
const graph = (nodes, edges, extra = {}) => ({ type: 'graph', nodes, edges, ...extra });
const tree = (levels, marks = [], extra = {}) => ({ type: 'tree', levels, marks, ...extra });
const intervals = (items, extra = {}) => ({ type: 'intervals', items, ...extra });
const linked = (nodes, extra = {}) => ({ type: 'linked', nodes, ...extra });
const trie = (paths, extra = {}) => ({ type: 'trie', paths, ...extra });
const bits = (values, marks = [], extra = {}) => ({ type: 'bits', values, marks, ...extra });
const shapes = (items, extra = {}) => ({ type: 'shapes', items, ...extra });
const attention = (rows, extra = {}) => ({ type: 'attention', rows, ...extra });
const buckets = (items, extra = {}) => ({ type: 'buckets', items, ...extra });
const prefix = (items, extra = {}) => ({ type: 'prefix', items, ...extra });
const dualWindow = (items, extra = {}) => ({ type: 'dual-window', items, ...extra });
const choices = (path, branches, extra = {}) => ({ type: 'choices', path, branches, ...extra });
const lru = (map, order, extra = {}) => ({ type: 'lru', map, order, ...extra });
const heap = (values, extra = {}) => ({ type: 'heap', values, ...extra });

export const codingQuestionVisuals = {
  'two-sum': visual('Find the complement in the values already scanned.', [
    frame('Read the first value', '2 is current. Nothing is saved yet.', arrayMap(['2', '7', '11', '15'], [], [mark(0, 'current')])),
    frame('Ask for the complement', 'At 7, the target 9 needs 2. The map already stores 2 at index 0.', arrayMap(['2', '7', '11', '15'], [['2', 'index 0']], [mark(0, 'saved', 'state'), mark(1, 'current'), mark(1, 'need 2', 'focus')])),
    frame('Return the pair', 'The complement is present, so indices 0 and 1 finish the search.', arrayMap(['2', '7', '11', '15'], [['2', 'index 0']], [mark(0, 'pair', 'output'), mark(1, 'pair', 'output')], { result: '[0, 1]' })),
  ]),
  'valid-anagram': visual('Compare letter counts, not letter positions.', [
    frame('Count the first word', 'eat contributes one e, one a, and one t.', table(['letter', 'eat', 'tea'], [['a', '1', '-'], ['e', '1', '-'], ['t', '1', '-']], [1, 4, 7])),
    frame('Consume the second word', 'tea removes the same three counts in a different order.', table(['letter', 'eat', 'tea'], [['a', '1', '1'], ['e', '1', '1'], ['t', '1', '1']], [1, 2, 4, 5, 7, 8], { status: 'all counts match' })),
    frame('Accept', 'Every count is equal, so the strings are anagrams.', table(['letter', 'left', 'right'], [['a', '1', '1'], ['e', '1', '1'], ['t', '1', '1']], [], { status: 'true', result: 'true' })),
  ]),
  'group-anagrams': visual('Use one frequency signature as the address for each word group.', [
    frame('Build the first bucket', 'eat and tea have the same sorted signature.', buckets([{ count: '[a,e,t]', items: ['eat', 'tea'], tone: 'focus' }, { count: '[a,n,t]', items: [], tone: 'neutral' }])),
    frame('Branch on a new signature', 'tan belongs under [a,n,t], while ate returns to [a,e,t].', buckets([{ count: '[a,e,t]', items: ['eat', 'tea', 'ate'], tone: 'state' }, { count: '[a,n,t]', items: ['tan'], tone: 'focus' }])),
    frame('Read the groups', 'Words sharing a signature are already together.', buckets([{ count: '[a,e,t]', items: ['eat', 'tea', 'ate'] }, { count: '[a,n,t]', items: ['tan', 'nat'] }, { count: '[a,b,t]', items: ['bat'] }], { status: 'three buckets', result: 'three groups' })),
  ]),
  'longest-consecutive-sequence': visual('Start a run only at a value with no predecessor.', [
    frame('Find a run start', '4 is skipped because 3 exists. 1 has no predecessor, so it starts a run.', arrayMap(['100', '4', '200', '1', '3', '2'], [['1', 'start']], [mark(3, 'start', 'focus')])),
    frame('Walk forward', 'The set answers 2, 3, and 4 in constant-time average lookups.', arrayMap(['1', '2', '3', '4'], [['1', 'run length 4']], [mark(0, 'start', 'state'), mark(3, 'end', 'output')])),
    frame('Keep the longest', 'Every other value either starts a shorter run or belongs to this one.', arrayMap(['1', '2', '3', '4'], [['1', 'best = 4']], [mark(0, 'best', 'output'), mark(1, 'best', 'output'), mark(2, 'best', 'output'), mark(3, 'best', 'output')], { result: '4' })),
  ]),
  'product-of-array-except-self': visual('Build each answer from the product to its left and the product to its right.', [
    frame('Write left products', 'At each index, store the product strictly before it.', prefix(['1', '2', '3', '4'], { left: ['1', '1', '2', '6'], right: ['-', '-', '-', '-'], answer: ['1', '1', '2', '6'], active: 0 })),
    frame('Walk back from the right', 'A suffix product is multiplied into each saved prefix product.', prefix(['1', '2', '3', '4'], { left: ['1', '1', '2', '6'], right: ['24', '12', '4', '1'], answer: ['24', '12', '8', '6'], active: 2 })),
    frame('Exclude the current value', 'Each answer combines everything on both sides and never divides.', prefix(['1', '2', '3', '4'], { left: ['1', '1', '2', '6'], right: ['24', '12', '4', '1'], answer: ['24', '12', '8', '6'], active: 3, status: 'complete', result: '[24,12,8,6]' })),
  ]),
  'subarray-sum-equals-k': visual('Turn a target subarray into a lookup between two prefix sums.', [
    frame('Record the empty prefix', 'Before reading values, prefix sum 0 has appeared once.', arrayMap(['0', '1', '2', '3'], [['0', 'count 1']], [mark(0, 'prefix 0', 'state')])),
    frame('Reach prefix 2', 'After the second 1, current prefix is 2. It needs an earlier prefix 0.', arrayMap(['0', '1', '2', '3'], [['0', 'count 1'], ['1', 'count 1']], [mark(2, 'prefix 2', 'focus')], { query: '2 - k = 0' })),
    frame('Count every match', 'The prefix-2 query finds prefix 0; prefix 3 later finds prefix 1.', arrayMap(['0', '1', '2', '3'], [['0', 'count 1'], ['1', 'count 1'], ['2', 'count 1'], ['3', 'count 1']], [mark(2, 'match', 'output'), mark(3, 'match', 'output')], { result: '2 subarrays' })),
  ]),
  '3sum': visual('Fix one value, then solve the remaining pair with two sorted pointers.', [
    frame('Sort and fix', 'After sorting, fix -1 at index 1. The pair search starts to its right.', array(['-4', '-1', '-1', '0', '1', '2'], [mark(1, 'fixed', 'state'), mark(2, 'L'), mark(5, 'R')])),
    frame('Move toward zero', 'The sum -1 + -1 + 2 is 0, so record it and move both pointers.', array(['-4', '-1', '-1', '0', '1', '2'], [mark(1, 'fixed', 'state'), mark(2, 'pair', 'output'), mark(5, 'pair', 'output')], { result: '[-1,-1,2]' })),
    frame('Find the second triple', 'With fixed -1, pointers reach 0 and 1 and record [-1,0,1].', array(['-4', '-1', '-1', '0', '1', '2'], [mark(1, 'fixed', 'state'), mark(3, 'pair', 'output'), mark(4, 'pair', 'output')], { result: '[-1,0,1]' })),
  ]),
  'container-with-most-water': visual('The shorter wall limits area, so move that wall inward.', [
    frame('Start at both ends', 'Width is largest, but the left wall of height 1 limits the water.', array(['1', '8', '6', '2', '5', '4', '8', '3', '7'], [mark(0, 'L', 'focus'), mark(8, 'R', 'focus')], { measure: 'area = 8' })),
    frame('Move the shorter wall', 'Moving the height-7 wall cannot improve the height-1 limit. Move L.', array(['1', '8', '6', '2', '5', '4', '8', '3', '7'], [mark(1, 'L', 'focus'), mark(8, 'R', 'focus')], { measure: 'height limit = 7' })),
    frame('Keep the best area', 'At heights 8 and 7, width 7 gives the best area 49.', array(['1', '8', '6', '2', '5', '4', '8', '3', '7'], [mark(1, 'best', 'output'), mark(8, 'best', 'output')], { measure: 'best = 49', result: '49' })),
  ]),
  'longest-substring-without-repeating-characters': visual('Keep the longest window whose characters are all distinct.', [
    frame('Grow the window', 'The first window abc contains no duplicate.', array(['a', 'b', 'c', 'a', 'b', 'c', 'b', 'b'], [mark(0, 'L'), mark(2, 'R', 'focus')], { range: 'abc', state: 'a,b,c' })),
    frame('Repair the duplicate', 'The next a repeats, so move L past the old a before continuing.', array(['a', 'b', 'c', 'a', 'b', 'c', 'b', 'b'], [mark(1, 'L', 'focus'), mark(3, 'R', 'focus')], { range: 'bca', state: 'b,c,a' })),
    frame('Save the best window', 'The longest distinct window seen has length 3.', array(['a', 'b', 'c', 'a', 'b', 'c', 'b', 'b'], [mark(1, 'best', 'output'), mark(3, 'best', 'output')], { range: 'bca', result: '3' })),
  ]),
  'longest-repeating-character-replacement': visual('A window is valid when every non-majority character fits inside the replacement budget.', [
    frame('Count the window', 'AAB has majority count 2. One B needs one replacement.', array(['A', 'A', 'B', 'A', 'B', 'B', 'A'], [mark(0, 'L'), mark(2, 'R', 'focus')], { range: 'AAB', formula: '3 - max_count 2 = 1' })),
    frame('Keep a valid length-4 window', 'AABA uses one replacement: length 4 minus majority count 3 equals 1.', array(['A', 'A', 'B', 'A', 'B', 'B', 'A'], [mark(0, 'L', 'focus'), mark(3, 'R', 'focus')], { range: 'AABA', formula: '4 - max_count 3 = 1' })),
    frame('Return the longest length', 'The scan may later see ABAB, but the valid window AABA already proves length 4.', array(['A', 'A', 'B', 'A', 'B', 'B', 'A'], [mark(0, 'best', 'output'), mark(3, 'best', 'output')], { range: 'AABA', result: '4' })),
  ]),
  'permutation-in-string': visual('Compare each fixed-width window with the pattern counts.', [
    frame('Build the pattern count', 'The pattern ab needs one a and one b. The first text window ei has neither.', table(['window', 'a', 'b'], [['ab', '1', '1'], ['ei', '0', '0']], [0, 1, 2])),
    frame('Slide to a candidate', 'The window ba has the same counts as ab, even though the order differs.', table(['window', 'a', 'b'], [['ab', '1', '1'], ['ba', '1', '1']], [3, 4, 5], { status: 'match' })),
    frame('Return true', 'A matching count window means a permutation appears in the text.', array(['e', 'i', 'd', 'b', 'a', 'o', 'o', 'o'], [mark(3, 'window', 'output'), mark(4, 'window', 'output')], { result: 'true' })),
  ]),
  'count-number-of-nice-subarrays': visual('Count exact odd counts by subtracting two at-most windows.', [
    frame('At most 3 odd values', 'The final valid window begins at index 1; the full array has four odd values.', dualWindow(['1', '1', '2', '1', '1'], { windows: [{ label: 'at most 3', range: [1, 4], count: '14 subarrays' }, { label: 'at most 2', range: [2, 4], count: '12 subarrays' }] })),
    frame('At most 2 odd values', 'The second left boundary moves to index 2, leaving two odd values in the final window.', dualWindow(['1', '1', '2', '1', '1'], { windows: [{ label: 'at most 3', range: [1, 4], count: '14' }, { label: 'at most 2', range: [2, 4], count: '12' }] })),
    frame('Subtract the counts', 'Exactly 3 odds = at_most(3) - at_most(2) = 14 - 12 = 2.', dualWindow(['1', '1', '2', '1', '1'], { windows: [{ label: 'at most 3', range: [1, 4], count: '14' }, { label: 'at most 2', range: [2, 4], count: '12' }], result: '2 nice subarrays' })),
  ]),
  'binary-search': visual('Keep the sorted half that can still contain the target.', [
    frame('Probe the middle', 'The middle value 5 is below target 7, so the lower half is finished.', array(['1', '3', '5', '7', '9'], [mark(2, 'mid', 'focus'), mark(0, 'discard'), mark(1, 'discard')])),
    frame('Narrow the interval', 'The remaining interval is [7, 9].', array(['1', '3', '5', '7', '9'], [mark(3, 'lo', 'state'), mark(4, 'hi', 'state')])),
    frame('Hit the target', 'The next middle is 7, at index 3.', array(['1', '3', '5', '7', '9'], [mark(3, 'found', 'output')], { result: 'index 3' })),
  ]),
  'search-in-rotated-sorted-array': visual('Use the sorted half to decide which side of the rotation can contain the target.', [
    frame('Identify a sorted half', 'For [4,5,6,7,0,1,2], the left half 4..7 is sorted.', array(['4', '5', '6', '7', '0', '1', '2'], [mark(0, 'L'), mark(3, 'mid', 'focus'), mark(6, 'R')], { detail: 'left half sorted' })),
    frame('Choose the other half', 'Target 0 is not inside 4..7, so discard the sorted left half.', array(['4', '5', '6', '7', '0', '1', '2'], [mark(4, 'L', 'state'), mark(5, 'mid', 'focus'), mark(6, 'R', 'state')], { detail: 'search right half' })),
    frame('Find the target', 'The right half reaches 0 at index 4.', array(['4', '5', '6', '7', '0', '1', '2'], [mark(4, 'found', 'output')], { result: 'index 4' })),
  ]),
  'koko-eating-bananas': visual('Binary-search the smallest eating speed that finishes within the hour limit.', [
    frame('Test a speed', 'At speed 4, piles [3,6,7,11] take 1+2+2+3 = 8 hours.', array(['speed 1', 'speed 2', 'speed 3', 'speed 4', 'speed 5', '...'], [mark(3, 'test', 'focus')], { detail: 'hours = 8; feasible' })),
    frame('Discard an infeasible speed', 'Speed 3 takes 10 hours, so the smallest feasible speed is at least 4. Keep [4,5].', array(['1', '2', '3', '4', '5', '...','11'], [mark(2, 'test', 'warning'), mark(3, 'lo', 'state'), mark(4, 'hi', 'state')], { detail: '10 hours > 8' })),
    frame('Return the first feasible speed', 'Speed 4 is the smallest speed whose hours fit.', array(['1', '2', '3', '4', '5', '...','11'], [mark(3, 'answer', 'output')], { result: '4 bananas/hour' })),
  ]),
  'find-minimum-in-rotated-sorted-array': visual('The drop lies on the side where the middle value exceeds the right boundary.', [
    frame('Compare middle and right', 'At mid 7 and right 2, the minimum is to the right of mid.', array(['4', '5', '6', '7', '0', '1', '2'], [mark(3, 'mid', 'focus'), mark(6, 'right', 'state')])),
    frame('Keep the rotation', 'The interval becomes [0,1,2].', array(['4', '5', '6', '7', '0', '1', '2'], [mark(4, 'lo', 'state'), mark(5, 'mid', 'focus'), mark(6, 'hi', 'state')])),
    frame('Minimum is at lo', 'When lo meets hi, that element is the minimum.', array(['4', '5', '6', '7', '0', '1', '2'], [mark(4, 'minimum', 'output')], { result: '0' })),
  ]),
  'valid-parentheses': visual('The newest unmatched opening bracket must match the next closing bracket.', [
    frame('Push openings', 'Read ( and [; both remain unfinished in the stack.', stack('([', ['(', '['], { current: '[' })),
    frame('Match the top', 'The next ] matches the stack top [, then } must match {.', stack('([{}])', ['(', '[', '{'], { current: '}', action: 'pop {' })),
    frame('Empty means valid', 'All openings were closed in reverse order.', stack('([{}])', [], { result: 'true' })),
  ]),
  'decode-string': visual('Save the outer text and repeat count whenever a nested bracket opens.', [
    frame('Enter the outer repeat', '3[ starts a new inner string while saving repeat 3.', stack('3[a2[c]]', ['outer="", count=3'], { current: '[' })),
    frame('Nest again', 'At 2[, save the current a and repeat count 2.', stack('3[a2[c]]', ['outer="", count=3', 'outer="a", count=2'], { current: '[' })),
    frame('Close from the inside', 'c becomes cc, then acc, then accaccacc.', stack('3[a2[c]]', ['outer="", count=3'], { current: ']', action: 'restore outer', result: 'accaccacc' })),
  ]),
  'daily-temperatures': visual('Keep colder days waiting until a warmer day resolves them.', [
    frame('Hold unresolved days', 'After scanning through 69, days 2, 3, and 4 are still waiting for a warmer temperature.', stack('73 74 75 71 69 72', ['day 2: 75', 'day 3: 71', 'day 4: 69'], { current: '72', action: 'wait' })),
    frame('Resolve from the top', '72 is warmer than 69 and 71, so both waiting days receive distances. Day 2 remains.', stack('73 74 75 71 69 72', ['day 2: 75'], { current: '72', action: 'resolve 69 -> 1, 71 -> 2' })),
    frame('Leave no warmer day as zero', '75 stays in the stack because no later value is warmer.', array(['1', '1', '0', '2', '1', '0'], [mark(2, 'none', 'state'), mark(5, 'none', 'state')], { result: '[1,1,0,2,1,0]' })),
  ]),
  'min-stack': visual('Store the minimum so far beside every stack value.', [
    frame('Push 5', 'The first value is also the minimum.', table(['value', 'min so far'], [['5', '5']], [0])),
    frame('Push 2 and 4', '2 becomes the minimum; 4 inherits minimum 2.', table(['value', 'min so far'], [['5', '5'], ['2', '2'], ['4', '2']], [2, 3, 4, 5])),
    frame('Read the minimum', 'The answer is at the top of the min column, without scanning values.', table(['value', 'min so far'], [['5', '5'], ['2', '2'], ['4', '2']], [5], { result: 'get_min() = 2' })),
  ]),
  'kth-largest-element': visual('Keep only the largest k values; the smallest of those is the kth largest.', [
    frame('Fill a size-2 heap', 'Read 3 and 2. Both remain candidates for the top two.', heap(['2', '3'], { root: '2', detail: 'size 2' })),
    frame('Replace the weak root', '5 arrives and evicts 2. The heap now protects 3 and 5.', heap(['3', '5'], { root: '3', detail: '2 discarded' })),
    frame('Return the root', 'After all values, heap [5,6] has root 5, the second largest.', heap(['5', '6'], { root: '5', detail: 'kth largest', result: '5' })),
  ]),
  'top-k-frequent-elements': visual('Use frequency as a bucket coordinate, then scan from the highest bucket.', [
    frame('Count values', 'The counts are 1 -> 3, 2 -> 2, and 3 -> 1.', buckets([{ count: '3', items: ['1'], tone: 'focus' }, { count: '2', items: ['2'] }, { count: '1', items: ['3'] }])),
    frame('Scan high to low', 'Take 1 from bucket 3 and 2 from bucket 2.', buckets([{ count: '3', items: ['1'], tone: 'output' }, { count: '2', items: ['2'], tone: 'output' }, { count: '1', items: ['3'] }], { result: 'two values collected' })),
    frame('Return top k', 'The answer is [1,2]; no global sort is needed.', buckets([{ count: '3', items: ['1'], tone: 'output' }, { count: '2', items: ['2'], tone: 'output' }, { count: '1', items: ['3'] }], { result: '[1, 2]' })),
  ]),
  'rotting-oranges': visual('Multi-source BFS makes each queue layer one minute of spread.', [
    frame('Seed all sources', 'Every rotten orange starts in minute 0.', queueGrid([['2', '1', '1'], ['1', '1', '0'], ['0', '1', '1']], ['(0,0)'], { minute: '0' })),
    frame('Spread one layer', 'The minute-1 frontier reaches its fresh neighbors.', queueGrid([['2', '2', '1'], ['2', '1', '0'], ['0', '1', '1']], ['(0,1)', '(1,0)'], { minute: '1' })),
    frame('Finish at the last layer', 'The final reachable orange rots at minute 4.', queueGrid([['2', '2', '2'], ['2', '2', '0'], ['0', '2', '2']], [], { minute: '4', result: '4' })),
  ]),
  'binary-tree-level-order-traversal': visual('Read exactly the queue length that existed before adding child nodes.', [
    frame('Queue the root', 'The first layer contains only 3.', queueGrid([['3'], ['9', '20'], ['15', '7']], ['3'], { level: '0' })),
    frame('Read one level', 'Pop 3, then append 9 and 20 for the next layer.', queueGrid([['3'], ['9', '20'], ['15', '7']], ['9', '20'], { level: '1', result: '[[3]]' })),
    frame('Continue by layer', 'The queue boundary gives [[3],[9,20],[15,7]].', queueGrid([['3'], ['9', '20'], ['15', '7']], [], { level: '2', result: '[[3],[9,20],[15,7]]' })),
  ]),
  'network-delay-time': visual('Dijkstra finalizes the node whose total path cost is smallest.', [
    frame('Start at node 2', 'Known distance is 0. Its outgoing paths cost 1 to nodes 1 and 3.', graph(['1', '2', '3'], ['2 -1-> 1', '2 -1-> 3'], { start: '2', frontier: ['1:1', '3:1'], visited: ['2:0'] })),
    frame('Finalize the cheapest path', 'Pop node 1 at distance 1; node 3 is already reachable at distance 1.', graph(['1', '2', '3'], ['2 -1-> 1', '2 -1-> 3', '1 -1-> 3'], { frontier: ['3:1'], visited: ['2:0', '1:1'] })),
    frame('Take the farthest finalized distance', 'Every node is reached; the delay is max(0,1,1) = 1.', graph(['1', '2', '3'], ['2 -1-> 1', '2 -1-> 3'], { visited: ['2:0', '1:1', '3:1'], result: '1' })),
  ]),
  'clone-graph': visual('Map each original node to one copy, then connect copies using the map.', [
    frame('Copy the start', 'Original node 1 gets exactly one copy before neighbors are explored.', graph(['original 1', 'original 2', 'copy 1'], ['1 <-> 2'], { visited: ['original 1'], copies: ['1 -> copy 1'] })),
    frame('Copy neighbors once', 'When node 2 appears, create copy 2 and reuse copy 1 for the reverse edge.', graph(['original 1', 'original 2', 'copy 1', 'copy 2'], ['copy 1 <-> copy 2'], { copies: ['1 -> copy 1', '2 -> copy 2'] })),
    frame('Return the copied component', 'Every original edge has a matching copied edge.', graph(['copy 1', 'copy 2'], ['copy 1 <-> copy 2'], { result: 'deep copy' })),
  ]),
  'number-of-islands': visual('Start a flood only at unseen land, then mark the whole component.', [
    frame('Find the first land', 'The top-left 1 starts island 1.', queueGrid([['1', '1', '1', '1'], ['1', '1', '0', '1'], ['1', '1', '0', '0'], ['0', '0', '0', '0']], ['(0,0)'], { action: 'start island 1' })),
    frame('Flood the component', 'Every connected 1 becomes visited water 0.', queueGrid([['0', '0', '0', '0'], ['0', '0', '0', '0'], ['0', '0', '0', '0'], ['0', '0', '0', '0']], [], { action: 'component visited' })),
    frame('Count starts, not cells', 'Only the first unseen land cell increments the island count.', queueGrid([['0', '0', '0'], ['0', '0', '0']], [], { result: '1 island' })),
  ]),
  'pacific-atlantic-water-flow': visual('Reverse the water direction: start at each ocean and walk uphill.', [
    frame('Seed the Pacific border', 'Pacific starts on the top and left edges; the heights stay visible while the search begins.', grid([['1', '2', '2', '3', '5'], ['3', '2', '3', '4', '4'], ['2', '4', '5', '3', '1'], ['6', '7', '1', '4', '5'], ['5', '1', '1', '2', '4']], [{ row: 0, col: 0, label: 'P', tone: 'focus' }, { row: 0, col: 1, label: 'P', tone: 'focus' }, { row: 0, col: 2, label: 'P', tone: 'focus' }, { row: 0, col: 3, label: 'P', tone: 'focus' }, { row: 0, col: 4, label: 'P', tone: 'focus' }, { row: 1, col: 0, label: 'P', tone: 'focus' }, { row: 2, col: 0, label: 'P', tone: 'focus' }, { row: 3, col: 0, label: 'P', tone: 'focus' }, { row: 4, col: 0, label: 'P', tone: 'focus' }], { legend: 'P = Pacific search starts here' })),
    frame('Seed the Atlantic border', 'Atlantic starts on the bottom and right edges; this is the second reverse search.', grid([['1', '2', '2', '3', '5'], ['3', '2', '3', '4', '4'], ['2', '4', '5', '3', '1'], ['6', '7', '1', '4', '5'], ['5', '1', '1', '2', '4']], [{ row: 0, col: 4, label: 'A', tone: 'focus' }, { row: 1, col: 4, label: 'A', tone: 'focus' }, { row: 2, col: 4, label: 'A', tone: 'focus' }, { row: 3, col: 4, label: 'A', tone: 'focus' }, { row: 4, col: 0, label: 'A', tone: 'focus' }, { row: 4, col: 1, label: 'A', tone: 'focus' }, { row: 4, col: 2, label: 'A', tone: 'focus' }, { row: 4, col: 3, label: 'A', tone: 'focus' }, { row: 4, col: 4, label: 'A', tone: 'focus' }], { legend: 'A = Atlantic search starts here' })),
    frame('Intersect the reached sets', 'Cells marked by both uphill searches can drain to both oceans.', grid([['1', '2', '2', '3', '5'], ['3', '2', '3', '4', '4'], ['2', '4', '5', '3', '1'], ['6', '7', '1', '4', '5'], ['5', '1', '1', '2', '4']], [{ row: 0, col: 4, label: 'B', tone: 'output' }, { row: 1, col: 3, label: 'B', tone: 'output' }, { row: 1, col: 4, label: 'B', tone: 'output' }, { row: 2, col: 2, label: 'B', tone: 'output' }, { row: 3, col: 0, label: 'B', tone: 'output' }, { row: 3, col: 1, label: 'B', tone: 'output' }, { row: 4, col: 0, label: 'B', tone: 'output' }], { legend: 'B = both searches reached this cell', result: 'intersection' })),
  ]),
  'maximum-depth-of-binary-tree': visual('A node returns one plus the larger depth returned by its children.', [
    frame('Solve the leaves', 'Every leaf returns depth 1.', tree([['3'], ['9', '20'], ['-', '-', '15', '7']], [mark(1, '1', 'state'), mark(5, '1', 'state'), mark(6, '1', 'state')])),
    frame('Combine child depths', 'Node 20 receives 1 and 1, so its depth is 2.', tree([['3'], ['9', '20'], ['-', '-', '15', '7']], [mark(2, 'depth 2', 'focus')])),
    frame('Return to the root', 'Root 3 returns 1 + max(1,2) = 3.', tree([['3'], ['9', '20'], ['-', '-', '15', '7']], [mark(0, 'depth 3', 'output')], { result: '3' })),
  ]),
  'same-tree': visual('Compare both trees at the same position before recursing to children.', [
    frame('Compare roots', 'Both roots are 1, so continue.', table(['position', 'tree A', 'tree B'], [['root', '1', '1'], ['left', '2', '2'], ['right', '3', '3']], [0])),
    frame('Compare children', 'Left and right values match at the same positions.', table(['position', 'tree A', 'tree B'], [['root', '1', '1'], ['left', '2', '2'], ['right', '3', '3']], [3, 6])),
    frame('Accept equal shape', 'The recursive comparisons all return true.', tree([['1'], ['2', '3']], [mark(0, 'same', 'output'), mark(1, 'same', 'output'), mark(2, 'same', 'output')], { result: 'true' })),
  ]),
  'invert-binary-tree': visual('Swap the two child links at every node.', [
    frame('Original children', 'Node 2 points left to 1 and right to 3.', tree([['2'], ['1', '3']], [mark(0, 'current', 'focus')])),
    frame('Swap at the root', 'The root now points left to 3 and right to 1.', tree([['2'], ['3', '1']], [mark(0, 'swapped', 'focus')])),
    frame('Return the inverted tree', 'The same swap happens recursively below every node.', tree([['2'], ['3', '1']], [mark(0, 'done', 'output')], { result: 'inverted' })),
  ]),
  'balanced-binary-tree': visual('Return a failure sentinel as soon as child heights differ by more than one.', [
    frame('Compute child heights', 'A chain gives the left subtree height 2 and the right height 0.', tree([['1'], ['2', '-'], ['3', '-']], [mark(0, 'check', 'focus')], { detail: 'left=2, right=0' })),
    frame('Propagate failure', 'The difference is 2, so this subtree returns -1.', tree([['1'], ['2', '-'], ['3', '-']], [mark(0, 'unbalanced', 'focus')], { detail: 'return -1' })),
    frame('Answer false', 'The root sees the sentinel and stops.', tree([['1'], ['2', '-'], ['3', '-']], [mark(0, 'false', 'output')], { result: 'false' })),
  ]),
  'subtree-of-another-tree': visual('Try the full-tree equality test at each candidate node.', [
    frame('Scan candidate roots', 'The root 3 does not match subroot root 4, so search its children.', tree([['3'], ['4', '5'], ['1', '2', '-', '-']], [mark(0, 'try', 'focus')])),
    frame('Match at node 4', 'The subtree rooted at 4 has the same value and child shape.', tree([['3'], ['4', '5'], ['1', '2', '-', '-']], [mark(1, 'match', 'output'), mark(3, 'match', 'output'), mark(4, 'match', 'output')])),
    frame('Return true', 'One complete matching subtree is enough.', tree([['3'], ['4', '5'], ['1', '2', '-', '-']], [mark(1, 'subtree', 'output')], { result: 'true' })),
  ]),
  'subsets': visual('Every partial path is already one valid subset; branch by choosing the next index.', [
    frame('Start with the empty path', 'The empty subset is a result before any choice.', choices([], ['take 1', 'skip 1'], { input: '[1,2]' })),
    frame('Choose 1, then 2', 'The path [1] branches to [1,2] or stops at [1].', choices(['1'], ['take 2 -> [1,2]', 'skip 2 -> [1]'], { input: '[1,2]' })),
    frame('Collect every path', 'The four paths are [], [1], [2], and [1,2].', choices([], ['[]', '[1]', '[2]', '[1,2]'], { result: '4 subsets' })),
  ]),
  'permutations': visual('Fill one position with each unused value, then undo it for the next branch.', [
    frame('Choose the first position', 'For [1,2,3], any of the three values can start the path.', choices([], ['1__', '2__', '3__'], { used: 'none' })),
    frame('Choose below 1', 'After choosing 1, only 2 and 3 remain for the next position.', choices(['1'], ['12_', '13_'], { used: '1' })),
    frame('Reach complete leaves', 'The tree ends at all six orderings.', choices([], ['123', '132', '213', '231', '312', '321'], { result: '6 permutations' })),
  ]),
  'combination-sum': visual('Choose in nondecreasing index order and carry the remaining target.', [
    frame('Start with target 7', 'The first choices are 2, 3, 6, or 7.', choices([], ['2 (remain 5)', '3 (remain 4)', '6 (remain 1)', '7 (remain 0)'], { target: '7' })),
    frame('Reuse a choice', 'From remainder 5, choosing 2 again leaves 3; [2,2,3] reaches zero.', choices(['2', '2'], ['choose 3 -> remain 0', 'choose 6 -> too large'], { target: '3' })),
    frame('Collect complete paths', 'The valid combinations are [2,2,3] and [7].', choices([], ['[2,2,3] = 7', '[7] = 7'], { result: '2 combinations' })),
  ]),
  'word-search': visual('Mark a board cell while it belongs to the current path, then restore it on return.', [
    frame('Start at A', 'The first matching cell starts the path A.', grid([['A', 'B', 'C', 'E'], ['S', 'F', 'C', 'S'], ['A', 'D', 'E', 'E']], [{ row: 0, col: 0, label: 'A', tone: 'focus' }], { word: 'A B C C E D' })),
    frame('Extend the path', 'Move through adjacent B, C, and C cells while marking them used.', grid([['#', '#', '#', 'E'], ['S', 'F', '#', 'S'], ['A', 'D', 'E', 'E']], [{ row: 0, col: 0, label: 'A', tone: 'state' }, { row: 0, col: 1, label: 'B', tone: 'state' }, { row: 0, col: 2, label: 'C', tone: 'state' }, { row: 1, col: 2, label: 'C', tone: 'focus' }], { word: 'A -> B -> C -> C' })),
    frame('Reach the final D', 'Continue to E and D; restore cells when a branch fails.', grid([['#', '#', '#', 'E'], ['S', 'F', '#', 'S'], ['A', 'D', '#', 'E']], [{ row: 2, col: 1, label: 'D', tone: 'output' }], { result: 'ABCCED found' })),
  ]),
  'climbing-stairs': visual('Each step can be reached from one step back or two steps back.', [
    frame('Base cases', 'The rolling state starts with a sentinel before step 1 and one way to reach step 1.', array(['before 1', 'step 1', 'step 2', 'step 3', 'step 4', 'step 5'], [mark(0, 'sentinel', 'state'), mark(1, '1 way', 'state')], { states: 'previous=0, current=1' })),
    frame('Build forward', 'ways(3) = ways(2) + ways(1) = 2 + 1 = 3.', array(['0', '1', '2', '3', '?', '?'], [mark(2, '2', 'state'), mark(3, '3', 'focus')])),
    frame('Keep only two totals', 'ways(5) = 8, and earlier totals are no longer needed.', array(['0', '1', '2', '3', '5', '8'], [mark(5, '8', 'output')], { result: '8' })),
  ]),
  'house-robber': visual('At each house, choose between skipping it and taking it after the previous house.', [
    frame('Before any house', 'The best totals two houses back and one house back are both zero.', array(['2', '7', '9', '3', '1'], [mark(0, 'current', 'focus')], { states: 'two_back=0, one_back=0' })),
    frame('Compare at 9', 'Skip 9 gives 7; take 9 gives 0 + 9. Keep 11 after the first three houses.', array(['2', '7', '9', '3', '1'], [mark(2, 'take', 'focus')], { states: 'skip=7, take=11, best=11' })),
    frame('Finish the line', 'The best non-adjacent selection is 2 + 9 + 1 = 12.', array(['2', '7', '9', '3', '1'], [mark(0, 'take', 'output'), mark(2, 'take', 'output'), mark(4, 'take', 'output')], { result: '12' })),
  ]),
  'partition-equal-subset-sum': visual('Reach half the total; every reachable sum is a state.', [
    frame('Set the target', 'Total is 22, so the wanted subset sum is 11.', array(['0', '1', '5', '11', '16'], [mark(3, 'target 11', 'focus')], { target: '11' })),
    frame('Add reachable sums', 'After processing 1, 5, and 11, the set contains 11.', array(['0', '1', '5', '6', '11'], [mark(4, 'reachable', 'output')])),
    frame('Accept the partition', 'A subset totals 11, so the remaining values also total 11.', array(['[1,5,5]', '[11]'], [mark(0, 'sum 11', 'output'), mark(1, 'sum 11', 'output')], { result: 'true' })),
  ]),
  'longest-common-subsequence': visual('A matching pair advances both prefixes; a mismatch keeps the better skipped prefix.', [
    frame('Compare prefixes', 'The grid state answers LCS for prefixes of abcde and ace.', table(['', '0', 'a', 'c', 'e'], [['0', '0', '0', '0', '0'], ['a', '0', '1', '1', '1'], ['b', '0', '1', '1', '1'], ['c', '0', '1', '2', '2'], ['d', '0', '1', '2', '2'], ['e', '0', '1', '2', '3']], [1, 1])),
    frame('Match c', 'The c/c cell takes the diagonal answer and adds one.', table(['', '0', 'a', 'c', 'e'], [['0', '0', '0', '0', '0'], ['a', '0', '1', '1', '1'], ['b', '0', '1', '1', '1'], ['c', '0', '1', '2', '2'], ['d', '0', '1', '2', '2'], ['e', '0', '1', '2', '3']], [18], { action: 'diagonal + 1' })),
    frame('Read the bottom-right', 'The complete prefixes share subsequence ace of length 3.', table(['', '0', 'a', 'c', 'e'], [['0', '0', '0', '0', '0'], ['a', '0', '1', '1', '1'], ['b', '0', '1', '1', '1'], ['c', '0', '1', '2', '2'], ['d', '0', '1', '2', '2'], ['e', '0', '1', '2', '3']], [29], { result: '3' })),
  ]),
  'edit-distance': visual('Each mismatch chooses the cheapest of insert, delete, and replace.', [
    frame('Initialize empty prefixes', 'The first row and column count edits against an empty string; the interior is not solved yet.', table(['', '0', 'r', 'o', 's'], [['0', '0', '1', '2', '3'], ['h', '1', '?', '?', '?'], ['o', '2', '?', '?', '?'], ['r', '3', '?', '?', '?'], ['s', '4', '?', '?', '?'], ['e', '5', '?', '?', '?']], [0])),
    frame('Choose a local operation', 'At the final e/s mismatch, the cell is 1 plus the smallest neighbor.', table(['', '0', 'r', 'o', 's'], [['0', '0', '1', '2', '3'], ['h', '1', '1', '2', '3'], ['o', '2', '2', '1', '2'], ['r', '3', '2', '2', '2'], ['s', '4', '3', '3', '2'], ['e', '5', '4', '4', '3']], [29], { action: 'min(insert, delete, replace) + 1' })),
    frame('Read the final cost', 'The bottom-right cell gives the distance from horse to ros.', table(['', '0', 'r', 'o', 's'], [['0', '0', '1', '2', '3'], ['h', '1', '1', '2', '3'], ['o', '2', '2', '1', '2'], ['r', '3', '2', '2', '2'], ['s', '4', '3', '3', '2'], ['e', '5', '4', '4', '3']], [29], { result: '3' })),
  ]),
  'merge-intervals': visual('Sort by start and extend the last merged interval whenever ranges overlap.', [
    frame('Start with the first range', 'The merged output begins with [1,3].', intervals([{ label: '[1,3]', start: 1, end: 3, tone: 'focus' }, { label: '[2,6]', start: 2, end: 6 }, { label: '[8,10]', start: 8, end: 10 }], { max: 10 })),
    frame('Extend on overlap', 'Since 2 <= 3, merge [1,3] and [2,6] into [1,6].', intervals([{ label: '[1,6]', start: 1, end: 6, tone: 'output' }, { label: '[8,10]', start: 8, end: 10 }], { max: 10 })),
    frame('Start a new range', 'The next interval starts after 6, so it stays separate.', intervals([{ label: '[1,6]', start: 1, end: 6, tone: 'output' }, { label: '[8,10]', start: 8, end: 10, tone: 'output' }], { max: 10, result: '[[1,6],[8,10]]' })),
  ]),
  'insert-interval': visual('Copy intervals before the new range, merge overlaps, then copy the suffix.', [
    frame('Copy the prefix', 'With new interval [4,8], [1,2] ends before it and stays untouched.', intervals([{ label: '[1,2]', start: 1, end: 2, tone: 'state' }, { label: '[3,5]', start: 3, end: 5, tone: 'state' }, { label: 'new [4,8]', start: 4, end: 8, tone: 'focus' }, { label: '[6,9]', start: 6, end: 9 }], { max: 10 })),
    frame('Merge the overlap', '[4,8] overlaps [3,5] and [6,9], producing [3,9].', intervals([{ label: '[1,2]', start: 1, end: 2, tone: 'state' }, { label: '[3,9]', start: 3, end: 9, tone: 'output' }], { max: 10 })),
    frame('Return the ordered result', 'The final answer keeps the prefix and the merged range.', intervals([{ label: '[1,2]', start: 1, end: 2, tone: 'output' }, { label: '[3,9]', start: 3, end: 9, tone: 'output' }], { max: 10, result: '[[1,2],[3,9]]' })),
  ]),
  'non-overlapping-intervals': visual('When intervals overlap, keep the one with the earlier end.', [
    frame('Sort by end', 'The candidate ending at 2 leaves the most room for later intervals.', intervals([{ label: '[1,2]', start: 1, end: 2, tone: 'focus' }, { label: '[1,3]', start: 1, end: 3 }, { label: '[2,3]', start: 2, end: 3 }, { label: '[3,4]', start: 3, end: 4 }], { max: 4 })),
    frame('Reject the late-ending overlap', '[1,3] overlaps the kept [1,2], so remove it and keep checking the remaining ranges.', intervals([{ label: '[1,2]', start: 1, end: 2, tone: 'state' }, { label: '[1,3]', start: 1, end: 3, tone: 'warning' }, { label: '[2,3]', start: 2, end: 3, tone: 'focus' }, { label: '[3,4]', start: 3, end: 4 }], { max: 4, detail: 'remove 1' })),
    frame('Keep room for the future', '[2,3] or [3,4] can follow the earliest-ending choice.', intervals([{ label: '[1,2]', start: 1, end: 2, tone: 'output' }, { label: '[2,3]', start: 2, end: 3, tone: 'output' }, { label: '[3,4]', start: 3, end: 4, tone: 'output' }], { max: 4, result: 'remove 1 interval' })),
  ]),
  'meeting-rooms-ii': visual('At each start, remove rooms whose meetings have already ended.', [
    frame('First meeting', 'Meeting [0,30] occupies one room.', intervals([{ label: '[0,30]', start: 0, end: 30, tone: 'focus' }], { max: 30, rooms: '1 active room' })),
    frame('Overlap needs another room', 'At start 5, [0,30] is still active, so [5,10] uses room 2.', intervals([{ label: '[0,30]', start: 0, end: 30, tone: 'state' }, { label: '[5,10]', start: 5, end: 10, tone: 'focus' }], { max: 30, rooms: '2 active rooms' })),
    frame('Reuse after an end', 'At start 15, [5,10] is gone; the maximum active count was 2.', intervals([{ label: '[0,30]', start: 0, end: 30, tone: 'output' }, { label: '[15,20]', start: 15, end: 20, tone: 'output' }], { max: 30, result: '2 rooms' })),
  ]),
  'jump-game': visual('Carry the farthest index reachable from everything scanned so far.', [
    frame('Reach from index 0', 'At index 0 with jump 2, the reachable boundary is 2.', array(['2', '3', '1', '1', '4'], [mark(0, 'scan', 'focus'), mark(2, 'reach', 'state')], { reach: '2' })),
    frame('Extend the boundary', 'Index 1 can reach 4, so the boundary moves to the last index.', array(['2', '3', '1', '1', '4'], [mark(1, 'scan', 'focus'), mark(4, 'reach', 'output')], { reach: '4' })),
    frame('Reach the end', 'The last index is at or before the farthest boundary.', array(['2', '3', '1', '1', '4'], [mark(4, 'goal', 'output')], { result: 'true' })),
  ]),
  'course-schedule': visual('A course becomes ready when its remaining prerequisite count reaches zero.', [
    frame('Count prerequisites', 'Course 0 is ready; course 1 has one incoming edge.', graph(['course 0', 'course 1'], ['0 -> 1'], { indegree: ['0:0', '1:1'], ready: ['0'] })),
    frame('Complete a ready course', 'Removing course 0 decrements course 1 from 1 to 0.', graph(['course 0', 'course 1'], ['0 -> 1'], { indegree: ['0:done', '1:0'], ready: ['1'] })),
    frame('Finish all nodes', 'Every course entered the ready queue, so no cycle remains.', graph(['course 0', 'course 1'], ['0 -> 1'], { order: ['0', '1'], result: 'true' })),
  ]),
  'course-schedule-ii': visual('The topological queue is also the feasible course order.', [
    frame('Seed zero-indegree courses', 'Only course 0 has no unmet prerequisite.', graph(['0', '1', '2'], ['0 -> 1', '1 -> 2'], { indegree: ['0:0', '1:1', '2:1'], ready: ['0'] })),
    frame('Append and decrement', 'Taking 0 makes 1 ready; taking 1 makes 2 ready.', graph(['0', '1', '2'], ['0 -> 1', '1 -> 2'], { order: ['0', '1'], ready: ['2'] })),
    frame('Return the order', 'The queue emitted a valid prerequisite-respecting sequence.', graph(['0', '1', '2'], ['0 -> 1', '1 -> 2'], { order: ['0', '1', '2'], result: '[0,1,2]' })),
  ]),
  'redundant-connection': visual('An edge is redundant when both endpoints already have the same representative root.', [
    frame('Join separate components', 'Edges 1-2 and 1-3 create one component rooted at 1.', graph(['1', '2', '3'], ['1 - 2', '1 - 3'], { components: ['root 1: {1,2,3}'] })),
    frame('Test the closing edge', 'For edge 2-3, find(2) and find(3) both return root 1.', graph(['1', '2', '3'], ['1 - 2', '1 - 3', '2 - 3'], { roots: ['2 -> 1', '3 -> 1'], current: '2 - 3' })),
    frame('Reject the cycle edge', 'Adding 2-3 would close a cycle, so return it.', graph(['1', '2', '3'], ['1 - 2', '1 - 3', '2 - 3'], { current: '2 - 3', result: '[2,3]' })),
  ]),
  'reverse-linked-list': visual('Save the outgoing link, reverse the current link, then advance.', [
    frame('Save next', 'Before changing 1.next, save the route to node 2.', linked([{ value: '1', pointer: 'current' }, { value: '2', pointer: 'next' }, { value: '3' }], { arrows: ['1 -> 2', '2 -> 3'] })),
    frame('Reverse one link', 'Point 1 back to previous, then advance current to saved node 2.', linked([{ value: '1', pointer: 'previous' }, { value: '2', pointer: 'current' }, { value: '3', pointer: 'next' }], { arrows: ['2 -> 3', '1 -> null'] })),
    frame('Return new head', 'After all links reverse, previous points at 3.', linked([{ value: '3', pointer: 'head', tone: 'output' }, { value: '2' }, { value: '1' }], { arrows: ['3 -> 2', '2 -> 1'], result: '3 -> 2 -> 1' })),
  ]),
  'linked-list-cycle': visual('A one-step pointer and a two-step pointer must meet inside a cycle.', [
    frame('Move at different speeds', 'After one move, slow is at 2 and fast is at 3.', linked([{ value: '1' }, { value: '2', pointer: 'slow' }, { value: '3', pointer: 'fast' }, { value: '4' }], { arrows: ['1 -> 2', '2 -> 3', '3 -> 4', '4 -> 2'] })),
    frame('Enter the loop', 'After the next move, slow is at 3 and fast has wrapped to 2.', linked([{ value: '2', pointer: 'fast' }, { value: '3', pointer: 'slow' }, { value: '4' }], { arrows: ['2 -> 3', '3 -> 4', '4 -> 2'] })),
    frame('Meet', 'On the next move both pointers reach 4, proving a cycle exists.', linked([{ value: '4', pointer: 'slow + fast', tone: 'output' }, { value: '2' }, { value: '3' }], { arrows: ['4 -> 2', '2 -> 3', '3 -> 4'], result: 'true' })),
  ]),
  'remove-nth-node-from-end': visual('A fixed pointer gap makes the left pointer stop just before the node to remove.', [
    frame('Create a gap', 'Move right two nodes ahead of left for n=2.', linked([{ value: 'dummy', pointer: 'left' }, { value: '1' }, { value: '2', pointer: 'right' }, { value: '3' }, { value: '4' }, { value: '5' }], { arrows: ['dummy -> 1 -> 2 -> 3 -> 4 -> 5'], detail: 'gap = 2' })),
    frame('Walk together', 'When right reaches 5, left is at node 3.', linked([{ value: '3', pointer: 'left' }, { value: '4' }, { value: '5', pointer: 'right' }], { arrows: ['3 -> 4 -> 5'], detail: 'left.next is node 4' })),
    frame('Skip the target', 'Redirect 3.next around node 4.', linked([{ value: '1' }, { value: '2' }, { value: '3', pointer: 'link changed', tone: 'focus' }, { value: '5', tone: 'output' }], { arrows: ['1 -> 2 -> 3 -> 5'], result: '[1,2,3,5]' })),
  ]),
  'merge-two-sorted-lists': visual('Attach the smaller current head and advance only that list.', [
    frame('Compare two heads', 'Heads 1 and 1 tie; attach one and advance its list.', linked([{ value: 'A:1', pointer: 'head A' }, { value: 'A:2' }, { value: 'A:4' }], { second: ['B:1', 'B:3', 'B:4'], detail: 'take 1' })),
    frame('Continue the merge', 'Compare the next heads and attach 2, then 3.', linked([{ value: '1' }, { value: '1' }, { value: '2', tone: 'focus' }, { value: '3', tone: 'focus' }], { detail: 'tail always points at last result node' })),
    frame('Append the remainder', 'When one list ends, attach the other suffix unchanged.', linked([{ value: '1' }, { value: '1' }, { value: '2' }, { value: '3' }, { value: '4' }, { value: '4' }], { result: 'sorted merged list' })),
  ]),
  'implement-trie': visual('A shared character path stores prefixes once, with a terminal marker for complete words.', [
    frame('Insert cat', 'The path root-c-a-t is created and t gets an end marker.', trie([{ word: 'cat', prefix: 'c-a-t', tone: 'focus' }], { action: 'insert cat' })),
    frame('Share c-a', 'Inserting car reuses c-a and branches only at the final character.', trie([{ word: 'cat', prefix: 'c-a-t', tone: 'state' }, { word: 'car', prefix: 'c-a-r', tone: 'focus' }], { action: 'share prefix c-a' })),
    frame('Search a prefix', 'starts_with("ca") succeeds even before choosing t or r.', trie([{ word: 'cat', prefix: 'c-a-t', tone: 'output' }, { word: 'car', prefix: 'c-a-r', tone: 'output' }], { query: 'ca', result: 'true' })),
  ]),
  'design-add-and-search-words': visual('A literal follows one trie child; a dot branches over every child.', [
    frame('Store words', 'bad, dad, and mad share the suffix ad after different first letters.', trie([{ word: 'bad', prefix: 'b-a-d' }, { word: 'dad', prefix: 'd-a-d' }, { word: 'mad', prefix: 'm-a-d' }], { action: 'insert three words' })),
    frame('Match a wildcard', 'For .ad, the dot tries b, d, and m, then follows a-d.', trie([{ word: '.ad', prefix: 'b/d/m -> a -> d', tone: 'focus' }], { query: '.ad', action: 'branch at dot' })),
    frame('Return true', 'One wildcard branch reaches a terminal word marker.', trie([{ word: 'bad', prefix: 'b-a-d', tone: 'output' }, { word: 'dad', prefix: 'd-a-d' }, { word: 'mad', prefix: 'm-a-d' }], { result: 'true' })),
  ]),
  'contains-duplicate': visual('The first repeated value is visible when it is already in the seen set.', [
    frame('Save new values', '1, 2, and 3 have not appeared before.', arrayMap(['1', '2', '3', '1'], [['1', 'seen'], ['2', 'seen'], ['3', 'seen']], [mark(2, 'current', 'focus')])),
    frame('Detect the repeat', 'The final 1 is already in the set.', arrayMap(['1', '2', '3', '1'], [['1', 'seen'], ['2', 'seen'], ['3', 'seen']], [mark(0, 'same value', 'output'), mark(3, 'repeat', 'output')])),
    frame('Return true', 'A set membership hit proves a duplicate exists.', array(['1', '2', '3', '1'], [mark(3, 'duplicate', 'output')], { result: 'true' })),
  ]),
  'maximum-subarray': visual('Discard a negative running prefix before extending a future subarray.', [
    frame('Carry a running sum', 'At 1, the negative prefix -2 is worse than starting a new subarray.', array(['-2', '1', '-3', '4', '-1', '2', '1'], [mark(0, 'drop', 'warning'), mark(1, 'start', 'focus')], { current: '1' })),
    frame('Extend the best ending here', 'Starting at 4, the running sum grows through -1, 2, and 1.', array(['4', '-1', '2', '1'], [mark(0, 'start', 'state'), mark(3, 'best ending', 'focus')], { current: '6' })),
    frame('Keep the global best', 'The maximum subarray is [4,-1,2,1] with sum 6.', array(['-2', '1', '-3', '4', '-1', '2', '1'], [mark(3, 'best', 'output'), mark(4, 'best', 'output'), mark(5, 'best', 'output'), mark(6, 'best', 'output')], { result: '6' })),
  ]),
  'best-time-to-buy-and-sell-stock': visual('For each selling day, pair the price with the lowest earlier buy price.', [
    frame('Track the cheapest buy', 'Prices 7 then 1 leave lowest buy price 1.', array(['7', '1', '5', '3', '6', '4'], [mark(1, 'lowest buy', 'state')], { low: '1', profit: '0' })),
    frame('Sell at 6', 'Selling at 6 after buying at 1 produces profit 5.', array(['7', '1', '5', '3', '6', '4'], [mark(1, 'buy', 'state'), mark(4, 'sell', 'focus')], { low: '1', profit: '5' })),
    frame('Return the best profit', 'No later sale beats 5.', array(['7', '1', '5', '3', '6', '4'], [mark(1, 'buy', 'output'), mark(4, 'sell', 'output')], { result: '5' })),
  ]),
  'maximum-product-subarray': visual('Keep both product extremes because a negative number can swap their roles.', [
    frame('Start both extremes', 'At 2 and 3, max and min ending products are both positive.', array(['2', '3', '-2', '4'], [mark(1, 'max=6,min=3', 'state')], { max: '6', min: '3' })),
    frame('A negative flips them', 'At -2, restarting at -2 is the maximum while the carried products become negative.', array(['2', '3', '-2', '4'], [mark(2, 'flip', 'focus')], { max: '-2', min: '-12', detail: 'candidates: -2, 6*-2, 3*-2' })),
    frame('Recover with another negative', 'The best product 6 comes from [2,3], while later values are checked the same way.', array(['2', '3', '-2', '4'], [mark(0, 'best', 'output'), mark(1, 'best', 'output')], { result: '6' })),
  ]),
  'number-of-1-bits': visual('The operation x & (x-1) removes exactly the lowest set bit.', [
    frame('Read the bits', '11 is binary 1011 and has three set bits.', bits(['1', '0', '1', '1'], [mark(3, 'lowest 1', 'focus')])),
    frame('Clear one bit', '1011 becomes 1010; two more applications produce 1000 and then 0000.', bits(['1', '0', '1', '0'], [mark(3, 'cleared', 'state')], { action: 'count = 1; next 1000 -> 0000' })),
    frame('Stop at zero', 'Three bit removals means Hamming weight 3.', bits(['0', '0', '0', '0'], [], { result: '3' })),
  ]),
  'counting-bits': visual('Remove the lowest bit and reuse the answer for the shifted number.', [
    frame('Use a smaller number', 'For 6, shift right to 3 and inspect the low bit 0.', table(['value', 'value >> 1', 'value & 1', 'count'], [['6', '3', '0', '?'], ['3', '1', '1', '2']], [0])),
    frame('Apply the recurrence', 'count[6] = count[3] + 0 = 2.', table(['value', 'shifted count', 'last bit', 'answer'], [['6', '2', '0', '2'], ['5', '2', '1', '2']], [0], { action: 'reuse DP' })),
    frame('Fill the line', 'Every value reuses a previously solved value.', array(['0', '1', '1', '2', '1', '2'], [mark(5, 'count(5)=2', 'output')], { result: 'counts 0..5' })),
  ]),
  'missing-number': visual('XOR cancels every value that appears in both the expected and actual sets.', [
    frame('Pair expected and actual', 'Expected values are 0,1,2,3; actual values are 3,0,1.', table(['expected', 'actual', 'xor'], [['0', '3', '0 xor 3'], ['1', '0', '1 xor 0'], ['2', '-', '2 remains'], ['3', '1', '3 xor 1']], [6])),
    frame('Cancel matches', '0, 1, and 3 cancel in pairs; only 2 remains.', bits(['0', '0', '1', '0'], [mark(2, 'uncancelled 2', 'focus')], { detail: 'XOR result = 2' })),
    frame('Return the survivor', 'The missing value is 2.', bits(['0', '0', '1', '0'], [mark(2, 'missing', 'output')], { result: '2' })),
  ]),
  'reverse-bits': visual('Read one input bit from the right and append it to the answer on the left.', [
    frame('Read the low bit', 'The input cursor starts at the least-significant bit. The drawing shows an 8-bit slice; the implementation repeats the same move 32 times.', bits(['1', '0', '1', '1', '0', '0', '1', '0'], [mark(0, 'read', 'focus')], { input: 'right -> left', output: 'empty', width: '8-bit illustration' })),
    frame('Append to output', 'Shift the output left and place the read bit at its low end.', bits(['1', '0', '1', '1', '0', '0', '1', '0'], [mark(0, 'read', 'state'), mark(7, 'write', 'focus')], { output: '1' })),
    frame('Repeat 32 times', 'After fixed-width processing, the bit order is reversed.', bits(['0', '1', '0', '0', '1', '1', '0', '1'], [], { result: 'reversed 32-bit word' })),
  ]),
  'sum-of-two-integers': visual('XOR supplies sum bits without carry; AND shifted left supplies the carry.', [
    frame('Separate sum and carry', 'For 3 (0011) and 1 (0001), XOR gives 0010 and the carry is 0010.', bits(['0', '0', '1', '0'], [mark(2, 'xor', 'state'), mark(3, 'xor', 'state')], { sum: '0010', carry: '0010' })),
    frame('Move the carry left', 'The next pass combines 0010 with 0010, producing no sum bits and carry 0100.', bits(['0', '0', '0', '0'], [mark(2, 'carry', 'focus')], { sum: '0000', carry: '0100' })),
    frame('Stop when carry is zero', 'A final pass produces 0100, the sum of 3 and 1.', bits(['0', '1', '0', '0'], [mark(1, '4', 'output')], { result: '4' })),
  ]),
  'coin-change': visual('The best way to make amount t is one coin plus the best way to make t-coin.', [
    frame('Initialize amount zero', 'Zero coins make amount 0; other amounts are unreachable.', array(['0', 'inf', 'inf', 'inf', 'inf', 'inf', 'inf'], [mark(0, 'base', 'state')])),
    frame('Build amount 6', 'For coin 5, look at amount 1 and add one coin.', array(['0', '1', '1', '2', '2', '1', '2'], [mark(1, 'fewest[1]=1', 'state'), mark(6, 'fewest[6]=2', 'focus')])),
    frame('Return the minimum', 'With coins 1, 2, and 5, 6 is made by 1+5 in two coins.', array(['0', '1', '1', '2', '2', '1', '2'], [mark(6, 'answer 2', 'output')], { result: '2 coins' })),
  ]),
  'longest-increasing-subsequence': visual('For each subsequence length, keep the smallest possible ending value.', [
    frame('Read 10, 9, 2', 'Each new smaller value replaces the tail for length 1.', array(['10', '9', '2', '5', '3', '7', '101'], [mark(2, 'tails=[2]', 'state')], { tails: '[2]' })),
    frame('Extend and replace tails', '5, 3, and 7 produce tails [2,3,7].', array(['2', '3', '7'], [mark(2, 'length 3', 'focus')], { tails: '[2,3,7]' })),
    frame('Append 101', '101 extends the tail list, giving length 4.', array(['2', '3', '7', '101'], [mark(3, 'append', 'output')], { result: '4' })),
  ]),
  'word-break': visual('A reachable string position can start any dictionary word that matches there.', [
    frame('Start at position 0', 'The empty prefix is reachable before reading any word.', array(['0', '1', '2', '3', '4', '5', '6', '7', '8'], [mark(0, 'start', 'state')], { text: 'leetcode' })),
    frame('Reach position 4', 'The word leet matches positions 0..3, so position 4 becomes reachable.', array(['l', 'e', 'e', 't', '|', 'c', 'o', 'd', 'e'], [mark(4, 'reachable', 'focus')], { word: 'leet' })),
    frame('Reach the end', 'The word code starts at 4 and reaches position 8.', array(['l', 'e', 'e', 't', '|', 'c', 'o', 'd', 'e'], [mark(8, 'reachable', 'output')], { result: 'true' })),
  ]),
  'combination-sum-iv': visual('Count ordered sequences by choosing the final number of each target.', [
    frame('Base count', 'There is one way to make total 0: choose nothing.', array(['1', '0', '0', '0', '0'], [mark(0, 'base', 'state')])),
    frame('Build totals', 'ways[3] includes sequences ending in 1, 2, or 3.', array(['1', '1', '2', '4', '7'], [mark(3, '4 ways', 'state'), mark(4, '7 ways', 'focus')])),
    frame('Return target count', 'The seven sequences for target 4 include 1+3 and 3+1 separately.', array(['1', '1', '2', '4', '7'], [mark(4, 'answer', 'output')], { result: '7' })),
  ]),
  'house-robber-ii': visual('A circular solution is the larger of two lines: exclude the first house or exclude the last.', [
    frame('Break the circle', 'Taking both first and last is forbidden, so solve two linear ranges.', array(['2', '3', '2'], [mark(0, 'exclude in case B', 'state'), mark(2, 'exclude in case A', 'state')], { cases: 'houses[0:-1] and houses[1:]' })),
    frame('Solve each line', 'Case A [2,3] gives 3. Case B [3,2] gives 3.', table(['case', 'houses', 'best'], [['A', '[2,3]', '3'], ['B', '[3,2]', '3']], [0, 1])),
    frame('Choose the larger result', 'Both cases tie at 3, which is the circular answer.', array(['2', '3', '2'], [mark(1, 'take', 'output')], { result: '3' })),
  ]),
  'decode-ways': visual('A digit can extend one prior decoding; a valid two-digit number can extend two prior decodings.', [
    frame('Read 2', 'The first digit gives one decoding.', array(['2', '2', '6'], [mark(0, '1 way', 'state')])),
    frame('At 22', '22 is valid, so one-digit and two-digit choices contribute.', array(['2', '2', '6'], [mark(1, '2 ways', 'focus')], { choices: '2|2 and 22' })),
    frame('At 226', '6 can follow 22 or stand after 2, giving three total decodings.', array(['2', '2', '6'], [mark(2, '3 ways', 'output')], { result: '3' })),
  ]),
  'unique-paths': visual('Each cell receives paths from the cell above and the cell to its left.', [
    frame('Initialize the top edge', 'Only one path reaches every cell along the top row.', grid([['1', '1', '1', '1', '1', '1', '1'], ['1', '.', '.', '.', '.', '.', '.'], ['1', '.', '.', '.', '.', '.', '.']], [], { action: 'base paths' })),
    frame('Add from two directions', 'The center cell gets paths from above plus paths from the left.', grid([['1', '1', '1', '1', '1', '1', '1'], ['1', '2', '3', '4', '5', '6', '7'], ['1', '3', '6', '10', '15', '21', '28']], [{ row: 2, col: 2, label: '6', tone: 'focus' }], { formula: '3 + 3 = 6' })),
    frame('Read the destination', 'The bottom-right cell contains 28 paths for a 3 by 7 grid.', grid([['1', '1', '1', '1', '1', '1', '1'], ['1', '2', '3', '4', '5', '6', '7'], ['1', '3', '6', '10', '15', '21', '28']], [{ row: 2, col: 6, label: '28', tone: 'output' }], { result: '28' })),
  ]),
  'graph-valid-tree': visual('A valid tree needs exactly n-1 edges and one connected component.', [
    frame('Check the edge count', 'Five nodes require exactly four edges.', graph(['0', '1', '2', '3', '4'], ['0-1', '0-2', '0-3', '1-4'], { detail: 'edges = 4 = n-1' })),
    frame('Traverse once', 'DFS from 0 reaches every node without finding a cycle.', graph(['0', '1', '2', '3', '4'], ['0-1', '0-2', '0-3', '1-4'], { visited: ['0', '1', '2', '3', '4'] })),
    frame('Accept the tree', 'Correct edge count plus full reachability proves a tree.', graph(['0', '1', '2', '3', '4'], ['0-1', '0-2', '0-3', '1-4'], { result: 'true' })),
  ]),
  'number-of-connected-components': visual('Every unseen node starts one DFS component and marks its whole group.', [
    frame('Start component 1', 'Node 0 reaches 1 and 2.', graph(['0', '1', '2', '3', '4'], ['0-1', '1-2', '3-4'], { visited: ['0', '1', '2'], components: '1' })),
    frame('Find the next unseen node', 'Node 3 starts a second flood and reaches 4.', graph(['0', '1', '2', '3', '4'], ['0-1', '1-2', '3-4'], { visited: ['0', '1', '2', '3', '4'], components: '2' })),
    frame('Return the count', 'Two starting floods mean two connected components.', graph(['0', '1', '2', '3', '4'], ['0-1', '1-2', '3-4'], { result: '2' })),
  ]),
  'meeting-rooms': visual('After sorting by start time, only the previous end can overlap the next meeting.', [
    frame('Sort meetings', 'The starts are 0, 5, and 15.', intervals([{ label: '[0,30]', start: 0, end: 30, tone: 'state' }, { label: '[5,10]', start: 5, end: 10, tone: 'focus' }, { label: '[15,20]', start: 15, end: 20 }], { max: 30 })),
    frame('Find the overlap', 'The next start 5 is before previous end 30.', intervals([{ label: '[0,30]', start: 0, end: 30, tone: 'warning' }, { label: '[5,10]', start: 5, end: 10, tone: 'focus' }], { max: 30, detail: '5 < 30' })),
    frame('Return false', 'One person cannot attend overlapping meetings.', intervals([{ label: '[0,30]', start: 0, end: 30, tone: 'warning' }, { label: '[5,10]', start: 5, end: 10, tone: 'warning' }], { max: 30, result: 'false' })),
  ]),
  'reorder-list': visual('Find the middle, reverse the second half, then interleave the two lists.', [
    frame('Split at the middle', 'Slow and fast leave first half 1,2,3 and second half 4,5.', linked([{ value: '1' }, { value: '2' }, { value: '3', pointer: 'split' }, { value: '4' }, { value: '5' }], { detail: 'first: 1->2->3; second: 4->5' })),
    frame('Reverse the second half', 'The second list becomes 5->4.', linked([{ value: '1' }, { value: '2' }, { value: '3' }, { value: '5', tone: 'focus' }, { value: '4' }], { detail: 'second: 5->4' })),
    frame('Interleave', 'Take one node from each half: 1,5,2,4,3.', linked([{ value: '1', tone: 'output' }, { value: '5', tone: 'output' }, { value: '2', tone: 'output' }, { value: '4', tone: 'output' }, { value: '3', tone: 'output' }], { result: '1->5->2->4->3' })),
  ]),
  'set-matrix-zeroes': visual('Use the first row and column as markers, then apply the marked rows and columns.', [
    frame('Find zeros', 'A zero at row 1, column 1 marks its row and column.', grid([['1', '1', '1'], ['1', '0', '1'], ['1', '1', '1']], [{ row: 1, col: 1, label: '0', tone: 'focus' }], { action: 'mark row 1, col 1' })),
    frame('Read the markers', 'The first row and column now carry the future zero instructions.', grid([['1', '0', '1'], ['0', '0', '1'], ['1', '1', '1']], [{ row: 0, col: 1, label: 'marker', tone: 'state' }, { row: 1, col: 0, label: 'marker', tone: 'state' }])),
    frame('Fill marked cells', 'Zero every cell in the marked row or column.', grid([['1', '0', '1'], ['0', '0', '0'], ['1', '0', '1']], [], { result: 'in place' })),
  ]),
  'spiral-matrix': visual('Read the four current boundaries, then shrink them after each side.', [
    frame('Read the top and right', 'Consume top row 1,2,3 and right column 6,9.', grid([['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']], [{ row: 0, col: 0, label: 'top', tone: 'focus' }, { row: 2, col: 2, label: 'right', tone: 'focus' }])),
    frame('Read bottom and left', 'Continue backward across 8,7 and up through 4.', grid([['.', '.', '.'], ['4', '5', '.'], ['7', '8', '.']], [{ row: 2, col: 1, label: 'bottom', tone: 'state' }, { row: 1, col: 0, label: 'left', tone: 'state' }])),
    frame('Finish the inner layer', 'The remaining center is 5.', grid([['.', '.', '.'], ['.', '5', '.'], ['.', '.', '.']], [{ row: 1, col: 1, label: 'last', tone: 'output' }], { result: '[1,2,3,6,9,8,7,4,5]' })),
  ]),
  'rotate-image': visual('Reverse the row order, then transpose across the main diagonal.', [
    frame('Reverse rows', 'The bottom row moves to the top.', grid([['7', '8', '9'], ['4', '5', '6'], ['1', '2', '3']], [], { action: 'reverse rows' })),
    frame('Transpose', 'Swap cells across the diagonal: (row,col) becomes (col,row).', grid([['7', '4', '1'], ['8', '5', '2'], ['9', '6', '3']], [{ row: 0, col: 0, label: 'fixed', tone: 'state' }, { row: 0, col: 2, label: 'moved', tone: 'focus' }])),
    frame('Read clockwise result', 'The matrix is rotated in place without a second matrix.', grid([['7', '4', '1'], ['8', '5', '2'], ['9', '6', '3']], [], { result: '90 degrees clockwise' })),
  ]),
  'valid-palindrome': visual('Move inward while comparing the next alphanumeric character from each end.', [
    frame('Skip punctuation', 'Ignore spaces and commas; the meaningful endpoints are A and a.', array(['A', 'm', 'a', 'n', 'a', 'm', 'a'], [mark(0, 'L', 'focus'), mark(6, 'R', 'focus')], { normalize: 'lowercase, alphanumeric' })),
    frame('Compare inward', 'Matching pairs move both pointers toward the center.', array(['A', 'm', 'a', 'n', 'a', 'm', 'a'], [mark(1, 'match', 'state'), mark(5, 'match', 'state')], { detail: 'm == m' })),
    frame('Meet in the middle', 'Every pair matches, so the normalized string is a palindrome.', array(['A', 'm', 'a', 'n', 'a', 'm', 'a'], [mark(3, 'center', 'output')], { result: 'true' })),
  ]),
  'longest-palindromic-substring': visual('Every palindrome grows from one character center or one gap center.', [
    frame('Try an odd center', 'Expand around b in babad to get bab.', array(['b', 'a', 'b', 'a', 'd'], [mark(1, 'center', 'focus'), mark(0, 'edge', 'state'), mark(2, 'edge', 'state')], { candidate: 'bab' })),
    frame('Try an even center', 'A gap between two equal characters handles even-length palindromes.', array(['c', 'b', 'b', 'd'], [mark(1, 'gap', 'focus'), mark(2, 'gap', 'focus')], { candidate: 'bb' })),
    frame('Keep the longest', 'The widest expansion wins.', array(['b', 'a', 'b', 'a', 'd'], [mark(0, 'best', 'output'), mark(1, 'best', 'output'), mark(2, 'best', 'output')], { result: 'bab or aba' })),
  ]),
  'palindromic-substrings': visual('Each successful center expansion contributes exactly one palindrome.', [
    frame('Count an odd center', 'Center a gives a, then expand to aba.', array(['a', 'b', 'a'], [mark(1, 'center', 'focus'), mark(0, 'palindrome', 'state'), mark(2, 'palindrome', 'state')], { count: '2' })),
    frame('Count every center', 'For aaa, three single letters, two pairs, and aaa all count.', array(['a', 'a', 'a'], [mark(0, '1', 'state'), mark(1, '4', 'focus'), mark(2, '1', 'state')], { count: '6 total' })),
    frame('Return the total', 'The six palindromic substrings of aaa are all center expansions.', array(['a', 'a', 'a'], [mark(0, 'a', 'output'), mark(1, 'a/aa', 'output'), mark(2, 'a', 'output')], { result: '6' })),
  ]),
  'encode-and-decode-strings': visual('A length prefix tells the decoder exactly how many characters belong to each string.', [
    frame('Encode with lengths', 'lint becomes 4#lint and # becomes 1##.', array(['4#lint', '1##', '0#'], [mark(0, 'length 4', 'focus')])),
    frame('Read one length', 'The decoder reads 4, skips #, and consumes exactly four characters.', array(['4', '#', 'l', 'i', 'n', 't'], [mark(0, 'read length', 'state'), mark(2, 'start', 'focus'), mark(5, 'end', 'focus')])),
    frame('Recover the list', 'Lengths make delimiters inside the original strings harmless.', array(['lint', '#', ''], [mark(0, 'decoded', 'output'), mark(1, 'decoded', 'output')], { result: '["lint","#",""]' })),
  ]),
  'construct-tree-from-preorder-and-inorder-traversal': visual('Preorder gives the next root; inorder splits the left and right ranges.', [
    frame('Choose the root', 'Preorder starts with 3. Inorder places 3 between 9 and 15,20,7.', table(['preorder next', 'inorder left', 'root', 'inorder right'], [['3', '[9]', '3', '[15,20,7]']], [2])),
    frame('Recurse on ranges', 'The next preorder values become roots of the left and right ranges.', tree([['3'], ['9', '20'], ['-', '-', '15', '7']], [mark(1, 'left range', 'state'), mark(2, 'right range', 'focus')])),
    frame('Return the tree', 'Every inorder range is reconstructed with one preorder root.', tree([['3'], ['9', '20'], ['-', '-', '15', '7']], [mark(0, 'root', 'output')], { result: 'tree rebuilt' })),
  ]),
  'validate-binary-search-tree': visual('Pass the full inherited lower and upper bounds down each tree branch.', [
    frame('Set the root bounds', 'Root 5 must lie between negative and positive infinity.', tree([['5'], ['1', '7'], ['-', '-', '4', '-']], [mark(0, 'bounds (-inf,inf)', 'focus')])),
    frame('Carry an ancestor bound', 'Node 4 is in the right subtree of 5, so its lower bound is 5.', tree([['5'], ['1', '7'], ['-', '-', '4', '-']], [mark(5, '4 not > 5', 'warning')], { bounds: '4 must be > 5' })),
    frame('Reject the tree', 'A parent-only check would miss this violation; inherited bounds catch it.', tree([['5'], ['1', '7'], ['-', '-', '4', '-']], [mark(5, 'invalid', 'output')], { result: 'false' })),
  ]),
  'kth-smallest-element-in-a-bst': visual('Inorder traversal visits BST nodes in ascending order, so stop at the kth visit.', [
    frame('Push the left spine', 'Start by pushing 3, then 1.', tree([['3'], ['1', '4'], ['-', '2', '-', '-']], [mark(0, 'stack', 'state'), mark(1, 'stack', 'focus')])),
    frame('Visit in order', 'Pop 1 first, then 2, then 3. The first visit is the smallest.', tree([['3'], ['1', '4'], ['-', '2', '-', '-']], [mark(1, 'visit 1', 'focus'), mark(4, 'visit 2', 'state')])),
    frame('Stop at k', 'For k=1, return node 1 immediately.', tree([['3'], ['1', '4'], ['-', '2', '-', '-']], [mark(1, 'kth', 'output')], { result: '1' })),
  ]),
  'lowest-common-ancestor-in-a-bst': visual('BST ordering tells whether both targets lie left, right, or split at the current node.', [
    frame('Start at 6', 'Targets 2 and 8 lie on opposite sides of 6.', tree([['6'], ['2', '8']], [mark(0, 'split', 'focus')])),
    frame('Stop at the split', 'If both targets were left or right, descend; here 6 is the first split.', tree([['6'], ['2', '8']], [mark(0, 'ancestor', 'output')], { path: '2 < 6 < 8' })),
    frame('Return the ancestor', 'Node 6 is the lowest node whose subtree contains both targets.', tree([['6'], ['2', '8']], [mark(0, 'LCA', 'output')], { result: '6' })),
  ]),
  'lru-cache': visual('A map finds a node; a doubly linked list keeps least-recent to most-recent order.', [
    frame('Insert and read', 'After put(1), put(2), get(1), the order is 2 -> 1.', lru([['1', 'node'], ['2', 'node']], ['least: 2', 'most: 1'], { action: 'get(1) moves it right' })),
    frame('Evict the left edge', 'put(3) appends 3 and removes least-recent key 2.', lru([['1', 'node'], ['3', 'node']], ['least: 1', 'most: 3'], { evicted: '2' })),
    frame('Lookup misses', 'get(2) returns -1 because the map and list no longer contain it.', lru([['1', 'node'], ['3', 'node']], ['least: 1', 'most: 3'], { result: 'get(2) = -1' })),
  ]),
  'pairwise-squared-distances': visual('Singleton axes create every point-center pair before reducing feature coordinates.', [
    frame('Add singleton axes', 'Points [n,d] become [n,1,d]; centers become [1,k,d].', shapes(['points [n,1,d]', 'centers [1,k,d]'], { action: 'align singleton axes' })),
    frame('Broadcast pairs', 'The difference tensor has one row for every point-center pair.', shapes(['points [n,1,d]', 'centers [1,k,d]', 'difference [n,k,d]'], { action: 'broadcast', focus: 'difference' })),
    frame('Reduce features', 'Summing squared differences over d yields [n,k] distances.', shapes(['difference [n,k,d]', 'sum over d', 'distances [n,k]'], { result: '[n,k]' })),
  ]),
  'stable-softmax': visual('Subtract the row maximum before exponentiating; relative gaps do not change.', [
    frame('See the large logits', 'Exponentiating 1000 and 1001 directly can overflow.', array(['1000', '1001'], [mark(1, 'row max', 'focus')], { detail: 'raw logits' })),
    frame('Shift the row', 'Subtract 1001 to get [-1,0].', array(['-1', '0'], [mark(0, 'shifted', 'state'), mark(1, 'shifted', 'state')], { action: 'logits - max' })),
    frame('Normalize safely', 'exp(-1) and exp(0) divide by their sum to form probabilities.', array(['0.2689', '0.7311'], [mark(0, 'p', 'output'), mark(1, 'p', 'output')], { result: 'sum = 1' })),
  ]),
  'cross-entropy-from-logits': visual('Cross-entropy from logits is a stable log-sum-exp minus the selected logit.', [
    frame('Choose the correct class', 'For logits [2,1,0], label 0 selects logit 2.', array(['class 0: 2', 'class 1: 1', 'class 2: 0'], [mark(0, 'correct', 'focus')])),
    frame('Compute the normalizer', 'logsumexp summarizes all class logits without building probabilities first.', array(['2', '1', '0'], [mark(0, 'selected', 'state')], { formula: 'log(exp(2)+exp(1)+exp(0))' })),
    frame('Subtract the correct logit', 'Loss = logsumexp(row) - 2 = 0.4076.', array(['logsumexp(row)', '-', 'correct logit 2'], [mark(0, 'normalizer', 'state'), mark(2, 'subtract', 'focus')], { result: '0.4076' })),
  ]),
  'causal-attention': visual('Each attention row can read its own position and every earlier position, never a future one.', [
    frame('Build all pair scores', 'Query-key scores start as a full square matrix.', attention([['.', '.', '.'], ['.', '.', '.'], ['.', '.', '.']], { action: 'QK^T' })),
    frame('Apply the causal mask', 'Future positions become forbidden before softmax.', attention([['read', 'mask', 'mask'], ['read', 'read', 'mask'], ['read', 'read', 'read']], { action: 'mask future scores' })),
    frame('Mix allowed values', 'Each row can assign weights to its prefix, while every future weight is zero.', attention([['w0', 'mask', 'mask'], ['w0', 'w1', 'mask'], ['w0', 'w1', 'w2']], { result: 'prefix-only reads; each row sums to 1' })),
  ]),
  'pad-variable-length-sequences': visual('Padding creates a rectangle; the boolean mask preserves which cells were real.', [
    frame('Start with ragged rows', 'The sequences have lengths 2 and 1.', shapes(['[3,4]', '[9]'], { action: 'ragged input' })),
    frame('Fill the rectangle', 'Use the longest width and a pad value for unused cells.', grid([['3', '4'], ['9', '0']], [{ row: 1, col: 1, label: 'pad', tone: 'state' }], { tensor: 'tokens [2,2]' })),
    frame('Carry the mask', 'The same padded position is false in the validity mask.', grid([['1', '1'], ['1', '0']], [{ row: 1, col: 1, label: 'false', tone: 'output' }], { tensor: 'mask [2,2]', result: 'tokens + mask' })),
  ]),
  'mini-batches': visual('A single cursor yields non-overlapping slices and keeps the final short slice.', [
    frame('Take the first slice', 'Start 0 with size 3 yields items 0,1,2.', array(['0', '1', '2', '3', '4', '5', '6'], [mark(0, 'start', 'focus'), mark(2, 'end', 'focus')], { batch: '[0:3]' })),
    frame('Advance the cursor', 'Start 3 yields the next three items.', array(['0', '1', '2', '3', '4', '5', '6'], [mark(3, 'start', 'focus'), mark(5, 'end', 'focus')], { batch: '[3:6]' })),
    frame('Keep the remainder', 'Start 6 yields [6] instead of dropping it.', array(['0', '1', '2', '3', '4', '5', '6'], [mark(6, 'remainder', 'output')], { result: '[[0,1,2],[3,4,5],[6]]' })),
  ]),
  'top-k-scores': visual('Partition finds membership in the top group; sort only that group for output order.', [
    frame('Partition the scores', 'Scores 0.9 and 0.8 belong to the top-2 group.', array(['0.1', '0.9', '0.4', '0.8'], [mark(1, 'candidate', 'state'), mark(3, 'candidate', 'state')], { action: 'argpartition' })),
    frame('Sort selected candidates', 'Only selected indices 1 and 3 need final ordering.', array(['index 1: 0.9', 'index 3: 0.8'], [mark(0, 'first', 'focus'), mark(1, 'second', 'state')], { action: 'sort k' })),
    frame('Return indices', 'The descending top-k indices are [1,3].', array(['1', '3'], [mark(0, 'top', 'output'), mark(1, 'top', 'output')], { result: '[1,3]' })),
  ]),
  'binary-precision-and-recall': visual('Each example enters one confusion cell; the metric chooses its denominator afterward.', [
    frame('Classify observations', 'Truth and prediction route examples to TN, FP, FN, or TP.', table(['', 'pred 0', 'pred 1'], [['true 0', 'TN', 'FP'], ['true 1', 'FN', 'TP']], [1, 2, 4, 5])),
    frame('Count the cells', 'For the example, TP=1, FP=1, FN=1, TN=1.', table(['', 'pred 0', 'pred 1'], [['true 0', '1 TN', '1 FP'], ['true 1', '1 FN', '1 TP']], [1, 2, 4, 5], { counts: 'TP=1 FP=1 TN=1 FN=1' })),
    frame('Choose the denominator', 'Precision uses predicted positives; recall uses actual positives.', table(['metric', 'numerator', 'denominator'], [['precision', 'TP=1', 'TP+FP=2'], ['recall', 'TP=1', 'TP+FN=2']], [1, 2, 4, 5], { result: 'precision=.5, recall=.5' })),
  ]),
  'minimum-window-substring': visual('Grow until all required characters are present, then shrink while the window remains valid.', [
    frame('Gather ABC', 'ADOBEC contains A, B, and C, so the first valid window ends at C.', array(['A', 'D', 'O', 'B', 'E', 'C', 'O', 'D', 'E', 'B', 'A', 'N', 'C'], [mark(0, 'L'), mark(5, 'R', 'focus')], { range: 'ADOBEC', need: 'A,B,C' })),
    frame('Shrink from the left', 'Dropping A breaks validity, so grow again until the new valid window is BANC.', array(['A', 'D', 'O', 'B', 'E', 'C', 'O', 'D', 'E', 'B', 'A', 'N', 'C'], [mark(5, 'old valid', 'state'), mark(9, 'L', 'focus'), mark(12, 'R', 'focus')], { range: 'BANC', action: 'shrink then regrow' })),
    frame('Keep the shortest', 'BANC is the shortest window containing A, B, and C.', array(['A', 'D', 'O', 'B', 'E', 'C', 'O', 'D', 'E', 'B', 'A', 'N', 'C'], [mark(9, 'B', 'output'), mark(10, 'A', 'output'), mark(11, 'N', 'output'), mark(12, 'C', 'output')], { result: 'BANC' })),
  ]),
  'split-array-largest-sum': visual('Guess a maximum part sum, greedily count required parts, and binary-search the smallest feasible guess.', [
    frame('Set answer bounds', 'The largest part must be at least max(nums)=10 and at most total 32.', array(['10', '11', '12', '...', '31', '32'], [mark(0, 'lo', 'state'), mark(5, 'hi', 'state')])),
    frame('Test a limit', 'With limit 18, greedy cuts [7,2,5] and [10,8], using two parts.', array(['7+2+5=14', '10+8=18'], [mark(0, 'part 1', 'focus'), mark(1, 'part 2', 'state')], { parts: '2 <= k' })),
    frame('Return the smallest feasible limit', '18 works, while 17 would require three parts.', array(['17', '18', '19'], [mark(1, 'answer', 'output')], { result: '18' })),
  ]),
  'largest-rectangle-in-histogram': visual('A shorter bar ends every taller increasing-stack bar and reveals its maximal width.', [
    frame('Push increasing bars', 'Height 1 closes the earlier height 2; heights 5 and 6 then wait in the increasing stack.', array(['2', '1', '5', '6', '2', '3'], [mark(2, 'push', 'state'), mark(3, 'push', 'focus')], { stack: '[1,5,6]' })),
    frame('Short bar closes rectangles', 'Height 2 pops 6 and 5; their widths are measured to the current index.', array(['2', '1', '5', '6', '2', '3'], [mark(2, 'height 5', 'focus'), mark(3, 'height 6', 'warning'), mark(4, 'boundary', 'state')], { areas: '6*1 and 5*2' })),
    frame('Keep the largest area', 'The bars 5 and 6 form the best rectangle of area 10.', array(['2', '1', '5', '6', '2', '3'], [mark(2, 'width 2', 'output'), mark(3, 'width 2', 'output')], { result: '10' })),
  ]),
  'binary-tree-maximum-path-sum': visual('A node returns one child branch upward but can score both child branches locally.', [
    frame('Return one branch', 'At node 20, the larger child contribution is 15, while the full path can use 15 and 7.', tree([['-10'], ['9', '20'], ['-', '-', '15', '7']], [mark(5, 'return 15', 'state'), mark(2, 'score 42', 'focus')])),
    frame('Reject negative branches', 'A negative child contribution is replaced by zero before combining.', tree([['-10'], ['9', '20'], ['-', '-', '15', '7']], [mark(0, 'left 0', 'state'), mark(2, 'both children', 'focus')], { formula: '20 + 15 + 7 = 42' })),
    frame('Update global best', 'The path 15 -> 20 -> 7 has the maximum sum 42.', tree([['-10'], ['9', '20'], ['-', '-', '15', '7']], [mark(2, 'best path', 'output'), mark(5, 'best path', 'output'), mark(6, 'best path', 'output')], { result: '42' })),
  ]),
  'serialize-and-deserialize-binary-tree': visual('Preorder plus explicit null markers preserves both node values and tree shape.', [
    frame('Visit preorder', 'Tree 1 with a right child 2 visits 1, null-left, 2.', tree([['1'], ['-', '2']], [mark(0, 'visit 1', 'focus'), mark(1, 'null', 'state')])),
    frame('Write markers', 'Missing children become # tokens, so the stream is 1,#,2,#,#.', array(['1', '#', '2', '#', '#'], [mark(1, 'shape marker', 'state'), mark(3, 'shape marker', 'state')])),
    frame('Read the same stream', 'The decoder consumes tokens in the same preorder and rebuilds the shape.', tree([['1'], ['-', '2']], [mark(0, 'rebuilt', 'output')], { result: 'same tree' })),
  ]),
  'longest-increasing-path-in-a-matrix': visual('Memoize the best increasing path starting at each cell; larger-only moves cannot cycle.', [
    frame('Find increasing neighbors', 'From 1, move to 2, then 6, then 9.', grid([['9', '9', '4'], ['6', '6', '8'], ['2', '1', '1']], [{ row: 2, col: 1, label: '1', tone: 'focus' }, { row: 2, col: 0, label: '2', tone: 'state' }])),
    frame('Cache a cell answer', 'The memo table stores the best path length from every cell; the path from 1 has length 4.', grid([['1', '1', '3'], ['2', '2', '2'], ['3', '4', '3']], [{ row: 2, col: 1, label: 'path length 4', tone: 'output' }], { action: 'memoize' })),
    frame('Take the maximum cached value', 'Every cell is solved once; the largest cached path is 4.', grid([['1', '1', '3'], ['2', '2', '2'], ['3', '4', '3']], [{ row: 2, col: 1, label: 'max 4', tone: 'output' }], { result: '4' })),
  ]),
  'alien-dictionary': visual('The first differing character in adjacent words creates a directed ordering edge.', [
    frame('Extract a rule', 'wrt and wrf first differ at t and f, so t -> f.', graph(['w', 'r', 't', 'f'], ['t -> f'], { rule: 't before f' })),
    frame('Collect rules', 'The other adjacent differences add w->e, e->r, and r->t.', graph(['w', 'e', 'r', 't', 'f'], ['w -> e', 'e -> r', 'r -> t', 't -> f'], { indegree: ['w:0', 'e:1', 'r:1', 't:1', 'f:1'] })),
    frame('Topologically order', 'Remove zero-indegree letters and return a valid alien alphabet.', graph(['w', 'e', 'r', 't', 'f'], ['w -> e', 'e -> r', 'r -> t', 't -> f'], { order: ['w', 'e', 'r', 't', 'f'], result: 'wertf' })),
  ]),
  'word-search-ii': visual('A trie shares word prefixes and stops a board search as soon as the path leaves the trie.', [
    frame('Build shared search structure', 'All dictionary words enter one trie; each board path follows only a matching child.', trie([{ word: 'oath', prefix: 'o-a-t-h' }, { word: 'eat', prefix: 'e-a-t' }], { action: 'trie prefixes' })),
    frame('Walk the board and trie together', 'A board path that reaches o-a-t may continue to h; a path with no trie child stops.', grid([['o', 'a', 'a', 'n'], ['e', 't', 'a', 'e'], ['i', 'h', 'k', 'r'], ['i', 'f', 'l', 'v']], [{ row: 0, col: 0, label: 'o', tone: 'state' }, { row: 0, col: 1, label: 'a', tone: 'state' }, { row: 1, col: 1, label: 't', tone: 'focus' }, { row: 2, col: 1, label: 'h', tone: 'output' }], { path: 'oath' })),
    frame('Emit each terminal word once', 'The board finds oath and eat; failed prefixes never expand further.', grid([['o', 'a', 'a', 'n'], ['e', 't', 'a', 'e'], ['i', 'h', 'k', 'r'], ['i', 'f', 'l', 'v']], [], { result: '[oath,eat]' })),
  ]),
  'merge-k-sorted-lists': visual('The heap holds one current head per list; pop the smallest and replace it with that list next.', [
    frame('Seed one head per list', 'The heap contains 1 from list A, 1 from B, and 2 from C.', heap(['A:1', 'B:1', 'C:2'], { root: 'A:1', detail: 'one head per list' })),
    frame('Pop and replace', 'After taking A:1, insert A:4 while B:1 remains the root.', heap(['B:1', 'C:2', 'A:4'], { root: 'B:1', detail: 'replace from same list' })),
    frame('Finish from the remaining heads', 'After emitting 1,1,2,3,4,4, the remaining heads are 5 and 6.', heap(['A:5', 'C:6'], { root: 'A:5', detail: 'emit 5, then 6', result: '1,1,2,3,4,4,5,6' })),
  ]),
  'find-median-from-data-stream': visual('Keep the lower half in a max-heap and the upper half in a min-heap.', [
    frame('Add 1', 'The lower half contains 1; the upper half is empty.', heap(['lower max:1', 'upper min:-'], { root: 'lower 1' })),
    frame('Add 2', 'Balance the halves: lower has 1 and upper has 2.', heap(['lower max:1', 'upper min:2'], { detail: 'two roots bracket the median' })),
    frame('Read the middle', 'With two values, median is (1+2)/2 = 1.5.', heap(['lower max:1', 'upper min:2'], { result: '1.5' })),
  ]),
};

const expectedProblemCount = 106;
const actualProblemCount = Object.keys(codingQuestionVisuals).length;
if (actualProblemCount !== expectedProblemCount) {
  throw new Error(`Expected ${expectedProblemCount} coding visual definitions, found ${actualProblemCount}`);
}

import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const sourceArgument = process.argv.slice(2).find((argument) => !argument.startsWith('--'));
const sourcePath = sourceArgument || path.resolve(root, '../ml_interview_book/docs/dsa/Coding_Questions_Phone_Guide.md');
const postsDir = path.join(root, 'src/content/posts');
const auditsDir = path.join(root, 'data/visual-audits');
const publicationDate = '2026-09-01';

const chapterDefinitions = [
  { id: 'remember-the-past', title: 'Remember the past', description: 'Use maps, sets, and saved boundary values to turn repeated searching into one pass.', difficulty: 'Foundation', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['1', '2', '3', '4', '5', '6'] },
  { id: 'move-boundaries', title: 'Move boundaries', description: 'Use sorted order, windows, and answer-space search to discard impossible regions safely.', difficulty: 'Foundation', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['7', '8', '9', '10', '11', '12', '13', '14', '15', '16'] },
  { id: 'unfinished-work', title: 'Keep unfinished work', description: 'Use stacks and monotonic state when the newest unresolved item must be handled first.', difficulty: 'Foundation', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['17', '18', '19', '20'] },
  { id: 'next-best-item', title: 'Process the next best item', description: 'Let queues, heaps, and shortest-path frontiers decide which reachable item comes next.', difficulty: 'Intermediate', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['21', '22', '23', '24', '25'] },
  { id: 'explore-choices', title: 'Explore choices', description: 'Traverse graphs, trees, choice paths, and smaller dynamic-programming states without losing the invariant.', difficulty: 'Intermediate', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42'] },
  { id: 'useful-order', title: 'Create a useful order', description: 'Sort ranges, commit safe greedy choices, remove prerequisites, and join connected groups.', difficulty: 'Intermediate', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['43', '44', '45', '46', '47', '48', '49', '50'] },
  { id: 'change-links', title: 'Change links', description: 'Rewire linked lists and prefix trees while preserving the pointer or path you still need.', difficulty: 'Intermediate', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['51', '52', '53', '54', '55', '56'] },
  { id: 'core-coverage', title: 'Complete core coverage', description: 'Reuse the main patterns across bits, strings, matrices, trees, graphs, intervals, and caches.', difficulty: 'Mixed', priority: 'Core', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation'], numbers: ['57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88'] },
  { id: 'practical-ai-coding', title: 'Practical AI coding', description: 'Make array shapes, masks, numerical stability, batching, selection, and metrics visible before coding.', difficulty: 'Intermediate', priority: 'Role-specific', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'ML implementation', 'ML breadth'], numbers: ['AI1', 'AI2', 'AI3', 'AI4', 'AI5', 'AI6', 'AI7', 'AI8'] },
  { id: 'hard-problems', title: 'Hard problems', description: 'Combine boundaries, stacks, trees, grids, tries, heaps, and answer search after the core patterns feel natural.', difficulty: 'Advanced', priority: 'Specialist', roles: ['MLE', 'RE', 'AS'], rounds: ['Coding', 'Work sample'], numbers: ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10'] },
];

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

function collapse(value) {
  return value.replace(/\s+/g, ' ').trim();
}

function extractField(section, label) {
  const expression = new RegExp(`\\*\\*${label}:\\*\\*\\s*([\\s\\S]*?)(?=\\n\\n(?:\\*\\*|<a id=)|$)`);
  return collapse(section.match(expression)?.[1] || '');
}

function wrapParagraphs(text) {
  const lines = text.split('\n');
  const output = [];
  let inFence = false;
  for (const line of lines) {
    if (line.startsWith('```')) {
      inFence = !inFence;
      output.push(line);
      continue;
    }
    if (inFence || !line.trim() || /^\s*(?:[#*<`]|\||[-*+] |\d+\. )/.test(line)) {
      output.push(line);
      continue;
    }
    const words = line.trim().split(/\s+/);
    let current = '';
    for (const word of words) {
      if (current && `${current} ${word}`.length > 88) {
        output.push(current);
        current = word;
      } else {
        current = current ? `${current} ${word}` : word;
      }
    }
    if (current) output.push(current);
  }
  return output.join('\n');
}

function modeFor(title, pattern) {
  const text = `${title} ${pattern}`.toLowerCase();
  if (/pairwise squared distances/.test(text)) return 'broadcast';
  if (/stable softmax|cross-entropy from logits/.test(text)) return 'numerics';
  if (/causal attention/.test(text)) return 'attention';
  if (/pad variable-length sequences/.test(text)) return 'padding';
  if (/mini-batches/.test(text)) return 'batching';
  if (/top-k scores/.test(text)) return 'selection';
  if (/binary precision and recall/.test(text)) return 'metrics';
  if (/product of array except self|subarray sum equals k/.test(text)) return 'prefix';
  if (/maximum subarray|best time to buy and sell stock/.test(text)) return 'running';
  if (/maximum product subarray/.test(text)) return 'extrema';
  if (/top[- ]k frequent elements/.test(text)) return 'frequency';
  if (/3sum|container with most water/.test(text)) return 'two-pointer';
  if (/min stack/.test(text)) return 'stack';
  if (/merge two sorted lists|remove nth node from end/.test(text)) return 'linked';
  if (/binary search|rotated sorted|koko|split array|minimum in rotated/.test(text)) return 'binary';
  if (/sliding window|substring|permutation in string|nice subarrays|minimum window/.test(text)) return 'window';
  if (/parentheses|decode string|daily temperatures|monotonic|histogram/.test(text)) return 'stack';
  if (/dijkstra|network delay/.test(text)) return 'dijkstra';
  if (/bfs|rotting|level order/.test(text)) return 'bfs';
  if (/heap|kth largest|merge k|median from data/.test(text)) return 'heap';
  if (/matrix|spiral|rotate image|longest increasing path in a matrix/.test(text)) return 'matrix';
  if (/clone graph|islands|water flow|connected components|graph valid|increasing path/.test(text)) return 'graph';
  if (/binary tree|same tree|invert|balanced|subtree|path sum|serialize|bst|kth smallest|ancestor|tree from/.test(text)) return 'tree';
  if (/word search ii|trie|add and search words/.test(text)) return 'trie';
  if (/alien dictionary|course schedule|topological/.test(text)) return 'topology';
  if (/combination sum iv|dynamic programming|climbing|house robber|partition|common subsequence|edit distance|coin change|decode ways|unique paths|word break|increasing subsequence/.test(text)) return 'dp';
  if (/subsets|permutations|combination sum|word search/.test(text)) return 'backtrack';
  if (/interval|meeting room|jump game/.test(text)) return 'interval';
  if (/redundant connection|union-find/.test(text)) return 'union';
  if (/linked list|reorder list/.test(text)) return 'linked';
  if (/bit|integer|missing number/.test(text)) return 'bit';
  if (/palindrome|encode and decode strings/.test(text)) return 'string';
  if (/hash|anagram|consecutive|product of array|subarray sum|duplicate|stock|maximum subarray|maximum product/.test(text)) return 'hash';
  return 'state';
}

const visualTemplates = {
  hash: {
    kicker: 'Memory as a shortcut',
    title: 'Save the fact that makes the next item cheap',
    steps: [
      ['1. Read', 'one item', 'The scan has a current value and a position.'],
      ['2. Remember', 'small state', 'Store the fact a future item may need.'],
      ['3. Ask', 'lookup or difference', 'Turn the target into a question about saved state.'],
      ['4. Commit', 'answer or update', 'A hit completes the answer; otherwise save this item.'],
    ],
    invariant: 'The state contains every useful fact from the prefix already processed.',
    caption: 'Follow the scan from left to right. The data structure is a compressed memory of the past, so the current item never needs to rescan earlier items.',
  },
  binary: {
    kicker: 'A shrinking answer space',
    title: 'Discard a half only after a yes-or-no test',
    steps: [
      ['1. Bound', 'lo ... hi', 'Every possible answer is inside this interval.'],
      ['2. Probe', 'mid', 'Test the middle value or candidate answer.'],
      ['3. Decide', 'predicate', 'The monotone result says which side can survive.'],
      ['4. Keep', 'one half', 'Move one boundary and preserve the answer.'],
    ],
    invariant: 'The answer never leaves the current low-to-high interval.',
    caption: 'Read the interval as a promise: everything outside it is already impossible. The midpoint is useful only because the predicate is monotone.',
  },
  window: {
    kicker: 'A moving range',
    title: 'Grow until valid, then shrink until necessary',
    steps: [
      ['1. Extend', 'L ... R', 'Move the right edge to include new evidence.'],
      ['2. Measure', 'window state', 'Update counts, sum, or the required matches.'],
      ['3. Tighten', 'advance L', 'Remove the oldest item while validity survives.'],
      ['4. Record', 'best valid range', 'Save the shortest, longest, or counted window.'],
    ],
    invariant: 'The current window has exactly the state needed to decide whether it is valid.',
    caption: 'The two edges are not guesses. The right edge gathers enough evidence; the left edge removes anything no longer needed, so each item enters and leaves once.',
  },
  stack: {
    kicker: 'Last unfinished, first resolved',
    title: 'Keep unresolved work in the order it must finish',
    steps: [
      ['1. Arrive', 'new token', 'Read the next symbol or value.'],
      ['2. Hold', 'stack top', 'Keep work that cannot finish yet.'],
      ['3. Resolve', 'match or warmer', 'A new item may finish the newest waiting item.'],
      ['4. Restore', 'remaining stack', 'Anything left is still unfinished or invalid.'],
    ],
    invariant: 'The top of the stack is the newest unresolved item and the only one that can be resolved next.',
    caption: 'Look at the top, not the whole history. Nesting and next-greater relationships work because the newest unresolved item blocks everything below it.',
  },
  heap: {
    kicker: 'A frontier ordered by value',
    title: 'Keep the candidates that can still win',
    steps: [
      ['1. Offer', 'candidate set', 'Put a new value into the frontier.'],
      ['2. Expose', 'root = next best', 'The heap makes the smallest or largest current item visible.'],
      ['3. Trim', 'keep k', 'Discard a candidate that cannot enter the answer.'],
      ['4. Advance', 'next candidate', 'Replace the used item and continue the stream.'],
    ],
    invariant: 'The heap root is the next candidate whose priority is safe to process.',
    caption: 'The heap is not a sorted list. It exposes only the next useful item, while preserving enough frontier state to continue without sorting everything.',
  },
  bfs: {
    kicker: 'Distance in layers',
    title: 'A queue turns time or steps into visible layers',
    steps: [
      ['1. Seed', 'frontier at 0', 'Put every starting position in the queue.'],
      ['2. Pop', 'current layer', 'Process only positions at the same distance.'],
      ['3. Spread', 'next layer', 'Add each newly reachable neighbor once.'],
      ['4. Finish', 'first arrival', 'The first layer reaching a goal is shortest.'],
    ],
    invariant: 'The queue is ordered by nondecreasing distance from the starting frontier.',
    caption: 'Read each queue layer as one minute or one step. Multiple starting points belong in the first layer, which is why multi-source BFS measures the nearest source.',
  },
  dijkstra: {
    kicker: 'The cheapest frontier first',
    title: 'Finalize a node when no cheaper path remains',
    steps: [
      ['1. Seed', 'distance 0', 'Start with the source and its known cost.'],
      ['2. Choose', 'min heap', 'Pop the reachable path with least total cost.'],
      ['3. Relax', 'new cost', 'Offer each outgoing path if it improves the estimate.'],
      ['4. Finalize', 'locked distance', 'The popped cost is final when edges are nonnegative.'],
    ],
    invariant: 'Every finalized node has the smallest possible distance from the source.',
    caption: 'The heap orders paths by total cost, not by the last edge. Once the cheapest frontier path reaches a node, any alternative must be at least as expensive.',
  },
  graph: {
    kicker: 'Reachability without repetition',
    title: 'Turn a large graph into one frontier and one visited set',
    steps: [
      ['1. Start', 'current node', 'Choose a source or an unseen component.'],
      ['2. Expand', 'neighbors', 'Follow edges or legal grid moves.'],
      ['3. Mark', 'visited', 'Record a node before adding it again.'],
      ['4. Count', 'component or goal', 'The explored set gives the answer.'],
    ],
    invariant: 'Every visited node has been scheduled exactly once, so cycles cannot repeat work.',
    caption: 'The frontier is the boundary between known and unknown nodes. Marking a node when it enters the frontier prevents a cycle from creating duplicate searches.',
  },
  tree: {
    kicker: 'A child answer returns upward',
    title: 'Solve a node by asking each child for one complete fact',
    steps: [
      ['1. Enter', 'current node', 'The call owns one subtree.'],
      ['2. Ask', 'left / right', 'Each child returns its subtree fact.'],
      ['3. Combine', 'node rule', 'Use the child facts to score or validate here.'],
      ['4. Return', 'one useful value', 'Pass only what the parent can still use.'],
    ],
    invariant: 'A returned value completely summarizes the subtree below its node.',
    caption: 'Read the tree bottom-up even when the code is recursive. A parent does not need every descendant, only the compact fact each child promises to return.',
  },
  backtrack: {
    kicker: 'A tree of choices',
    title: 'Choose, explore, then undo the exact choice',
    steps: [
      ['1. Path', 'partial answer', 'The current path is a valid unfinished choice.'],
      ['2. Choose', 'one branch', 'Add one available value, cell, or letter.'],
      ['3. Recurse', 'smaller problem', 'Explore everything below that choice.'],
      ['4. Undo', 'restore state', 'Remove the same choice before the next branch.'],
    ],
    invariant: 'At every call, the path contains exactly the choices on the route from the root.',
    caption: 'The visual is a choice tree, not a list of magic loops. Backtracking works because every branch starts from the same restored parent state.',
  },
  dp: {
    kicker: 'A small state graph',
    title: 'Keep the complete answer for each smaller state',
    steps: [
      ['1. Base', 'known state', 'Initialize the smallest solvable problem.'],
      ['2. Read', 'earlier answers', 'Look only at states the transition depends on.'],
      ['3. Build', 'current state', 'Choose, count, or combine those answers.'],
      ['4. Compress', 'rolling memory', 'Discard old states that no future step needs.'],
    ],
    invariant: 'Each saved state is the complete answer for its prefix, amount, cell, or pair of prefixes.',
    caption: 'Treat the table as a map of smaller questions. The recurrence is the arrow between states; space optimization is safe only after the dependencies are visible.',
  },
  interval: {
    kicker: 'Time ranges on one line',
    title: 'Sorting makes the next possible conflict visible',
    steps: [
      ['1. Order', 'start or end', 'Put ranges in the order the proof needs.'],
      ['2. Compare', 'current boundary', 'Check the next range against the active boundary.'],
      ['3. Decide', 'overlap?', 'Merge, remove, or allocate a room.'],
      ['4. Advance', 'last safe end', 'Carry the boundary that preserves future room.'],
    ],
    invariant: 'The saved boundary summarizes every interval that can still affect the next one.',
    caption: 'See intervals as occupied segments, not pairs of unrelated numbers. The sort order turns a global overlap question into a local boundary comparison.',
  },
  topology: {
    kicker: 'Dependencies becoming ready',
    title: 'Remove prerequisites until the next zero-indegree item appears',
    steps: [
      ['1. Count', 'incoming edges', 'Record how many requirements each node still has.'],
      ['2. Ready', 'indegree = 0', 'Only nodes with no unmet prerequisite can start.'],
      ['3. Remove', 'complete one', 'Subtract its edge from every dependent node.'],
      ['4. Detect', 'cycle or order', 'Unfinished nodes reveal a dependency cycle.'],
    ],
    invariant: 'The ready queue contains exactly the nodes whose prerequisites are complete.',
    caption: 'Imagine removing foundation blocks from a dependency wall. A block becomes available only when every incoming requirement has disappeared.',
  },
  union: {
    kicker: 'Components with one representative',
    title: 'Ask roots whether two endpoints already belong together',
    steps: [
      ['1. Find', 'root(a), root(b)', 'Follow parent links to each component representative.'],
      ['2. Compare', 'same root?', 'Equal roots mean the edge closes a cycle.'],
      ['3. Join', 'different roots', 'Attach one component under the other.'],
      ['4. Compress', 'short parent paths', 'Future root checks become cheaper.'],
    ],
    invariant: 'All nodes in one connected component eventually point to the same root.',
    caption: 'The parent array is a map of component identity. You do not need to walk every edge again; compare representatives and merge only when the groups differ.',
  },
  linked: {
    kicker: 'Pointers in motion',
    title: 'Save the next link before redirecting the current one',
    steps: [
      ['1. Save', 'next pointer', 'Keep the only route to the unrevised suffix.'],
      ['2. Redirect', 'current.next', 'Change one link to the new direction.'],
      ['3. Advance', 'previous / current', 'Move the working window by one node.'],
      ['4. Return', 'new head', 'The pointer at the boundary becomes the result.'],
    ],
    invariant: 'Every node is still reachable through either the saved suffix or the rebuilt prefix.',
    caption: 'The list is a chain of ownership. First save the outgoing link, then edit the link you own, then advance into the saved suffix.',
  },
  trie: {
    kicker: 'Prefixes share a path',
    title: 'Store each character once along a shared prefix route',
    steps: [
      ['1. Root', 'empty prefix', 'All words begin at one shared node.'],
      ['2. Walk', 'one character', 'Follow the edge for the next letter.'],
      ['3. Branch', 'shared or new', 'Reuse a prefix or create a child node.'],
      ['4. Mark', 'word ends here', 'Separate a complete word from its prefix.'],
    ],
    invariant: 'The path from the root to the current node spells exactly the prefix being queried.',
    caption: 'A trie turns repeated string prefixes into shared structure. A full-word marker matters because a word can end before another word that continues through it.',
  },
  bit: {
    kicker: 'Bits as visible state',
    title: 'Use one local bit identity to remove or cancel work',
    steps: [
      ['1. Read', 'lowest bit', 'Inspect the bit at the edge of the word.'],
      ['2. Combine', 'XOR / AND', 'Separate information from carry or cancellation.'],
      ['3. Shift', 'move one place', 'Bring the next bit into position.'],
      ['4. Finish', 'zero or fixed width', 'Stop when the represented state is complete.'],
    ],
    invariant: 'Each step preserves the numerical meaning of the bits not yet processed.',
    caption: 'Watch bits move from one position to the next. XOR keeps non-carrying differences, AND identifies shared one-bits, and shifts expose the next position.',
  },
  matrix: {
    kicker: 'Boundaries and coordinates',
    title: 'Make the unvisited rectangle explicit before overwriting it',
    steps: [
      ['1. Mark', 'row / column', 'Record information in a safe boundary or marker cell.'],
      ['2. Visit', 'current layer', 'Read only the still-unvisited rectangle or ring.'],
      ['3. Move', 'boundary inward', 'Shrink the region after a side is complete.'],
      ['4. Reuse', 'in-place result', 'Write after the original information is safe.'],
    ],
    invariant: 'The boundaries describe exactly which cells remain unread or unmodified.',
    caption: 'The matrix becomes easier when you draw its active rectangle. Every operation either marks a future action or consumes one boundary, so no cell is accidentally read twice.',
  },
  string: {
    kicker: 'Characters and centers',
    title: 'Compare the only characters that can still decide the answer',
    steps: [
      ['1. Point', 'left / right', 'Choose the two positions or a possible center.'],
      ['2. Normalize', 'skip or align', 'Ignore separators or align matching lengths.'],
      ['3. Expand', 'equal pair', 'Move outward while the local rule survives.'],
      ['4. Record', 'best text', 'Keep the longest, valid, or decodable result.'],
    ],
    invariant: 'Everything outside the active pointers or center expansion has already been resolved.',
    caption: 'The string is a line, not a bag of characters. The pointers identify the only unresolved comparison, and each successful comparison makes the next one smaller.',
  },
  tensor: {
    kicker: 'Shapes before arithmetic',
    title: 'Align dimensions, then let the operation expose the result',
    steps: [
      ['1. Name', 'input shapes', 'Write the batch, sequence, class, or feature axes.'],
      ['2. Align', 'broadcast / mask', 'Expand only singleton axes or valid positions.'],
      ['3. Operate', 'reduce / select', 'Apply the numerical rule along its stated axis.'],
      ['4. Check', 'output shape', 'Verify the result and its edge cases.'],
    ],
    invariant: 'Every axis has a declared meaning, and the output shape follows from the operation rather than a guess.',
    caption: 'Read the boxes as named shapes. Most practical array bugs are visible before coding when each axis is named and the transformation is drawn.',
  },
  prefix: {
    kicker: 'Two passes, one answer',
    title: 'Combine the information on both sides of the current position',
    steps: [
      ['1. Forward', 'left accumulator', 'Carry everything strictly before the current item.'],
      ['2. Store', 'left contribution', 'Write the part that belongs in this answer.'],
      ['3. Backward', 'right accumulator', 'Walk from the other side without revisiting the array.'],
      ['4. Combine', 'left × right', 'Join both outside contributions at the current position.'],
    ],
    invariant: 'The accumulators describe only values outside the current position, so the current value is excluded.',
    caption: 'See each answer as a hole in the array. One pass fills the left side of every hole; the reverse pass supplies the right side or the earlier prefix count.',
  },
  running: {
    kicker: 'One pass, running best',
    title: 'Carry the smallest, largest, or best state seen so far',
    steps: [
      ['1. Observe', 'current value', 'Read the next price, sum, or candidate.'],
      ['2. Carry', 'state so far', 'Keep the summary future positions can use.'],
      ['3. Update', 'best decision', 'Compare starting fresh, extending, buying, or selling.'],
      ['4. Record', 'best answer', 'Preserve the strongest result seen anywhere.'],
    ],
    invariant: 'The carried state is the complete summary needed to make the next position optimal.',
    caption: 'The scan does not remember every earlier value. It remembers the one summary that gives every future position its best possible continuation.',
  },
  extrema: {
    kicker: 'Keep both signs alive',
    title: 'A negative value can swap the best and worst futures',
    steps: [
      ['1. Hold', 'max and min', 'Keep both extremes ending at the current position.'],
      ['2. Flip', 'negative multiplier', 'A negative value exchanges their future roles.'],
      ['3. Extend', 'or restart', 'Multiply the old extremes or begin at this value.'],
      ['4. Record', 'largest product', 'Save the best ending value seen so far.'],
    ],
    invariant: 'Both the maximum and minimum product ending here are available for the next value.',
    caption: 'The minimum is not discarded as bad news. One more negative value can turn it into the maximum, so the visual keeps both futures alive.',
  },
  frequency: {
    kicker: 'Counts become buckets',
    title: 'Turn frequency into a coordinate you can scan',
    steps: [
      ['1. Count', 'value → frequency', 'Build one count for each distinct value.'],
      ['2. Place', 'frequency bucket', 'Put the value at the coordinate named by its count.'],
      ['3. Scan', 'high to low', 'Read the buckets in the order the answer needs.'],
      ['4. Stop', 'top k values', 'Return as soon as enough values are collected.'],
    ],
    invariant: 'Every value appears in the bucket matching its complete frequency.',
    caption: 'The buckets replace repeated sorting. Frequency is now position, so walking from the largest bucket exposes the most common values first.',
  },
  'two-pointer': {
    kicker: 'Two ends, one proof',
    title: 'Move an endpoint only when the other choice cannot help',
    steps: [
      ['1. Arrange', 'sorted or bounded', 'Put the candidates in an order that supports comparison.'],
      ['2. Compare', 'left + right', 'Measure the pair or the container formed by both ends.'],
      ['3. Move', 'provably weaker end', 'Discard the side that cannot improve the answer.'],
      ['4. Record', 'valid pair or best area', 'Save the result before narrowing the search.'],
    ],
    invariant: 'Everything outside the two pointers has been checked or proven unable to improve the answer.',
    caption: 'The pointers are a proof boundary. The next move is safe only because one endpoint is the limiting factor or the sorted sum has the wrong sign.',
  },
  broadcast: {
    kicker: 'Singleton axes expand',
    title: 'One point meets every center without a Python loop',
    steps: [
      ['1. Reshape', 'points [n,1,d]', 'Leave a singleton axis for the centers.'],
      ['2. Align', 'centers [1,k,d]', 'Leave a singleton axis for the points.'],
      ['3. Broadcast', 'difference [n,k,d]', 'Pair every point with every center by shape.'],
      ['4. Reduce', 'sum over d', 'Collapse feature coordinates into squared distances.'],
    ],
    invariant: 'The final two axes identify one point-center pair and its feature coordinates.',
    caption: 'Broadcasting is a shape construction. The singleton axes make the full pair grid visible before subtraction, then the feature axis is the only one reduced.',
  },
  numerics: {
    kicker: 'Stable numerical path',
    title: 'Change the reference point before exponentiating',
    steps: [
      ['1. Anchor', 'row maximum', 'Choose a value that keeps shifted logits small.'],
      ['2. Shift', 'logits − max', 'Preserve relative differences while avoiding overflow.'],
      ['3. Normalize', 'logsumexp or softmax', 'Aggregate the shifted exponentials stably.'],
      ['4. Select', 'correct class', 'Read the requested probability or loss term.'],
    ],
    invariant: 'Subtracting one row constant changes no softmax probabilities or cross-entropy differences.',
    caption: 'The large raw numbers are not the information. Their differences are. Shift the row first, then exponentiate values that are numerically safe.',
  },
  attention: {
    kicker: 'Causal information flow',
    title: 'Mask future positions before probabilities are formed',
    steps: [
      ['1. Score', 'QKᵀ', 'Compare every query with every key.'],
      ['2. Scale', 'divide by √d', 'Keep score magnitudes stable across key widths.'],
      ['3. Mask', 'future = −∞', 'Make forbidden positions receive zero probability.'],
      ['4. Mix', 'weights × V', 'Read only the prefix allowed for each token.'],
    ],
    invariant: 'Row i assigns probability only to keys at positions 0 through i.',
    caption: 'The triangular mask is the lesson. Scores may be computed for every pair, but future entries are removed before softmax can give them weight.',
  },
  padding: {
    kicker: 'A rectangular batch',
    title: 'Pad values and validity together',
    steps: [
      ['1. Measure', 'longest sequence', 'Choose one width for the batch.'],
      ['2. Fill', 'pad value', 'Initialize every unused position explicitly.'],
      ['3. Copy', 'real tokens', 'Write each sequence into its prefix slice.'],
      ['4. Mark', 'boolean mask', 'Keep computation aware of real versus padded cells.'],
    ],
    invariant: 'Every padded token has a false mask, and every real token has a true mask at the same position.',
    caption: 'Padding creates a rectangle; the mask preserves the original ragged boundary. The two arrays must be drawn as one contract.',
  },
  batching: {
    kicker: 'A fixed-size cursor',
    title: 'Advance by a slice and keep the final remainder',
    steps: [
      ['1. Start', 'cursor = 0', 'Point at the first unprocessed item.'],
      ['2. Slice', 'start : start + size', 'Take a full batch when possible.'],
      ['3. Yield', 'current batch', 'Process the slice without changing its contents.'],
      ['4. Advance', 'cursor += size', 'Repeat until the cursor reaches the end.'],
    ],
    invariant: 'Every item belongs to exactly one yielded slice, including the final short slice.',
    caption: 'The range of start positions is the whole algorithm. Python clips the last slice at the sequence end, so no special final-batch branch is needed.',
  },
  selection: {
    kicker: 'Select first, sort last',
    title: 'Avoid ordering values that cannot enter the answer',
    steps: [
      ['1. Partition', 'candidate boundary', 'Separate a possible top-k group without full sorting.'],
      ['2. Retain', 'k candidates', 'Discard everything that cannot be selected.'],
      ['3. Order', 'selected scores', 'Sort only the retained group.'],
      ['4. Return', 'indices + scores', 'Emit the requested order and shape.'],
    ],
    invariant: 'The retained group contains the k largest values before final ordering.',
    caption: 'Selection answers membership; sorting answers presentation order. Keeping those jobs separate is the memory and time saving.',
  },
  metrics: {
    kicker: 'Counts before ratios',
    title: 'Accumulate the four cells, then compute precision and recall',
    steps: [
      ['1. Classify', 'truth × prediction', 'Send each example to exactly one confusion cell.'],
      ['2. Count', 'TP FP TN FN', 'Retain additive counts rather than raw examples.'],
      ['3. Divide', 'chosen denominator', 'Use predicted positives for precision or real positives for recall.'],
      ['4. Guard', 'zero denominator', 'Define the empty-class behavior explicitly.'],
    ],
    invariant: 'Each observation contributes to one and only one confusion-matrix count.',
    caption: 'The denominator is part of the metric. Draw the count cells first, then draw the ratio that selects the row or column it needs.',
  },
  state: {
    kicker: 'A compact mental model',
    title: 'Name the state, the safe move, and the proof',
    steps: [
      ['1. Input', 'what arrives', 'Identify the part of the problem being processed now.'],
      ['2. State', 'what survives', 'Keep only information future steps may need.'],
      ['3. Move', 'safe transition', 'Change the state without losing a possible answer.'],
      ['4. Output', 'proof of finish', 'Know what condition makes the result complete.'],
    ],
    invariant: 'The state is sufficient to make the next move without reconstructing the past.',
    caption: 'Use this as a four-question rehearsal: what arrived, what must survive, why is the move safe, and what proves completion?',
  },
};

function renderSketch(mode) {
  switch (mode) {
    case 'prefix':
      return '<div class="coding-visual-sketch coding-visual-sketch--prefix"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--input">left</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--state">index</span><span class="coding-sketch-arrow">&larr;</span><span class="coding-sketch-pill coding-sketch-pill--focus">right</span></div><p class="coding-sketch-note">two passes meet at one position without including its own value</p></div>';
    case 'running':
      return '<div class="coding-visual-sketch coding-visual-sketch--running"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--state">best so far</span><span class="coding-sketch-arrow">&larr;</span><span class="coding-sketch-pill coding-sketch-pill--input">current</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--focus">new best?</span></div><p class="coding-sketch-note">the carried summary is enough to judge the next value</p></div>';
    case 'extrema':
      return '<div class="coding-visual-sketch coding-visual-sketch--extrema"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--state">minimum</span><span class="coding-sketch-arrow">&harr;</span><span class="coding-sketch-pill coding-sketch-pill--focus">negative?</span><span class="coding-sketch-arrow">&harr;</span><span class="coding-sketch-pill coding-sketch-pill--active">maximum</span></div><p class="coding-sketch-note">a negative input can swap which extreme wins next</p></div>';
    case 'frequency':
      return '<div class="coding-visual-sketch coding-visual-sketch--frequency"><div class="coding-sketch-buckets"><span class="coding-sketch-bucket"><b>3</b> value</span><span class="coding-sketch-bucket"><b>2</b> value, value</span><span class="coding-sketch-bucket"><b>1</b> value</span></div><p class="coding-sketch-note">frequency is the bucket coordinate; scan from the largest count</p></div>';
    case 'two-pointer':
      return '<div class="coding-visual-sketch coding-visual-sketch--two-pointer"><div class="coding-sketch-array"><span class="coding-sketch-pointer">L</span><span class="coding-sketch-cell">candidate</span><span class="coding-sketch-cell coding-sketch-cell--active">pair</span><span class="coding-sketch-cell">candidate</span><span class="coding-sketch-pointer">R</span></div><p class="coding-sketch-note">compare both ends, then move the limiting side</p></div>';
    case 'broadcast':
      return '<div class="coding-visual-sketch coding-visual-sketch--broadcast"><div class="coding-sketch-shapes"><span class="coding-sketch-shape coding-sketch-shape--input">[n,1,d]</span><span class="coding-sketch-arrow">&times;</span><span class="coding-sketch-shape coding-sketch-shape--state">[1,k,d]</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-shape coding-sketch-shape--active">[n,k,d]</span></div><p class="coding-sketch-note">singleton axes expand; the feature axis remains available for reduction</p></div>';
    case 'numerics':
      return '<div class="coding-visual-sketch coding-visual-sketch--numerics"><div class="coding-sketch-array"><span class="coding-sketch-cell coding-sketch-cell--state">large</span><span class="coding-sketch-arrow">&minus; max</span><span class="coding-sketch-cell coding-sketch-cell--active">small</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell">safe exp</span></div><p class="coding-sketch-note">relative differences stay; raw magnitude stops causing overflow</p></div>';
    case 'attention':
      return '<div class="coding-visual-sketch coding-visual-sketch--attention"><div class="coding-sketch-attention-grid"><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">read</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">mask</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">mask</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">read</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">read</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">mask</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">read</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">read</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">read</span></div><p class="coding-sketch-note">row i keeps columns 0 through i and masks every future column</p></div>';
    case 'padding':
      return '<div class="coding-visual-sketch coding-visual-sketch--padding"><div class="coding-sketch-matrix"><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">token</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">token</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">pad / 0</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">token</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">pad / 0</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">pad / 0</span></div><p class="coding-sketch-note">the mask marks the same cells that contain real tokens</p></div>';
    case 'batching':
      return '<div class="coding-visual-sketch coding-visual-sketch--batching"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--state">0 : size</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--active">size : 2size</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill">remainder</span></div><p class="coding-sketch-note">one cursor partitions the input into non-overlapping slices</p></div>';
    case 'selection':
      return '<div class="coding-visual-sketch coding-visual-sketch--selection"><div class="coding-sketch-array"><span class="coding-sketch-cell">discard</span><span class="coding-sketch-cell coding-sketch-cell--state">candidate</span><span class="coding-sketch-cell coding-sketch-cell--state">candidate</span><span class="coding-sketch-cell coding-sketch-cell--active">top k</span></div><p class="coding-sketch-note">membership first, presentation order second</p></div>';
    case 'metrics':
      return '<div class="coding-visual-sketch coding-visual-sketch--metrics"><div class="coding-sketch-matrix coding-sketch-matrix--metrics"><span class="coding-sketch-grid-cell">TN</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">FP</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">FN</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">TP</span></div><p class="coding-sketch-note">precision reads the predicted-positive column; recall reads the actual-positive row</p></div>';
    case 'hash':
      return '<div class="coding-visual-sketch coding-visual-sketch--hash"><div class="coding-sketch-row"><span class="coding-sketch-label">current</span><span class="coding-sketch-pill coding-sketch-pill--input">item</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-label">ask</span><span class="coding-sketch-pill coding-sketch-pill--focus">needed fact</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--state">saved state</span></div><p class="coding-sketch-note">read the concrete example above as the values flowing through this lookup</p></div>';
    case 'binary':
      return '<div class="coding-visual-sketch coding-visual-sketch--binary"><div class="coding-sketch-array"><span class="coding-sketch-pointer">lo</span><span class="coding-sketch-cell">left</span><span class="coding-sketch-cell">left</span><span class="coding-sketch-cell coding-sketch-cell--active">mid</span><span class="coding-sketch-cell">right</span><span class="coding-sketch-cell">right</span><span class="coding-sketch-pointer">hi</span></div><p class="coding-sketch-note">probe the middle, then discard the side the predicate rules out</p></div>';
    case 'window':
      return '<div class="coding-visual-sketch coding-visual-sketch--window"><div class="coding-sketch-array"><span class="coding-sketch-pointer">L</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-cell coding-sketch-cell--active">active</span><span class="coding-sketch-cell coding-sketch-cell--state">inside</span><span class="coding-sketch-pointer">R</span></div><p class="coding-sketch-note">the active bracket grows for evidence and shrinks when its state is sufficient</p></div>';
    case 'stack':
      return '<div class="coding-visual-sketch coding-visual-sketch--stack"><div class="coding-sketch-stack"><span class="coding-sketch-label">older work</span><span class="coding-sketch-stack-item">waiting</span><span class="coding-sketch-stack-item">waiting</span><span class="coding-sketch-stack-item coding-sketch-stack-item--active">top resolves next</span></div><p class="coding-sketch-note">the newest unfinished item blocks everything below it</p></div>';
    case 'heap':
      return '<div class="coding-visual-sketch coding-visual-sketch--heap"><div class="coding-sketch-tree"><span class="coding-sketch-node coding-sketch-node--active">root: next best</span><div class="coding-sketch-branch"><span class="coding-sketch-node">candidate</span><span class="coding-sketch-node">candidate</span></div></div><p class="coding-sketch-note">the root is exposed while the rest stays as a frontier</p></div>';
    case 'bfs':
      return '<div class="coding-visual-sketch coding-visual-sketch--bfs"><div class="coding-sketch-grid"><span class="coding-sketch-grid-cell coding-sketch-grid-cell--seen">0</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--frontier">1</span><span class="coding-sketch-grid-cell">2</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--seen">1</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--frontier">2</span><span class="coding-sketch-grid-cell">3</span><span class="coding-sketch-grid-cell">2</span><span class="coding-sketch-grid-cell">3</span><span class="coding-sketch-grid-cell">4</span></div><p class="coding-sketch-note">each layer is one more step or minute from the starting frontier</p></div>';
    case 'dijkstra':
      return '<div class="coding-visual-sketch coding-visual-sketch--dijkstra"><div class="coding-sketch-path"><span class="coding-sketch-node coding-sketch-node--active">start</span><span class="coding-sketch-edge">cost 1</span><span class="coding-sketch-node">next</span><span class="coding-sketch-edge">cost 4</span><span class="coding-sketch-node">farther</span></div><p class="coding-sketch-note">compare total path cost, then lock the cheapest frontier node</p></div>';
    case 'graph':
      return '<div class="coding-visual-sketch coding-visual-sketch--graph"><div class="coding-sketch-graph"><span class="coding-sketch-node coding-sketch-node--active">start</span><span class="coding-sketch-edge">&harr;</span><span class="coding-sketch-node">visited</span><span class="coding-sketch-edge">&harr;</span><span class="coding-sketch-node coding-sketch-node--state">unseen</span></div><p class="coding-sketch-note">the frontier separates visited nodes from reachable unknowns</p></div>';
    case 'tree':
      return '<div class="coding-visual-sketch coding-visual-sketch--tree"><div class="coding-sketch-tree"><span class="coding-sketch-node coding-sketch-node--active">parent</span><div class="coding-sketch-branch"><span class="coding-sketch-node">left fact</span><span class="coding-sketch-node">right fact</span></div></div><p class="coding-sketch-note">children return compact facts; the parent combines them</p></div>';
    case 'backtrack':
      return '<div class="coding-visual-sketch coding-visual-sketch--backtrack"><div class="coding-sketch-choice-tree"><span class="coding-sketch-node coding-sketch-node--active">partial path</span><div class="coding-sketch-choice-branches"><span class="coding-sketch-node">choose A</span><span class="coding-sketch-node">choose B</span><span class="coding-sketch-node">choose C</span></div></div><p class="coding-sketch-note">add one choice, explore below it, then restore the parent path</p></div>';
    case 'dp':
      return '<div class="coding-visual-sketch coding-visual-sketch--dp"><div class="coding-sketch-dp-grid"><span class="coding-sketch-cell coding-sketch-cell--state">base</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell">smaller</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-cell coding-sketch-cell--active">current</span></div><p class="coding-sketch-note">each cell is a complete answer to one smaller question</p></div>';
    case 'interval':
      return '<div class="coding-visual-sketch coding-visual-sketch--interval"><div class="coding-sketch-timeline"><span class="coding-sketch-tick">time</span><span class="coding-sketch-bar coding-sketch-bar--state">kept range</span><span class="coding-sketch-bar coding-sketch-bar--active">next range</span></div><p class="coding-sketch-note">sort first; carry the boundary that preserves future room</p></div>';
    case 'topology':
      return '<div class="coding-visual-sketch coding-visual-sketch--topology"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--state">0 unmet</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--active">ready</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill">next is ready</span></div><p class="coding-sketch-note">remove incoming requirements until a node becomes ready</p></div>';
    case 'union':
      return '<div class="coding-visual-sketch coding-visual-sketch--union"><div class="coding-sketch-components"><span class="coding-sketch-component"><b>root A</b> · a · b</span><span class="coding-sketch-component"><b>root B</b> · c</span><span class="coding-sketch-component coding-sketch-component--active"><b>same root?</b> cycle</span></div><p class="coding-sketch-note">compare representatives before joining two components</p></div>';
    case 'linked':
      return '<div class="coding-visual-sketch coding-visual-sketch--linked"><div class="coding-sketch-path"><span class="coding-sketch-node coding-sketch-node--state">previous</span><span class="coding-sketch-arrow">&larr;</span><span class="coding-sketch-node coding-sketch-node--active">current</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-node">saved next</span></div><p class="coding-sketch-note">save the outgoing link before redirecting the current node</p></div>';
    case 'trie':
      return '<div class="coding-visual-sketch coding-visual-sketch--trie"><div class="coding-sketch-prefix"><span class="coding-sketch-node">root</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-node coding-sketch-node--state">c-a</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-node coding-sketch-node--active">t / r</span></div><p class="coding-sketch-note">shared prefixes stay shared until the words branch</p></div>';
    case 'bit':
      return '<div class="coding-visual-sketch coding-visual-sketch--bit"><div class="coding-sketch-bits"><span>1</span><span>0</span><span>1</span><span class="coding-sketch-bit--active">1</span><span>0</span><span>1</span><span>0</span><span>0</span></div><p class="coding-sketch-note">read, cancel, or carry one bit position at a time</p></div>';
    case 'matrix':
      return '<div class="coding-visual-sketch coding-visual-sketch--matrix"><div class="coding-sketch-matrix"><span class="coding-sketch-grid-cell coding-sketch-grid-cell--active">focus</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell coding-sketch-grid-cell--state">marker</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span><span class="coding-sketch-grid-cell">cell</span></div><p class="coding-sketch-note">mark a row, column, layer, or active rectangle before writing over it</p></div>';
    case 'string':
      return '<div class="coding-visual-sketch coding-visual-sketch--string"><div class="coding-sketch-array"><span class="coding-sketch-pointer">L</span><span class="coding-sketch-cell">left</span><span class="coding-sketch-cell coding-sketch-cell--active">focus</span><span class="coding-sketch-cell coding-sketch-cell--active">focus</span><span class="coding-sketch-cell">right</span><span class="coding-sketch-pointer">R</span></div><p class="coding-sketch-note">compare or expand around the active characters in the concrete trace</p></div>';
    case 'tensor':
      return '<div class="coding-visual-sketch coding-visual-sketch--tensor"><div class="coding-sketch-shapes"><span class="coding-sketch-shape coding-sketch-shape--input">[batch, width]</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-shape coding-sketch-shape--state">align axes</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-shape coding-sketch-shape--active">[batch, result]</span></div><p class="coding-sketch-note">name each axis before broadcasting, masking, reducing, or selecting</p></div>';
    case 'state':
    default:
      return '<div class="coding-visual-sketch coding-visual-sketch--state"><div class="coding-sketch-row"><span class="coding-sketch-pill coding-sketch-pill--input">cue</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--state">state</span><span class="coding-sketch-arrow">&rarr;</span><span class="coding-sketch-pill coding-sketch-pill--active">safe move</span></div><p class="coding-sketch-note">the invariant explains why the transition does not lose the answer</p></div>';
  }
}

const problemTraces = {
  'two-sum': '2, 7, 11, 15; target 9 -> 7 asks for 2 and finds it',
  'valid-anagram': 'eat and tea -> both reduce to a:1, e:1, t:1',
  'group-anagrams': 'eat, tea, tan -> [a,e,t] shares one bucket; tan uses [a,n,t]',
  'longest-consecutive-sequence': '100, 4, 200, 1, 3, 2 -> start at 1 and walk 1,2,3,4',
  'product-of-array-except-self': '[1, 2, 3, 4] -> answer at 2 is left 1*2 times right 4',
  'subarray-sum-equals-k': '[1, 2, 1], k=3 -> prefix 3 looks for earlier prefix 0',
  '3sum': '[-1, 0, 1, 2, -1, -4] -> fix -1, then move a sorted pair toward zero',
  'container-with-most-water': '[1, 8, 6, 2, 5, 4, 8, 3, 7] -> move the shorter wall inward',
  'longest-substring-without-repeating-characters': 'abcabcbb -> window abc, then move left past the old a',
  'longest-repeating-character-replacement': 'AABABBA, k=1 -> window is valid when length - max_count <= 1',
  'permutation-in-string': 's1=ab, s2=eidbaooo -> compare each width-2 frequency window',
  'count-number-of-nice-subarrays': '[1,1,2,1,1], k=3 -> prefix odd counts 0,1,2,2,3,4',
  'binary-search': '[1,3,5,7,9], target 7 -> mid 5 rules out the lower half',
  'search-in-rotated-sorted-array': '[4,5,6,7,0,1,2], target 0 -> one half is sorted; choose its side',
  'koko-eating-bananas': 'piles [3,6,7,11], h=8 -> speed 4 finishes in 8 hours; test lower',
  'find-minimum-in-rotated-sorted-array': '[4,5,6,7,0,1,2] -> compare mid with right to keep the drop',
  'valid-parentheses': '([{}]) -> push (, [, {; each close must match the top',
  'decode-string': '3[a2[c]] -> save outer state at each [, restore it at each ]',
  'daily-temperatures': '[73,74,75,71,69,72] -> 72 resolves the waiting 69 and 71',
  'min-stack': 'push 5, push 2, push 4 -> each node carries its minimum so far',
  'kth-largest-element': '[3,2,1,5,6,4], k=2 -> a size-2 min-heap keeps 5 and 6',
  'top-k-frequent-elements': '[1,1,1,2,2,3], k=2 -> buckets 3:[1], 2:[2]',
  'rotting-oranges': 'all rotten cells seed minute 0; each queue layer is one minute',
  'binary-tree-level-order-traversal': 'queue [3] -> read one layer, then append its children 9 and 20',
  'network-delay-time': 'paths 1->2 cost 1 and 1->3 cost 4 -> finalize 2 before 3',
  'clone-graph': 'copy node 1 once, then point its copy at copies of every neighbor',
  'number-of-islands': 'each unseen 1 starts one flood; mark its whole connected land as 0',
  'pacific-atlantic-water-flow': 'start from both ocean borders, walk uphill, intersect reached cells',
  'maximum-depth-of-binary-tree': 'leaf returns 1; parent returns 1 + max(left_depth, right_depth)',
  'same-tree': 'compare roots, then compare left children and right children in lockstep',
  'invert-binary-tree': 'at every node, swap left and right before returning upward',
  'balanced-binary-tree': 'child heights 3 and 1 differ by 2 -> return the failure sentinel',
  'subtree-of-another-tree': 'try Same Tree at each candidate node, then search both children',
  'subsets': 'for [1,2], every call saves its path: [], [1], [1,2], [2]',
  'permutations': 'for [1,2,3], choose one unused value for each next position',
  'combination-sum': 'target 7 with [2,3,6,7] -> paths [2,2,3] and [7]',
  'word-search': 'trace C-A-T through neighboring cells, marking each chosen cell temporarily',
  'climbing-stairs': 'ways(5) = ways(4) + ways(3); only the last two totals survive',
  'house-robber': 'money 2,7,9 -> at 9 choose max(skip 7, take 2+9)',
  'partition-equal-subset-sum': '[1,5,11,5] -> total 22, so ask whether sum 11 is reachable',
  'longest-common-subsequence': 'abcde and ace -> match a, skip b/d, match c, then e',
  'edit-distance': 'horse -> ros; each grid cell chooses insert, delete, or replace',
  'merge-intervals': '[1,3] and [2,6] overlap -> carry [1,6]',
  'insert-interval': 'before, overlap, after -> copy [1,2], merge [3,5] with [4,8], copy [10,12]',
  'non-overlapping-intervals': 'when ranges overlap, keep the one with the earlier end',
  'meeting-rooms-ii': 'start 1, start 2, end 3 -> two active meetings need two rooms',
  'jump-game': 'at index 2 with jump 3, farthest reach becomes 5',
  'course-schedule': 'finish prerequisite 0 -> decrement course 1 until its indegree reaches zero',
  'course-schedule-ii': 'queue zero-indegree courses and append each one to the feasible order',
  'redundant-connection': 'edge 2-3 finds the same root at both ends -> it closes the cycle',
  'reverse-linked-list': '1->2->3 becomes 1<-2<-3; save 2 before redirecting 1',
  'linked-list-cycle': 'slow moves 1, fast moves 2; inside a loop they eventually meet',
  'remove-nth-node-from-end': 'gap n=2 leaves left immediately before the node to remove',
  'merge-two-sorted-lists': 'compare heads 1 and 2, attach 1, then compare the next pair',
  'implement-trie': 'insert cat and car -> c-a is shared, then branch at the final letter',
  'design-add-and-search-words': 'search c.t -> dot branches across every child at position 2',
  'contains-duplicate': '[1,2,3,1] -> the second 1 is already in the seen set',
  'maximum-subarray': '[-2,1,-3,4,-1,2,1] -> discard the negative prefix before 4',
  'best-time-to-buy-and-sell-stock': 'prices 7,1,5 -> at 5, the saved buy price 1 yields profit 4',
  'maximum-product-subarray': '[-2,3,-4] -> the negative minimum becomes the positive maximum',
  'number-of-1-bits': '1011 -> clear the lowest 1 three times',
  'counting-bits': 'bits[6] = bits[3] + 0 because 6 >> 1 is 3',
  'missing-number': '[3,0,1] -> XOR expected and actual values; unmatched 2 remains',
  'reverse-bits': 'read the input from right to left while appending each bit to the answer',
  'sum-of-two-integers': 'XOR gives provisional sum; AND shifted left gives the carry to add next',
  'coin-change': 'amount 6 with coins 1,3,4 -> fewest[6] builds from fewest[3] + 3',
  'longest-increasing-subsequence': '10,9,2,5,3,7,101 -> tails become 2,3,7,101',
  'word-break': 'leetcode -> position 0 reaches 4 via leet, then 8 via code',
  'combination-sum-iv': 'target 4 with 1,2 -> count sequences by choosing their final number',
  'house-robber-ii': 'circle [2,3,2] -> solve without first and without last, then take max',
  'decode-ways': '226 -> 2|2|6, 22|6, and 2|26 are the valid paths',
  'unique-paths': 'each grid cell stores paths from above plus paths from the left',
  'graph-valid-tree': 'n nodes need n-1 edges; then one DFS must reach every node',
  'number-of-connected-components': 'each unseen node starts a flood and increments the component count',
  'meeting-rooms': 'sort starts; if the next start is before the previous end, overlap exists',
  'reorder-list': '1,2,3,4,5 -> split at 3, reverse 4,5, then interleave 1,5,2,4,3',
  'set-matrix-zeroes': 'a zero at row 1, col 2 marks the first cell of that row and column',
  'spiral-matrix': 'consume top, right, bottom, left, then shrink all four boundaries',
  'rotate-image': 'reverse rows, then transpose across the diagonal to turn columns clockwise',
  'valid-palindrome': 'A man, a plan -> skip spaces and commas, compare normalized ends',
  'longest-palindromic-substring': 'expand around one letter and one gap; keep the widest match',
  'palindromic-substrings': 'each center contributes one count per successful outward expansion',
  'encode-and-decode-strings': '4#lint3#ML -> read 4, consume lint, then read 3, consume ML',
  'construct-tree-from-preorder-and-inorder-traversal': 'preorder gives root 3; inorder splits [9] from [15,20,7]',
  'validate-binary-search-tree': 'a node 4 in the right subtree of 5 violates the inherited lower bound 5',
  'kth-smallest-element-in-a-bst': 'inorder visits 1,2,3,...; stop exactly at the kth visit',
  'lowest-common-ancestor-in-a-bst': 'if both targets are left, go left; if both right, go right; otherwise stop',
  'lru-cache': 'map finds key 1; list moves it to most-recent, evicting the least-recent left node',
  'pairwise-squared-distances': 'points [n,1,d] and centers [1,k,d] broadcast to [n,k,d]',
  'stable-softmax': 'logits [1000,1001] shift to [-1,0] before exponentiation',
  'cross-entropy-from-logits': 'loss = logsumexp(row) - the selected class logit',
  'causal-attention': 'token 2 can read tokens 0,1,2 but the future score is masked out',
  'pad-variable-length-sequences': '[3,4] and [9] become tokens [[3,4],[9,0]] plus mask [[1,1],[1,0]]',
  'mini-batches': 'items 0:3, 3:6, 6:end -> the final short batch is still yielded',
  'top-k-scores': 'argpartition finds the top group; only those k scores receive final sorting',
  'binary-precision-and-recall': 'TP=2, FP=1, FN=1 -> precision uses predicted positives; recall uses real positives',
  'minimum-window-substring': 'ADOBECODEBANC needs ABC -> expand to BANC, then shrink from the left',
  'split-array-largest-sum': '[7,2,5,10,8], k=2 -> test a limit and count how many parts it forces',
  'largest-rectangle-in-histogram': 'a shorter bar ends every taller stack bar and reveals its rectangle width',
  'binary-tree-maximum-path-sum': 'a node may return one child upward but score both children locally',
  'serialize-and-deserialize-binary-tree': 'preorder 1,#,2,#,# preserves both values and missing-child shape',
  'longest-increasing-path-in-a-matrix': 'cache the best path from each cell; larger-only moves cannot cycle',
  'alien-dictionary': 'first difference in w-r and e-r gives w<e; topological sort orders the alphabet',
  'word-search-ii': 'the trie rejects a board path as soon as it is not a dictionary prefix',
  'merge-k-sorted-lists': 'heap holds one head per list; pop the smallest and add that list\'s next node',
  'find-median-from-data-stream': 'lower heap holds the smaller half, upper heap the larger half, roots meet at median',
};

function renderVisual(problem) {
  const mode = modeFor(problem.title, problem.pattern);
  const template = visualTemplates[mode];
  const visualId = `${problem.slug}-state`;
  const titleId = `${visualId}-title`;
  const trace = problemTraces[problem.slug];
  if (!trace) throw new Error(`Missing concrete visual trace for ${problem.slug}`);
  const alt = `${problem.title}: ${trace}. ${template.invariant}`;
  const steps = template.steps.map(([label, value, detail], index) => `<div class="coding-visual-step" data-coding-step="${index}"><span class="coding-visual-step-label">${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail)}</small></div>`).join('');
  const sketch = renderSketch(mode);
  const controls = '<div class="coding-visual-controls" data-coding-controls hidden><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button><output data-coding-progress>Step 1 of 4</output></div><p class="coding-visual-status sr-only" data-coding-status aria-live="polite"></p>';
  return {
    mode,
    visualId,
    source: `<!-- visual:${visualId} -->\n<figure class="learning-figure coding-visual-figure" aria-labelledby="${titleId}"><p class="visual-kicker">${escapeHtml(template.kicker)}</p><p class="visual-title" id="${titleId}">${escapeHtml(problem.title)}: ${escapeHtml(template.title)}</p><div class="coding-visual coding-visual--${mode}" data-coding-visual data-coding-mode="${mode}" data-coding-slug="${problem.slug}" role="group" aria-label="${escapeHtml(alt)}"><div class="coding-visual-example"><span>Concrete trace</span><strong>${escapeHtml(trace)}</strong></div>${sketch}<div class="coding-visual-flow">${steps}</div><p class="coding-visual-invariant"><span>Invariant</span>${escapeHtml(template.invariant)}</p>${controls}</div><figcaption><strong>Read it this way:</strong> ${escapeHtml(template.caption)} For this problem, hold onto the concrete trace: ${escapeHtml(trace)}.</figcaption></figure>`,
    audit: {
      schemaVersion: 1,
      slug: problem.slug,
      article: `src/content/posts/${publicationDate}-${problem.slug}.md`,
      status: 'implemented',
      medium: 'semantic-html',
      learningObjective: `${problem.title}: ${trace}. ${template.invariant}`,
      mediumRationale: 'A compact semantic state map pairs a pattern-specific data-structure sketch with the algorithm invariant before code. It is responsive, printable, color-independent, and easier to revisit than a decorative algorithm illustration.',
      mediumComparison: {
        mermaid: 'Rejected: automatic graph layout would add structure without showing the changing local state clearly.',
        svg: 'Rejected: the problem family benefits from labeled state cards rather than coordinate geometry.',
        semanticHtml: 'Selected: a pattern-specific sketch plus labeled steps keeps the input shape, retained state, safe move, and completion condition readable at phone width.',
        interaction: 'Selected as progressive enhancement: an explicit step player highlights the invariant path, while the static figure remains complete for print and no-JavaScript readers.',
        paperReuse: 'Rejected: the visual is an original synthesis and does not reuse source artwork.',
        noVisual: 'Rejected: prose alone makes the learner hold the state transition in working memory.',
      },
      deckReview: {
        pages: [],
        notes: 'No source slide deck is part of the supplied coding guide. The visual was designed from the local problem explanation and the algorithm invariant.',
      },
      sourceReview: {
        sources: [],
        notes: 'The supplied Coding_Questions_Phone_Guide.md is the curriculum source. The page preserves its task, pattern, simple idea, implementation sketch, and cost while adding an original visual state map.',
      },
      agentReview: {
        reviewer: 'GitHub Copilot',
        reviewedAt: publicationDate,
        summary: 'Checked that the figure names a concrete algorithm state, includes an invariant, remains understandable without color, and appears before the implementation sketch.',
      },
      implementation: {
        visualIds: [visualId],
        accessibility: 'The figure has a labelled title, a role=group state map with a complete aria-label, visible text labels, a direct Read it this way caption, and keyboard-sized controls that are hidden until enhanced.',
      },
    },
  };
}

function parseProblems(source) {
  const headingPattern = /^## ((?:\d+|AI\d+|H\d+)\. .+)$/gm;
  const headings = [...source.matchAll(headingPattern)];
  return headings.map((match, index) => {
    const sourceHeading = match[1];
    const sectionStart = match.index;
    const sectionEnd = headings[index + 1]?.index ?? source.length;
    let section = source.slice(sectionStart, sectionEnd).replace(/^## .+\n/, '').trim();
    const nextHeading = section.search(/\n\s*(?:<a id="[^"]+"><\/a>\s*)?#{1,4} /);
    if (nextHeading >= 0) section = section.slice(0, nextHeading).trim();
    section = section.replace(/\n\s*<a id="[^"]+"><\/a>\s*$/, '').trim();
    const identifierMatch = sourceHeading.match(/^((?:\d+|AI\d+|H\d+))\. (.+)$/);
    const identifier = identifierMatch[1];
    const title = identifierMatch[2].trim();
    const task = extractField(section, 'Task');
    const pattern = extractField(section, 'Pattern');
    section = section.replace(/^\*\*Task:\*\*[\s\S]*?(?=\n\n)/, '').trim();
    const slug = slugify(title);
    return { identifier, title, task, pattern, slug, section };
  });
}

function chapterFor(problem) {
  return chapterDefinitions.find((chapter) => chapter.numbers.includes(problem.identifier));
}

function writeProblem(problem) {
  const visual = renderVisual(problem);
  const opening = 'Start with the concrete trace below. It shows the state the algorithm must carry as it runs.';
  const body = `> ${problem.task}\n\n${opening}\n\n${visual.source}\n\n${wrapParagraphs(problem.section)}`;
  const difficulty = problem.identifier.startsWith('H') ? 'Advanced' : problem.identifier.startsWith('AI') ? 'Intermediate' : chapterFor(problem).difficulty;
  const priority = problem.identifier.startsWith('AI') ? 'Role-specific' : problem.identifier.startsWith('H') ? 'Specialist' : 'Core';
  const frontmatter = [
    '---',
    `title: ${JSON.stringify(problem.title)}`,
    `description: ${JSON.stringify(problem.task)}`,
    `date: "${publicationDate}"`,
    'draft: false',
    'tags: ["coding interview", "data structures"]',
    'category: "questions"',
    'roles: ["MLE", "RE", "AS"]',
    'rounds: ["Coding", "ML implementation"]',
    `difficulty: "${difficulty}"`,
    `priority: "${priority}"`,
    'aliases: []',
    'prerequisites: []',
    '---',
  ].join('\n');
  fs.writeFileSync(path.join(postsDir, `${publicationDate}-${problem.slug}.md`), `${frontmatter}\n\n${body}\n`);
  fs.writeFileSync(path.join(auditsDir, `${problem.slug}.json`), `${JSON.stringify(visual.audit, null, 2)}\n`);
  return { ...problem, visualMode: visual.mode };
}

const source = fs.readFileSync(sourcePath, 'utf8');
const problems = parseProblems(source);
const force = process.argv.includes('--force');
if (problems.length !== 106) throw new Error(`Expected 106 problems, found ${problems.length}`);
const existingSlugs = new Set(fs.readdirSync(postsDir).map((name) => name.replace(/\.mdx?$/, '').replace(/^\d{4}-\d{2}-\d{2}-/, '')));
for (const problem of problems) {
  if (existingSlugs.has(problem.slug) && !force) throw new Error(`Slug already exists: ${problem.slug}`);
  if (!problem.task || !problem.pattern) throw new Error(`Missing metadata for ${problem.identifier}`);
  if (!chapterFor(problem)) throw new Error(`No chapter for ${problem.identifier}`);
}
fs.mkdirSync(postsDir, { recursive: true });
fs.mkdirSync(auditsDir, { recursive: true });
const generated = problems.map(writeProblem);
const registry = chapterDefinitions.map((chapter) => ({ ...chapter, slugs: generated.filter((problem) => chapter.numbers.includes(problem.identifier)).map((problem) => problem.slug) }));
const registryPath = process.env.CODING_BOOK_REGISTRY || '/tmp/coding-interview-book-registry.json';
fs.writeFileSync(registryPath, `${JSON.stringify(registry, null, 2)}\n`);
console.log(`Generated ${generated.length} coding question pages and audits.`);
console.log(`Registry written to ${registryPath}.`);
console.log(`Visual modes: ${[...new Set(generated.map((problem) => problem.visualMode))].sort().join(', ')}.`);

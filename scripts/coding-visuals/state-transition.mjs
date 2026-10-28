import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';

const escapeHtml = (value) => String(value)
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');

function display(value) {
  if (Array.isArray(value)) return `[${value.map(display).join(', ')}]`;
  if (value === null) return 'null';
  if (typeof value === 'object') {
    if (value.from !== undefined && value.to !== undefined) return `${display(value.from)} -> ${display(value.to)}`;
    return value.label ?? value.id ?? JSON.stringify(value);
  }
  return String(value);
}

export function retainedState(scene) {
  const entries = new Map();
  const add = (identity, label, value, priority = 1) => {
    if (value !== undefined) entries.set(identity, { label, value: display(value), priority });
  };
  for (const pointer of Array.isArray(scene.marks) ? scene.marks : []) {
    const identity = pointer.key ?? pointer.label?.split(/[=:]/)[0] ?? `${pointer.row}:${pointer.col}`;
    const position = pointer.index ?? (pointer.row === undefined ? undefined : `(${pointer.row}, ${pointer.col})`);
    add(`pointer:${identity}`, pointer.key ?? pointer.label ?? 'position', position, 4);
  }
  for (const [key, value] of Array.isArray(scene.map) ? scene.map : []) add(`map:${key}`, `key ${key}`, value, 5);
  const values = Array.isArray(scene.items) ? scene.items : Array.isArray(scene.values) ? scene.values : [];
  for (const [index, value] of values.entries()) {
    add(`value:${index}`, `index ${index}`, value, 1);
  }
  for (const [rowIndex, row] of (Array.isArray(scene.rows) ? scene.rows : []).entries()) {
    if (Array.isArray(row)) {
      for (const [columnIndex, value] of row.entries()) {
        add(`cell:${rowIndex}:${columnIndex}`, `${scene.columns?.[columnIndex] ?? `column ${columnIndex}`} / row ${rowIndex}`, value, 2);
      }
    }
  }
  for (const key of ['queue', 'stack', 'visited', 'path', 'prefix', 'bits', 'order', 'edges', 'stateNodes']) {
    if (Array.isArray(scene[key])) add(key, key.replaceAll(/([A-Z])/g, ' $1').toLowerCase(), scene[key], 3);
  }
  for (const [index, node] of (Array.isArray(scene.nodes) ? scene.nodes : []).entries()) {
    const identity = typeof node === 'object' && node !== null ? node.id ?? index : index;
    const value = typeof node === 'object' && node !== null ? node.value ?? node.label ?? node.id : node;
    add(`node:${identity}`, `node ${identity}`, value, 2);
  }
  for (const [index, group] of (Array.isArray(scene.groups) ? scene.groups : []).entries()) {
    add(`group:${index}`, group.label ?? `group ${index}`, group.values, 2);
  }
  const structural = new Set(['type', 'label', 'mapLabel', 'arraySummary', 'gridSummary', 'motion', 'fitArray', 'arrayColumns', 'width', 'height']);
  for (const [key, value] of Object.entries(scene)) {
    if (structural.has(key) || !['string', 'number', 'boolean'].includes(typeof value)) continue;
    if (display(value).length > 64) continue;
    add(`state:${key}`, key.replaceAll(/([A-Z])/g, ' $1').replaceAll('-', ' ').toLowerCase(), value, 3);
  }
  return entries;
}

export function checkpointChanges(scene, previous) {
  const current = retainedState(scene);
  const before = previous ? retainedState(previous) : new Map();
  const identities = new Set([...current.keys(), ...before.keys()]);
  return [...identities].flatMap((identity) => {
    const next = current.get(identity);
    const prior = before.get(identity);
    if (prior?.value === next?.value) return [];
    return [{ identity, label: next?.label ?? prior.label, before: prior?.value, after: next?.value, priority: next?.priority ?? prior.priority }];
  }).sort((left, right) => right.priority - left.priority);
}

function mechanism(scene) {
  const label = (horizontal, vertical, value, tone = 'c-text', size = 14) => `<text x="${horizontal}" y="${vertical}" style="font: ${size}px var(--font-mono);text-anchor:middle;fill:var(--${tone})">${escapeHtml(value)}</text>`;
  const line = (fromX, fromY, toX, toY, tone = 'viz-neutral-stroke', dashed = false) => `<path d="M${fromX} ${fromY}L${toX} ${toY}" style="fill:none;stroke:var(--${tone});stroke-width:1.75${dashed ? ';stroke-dasharray:4 3' : ''}"></path>`;
  const cell = (horizontal, vertical, width, value, tone = 'neutral') => `<rect x="${horizontal}" y="${vertical}" width="${width}" height="33" rx="2" style="fill:var(--viz-${tone}-bg);stroke:var(--viz-${tone}-stroke)"></rect>${label(horizontal + width / 2, vertical + 22, value)}`;
  let body = '';
  let height;
  let description;
  if (Number.isFinite(scene.lookupTarget)) {
    const pointer = scene.marks.find((mark) => mark.key === 'scan-cursor');
    const index = pointer.index;
    const value = Number(scene.items[index]);
    const needed = scene.lookupTarget - value;
    const saved = scene.map.find(([key]) => Number(key) === needed);
    const earlier = saved && Number(String(saved[1]).replace('index ', '')) < index;
    const queryPhase = ['miss', 'hit', 'return'].includes(scene.lookupPhase);
    description = queryPhase ? `Complement lookup: ${scene.lookupTarget} minus ${value} is ${needed}. ${earlier ? `Map entry ${needed} points to ${saved[1]}, an earlier index.` : 'No earlier matching index is in the displayed map.'}` : 'Checkpoint map after storage or before the next query; storage is not a successful lookup.';
    body = label(180, 25, `CURRENT i=${index}; value=${value}`)
      + cell(15, 52, 94, scene.lookupTarget, 'input') + label(124, 75, '-')
      + cell(139, 52, 82, value, 'focus') + label(237, 75, '=')
      + cell(253, 52, 92, needed, 'state')
      + label(180, 119, queryPhase ? 'QUERY BEFORE STORING CURRENT' : scene.lookupPhase === 'saved' ? 'AFTER THE STORE' : 'BEFORE THE QUERY', 'c-text-soft', 12)
      + line(299, 85, 299, 145, 'viz-state-stroke');
    const mapRows = scene.map.length ? scene.map : [['empty', 'no earlier indices']];
    for (const [row, [key, location]] of mapRows.entries()) {
      body += cell(25, 148 + row * 43, 100, key, earlier && Number(key) === needed ? 'output' : 'neutral')
        + label(150, 171 + row * 43, '->')
        + cell(180, 148 + row * 43, 160, location, earlier && Number(key) === needed ? 'output' : 'neutral');
    }
    height = 220 + mapRows.length * 43;
    body += label(180, height - 23, queryPhase ? earlier ? `${needed} -> ${saved[1]}; return [${String(saved[1]).replace('index ', '')},${index}]` : 'MISS: current cannot pair with itself.' : 'Store for a later index, never itself.', 'c-text-soft', 12);
  } else if (scene.coveredRange && scene.total) {
    const pointer = (key) => scene.marks.find((mark) => mark.key === key)?.index;
    const fixed = pointer('fixed');
    const left = pointer('left');
    const right = pointer('right');
    if (left === undefined || right === undefined) return '';
    const total = Number(scene.items[fixed]) + Number(scene.items[left]) + Number(scene.items[right]);
    description = `Sorted pair range ${left} through ${right} with fixed index ${fixed}. The selected sum is ${total}. ${total < 0 ? 'Even the largest remaining partner is too small for this left value; the entire left row of pairs is rejected.' : total > 0 ? 'Even the smallest remaining partner is too large for this right value; the entire right column is rejected.' : 'The selected values form a zero-sum triple.'}`;
    for (const [index, value] of scene.items.entries()) body += cell(15 + index * 56, 32, 49, value, index === fixed ? 'state' : index >= left && index <= right ? 'focus' : 'neutral') + label(39.5 + index * 56, 23, index, 'c-text-soft', 12);
    body += line(15 + left * 56, 78, 64 + right * 56, 78, 'viz-focus-stroke')
      + label(180, 113, scene.total, 'c-text', 14);
    const rejected = [];
    if (total < 0) for (let partner = left + 1; partner <= right; partner++) rejected.push(`(${left},${partner})`);
    if (total > 0) for (let partner = left; partner < right; partner++) rejected.push(`(${partner},${right})`);
    body += label(180, 148, total === 0 ? 'KEEP the triple; move both pair bounds' : total < 0 ? 'Every pair using this L is too small' : 'Every pair using this R is too large', total === 0 ? 'viz-output-stroke' : 'viz-warning-stroke', 12)
      + label(180, 180, total === 0 ? `[${scene.items[fixed]}, ${scene.items[left]}, ${scene.items[right]}]` : rejected.join(' '), 'c-text', 12);
    height = 204;
  } else if (scene.preorder && scene.inorder) {
    const preorder = JSON.parse(scene.preorder);
    const inorder = JSON.parse(scene.inorder);
    const root = Number(scene.activeRoot);
    const preIndex = preorder.indexOf(root);
    const inIndex = inorder.indexOf(root);
    description = `Both traversals stay visible. Root ${root} links preorder position ${preIndex} to inorder position ${inIndex}; inorder positions left and right of the root determine subtree regions within the active call.`;
    body = label(180, 23, 'PREORDER: take the next root', 'c-text-soft', 12);
    for (const [index, value] of preorder.entries()) body += cell(26 + index * 64, 42, 52, value, value === root ? 'focus' : 'neutral');
    body += line(52 + preIndex * 64, 75, 52 + inIndex * 64, 136, 'viz-focus-stroke', true)
      + label(180, 122, 'INORDER: split around that root', 'c-text-soft', 12);
    for (const [index, value] of inorder.entries()) body += cell(26 + index * 64, 137, 52, value, value === root ? 'focus' : 'neutral');
    body += label(180, 206, scene.split ?? 'First split: [9] | 3 | [15,20,7]', 'c-text', 12)
      + label(180, 239, 'Tree uses consumed preorder roots only.', 'c-text-soft', 12);
    height = 263;
  } else if (scene.input === 'aaa' && scene.runningTotal !== undefined) {
    const intervals = [[0, 1, 0], [0, 2, 0], [1, 2, 1], [0, 3, 1], [1, 3, 1], [2, 3, 2]];
    const count = Number(scene.runningTotal);
    description = `Six distinct palindrome occurrences in aaa, using half-open intervals. Solid intervals have been counted at this checkpoint, ${count} in total; dashed intervals have not. Equal text at distinct positions still represents different occurrences.`;
    body = label(180, 24, `COUNTED ${count} / 6; intervals are [start,end)`);
    for (let index = 0; index < 3; index++) body += cell(150 + index * 58, 49, 48, 'a', 'input') + label(174 + index * 58, 42, index, 'c-text-soft', 12);
    for (const [index, [start, end, center]] of intervals.entries()) {
      const vertical = 111 + index * 35;
      body += label(73, vertical + 4, `m${center} [${start},${end})`, 'c-text-soft', 12)
        + line(150 + start * 58, vertical, 140 + end * 58, vertical, index < count ? 'viz-output-stroke' : 'viz-neutral-stroke', index >= count);
    }
    body += label(180, 355, '6 interval occurrences; only 3 distinct strings.', 'c-text-soft', 12);
    height = 380;
  } else if (scene.items?.length === 32 && scene.indexLabels?.[0] === 31) {
    const source = scene.items.join('');
    const processed = scene.sourceBitPosition === undefined ? 0 : Number(scene.sourceBitPosition) + 1;
    const bit = Math.max(0, processed - 1);
    const finalOrder = [...source].reverse().map((value, index) => index < processed ? value : '.').join('');
    const spaced = (value) => value.match(/.{8}/g).join(' ');
    const position = (index) => 29 + index * 9.1 + Math.floor(index / 8) * 4;
    const strip = (value, vertical) => [...value].map((digit, index) => label(position(index), vertical, digit)).join('');
    description = `Fixed-width reversal has processed ${processed} of thirty-two bits. Original source position ${bit} maps to final destination ${31 - bit}. Dots mark final positions not processed yet; the actual current accumulator is shown separately.`;
    body = label(180, 23, `${processed} / 32 steps; original i -> final 31-i`)
      + label(180, 57, 'ORIGINAL input: position 31 -> 0', 'c-text-soft', 12)
      + strip(source, 86)
      + line(position(31 - bit), 96, position(bit), 135, 'viz-focus-stroke', true)
      + strip(finalOrder, 160)
      + label(180, 190, 'FINAL destinations, not the current accumulator', 'c-text-soft', 12)
      + label(180, 231, `current: ${spaced(scene.answerBits ?? scene.answer)}`, 'c-text', 12)
      + label(180, 269, processed >= 26 && processed < 32 ? `Input is zero; ${32 - processed} required shifts remain.` : 'Zero bits need output positions too.', 'c-text-soft', 12);
    height = 294;
  } else return '';
  return `<div class="coding-mechanism"><svg viewBox="0 0 360 ${height}" role="img" aria-label="${escapeHtml(description)}" style="display:block;width:100%;max-width:34rem;margin:auto"><title>${escapeHtml(description)}</title>${body}</svg></div>`;
}

export function renderStateTransition(scene, previous) {
  const changes = checkpointChanges(scene, previous);
  const diagram = mechanism(scene);
  if (!changes.length) return diagram;
  const selected = changes.slice(0, 3);
  const rows = selected.map((change) => {
    const before = change.before === undefined ? '<span class="coding-state-absent">not shown</span>' : escapeHtml(change.before);
    const after = change.after === undefined ? '<span class="coding-state-absent">not shown</span>' : escapeHtml(change.after);
    const prior = previous ? `<span class="coding-state-before">${before}</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span>` : '';
    return `<div class="coding-state-change"><dt>${escapeHtml(change.label)}</dt><dd>${prior}<strong class="coding-state-after">${after}</strong></dd></div>`;
  }).join('');
  return `${diagram}<div class="coding-state-comparison" role="group" aria-label="${previous ? 'Selected values shown at consecutive authored checkpoints' : 'Selected values shown at the first checkpoint'}"><dl>${rows}</dl></div>`;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const before = { type: 'array-map', items: [3, 2, 4], marks: [{ index: 0, label: 'i=0', key: 'scan' }], map: [] };
  const after = { ...before, marks: [{ index: 1, label: 'i=1', key: 'scan' }], map: [[3, 0]] };
  const frozen = JSON.stringify([before, after]);
  assert.deepEqual(checkpointChanges(after, before).map(({ label, before, after }) => ({ label, before, after })), [
    { label: 'key 3', before: undefined, after: '0' },
    { label: 'scan', before: '0', after: '1' },
  ]);
  assert.match(renderStateTransition(after, before), /key 3/);
  assert.match(renderStateTransition(after, before), /not shown/);
  assert.match(renderStateTransition(before, after), /not shown/);
  assert.doesNotMatch(renderStateTransition(before), /coding-state-before|coding-state-arrow|not shown/);
  assert.doesNotMatch(renderStateTransition(after, before), /not retained|no longer retained|&empty;/);
  assert.doesNotMatch(renderStateTransition(after, before), /data-motion-key|\bid=/);
  assert.equal(renderStateTransition(before, before), '');
  assert.match(renderStateTransition({ type: 'array', items: ['<script>'] }), /&lt;script&gt;/);
  assert.match(renderStateTransition({ type: 'attention', values: 'V0', rows: 'Q0' }), /V0/);
  assert.doesNotMatch(renderStateTransition({ type: 'graph', nodes: [{ id: 'A', x: 90, y: 80 }], width: 360 }), />360<|>90<|>80</);
  assert.equal(JSON.stringify([before, after]), frozen);
  const { parseFragment } = await import('parse5');
  const elements = (source, name) => {
    const found = [];
    function walk(node) {
      if (node.nodeName === name) found.push({ attributes: Object.fromEntries((node.attrs ?? []).map((attribute) => [attribute.name, attribute.value])), text: (node.childNodes ?? []).map((child) => child.value ?? '').join('') });
      for (const child of node.childNodes ?? []) walk(child);
    }
    walk(parseFragment(source));
    return found;
  };
  const bitSource = (43261596).toString(2).padStart(32, '0');
  const bitResult = [...bitSource].reverse().join('');
  assert.equal(Number.parseInt(bitResult, 2), 964176192);
  for (let processed = 0; processed <= 32; processed++) {
    const scene = { items: [...bitSource], indexLabels: Array.from({ length: 32 }, (_, index) => 31 - index), sourceBitPosition: processed ? processed - 1 : undefined, answerBits: Number.parseInt(bitResult.slice(0, processed) || '0', 2).toString(2).padStart(32, '0') };
    const drawing = mechanism(scene);
    const labels = elements(drawing, 'text');
    const original = labels.filter(({ attributes }) => attributes.y === '86');
    const destinations = labels.filter(({ attributes }) => attributes.y === '160');
    assert.equal(original.map(({ text }) => text).join(''), bitSource);
    assert.equal(destinations.map(({ text }) => text).join(''), bitResult.slice(0, processed).padEnd(32, '.'));
    if (processed) {
      const endpoints = elements(drawing, 'path')[0].attributes.d.match(/-?\d+(?:\.\d+)?/g).map(Number);
      assert.equal(endpoints[0], Number(original[32 - processed].attributes.x), 'bit connector must start at the actual source digit');
      assert.equal(endpoints[2], Number(destinations[processed - 1].attributes.x), 'bit connector must end at the actual final destination digit');
    }
  }
  for (let counted = 0; counted <= 6; counted++) {
    const drawing = mechanism({ input: 'aaa', runningTotal: counted });
    const intervals = elements(drawing, 'text').map(({ text }) => text).filter((text) => /^m\d \[\d,\d\)$/.test(text));
    assert.equal(new Set(intervals).size, 6);
    assert.equal(elements(drawing, 'path').filter(({ attributes }) => !attributes.style.includes('stroke-dasharray')).length, counted);
    assert.deepEqual(intervals, ['m0 [0,1)', 'm0 [0,2)', 'm1 [1,2)', 'm1 [0,3)', 'm1 [1,3)', 'm2 [2,3)']);
  }
  const pairScene = { items: [-4, -2, 1, 2], coveredRange: '1..3', total: '-4', marks: [{ key: 'fixed', index: 0 }, { key: 'left', index: 1 }, { key: 'right', index: 3 }] };
  assert.match(mechanism(pairScene), /\(1,2\) \(1,3\)/);
  assert.doesNotMatch(mechanism(pairScene), /\(2,3\)/);
  assert.match(mechanism({ ...pairScene, items: [-1, 0, 2, 3], total: '2' }), /\(1,3\) \(2,3\)/);
  const { codingQuestionVisuals } = await import('../coding-question-visuals.mjs');
  let compared = 0;
  for (const definition of Object.values(codingQuestionVisuals)) {
    const original = JSON.stringify(definition);
    let visible = 0;
    for (const [index, item] of definition.frames.entries()) {
      const rendered = renderStateTransition(item.scene, definition.frames[index - 1]?.scene);
      assert.doesNotMatch(rendered, /undefined|\[object Object\]|data-motion-key|\bid=/, `${definition.slug}/${item.key}`);
      if (rendered) visible++;
      compared++;
    }
    assert(visible, `${definition.slug} needs at least one concrete retained-state comparison`);
    assert.equal(JSON.stringify(definition), original, `${definition.slug} was mutated`);
  }
  console.log(`State comparisons: ${Object.keys(codingQuestionVisuals).length} illustrations / ${compared} checkpoints; escaping, identity, and immutability checks passed.`);
}
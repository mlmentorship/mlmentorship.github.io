import assert from 'node:assert/strict';
import { visualDrawingTools } from './illustration-arguments.mjs';

const { text, line, circle, color, fill, escape } = visualDrawingTools;

function rectangle(horizontal, vertical, width, height, tone = 'input', dashed = false) {
  return `<rect x="${horizontal}" y="${vertical}" width="${width}" height="${height}" rx="3" style="fill:${fill(tone)};stroke:${color(tone)};stroke-width:1.75${dashed ? ';stroke-dasharray:5 4' : ''}"></rect>`;
}

function arrow(fromX, fromY, toX, toY, tone = 'focus') {
  const angle = Math.atan2(toY - fromY, toX - fromX);
  const firstX = toX - 7 * Math.cos(angle - Math.PI / 6);
  const firstY = toY - 7 * Math.sin(angle - Math.PI / 6);
  const secondX = toX - 7 * Math.cos(angle + Math.PI / 6);
  const secondY = toY - 7 * Math.sin(angle + Math.PI / 6);
  return line(fromX, fromY, toX, toY, tone) + line(firstX, firstY, toX, toY, tone) + line(secondX, secondY, toX, toY, tone);
}

function node(horizontal, vertical, width, label, tone = 'input') {
  return rectangle(horizontal, vertical, width, 48, tone) + text(horizontal + width / 2, vertical + 21, label, { width: width - 12, anchor: 'middle', weight: 700, tone });
}

function cells(values, horizontal, vertical, tone = 'input', width = 28) {
  const cellWidth = Math.max(width, ...values.map(value => String(value).length * 9 + 10));
  return values.map((value, index) => rectangle(horizontal + index * cellWidth, vertical, cellWidth - 2, 28, tone) + text(horizontal + index * cellWidth + (cellWidth - 2) / 2, vertical + 19, value, { width: cellWidth - 4, anchor: 'middle', weight: 700 })).join('');
}

function adamBuffers() {
  const gradient = [2, 4];
  const direction = gradient.map(value => .5 * value);
  const squared = gradient.map(value => .5 * value * value);
  const correctedDirection = direction.map(value => value / .5);
  const correctedSquared = squared.map(value => value / .5);
  const moves = correctedDirection.map((value, index) => .1 * value / Math.sqrt(correctedSquared[index]));
  assert.deepEqual(direction, [1, 2]);
  assert.deepEqual(squared, [2, 8]);
  assert.deepEqual(moves, [.1, .1]);
  let body = text(20, 25, 'One gradient feeds two memories', { weight: 700 });
  body += node(118, 53, 124, 'Gradient') + cells(gradient, 152, 108);
  body += arrow(145, 135, 90, 178) + arrow(215, 135, 270, 178);
  body += node(20, 184, 142, 'Direction memory', 'focus') + node(198, 184, 142, 'Size memory', 'warning');
  body += cells(direction, 63, 246, 'focus') + cells(squared, 241, 246, 'warning');
  body += arrow(90, 278, 155, 316) + arrow(270, 278, 205, 316);
  body += node(100, 323, 160, 'Correct zero-start bias', 'neutral');
  body += arrow(180, 374, 180, 397) + node(94, 405, 172, 'Scale direction by size', 'output');
  body += cells(moves, 132, 467, 'output', 48);
  body += text(180, 520, 'Unequal gradients; equal first moves', { width: 320, anchor: 'middle', weight: 700 });
  body += text(20, 577, 'Next gradient: the second input is zero', { weight: 700 });
  body += cells([2, 0], 20, 604) + arrow(86, 618, 144, 618) + cells([1.5, 1], 155, 604, 'focus');
  body += text(20, 664, 'Direction memory still contains one.', { width: 320 });
  return { body, height: 700 };
}

function alienPrefix() {
  let body = text(20, 26, 'A shared prefix supplies no new letter rule', { weight: 700 });
  for (const [index, allowed] of [false, true].entries()) {
    const top = 77 + index * 220;
    const first = allowed ? ['a', 'b'] : ['a', 'b', 'c'];
    const second = allowed ? ['a', 'b', 'c'] : ['a', 'b'];
    body += text(20, top, allowed ? 'Shorter word first' : 'Longer word first', { weight: 700 });
    body += cells(first, 30, top + 26, 'input', 42) + cells(second, 30, top + 93, 'output', 42);
    body += line(51, top + 56, 51, top + 90, 'focus') + line(93, top + 56, 93, top + 90, 'focus');
    body += text(230, top + 60, allowed ? 'valid' : 'invalid', { tone: allowed ? 'output' : 'warning', weight: 700, width: 100 });
    if (!allowed) body += line(170, top + 35, 198, top + 84, 'warning') + line(198, top + 35, 170, top + 84, 'warning');
  }
  body += text(20, 520, 'Changing the alphabet cannot fix a longer word placed before its entire prefix.', { width: 320 });
  return { body, height: 595 };
}

function collectiveContracts(pair) {
  const inputs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
  const reduced = inputs[0].map((_, index) => inputs.reduce((sum, row) => sum + row[index], 0));
  const joined = inputs.flat();
  assert.deepEqual(reduced, [12, 15, 18]);
  let body = text(20, 25, 'Same three ranks; different required outputs', { weight: 700 });
  inputs.forEach((row, index) => { body += text(24, 65 + index * 37, `R${index}`, { width: 40 }) + cells(row, 78, 46 + index * 37, ['input', 'focus', 'warning'][index]); });
  const contracts = [
    ['Reduce: add, root only', [reduced, null, null]],
    ['All-reduce: add, everyone', [reduced, reduced, reduced]],
    ['Gather: join, root only', [joined, null, null]],
    ['All-gather: join, everyone', [joined, joined, joined]],
    ['Reduce-scatter: add, split ownership', reduced.map(value => [value])],
    ['Broadcast: copy the root', Array(3).fill(inputs[0])],
  ];
  contracts.slice(pair * 2, pair * 2 + 2).forEach(([label, rows], index) => {
    const top = 195 + index * 163;
    body += text(20, top, label, { width: 320, weight: 700 });
    rows.forEach((row, rank) => {
      body += text(24, top + 40 + rank * 33, `R${rank}`, { width: 40 });
      body += row?.length === 9 ? inputs.map((block, origin) => cells(block, 78 + origin * 84, top + 21 + rank * 33, ['input', 'focus', 'warning'][origin])).join('') : row ? cells(row, 78, top + 21 + rank * 33, 'output', 40) : rectangle(78, top + 21 + rank * 33, 118, 28, 'neutral', true);
    });
  });
  body += text(20, 535, 'Empty outline: that rank receives no result.', { width: 320 });
  return { body, height: 590 };
}

function adamDecay() {
  const request = .001;
  assert.equal(request / 10, .0001);
  let body = text(20, 25, 'The decay request is the same: 0.001', { weight: 700 });
  body += text(20, 69, 'Frozen scale 10', { weight: 700 });
  body += circle(180, 113, 20, 'input') + text(180, 118, 'w', { anchor: 'middle', width: 30, weight: 700 });
  body += arrow(163, 131, 92, 158, 'input') + arrow(197, 131, 274, 158, 'output');
  body += node(30, 166, 126, 'Through scale', 'input') + node(204, 166, 126, 'Direct decay', 'output');
  body += arrow(92, 219, 92, 266, 'input') + arrow(274, 219, 274, 266, 'output');
  body += rectangle(42, 277, 10, 36, 'input') + rectangle(220, 277, 100, 36, 'output');
  body += text(92, 344, '.0001', { anchor: 'middle', width: 100, tone: 'input', weight: 700 }) + text(274, 344, '.001', { anchor: 'middle', width: 100, tone: 'output', weight: 700 });
  body += text(20, 393, 'Frozen scale 1: both shrink equally', { weight: 700 });
  body += rectangle(42, 432, 100, 36, 'input') + rectangle(220, 432, 100, 36, 'output');
  body += text(92, 498, '.001', { anchor: 'middle', width: 100, tone: 'input', weight: 700 }) + text(274, 498, '.001', { anchor: 'middle', width: 100, tone: 'output', weight: 700 });
  return { body, height: 530 };
}

function anchorDecode() {
  const anchor = { center: [10, 20], size: [4, 8] };
  const delta = [.5, -.25, Math.log(2), 0];
  const decoded = { center: anchor.center.map((value, index) => value + anchor.size[index] * delta[index]), size: anchor.size.map((value, index) => value * Math.exp(delta[index + 2])) };
  assert.deepEqual(decoded.center, [12, 18]);
  assert.deepEqual(decoded.size, [8, 8]);
  const scale = 20;
  const origin = [20, 580];
  const coordinate = ([horizontal, vertical]) => [origin[0] + horizontal * scale, origin[1] - vertical * scale];
  const drawBox = (box, tone, dashed) => {
    const corner = coordinate([box.center[0] - box.size[0] / 2, box.center[1] + box.size[1] / 2]);
    const center = coordinate(box.center);
    return rectangle(corner[0], corner[1], box.size[0] * scale, box.size[1] * scale, tone, dashed) + circle(center[0], center[1], 4, tone, true);
  };
  let body = text(20, 25, 'Shift the center, then resize the reference', { weight: 700 });
  body += drawBox(decoded, 'output', false) + drawBox(anchor, 'input', true);
  const oldCenter = coordinate(anchor.center);
  const newCenter = coordinate(decoded.center);
  body += arrow(...oldCenter, ...newCenter, 'focus');
  body += text(24, 92, 'Reference: center 10, 20', { width: 300, tone: 'input' });
  body += text(24, 114, 'Width 4; height 8', { width: 300, tone: 'input' });
  body += text(24, 448, 'Decoded: center 12, 18', { width: 300, tone: 'output' });
  body += text(24, 473, 'Width 8; height 8', { width: 300, tone: 'output' });
  body += text(24, 523, 'Half a reference width moves right.', { width: 300 });
  body += text(24, 567, 'A quarter reference height moves down.', { width: 300 });
  body += text(24, 611, 'Width doubles; height stays unchanged.', { width: 300 });
  return { body, height: 650 };
}

function ecosystemFeedback() {
  let body = text(20, 25, 'One ranking policy enters two feedback loops', { weight: 700 });
  body += node(112, 67, 136, 'Ranking policy', 'focus');
  body += arrow(145, 118, 80, 151) + arrow(215, 118, 279, 151);
  body += node(20, 161, 130, 'Viewer exposure') + node(210, 161, 130, 'Creator opportunity', 'warning');
  body += arrow(85, 213, 85, 247) + arrow(275, 213, 275, 247);
  body += node(20, 256, 130, 'Viewer response', 'output') + node(210, 256, 130, 'Creator response', 'warning');
  body += text(84, 350, 'earlier', { width: 100, anchor: 'middle', tone: 'output' }) + text(276, 350, 'later', { width: 100, anchor: 'middle', tone: 'warning' });
  body += arrow(85, 309, 85, 383, 'output') + arrow(275, 309, 275, 383, 'warning');
  body += node(20, 393, 130, 'Fast readout', 'output') + node(210, 393, 130, 'Future supply', 'warning');
  body += `<path d="M20 417H8V91H108M340 417H352V91H251" style="fill:none;stroke:${color('neutral')};stroke-width:1.75"></path>`;
  body += arrow(8, 91, 108, 91, 'output') + arrow(352, 91, 251, 91, 'warning');
  body += line(174, 149, 174, 445, 'neutral', true);
  body += text(20, 495, 'The fast readout can close before the supply response is observed.', { width: 320 });
  return { body, height: 570 };
}

function assistantRequest() {
  let body = text(20, 25, 'The caret divides fixed code sides', { weight: 700 });
  body += cells(['x =', '?', '+ 1'], 45, 66, 'input', 90);
  body += node(20, 127, 130, 'Permitted context') + node(210, 127, 130, 'Request v7', 'focus');
  body += arrow(85, 178, 145, 215) + arrow(275, 178, 215, 215);
  body += node(108, 222, 144, 'Fill middle', 'output');
  body += cells(['x ='], 45, 296, 'input', 90) + cells(['2'], 135, 296, 'output', 90) + cells(['+ 1'], 225, 296, 'input', 90);
  body += text(20, 352, 'Reply remains bound to request v7', { weight: 700 });
  body += text(20, 394, 'Recheck version, deadline, and policy', { width: 320 });
  body += arrow(162, 418, 85, 444, 'output') + arrow(198, 418, 275, 444, 'warning');
  body += node(20, 452, 130, 'All pass: apply', 'output') + node(210, 452, 130, 'Any fails: drop', 'warning');
  return { body, height: 540 };
}

function codingRebuild() {
  const values = [3, 2, 4];
  const current = 2;
  const target = 6;
  const saved = new Map(values.slice(0, current).map((value, index) => [value, index]));
  assert.equal(saved.get(target - values[current]), 1);
  let body = text(20, 25, 'Cue: find an earlier partner', { weight: 700 });
  body += cells(values.slice(0, current), 38, 65, 'input', 68) + cells([values[current]], 174, 65, 'focus', 68);
  values.forEach((value, index) => { body += text(70 + index * 68, 124, `index ${index}`, { width: 60, anchor: 'middle' }); });
  body += text(280, 83, 'target 6', { width: 70, tone: 'output', weight: 700 });
  body += text(20, 171, 'State: values keep earlier indices', { weight: 700 });
  body += circle(80, 229, 24, 'input') + text(80, 235, '2', { width: 40, anchor: 'middle', weight: 700 });
  body += arrow(109, 229, 220, 229) + circle(250, 229, 24, 'output') + text(250, 235, '1', { width: 40, anchor: 'middle', weight: 700 });
  body += text(80, 279, 'value', { width: 70, anchor: 'middle' }) + text(250, 279, 'index', { width: 70, anchor: 'middle' });
  body += text(20, 325, 'Invariant: the current index is absent', { weight: 700 });
  body += cells([1], 64, 363, 'output', 56) + line(159, 350, 159, 406, 'neutral', true) + cells([2], 221, 363, 'focus', 56);
  body += text(92, 431, 'earlier', { width: 90, anchor: 'middle' }) + text(249, 431, 'current', { width: 90, anchor: 'middle' });
  body += text(20, 476, 'Move: query before storing', { weight: 700 });
  body += cells([1, 2], 123, 519, 'output', 56) + text(180, 586, 'Two distinct original positions', { width: 300, anchor: 'middle' });
  return { body, height: 620 };
}

function moderationFeedback() {
  let body = text(20, 25, 'A score enters a versioned decision route', { weight: 700 });
  body += node(20, 65, 130, 'Scores M7') + node(210, 65, 130, 'Policy P7', 'focus') + arrow(154, 89, 206, 89);
  body += text(85, 145, '0.62', { width: 70, anchor: 'middle', tone: 'input', weight: 700 });
  body += arrow(275, 117, 213, 173) + node(110, 181, 140, 'Human review P7', 'focus');
  body += arrow(180, 233, 180, 264) + node(110, 272, 140, 'Action C1', 'output');
  body += arrow(147, 324, 85, 358, 'output') + arrow(213, 324, 275, 358, 'warning');
  body += node(20, 366, 130, 'Outcome log', 'neutral') + node(210, 366, 130, 'Appeal?', 'warning');
  body += arrow(85, 418, 85, 453, 'output') + arrow(275, 418, 275, 453, 'warning');
  body += node(20, 461, 130, 'Audited evidence', 'output') + node(210, 461, 130, 'Adjudicate', 'warning');
  body += arrow(206, 485, 154, 485, 'warning');
  body += `<path d="M20 485H8V89H16" style="fill:none;stroke:${color('neutral')};stroke-width:1.75;stroke-dasharray:5 4"></path>`;
  body += arrow(8, 89, 16, 89, 'neutral');
  return { body, height: 540 };
}

function impactSettlement() {
  const capacity = 20;
  const states = [
    { label: 'Reserve twelve before dispatch', held: 12, confirmed: 0 },
    { label: 'Timeout: external outcome unknown', held: 12, confirmed: 0 },
    { label: 'Receipt: twelve effects confirmed', held: 0, confirmed: 12 },
  ];
  let body = text(20, 25, 'Granted work still needs impact admission', { weight: 700 });
  body += node(20, 69, 96, 'Grant') + node(132, 69, 96, 'Reserve', 'focus') + node(244, 69, 96, 'Effect', 'output');
  body += arrow(119, 93, 129, 93) + arrow(231, 93, 241, 93);
  body += text(20, 151, 'Amber: held. Green: confirmed.', { width: 320 });
  states.forEach((state, index) => {
    const free = capacity - state.held - state.confirmed;
    assert.equal(free, 8);
    const top = 203 + index * 108;
    body += text(20, top, state.label, { weight: 700 });
    for (let slot = 0; slot < capacity; slot += 1) body += rectangle(20 + slot * 16, top + 27, 14, 26, slot < state.held ? 'focus' : slot < state.held + state.confirmed ? 'output' : 'neutral', slot >= state.held + state.confirmed);
    body += text(20, top + 80, `${state.held} held; ${state.confirmed} confirmed; ${free} free`, { width: 320 });
  });
  body += text(20, 557, 'Next request twelve: only eight fit', { weight: 700 });
  for (let slot = 0; slot < 12; slot += 1) body += rectangle(20 + slot * 26, 595, 24, 28, slot < 8 ? 'neutral' : 'warning', slot < 8);
  return { body, height: 660 };
}

function artifactBinding() {
  let body = text(20, 25, 'Repository R; state identities differ', { weight: 700 });
  body += node(20, 67, 130, 'Patch r1, o7') + node(210, 67, 130, 'Target r2, o8', 'focus');
  body += arrow(85, 118, 85, 153) + node(20, 162, 130, 'Checks r1, o7', 'input');
  body += arrow(153, 91, 201, 91, 'warning') + text(180, 146, 'mismatch', { width: 100, anchor: 'middle', tone: 'warning', weight: 700 });
  body += text(20, 249, 'Clean text application is not validation', { weight: 700 });
  body += arrow(275, 118, 275, 289, 'focus') + node(204, 298, 136, 'Reconcile r2, o8', 'focus');
  body += arrow(272, 350, 272, 385, 'output') + node(204, 394, 136, 'Retest r2, o8', 'output');
  body += arrow(272, 446, 213, 481, 'output') + node(90, 489, 180, 'Developer acceptance', 'output');
  body += `<path d="M20 91H8V322H200" style="fill:none;stroke:${color('input')};stroke-width:1.75;stroke-dasharray:5 4"></path>` + arrow(8, 322, 200, 322, 'input');
  return { body, height: 575 };
}

function redTeamCycle() {
  let body = text(20, 25, 'One scoped finding enters an owned loop', { weight: 700 });
  const stages = [['Scoped attack', 'input'], ['Saved reproduction', 'input'], ['Owned mitigation', 'focus'], ['Adaptive retest', 'output']];
  stages.forEach(([label, tone], index) => {
    const top = 67 + index * 94;
    body += node(20, top, 154, label, tone);
    if (index < stages.length - 1) body += arrow(97, top + 53, 97, top + 85, tone);
  });
  body += node(210, 161, 130, 'Legitimate tasks', 'input');
  body += arrow(275, 214, 275, 290, 'input') + arrow(178, 373, 231, 413, 'output') + arrow(275, 294, 275, 410, 'input');
  body += node(204, 422, 136, 'Release decision', 'focus');
  body += arrow(200, 446, 158, 446, 'output') + node(20, 422, 134, 'Regression cases', 'output');
  body += `<path d="M340 446H352V91H180" style="fill:none;stroke:${color('neutral')};stroke-width:1.75;stroke-dasharray:5 4"></path>` + arrow(352, 91, 180, 91, 'neutral');
  body += text(20, 521, 'Production evidence updates the scope.', { width: 320 });
  return { body, height: 570 };
}

function flashTileResidency() {
  let body = text(20, 25, 'Query stays; each temporary score tile leaves', { weight: 700 });
  for (let index = 0; index < 2; index += 1) {
    const top = 76 + index * 230;
    body += text(20, top, `Visit ${index + 1}`, { width: 90, weight: 700 }) + text(130, top, 'SRAM', { width: 90, tone: 'neutral', weight: 700 });
    body += rectangle(121, top + 12, 223, 158, 'neutral', true);
    body += node(130, top + 25, 88, 'Query', 'input') + node(235, top + 25, 99, 'Row state', 'focus');
    body += node(20, top + 103, 90, `K${index + 1}, V${index + 1}`, 'input');
    body += node(130, top + 103, 88, `Score tile ${index + 1}`, 'warning');
    body += arrow(174, top + 77, 174, top + 96, 'input') + arrow(114, top + 127, 126, top + 127, 'input');
    body += arrow(222, top + 127, 284, top + 127, 'warning') + arrow(284, top + 127, 284, top + 78, 'focus');
    body += text(236, top + 158, 'Merge', { width: 94, tone: 'focus' });
    body += rectangle(132, top + 187, 82, 28, 'warning');
    body += line(127, top + 183, 219, top + 219, 'warning') + line(219, top + 183, 127, top + 219, 'warning');
    body += text(20, top + 207, 'Discard', { width: 100, tone: 'warning' });
  }
  body += `<path d="M334 355H352V540H284" style="fill:none;stroke:${color('focus')};stroke-width:1.75"></path>`;
  body += arrow(284, 540, 284, 555, 'focus') + node(211, 565, 130, 'Final output', 'output');
  body += text(20, 565, 'Normalize the retained row', { width: 160, weight: 700 });
  return { body, height: 650 };
}

function gpuTileReuse() {
  let body = text(20, 25, 'One fitting tile serves four input rows', { weight: 700 });
  body += node(20, 80, 90, 'HBM', 'input') + node(215, 80, 125, 'On-chip tile', 'focus') + arrow(114, 104, 211, 104, 'input');
  body += text(20, 160, 'Load weights', { width: 100 }) + text(217, 160, 'Reuse weights', { width: 125, tone: 'focus' });
  body += rectangle(182, 203, 158, 264, 'neutral', true);
  body += `<path d="M340 104H352V413" style="fill:none;stroke:${color('focus')};stroke-width:1.75"></path>`;
  for (let index = 0; index < 4; index += 1) {
    const top = 233 + index * 60;
    body += circle(48, top, 19, 'input') + text(48, top + 5, `Q${index + 1}`, { width: 35, anchor: 'middle', weight: 700 });
    body += arrow(72, top, 194, top, 'input') + node(201, top - 24, 126, 'Register work', 'output');
    body += arrow(352, top, 331, top, 'focus');
  }
  body += text(20, 507, 'Four rows of arithmetic; one tile load', { weight: 700 });
  body += text(20, 555, 'Input reads and output writes remain.', { width: 320 });
  return { body, height: 610 };
}

function gcnSymmetricRow() {
  const degrees = [2, 3, 2, 1];
  const features = [1, 2, 3, 100];
  const coefficients = degrees.map((degree, index) => index === 3 ? 0 : 1 / Math.sqrt(3 * degree));
  const aggregate = coefficients.reduce((sum, value, index) => sum + value * features[index], 0);
  assert(Math.abs(aggregate - 2.2996598285221186) < 1e-12);
  let body = text(20, 25, 'B receives only its declared message routes', { weight: 700 });
  const centers = [54, 139, 224, 309];
  body += arrow(82, 99, 110, 99, 'input') + arrow(196, 99, 168, 99, 'input');
  centers.forEach(horizontal => { body += `<path d="M${horizontal - 14} 78C${horizontal - 42} 46 ${horizontal + 44} 46 ${horizontal + 15} 78" style="fill:none;stroke:${color('focus')};stroke-width:1.75"></path>` + arrow(horizontal + 22, 67, horizontal + 15, 78, 'focus'); });
  centers.forEach((horizontal, index) => { body += circle(horizontal, 99, 24, index === 1 ? 'focus' : 'input') + text(horizontal, 105, ['A', 'B', 'C', 'D'][index], { width: 40, anchor: 'middle', weight: 700 }) + text(horizontal, 155, features[index], { width: 60, anchor: 'middle' }); });
  body += text(20, 198, 'B row: both endpoint degrees count', { weight: 700 });
  body += cells(coefficients.map(value => value.toFixed(3)), 20, 239, 'focus', 80);
  body += cells(features, 20, 301, 'input', 80);
  centers.forEach(horizontal => { body += arrow(horizontal, 271, horizontal, 293, 'focus'); });
  [60, 140, 220, 300].forEach(horizontal => { body += arrow(horizontal, 336, 180, 372, 'input'); });
  body += node(114, 380, 132, `Sum ${aggregate.toFixed(3)}`, 'output');
  body += arrow(180, 432, 180, 466, 'output') + node(103, 475, 154, 'Scale 2, then ReLU', 'output');
  body += text(180, 576, (2 * aggregate).toFixed(3), { width: 100, anchor: 'middle', tone: 'output', weight: 700 });
  return { body, height: 610 };
}

function loraFrozenPath() {
  const input = [1, 0, 0, 0];
  const projection = [2, 0, 0, 0].reduce((sum, value, index) => sum + value * input[index], 0);
  const adapter = [0, 0, 0, 0].map(value => value * projection);
  assert.deepEqual(input.map((value, index) => value + adapter[index]), input);
  assert.equal(projection * 3, 6);
  let body = text(20, 25, 'One input takes the base and adapter routes', { weight: 700 });
  body += cells(input, 124, 68, 'input', 28);
  body += arrow(145, 102, 86, 137, 'input') + arrow(214, 102, 269, 137, 'input');
  body += node(20, 145, 132, 'Frozen W', 'neutral') + node(207, 145, 132, 'Trainable A', 'focus');
  body += cells(input, 28, 222, 'input', 28) + cells([projection], 257, 222, 'focus', 28);
  body += arrow(273, 254, 273, 278, 'focus') + node(207, 286, 132, 'B starts zero', 'focus');
  body += cells(adapter, 214, 363, 'focus', 28);
  body += arrow(86, 254, 164, 423, 'input') + arrow(269, 396, 199, 423, 'focus');
  body += circle(180, 446, 21, 'output') + text(180, 452, '+', { width: 30, anchor: 'middle', weight: 700 });
  body += cells(input, 124, 505, 'output', 28);
  body += text(20, 578, 'Initial output equals the frozen base', { weight: 700 });
  return { body, height: 630 };
}

function distillationOwner() {
  const temperature = 2;
  const probabilities = logits => {
    const weights = logits.map(value => Math.exp(value / temperature));
    const total = weights.reduce((sum, value) => sum + value, 0);
    return weights.map(value => value / total);
  };
  const target = probabilities([2, 1, 0]);
  const initial = probabilities([0, 0, 0]);
  const updated = probabilities(initial.map((value, index) => -.1 * temperature * (value - target[index])));
  assert(updated[0] > initial[0] && updated[2] < initial[2]);
  const labels = values => values.map(value => value.toFixed(3).slice(1));
  let body = text(20, 25, 'Fixed teacher; only the student changes', { weight: 700 });
  body += node(20, 69, 138, 'Frozen teacher', 'input') + node(202, 69, 138, 'Student', 'focus');
  body += cells(labels(target), 20, 155, 'input', 44) + cells(labels(initial), 202, 155, 'focus', 44);
  body += text(20, 222, 'Targets A, B, C', { width: 145 }) + text(202, 222, 'Predictions A, B, C', { width: 145 });
  body += arrow(91, 244, 148, 291, 'input') + arrow(273, 244, 212, 291, 'focus');
  body += node(115, 299, 130, 'Match targets', 'neutral');
  body += arrow(212, 352, 253, 379, 'output') + node(204, 387, 136, 'Student update', 'output');
  body += cells(labels(updated), 202, 466, 'output', 44);
  body += `<path d="M340 411H352V93H344" style="fill:none;stroke:${color('output')};stroke-width:1.75"></path>` + arrow(352, 93, 344, 93, 'output');
  body += text(20, 545, 'Teacher weights receive no update', { weight: 700 });
  return { body, height: 590 };
}

function encounterEvidence() {
  let body = text(20, 25, 'Task: patient P1, February encounter E2', { weight: 700 });
  body += rectangle(20, 77, 320, 87, 'warning') + text(35, 105, 'E1 / P1 / January 10', { weight: 700 });
  body += text(35, 136, 'No prior procedure recorded', { width: 270 });
  body += text(20, 194, 'Authentic passage; wrong encounter', { tone: 'warning', weight: 700 });
  body += rectangle(20, 236, 320, 98, 'input') + text(35, 264, 'E2 / P1 / February 10', { weight: 700 });
  body += text(35, 295, 'PX1 completed February 9', { width: 270 });
  body += arrow(180, 341, 180, 369, 'input') + node(20, 377, 320, 'Draft preserves event and source dates', 'focus');
  body += arrow(180, 432, 180, 463, 'focus') + node(90, 471, 180, 'Review exact draft', 'neutral');
  body += arrow(180, 526, 180, 557, 'output') + node(90, 565, 180, 'Accept summary', 'output');
  body += text(20, 658, 'No diagnosis, prescription, or clinical action', { tone: 'warning', weight: 700 });
  return { body, height: 710 };
}

function evalCaseCalibration() {
  const reference = [1, 0, 1];
  const judge = [1, 1, 1];
  assert.deepEqual(reference.map((value, index) => value === judge[index]), [true, false, true]);
  let body = text(20, 25, 'Calibrate on the same reviewed cases', { weight: 700 });
  body += text(20, 91, 'Production cases', { weight: 700 });
  body += cells(['C1', 'C2', 'C3'], 131, 127, 'input', 64);
  body += arrow(180, 166, 180, 194, 'input');
  body += rectangle(197, 200, 68, 111, 'warning', true);
  body += text(20, 224, 'Reference', { width: 100 }) + cells(['pass', 'fail', 'pass'], 131, 204, 'input', 64);
  body += text(20, 295, 'Judge', { width: 100 }) + cells(['pass', 'pass', 'pass'], 131, 275, 'focus', 64);
  body += arrow(230, 318, 180, 351, 'warning') + node(89, 359, 182, 'Inspect C2 disagreement', 'warning');
  body += node(89, 451, 182, 'Revise and retest judge', 'focus') + arrow(180, 414, 180, 443, 'focus');
  body += node(89, 560, 182, 'New production cases', 'input');
  body += `<path d="M82 584H8V147H96" style="fill:none;stroke:${color('input')};stroke-width:1.75"></path>` + arrow(87, 147, 96, 147, 'input');
  body += text(20, 665, 'New case mix reopens calibration', { weight: 700 });
  return { body, height: 710 };
}

function longContextStateAxes() {
  let body = text(20, 25, 'Different axes need different techniques', { weight: 700 });
  body += text(20, 93, 'Position coverage', { weight: 700 });
  body += cells([0, 1, 2, 3], 20, 126, 'input', 34) + cells([4, 5, 6, 7], 181, 126, 'warning', 34);
  body += text(20, 199, 'Training range', { width: 155 }) + text(181, 199, 'New range: validate', { width: 155 });
  body += text(20, 288, 'Temporary score residency', { weight: 700 });
  for (let row = 0; row < 4; row += 1) for (let column = 0; column < 4; column += 1) body += rectangle(30 + column * 18, 322 + row * 18, 16, 16, 'input');
  body += arrow(123, 356, 215, 356, 'focus');
  for (let row = 0; row < 2; row += 1) for (let column = 0; column < 2; column += 1) body += rectangle(254 + column * 18, 338 + row * 18, 16, 16, 'focus');
  body += text(20, 431, 'Full scores', { width: 140 }) + text(213, 431, 'Stream tiles', { width: 140 });
  body += text(20, 526, 'Persistent KV positions', { weight: 700 });
  body += cells([0, 1, 2, 3], 20, 565, 'input', 34);
  body += arrow(173, 581, 322, 581, 'focus') + arrow(322, 581, 322, 626, 'focus');
  body += cells([0, 1, 2, 3, 4, 5, 6, 7], 20, 637, 'output', 34);
  body += text(20, 722, 'More positions retain more logical KV', { weight: 700 });
  return { body, height: 765 };
}

function nearestOriginalRows() {
  const points = [[0, 0], [3, 0], [8, 0]];
  const query = [2, 0];
  const distances = points.map(point => Math.hypot(...point.map((value, index) => value - query[index])));
  assert.deepEqual(distances, [2, 1, 6]);
  assert.equal(distances.indexOf(Math.min(...distances)), 1);
  const horizontal = value => 26 + 35 * value;
  let body = text(20, 25, 'Measure each row; keep the nearest identity', { weight: 700 });
  body += text(horizontal(2), 105, 'Query 2', { anchor: 'middle', width: 150, tone: 'focus', weight: 700 });
  body += circle(horizontal(2), 138, 11, 'focus') + arrow(horizontal(2), 152, horizontal(2), 175, 'focus');
  body += line(26, 179, 330, 179, 'neutral');
  points.forEach((point, index) => { body += circle(horizontal(point[0]), 179, 10, 'input') + text(horizontal(point[0]), 217, point[0], { anchor: 'middle', width: 35 }); });
  points.forEach((point, index) => {
    const top = 272 + index * 81;
    const tone = index === 1 ? 'output' : 'input';
    body += text(20, top - 17, `Row ${index}: distance ${distances[index]}`, { weight: 700, tone });
    body += arrow(horizontal(2), top, horizontal(point[0]), top, tone);
  });
  body += node(20, 529, 320, 'Return original row index 1', 'output');
  return { body, height: 620 };
}

function trainingGateCycle() {
  assert.equal(-Math.log(.25), Math.log(4));
  let body = text(20, 25, 'Verify the same known signal before scaling', { weight: 700 });
  body += cells(['C0', 'C1', 'C2', 'C3'], 84, 78, 'input', 48);
  body += text(20, 143, 'Eight distinct examples per class', { width: 320 });
  body += node(85, 186, 190, 'Forward: current weights', 'neutral');
  body += text(20, 257, 'Initial:', { width: 65 });
  body += cells(['.25', '.25', '.25', '.25'], 80, 267, 'input', 50);
  body += arrow(180, 303, 180, 329, 'focus') + node(85, 337, 190, 'Backward computes gradient', 'focus');
  body += arrow(180, 392, 180, 421, 'focus') + node(85, 429, 190, 'Step changes bound weights', 'output');
  body += arrow(180, 484, 180, 513, 'output') + node(85, 521, 190, 'Retest the same examples', 'input');
  body += `<path d="M78 545H8V102H76" style="fill:none;stroke:${color('input')};stroke-width:1.75"></path>` + arrow(68, 102, 76, 102, 'input');
  body += text(20, 634, 'Only then test held-out generalization', { weight: 700 });
  return { body, height: 685 };
}

function searchExposureCycle() {
  let body = text(20, 25, 'The serving policy creates the learning record', { weight: 700 });
  body += node(85, 76, 190, 'Query and permitted context', 'input');
  body += arrow(144, 131, 89, 154, 'input') + arrow(216, 131, 273, 154, 'input');
  body += node(20, 162, 140, 'Lexical source', 'input') + node(200, 162, 140, 'Semantic source', 'input');
  body += arrow(90, 217, 145, 244, 'input') + arrow(270, 217, 215, 244, 'input');
  body += node(85, 252, 190, 'Eligibility and merge', 'focus');
  body += arrow(180, 307, 180, 331, 'focus') + node(85, 339, 190, 'Rank and list policy P1', 'focus');
  body += arrow(180, 394, 180, 420, 'focus') + node(20, 428, 320, 'Exposure: P1 / B / position 1', 'neutral');
  body += arrow(180, 483, 180, 509, 'neutral') + node(85, 517, 190, 'Join mature outcome', 'input');
  body += arrow(180, 572, 180, 594, 'input') + node(20, 602, 150, 'Evaluate and retrain', 'focus');
  body += arrow(175, 626, 192, 626, 'output') + node(200, 602, 140, 'Release checks', 'output');
  body += `<path d="M340 626H352V363H284" style="fill:none;stroke:${color('output')};stroke-width:1.75"></path>` + arrow(293, 363, 284, 363, 'output');
  body += text(20, 717, 'Unshown A has no observed outcome', { weight: 700 });
  return { body, height: 760 };
}

function exactQueryMerge() {
  const lexicalA = 1 / 61;
  const lexicalB = 1 / 62;
  const semanticB = 1 / 61;
  assert(lexicalB + semanticB > lexicalA);
  let body = text(20, 25, 'A can lead lexical retrieval and still be lost', { weight: 700 });
  body += text(20, 92, 'Lexical', { weight: 700 }) + text(210, 92, 'Semantic', { weight: 700 });
  body += node(20, 119, 130, 'A rank 1', 'input') + node(210, 119, 130, 'A absent', 'neutral');
  body += node(20, 201, 130, 'B rank 2', 'input') + node(210, 201, 130, 'B rank 1', 'input');
  body += text(20, 309, 'RRF offset 60: add only present-source scores', { weight: 700 });
  body += text(20, 378, 'A: 0.016393', { tone: 'input', weight: 700 }) + rectangle(20, 397, lexicalA * 8500, 28, 'input');
  body += text(20, 475, 'B: 0.032522', { tone: 'output', weight: 700 }) + rectangle(20, 494, (lexicalB + semanticB) * 8500, 28, 'output');
  body += node(20, 587, 320, 'Cap one keeps B; A never reaches ranking', 'warning');
  return { body, height: 680 };
}

function ragCorpusRequest() {
  let body = text(20, 25, 'Prepared corpus meets an authorized request', { weight: 700 });
  body += text(20, 91, 'Offline', { weight: 700 }) + text(206, 91, 'Online', { weight: 700 });
  body += node(20, 119, 140, 'Source D v3', 'input') + node(204, 119, 136, 'Query', 'input');
  body += arrow(90, 174, 90, 200, 'input') + arrow(272, 174, 272, 200, 'input');
  body += node(20, 208, 140, 'Chunk C7 v3', 'input') + node(204, 208, 136, 'Allowed scope', 'neutral');
  body += arrow(90, 263, 90, 289, 'input') + arrow(272, 263, 272, 289, 'input');
  body += node(20, 297, 140, 'Lexical / dense indexes', 'focus') + node(204, 297, 136, 'Retrieve / rank', 'focus');
  body += arrow(166, 321, 197, 321, 'input');
  body += arrow(272, 352, 272, 378, 'focus') + node(204, 386, 136, 'Evidence C7 v3', 'input');
  body += arrow(272, 441, 272, 467, 'input') + node(204, 475, 136, 'Claim and passage', 'focus');
  body += arrow(243, 530, 196, 556, 'focus') + node(85, 564, 190, 'Exact claim supported?', 'neutral');
  body += arrow(145, 619, 90, 654, 'output') + arrow(215, 619, 272, 654, 'warning');
  body += node(20, 662, 140, 'Release answer', 'output') + node(204, 662, 136, 'Review / refuse', 'warning');
  return { body, height: 755 };
}

function featureClockJoin() {
  let body = text(20, 25, 'Each feature joins with its own update clock', { weight: 700 });
  const labels = ['Batch snapshot', 'Stream state', 'Request context'];
  labels.forEach((label, index) => {
    const top = 97 + index * 108;
    body += node(20, top, 165, label, ['input', 'focus', 'neutral'][index]);
    body += circle(221, top + 24, 18, ['input', 'focus', 'neutral'][index]);
    body += line(221, top + 24, 221, top + 12, 'neutral') + line(221, top + 24, 230, top + 29, 'neutral');
    body += arrow(246, top + 24, 287, top + 24, 'input');
  });
  body += line(292, 121, 292, 390, 'input') + arrow(292, 390, 180, 430, 'input');
  body += node(20, 438, 320, 'One vector; preserve feature timestamps', 'output');
  body += arrow(180, 493, 180, 520, 'output') + node(89, 528, 182, 'Rank the current request', 'focus');
  body += text(20, 643, 'Feature age starts at its snapshot', { weight: 700 });
  return { body, height: 690 };
}

function environmentGraderHandoff() {
  let body = text(20, 25, 'The grader reads an actual environment episode', { weight: 700 });
  body += text(20, 95, 'Environment E1', { weight: 700 });
  body += node(20, 131, 90, 'State s0', 'input') + node(135, 131, 90, 'Action a1', 'focus') + node(250, 131, 90, 'State s1', 'input');
  body += arrow(115, 155, 129, 155, 'focus') + arrow(230, 155, 244, 155, 'focus');
  body += arrow(180, 186, 180, 219, 'input') + node(20, 227, 320, 'Episode: E1 / s0 / a1 / s1', 'neutral');
  body += arrow(180, 282, 180, 320, 'neutral');
  body += text(20, 348, 'Grader G1: two independent checks', { weight: 700 });
  body += arrow(143, 371, 90, 401, 'input') + arrow(217, 371, 270, 401, 'input');
  body += node(20, 409, 140, 'Outcome quality', 'output') + node(200, 409, 140, 'Allowed process', 'warning');
  body += arrow(90, 464, 143, 510, 'output') + arrow(270, 464, 217, 510, 'warning');
  body += node(85, 518, 190, 'Preserve both grade fields', 'neutral');
  body += text(20, 629, 'Quality cannot erase a hard process failure', { weight: 700 });
  return { body, height: 680 };
}

function shardedCrossProducts() {
  const left = [[1, 2], [3, 4]];
  const right = [[5, 6], [7, 8]];
  const result = left.map(row => right[0].map((_, column) => row.reduce((sum, value, index) => sum + value * right[index][column], 0)));
  assert.deepEqual(result, [[19, 22], [43, 50]]);
  let body = text(20, 25, 'Keep A rows; gather B columns before dot products', { weight: 700 });
  body += text(20, 98, 'R0: A row 0', { weight: 700 }) + text(200, 98, 'R1: A row 1', { weight: 700 });
  body += cells(left[0], 54, 129, 'input', 40) + cells(left[1], 234, 129, 'input', 40);
  body += text(20, 213, 'B column 0', { weight: 700 }) + text(200, 213, 'B column 1', { weight: 700 });
  body += cells([5, 7], 54, 244, 'focus', 40) + cells([6, 8], 234, 244, 'focus', 40);
  body += arrow(94, 280, 145, 314, 'focus') + arrow(274, 280, 215, 314, 'focus');
  body += node(85, 322, 190, 'Gather both B columns', 'focus');
  body += arrow(145, 377, 94, 411, 'focus') + arrow(215, 377, 274, 411, 'focus');
  body += node(20, 419, 140, 'A0 reads full B', 'output') + node(200, 419, 140, 'A1 reads full B', 'output');
  body += `<path d="M54 143H8V443H16M312 143H352V443H344" style="fill:none;stroke:${color('input')};stroke-width:1.75"></path>`;
  body += arrow(8, 443, 16, 443, 'input') + arrow(352, 443, 344, 443, 'input');
  body += cells(result[0], 54, 508, 'output', 40) + cells(result[1], 234, 508, 'output', 40);
  body += text(20, 602, '22 and 43 are new cross-products', { weight: 700 });
  return { body, height: 650 };
}

function transformerBudgetLedgers() {
  const stateBytes = [2, 2, 4, 4];
  assert.equal(stateBytes.reduce((sum, value) => sum + value, 0), 12);
  let body = text(20, 25, 'Work, model state, and KV have different units', { weight: 700 });
  body += text(20, 101, 'Training compute', { weight: 700 });
  ['Token 1', 'Token 2', 'Token 3'].forEach((label, index) => {
    body += node(20 + index * 110, 137, 100, label, 'input');
    body += arrow(70 + index * 110, 191, 70 + index * 110, 215, 'focus');
    body += rectangle(45 + index * 110, 223, 50, 28, 'focus');
  });
  body += text(20, 286, 'Active parameter work repeats per token', { width: 320 });
  body += text(20, 382, 'Training state: bytes per parameter', { weight: 700 });
  let horizontal = 20;
  ['W', 'G', 'M1', 'M2'].forEach((label, index) => {
    body += rectangle(horizontal, 422, stateBytes[index] * 20, 28, ['input', 'focus', 'neutral', 'neutral'][index]);
    body += text(horizontal + stateBytes[index] * 10, 487, `${label}: ${stateBytes[index]}`, { anchor: 'middle', width: stateBytes[index] * 20, weight: 700 });
    horizontal += stateBytes[index] * 20 + 6;
  });
  body += text(20, 576, 'Serving weights: W only, two bytes', { weight: 700 });
  body += rectangle(20, 614, 40, 28, 'input');
  body += text(20, 722, 'KV: one K/V pair per stored position', { weight: 700 });
  for (let position = 0; position < 4; position += 1) {
    body += rectangle(30 + position * 76, 763, 30, 28, 'focus') + rectangle(62 + position * 76, 763, 30, 28, 'output');
    body += text(61 + position * 76, 826, position, { anchor: 'middle', width: 45 });
  }
  body += text(20, 884, 'Repeat KV for each layer and KV head', { width: 320 });
  return { body, height: 930 };
}

function trieWildcardPrint() {
  assert.deepEqual(['bake', 'bed'].filter(word => /^b.d$/.test(word)), ['bed']);
  let body = text(20, 25, 'Pattern b.d: try each child for the dot', { weight: 700 });
  body += cells(['b', '.', 'd'], 117, 70, 'focus', 42);
  body += text(20, 128, 'Stored words: bake and bed', { weight: 700 });
  const points = [[180, 178, 'b', 'input'], [88, 273, 'a', 'warning'], [272, 273, 'e', 'output'], [88, 365, 'k', 'warning'], [272, 365, 'd', 'output'], [88, 450, 'e', 'input']];
  body += arrow(168, 194, 101, 256, 'warning') + arrow(192, 194, 259, 256, 'output');
  body += arrow(88, 291, 88, 346, 'warning') + arrow(272, 291, 272, 346, 'output');
  body += arrow(88, 383, 88, 431, 'input');
  body += circle(272, 365, 23, 'output') + circle(88, 450, 23, 'input');
  points.forEach(([horizontal, vertical, label, tone]) => { body += circle(horizontal, vertical, 18, tone) + text(horizontal, vertical + 5, label, { width: 28, anchor: 'middle', weight: 700, tone }); });
  body += text(180, 237, 'try both', { width: 88, anchor: 'middle', weight: 700 });
  body += text(128, 358, 'need d; only k', { width: 105, tone: 'warning' });
  body += text(272, 417, 'bed: true', { width: 112, anchor: 'middle', tone: 'output', weight: 700 });
  body += text(20, 508, 'Double ring: a stored word ends here.', { width: 320 });
  return { body, height: 545 };
}

function trieTerminalPrint() {
  assert(!['bake', 'bed'].includes('ba'));
  let body = text(20, 25, 'The complete query must end at a word flag', { weight: 700 });
  body += text(20, 92, 'Query ba', { weight: 700 }) + cells(['b', 'a'], 20, 116, 'input', 42);
  body += arrow(111, 130, 180, 130, 'input');
  body += circle(207, 130, 18, 'input') + text(207, 135, 'a', { width: 28, anchor: 'middle', weight: 700 });
  body += text(242, 127, 'no word flag', { width: 98, tone: 'warning' });
  body += text(20, 183, 'Existing prefix; false', { width: 320, tone: 'warning', weight: 700 });
  body += text(20, 256, 'Query b.d', { weight: 700 }) + cells(['b', '.', 'd'], 20, 280, 'focus', 42);
  body += arrow(153, 294, 205, 294, 'output');
  body += circle(232, 294, 23, 'output') + circle(232, 294, 18, 'output');
  body += text(232, 299, 'd', { width: 28, anchor: 'middle', tone: 'output', weight: 700 });
  body += text(272, 291, 'bed', { width: 65, tone: 'output', weight: 700 });
  body += text(20, 347, 'Complete stored word; true', { width: 320, tone: 'output', weight: 700 });
  body += text(20, 409, 'The dot consumes exactly one character.', { width: 320 });
  return { body, height: 455 };
}

export const lessonScenes = {
  'adam-two-memory-route': { title: 'Adam keeps direction history and size history before choosing a move', caption: 'Zero current input does not erase stored direction.', draw: adamBuffers },
  'adam-decay-bypass': { title: 'AdamW decay bypasses the adaptive gradient scale', caption: 'Bar lengths share one scale. The preconditioner is frozen to isolate decay, not simulate the full Adam recurrence.', draw: adamDecay },
  'alien-prefix-order': { title: 'A word cannot precede its complete shorter prefix', caption: 'Both words share every compared character.', draw: alienPrefix },
  'collective-add-ownership': { title: 'Reduction adds matching coordinates; the output owner changes', caption: 'Three ranks, three values per input row.', draw: () => collectiveContracts(0) },
  'collective-join-ownership': { title: 'Gathering preserves all input values; the output owner changes', caption: 'Input colors survive concatenation.', draw: () => collectiveContracts(1) },
  'collective-split-copy': { title: 'Reduce-scatter splits the sum; broadcast copies the root', caption: 'A split sum and a copied original have different values and owners.', draw: () => collectiveContracts(2) },
  'anchor-reference-decode': { title: 'The same reference box supplies units for the center shift and size change', caption: 'The reference is dashed; the decoded box is solid.', draw: anchorDecode },
  'ecosystem-two-feedback-loops': { title: 'Viewer feedback and creator supply return on different clocks', caption: 'Arrows show coupling, not a guaranteed gain or loss.', draw: ecosystemFeedback },
  'coding-assistant-request-route': { title: 'Frame one editor request, fill its middle, and recheck current acceptance', caption: 'Illustrative completion. Preserving code sides and authorized context does not bypass acceptance checks.', draw: assistantRequest },
  'coding-rebuild-four-questions': { title: 'Rebuild Two Sum from a cue, minimal state, an invariant, and a move', caption: 'Only the relevant saved map entry is shown. The target is six; the current four is at original index two.', draw: codingRebuild },
  'moderation-review-feedback': { title: 'Policy governs actions; review and appeals supply auditable corrections', caption: 'Appeals are selected corrections. Training needs audited, representative evidence; a score alone authorizes nothing.', draw: moderationFeedback },
  'agent-impact-settlement': { title: 'Reserve impact before effects; an unknown outcome keeps the reservation held', caption: 'This gate cannot enlarge a grant. Confirmed means authoritatively recorded, not guessed. An oversized next request is refused, not silently narrowed.', draw: impactSettlement },
  'coding-artifact-state-binding': { title: 'Patch evidence stays attached to the state that was tested', caption: 'Reconcile and retest under the original task and API contract. Current evidence still does not replace developer acceptance.', draw: artifactBinding },
  'red-team-evidence-cycle': { title: 'A finding becomes an owned mitigation, adaptive retest, and retained regression', caption: 'Release requires residual-risk and legitimate-task evidence. Unknown paths remain unknown, not safe.', draw: redTeamCycle },
  'flashattention-tile-residency': { title: 'Stream tiles, retain row summaries, discard scores, and produce the output', caption: 'The same query and row-state buffers persist. Each visit updates the maximum, normalization mass, and weighted-value numerator; temporary scores are not written as a global matrix.', draw: flashTileResidency },
  'gpu-fitting-tile-reuse': { title: 'Reuse a loaded weight tile across a batch before reading it again', caption: 'Illustrative work on one fitting tile. Reuse can reduce traffic per row; it does not remove input/output traffic or prove a measured speedup.', draw: gpuTileReuse },
  'gcn-symmetric-neighbor-row': { title: 'Graph routes become a symmetrically normalized message row', caption: 'Path A-B-C with isolated D and self-loops. Displayed coefficients are rounded; the sum uses exact endpoint-degree weights.', draw: gcnSymmetricRow },
  'lora-frozen-additive-path': { title: 'A low-rank adapter adds a trainable branch to frozen base weights', caption: 'Four-dimensional identity base, rank one, scale one. A projects to two; zero B makes the adapter a no-op. Both factors remain trainable.', draw: loraFrozenPath },
  'distillation-student-update': { title: 'The teacher supplies soft targets; the student learns to match them', caption: 'Classes A, B, C; temperature two. One soft-target-only update uses T-squared loss scaling and rate 0.1. Shares are rounded; teacher weights stay fixed.', draw: distillationOwner },
  'healthcare-encounter-evidence': { title: 'Select evidence for the requested encounter before accepting a summary', caption: 'The fictional P1/E2 task. E1 is authentic but cannot support the February claim. E2 records February 9 as event time and February 10 as source time. Review authorizes the exact summary, not clinical actions.', draw: encounterEvidence },
  'eval-matched-case-calibration': { title: 'Production cases connect reviewed judgments to scalable checks', caption: 'Three illustrative reviewed cases, not a deployment threshold. The judge disagrees on C2; inspect and retest rather than promote that score. New traffic reopens calibration. Hard requirements remain independent.', draw: evalCaseCalibration },
  'long-context-state-axes': { title: 'Positions, temporary scores, and retained KV are distinct state problems', caption: 'Illustrative four/eight positions. New positions need behavioral validation. Streaming score tiles changes residency, not dense arithmetic. More unique positions still retain more KV; allocation packing addresses a different cost.', draw: longContextStateAxes },
  'knn-query-original-indices': { title: 'Exact nearest-neighbor search scores every reference row', caption: 'The declared points [0,0], [3,0], [8,0] and query [2,0], with k one. Second coordinates match; horizontal lengths are Euclidean distances. Keep original row one, not its value or sorted position.', draw: nearestOriginalRows },
  'training-update-retest-cycle': { title: 'A verified update changes the weights before the known batch is rescored', caption: 'The declared 32-example, four-class fixture starts with equal quarter shares. Backward alone does not change weights. Verify the optimizer-bound update, then fit the same known signal; fitting does not establish held-out quality.', draw: trainingGateCycle },
  'search-exposure-learning-cycle': { title: 'Retain source, policy, and exposure identities from serving into learning', caption: 'The declared A/B example shows only B after the one-candidate cap. Record actual exposure and policy before joining mature outcomes. Release checks precede a learned policy change. Logging cannot create outcomes for unshown A.', draw: searchExposureCycle },
  'search-exact-query-merge': { title: 'Fusion can remove the exact lexical result before either ranker sees it', caption: 'The declared RRF example uses offset sixty and cap one. A is lexical first but absent from semantic retrieval. B receives both contributions and wins. Scores share one bar scale; a final ranker cannot restore excluded A.', draw: exactQueryMerge },
  'rag-corpus-request-join': { title: 'Indexing prepares evidence that retrieval selects for the current request', caption: 'Source D version three yields chunk C7 in the prepared indexes. The authorized request retrieves evidence and joins each claim to an exact supporting passage. Matching an identity is not enough to establish support; unsupported claims require review or refusal.', draw: ragCorpusRequest },
  'personalization-feature-clock-join': { title: 'Batch, stream, and request-time values join without sharing one freshness clock', caption: 'Qualitative update clocks, not claimed service intervals. Carry feature timestamps into one vector; judge each against its required freshness. A fast request does not make an old snapshot new.', draw: featureClockJoin },
  'environment-grader-episode': { title: 'Versioned transition evidence crosses from the environment into grading', caption: 'Illustrative one-action episode. Environment E1 owns transitions and trace; grader G1 owns outcome and process judgments over that same record. Keep a hard failure separate from quality instead of averaging it away.', draw: environmentGraderHandoff },
  'sharded-gather-cross-products': { title: 'Gather an input before computing the missing cross-index blocks', caption: 'A is [[1,2],[3,4]] and B is [[5,6],[7,8]]. Each rank retains one A row. Gathering both B columns enables complete output rows: retain diagonal 19 and 50, compute previously absent 22 and 43. Adding diagonal outputs alone cannot create them.', draw: shardedCrossProducts },
  'transformer-four-budget-ledgers': { title: 'Compute, training state, inference weights, and KV require separate ledgers', caption: 'Work blocks are qualitative, not a FLOP estimate. BF16 weights/gradients and FP32 Adam moments occupy twelve bytes per parameter without master weights; serving weights alone use two. KV follows stored positions, head width, KV heads, layers, and entry precision. Other buffers are excluded.', draw: transformerBudgetLedgers },
  'trie-wildcard-print-branches': { title: 'A dot tries child alternatives before the remaining suffix', caption: 'The same bake/bed fixture as the screen lesson. The a branch lacks d; the e branch reaches terminal bed.', draw: trieWildcardPrint },
  'trie-wildcard-print-terminal': { title: 'A complete pattern succeeds only at a stored word terminal', caption: 'The existing ba prefix is not a word. The complete b.d pattern matches bed; each dot consumes one character.', draw: trieTerminalPrint },
};

export function renderLessonScene(identity) {
  const scene = lessonScenes[identity];
  assert(scene, `Missing lesson scene ${identity}`);
  const drawing = scene.draw();
  assert(drawing.height > 0 && !/NaN|Infinity|undefined/.test(drawing.body), identity);
  return `<figure class="learning-figure visual-lesson-scene" data-lesson-visual="${identity}" aria-labelledby="${identity}-title"><p class="visual-title" id="${identity}-title">${escape(scene.title)}</p><div class="visual-argument" data-lesson-art="${identity}"><svg viewBox="0 0 360 ${drawing.height}" role="img" aria-label="${escape(scene.title)}"><title>${escape(scene.title)}</title><desc>${escape(scene.caption)}</desc>${drawing.body}</svg></div><figcaption>${escape(scene.caption)}</figcaption></figure>`;
}
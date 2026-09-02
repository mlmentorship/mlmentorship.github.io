import { articleVisualTraceFallbacks } from './articleVisualTraceFallbacks.ts';

export type ArticleTraceTone = 'input' | 'focus' | 'state' | 'output' | 'warning' | 'neutral';

export interface ArticleTraceCell {
  key: string;
  value: string;
  tone?: ArticleTraceTone;
  detail?: string;
}

export interface ArticleTraceLane {
  label: string;
  cells: ArticleTraceCell[];
}

export interface ArticleTracePlotPoint {
  key: string;
  x: number;
  y: number;
  label: string;
  shape: 'circle' | 'square';
  tone?: ArticleTraceTone;
}

export interface ArticleTraceCentroid {
  key: string;
  label: string;
  x: number;
  y: number;
  previous?: { x: number; y: number };
}

export interface ArticleTraceFlowNode {
  key: string;
  x: number;
  y: number;
  label: string;
  value: string;
  tone?: ArticleTraceTone;
  gradient?: string;
}

export interface ArticleTraceFlowEdge {
  key: string;
  from: string;
  to: string;
  label: string;
  direction: 'forward' | 'backward';
}

export interface ArticleTracePlotScene {
  type: 'plot';
  ariaLabel: string;
  points: ArticleTracePlotPoint[];
  centroids: ArticleTraceCentroid[];
  activePoint?: string;
  annotations: string[];
  formula?: string;
}

export interface ArticleTraceFlowScene {
  type: 'flow';
  ariaLabel: string;
  nodes: ArticleTraceFlowNode[];
  edges: ArticleTraceFlowEdge[];
  annotations: string[];
}

export interface ArticleTraceLanesScene {
  type: 'lanes';
  ariaLabel: string;
  lanes: ArticleTraceLane[];
  annotations: string[];
  formula?: string;
}

export interface ArticleTraceTableScene {
  type: 'table';
  ariaLabel: string;
  columns: string[];
  rows: ArticleTraceCell[][];
  annotations: string[];
  formula?: string;
}

export interface ArticleTraceEvidenceStage {
  key: string;
  label: string;
  value: string;
  tone?: ArticleTraceTone;
}

export interface ArticleTraceEvidenceScene {
  type: 'evidence';
  ariaLabel: string;
  stages: ArticleTraceEvidenceStage[];
  annotations: string[];
}

export interface ArticleTraceGridScene {
  type: 'grid' | 'schedule';
  ariaLabel: string;
  columns: string[];
  rows: ArticleTraceLane[];
  header?: string;
  queue?: ArticleTraceCell[];
  annotations: string[];
  formula?: string;
}

export interface ArticleTraceSpeculativeScene {
  type: 'speculative';
  ariaLabel: string;
  draft: ArticleTraceCell[];
  decisions: ArticleTraceCell[];
  committed: ArticleTraceCell[];
  annotations: string[];
  formula?: string;
}

export type ArticleTraceScene =
  | ArticleTraceLanesScene
  | ArticleTraceTableScene
  | ArticleTraceEvidenceScene
  | ArticleTracePlotScene
  | ArticleTraceFlowScene
  | ArticleTraceGridScene
  | ArticleTraceSpeculativeScene;

export interface ArticleTraceFrame {
  key: string;
  label: string;
  note: string;
  scene: ArticleTraceScene;
}

export interface ArticleTraceReview {
  recognitionCue: string;
  invariant: string;
  transferLesson: string;
}

export interface ArticleVisualTrace {
  slug: string;
  visualId: string;
  title: string;
  objective: string;
  example: string;
  traceKind?: 'mechanism' | 'evidence';
  frames: ArticleTraceFrame[];
  review: ArticleTraceReview;
}

const cell = (key: string, value: string, tone: ArticleTraceTone = 'neutral', detail?: string): ArticleTraceCell => ({
  key,
  value,
  tone,
  ...(detail ? { detail } : {}),
});

export const articleVisualTraces: Readonly<Record<string, ArticleVisualTrace>> = Object.freeze({
  tokenization: {
    slug: 'tokenization',
    visualId: 'bpe-corpus-merge-trace',
    title: 'BPE learns a ranked merge list, then replays it',
    objective: 'Trace how global pair counts create reusable tokens and leave rarer words in smaller pieces.',
    example: 'map x5, maps x3, cap x2; encode maps, cap, and traps after three learned merges.',
    traceKind: 'mechanism',
    frames: [
      {
        key: 'rank-1-ap',
        label: 'Rank 1: merge ap',
        note: 'Count adjacent pairs across every occurrence before changing the segmentation.',
        scene: {
          type: 'lanes',
          ariaLabel: 'BPE rank one counts a plus p ten times across the toy corpus.',
          lanes: [
            { label: 'corpus', cells: [cell('corpus-map', 'm a p x5', 'input'), cell('corpus-maps', 'm a p s x3', 'input'), cell('corpus-cap', 'c a p x2', 'input')] },
            { label: 'largest pair', cells: [cell('pair-ap', 'a + p = 5 + 3 + 2 = 10', 'focus')] },
            { label: 'vocabulary move', cells: [cell('merge-ap', 'add ap', 'state', 'rank 1')] },
          ],
          annotations: ['Only the current segmentation is counted.', 'The same pair is merged everywhere.'],
          formula: 'a + p -> ap',
        },
      },
      {
        key: 'rank-2-map',
        label: 'Rank 2: merge map',
        note: 'Recount after ap exists; the new symbol changes which pair is now most frequent.',
        scene: {
          type: 'lanes',
          ariaLabel: 'BPE rank two counts m plus ap eight times after the first merge.',
          lanes: [
            { label: 'corpus', cells: [cell('corpus-map', 'm ap x5', 'state'), cell('corpus-maps', 'm ap s x3', 'state'), cell('corpus-cap', 'c ap x2', 'input')] },
            { label: 'largest pair', cells: [cell('pair-map', 'm + ap = 5 + 3 = 8', 'focus')] },
            { label: 'vocabulary move', cells: [cell('merge-map', 'add map', 'state', 'rank 2')] },
          ],
          annotations: ['cap keeps c + ap because it has no m + ap pair.', 'The rank list is now ap, map.'],
          formula: 'm + ap -> map',
        },
      },
      {
        key: 'rank-3-replay',
        label: 'Rank 3: replay the learned order',
        note: 'After maps is learned, encode new words by applying ap, then map, then maps in order.',
        scene: {
          type: 'lanes',
          ariaLabel: 'BPE rank three learns maps and replays the ordered merges on three words.',
          lanes: [
            { label: 'corpus after map', cells: [cell('corpus-map', 'map x5', 'state'), cell('corpus-maps', 'map s x3', 'state'), cell('corpus-cap', 'c ap x2', 'input')] },
            { label: 'largest pair', cells: [cell('pair-maps', 'map + s = 3', 'focus')] },
            { label: 'replay on new text', cells: [cell('encoded-maps', 'maps -> [maps]', 'output'), cell('encoded-cap', 'cap -> [c] [ap]', 'output'), cell('encoded-traps', 'traps -> [t] [r] [ap] [s]', 'output')] },
          ],
          annotations: ['Frequent patterns become larger reusable tokens.', 'Unknown words still fall back to available byte pieces.'],
          formula: 'learned order: ap, map, maps',
        },
      },
    ],
    review: {
      recognitionCue: 'When tokenization is learned from a corpus, ask which adjacent pair is most frequent in the current segmentation.',
      invariant: 'Every merge is global for that round, and encoding replays the learned merges in rank order.',
      transferLesson: 'Use the same count, merge, recount loop to reason about BPE vocabulary growth and token-count changes.',
    },
  },
  'k-means-clustering': {
    slug: 'k-means-clustering',
    visualId: 'kmeans-assign-then-update',
    title: 'One Lloyd iteration separates assignment from movement',
    objective: 'Trace nearest-centroid assignment followed by the mean update that moves each centroid.',
    example: 'Circles at (50,80), (70,110), (90,80); squares at (230,70), (250,100), (270,70).',
    traceKind: 'mechanism',
    frames: [
      {
        key: 'assign-fixed-centers',
        label: 'Assignment: centers fixed',
        note: 'Compare each point with A0=(110,130) and B0=(210,120); membership changes, centers do not.',
        scene: {
          type: 'plot',
          ariaLabel: 'Six points are assigned to two fixed initial centroids.',
          points: [
            { key: 'point-a1', x: 50, y: 80, label: 'p1', shape: 'circle', tone: 'input' },
            { key: 'point-a2', x: 70, y: 110, label: 'p2', shape: 'circle', tone: 'input' },
            { key: 'point-a3', x: 90, y: 80, label: 'p3', shape: 'circle', tone: 'focus' },
            { key: 'point-b1', x: 230, y: 70, label: 'p4', shape: 'square', tone: 'output' },
            { key: 'point-b2', x: 250, y: 100, label: 'p5', shape: 'square', tone: 'output' },
            { key: 'point-b3', x: 270, y: 70, label: 'p6', shape: 'square', tone: 'output' },
          ],
          centroids: [
            { key: 'centroid-a', label: 'A0 (110,130)', x: 110, y: 130 },
            { key: 'centroid-b', label: 'B0 (210,120)', x: 210, y: 120 },
          ],
          activePoint: 'point-a3',
          annotations: ['p3: d2(A0)=2,900; d2(B0)=16,000.', 'Circles join A; squares join B.'],
          formula: 'C(i) = argmin distance(x_i, centroid_j)',
        },
      },
      {
        key: 'update-means',
        label: 'Update: memberships fixed',
        note: 'Hold each membership fixed and move the centroid to the arithmetic mean of its assigned points.',
        scene: {
          type: 'plot',
          ariaLabel: 'Two fixed memberships move centroids to their assigned point means.',
          points: [
            { key: 'point-a1', x: 50, y: 80, label: 'p1', shape: 'circle', tone: 'input' },
            { key: 'point-a2', x: 70, y: 110, label: 'p2', shape: 'circle', tone: 'input' },
            { key: 'point-a3', x: 90, y: 80, label: 'p3', shape: 'circle', tone: 'input' },
            { key: 'point-b1', x: 230, y: 70, label: 'p4', shape: 'square', tone: 'output' },
            { key: 'point-b2', x: 250, y: 100, label: 'p5', shape: 'square', tone: 'output' },
            { key: 'point-b3', x: 270, y: 70, label: 'p6', shape: 'square', tone: 'output' },
          ],
          centroids: [
            { key: 'centroid-a', label: 'A1 (70,90)', x: 70, y: 90, previous: { x: 110, y: 130 } },
            { key: 'centroid-b', label: 'B1 (250,80)', x: 250, y: 80, previous: { x: 210, y: 120 } },
          ],
          annotations: ['A1=((50+70+90)/3,(80+110+80)/3)=(70,90).', 'B1=((230+250+270)/3,(70+100+70)/3)=(250,80).'],
          formula: 'mu_j = mean of points assigned to j',
        },
      },
      {
        key: 'reassign-stop',
        label: 'Reassign and check',
        note: 'Recompute nearest centers; unchanged memberships mean this toy example has reached a fixed point.',
        scene: {
          type: 'plot',
          ariaLabel: 'The updated centroids preserve all six memberships, so the iteration stops.',
          points: [
            { key: 'point-a1', x: 50, y: 80, label: 'p1', shape: 'circle', tone: 'input' },
            { key: 'point-a2', x: 70, y: 110, label: 'p2', shape: 'circle', tone: 'input' },
            { key: 'point-a3', x: 90, y: 80, label: 'p3', shape: 'circle', tone: 'focus' },
            { key: 'point-b1', x: 230, y: 70, label: 'p4', shape: 'square', tone: 'output' },
            { key: 'point-b2', x: 250, y: 100, label: 'p5', shape: 'square', tone: 'output' },
            { key: 'point-b3', x: 270, y: 70, label: 'p6', shape: 'square', tone: 'output' },
          ],
          centroids: [
            { key: 'centroid-a', label: 'A1 (70,90)', x: 70, y: 90 },
            { key: 'centroid-b', label: 'B1 (250,80)', x: 250, y: 80 },
          ],
          activePoint: 'point-a3',
          annotations: ['p3: d2(A1)=500; d2(B1)=25,600.', 'No membership changes; stop at this local fixed point.'],
          formula: 'repeat until assignments stop changing',
        },
      },
    ],
    review: {
      recognitionCue: 'When a problem asks for clusters by squared Euclidean distance, separate nearest-center assignment from mean recomputation.',
      invariant: 'Assignment holds centers fixed; update holds memberships fixed; each update uses the assigned-point mean.',
      transferLesson: 'Reuse the two-phase invariant for k-means++, mini-batch k-means, and vector-quantization codebooks.',
    },
  },
  backpropagation: {
    slug: 'backpropagation',
    visualId: 'backprop-forward-reverse-trace',
    title: 'Backpropagation reuses a cached forward graph in reverse',
    objective: 'Trace cached values forward and upstream gradients backward through local derivatives.',
    example: 'x=2, w=3; z1=xw=6; z2=z1+1=7; L=z2^2=49.',
    traceKind: 'mechanism',
    frames: [
      {
        key: 'forward-cache',
        label: 'Forward: compute and cache',
        note: 'The scalar graph computes L once and retains x, w, z1, and z2 for the reverse sweep.',
        scene: {
          type: 'flow',
          ariaLabel: 'Inputs x and w flow through multiply, addition, and square to loss 49.',
          nodes: [
            { key: 'node-x', x: 70, y: 72, label: 'x', value: '2', tone: 'input' },
            { key: 'node-w', x: 70, y: 182, label: 'w', value: '3', tone: 'input' },
            { key: 'node-z1', x: 220, y: 127, label: 'z1', value: '6', tone: 'state' },
            { key: 'node-z2', x: 380, y: 127, label: 'z2', value: '7', tone: 'state' },
            { key: 'node-loss', x: 540, y: 127, label: 'L', value: '49', tone: 'output' },
          ],
          edges: [
            { key: 'edge-x-z1', from: 'node-x', to: 'node-z1', label: 'multiply', direction: 'forward' },
            { key: 'edge-w-z1', from: 'node-w', to: 'node-z1', label: 'multiply', direction: 'forward' },
            { key: 'edge-z1-z2', from: 'node-z1', to: 'node-z2', label: '+1', direction: 'forward' },
            { key: 'edge-z2-loss', from: 'node-z2', to: 'node-loss', label: 'square', direction: 'forward' },
          ],
          annotations: ['Forward values are saved before backward begins.', 'The graph is the data structure reverse mode traverses.'],
        },
      },
      {
        key: 'reverse-to-z1',
        label: 'Backward: seed and propagate',
        note: 'Seed dL/dL=1, then multiply the upstream gradient by each local derivative.',
        scene: {
          type: 'flow',
          ariaLabel: 'A reverse sweep carries gradients 1, 14, and 14 back from the loss to z1.',
          nodes: [
            { key: 'node-x', x: 70, y: 72, label: 'x', value: '2', tone: 'input' },
            { key: 'node-w', x: 70, y: 182, label: 'w', value: '3', tone: 'input' },
            { key: 'node-z1', x: 220, y: 127, label: 'z1', value: '6', tone: 'state', gradient: 'dL/dz1 = 14' },
            { key: 'node-z2', x: 380, y: 127, label: 'z2', value: '7', tone: 'state', gradient: 'dL/dz2 = 14' },
            { key: 'node-loss', x: 540, y: 127, label: 'L', value: '49', tone: 'output', gradient: 'seed = 1' },
          ],
          edges: [
            { key: 'edge-z2-loss', from: 'node-loss', to: 'node-z2', label: '1 x 2z2 = 14', direction: 'backward' },
            { key: 'edge-z1-z2', from: 'node-z2', to: 'node-z1', label: '14 x 1 = 14', direction: 'backward' },
          ],
          annotations: ['The square contributes local derivative 2z2=14.', 'The addition contributes local derivative 1.'],
        },
      },
      {
        key: 'reverse-to-parameters',
        label: 'Backward: split at multiply',
        note: 'The shared upstream gradient branches through z1=xw to produce both parameter gradients.',
        scene: {
          type: 'flow',
          ariaLabel: 'The reverse sweep produces gradients 42 for x and 28 for w.',
          nodes: [
            { key: 'node-x', x: 70, y: 72, label: 'x', value: '2', tone: 'input', gradient: 'dL/dx = 42' },
            { key: 'node-w', x: 70, y: 182, label: 'w', value: '3', tone: 'input', gradient: 'dL/dw = 28' },
            { key: 'node-z1', x: 220, y: 127, label: 'z1', value: '6', tone: 'state', gradient: 'dL/dz1 = 14' },
            { key: 'node-z2', x: 380, y: 127, label: 'z2', value: '7', tone: 'state', gradient: 'dL/dz2 = 14' },
            { key: 'node-loss', x: 540, y: 127, label: 'L', value: '49', tone: 'output', gradient: 'seed = 1' },
          ],
          edges: [
            { key: 'edge-z2-loss', from: 'node-loss', to: 'node-z2', label: '1 x 2z2 = 14', direction: 'backward' },
            { key: 'edge-z1-z2', from: 'node-z2', to: 'node-z1', label: '14 x 1 = 14', direction: 'backward' },
            { key: 'edge-x-z1', from: 'node-z1', to: 'node-x', label: '14 x w = 42', direction: 'backward' },
            { key: 'edge-w-z1', from: 'node-z1', to: 'node-w', label: '14 x x = 28', direction: 'backward' },
          ],
          annotations: ['dz1/dx=w=3, so dL/dx=14x3=42.', 'dz1/dw=x=2, so dL/dw=14x2=28.'],
        },
      },
    ],
    review: {
      recognitionCue: 'When the objective is one scalar built from differentiable operations, cache the forward intermediates and reverse the graph.',
      invariant: 'Every adjoint equals upstream gradient times a local derivative; shared nodes sum downstream contributions before propagating.',
      transferLesson: 'Use the same reverse accumulation for computation graphs, neural networks, and scalar reverse-mode autodiff.',
    },
  },
  'pipeline-parallelism': {
    slug: 'pipeline-parallelism',
    visualId: 'pipeline-fill-drain-bubble',
    title: 'Pipeline parallelism exposes a fill, full, and drain wave',
    objective: 'Trace a three-stage, four-micro-batch schedule and count its idle bubble.',
    example: 'P=3 stages, M=4 micro-batches, six time slots.',
    traceKind: 'mechanism',
    frames: [
      {
        key: 'fill-wave',
        label: 'Fill: the wave enters',
        note: 'A micro-batch cannot reach a later stage until it finishes the earlier stage.',
        scene: {
          type: 'grid',
          ariaLabel: 'The first three time slots show the pipeline fill wave across three stages.',
          columns: ['1', '2', '3', '4', '5', '6'],
          rows: [
            { label: 'S1', cells: [cell('s1-t1', 'F1', 'input'), cell('s1-t2', 'F2', 'input'), cell('s1-t3', 'F3', 'input'), cell('s1-t4', 'idle'), cell('s1-t5', 'idle'), cell('s1-t6', 'idle')] },
            { label: 'S2', cells: [cell('s2-t1', 'idle'), cell('s2-t2', 'F1', 'state'), cell('s2-t3', 'F2', 'state'), cell('s2-t4', 'idle'), cell('s2-t5', 'idle'), cell('s2-t6', 'idle')] },
            { label: 'S3', cells: [cell('s3-t1', 'idle'), cell('s3-t2', 'idle'), cell('s3-t3', 'F1', 'output'), cell('s3-t4', 'idle'), cell('s3-t5', 'idle'), cell('s3-t6', 'idle')] },
          ],
          annotations: ['F1 moves diagonally through adjacent stages.', 'The first useful slot at S3 appears at time 3.'],
          formula: 'dependency depth = P - 1 = 2 transitions',
        },
      },
      {
        key: 'full-pipeline',
        label: 'Full: all stages work',
        note: 'The middle of the schedule has no idle stage-slot while the four micro-batches are in flight.',
        scene: {
          type: 'grid',
          ariaLabel: 'All three stages are active during the full middle of the pipeline schedule.',
          columns: ['1', '2', '3', '4', '5', '6'],
          rows: [
            { label: 'S1', cells: [cell('s1-t1', 'F1', 'input'), cell('s1-t2', 'F2', 'input'), cell('s1-t3', 'F3', 'input'), cell('s1-t4', 'F4', 'input'), cell('s1-t5', 'idle'), cell('s1-t6', 'idle')] },
            { label: 'S2', cells: [cell('s2-t1', 'idle'), cell('s2-t2', 'F1', 'state'), cell('s2-t3', 'F2', 'state'), cell('s2-t4', 'F3', 'state'), cell('s2-t5', 'F4', 'state'), cell('s2-t6', 'idle')] },
            { label: 'S3', cells: [cell('s3-t1', 'idle'), cell('s3-t2', 'idle'), cell('s3-t3', 'F1', 'output'), cell('s3-t4', 'F2', 'output'), cell('s3-t5', 'F3', 'output'), cell('s3-t6', 'F4', 'output')] },
          ],
          annotations: ['F1, F2, F3, F4 occupy different stages at once.', 'Only the fill and drain ends are idle.'],
          formula: 'useful work = P x M = 3 x 4 = 12 stage-slots',
        },
      },
      {
        key: 'drain-count',
        label: 'Drain: count the bubble',
        note: 'After S1 finishes its forwards, later stages finish the remaining micro-batches while early slots go idle.',
        scene: {
          type: 'grid',
          ariaLabel: 'The completed schedule contains twelve active and six idle stage-slots.',
          columns: ['1', '2', '3', '4', '5', '6'],
          rows: [
            { label: 'S1', cells: [cell('s1-t1', 'F1', 'input'), cell('s1-t2', 'F2', 'input'), cell('s1-t3', 'F3', 'input'), cell('s1-t4', 'F4', 'input'), cell('s1-t5', 'idle', 'warning'), cell('s1-t6', 'idle', 'warning')] },
            { label: 'S2', cells: [cell('s2-t1', 'idle', 'warning'), cell('s2-t2', 'F1', 'state'), cell('s2-t3', 'F2', 'state'), cell('s2-t4', 'F3', 'state'), cell('s2-t5', 'F4', 'state'), cell('s2-t6', 'idle', 'warning')] },
            { label: 'S3', cells: [cell('s3-t1', 'idle', 'warning'), cell('s3-t2', 'idle', 'warning'), cell('s3-t3', 'F1', 'output'), cell('s3-t4', 'F2', 'output'), cell('s3-t5', 'F3', 'output'), cell('s3-t6', 'F4', 'output')] },
          ],
          annotations: ['Fill idle slots = 3; drain idle slots = 3.', 'Bubble = 6 / 18 = 1 / 3.'],
          formula: 'bubble = (P - 1) / (M + P - 1) = 2 / 6 = 1 / 3',
        },
      },
    ],
    review: {
      recognitionCue: 'When layers are split across stages and a batch can be divided into micro-batches, draw stage rows against time slots.',
      invariant: 'A micro-batch advances one stage per dependency step; increasing M adds full middle work without increasing fill or drain depth.',
      transferLesson: 'Reuse the fill/full/drain schedule to compare GPipe, 1F1B, interleaving, and bubble formulas.',
    },
  },
  'continuous-batching': {
    slug: 'continuous-batching',
    visualId: 'continuous-batching-reuses-finished-slot',
    title: 'Continuous batching rebuilds the active set at each step',
    objective: 'Trace request C replacing completed request B while request A keeps decoding.',
    example: 'A runs five decode steps, B runs two, and queued C needs three steps.',
    traceKind: 'mechanism',
    frames: [
      {
        key: 'shared-prefix',
        label: 'Shared prefix: B finishes',
        note: 'Both schedulers produce the same first two iterations; B reaches completion while A remains active.',
        scene: {
          type: 'schedule',
          ariaLabel: 'The first two decode steps contain A and B while C waits in the queue.',
          header: 'slot',
          columns: ['1', '2', '3', '4', '5'],
          rows: [
            { label: 'slot 1', cells: [cell('a-t1', 'A', 'input'), cell('a-t2', 'A', 'input'), cell('a-t3', 'A', 'input'), cell('a-t4', 'A', 'input'), cell('a-t5', 'A', 'input')] },
            { label: 'slot 2', cells: [cell('b-t1', 'B', 'state'), cell('b-t2', 'B', 'state'), cell('slot2-t3', 'free'), cell('slot2-t4', 'free'), cell('slot2-t5', 'free')] },
          ],
          queue: [cell('request-c', 'C queued', 'focus')],
          annotations: ['B completes after step 2.', 'A still owns its KV cache and continues.'],
        },
      },
      {
        key: 'static-membership',
        label: 'Static batching: leave the slot idle',
        note: 'A fixed batch cannot admit C until A also finishes, so B slot positions 3-5 do no work.',
        scene: {
          type: 'schedule',
          ariaLabel: 'Static batching leaves slot two idle after B finishes and keeps C queued.',
          header: 'slot',
          columns: ['1', '2', '3', '4', '5'],
          rows: [
            { label: 'slot 1', cells: [cell('a-t1', 'A', 'input'), cell('a-t2', 'A', 'input'), cell('a-t3', 'A', 'input'), cell('a-t4', 'A', 'input'), cell('a-t5', 'A', 'input')] },
            { label: 'slot 2', cells: [cell('b-t1', 'B', 'state'), cell('b-t2', 'B', 'state'), cell('slot2-t3', 'idle', 'warning'), cell('slot2-t4', 'idle', 'warning'), cell('slot2-t5', 'idle', 'warning')] },
          ],
          queue: [cell('request-c', 'C queued', 'focus')],
          annotations: ['Useful slots = 7 / 10.', 'The completed request still occupies the batch shape.'],
          formula: 'static utilization = 7 / 10',
        },
      },
      {
        key: 'continuous-replacement',
        label: 'Continuous batching: admit C',
        note: "At the next iteration, free B's KV blocks and admit C into the same slot while A stays active.",
        scene: {
          type: 'schedule',
          ariaLabel: 'Continuous batching replaces B with C at step three and fills all ten slot positions.',
          header: 'slot',
          columns: ['1', '2', '3', '4', '5'],
          rows: [
            { label: 'slot 1', cells: [cell('a-t1', 'A', 'input'), cell('a-t2', 'A', 'input'), cell('a-t3', 'A', 'input'), cell('a-t4', 'A', 'input'), cell('a-t5', 'A', 'input')] },
            { label: 'slot 2', cells: [cell('b-t1', 'B', 'state'), cell('b-t2', 'B', 'state'), cell('request-c', 'C', 'output'), cell('c-t4', 'C', 'output'), cell('c-t5', 'C', 'output')] },
          ],
          annotations: ['C enters only after B completes at the iteration boundary.', 'Useful slots = 10 / 10; each request still owns its own KV state.'],
          formula: 'continuous utilization = 10 / 10',
        },
      },
    ],
    review: {
      recognitionCue: 'When requests finish at different decode lengths, inspect the active set at every iteration boundary.',
      invariant: 'Unfinished requests retain their KV state; completed requests release capacity before queued work is admitted.',
      transferLesson: 'Reuse the replacement rule for paged KV allocation, chunked prefill, and any iteration-level serving scheduler.',
    },
  },
  'activation-functions': {
    slug: 'activation-functions',
    visualId: 'activation-gradient-regions',
    title: 'Activation functions change both output and local slope',
    objective: 'Compare sigmoid, ReLU, and exact GELU at negative, zero, and positive inputs.',
    example: 'Evaluate each function and derivative at x=-2, x=0, and x=2.',
    traceKind: 'mechanism',
    frames: [
      {
        key: 'negative-input',
        label: 'Negative input: x = -2',
        note: 'The same input exposes sigmoid saturation, ReLU deadness, and GELU negative shoulder.',
        scene: {
          type: 'table',
          ariaLabel: 'Activation outputs and derivatives at x equals negative two.',
          columns: ['function', 'output f(-2)', 'slope f\'(-2)'],
          rows: [
            [cell('activation-sigmoid-name', 'sigmoid', 'input'), cell('activation-sigmoid-output', '0.119', 'state'), cell('activation-sigmoid-slope', '0.105', 'state')],
            [cell('activation-relu-name', 'ReLU', 'input'), cell('activation-relu-output', '0', 'state'), cell('activation-relu-slope', '0', 'warning')],
            [cell('activation-gelu-name', 'exact GELU', 'input'), cell('activation-gelu-output', '-0.0455', 'focus'), cell('activation-gelu-slope', '-0.0852', 'focus')],
          ],
          annotations: ['Sigmoid is already near its lower asymptote.', 'ReLU passes no negative activation or gradient.'],
          formula: 'GELU(x) = x Phi(x)',
        },
      },
      {
        key: 'zero-input',
        label: 'Midpoint: x = 0',
        note: 'At zero, sigmoid is halfway to its range, ReLU reaches its kink, and exact GELU crosses the origin smoothly.',
        scene: {
          type: 'table',
          ariaLabel: 'Activation outputs and derivatives at x equals zero.',
          columns: ['function', 'output f(0)', 'slope f\'(0)'],
          rows: [
            [cell('activation-sigmoid-name', 'sigmoid', 'input'), cell('activation-sigmoid-output', '0.5', 'state'), cell('activation-sigmoid-slope', '0.25', 'state')],
            [cell('activation-relu-name', 'ReLU', 'input'), cell('activation-relu-output', '0', 'focus'), cell('activation-relu-slope', 'kink convention', 'focus')],
            [cell('activation-gelu-name', 'exact GELU', 'input'), cell('activation-gelu-output', '0', 'focus'), cell('activation-gelu-slope', '0.5', 'focus')],
          ],
          annotations: ['ReLU derivative exactly at zero depends on the implementation convention.', 'The exact GELU derivative at zero is Phi(0)=0.5.'],
          formula: 'sigmoid\'(0) = 0.5(1 - 0.5) = 0.25',
        },
      },
      {
        key: 'positive-input',
        label: 'Positive input: x = 2',
        note: 'ReLU keeps a unit slope, GELU continues smoothly above one, and sigmoid remains bounded near saturation.',
        scene: {
          type: 'table',
          ariaLabel: 'Activation outputs and derivatives at x equals two.',
          columns: ['function', 'output f(2)', 'slope f\'(2)'],
          rows: [
            [cell('activation-sigmoid-name', 'sigmoid', 'input'), cell('activation-sigmoid-output', '0.881', 'state'), cell('activation-sigmoid-slope', '0.105', 'state')],
            [cell('activation-relu-name', 'ReLU', 'input'), cell('activation-relu-output', '2', 'output'), cell('activation-relu-slope', '1', 'output')],
            [cell('activation-gelu-name', 'exact GELU', 'input'), cell('activation-gelu-output', '1.9545', 'output'), cell('activation-gelu-slope', '1.0852', 'output')],
          ],
          annotations: ['Sigmoid derivative is small in both tails.', 'GELU positive derivative can exceed one near x=2.'],
          formula: 'GELU\'(2) = Phi(2) + 2 phi(2) = 1.0852',
        },
      },
    ],
    review: {
      recognitionCue: 'When an activation choice affects gradient flow, compare both f(x) and f\'(x) on the input region the layer sees.',
      invariant: 'The output curve and its local slope are separate facts; a bounded output does not imply a useful gradient.',
      transferLesson: 'Reuse the output-and-slope check for initialization, saturation diagnosis, and activation replacement decisions.',
    },
  },
  'attention-mechanism': {
    slug: 'attention-mechanism',
    visualId: 'attention-keys-choose-values-contribute',
    title: 'Attention uses keys to choose weights and values to carry content',
    objective: 'Trace one query from key similarities to normalized weights and a weighted value sum.',
    example: 'Weights alpha=(0.60,0.30,0.10) and values v=(1,4,7).',
    traceKind: 'mechanism',
    frames: [
      {
        key: 'score-normalize',
        label: 'Score and normalize',
        note: 'The query compares with keys first; softmax turns those comparisons into coefficients.',
        scene: {
          type: 'lanes',
          ariaLabel: 'One query produces three normalized attention weights from three keys.',
          lanes: [
            { label: 'query -> keys', cells: [cell('key-1', 'q vs k1', 'input'), cell('key-2', 'q vs k2', 'input'), cell('key-3', 'q vs k3', 'input')] },
            { label: 'weights', cells: [cell('alpha-1', 'alpha1 = 0.60', 'focus'), cell('alpha-2', 'alpha2 = 0.30', 'focus'), cell('alpha-3', 'alpha3 = 0.10', 'focus')] },
          ],
          annotations: ['Keys affect the coefficients.', 'The weights sum to one and values have not been used yet.'],
          formula: 'alpha = softmax(q K^T / sqrt(d))',
        },
      },
      {
        key: 'route-values',
        label: 'Route the value payload',
        note: 'Carry each normalized weight to the value paired with the same key.',
        scene: {
          type: 'lanes',
          ariaLabel: 'The three attention weights multiply their paired values one, four, and seven.',
          lanes: [
            { label: 'weights', cells: [cell('alpha-1', '0.60', 'focus'), cell('alpha-2', '0.30', 'focus'), cell('alpha-3', '0.10', 'focus')] },
            { label: 'values', cells: [cell('value-1', 'v1 = 1', 'input'), cell('value-2', 'v2 = 4', 'input'), cell('value-3', 'v3 = 7', 'input')] },
            { label: 'contributions', cells: [cell('contribution-1', '0.60 x 1 = 0.60', 'state'), cell('contribution-2', '0.30 x 4 = 1.20', 'state'), cell('contribution-3', '0.10 x 7 = 0.70', 'state')] },
          ],
          annotations: ['Each weight stays paired with its own value.', 'The key score chooses the coefficient, not the payload.'],
          formula: 'alpha_i v_i',
        },
      },
      {
        key: 'aggregate-output',
        label: 'Aggregate the output',
        note: 'Add the weighted value contributions into one attention output.',
        scene: {
          type: 'lanes',
          ariaLabel: 'The weighted value contributions sum to an attention output of two point five.',
          lanes: [
            { label: 'contributions', cells: [cell('contribution-1', '0.60', 'state'), cell('contribution-2', '1.20', 'state'), cell('contribution-3', '0.70', 'state')] },
            { label: 'output', cells: [cell('attention-output', '0.60 + 1.20 + 0.70 = 2.50', 'output')] },
          ],
          annotations: ['The output is a convex combination of the values.', 'Changing keys changes the mixture weights; changing values changes the payload.'],
          formula: 'attention(q,K,V) = sum_i alpha_i v_i = 2.50',
        },
      },
    ],
    review: {
      recognitionCue: 'When a query must select information from a sequence, separate the similarity path through QK from the payload path through V.',
      invariant: 'Attention weights are normalized coefficients paired with the corresponding values before summation.',
      transferLesson: 'Reuse the key-weight/value-payload split for self-attention, cross-attention, and cached decoding.',
    },
  },
  'batchnorm-vs-layernorm': {
    slug: 'batchnorm-vs-layernorm',
    visualId: 'normalization-shared-statistics',
    title: 'Normalization behavior follows the axes that share statistics',
    objective: 'Compare the entries grouped by BatchNorm and transformer LayerNorm.',
    example: 'BatchNorm on (N,C,H,W) and LayerNorm on (B,T,D).',
    traceKind: 'mechanism',
    frames: [
      {
        key: 'batchnorm-axes',
        label: 'BatchNorm: group by channel',
        note: 'For each channel C, BatchNorm shares statistics over samples and spatial positions.',
        scene: {
          type: 'lanes',
          ariaLabel: 'BatchNorm groups N and H by each channel C while preserving separate channel statistics.',
          lanes: [
            { label: 'tensor shape', cells: [cell('bn-shape', '(N, C, H, W)', 'input')] },
            { label: 'one statistic', cells: [cell('bn-stat-c0', 'mu_C0, sigma_C0 over N,H,W', 'focus'), cell('bn-stat-c1', 'mu_C1, sigma_C1 over N,H,W', 'focus'), cell('bn-stat-c2', 'mu_C2, sigma_C2 over N,H,W', 'focus')] },
            { label: 'parameters', cells: [cell('bn-params', 'gamma,beta shape = C', 'state')] },
          ],
          annotations: ['Rows from different samples share a channel statistic.', 'Different channels do not share that statistic.'],
        },
      },
      {
        key: 'layernorm-axes',
        label: 'LayerNorm: group by token row',
        note: 'For each token position (b,t), LayerNorm computes statistics over the feature dimension D only.',
        scene: {
          type: 'lanes',
          ariaLabel: 'Transformer LayerNorm groups the feature dimension D within each token row.',
          lanes: [
            { label: 'tensor shape', cells: [cell('ln-shape', '(B, T, D)', 'input')] },
            { label: 'one statistic', cells: [cell('ln-stat-row-a', 'mu_(b,t), sigma_(b,t) over D', 'focus'), cell('ln-stat-row-b', 'independent for each token row', 'focus')] },
            { label: 'parameters', cells: [cell('ln-params', 'gamma,beta shape = D', 'state')] },
          ],
          annotations: ['Features in one token row share a statistic.', 'Different batch items and token positions do not mix.'],
        },
      },
      {
        key: 'train-eval-consequence',
        label: 'The axis choice changes runtime behavior',
        note: 'BatchNorm switches to stored running statistics at evaluation; LayerNorm uses the same per-token computation in train and eval.',
        scene: {
          type: 'lanes',
          ariaLabel: 'BatchNorm and LayerNorm have different train and evaluation behavior because their shared axes differ.',
          lanes: [
            { label: 'BatchNorm', cells: [cell('bn-train', 'train: batch statistics', 'state'), cell('bn-eval', 'eval: running statistics', 'output'), cell('bn-batch-one', 'batch-size-one sensitive', 'warning')] },
            { label: 'LayerNorm', cells: [cell('ln-train', 'train: per-token statistics', 'state'), cell('ln-eval', 'eval: same computation', 'output'), cell('ln-variable', 'works with variable lengths', 'output')] },
          ],
          annotations: ['The layer name is not the grouping rule.', 'Read the normalized axes first, then infer train/eval behavior.'],
        },
      },
    ],
    review: {
      recognitionCue: 'When a normalization question gives tensor dimensions, mark the axes reduced by the mean and variance before discussing behavior.',
      invariant: 'The normalized axes define which entries share statistics; parameter shape follows the remaining feature axis.',
      transferLesson: 'Reuse axis tracing for group normalization, RMSNorm, packed sequences, and distributed batch-statistics questions.',
    },
  },
  'svd-and-pca': {
    slug: 'svd-and-pca',
    visualId: 'pca-rank-one-projection',
    title: 'PCA keeps the coordinate along PC1 and discards the perpendicular residual',
    objective: 'Trace mean-centering, orthogonal projection, and rank-1 reconstruction.',
    example: 'Six mean-centered samples, PC1 along greatest variance, and PC2 perpendicular to PC1.',
    traceKind: 'mechanism',
    frames: [
      {
        key: 'center-and-orient',
        label: 'Center and orient',
        note: 'Subtract the mean, then choose orthogonal directions with PC1 aligned to the greatest sample variance.',
        scene: {
          type: 'lanes',
          ariaLabel: 'Six mean-centered samples are described relative to orthogonal principal directions.',
          lanes: [
            { label: 'samples', cells: [cell('pca-samples', 'six centered samples', 'input'), cell('pca-mean', 'mean = 0', 'state')] },
            { label: 'basis', cells: [cell('pca-pc1', 'PC1: greatest variance', 'focus'), cell('pca-pc2', 'PC2: perpendicular', 'state')] },
          ],
          annotations: ['Centering makes the origin the sample mean.', 'The directions are orthogonal, so each sample has two coordinates.'],
          formula: 'A = U Sigma V^T',
        },
      },
      {
        key: 'project-onto-pc1',
        label: 'Project orthogonally',
        note: 'Drop each centered sample along PC2 until it reaches its coordinate on PC1.',
        scene: {
          type: 'lanes',
          ariaLabel: 'Each centered sample maps to an orthogonal projection on PC1 and leaves a perpendicular residual.',
          lanes: [
            { label: 'input', cells: [cell('pca-samples', 'sample x', 'input')] },
            { label: 'projection', cells: [cell('pca-projection', 'x projected onto PC1', 'focus'), cell('pca-pc1-coordinate', 'retain PC1 coordinate', 'state')] },
            { label: 'residual', cells: [cell('pca-residual', 'x - xhat is perpendicular to PC1', 'warning')] },
          ],
          annotations: ['The projection is the closest point on the PC1 line.', 'The residual is the part rank 1 will discard.'],
          formula: 'xhat = projection_PC1(x)',
        },
      },
      {
        key: 'rank-one-reconstruction',
        label: 'Reconstruct at rank 1',
        note: 'Keep the PC1 coordinate as xhat and measure reconstruction error with the discarded residual.',
        scene: {
          type: 'lanes',
          ariaLabel: 'Rank one PCA retains the PC1 reconstruction and discards the perpendicular residual.',
          lanes: [
            { label: 'kept', cells: [cell('pca-reconstruction', 'xhat on PC1', 'output'), cell('pca-z', 'Z = Xtilde V1', 'output')] },
            { label: 'discarded', cells: [cell('pca-residual', 'x - xhat', 'warning'), cell('pca-error', 'reconstruction error', 'warning')] },
          ],
          annotations: ['Rank 1 keeps one coordinate per sample.', 'Increasing k retains more orthogonal directions and reduces reconstruction error.'],
          formula: 'Z = Xtilde V_k = U_k Sigma_k',
        },
      },
    ],
    review: {
      recognitionCue: 'When a dimensionality-reduction question asks what PCA retains, identify the high-variance subspace and project onto it.',
      invariant: 'Center first; the discarded residual is orthogonal to the retained principal subspace.',
      transferLesson: 'Reuse projection and residual reasoning for truncated SVD, low-rank reconstruction, and linear autoencoder comparisons.',
    },
  },
  'roc-pr-auc': {
    slug: 'roc-pr-auc',
    visualId: 'roc-pr-same-operating-point',
    title: 'One threshold produces different ROC and PR views of the same counts',
    objective: 'Trace one imbalanced confusion-count state into ROC and precision-recall coordinates.',
    example: '10 positives, 1,000 negatives, TP=8, FN=2, FP=50, TN=950.',
    traceKind: 'mechanism',
    frames: [
      {
        key: 'confusion-counts',
        label: 'Count the threshold outcome',
        note: 'Partition the same threshold predictions by truth before computing any rate.',
        scene: {
          type: 'table',
          ariaLabel: 'Confusion counts for ten positive and one thousand negative examples.',
          columns: ['truth group', 'predicted positive', 'predicted negative'],
          rows: [
            [cell('roc-positive-truth', 'actual positive: 10', 'input'), cell('roc-tp', 'TP = 8', 'output'), cell('roc-fn', 'FN = 2', 'warning')],
            [cell('roc-negative-truth', 'actual negative: 1,000', 'input'), cell('roc-fp', 'FP = 50', 'warning'), cell('roc-tn', 'TN = 950', 'state')],
          ],
          annotations: ['The positive prevalence is 10/(10+1,000), about 1%.', 'These four counts are reused in both metric views.'],
        },
      },
      {
        key: 'roc-coordinate',
        label: 'ROC normalizes within truth class',
        note: 'False-positive rate divides by all negatives while true-positive rate divides by all positives.',
        scene: {
          type: 'table',
          ariaLabel: 'The threshold maps to ROC coordinates zero point zero five and zero point eight zero.',
          columns: ['ROC coordinate', 'numerator', 'denominator', 'value'],
          rows: [
            [cell('roc-fpr-label', 'FPR', 'focus'), cell('roc-fpr-num', 'FP = 50', 'warning'), cell('roc-fpr-den', 'FP + TN = 1,000', 'state'), cell('roc-fpr-value', '0.05', 'focus')],
            [cell('roc-tpr-label', 'TPR', 'focus'), cell('roc-tpr-num', 'TP = 8', 'output'), cell('roc-tpr-den', 'TP + FN = 10', 'state'), cell('roc-tpr-value', '0.80', 'focus')],
          ],
          annotations: ['ROC point = (FPR, TPR) = (0.05, 0.80).', 'The class prior is hidden by the within-class denominators.'],
          formula: 'FPR = FP/(FP+TN); TPR = TP/(TP+FN)',
        },
      },
      {
        key: 'pr-coordinate',
        label: 'PR normalizes within predicted alerts',
        note: 'Precision divides true alerts by all predicted alerts, exposing the fifty false alarms.',
        scene: {
          type: 'table',
          ariaLabel: 'The same threshold maps to recall zero point eight zero and precision zero point one three seven nine.',
          columns: ['PR coordinate', 'numerator', 'denominator', 'value'],
          rows: [
            [cell('pr-recall-label', 'recall', 'focus'), cell('pr-recall-num', 'TP = 8', 'output'), cell('pr-recall-den', 'TP + FN = 10', 'state'), cell('pr-recall-value', '0.80', 'focus')],
            [cell('pr-precision-label', 'precision', 'focus'), cell('pr-precision-num', 'TP = 8', 'output'), cell('pr-precision-den', 'TP + FP = 58', 'warning'), cell('pr-precision-value', '8/58 = 0.1379', 'warning')],
          ],
          annotations: ['The random PR baseline is prevalence = 10/1,010 = 0.0099.', 'AUC summarizes a curve, not this one operating point.'],
          formula: 'precision = TP/(TP+FP); recall = TP/(TP+FN)',
        },
      },
    ],
    review: {
      recognitionCue: 'When positives are rare, write the four confusion counts before choosing ROC or precision-recall metrics.',
      invariant: 'The threshold counts do not change; only the denominator used by each view changes.',
      transferLesson: 'Reuse the count-first method for threshold selection, alert review capacity, and prevalence-shift analysis.',
    },
  },
  calibration: {
    slug: 'calibration',
    visualId: 'calibration-reliability-gap',
    title: 'Calibration compares predicted confidence with observed correctness',
    objective: 'Trace reliability coordinates into calibration gaps and their weighted ECE terms.',
    example: 'Reliability points (0.20,0.12), (0.40,0.28), (0.60,0.43), (0.80,0.60), (0.90,0.72).',
    traceKind: 'mechanism',
    frames: [
      {
        key: 'reliability-points',
        label: 'Place the reliability points',
        note: 'Each bin contributes one point with mean predicted confidence on x and observed correctness on y.',
        scene: {
          type: 'table',
          ariaLabel: 'Five reliability points compare confidence with observed accuracy.',
          columns: ['bin confidence', 'observed accuracy', 'position'],
          rows: [
            [cell('cal-bin-20', '0.20', 'input'), cell('cal-acc-20', '0.12', 'state'), cell('cal-pos-20', 'below y=x', 'focus')],
            [cell('cal-bin-40', '0.40', 'input'), cell('cal-acc-40', '0.28', 'state'), cell('cal-pos-40', 'below y=x', 'focus')],
            [cell('cal-bin-60', '0.60', 'input'), cell('cal-acc-60', '0.43', 'state'), cell('cal-pos-60', 'below y=x', 'focus')],
            [cell('cal-bin-80', '0.80', 'input'), cell('cal-acc-80', '0.60', 'state'), cell('cal-pos-80', 'below y=x', 'focus')],
            [cell('cal-bin-90', '0.90', 'input'), cell('cal-acc-90', '0.72', 'state'), cell('cal-pos-90', 'below y=x', 'focus')],
          ],
          annotations: ['The diagonal y=x is perfect calibration.', 'Every listed point is below the diagonal, so the model is overconfident.'],
        },
      },
      {
        key: 'absolute-gaps',
        label: 'Measure each calibration gap',
        note: 'The vertical distance to y=x is the absolute difference between observed accuracy and confidence.',
        scene: {
          type: 'table',
          ariaLabel: 'Absolute calibration gaps are zero point zero eight, zero point one two, zero point one seven, zero point two zero, and zero point one eight.',
          columns: ['bin confidence', 'observed accuracy', 'absolute gap'],
          rows: [
            [cell('cal-bin-20', '0.20', 'input'), cell('cal-acc-20', '0.12', 'state'), cell('cal-gap-20', '0.08', 'focus')],
            [cell('cal-bin-40', '0.40', 'input'), cell('cal-acc-40', '0.28', 'state'), cell('cal-gap-40', '0.12', 'focus')],
            [cell('cal-bin-60', '0.60', 'input'), cell('cal-acc-60', '0.43', 'state'), cell('cal-gap-60', '0.17', 'focus')],
            [cell('cal-bin-80', '0.80', 'input'), cell('cal-acc-80', '0.60', 'state'), cell('cal-gap-80', '0.20', 'warning')],
            [cell('cal-bin-90', '0.90', 'input'), cell('cal-acc-90', '0.72', 'state'), cell('cal-gap-90', '0.18', 'focus')],
          ],
          annotations: ['The largest listed gap is the 0.80-confidence bin: 0.80 - 0.60 = 0.20.', 'Gap magnitude does not include how many examples occupy a bin.'],
          formula: 'gap_m = |accuracy_m - confidence_m|',
        },
      },
      {
        key: 'weighted-ece',
        label: 'Weight gaps to form ECE',
        note: 'ECE weights each gap by the fraction of examples in that bin; the bin populations are required for a numeric total.',
        scene: {
          type: 'table',
          ariaLabel: 'Expected calibration error weights each listed calibration gap by its bin population.',
          columns: ['bin', 'absolute gap', 'ECE contribution'],
          rows: [
            [cell('cal-bin-20', '0.20', 'input'), cell('cal-gap-20', '0.08', 'focus'), cell('cal-ece-20', '(n_20/n) x 0.08', 'state')],
            [cell('cal-bin-40', '0.40', 'input'), cell('cal-gap-40', '0.12', 'focus'), cell('cal-ece-40', '(n_40/n) x 0.12', 'state')],
            [cell('cal-bin-60', '0.60', 'input'), cell('cal-gap-60', '0.17', 'focus'), cell('cal-ece-60', '(n_60/n) x 0.17', 'state')],
            [cell('cal-bin-80', '0.80', 'input'), cell('cal-gap-80', '0.20', 'warning'), cell('cal-ece-80', '(n_80/n) x 0.20', 'warning')],
            [cell('cal-bin-90', '0.90', 'input'), cell('cal-gap-90', '0.18', 'focus'), cell('cal-ece-90', '(n_90/n) x 0.18', 'state')],
          ],
          annotations: ['No total ECE is claimed because the figure gives no n_m values.', 'A calibration plot and an ECE number answer related but different questions.'],
          formula: 'ECE = sum_m (n_m/n) |accuracy_m - confidence_m|',
        },
      },
    ],
    review: {
      recognitionCue: 'When a model reports probabilities, group predictions by confidence and compare empirical correctness with the diagonal.',
      invariant: 'Perfect calibration means observed accuracy equals predicted confidence in every bin.',
      transferLesson: 'Reuse the gap-then-weight method for ECE, reliability plots, temperature scaling, and selective prediction.',
    },
  },
  'speculative-decoding': {
    slug: 'speculative-decoding',
    visualId: 'speculative-decoding-first-rejection-boundary',
    title: 'Speculative decoding commits only the accepted prefix',
    objective: 'Trace a five-token draft through one target verification pass and its first rejection.',
    example: 'K=5: draft positions 1 and 2 are accepted, position 3 is rejected, and positions 4 and 5 are discarded.',
    traceKind: 'mechanism',
    frames: [
      {
        key: 'draft-verify',
        label: 'Draft, then verify',
        note: 'The small model drafts five positions serially; the target scores all five in one parallel pass.',
        scene: {
          type: 'speculative',
          ariaLabel: 'Five draft positions move from serial proposal to one parallel target verification pass.',
          draft: [cell('draft-1', 'x1', 'input'), cell('draft-2', 'x2', 'input'), cell('draft-3', 'x3', 'input'), cell('draft-4', 'x4', 'input'), cell('draft-5', 'x5', 'input')],
          decisions: [cell('decision-1', 'target pM(x1)', 'state'), cell('decision-2', 'target pM(x2)', 'state'), cell('decision-3', 'target pM(x3)', 'state'), cell('decision-4', 'target pM(x4)', 'state'), cell('decision-5', 'target pM(x5)', 'state')],
          committed: [],
          annotations: ['m: K serial draft steps.', 'M: one forward pass over the drafted block.'],
          formula: 'alpha_i = min(1, pM(x_i) / pm(x_i))',
        },
      },
      {
        key: 'first-rejection',
        label: 'First rejection at position 3',
        note: 'Sweep from left to right; after the first rejection, later draft tokens no longer have a trusted prefix.',
        scene: {
          type: 'speculative',
          ariaLabel: 'Draft tokens one and two are accepted, token three is rejected, and tokens four and five are discarded.',
          draft: [cell('draft-1', 'x1', 'output'), cell('draft-2', 'x2', 'output'), cell('draft-3', 'x3', 'warning'), cell('draft-4', 'x4', 'neutral'), cell('draft-5', 'x5', 'neutral')],
          decisions: [cell('decision-1', 'ACCEPT', 'output'), cell('decision-2', 'ACCEPT', 'output'), cell('decision-3', 'REJECT', 'warning'), cell('decision-4', 'DISCARD', 'neutral'), cell('decision-5', 'DISCARD', 'neutral')],
          committed: [],
          annotations: ['The rejection boundary is i*=3.', 'Do not evaluate the suffix as if its old prefix still existed.'],
          formula: 'accept with probability alpha_i; stop at first rejection',
        },
      },
      {
        key: 'commit-restart',
        label: 'Commit, replace, restart',
        note: 'Keep x1 and x2, sample a replacement from q at position 3, discard x4 and x5, then continue from the replacement.',
        scene: {
          type: 'speculative',
          ariaLabel: 'The cycle commits two accepted draft tokens and one corrected replacement before restarting.',
          draft: [cell('draft-1', 'x1', 'output'), cell('draft-2', 'x2', 'output'), cell('draft-3', 'discarded', 'neutral'), cell('draft-4', 'discarded', 'neutral'), cell('draft-5', 'discarded', 'neutral')],
          decisions: [cell('decision-1', 'keep', 'output'), cell('decision-2', 'keep', 'output'), cell('decision-3', 'replace', 'focus'), cell('decision-4', 'drop suffix', 'neutral'), cell('decision-5', 'drop suffix', 'neutral')],
          committed: [cell('committed-1', 'x1', 'output'), cell('committed-2', 'x2', 'output'), cell('replacement', 'replacement from q', 'focus')],
          annotations: ['q(x)=normalize(max(0,pM(x)-pm(x))).', 'The committed stream still has target distribution pM.'],
          formula: 'accepted prefix + q replacement -> next cycle',
        },
      },
    ],
    review: {
      recognitionCue: 'When a small model proposes several autoregressive tokens for a large model, locate the first accept/reject boundary.',
      invariant: 'After the first rejection, every later draft is discarded; the corrected residual sample preserves the target distribution.',
      transferLesson: 'Reuse the prefix-boundary rule for tree speculation, self-speculation, and KV-cache commit logic.',
    },
  },
});

const articleVisualTraceFallbackMap = articleVisualTraceFallbacks as unknown as Readonly<Record<string, ArticleVisualTrace>>;

export function getArticleVisualTrace(slug: string): ArticleVisualTrace | undefined {
  return articleVisualTraces[slug] ?? articleVisualTraceFallbackMap[slug];
}
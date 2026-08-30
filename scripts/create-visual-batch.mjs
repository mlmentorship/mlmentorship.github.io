import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import process from 'node:process';

const root = resolve(import.meta.dirname, '..');
const checker = resolve(root, 'scripts/check-visual-coverage.mjs');

function option(name, fallback) {
  const prefix = `--${name}=`;
  return process.argv.find((arg) => arg.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

const count = Math.max(1, Number(option('count', '6')) || 6);
const output = resolve(root, option('output', 'scratch/visual-batch.generated.json'));
const deckPath = '/mnt/c/Users/saghi/Downloads/Basic_ML_CS_concepts (1).pdf';
const statusOutput = execFileSync(process.execPath, [checker, '--next=10000'], {
  cwd: root,
  encoding: 'utf8',
});

const entries = statusOutput
  .split('\n')
  .filter((line) => line.startsWith('  ') && line.includes('\t'))
  .map((line) => {
    const [slug, status, category, file, ...titleParts] = line.trim().split('\t');
    return { slug, status, category, file, title: titleParts.join('\t') };
  })
  .slice(0, count);

if (entries.length === 0) {
  console.log('Visual coverage is complete; no batch generated.');
  process.exit(0);
}

function researchDescription(entry) {
  const prior = entry.status === 'planned'
    ? 'A planned audit already exists; independently verify it rather than trusting its medium or sources, then replace it with the final result. '
    : '';
  return `Own exactly one article and its audit sidecar: ${entry.file} and data/visual-audits/${entry.slug}.json. ${prior}Read the complete article and identify the hardest mental model to form from prose. Inspect the full local source deck at ${deckPath}; examine neighboring slides and record exact relevant page numbers, or document why none applies. Research primary papers, official documentation, or authoritative textbooks. Verify figure licensing explicitly; default to an original redraw with citation when reuse permission is unclear. Compare Mermaid, deterministic SVG, semantic HTML, interaction, paper reuse, and no visual. Implement the smallest useful spatial explanation, or record a rigorous no-visual decision. Add one learning objective, a direct Read it this way caption, accessible text, dark/light/mobile/print behavior, a unique article marker, and a complete schemaVersion 1 sidecar. Do not edit shared styles, scripts, configuration, or other articles. Reuse established visual primitives. Run: npm run check:visuals`;
}

const tasks = entries.map((entry, index) => ({
  id: `1.${index + 1}`,
  name: `Audit and visualize ${entry.title}`,
  priority: 101 + index,
  status: 'not-started',
  description: researchDescription(entry),
  files: [entry.file, `data/visual-audits/${entry.slug}.json`],
  acceptanceCriteria: [
    'The audit records exact deck review, authoritative sources, licensing treatment, medium comparison, reviewer identity, and one learning objective',
    'Any implemented visual is original or clearly licensed, technically correct, accessible, and useful without relying on color',
    'The article and sidecar have matching unique visual IDs, or the sidecar gives a substantive no-visual rationale',
    'The implementation is readable at 390 CSS pixels and compatible with dark mode and print',
    'npm run check:visuals passes',
  ],
  dependencies: [],
}));

const dependencyIds = tasks.map((task) => task.id);
const integrationFiles = [
  ...entries.flatMap((entry) => [entry.file, `data/visual-audits/${entry.slug}.json`]),
  'src/styles/global.css',
  'src/styles/variables.css',
  'scripts/check-visual-coverage.mjs',
  'docs/VISUAL_LEARNING_SYSTEM.md',
];
const expectedCompleteIncrease = entries.length;

tasks.push({
  id: '2.1',
  name: 'Cross-review and integrate visual batch',
  priority: 201,
  status: 'not-started',
  description: `Review all ${entries.length} article outcomes together against docs/VISUAL_LEARNING_SYSTEM.md, their source articles, the deck, and cited primary sources. Do not assume agent licensing claims are correct. Reject decorative visuals, copied figures without verified permission, incorrect equations or geometry, color-only meaning, malformed raw HTML, unreadable labels, duplicate IDs, and unnecessary interaction. Fix shared styles only when multiple articles need the same primitive. Run generated-output validation, normal and disabled-Workbook production builds, and browser checks in light, dark, 390-pixel mobile, and print-oriented layouts. Confirm the number of implemented plus no-visual outcomes increases by exactly ${expectedCompleteIncrease}, with no unrelated article edits. Run: npm run check && PUBLIC_PREP_TOOLS=false npm run build && npm run build`,
  files: [...new Set(integrationFiles)],
  acceptanceCriteria: [
    `Exactly ${entries.length} queued articles move from planned or unreviewed to implemented or justified no-visual status`,
    'Every generated visual retains intact accessible markup and a direct explanatory caption',
    'All new visuals pass technical, provenance, desktop, mobile, theme, and print review',
    'Normal and disabled-Workbook builds and all link checks pass',
  ],
  dependencies: dependencyIds,
});

const plan = {
  title: `Systematic Visual Learning Batch: ${entries[0].slug} to ${entries.at(-1).slug}`,
  description: `Independently research ${entries.length} queued articles and integrate one evidence-backed visual outcome per article.`,
  phases: [
    {
      id: 'phase-1',
      name: 'Independent per-article visual research',
      priority: 100,
      tasks: tasks.slice(0, -1),
    },
    {
      id: 'phase-2',
      name: 'Cross-review and integration',
      priority: 200,
      tasks: tasks.slice(-1),
    },
  ],
};

mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, `${JSON.stringify(plan, null, 2)}\n`);
console.log(`Generated ${relative(root, output)} with ${entries.length} independent article tasks:`);
for (const entry of entries) console.log(`  ${entry.slug} (${entry.status})`);

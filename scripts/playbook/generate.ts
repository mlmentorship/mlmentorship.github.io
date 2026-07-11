#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { buildCatalog } from './catalog';
import { buildPersonalizedPlaybook, describePlan } from './engine';
import { renderPlaybookHtml } from './render';
import { parseIntake } from './schema';
import { renderPdf } from './pdf';

interface CliOptions {
  intake?: string;
  output?: string;
  htmlOnly: boolean;
  anonymize: boolean;
  help: boolean;
}

function parseArgs(argv: string[]): CliOptions {
  const options: CliOptions = { htmlOnly: false, anonymize: false, help: false };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--intake' || argument === '-i') options.intake = argv[++index];
    else if (argument === '--out' || argument === '-o') options.output = argv[++index];
    else if (argument === '--html-only' || argument === '--no-pdf') options.htmlOnly = true;
    else if (argument === '--anonymize') options.anonymize = true;
    else if (argument === '--help' || argument === '-h') options.help = true;
    else throw new Error(`Unknown argument: ${argument}`);
  }
  return options;
}

function usage(): string {
  return `Personalized mlmentorship playbook generator

Usage:
  npm run playbook:generate -- --intake <intake.json> [--out <directory>] [--html-only] [--anonymize]

Required intake fields:
  candidateName, role, targetLevel, startDate, weeks, hoursPerWeek,
  rounds, domainTracks, selfRatings (all eight areas)

Outputs:
  plan.json                 Structured source of truth for the future app
  playbook.html             Print-ready, self-contained playbook
  playbook.pdf              A4 PDF with clickable resource links
  delivery-manifest.json    File hashes and plan ID for fulfillment records

Privacy:
  Outputs contain candidate-provided personal data by default. Keep intake and
  generated artifacts outside the public repository. --anonymize replaces the
  candidate name and removes every free-text intake field.

Environment:
  CHROME_PATH               Optional Chrome/Chromium executable override
  PLAYBOOK_PDF_TIMEOUT_MS   PDF generation timeout (minimum 5000; default 60000)`;
}

function fileHash(filePath: string): string {
  return createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function slugify(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'candidate';
}

export async function generatePlaybook(options: {
  intakePath: string;
  outputDirectory?: string;
  htmlOnly?: boolean;
  anonymize?: boolean;
  repoRoot?: string;
}) {
  const repoRoot = path.resolve(options.repoRoot ?? process.cwd());
  const intakePath = path.resolve(options.intakePath);
  const parsedIntake = parseIntake(JSON.parse(fs.readFileSync(intakePath, 'utf8')));
  const intake = options.anonymize
    ? {
      ...parsedIntake,
      candidateName: 'Candidate',
      experienceSummary: undefined,
      constraints: [],
      priorities: [],
    }
    : parsedIntake;
  const catalog = buildCatalog(repoRoot);
  const playbook = buildPersonalizedPlaybook(intake, catalog);
  const outputDirectory = path.resolve(
    options.outputDirectory ?? path.join(repoRoot, 'artifacts/playbooks', `${slugify(intake.candidateName)}-${playbook.planId}`),
  );
  fs.mkdirSync(outputDirectory, { recursive: true });

  const planPath = path.join(outputDirectory, 'plan.json');
  const htmlPath = path.join(outputDirectory, 'playbook.html');
  const pdfPath = path.join(outputDirectory, 'playbook.pdf');
  fs.writeFileSync(planPath, `${JSON.stringify(playbook, null, 2)}\n`);
  fs.writeFileSync(htmlPath, renderPlaybookHtml(playbook));
  if (!options.htmlOnly) await renderPdf(htmlPath, pdfPath);

  const files = [planPath, htmlPath, ...(!options.htmlOnly ? [pdfPath] : [])];
  const manifest = {
    schemaVersion: 1,
    planId: playbook.planId,
    engineVersion: playbook.engineVersion,
    candidateName: playbook.generatedFor,
    containsPersonalData: !options.anonymize,
    handling: options.anonymize
      ? 'Anonymized output; direct identity and all free-text intake fields were removed.'
      : 'Contains candidate-provided personal data. Store privately and do not commit or publish.',
    intakeFile: path.basename(intakePath),
    createdAt: new Date().toISOString(),
    files: files.map((filePath) => ({
      name: path.basename(filePath),
      bytes: fs.statSync(filePath).size,
      sha256: fileHash(filePath),
    })),
  };
  const manifestPath = path.join(outputDirectory, 'delivery-manifest.json');
  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

  return { playbook, outputDirectory, planPath, htmlPath, pdfPath: options.htmlOnly ? undefined : pdfPath, manifestPath };
}

async function main() {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) {
      console.log(usage());
      return;
    }
    if (!options.intake) throw new Error('--intake is required');
    const result = await generatePlaybook({
      intakePath: options.intake,
      outputDirectory: options.output,
      htmlOnly: options.htmlOnly,
      anonymize: options.anonymize,
    });
    console.log(`Generated ${result.playbook.planId}: ${describePlan(result.playbook)}`);
    console.log(`Output: ${result.outputDirectory}`);
    console.log(`Plan: ${result.planPath}`);
    console.log(`HTML: ${result.htmlPath}`);
    if (result.pdfPath) console.log(`PDF: ${result.pdfPath}`);
    console.log(`Manifest: ${result.manifestPath}`);
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    console.error('\nRun with --help for intake and output details.');
    process.exitCode = 1;
  }
}

const isDirectExecution = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname);
if (isDirectExecution) void main();

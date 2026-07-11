import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';

const COMMON_CHROME_PATHS = [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
];

export function findChrome(): string | undefined {
  const configured = process.env.CHROME_PATH;
  if (configured && fs.existsSync(configured)) return configured;
  return COMMON_CHROME_PATHS.find((candidate) => fs.existsSync(candidate));
}

function validPdf(pdfPath: string): boolean {
  if (!fs.existsSync(pdfPath)) return false;
  const size = fs.statSync(pdfPath).size;
  if (size < 20_000) return false;
  return fs.readFileSync(pdfPath).subarray(0, 5).toString('ascii') === '%PDF-';
}

export async function renderPdf(htmlPath: string, pdfPath: string, chromePath = findChrome()): Promise<void> {
  if (!chromePath) {
    throw new Error('Chrome/Chromium was not found. Set CHROME_PATH or use --html-only.');
  }

  const resolvedHtml = path.resolve(htmlPath);
  const resolvedPdf = path.resolve(pdfPath);
  fs.mkdirSync(path.dirname(resolvedPdf), { recursive: true });
  const userDataDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'mlmentorship-pdf-'));

  try {
    fs.rmSync(resolvedPdf, { force: true });
    const child = spawn(chromePath, [
      '--headless=new',
      '--disable-gpu',
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--disable-background-networking',
      '--disable-component-update',
      '--disable-default-apps',
      '--disable-extensions',
      '--disable-sync',
      '--metrics-recording-only',
      '--no-first-run',
      '--allow-file-access-from-files',
      '--print-to-pdf-no-header',
      `--user-data-dir=${userDataDirectory}`,
      `--print-to-pdf=${resolvedPdf}`,
      pathToFileURL(resolvedHtml).href,
    ], { stdio: ['ignore', 'ignore', 'pipe'] });

    let stderr = '';
    child.stderr?.on('data', (chunk) => { stderr += String(chunk); });
    let previousSize = -1;
    let stableChecks = 0;
    const configuredTimeout = Number(process.env.PLAYBOOK_PDF_TIMEOUT_MS ?? 60_000);
    const timeoutMs = Number.isFinite(configuredTimeout) && configuredTimeout >= 5_000
      ? configuredTimeout
      : 60_000;
    const deadline = Date.now() + timeoutMs;

    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 150));
      if (validPdf(resolvedPdf)) {
        const size = fs.statSync(resolvedPdf).size;
        stableChecks = size === previousSize ? stableChecks + 1 : 0;
        previousSize = size;
        if (stableChecks >= 3) break;
      }
      if (child.exitCode !== null && !validPdf(resolvedPdf)) {
        throw new Error(`Chrome PDF rendering failed (${child.exitCode}):\n${stderr}`);
      }
    }

    if (!validPdf(resolvedPdf)) {
      child.kill('SIGKILL');
      throw new Error(`Chrome did not produce a valid PDF within ${Math.round(timeoutMs / 1000)} seconds.\n${stderr}`);
    }

    if (child.exitCode === null) child.kill('SIGTERM');
    await new Promise<void>((resolve) => {
      if (child.exitCode !== null) return resolve();
      const force = setTimeout(() => { child.kill('SIGKILL'); resolve(); }, 2_000);
      child.once('exit', () => { clearTimeout(force); resolve(); });
    });
  } finally {
    fs.rmSync(userDataDirectory, { recursive: true, force: true });
  }
}

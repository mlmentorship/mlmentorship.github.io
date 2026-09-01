import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const problemsDir = path.join(path.dirname(fileURLToPath(import.meta.url)), 'problems');
const moduleNames = fs.readdirSync(problemsDir)
  .filter((name) => name.endsWith('.mjs'))
  .sort();

export const visualDefinitions = await Promise.all(
  moduleNames.map(async (name) => (await import(pathToFileURL(path.join(problemsDir, name)).href)).default),
);

export const codingQuestionVisuals = Object.freeze(Object.fromEntries(
  visualDefinitions.map((definition) => [definition.slug, definition]),
));

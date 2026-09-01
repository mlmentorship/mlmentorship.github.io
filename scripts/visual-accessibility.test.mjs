import assert from 'node:assert/strict';
import test from 'node:test';

import { unresolvedAccessibilityReferences } from './visual-accessibility.mjs';

test('accepts accessibility references that resolve within a figure', () => {
  const figure = `
    <figure aria-labelledby="figure-title" aria-describedby="figure-description">
      <h3 id="figure-title">Learning question</h3>
      <p id="figure-description">A complete text alternative.</p>
      <svg role="img" aria-labelledby="svg-title svg-description">
        <title id="svg-title">Chart title</title>
        <desc id="svg-description">Chart description</desc>
      </svg>
    </figure>`;

  assert.deepEqual(unresolvedAccessibilityReferences(figure), []);
});

test('reports missing and duplicate accessibility targets', () => {
  const figure = `
    <figure aria-labelledby="shared-title missing-title">
      <h3 id="shared-title">First title</h3>
      <p id="shared-title">Duplicate title</p>
      <svg role="img" aria-describedby="missing-description"></svg>
    </figure>`;

  assert.deepEqual(unresolvedAccessibilityReferences(figure), [
    'aria-labelledby references duplicate #shared-title',
    'aria-labelledby references missing #missing-title',
    'aria-describedby references missing #missing-description',
  ]);
});

test('reports duplicate IDs even when no accessibility attribute references them', () => {
  const figure = '<figure><span id="reused"></span><span id="reused"></span></figure>';

  assert.deepEqual(unresolvedAccessibilityReferences(figure), ['duplicate id #reused']);
});

---
title: "Cross-Entropy From Logits"
description: "Compute mean multiclass cross-entropy from logits and integer labels."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Intermediate"
priority: "Role-specific"
aliases: []
prerequisites: []
---

Compute the average categorical prediction loss from a batch of raw class scores and correct class labels.

## Problem

**Input.** Float64 `logits` with shape `(batch, classes)` and integer `labels` with shape `(batch,)`.

**Output.** Return one floating-point mean negative log-probability for the labeled classes, using natural logarithms.
Each row's class probabilities are its softmax distribution; the input scores are not already probabilities.

**Rules.** Both dimensions are positive, and every label is a valid zero-based class index.
Require finite logits with representable differences. Repeated labels are valid; preserve both inputs.
Empty means and ignored labels are outside this reference contract.

**Example.** Logits `[[0, 0], [0, 0]]` and labels `[0, 1]` return approximately `0.693147`.
Each correct class has probability one half, so both observations contribute the same loss.

## Picture

For logits `[[2,1,0],[0,1,2]]` and labels `[0,1]`, losses are approximately `0.407606` and `1.407606`.
Their mean is approximately `0.907606`.
Each label selects its own row, not a shared column slice.

<!-- visual:cross-entropy-from-logits-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="cross-entropy-from-logits-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="cross-entropy-from-logits-state-title">The target logit changes the loss even when the normalizer stays equal</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="cross-entropy-from-logits" role="group" tabindex="0" aria-label="Cross-Entropy From Logits: Compute stable log-normalizers per example, subtract the indexed correct logits, then average example losses."><div class="coding-visual-example"><span>Input and goal</span><strong>Compute mean multiclass cross-entropy from logits and integer labels.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" data-frame-key="inputs" role="group" aria-label="Read logits and integer labels"><div class="coding-trace-frame-heading"><span>Read logits and integer labels</span><strong>Use logits [[2,1,0],[0,1,2]] with labels [0,1]. Each label selects one class in its own row.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">example</th><th scope="col">class 0</th><th scope="col">class 1</th><th scope="col">class 2</th><th scope="col">label</th></tr></thead><tbody><tr><td class="" data-motion-key="table-0-0">0</td><td class="is-active" data-motion-key="table-0-1">2</td><td class="" data-motion-key="table-0-2">1</td><td class="" data-motion-key="table-0-3">0</td><td class="" data-motion-key="table-0-4">0</td></tr><tr><td class="" data-motion-key="table-1-0">1</td><td class="" data-motion-key="table-1-1">0</td><td class="" data-motion-key="table-1-2">1</td><td class="" data-motion-key="table-1-3">2</td><td class="is-active" data-motion-key="table-1-4">1</td></tr></tbody></table></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at the first checkpoint"><dl><div class="coding-state-change"><dt>example / row 0</dt><dd><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>class 0 / row 0</dt><dd><strong class="coding-state-after">2</strong></dd></div><div class="coding-state-change"><dt>class 1 / row 0</dt><dd><strong class="coding-state-after">1</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="1" data-frame-key="shift" hidden role="group" aria-label="Shift each row by its maximum"><div class="coding-trace-frame-heading"><span>Shift each row by its maximum</span><strong>Both row maxima are 2. Shifted rows become [0,-1,-2] and [-2,-1,0], keeping the selected entries 0 and -1.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">example</th><th scope="col">class 0</th><th scope="col">class 1</th><th scope="col">class 2</th><th scope="col">label</th></tr></thead><tbody><tr><td class="" data-motion-key="table-0-0">0</td><td class="is-active" data-motion-key="table-0-1">0</td><td class="" data-motion-key="table-0-2">-1</td><td class="" data-motion-key="table-0-3">-2</td><td class="" data-motion-key="table-0-4">0</td></tr><tr><td class="" data-motion-key="table-1-0">1</td><td class="" data-motion-key="table-1-1">-2</td><td class="" data-motion-key="table-1-2">-1</td><td class="is-active" data-motion-key="table-1-3">0</td><td class="" data-motion-key="table-1-4">1</td></tr></tbody></table></div><div class="coding-trace-meta"><span><b>maxima</b>[2,2]</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>maxima</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">[2,2]</strong></dd></div><div class="coding-state-change"><dt>class 0 / row 0</dt><dd><span class="coding-state-before">2</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>class 1 / row 0</dt><dd><span class="coding-state-before">1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">-1</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="2" data-frame-key="normalizers" hidden role="group" aria-label="Compute row log-normalizers"><div class="coding-trace-frame-heading"><span>Compute row log-normalizers</span><strong>Each shifted row has exponential sum 1 + 0.3679 + 0.1353 = 1.5032, so log_normalizer = log(1.5032) = 0.4076.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">example</th><th scope="col">exp sum</th><th scope="col">log normalizer</th></tr></thead><tbody><tr><td class="" data-motion-key="table-0-0">0</td><td class="" data-motion-key="table-0-1">1.5032</td><td class="is-active" data-motion-key="table-0-2">0.4076</td></tr><tr><td class="" data-motion-key="table-1-0">1</td><td class="" data-motion-key="table-1-1">1.5032</td><td class="is-active" data-motion-key="table-1-2">0.4076</td></tr></tbody></table></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>maxima</dt><dd><span class="coding-state-before">[2,2]</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after"><span class="coding-state-absent">not shown</span></strong></dd></div><div class="coding-state-change"><dt>exp sum / row 0</dt><dd><span class="coding-state-before">0</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1.5032</strong></dd></div><div class="coding-state-change"><dt>log normalizer / row 0</dt><dd><span class="coding-state-before">-1</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0.4076</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="3" data-frame-key="losses" hidden role="group" aria-label="Subtract each selected logit"><div class="coding-trace-frame-heading"><span>Subtract each selected logit</span><strong>Example 0 loss is 0.4076 - 0 = 0.4076. Example 1 label 1 selects shifted logit -1, so loss is 0.4076 - (-1) = 1.4076.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">example</th><th scope="col">normalizer</th><th scope="col">correct shifted logit</th><th scope="col">loss</th></tr></thead><tbody><tr><td class="" data-motion-key="table-0-0">0</td><td class="" data-motion-key="table-0-1">0.4076</td><td class="" data-motion-key="table-0-2">0</td><td class="is-active" data-motion-key="table-0-3">0.4076</td></tr><tr><td class="" data-motion-key="table-1-0">1</td><td class="" data-motion-key="table-1-1">0.4076</td><td class="" data-motion-key="table-1-2">-1</td><td class="is-active" data-motion-key="table-1-3">1.4076</td></tr></tbody></table></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>normalizer / row 0</dt><dd><span class="coding-state-before">1.5032</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0.4076</strong></dd></div><div class="coding-state-change"><dt>correct shifted logit / row 0</dt><dd><span class="coding-state-before">0.4076</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0</strong></dd></div><div class="coding-state-change"><dt>loss / row 0</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">0.4076</strong></dd></div></dl></div></div><div class="coding-trace-frame" data-coding-frame="4" data-frame-key="mean" hidden role="group" aria-label="Average the batch"><div class="coding-trace-frame-heading"><span>Average the batch</span><strong>Mean cross-entropy = (0.4076 + 1.4076) / 2 = 0.9076.</strong></div><div class="coding-trace-table-wrap"><table class="coding-trace-table"><thead><tr><th scope="col">example</th><th scope="col">loss</th></tr></thead><tbody><tr><td class="" data-motion-key="table-0-0">0</td><td class="" data-motion-key="table-0-1">0.4076</td></tr><tr><td class="" data-motion-key="table-1-0">1</td><td class="" data-motion-key="table-1-1">1.4076</td></tr><tr><td class="" data-motion-key="table-2-0">mean</td><td class="is-active" data-motion-key="table-2-1">0.9076</td></tr></tbody></table></div><div class="coding-trace-meta"><span><b>arithmetic</b>1.8152 / 2 = 0.9076</span><span><b>result</b>mean = 0.9076</span></div><div class="coding-state-comparison" role="group" aria-label="Selected values shown at consecutive authored checkpoints"><dl><div class="coding-state-change"><dt>arithmetic</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1.8152 / 2 = 0.9076</strong></dd></div><div class="coding-state-change"><dt>result</dt><dd><span class="coding-state-before"><span class="coding-state-absent">not shown</span></span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">mean = 0.9076</strong></dd></div><div class="coding-state-change"><dt>loss / row 1</dt><dd><span class="coding-state-before">0.4076</span><span class="coding-state-arrow" role="img" aria-label="shown next as">&rarr;</span><strong class="coding-state-after">1.4076</strong></dd></div></dl></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 5</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Read logits and integer labels</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Shift each row by its maximum</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Compute row log-normalizers</strong></button><button type="button" data-coding-frame-button="3"><span>4</span><strong>Subtract each selected logit</strong></button><button type="button" data-coding-frame-button="4"><span>5</span><strong>Average the batch</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><dl class="coding-visual-lessons" aria-label="Pattern recognition and transfer"><div class="coding-visual-lesson"><dt>Recognize it</dt><dd data-coding-review="recognitionCue">Use it when multiclass logits and integer labels require negative log likelihood without explicitly forming potentially unstable probabilities.</dd></div><div class="coding-visual-lesson"><dt>Keep true</dt><dd data-coding-review="invariant">For each row, log_normalizer minus the shifted selected logit equals negative log probability of the label; max shifting cancels from both terms.</dd></div><div class="coding-visual-lesson"><dt>Reuse it</dt><dd data-coding-review="transferLesson">For stable classification losses, compute log-normalization in score space, gather the target score by aligned row indices, and reduce only after producing per-example losses.</dd></div></dl></div><figcaption><strong>Read it this way:</strong> Use logits [[2,1,0],[0,1,2]] with labels [0,1]. Each label selects one class in its own row. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition. <span class="visual-reader-check" data-reader-check="cross-entropy-from-logits-state"><strong>Check:</strong> Why do equal normalizers produce losses 0.4076 and 1.4076 for the two rows?</span></figcaption></figure>

The [cross-entropy owner](/concepts/cross-entropy-softmax/) derives the likelihood and gradient.
Subtracting the same row maximum from every logit preserves probabilities while preventing positive exponential arguments.

## Code

```python
import numpy as np

def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
   shifted = logits - np.max(logits, axis=1, keepdims=True)
   log_normalizer = np.log(np.sum(np.exp(shifted), axis=1))
   correct_logits = shifted[np.arange(len(labels)), labels]
   return float(np.mean(log_normalizer - correct_logits))
```

**Cost:** $O(batch\times classes)$ arithmetic and temporary storage, despite the scalar result.

## Change the Requirement

### Weight the Examples

**New task:** Return a weighted mean using nonnegative finite weights with a positive representable total.

**Picture and change:** Keep per-row losses until weighting; divide by weight sum rather than batch size.

**Code:** Retain the original logits and label contract.

```python
def weighted_cross_entropy(logits, labels, weights) -> float:
   weights = np.asarray(weights, dtype=np.float64)
   if weights.shape != labels.shape or np.any(~np.isfinite(weights)) or np.any(weights < 0):
      raise ValueError("weights must match labels and be finite and nonnegative")
   total = weights.sum()
   if not np.isfinite(total) or total <= 0:
      raise ValueError("weight total must be positive and representable")
   shifted = logits - logits.max(axis=1, keepdims=True)
   normalizer = np.log(np.exp(shifted).sum(axis=1))
   losses = normalizer - shifted[np.arange(len(labels)), labels]
   return float(np.sum(losses * (weights / total)))
```

**Check:** Weights `[3,1]` change the example mean to approximately `0.657606`; `[0,0]` raises `ValueError`.

**Use when:** Example weights define the intended reporting or training population.

### Ignore Selected Labels

**New task:** Exclude rows with label `-1` from the mean.

**Picture and change:** Filter both logits and labels before gathering or reducing.

**Code:** Nonignored labels retain the valid class-index range.

```python
def cross_entropy_ignoring(logits, labels, ignored: int = -1) -> float:
   valid = labels != ignored
   if not np.any(valid):
      raise ValueError("mean loss needs at least one retained example")
   return cross_entropy(logits[valid], labels[valid])
```

**Check:** Ignoring the second row leaves loss approximately `0.407606`; ignoring both rows raises `ValueError`.

**Use when:** Padding is not a labeled observation.
Boolean selection allocates copies; both alternatives retain linear-in-logits time and storage bounds.

<!-- learning:core-end -->
<div data-learning-extension>

## Reconstruct

Why not pass ignored negative labels directly to the reference?

<details>
<summary>Reasoned answer</summary>

NumPy treats `-1` as the last column index, not an ignore instruction.
For the first example row, this silently selects class two and gives approximately `2.407606` instead of excluding the example.
The gathering operation cannot infer the meaning of a sentinel.
Define filtering and the empty-result policy before evaluating the reduction.

</details>

</div>

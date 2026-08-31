---
title: "Investigate a black-box model behavior in 90 minutes"
description: "Turn an observation into competing hypotheses, discriminating probes, boundary conditions, and one decision-changing result."
date: "2026-07-11"
draft: false
tags: ["questions"]
category: "questions"
---

> You can query an undocumented model through an API. Investigate one surprising behavior and present your conclusion, evidence, uncertainty, and next experiment.

The senior signal is experimental compression: how much uncertainty did each query remove? A hundred prompt variations are weaker than six probes chosen because competing hypotheses predict different outcomes.

## Begin with a behavioral claim

A useful claim is observable and bounded:

> "When tool output contains an instruction that conflicts with the user request, the model follows the tool instruction more often if the requested action is low risk."

This is better than "the model trusts tools" because it identifies the input condition, behavior, comparison, and likely interaction. It can be falsified.

Do not begin with a hidden-architecture story. Many mechanisms can generate the same output. First establish behavior.

## Build competing hypotheses

For the claim above:

- **H1, source priority:** tool text is treated as higher priority than user context.
- **H2, lexical trigger:** a small set of override phrases causes the behavior regardless of source.
- **H3, risk gate:** the model follows conflicting text only when a separate risk classifier allows the action.
- **H4, recency:** the last instruction wins, and source labels are incidental.

Now choose probes where predictions diverge. Move the same phrase between context and tool output. Change order without changing content. Vary risk while holding the conflict constant. Paraphrase the trigger. Include a control with no conflict.

<!-- visual:black-box-discriminating-probes -->
<figure class="learning-figure" aria-labelledby="black-box-probes-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="black-box-probes-title">Choose the next query by asking where the hypotheses disagree.</p>
	<div class="visual-grid--two" role="group" aria-label="Comparison of an ambiguous observation and four matched discriminating probes">
		<section class="visual-panel" aria-labelledby="black-box-ambiguous-title">
			<h4 id="black-box-ambiguous-title">One observation fits every story</h4>
			<p>Setup: an exact override phrase appears last in tool output and requests a low-risk action. The model follows it.</p>
			<table class="cm-grid" aria-label="Four hypotheses that all predict the observed behavior">
				<thead><tr><th scope="col">hypothesis</th><th scope="col">prediction</th></tr></thead>
				<tbody>
					<tr><th scope="row">H1 source priority</th><td>follow conflict</td></tr>
					<tr><th scope="row">H2 lexical trigger</th><td>follow conflict</td></tr>
					<tr><th scope="row">H3 risk gate</th><td>follow conflict</td></tr>
					<tr><th scope="row">H4 recency</th><td>follow conflict</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">result: 4 explanations remain compatible</p>
		</section>
		<section class="visual-panel" aria-labelledby="black-box-discriminate-title">
			<h4 id="black-box-discriminate-title">Change one factor; predict before querying</h4>
			<p>Hold the other factors fixed. A switch back to the user request singles out one explanation.</p>
			<table class="cm-grid" aria-label="Four one-factor probes and the hypothesis uniquely supported if behavior switches">
				<thead><tr><th scope="col">one change</th><th scope="col">switch supports</th><th scope="col">rivals predict</th></tr></thead>
				<tbody>
					<tr><th scope="row">tool to context</th><td class="cm-selected"><strong>H1</strong>stop</td><td>follow</td></tr>
					<tr><th scope="row">paraphrase trigger</th><td class="cm-selected"><strong>H2</strong>stop</td><td>follow</td></tr>
					<tr><th scope="row">low to high risk</th><td class="cm-selected"><strong>H3</strong>stop</td><td>follow</td></tr>
					<tr><th scope="row">conflict first, user last</th><td class="cm-selected"><strong>H4</strong>stop</td><td>follow</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">best next probe = largest useful prediction split</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> the first query confirms the behavior but cannot choose among H1-H4 because all four predict the same outcome. Spend the next query on a matched contrast: change only source, phrase, risk, or order and write the predictions first. If behavior switches only under one contrast, that result removes more uncertainty than another unstructured prompt variation. These predictions illustrate the stated hypotheses, not hidden model internals. Original synthesis informed by <a href="https://doi.org/10.1126/science.146.3642.347">Platt's strong-inference method</a> and <a href="https://aclanthology.org/2020.acl-main.442/">CheckList behavioral testing</a>.</figcaption>
</figure>

## Keep an experiment ledger

For each probe, record:

| Field | Purpose |
| --- | --- |
| Hypothesis | Which explanation does this test? |
| Manipulation | What changed from control? |
| Predicted outcomes | What does each hypothesis expect? |
| Observation | Exact decision, score, or output |
| Update | Which beliefs moved, and how? |
| Next probe | Highest-information follow-up |

If the model is stochastic, repeated samples and fixed decoding settings matter. If the API hides probability, use a categorical rubric and report uncertainty honestly.

## What an L4 answer sounds like

> "I tried many jailbreak prompts. Some worked, so the model is vulnerable to prompt injection. I would add more safety training."

The conclusion is broader than the evidence. There is no control, no denominator, no competing explanation, and no boundary condition. "More safety training" is not connected to a measured failure mechanism.

## What an L5 answer adds

An L5 candidate states a bounded claim, creates competing hypotheses, varies one factor at a time, and then tests an interaction. They preserve negative results and distinguish deterministic checks from model-graded judgments.

Their readout fits five minutes:

1. claim;
2. experiment matrix;
3. result that most changed belief;
4. boundary or uncertainty;
5. next experiment.

They do not show every prompt. They show the evidence needed for the conclusion.

## What an L6 answer adds

An L6 candidate connects the investigation to a decision. They ask whether the observed behavior is prevalent, severe, reachable in production, and mitigated elsewhere in the system. A black-box curiosity becomes actionable only when tied to a threat model, product surface, training intervention, or evaluation gap.

They also anticipate evaluation traps:

- Prompt families are correlated, so 100 paraphrases are not 100 independent failures.
- A model grader may share the target model's blind spot.
- Safety behavior can improve while useful behavior regresses.
- A static prompt set can be overfit by post-training.
- API nondeterminism can look like an interaction effect.
- Model-version changes can invalidate the conclusion.

They version prompts, model, decoding, tools, and grader. They define the result that would block launch or change training.

## Tells that get you a strong-hire vote

- The first claim is falsifiable and bounded.
- At least three hypotheses predict different probe outcomes.
- Controls precede interactions.
- Negative results remain in the evidence table.
- You report uncertainty and stochasticity.
- The readout prioritizes one decision-changing result.
- You propose the next experiment by information value.
- You avoid claiming hidden mechanisms from outputs alone.

## Tells that get you down-leveled

- A large prompt dump with no experiment design.
- Counting paraphrases as independent evidence.
- Selecting only successful attacks.
- Changing multiple factors in every probe.
- Treating a model judge as ground truth.
- Inferring internals from one behavior.
- Ending with "collect more data" rather than a specific next test.

## Common follow-up

"Your leading hypothesis explains 90% of observations. What do you do with the other 10%?"

First determine whether the residual is noise, a mislabeled outcome, or a missing condition. Cluster residuals by input features and rerun exact cases under controlled decoding. A theory that predicts only the majority may still be useful, but the tail can reveal the safety-critical interaction. Do not average it away before inspecting severity and reachability.

Use the [black-box behavior lab](/prep/labs/black-box/) for an executable attempt.

*Related: [design an ablation study](/questions/design-ablation-study/), [critique an ML paper](/questions/critique-ml-paper/), and [chain-of-thought monitorability](/concepts/chain-of-thought-monitorability/).*

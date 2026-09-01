---
title: "What's the most over-rated technique in ML right now?"
description: "A trap question that rewards taste. Strong opinions, defended with reasoning, are the senior signal. Weak opinions or 'I don't know' both lose."
date: "2025-12-07"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: behavioral and culture-fit rounds at senior loops; Anthropic, OpenAI, and similar opinion-rich teams especially.*

The question tests whether you have *taste*: can you form a defensible opinion about something the field overhypes? "I don't have one" is a bad answer. So is "everything is great."

## What an L4 answer sounds like

> "I don't think I have strong opinions on that. They all seem useful in different contexts."

Honest, unhelpful. You've consumed techniques without forming judgments about them.

## What an L5 answer sounds like

A strong answer picks a real target and defends it. Some defensible candidates in 2026:

> "Chain-of-thought prompting as a *generic* improvement. CoT helps for problems with explicit reasoning structure; it doesn't help for problems where the bottleneck is knowledge retrieval or pattern matching. The 'always add chain-of-thought' folk wisdom wastes tokens on tasks that don't benefit, and the cost of that compounds at scale.
>
> The right framing: CoT is a tool for problems where the model can do the reasoning if prompted to lay it out, not a free quality boost."

Or:

> "Knowledge distillation as a one-size-fits-all. Distilling a large model into a small one has a quality ceiling; for many tasks the small model lacks the capacity to absorb what the teacher knows, no matter how good the distillation procedure. Better framing: distillation is useful for compressing within a quality envelope, not for stretching the envelope.
>
> The 7B-distilled-from-70B family of models is a perfect example. They're impressive on benchmarks where 7B is enough; they fall over on tasks that genuinely need 70B parameters."

Or:

> "Synthetic data as a free fix for data shortages. Synthetic data works well when (a) the generator model is much stronger than the student, (b) the synthetic data is filtered by an even stronger judge, and (c) the use case is bounded. It fails when the synthetic data has systematic biases of the generator that the student inherits and amplifies."

The key qualities:
- Specific, not vague.
- Defends with a *reason*, not just dislikes the term.
- Acknowledges where the technique *does* work.

<!-- visual:overrated-technique-claim-stress-test -->
<figure class="learning-figure" aria-labelledby="overrated-technique-stress-test-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="overrated-technique-stress-test-title">What turns a hot take into a defensible technical judgment?</p>
	<div class="visual-grid--two" role="group" aria-label="Side-by-side comparison of an unsupported hot take and a bounded, testable critique">
		<section class="visual-panel" aria-labelledby="hot-take-title">
			<h4 id="hot-take-title">Hot take: “CoT is overrated.”</h4>
			<p>A verdict without a scope or mechanism gives the interviewer nothing technical to test.</p>
			<table class="cm-grid" aria-label="Missing parts of the unsupported claim">
				<tbody>
					<tr><th scope="row">Target</th><td>Which use?</td></tr>
					<tr><th scope="row">Mechanism</th><td>Why?</td></tr>
					<tr><th scope="row">Boundary</th><td>When useful?</td></tr>
					<tr><th scope="row">Evidence</th><td>What changed?</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">Result: contrarian preference</p>
		</section>
		<section class="visual-panel" aria-labelledby="bounded-critique-title">
			<h4 id="bounded-critique-title">Bounded critique: generic CoT use</h4>
			<p>Make the claim narrower, causal, and falsifiable while preserving the case where the technique earns its cost.</p>
			<table class="cm-grid" aria-label="Four parts of a defensible critique">
				<tbody>
					<tr><th scope="row">Target</th><td class="cm-selected">Generic use on every task</td></tr>
					<tr><th scope="row">Mechanism</th><td>Extra reasoning tokens cannot supply missing knowledge</td></tr>
					<tr><th scope="row">Boundary</th><td>Useful when explicit intermediate reasoning is the bottleneck</td></tr>
					<tr><th scope="row">Evidence</th><td>Compare task metric and token cost with and without it</td></tr>
				</tbody>
			</table>
			<p class="cm-equation">Result: a claim experience can update</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> move down the right-hand column before defending the verdict: name the exact use you oppose, explain its failure mechanism, mark the boundary where it works, and state what evidence would change your mind. “Overrated” should mean over-applied beyond an evidence-backed envelope, not universally useless. This is an original comparison informed by the task-specific results in <a href="https://arxiv.org/abs/2201.11903">Wei et al.'s chain-of-thought study</a>, the deployment framing in <a href="https://arxiv.org/abs/1503.02531">Hinton et al.'s distillation paper</a>, and the failure conditions studied in <a href="https://arxiv.org/abs/2305.17493">Shumailov et al.'s generated-data work</a>.</figcaption>
</figure>

## What an L6 answer adds

The L6 version often:
- Picks a target that's currently fashionable, not a dead horse.
- Has *experience-based* reasoning ("we tried this and saw [specific failure]") rather than abstract critique.
- Acknowledges the technique's appeal and explains *why* people overhype it.
- Avoids being a contrarian for its own sake; the criticism is constructive.

## Tells that get you a strong-hire vote

- You pick a **specific, current** target.
- You explain the **mechanism of overhype**, not just the existence.
- You acknowledge **where the technique works**.
- You can defend the position under follow-up.

## Tells that get you down-leveled

- "I don't have strong opinions."
- "Everything has its place." (Diplomatic but vacuous.)
- Picking dead horses (LSTMs, GANs, mainframes).
- Holding the position weakly ("well, maybe it's not really overrated...").
- Negativity without substance.

## Mirror question to be ready for

The interviewer often follows with "What's *underrated* right now?" Defensible candidates: 

- **Eval engineering** as a discipline. Most teams underinvest because eval doesn't ship. The teams that invest see disproportionate returns.
- **BM25 / lexical retrieval.** Embeddings get the attention; hybrid retrieval (dense + lexical) consistently beats either alone, but the lexical half gets neglected in many production systems.
- **Calibration.** Model accuracy gets headlines; calibrated probabilities matter more for downstream decisions and are routinely ignored.
- **Distillation onto a fixed quality target.** Most distillation discussions focus on 'can the student match the teacher?' The more useful question is 'can the student hit the quality the product needs at lower cost?'

Pick whichever you can defend with technical content.

## Common follow-up

"What if your interviewer enthusiastically uses the technique you just called overrated?"

The L6 answer:

> "I'd ask what they're using it for and whether it's actually moving their metric. If yes, my generic critique doesn't apply to their case; the technique is right for their problem and that's the point. The criticism is about *generic application*, not about whether it ever works. If their use case matches the criticism, that's a productive conversation about evidence; not a fight about positions."

---

*Related: [The 5 things every applied scientist interview is testing for](/guides/five-things-as-interview-tests/), [senior through senior-principal ML scope](/guides/l5-vs-l6-faang-ml/).*

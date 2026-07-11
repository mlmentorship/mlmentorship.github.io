---
title: "Frontier-lab proof of work: the 100-word claim, artifacts, and references"
description: "Make one exceptional contribution easy to verify, then choose references who can independently describe the same decisions and working style."
date: "2026-07-11"
draft: false
tags: ["guides"]
category: "guides"
---

A strong frontier-lab application makes one claim easy to verify: you owned difficult work, produced evidence, and can explain what changed because of it. A long technology list is weaker than one contribution with a clear decision thread.

The same proof should survive three surfaces:

1. a 100-word exceptional-work statement;
2. a technical artifact or presentation;
3. an independent reference.

If those tell incompatible stories, the application feels inflated even when every sentence is technically true.

## Choose one contribution

Select work with:

- a hard technical or research constraint;
- a decision you personally owned;
- evidence or an artifact;
- measurable consequence;
- a failure, trade-off, or changed belief;
- relevance to the target role.

Do not choose by employer prestige. Choose by ownership and evidence.

## The 100-word structure

Use five moves:

1. **Problem:** one sentence on stakes and constraint.
2. **Ownership:** one sentence on what you personally decided or built.
3. **Technical difficulty:** one sentence on the mechanism or bottleneck.
4. **Evidence:** one sentence on measured outcome or research result.
5. **Why it matters:** one sentence on adoption, learning, or transfer.

Example:

> I owned the evaluation and launch decision for a multilingual retrieval model serving 18 markets. Offline gains hid a severe low-resource-language regression, so I built a slice-aware benchmark, traced the failure to negative-sampling imbalance, and changed both training data and the release gate. The revised model recovered the harmed slices while preserving the aggregate gain, then became the default for two adjacent teams. The important result was not the model architecture; it was making language-level evidence a launch requirement before the next incident.

This is 77 words. It gives scope, ownership, mechanism, evidence, and system impact without pretending one person built everything.

## Artifact selection

Use the artifact that makes the central claim easiest to inspect:

- repository or pull request for implementation quality;
- paper, experiment report, or notebook for research judgment;
- design document for architecture and trade-offs;
- benchmark or dataset card for evaluation and data work;
- incident retrospective for debugging and operational ownership;
- talk or deck for communication and project defense;
- product or demo for shipped user value.

An artifact does not need to be public if confidentiality prevents it. Create a sanitized architecture, synthetic example, or precise verbal reconstruction. Never leak employer information to prove credibility.

## Prepare the verification path

For every claim, know:

- what evidence exists;
- what is observed versus estimated;
- what the team owned;
- what you owned;
- who can independently verify it;
- which details are confidential;
- what failed or remained unresolved.

The interviewer will often probe the boundary because inflated ownership is common and easy to detect.

## References are independent evidence

Choose people who directly observed the relevant behavior. A close collaborator who saw the decisions is usually more useful than a senior executive who knows only the outcome.

Strong coverage across two or three references can include:

- former manager: scope, judgment, reliability, growth;
- technical peer: depth, collaboration, code or research quality;
- cross-functional partner: influence, communication, user or product judgment;
- research mentor: hypothesis quality, rigor, independence, response to failure.

Ask permission early. Share the role, current resume, and the work you expect to discuss. Do not script praise. Remind them of accurate context so they can provide specific, honest evidence.

## What references may expose

A reference can reveal:

- the candidate claimed team work as personal ownership;
- the impact number lacks context;
- the candidate was brilliant but unreliable;
- conflict stories omitted their contribution to the conflict;
- a project succeeded despite, not because of, the highlighted decision;
- the candidate changed materially since an old failure.

The right response is alignment with reality, not coaching everyone into identical language.

## Build one evidence table

| Claim | Artifact | Your ownership | Team ownership | Independent witness | Caveat |
| --- | --- | --- | --- | --- | --- |
| Improved low-resource retrieval | Slice benchmark and launch report | Eval design, diagnosis, launch gate | Training platform and rollout | Manager, localization lead | Causal contribution bounded by concurrent data cleanup |

If a major claim has no artifact, no direct witness, and no bounded caveat, rewrite it.

## Common failure modes

- A 100-word biography instead of one contribution.
- Technology names replacing the decision.
- Metrics with no baseline, denominator, or attribution.
- Public artifacts unrelated to the claimed work.
- References chosen only by title.
- Asking a reference after the company contacts them.
- Feeding references a script that erases independent judgment.
- Sharing confidential code or data as proof.
- Describing only success and leaving no sign of learning.

## Preparation checklist

1. Write the 100-word statement.
2. Highlight every ownership verb and verify it is literally true.
3. Attach one artifact or sanitized reconstruction.
4. Prepare the 30-minute project presentation around the same decisions.
5. Build the claim-to-evidence table.
6. Ask references and confirm contact details.
7. Tell each reference which real work may come up.
8. Reconcile any difference between resume, deck, artifact, and reference context.

The goal is not a coordinated performance. It is a coherent body of evidence.

*Related: [present a technical ML project](/questions/present-technical-ml-project/), [your most ambitious project](/questions/most-ambitious-project/), and [technical presentation practice](/prep/presentation/).*

---
title: "Frontier AI lab interviews in 2026: prepare for the format, not the logo"
description: "OpenAI, Anthropic, DeepMind, Meta, and xAI now test different combinations of codebases, presentations, research work, AI tools, and technical depth."
date: "2026-07-11"
updated: "2026-07-11"
draft: false
tags: ["guides"]
category: "guides"
---

The highest-leverage preparation decision is identifying the actual work sample. A candidate expecting whiteboard ML design can waste weeks if the round is an existing codebase, a black-box notebook, an accelerator trace, or a technical project presentation.

The company name does not determine the loop. Team, role, level, and current hiring experiment do.

## What is actually changing

Three changes matter.

First, realistic work samples are expanding. Official company pages now describe pair coding, take-homes, CodeSignal, Colab, AI-assisted CoderPad, and role-specific technical assessments. Current reports add multi-file projects, research discussions, presentations, and practical ML debugging.

Second, AI policy is part of the assessment. Meta says candidates are expected to use its authorized assistant in many interviews. Anthropic says no AI in take-homes or live interviews unless the round explicitly permits it. OpenAI's official guide does not publish one universal rule, while a current report describes an agentic-coding pilot. Permission cannot be inferred from company identity.

Third, surface fluency is cheaper. Interviewers can probe deeper with changed constraints, code review, causal experiments, and questions about why one line or assumption matters. Memorized answers have less room to hide.

## Evidence quality matters

Use this order:

1. official process and candidate-guidance pages;
2. role-specific instructions from recruiting;
3. reports that explain their method and recency;
4. first-person accounts for that candidate's role and date;
5. anonymous anecdotes only as questions to ask, never as facts.

Do not turn one team's performance take-home into "the Anthropic interview." Do not turn one Fellows experience into a full-time Research Engineer process. Do not publish an alleged exact prompt because an SEO page repeats it.

## Current format map

| Lab | Officially visible signal | Additional current report | Best targeted practice |
| --- | --- | --- | --- |
| OpenAI | Pair coding, take-home projects, technical tests, 4 to 6 final interviews | Practical multi-part coding, system design, presentation, agentic pilot | Codebase lab, presentation, inference design, math oral |
| Anthropic | Colab and CodeSignal, explicit AI rules, public performance challenge | Progressive coding, project depth, values round, research work samples | Closed-book coding, values, black-box research, performance |
| Google DeepMind | Recruiter, 2 or 3 skills calls, final team-lead interviews | Executable coding, mathematical ML, ML design | Math oral, ML implementation, design, paper critique |
| Meta | Authorized AI assistant in CoderPad and Mermaid design for many roles | Multi-file AI-assisted onsite format | Agentic codebase, AI review, testing, design with diagrams |
| xAI | Technical staff review, short technical screen, deep technical interviews | Public evidence remains sparse | Exceptional-work statement, domain depth, project defense |

See the [live source-labeled matrix](/prep/frontier-labs/) for links, verification dates, and caveats.

## OpenAI

The [official interview guide](https://openai.com/interview-guide/) says skills assessments vary by team and can include pair coding, take-home projects, and technical tests. Final interviews typically span 4 to 6 hours with 4 to 6 people. Engineering evaluation includes design, code quality, performance, tests, communication, and collaboration.

That alone should change preparation. Working code and test coverage are first-class signals, not a side effect of reaching the right idea.

A [June 2026 interviewing.io report](https://interviewing.io/openai-interview-questions), based on current conversations with OpenAI engineers, describes practical multi-part coding, architecture design, a technical presentation for many senior loops, and a beta agentic-coding round. These are reported formats, not universal policy.

Current job scope also shows where technical depth is moving. OpenAI roles now emphasize RL environments, graders, synthetic data, monitorability, distributed RL training, and debugging across model behavior, data, inference, orchestration, and agent harnesses. Job descriptions are not question lists, but they are better domain signals than recycled interview trivia.

## Anthropic

[Anthropic Careers](https://www.anthropic.com/careers) confirms live coding in Colab and CodeSignal. Its [candidate AI guidance](https://www.anthropic.com/candidate-ai-guidance) is explicit: no AI in take-homes or live interviews unless the instructions permit it.

The strongest public work sample is Anthropic's own [performance evaluation write-up](https://www.anthropic.com/engineering/AI-resistant-technical-evaluations). The task evolved from a simulated accelerator with profiling, scratchpad memory, SIMD, VLIW, and multicore reasoning into newer constrained programming puzzles. Anthropic also released the [original challenge](https://github.com/anthropics/original_performance_takehome).

The lesson is not that every candidate needs accelerator optimization. It is that some teams build assessments from actual work and expect tooling, measurement, and correctness under a longer horizon.

The current [Fellows posting](https://job-boards.greenhouse.io/anthropic/jobs/5023394008) confirms an application and reference check, technical assessments and interviews, and a research discussion. Workstreams include scalable oversight, AI control, mechanistic interpretability, security, ML systems, RL, synthetic pipelines, and model training diagnostics.

## Google DeepMind

The [official careers process](https://deepmind.google/careers/) publishes a 30-minute recruiter conversation, 2 or 3 skills interviews, and final interviews with team leads and leadership. Exact steps remain role-specific.

A [June 2026 Research Engineer synthesis](https://igotanoffer.com/en/advice/google-deepmind-research-engineer-interview) reports executable coding, mathematical ML fundamentals, ML design, and final team discussions in many loops. This matches the enduring Research Engineer bar: code that runs, derivations that survive follow-up, and experiments grounded in first principles.

The site deliberately does not become a generic Google algorithms curriculum. Candidates should use a dedicated source for that requirement and use mlmentorship for the ML depth layered on top.

## Meta

Meta's [official hiring FAQ](https://www.metacareers.com/hiring-process/) now says many interviews include an AI assistant and candidates are expected to use it. The assistant is built into CoderPad and includes Claude, ChatGPT, Gemini, and Meta models. Meta also says AI-native design uses Mermaid Markdown with the same assistant.

This is not a shortcut round. The candidate must understand unfamiliar code, direct the model, inspect output, debug, test, and explain. Outside AI tools remain unauthorized.

The format rewards engineers who treat an agent like a fast but untrusted collaborator. Prompt count is not the metric. Controlled progress is.

## xAI

The [official xAI careers page](https://x.ai/careers) says technical staff review applications, a short screen includes technical questions, and later sessions go deep on technical expertise. The application asks for exceptional work in 100 words or fewer.

That is enough to justify proof-of-work and project-defense preparation. It is not enough to justify publishing a stable list of xAI ML rounds. When public evidence is thin, say so.

## The preparation decision tree

Ask the recruiter six questions:

1. What are the exact round names and durations?
2. Is implementation blank-file, codebase, notebook, take-home, or trace-based?
3. Which AI tools are allowed in each round?
4. Is there a presentation, research discussion, or artifact review?
5. Which domain and team will technical depth target?
6. What does each round produce or score?

Then map format to drill:

- unfamiliar codebase plus AI: [agentic codebase lab](/prep/labs/agentic-codebase/);
- LLM training debugging: [broken training lab](/prep/labs/broken-training/);
- research notebook or API: [black-box lab](/prep/labs/black-box/);
- project presentation: [presentation protocol](/prep/presentation/);
- performance trace: [accelerator lab](/prep/labs/accelerator/);
- ML primitives: [implementation set](/prep/labs/implementation/);
- oral math: [math lab](/prep/labs/math-oral/);
- values or mission: [values packet](/prep/values/).

## What not to do

Do not complete every lab "just in case." Do not memorize alleged leaked questions. Do not assume AI permission. Do not confuse the company's research agenda with your exact interview. Do not let generic coding consume the ML judgment and work-sample practice the confirmed loop requires.

The process will change again. A maintainable preparation system starts from format evidence and transfers across prompts.

*Related: [frontier-lab process matrix](/prep/frontier-labs/), [role-specific paths](/prep/role-paths/), and [executable labs](/prep/labs/).*

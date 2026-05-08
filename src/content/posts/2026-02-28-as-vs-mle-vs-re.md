---
title: "Applied Scientist vs MLE vs Research Engineer: what these roles actually do"
description: "The role taxonomy is confusing because companies use the same titles to mean different things. Here's the actual decomposition, and which one you should target."
date: "2026-02-28"
draft: false
tags: ["guides"]
category: "guides"
---


**Pick the team, not the title.** Companies use Research Scientist, Applied Scientist, Research Engineer, and ML Engineer to mean different things. The actual day-to-day work is what matters for prep, fit, and trajectory.

## The basic decomposition

ML work decomposes into roughly four activities:

1. **Research**: framing novel problems, reading literature, designing experiments, writing papers
2. **Modeling**: data preparation, training models, hyperparameter tuning, evaluation
3. **Engineering**: serving infrastructure, training infrastructure, ops, scaling
4. **Product**: talking to users, defining what to build, owning end-to-end outcomes

Different roles weight these differently:

| Role | Research | Modeling | Engineering | Product |
|---|---|---|---|---|
| **Research Scientist (RS)** | 60% | 30% | 5% | 5% |
| **Applied Scientist (AS)** | 20% | 40% | 20% | 20% |
| **Research Engineer (RE)** | 25% | 30% | 40% | 5% |
| **ML Engineer (MLE)** | 5% | 30% | 50% | 15% |
| **Software Engineer (ML adjacent)** | 0% | 10% | 80% | 10% |

These percentages are rough averages.

## Role-by-role, in plain language

### Research Scientist

Think: a tenure-track position with better pay, no teaching, and no grant writing.

You're judged primarily by the *quality* of your scientific output: papers at top venues, citation impact, novel ideas that the field absorbs. You typically don't ship product directly; instead you publish, your ideas get incorporated into other teams' systems, and you're rewarded for influence.

Where it exists: foundation labs (OpenAI, Anthropic, DeepMind, Google Brain remnants, FAIR), research-heavy industry labs (MSR, IBM Research, NVIDIA Research), some startups.

Who it suits: People who genuinely love research, are willing to work on long-time-horizon problems, and don't mind not seeing their work in product. New PhDs from top labs sometimes start here; mid-career folks usually transition in from AS or RE roles after building a track record.

How competitive: Extremely. RS roles at top labs are *more* competitive than tenure-track CS faculty positions in many subfields.

### Applied Scientist

Think: a hybrid scientist + engineer + product partner. The "T-shaped" role.

You take open-ended problems from the business or product, scope them into ML problems, design and train models, work with engineers to deploy them, and own the outcome. You publish if the work is novel, but publishing is bonus, not the metric.

Where it exists: Amazon (the canonical AS role), Microsoft, Apple, Meta, Google, every big-tech ML org, well-funded LLM startups.

Who it suits: People who like ML *and* like seeing their work in production. PhDs who want their science to ship. Engineers who want to do science. The role is unusually broad, you're expected to do *some* research, *some* engineering, *some* product thinking. None of those at world-class level individually, but all at competent level together.

How competitive: Competitive but more abundant than RS. The bar is "can you scope, build, and ship." Strong PhDs with even modest production experience can land senior AS roles.

The most common target for PhDs transitioning to industry.

### Research Engineer

Think: an engineer who understands the science deeply enough to be useful in research-adjacent work.

REs are the people who scale up training runs, build the infrastructure that lets RS folks run experiments, implement papers in production-ready code, and bridge the gap between "the paper says it works" and "it works on our infrastructure."

Where it exists: Foundation labs (where the role is sometimes the *most valuable* on the team, you can't run a frontier model without REs), DeepMind, OpenAI, Anthropic. Less common at FAANG-style product orgs, where the work tends to be split between AS and MLE.

Who it suits: Strong engineers who like ML and the systems side. Often very strong CS undergrads or master's, sometimes PhDs in ML systems areas. Very compensated very well at frontier labs.

How competitive: Very. The combination of strong systems skills and ML literacy is rare.

### ML Engineer

Think: a software engineer specialized in ML systems.

You build the platforms and infrastructure that ML runs on: training pipelines, feature stores, model serving, monitoring, experimentation infrastructure, deployment tooling. You sometimes train models, but it's not your primary value. Your job is to make ML *reliable and scalable*.

Where it exists: Everywhere. Every company doing serious ML has MLEs.

Who it suits: Software engineers who want to specialize in ML systems. People who care more about reliability and scale than about novelty.

How competitive: Bar is mostly "are you a strong software engineer with enough ML literacy to be useful?" The ML literacy bar varies hugely by team, some teams want senior researchers; others want people who could swap to a non-ML SWE role tomorrow.

### Software Engineer (ML adjacent)

Think: a SWE who happens to work on a team that ships ML.

You may not train models at all; you build the application code, the APIs, the data pipelines, the user-facing features. You need to understand enough about ML to integrate it sensibly, but you're judged as a software engineer first.

This is increasingly the largest category at most companies: most of the code around an ML model is regular software.

## The decision tree

If you're trying to pick a role to target:

- **Do you have a PhD or equivalent research experience, and want to stay close to the science?** → Applied Scientist (with Research Scientist as a stretch goal at top labs).
- **Are you a strong engineer with deep ML interest, especially infra/systems?** → Research Engineer at a frontier lab if you can land it; otherwise senior MLE at a big company.
- **Are you an engineer who wants to work near ML but doesn't want to live in modeling?** → ML Engineer.
- **Are you trying to break into ML from a software background and don't have research experience?** → Start as a software engineer on an ML team, or as MLE if your team is willing to train you on the modeling side.

## Compensation patterns (US, mid-2026)

Roughly, at the same level (e.g., L5 / Senior at a FAANG):

- RS: highest base, often equity-heavy at frontier labs (can be 2&times;-5&times; FAANG total comp at OpenAI/Anthropic/xAI for senior roles)
- AS: high base, decent equity, comparable total to senior MLE at FAANG
- RE: at frontier labs, can rival or exceed RS comp; at FAANG, typically slightly above MLE
- MLE: standard senior SWE comp, sometimes with modest premium for the ML specialty
- SWE on ML team: standard senior SWE comp

The frontier labs (OpenAI, Anthropic, xAI, Google DeepMind) pay roughly 2&times; FAANG comp for equivalent roles in 2026. This is the largest comp delta in the industry. If you can get into one of these, the comp alone often justifies the move.

## What changes between levels

Across all roles, the senior trajectory looks similar:

- **Junior** (L4): execute well within a defined scope.
- **Mid** (L5): own a project area; make autonomous technical decisions.
- **Senior** (L6 / Staff): own a problem space; influence multiple teams; set technical direction; mentor.
- **Principal** (L7): cross-org technical leadership; long-time-horizon strategic decisions.

The difference between roles diminishes at higher levels. A Principal RS, AS, and RE all spend a lot of their time on the same things: cross-team influence, long-term technical direction, mentorship, strategic prioritization. They differ in *where* they apply that influence (research agendas vs. product roadmaps vs. infrastructure decisions) but not in the nature of the work.

## How to interview for each

The interviews are subtly different even when the formats look the same:

- **RS**: deep technical depth on your research area, paper review (sometimes literally walking through a paper you didn't write and critiquing it), open-ended research-design questions. Project deep-dive is the most heavily weighted round.
- **AS**: ML breadth, ML system design, behavioral, project deep-dive, sometimes coding. The "ship-it engineering" signal is often the differentiator vs. RS-level candidates.
- **RE**: coding-heavy (often LeetCode-style), systems-design heavy, ML breadth, project deep-dive on infrastructure work.
- **MLE**: coding-heavy, ML system design, ML breadth (lighter than AS), behavioral.

If you're targeting AS but interviewing like RS, you may bomb the system design or the engineering screen. If you're targeting MLE but interviewing like AS, you may not have prepared enough on coding. Read the job description carefully and ask the recruiter for the round breakdown.

## A note on title inflation

Titles are unreliable across companies. "Applied Scientist" at Amazon is a different role than "Applied Scientist" at most consulting firms. "Research Engineer" at OpenAI is roughly equivalent to "Member of Technical Staff" elsewhere, which is roughly equivalent to AS at FAANG. Check the actual work, not the title.

The signals that matter more than the title:

- Does the team ship to real users?
- What's the team's track record (papers, products, model releases)?
- What's the day-to-day mix of research / modeling / engineering / product?
- Does the role you'd hold do the work you want to do?

Two people with the same title at the same company often do very different jobs depending on the team. Pick the team, not the title.

---

*Related: [The 5 things every applied scientist interview is testing for](/guides/five-things-as-interview-tests/), [What L5 vs L6 actually means at FAANG ML](/guides/l5-vs-l6-faang-ml/).*

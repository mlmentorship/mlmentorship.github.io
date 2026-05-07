---
title: "What L5 vs L6 actually means at FAANG ML"
description: "Level lines are mostly invisible from the outside but sharp on the inside. A practical calibration of L4 through L7 in ML / Applied Scientist tracks."
date: "2026-05-07"
draft: false
tags: ["essays"]
category: "essays"
---


Most candidates get under-leveled because they describe their work accurately but for the wrong rubric. The level you're hired at decides scope, comp, trajectory, and how much weight your ideas carry in rooms you're not in.

The calibration below is specific to ML / Applied Scientist tracks at large tech companies (FAANG and equivalent), where the levels roughly map as:

- **L4** = SDE-2 / new MS or strong new PhD
- **L5** = Senior / mid-career IC, the company "default" level
- **L6** = Staff / "fully autonomous," the level most ICs cap at
- **L7** = Principal / scope = a whole org's technical direction

For each level: what they actually do, what they say in interviews that gives them away, and the most common reasons people get down-leveled.

## L4: "Tell me what to do and I'll do it well"

**Scope:** A single project, ~3 months. Bounded. Tech lead picks the approach.

**What L4 looks like in interviews:**
- Strong on fundamentals; can derive backprop, knows the ML canon.
- Implements clean code, debugs efficiently.
- Talks about *what they did*, less about *why*.
- Defers on architecture choice ("my mentor suggested...", "the team uses...").
- Eval is "I split 80/20 and computed accuracy."

**Why this is fine for L4:** execution within direction.

**Why this gets you down-leveled from L5:** at L5, "would I let this own a project?" requires more than execution.

## L5: "Give me a problem and I'll figure it out"

**Scope:** A single project area, 6-12 months. You pick the approach. Someone else picks the problem.

**What L5 looks like in interviews:**
- Scopes ambiguous problems before diving in: "what does the user actually want here?"
- Has opinions about trade-offs and can articulate them.
- Tells stories with a clear "I" thread, "I noticed X, I tried Y, that failed because Z, so I tried W."
- Eval includes online metrics, slice analysis, and at least one mention of "what would make me hold the launch."
- Can talk about a project that *failed* and what they learned.

**The L5 trap:** technically L6-strong but speaking like L5. You did the work but say "the team did X" instead of "I drove X." Use "I" statements.

**Down-leveled to L5:** at L6, "would I let this own team direction?" requires more than project ownership.

## L6: "Tell me the goal and I'll figure out the strategy"

**Scope:** A team or sub-org area, 12+ months. You pick the problems *and* the approaches. You influence other teams. Your judgment is trusted enough that disagreeing with you requires evidence.

**What L6 looks like in interviews:**
- Reframes the question. Asked "how would you build X?" they often respond "before I answer, I want to push back on whether X is what we should be building, here's an alternative framing." (When done well, this is the strongest L6 signal there is. When done poorly, it's L5 trying too hard.)
- Talks about *systems*, not models. "We had three teams shipping similar things, so I built a shared platform for X" is a more L6 story than "my model got 4% better."
- Has at least one story about *making a wrong technical bet and recovering from it*.
- Has at least one story about *changing someone else's mind*, with details on how.
- Has at least one story about *deciding not to do something*, killed projects count for more than launched ones.
- Eval discussion includes the offline-online correlation, second-order effects, and what they'd report up the chain.

**The L6 trap:** sounding too senior. Senior interviewers smell BS. If your answer to every question is "well, at the strategic level..." you sound like a manager who hasn't shipped in 3 years. L6 is *both* hands-on technical and strategic. You need stories at both levels.

## L7: "Tell me the constraint and I'll find the strategy nobody saw"

**Scope:** Multiple orgs, multi-year. You set the technical direction that hundreds of people work within.

L7 calibration is harder to summarize publicly because the bar varies more across companies. Common patterns:

- L7 candidates have built something that *exists in the world*, a system, an open-source project, a research line, that other people use without knowing them.
- They talk about decisions in terms of multi-year industry trends, not next-quarter metrics.
- The interview is less about whether they can answer questions and more about whether the interviewer learns something.

If you're at this level, this essay isn't for you.

## The single most common down-leveling mistake

After watching many senior loops, the #1 reason strong candidates get down-leveled is:

**Underclaiming.**

Specifically: telling stories where the work was clearly L6 (designed a re-calibration system that several teams adopted) but framing them at L5 ("the team had a calibration issue and I helped fix it"). The interviewer writes down what you said, not what was true.

This is especially common among:
- People from research backgrounds (trained to credit collaborators)
- People from cultures with strong humility norms
- Women and underrepresented minorities (well-documented effect)
- Anyone whose prior team had a strong consensus culture

The fix is uncomfortable but mechanical: rewrite every story in your prep using "I" verbs, with "we" only when there were genuinely no individual contributions you can name. If you can't think of any, the story isn't strong enough, pick a different one.

## The single most common over-leveling mistake

The mirror image. Telling stories where the work was clearly L4 ("I implemented the team's design and it shipped") but framing them at L5 with vague language ("I drove X to shipping").

Senior interviewers ask follow-ups: "what was the specific decision you made?" "who else worked on it and what did they own?" "what would you do differently?" If your answers thin out, you're caught. The interview is over and you may have just damaged your reputation.

Don't claim what you can't substantiate at three layers of follow-up. The way through is to be specific and bounded: name what you owned, name what others owned, name the decisions that were yours. Specificity reads as senior; vagueness reads as inflation.

## How to use this for your loop

A week before the interview, pick a target level. Be honest. Then read your top 5 stories aloud and ask:

1. **Scope**: does the scope of this story match the target level? (Single project = L4-L5; team area = L5-L6; cross-team = L6-L7.)
2. **Authority**: in this story, did I pick the *what*, the *how*, or just executed?
3. **Recovery**: is there a moment where something went wrong and I drove the recovery?
4. **Influence**: did I change anyone else's mind or work?
5. **"I" verbs**: am I using passive voice or first-person plural where I should be using "I"?

Stories that don't pass these checks are L4-L5 stories. That's fine, but they shouldn't be your top stories if you're targeting L6.

## A word on company variation

Levels do not transfer cleanly. Amazon L6 is not Google L6 is not Meta E6 is not Anthropic Member of Technical Staff. The expectations described above are roughly consistent in spirit, but the level numbers are not. Read the role description carefully, ask the recruiter for the level rubric (most companies will share it), and calibrate to the specific role.

The goal is not "be L6." The goal is to be hired at the level that matches the work you actually do, with comp and scope that match the impact you actually want.

---

*Related: [The 5 things every applied scientist interview is testing for](/essays/five-things-as-interview-tests/). For calibration on which level your stories actually land at, [a mock interview](/interview/) is the fastest way to find out.*

---
title: "How would you do cold-start for a new user?"
description: "Cold-start is solved by combining minimal explicit signal, demographic and contextual fallbacks, and aggressive exploration in the first few sessions."
date: "2025-11-30"
draft: false
tags: ["questions"]
category: "questions"
---


> *Asked in: recsys interviews. The classic open-ended sub-question.*

The L4 candidate proposes "use popularity." The L6 candidate stages a transition from generic to personalized signal across the user's first several sessions.

## What an L4 answer sounds like

> "Show them popular items until they have some history, then switch to personalized."

The right idea, no nuance. Misses how to make the transition smooth, what intermediate signals to use, and how to bootstrap quickly.

## What an L5 answer sounds like

> "Cold-start has three sources of signal, used in sequence:
>
> 1. **Onboarding signal**: ask the user explicitly. Pick 5 favorite artists / genres / categories. Cheap to collect, surprisingly good signal. Many products skip this; they shouldn't.
>
> 2. **Contextual fallbacks**: time, location, device, referral source, demographic if available. A user signing up at 9am on weekday mobile has different intent than 11pm Friday on TV.
>
> 3. **Exploration in early sessions**: explicitly diversify recommendations to gather signal across categories quickly. A bandit-style approach (epsilon-greedy or Thompson sampling on the candidate distribution) bootstraps faster than pure exploitation.
>
> Transition: at session 1, almost all signal is from (1) and (2). After ~5-10 interactions, the personalization model starts to dominate. Hard threshold, gradual blend, or a learned mixing weight depending on signal volume."

This is L5. Three signals, sequenced over user-lifetime.

<!-- visual:cold-start-signal-handoff -->
<figure class="learning-figure" aria-labelledby="cold-start-handoff-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="cold-start-handoff-title">How does a recommender hand control from fallback signals to learned user behavior?</p>
	<div class="visual-grid--two" role="group" aria-label="Two stages of the cold-start signal handoff from fallback evidence to observed user behavior">
		<section class="visual-panel">
			<h4>1 · BEFORE BEHAVIOR EXISTS</h4>
			<p><strong>Start with available evidence</strong><br />Onboarding choices + request context + item content + safe population priors.</p>
			<p><strong>Compose the first slate</strong><br />Anchor it with likely-relevant items, then reserve some positions for diverse exploration.</p>
			<p><strong>What is missing</strong><br />There are no clicks, skips, saves, or watch-time observations for this user yet.</p>
		</section>
		<section class="visual-panel">
			<h4>2 · EACH RESPONSE UPDATES THE MIX</h4>
			<p><strong>Observe</strong><br />Exploratory impressions produce clicks, skips, saves, and dwell-time evidence.</p>
			<p><strong>Shift influence gradually</strong><br />Observed behavior gains influence; onboarding, context, and population fallbacks recede as confidence grows.</p>
			<p><strong>Keep learning</strong><br />Do not turn exploration off: preferences, intent, and the item catalog can still change.</p>
		</section>
	</div>
	<figcaption><strong>Read it this way:</strong> begin on the left with signals that exist at sign-up. The slate intentionally gathers feedback; carry that feedback to the right, where personalized behavior gains influence rather than switching on at one fixed interaction count. Exploration stays in the mix so the system can keep learning. Original schematic informed by <a href="https://doi.org/10.1145/564376.564421">Schein et al. on cold-start recommendation</a> and <a href="https://arxiv.org/abs/1003.0146">Li et al. on contextual-bandit feedback</a>.</figcaption>
</figure>

## What an L6 answer adds

> "...practical things:
>
> **The first session matters disproportionately.** Users decide whether to come back based on early experience. Cold-start that surfaces obvious popular hits is safe but boring; that loses long-term retention. The right balance: anchor on hits, sprinkle exploration.
>
> **Side information beats interaction signal at cold-start.** Item content (genre, description, audio embedding for music, visual embedding for video) lets you recommend before any user interaction with that item. Two-tower models that use both content and collaborative signal handle this naturally.
>
> **Synthetic warm-up via similar users.** A new user signing up with profile X can be matched to existing users with similar profiles; recommendations bootstrapped from those users' aggregated history. Useful when explicit onboarding signal is weak.
>
> **Multi-armed bandits for shelf composition** in early sessions. Try several recommendation strategies (popular, trending, content-based, demographic match) and use bandit logic to converge on what works for this user fast.
>
> **Cold-start as a metric, not just a feature.** Track 'time to good recommendations' (sessions until a personalized model performs well for a new user) as a release-gating metric. New cold-start improvements should reduce this number.
>
> **The dual problem (cold items)** is just as important. New items have no engagement signal; they need a content-based candidate source and an explicit boost in the first hours / days to gather signal."

## Tells that get you a strong-hire vote

- You name **three signals** (onboarding, context, exploration) and sequence them.
- You bring up **content-based features** for early personalization.
- You mention **bandits** for shelf-level exploration.
- You name **cold-start as a metric** to track over releases.
- You connect to the **dual cold-item problem**.

## Tells that get you down-leveled

- "Just show popular items."
- No onboarding-signal collection.
- No mention of exploration.
- Only solving the user side; ignoring cold-items.

## Common follow-up

"How would you measure how well your cold-start works?"

The L6 answer:

> "Three metrics: (1) Day-N retention for new users (do they come back), (2) time to first 'good' recommendation (sessions until a personalized model meets quality threshold for that user), (3) per-cohort engagement on first session vs same cohort at session 10. A/B cold-start changes against the previous cold-start strategy on new users only; the population is small so power matters and tests need to run longer than usual."

---

*Related: [Design YouTube's recommender](/questions/design-youtube-recommender/), [Design Spotify's homepage](/questions/design-spotify-homepage/), [System design case study: building personalized search ranking](/guides/personalized-search-ranking/).*

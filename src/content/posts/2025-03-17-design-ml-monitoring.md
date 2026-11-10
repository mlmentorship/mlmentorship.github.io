---
title: "Design ML monitoring"
description: "Separate service health, input contracts, model behavior, mature outcomes and causal comparisons; attach each alert to evidence and an owner."
date: "2025-03-17"
draft: false
tags: ["questions"]
category: "questions"
---


Design monitoring and an incident response when a model service looks healthy but its inputs and scores have changed.

## Problem

**Context.** An online model uses timestamped session features, while its outcome labels require a seven-day observation window.

**Task.** Specify signals, immediate actions, and later comparisons for the incident below, including owners and evidence for recovery.

**Constraints.** Distinguish service availability, feature freshness, distribution changes, and measured outcome quality.
Account for missing label joins and important slices; unavailable mature labels cannot establish current quality.

At 12:00, an ML endpoint meets its latency and error objectives.
Its session-feature snapshot has `as_of=11:40`, despite a two-minute maximum-age contract.
Model score distributions also changed.
The outcome definition requires seven days of observation.

Design the signals, immediate response and later comparison needed to decide what happened.
Specify how missing label joins and important slices affect the conclusion.

<span id="what-ml-monitoring-is-for"></span>

## Picture: Different Evidence Arrives at Different Times

The snapshot is twenty minutes old, exceeding the two-minute limit.
That is a data-contract breach now, even though requests return successfully.
The score change is another observable fact, not proof of lower accuracy.
Seven-day outcomes cannot establish current cohort quality before their window matures.

Use the declared fallback for affected requests while the data owner investigates.
It might use approved rules or a model that does not require the stale feature.
Do not replace missing values with zero unless that behavior was validated.
Record affected versions, cohorts, actions and exposure so later outcomes remain interpretable.

<!-- visual:ml-monitoring-observability-layers -->
<figure class="learning-figure editorial-illustration" aria-labelledby="ml-monitoring-observability-layers-edition-title"><p class="visual-title" id="ml-monitoring-observability-layers-edition-title">Act on known contract failures; keep mature quality tied to its covered cohort</p><div class="visual-argument" data-visual-argument="ml-monitoring-observability-layers" data-argument-kind="timeline" data-argument-composition="monitoring-evidence-and-coverage"><svg viewBox="0 0 360 1700" role="img" aria-label="Act on known contract failures; keep mature quality tied to its covered cohort. Separate endpoint health, data contracts, output behavior, and mature quality. A twenty-minute snapshot violates the two-minute limit now; changed scores alone do not prove worse accuracy. Use an approved fallback and retain affected identities. After seven days, evaluate eligible labels and join coverage by slice. Here ninety percent accuracy covers only joined records; missing joins leave full-cohort accuracy between eighty-one and ninety-one percent."><title>Act on known contract failures; keep mature quality tied to its covered cohort</title><desc>Separate endpoint health, data contracts, output behavior, and mature quality. A twenty-minute snapshot violates the two-minute limit now; changed scores alone do not prove worse accuracy. Use an approved fallback and retain affected identities. After seven days, evaluate eligible labels and join coverage by slice. Here ninety percent accuracy covers only joined records; missing joins leave full-cohort accuracy between eighty-one and ninety-one percent.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">At noon: distinct facts need distinct</tspan><tspan x="20" dy="18">responses</tspan></text><circle cx="51" cy="153" r="24" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><text x="51" y="159" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="51" dy="0">ok</tspan></text><text x="96" y="141" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="96" dy="0">Endpoint</tspan></text><text x="96" y="173" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="96" dy="0">Meets service objectives</tspan></text><circle cx="51" cy="263" r="24" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><text x="51" y="269" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="51" dy="0">!</tspan></text><text x="96" y="251" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="96" dy="0">Features</tspan></text><text x="96" y="283" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="96" dy="0">Snapshot from 11:40</tspan></text><circle cx="51" cy="373" r="24" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><text x="51" y="379" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--c-text)"><tspan x="51" dy="0">?</tspan></text><text x="96" y="361" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="96" dy="0">Scores</tspan></text><text x="96" y="393" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="96" dy="0">Distribution changed</tspan></text><text x="20" y="465" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Age 20 minutes; allowed maximum 2</tspan></text><rect x="40" y="524" width="28" height="23" style="fill:var(--viz-output-bg);stroke:var(--viz-output-stroke);stroke-width:1.75"></rect><text x="40" y="572" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="40" dy="0">Contract limit</tspan></text><rect x="40" y="584" width="280" height="23" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.75"></rect><text x="40" y="632" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="40" dy="0">Observed age</tspan></text><text x="20" y="710" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Data owner repairs freshness. Use a</tspan><tspan x="20" dy="18">validated fallback; save versions,</tspan><tspan x="20" dy="18">actions and exposure.</tspan></text><text x="20" y="837" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Quality for this cohort waits seven</tspan><tspan x="20" dy="18">days</tspan></text><path d="M45 933L315 933" style="fill:none;stroke:var(--viz-state-stroke);stroke-width:1.75"></path><circle cx="45" cy="933" r="6" style="fill:var(--viz-input-stroke);stroke:var(--viz-input-stroke);stroke-width:1.75"></circle><circle cx="315" cy="933" r="6" style="fill:var(--viz-state-stroke);stroke:var(--viz-state-stroke);stroke-width:1.75"></circle><text x="45" y="972" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="45" dy="0">Day 0</tspan></text><text x="315" y="972" style="font-size:14px;font-weight:400;text-anchor:end;fill:var(--c-text)"><tspan x="315" dy="0">Day 7</tspan></text><text x="20" y="1052" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Then: 1000 eligible; 900 joined; 810</tspan><tspan x="20" dy="18">correct</tspan></text><circle cx="89" cy="1140" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="109" cy="1140" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="129" cy="1140" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="149" cy="1140" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="169" cy="1140" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="189" cy="1140" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="209" cy="1140" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="229" cy="1140" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="249" cy="1140" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="269" cy="1140" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="89" cy="1158" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="109" cy="1158" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="129" cy="1158" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="149" cy="1158" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="169" cy="1158" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="189" cy="1158" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="209" cy="1158" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="229" cy="1158" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="249" cy="1158" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="269" cy="1158" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="89" cy="1176" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="109" cy="1176" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="129" cy="1176" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="149" cy="1176" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="169" cy="1176" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="189" cy="1176" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="209" cy="1176" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="229" cy="1176" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="249" cy="1176" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="269" cy="1176" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="89" cy="1194" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="109" cy="1194" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="129" cy="1194" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="149" cy="1194" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="169" cy="1194" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="189" cy="1194" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="209" cy="1194" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="229" cy="1194" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="249" cy="1194" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="269" cy="1194" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="89" cy="1212" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="109" cy="1212" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="129" cy="1212" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="149" cy="1212" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="169" cy="1212" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="189" cy="1212" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="209" cy="1212" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="229" cy="1212" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="249" cy="1212" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="269" cy="1212" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="89" cy="1230" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="109" cy="1230" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="129" cy="1230" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="149" cy="1230" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="169" cy="1230" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="189" cy="1230" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="209" cy="1230" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="229" cy="1230" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="249" cy="1230" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="269" cy="1230" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="89" cy="1248" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="109" cy="1248" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="129" cy="1248" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="149" cy="1248" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="169" cy="1248" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="189" cy="1248" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="209" cy="1248" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="229" cy="1248" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="249" cy="1248" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="269" cy="1248" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="89" cy="1266" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="109" cy="1266" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="129" cy="1266" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="149" cy="1266" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="169" cy="1266" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="189" cy="1266" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="209" cy="1266" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="229" cy="1266" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="249" cy="1266" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="269" cy="1266" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="89" cy="1284" r="5" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle><circle cx="109" cy="1284" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><circle cx="129" cy="1284" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><circle cx="149" cy="1284" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><circle cx="169" cy="1284" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><circle cx="189" cy="1284" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><circle cx="209" cy="1284" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><circle cx="229" cy="1284" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><circle cx="249" cy="1284" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><circle cx="269" cy="1284" r="5" style="fill:var(--viz-warning-stroke);stroke:var(--viz-warning-stroke);stroke-width:1.75"></circle><circle cx="89" cy="1302" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><circle cx="109" cy="1302" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><circle cx="129" cy="1302" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><circle cx="149" cy="1302" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><circle cx="169" cy="1302" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><circle cx="189" cy="1302" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><circle cx="209" cy="1302" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><circle cx="229" cy="1302" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><circle cx="249" cy="1302" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><circle cx="269" cy="1302" r="5" style="fill:var(--viz-neutral-bg);stroke:var(--viz-neutral-stroke);stroke-width:1.75"></circle><text x="20" y="1370" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Each dot: ten records. Green correct;</tspan><tspan x="20" dy="18">red wrong; hollow missing joins.</tspan></text><text x="20" y="1484" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Joined accuracy 90%; join coverage</tspan><tspan x="20" dy="18">90%</tspan></text><text x="20" y="1581" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="20" dy="0">Full-cohort bounds: 81 to 91 percent.</tspan><tspan x="20" dy="18">Check missingness by slice before</tspan><tspan x="20" dy="18">claiming recovery.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content"><svg viewBox="0 0 360 425" role="img" aria-labelledby="ml-monitoring-observability-layers-edition-svg-title ml-monitoring-observability-layers-edition-svg-desc" style="width:100%;max-width:34rem;display:block;margin:auto;font-family:var(--font-ui)"><title id="ml-monitoring-observability-layers-edition-svg-title">Freshness can fail now while outcome quality is still unknown</title><desc id="ml-monitoring-observability-layers-edition-svg-desc">At noon, an 11:40 snapshot is twenty minutes old and fails a two-minute freshness contract immediately. Outcome evidence matures after a seven-day window and requires request identity and join coverage. A changed score distribution does not establish changed quality or causal product impact.</desc><text x="180" y="22" style="font-size:12px;font-weight:400;fill:var(--c-text-soft);text-anchor:middle"><tspan x="180" dy="0">Same request, different evidence clocks</tspan></text><text x="18" y="68" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">At 12:00: freshness can fail NOW</tspan></text><g data-editorial-node><rect x="25" y="90" width="310" height="64" rx="3" style="fill:var(--viz-warning-bg);stroke:var(--viz-warning-stroke);stroke-width:1.3"></rect><text x="180" y="117.5" style="font-size:14px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">snapshot 11:40</tspan><tspan x="180" dy="19">age 20 min &gt; 2 min limit</tspan></text></g><text x="18" y="204" style="font-size:14px;font-weight:700;fill:var(--c-text);text-anchor:start"><tspan x="18" dy="0">Later: outcome evidence must mature</tspan></text><path d="M35 245L325 245" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><rect x="35" y="233" width="290" height="24" rx="3" style="fill:var(--viz-state-bg);stroke:var(--viz-state-stroke);stroke-width:1.3;stroke-dasharray:5 3"></rect><text x="35" y="281" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:start"><tspan x="35" dy="0">day 0</tspan></text><text x="325" y="281" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:end"><tspan x="325" dy="0">day 7</tspan></text><text x="180" y="324" style="font-size:13px;font-weight:700;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Mature label + request ID + join coverage</tspan></text><text x="180" y="378" style="font-size:12px;font-weight:400;fill:var(--c-text);text-anchor:middle"><tspan x="180" dy="0">Score drift alone does not establish a quality</tspan><tspan x="180" dy="19">change.</tspan></text></svg><div class="visual-derivation-caption">The stale snapshot breaches a contract immediately. Outcome quality needs mature labels and adequate joins. Score drift alone proves neither quality nor causal impact.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="ml-monitoring-observability-layers"><strong>Check:</strong> Which alert is justified immediately while model quality is still unknown?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="ml-monitoring-observability-layers">Separate endpoint health, data contracts, output behavior, and mature quality. A twenty-minute snapshot violates the two-minute limit now; changed scores alone do not prove worse accuracy. Use an approved fallback and retain affected identities. After seven days, evaluate eligible labels and join coverage by slice. Here ninety percent accuracy covers only joined records; missing joins leave full-cohort accuracy between eighty-one and ninety-one percent. <strong>Could all missing joins belong to one critical slice?</strong></span></figcaption></figure>

<span id="what-an-l5-answer-sounds-like"></span>

## Give Each Signal a Claim and Response

| Layer | Signal | Supported conclusion | Owner and response |
| --- | --- | --- | --- |
| Service | Latency, errors, throughput, resources | Endpoint operating condition | Service owner: capacity/failure runbook |
| Data | Schema, nulls, freshness, distributions | Contract breach or input change | Data owner: repair source; approved consumer fallback |
| Behavior | Score/rank distributions and slices | Outputs changed | ML owner: inspect versions, inputs and applicable baseline |
| Outcome | Mature labels, join coverage, product effects | Observed quality/impact in covered cohort | Product/ML: investigate; causal claim needs valid comparison |
| Cost | Cost/request, utilization, batching | Resource/economic change | Service/platform: evaluate throughput and task-success effects |

Declare which signals are proxies, directly observed outcomes or causal estimates.
An engagement proxy or confidence shift is not a substitute for correctness labels.

```text
PredictionRecord
	prediction_id, request_id, decision_time, cohort_id
	model_version, feature_version, feature_as_of, action_id
	policy_version, exposure_assignment, score, slice_keys
LabelRecord
	prediction_id, target_definition, outcome_time
	available_time, maturity_time, join_status, censoring_reason
Alert
	signal_id, window, baseline, slice, numerator, denominator
	magnitude, uncertainty, severity, owner, runbook, deadline
```

## Check the Denominator Before Trusting Quality

Suppose 1,000 mature label IDs are eligible, but only 900 join to predictions.
Of the joined rows, 810 are correct: reported accuracy is 90%, and join coverage is 90%.
Without resolving the missing rows, full-cohort accuracy could range from 81% to 91%.
A critical slice might account for all 100 missing joins.

Track eligibility, maturity, missingness and coverage by slice alongside quality.
Distinguish source delay, join failure and censoring from an observed negative outcome.

## Change 1: A Campaign Changes Normal Traffic

**New task.** Input distributions shift during a seasonal campaign.

**Picture and change.** Compare appropriate seasonal/context baselines, sample sizes and practical effect sizes.
Inspect contract breaches and mature quality rather than equating a small statistical p-value with degradation.

**Check.** A detectable distribution change can be harmless.
Historical context helps diagnosis but does not excuse a broken freshness limit.

**Use when.** Traffic composition has legitimate causes.

<span id="common-follow-up"></span>

## Change 2: One Failed Job Creates Many Alerts

**New task.** Many feature alerts share one upstream failure.

**Picture and change.** Group by lineage and root cause; separate pages, tickets and dashboard signals.
Retain consequential slice alerts, a named responder and the fallback state.
Record acknowledgment, action and resolution evidence for later threshold review.

**Check.** Silencing a root alert does not repair stale features.
There is no universal useful-alert percentage that validates the monitoring system.

**Use when.** Noise hides actionable incidents.

<span id="tells-that-get-you-down-leveled"></span>

## Failure: The Global Metric Stays Calm

A stable aggregate can hide lost label coverage or worse performance in a critical slice.
Monitor counts, uncertainty and severe failures, not just an average.

## Transfer Check

Can a shadow deployment establish a new model's lift in user outcomes?

### Reasoned Answer

Not generally. Shadow traffic can check integration and compare predictions without changing user exposure.
User effects require actual exposure with a valid experiment or justified identification assumptions.

<!-- learning:core-end -->
<div class="learning-extension" data-learning-extension>

<span id="what-an-l6-answer-adds"></span>

## Select Diagnostics, Not Universal Thresholds

KS tests compare continuous distributions; chi-square tests compare categorical counts under their assumptions.
PSI depends on binning and reference choice; it is not a universal quality test.
Consider dependence, repeated testing, sample size, effect magnitude and historical noise before setting thresholds.

Monitor schema and freshness contracts separately from statistical drift.
Track target distributions, label availability and calibration on supported mature cohorts.
An input shift alone does not establish a changed conditional relationship between inputs and targets.

Predetermine important product and safety slices, then investigate newly affected groups.
Sparse slices need uncertainty and sufficient observation, not automatic reassurance from a quiet test.
Govern sensitive attributes and trace access under the applicable data policy.

<span id="tells-that-get-you-a-strong-hire-vote"></span>

## Connect Observation to Operation

Service and data-contract breaches need explicit objectives, fallback rules and response deadlines.
Behavior alerts require diagnosis; retraining is not a default repair for stale inputs or broken joins.
Record incident ownership, affected versions, containment, recovery evidence and rollback scope.

Audit representative mature samples to evaluate proxies; retain selection and observation limits.
Version baselines, target definitions, thresholds and runbooks with model and feature changes.
Review alert volume, action usefulness, missed incidents and time to resolution without imposing an arbitrary success percentage.

*Related: [multi-team ML platform design](/questions/design-multi-team-ml-platform/), [ML data lineage and versioning](/concepts/ml-data-lineage-versioning/), [delayed and selective labels](/concepts/delayed-labels-selective-labels-feedback-loops/), and [A/B testing for ML systems](/concepts/ab-testing-for-ml/).*

</div>

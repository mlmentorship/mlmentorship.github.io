---
title: "Determinant and volume"
description: "The determinant of a matrix is the signed volume scaling factor of the linear map. Zero determinant means the map collapses dimensions."
date: "2025-12-02"
draft: false
tags: ["concepts"]
category: "concepts"
---

## Summary

A determinant gives the signed area or volume scale of a linear map.
Its magnitude measures the scale. Its sign records orientation.

Start with a unit square. Send its horizontal edge to $(2, 0.5)$ and its vertical edge to $(0.5, 1.25)$.
What area does the square occupy now? What happens if you exchange the two arrows?

## Geometric interpretation

Each column is the image of one basis vector. The two arrows bound the image of the square.
Follow their order in the first two panels. Then compare their span in the last panel.

<!-- visual:determinant-signed-area-collapse -->
<figure class="learning-figure" aria-labelledby="determinant-area-title">
	<p class="visual-kicker">Learning objective</p>
	<p class="visual-title" id="determinant-area-title">Columns scale signed area; fixed mass spreads out, while aligned columns collapse a dimension</p><div class="visual-argument" data-visual-argument="determinant-signed-area-collapse" data-argument-kind="geometry" data-argument-composition="unit-area-mass-and-collapse"><svg viewBox="0 0 360 2040" role="img" aria-label="Columns scale signed area; fixed mass spreads out, while aligned columns collapse a dimension. Columns send the square edges into a parallelogram. Unit area becomes 2.25, while probability mass stays fixed and density falls. Swapping column order reverses orientation, not unsigned area or density. Aligned columns send the square onto a line: area vanishes and distinct inputs can share an output. That collapsed map has no unique inverse or ordinary two-dimensional density."><title>Columns scale signed area; fixed mass spreads out, while aligned columns collapse a dimension</title><desc>Columns send the square edges into a parallelogram. Unit area becomes 2.25, while probability mass stays fixed and density falls. Swapping column order reverses orientation, not unsigned area or density. Aligned columns send the square onto a line: area vanishes and distinct inputs can share an output. That collapsed map has no unique inverse or ordinary two-dimensional density.</desc><text x="20" y="27" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Start: the unit square</tspan></text><path d="M55 220L325 220" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M80 245L80 68" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-mapped-area="1" d="M80 220L152 220L152 148L80 148Z" style="fill:var(--viz-output-bg);stroke:var(--viz-input-stroke);stroke-width:2"></path><g data-picture-preserved-mass-member="0,0" data-picture-mass-share="0.0625"><circle cx="89" cy="211" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,1" data-picture-mass-share="0.0625"><circle cx="107" cy="211" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,2" data-picture-mass-share="0.0625"><circle cx="125" cy="211" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,3" data-picture-mass-share="0.0625"><circle cx="143" cy="211" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,4" data-picture-mass-share="0.0625"><circle cx="89" cy="193" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,5" data-picture-mass-share="0.0625"><circle cx="107" cy="193" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,6" data-picture-mass-share="0.0625"><circle cx="125" cy="193" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,7" data-picture-mass-share="0.0625"><circle cx="143" cy="193" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,8" data-picture-mass-share="0.0625"><circle cx="89" cy="175" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,9" data-picture-mass-share="0.0625"><circle cx="107" cy="175" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,10" data-picture-mass-share="0.0625"><circle cx="125" cy="175" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,11" data-picture-mass-share="0.0625"><circle cx="143" cy="175" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,12" data-picture-mass-share="0.0625"><circle cx="89" cy="157" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,13" data-picture-mass-share="0.0625"><circle cx="107" cy="157" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,14" data-picture-mass-share="0.0625"><circle cx="125" cy="157" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="0,15" data-picture-mass-share="0.0625"><circle cx="143" cy="157" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><path d="M80 220L131.84 220M123.84 224L131.84 220L123.84 216" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5;stroke-linejoin:round"></path><path d="M152 220L152 168.16M156 176.16L152 168.16L148 176.16" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5;stroke-linejoin:round"></path><text x="164" y="246" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="164" dy="0">first</tspan></text><text x="80" y="126" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="80" dy="0">second</tspan></text><text x="20" y="307" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Area: 1; density: 1</tspan></text><text x="20" y="367" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Signed scale: positive 1. Mass: one.</tspan></text><text x="20" y="468" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Map: columns become the square edges</tspan></text><path d="M55 661L325 661" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M80 686L80 509" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-mapped-area="2.25" d="M80 661L224 625L260 535L116 571Z" style="fill:var(--viz-output-bg);stroke:var(--viz-input-stroke);stroke-width:2"></path><g data-picture-preserved-mass-member="1,0" data-picture-mass-share="0.0625"><circle cx="102.5" cy="645.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,1" data-picture-mass-share="0.0625"><circle cx="138.5" cy="636.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,2" data-picture-mass-share="0.0625"><circle cx="174.5" cy="627.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,3" data-picture-mass-share="0.0625"><circle cx="210.5" cy="618.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,4" data-picture-mass-share="0.0625"><circle cx="111.5" cy="622.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,5" data-picture-mass-share="0.0625"><circle cx="147.5" cy="613.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,6" data-picture-mass-share="0.0625"><circle cx="183.5" cy="604.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,7" data-picture-mass-share="0.0625"><circle cx="219.5" cy="595.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,8" data-picture-mass-share="0.0625"><circle cx="120.5" cy="600.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,9" data-picture-mass-share="0.0625"><circle cx="156.5" cy="591.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,10" data-picture-mass-share="0.0625"><circle cx="192.5" cy="582.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,11" data-picture-mass-share="0.0625"><circle cx="228.5" cy="573.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,12" data-picture-mass-share="0.0625"><circle cx="129.5" cy="577.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,13" data-picture-mass-share="0.0625"><circle cx="165.5" cy="568.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,14" data-picture-mass-share="0.0625"><circle cx="201.5" cy="559.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="1,15" data-picture-mass-share="0.0625"><circle cx="237.5" cy="550.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><path d="M80 661L183.68 635.08M176.88900249898268 640.9008550008721L183.68 635.08L174.94871749869202 633.1397149997094" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5;stroke-linejoin:round"></path><path d="M224 625L249.92 560.2M250.6627813527082 569.1133762324986L249.92 560.2L243.23496782562611 566.1422508216657" style="fill:none;stroke:var(--viz-input-stroke);stroke-width:2.5;stroke-linejoin:round"></path><text x="236" y="651" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-input-stroke)"><tspan x="236" dy="0">first</tspan></text><text x="116" y="549" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-input-stroke)"><tspan x="116" dy="0">second</tspan></text><text x="20" y="748" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Area: 2.25; density: 0.4444</tspan></text><text x="20" y="808" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Signed scale: positive 2.25. Mass:</tspan><tspan x="20" dy="18">one.</tspan></text><text x="20" y="909" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Swap columns: same region, opposite</tspan><tspan x="20" dy="18">order</tspan></text><path d="M55 1102L325 1102" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M80 1127L80 950" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-mapped-area="2.25" d="M80 1102L116 1012L260 976L224 1066Z" style="fill:var(--viz-output-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></path><g data-picture-preserved-mass-member="2,0" data-picture-mass-share="0.0625"><circle cx="102.5" cy="1086.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,1" data-picture-mass-share="0.0625"><circle cx="111.5" cy="1063.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,2" data-picture-mass-share="0.0625"><circle cx="120.5" cy="1041.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,3" data-picture-mass-share="0.0625"><circle cx="129.5" cy="1018.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,4" data-picture-mass-share="0.0625"><circle cx="138.5" cy="1077.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,5" data-picture-mass-share="0.0625"><circle cx="147.5" cy="1054.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,6" data-picture-mass-share="0.0625"><circle cx="156.5" cy="1032.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,7" data-picture-mass-share="0.0625"><circle cx="165.5" cy="1009.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,8" data-picture-mass-share="0.0625"><circle cx="174.5" cy="1068.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,9" data-picture-mass-share="0.0625"><circle cx="183.5" cy="1045.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,10" data-picture-mass-share="0.0625"><circle cx="192.5" cy="1023.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,11" data-picture-mass-share="0.0625"><circle cx="201.5" cy="1000.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,12" data-picture-mass-share="0.0625"><circle cx="210.5" cy="1059.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,13" data-picture-mass-share="0.0625"><circle cx="219.5" cy="1036.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,14" data-picture-mass-share="0.0625"><circle cx="228.5" cy="1014.25" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="2,15" data-picture-mass-share="0.0625"><circle cx="237.5" cy="991.75" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><path d="M80 1102L105.92 1037.2M106.66278135270821 1046.1133762324985L105.92 1037.2L99.23496782562614 1043.1422508216656" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5;stroke-linejoin:round"></path><path d="M116 1012L219.68 986.08M212.88900249898268 991.9008550008721L219.68 986.08L210.94871749869202 984.1397149997094" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5;stroke-linejoin:round"></path><text x="72" y="1038" style="font-size:14px;font-weight:700;text-anchor:end;fill:var(--viz-warning-stroke)"><tspan x="72" dy="0">first</tspan></text><text x="236" y="1092" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="236" dy="0">second</tspan></text><text x="20" y="1189" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Area: 2.25; density: 0.4444</tspan></text><text x="20" y="1249" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Signed scale: negative 2.25. Mass:</tspan><tspan x="20" dy="18">one.</tspan></text><text x="20" y="1350" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Align columns: lose one independent</tspan><tspan x="20" dy="18">direction</tspan></text><path d="M55 1543L325 1543" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path d="M80 1568L80 1391" style="fill:none;stroke:var(--viz-neutral-stroke);stroke-width:1.75"></path><path data-picture-mapped-area="0" d="M80 1543L224 1471L296 1435L152 1507Z" style="fill:var(--viz-output-bg);stroke:var(--viz-warning-stroke);stroke-width:2"></path><g data-picture-preserved-mass-member="3,0" data-picture-mass-share="0.0625"><circle cx="107" cy="1529.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,1" data-picture-mass-share="0.0625"><circle cx="143" cy="1511.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,2" data-picture-mass-share="0.0625"><circle cx="179" cy="1493.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,3" data-picture-mass-share="0.0625"><circle cx="215" cy="1475.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,4" data-picture-mass-share="0.0625"><circle cx="125" cy="1520.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,5" data-picture-mass-share="0.0625"><circle cx="161" cy="1502.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,6" data-picture-mass-share="0.0625"><circle cx="197" cy="1484.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,7" data-picture-mass-share="0.0625"><circle cx="233" cy="1466.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,8" data-picture-mass-share="0.0625"><circle cx="143" cy="1511.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,9" data-picture-mass-share="0.0625"><circle cx="179" cy="1493.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,10" data-picture-mass-share="0.0625"><circle cx="215" cy="1475.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,11" data-picture-mass-share="0.0625"><circle cx="251" cy="1457.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,12" data-picture-mass-share="0.0625"><circle cx="161" cy="1502.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,13" data-picture-mass-share="0.0625"><circle cx="197" cy="1484.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,14" data-picture-mass-share="0.0625"><circle cx="233" cy="1466.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><g data-picture-preserved-mass-member="3,15" data-picture-mass-share="0.0625"><circle cx="269" cy="1448.5" r="2.7" style="fill:var(--viz-output-stroke);stroke:var(--viz-output-stroke);stroke-width:1.75"></circle></g><path d="M80 1543L183.68 1491.16M178.3134368540005 1498.3154175279994L183.68 1491.16L174.73572809000083 1491.16" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5;stroke-linejoin:round"></path><path d="M224 1471L275.84 1445.08M270.4734368540005 1452.2354175279993L275.84 1445.08L266.89572809000083 1445.08" style="fill:none;stroke:var(--viz-warning-stroke);stroke-width:2.5;stroke-linejoin:round"></path><text x="236" y="1497" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--viz-warning-stroke)"><tspan x="236" dy="0">first</tspan></text><text x="152" y="1485" style="font-size:14px;font-weight:700;text-anchor:middle;fill:var(--viz-warning-stroke)"><tspan x="152" dy="0">second</tspan></text><text x="20" y="1630" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Area: zero. No unique inverse.</tspan></text><text x="20" y="1690" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">Distinct inputs meet on the same</tspan><tspan x="20" dy="18">line.</tspan></text><text x="20" y="1838" style="font-size:14px;font-weight:400;text-anchor:start;fill:var(--c-text-soft)"><tspan x="20" dy="0">Sixteen fixed marks track conserved</tspan><tspan x="20" dy="18">mass, not sampled observations.</tspan><tspan x="20" dy="18">Larger area means lower density.</tspan></text><text x="20" y="1944" style="font-size:14px;font-weight:700;text-anchor:start;fill:var(--c-text)"><tspan x="20" dy="0">A collapsed image has no ordinary</tspan><tspan x="20" dy="18">two-dimensional density. The regular</tspan><tspan x="20" dy="18">inverse-density rule requires a</tspan><tspan x="20" dy="18">nonsingular map.</tspan></text></svg></div><details class="visual-derivation"><summary>Derivation</summary><div class="visual-derivation-content">
	<div class="visual-panel">
		<svg viewBox="0 0 360 560" role="img" aria-labelledby="determinant-area-svg-title determinant-area-svg-desc">
			<title id="determinant-area-svg-title">Three ordered column pairs with positive, negative, and zero determinant</title>
			<desc id="determinant-area-svg-desc">The first panel shows columns a one equals two comma one half and a two equals one half comma one and one quarter, spanning a parallelogram of area two and one quarter in counterclockwise order, so the determinant is positive two and one quarter. The second panel uses the exact same parallelogram but swaps the column order: a one now points to one half comma one and one quarter and a two points to two comma one half. Its area remains two and one quarter, but the order is clockwise, so the determinant is negative two and one quarter. The third panel has columns b one equals two comma one and b two equals one comma one half. The second is half the first, so both arrows lie on one line, the parallelogram collapses to a segment, and the determinant is zero.</desc>
			<defs>
				<marker id="det-arrow-primary" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-forward" d="M0 0L7 3.5L0 7Z"></path></marker>
				<marker id="det-arrow-secondary" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path class="viz-arrow-backward" d="M0 0L7 3.5L0 7Z"></path></marker>
			</defs>
			<rect class="viz-plot-bg" x="12" y="12" width="336" height="164" rx="4"></rect>
			<text class="viz-callout" x="24" y="34">1 · Positive: area 2.25, orientation preserved</text>
			<path class="viz-gridline" d="M45 148H225 M65 160V48"></path>
			<polygon class="viz-node viz-node--focus" points="65,148 145,128 165,78 85,98"></polygon>
			<path d="M65 148L145 128" class="viz-roc-curve" marker-end="url(#det-arrow-primary)"></path>
			<path d="M65 148L85 98" class="viz-pr-curve" marker-end="url(#det-arrow-secondary)"></path>
			<circle class="viz-operating-point" cx="65" cy="148" r="3"></circle>
			<text class="viz-label" x="139" y="146">a₁ = (2, 0.5)</text>
			<text class="viz-label" x="91" y="92">a₂ = (0.5, 1.25)</text>
			<text class="viz-callout" x="238" y="83">counterclockwise</text>
			<text class="viz-node-value" x="290" y="104">det[a₁ a₂]</text>
			<text class="viz-node-label" x="290" y="129">= +2.25</text>
			<text class="viz-label" x="238" y="154">density = 4/9</text>
			<rect class="viz-plot-bg" x="12" y="190" width="336" height="164" rx="4"></rect>
			<text class="viz-callout" x="24" y="212">2 · Swap columns: same area, opposite sign</text>
			<path class="viz-gridline" d="M45 326H225 M65 338V226"></path>
			<polygon class="viz-node viz-node--focus" points="65,326 145,306 165,256 85,276"></polygon>
			<path d="M65 326L85 276" class="viz-roc-curve" marker-end="url(#det-arrow-primary)"></path>
			<path d="M65 326L145 306" class="viz-pr-curve" marker-end="url(#det-arrow-secondary)"></path>
			<circle class="viz-operating-point" cx="65" cy="326" r="3"></circle>
			<text class="viz-label" x="91" y="270">a₁ = (0.5, 1.25)</text>
			<text class="viz-label" x="139" y="324">a₂ = (2, 0.5)</text>
			<text class="viz-callout" x="238" y="261">clockwise</text>
			<text class="viz-node-value" x="290" y="282">det[a₁ a₂]</text>
			<text class="viz-node-label" x="290" y="307">= −2.25</text>
			<rect class="viz-plot-bg" x="12" y="368" width="336" height="180" rx="4"></rect>
			<text class="viz-callout" x="24" y="390">3 · Dependent columns: one direction is lost</text>
			<path class="viz-gridline" d="M45 522H225 M65 534V414"></path>
			<path d="M65 522L145 482" class="viz-roc-curve" marker-end="url(#det-arrow-primary)"></path>
			<path d="M65 522L105 502" class="viz-pr-curve" marker-end="url(#det-arrow-secondary)"></path>
			<circle class="viz-operating-point" cx="65" cy="522" r="3"></circle>
			<text class="viz-label" x="139" y="476">b₁ = (2, 1)</text>
			<text class="viz-label" x="107" y="516">b₂ = (1, 0.5) = ½b₁</text>
			<text class="viz-callout" x="238" y="447">collapsed segment</text>
			<text class="viz-node-value" x="290" y="468">det[b₁ b₂]</text>
			<text class="viz-node-label" x="290" y="493">= 0</text>
			<text class="viz-label" x="238" y="519">area = 0</text>
			<text class="viz-label" x="238" y="536">not invertible</text>
		</svg>
	</div>
	<div class="visual-derivation-caption"><strong>Read it this way:</strong> swapping the arrows changes their order but leaves the filled region unchanged. Only the determinant sign changes. In the last panel, both arrows lie on one line. There is no two-dimensional area.</div><div class="visual-derivation-caption"> <span class="visual-reader-check" data-reader-check="determinant-signed-area-collapse"><strong>Check:</strong> Why does swapping columns change the determinant sign but not the transformed density?</span></div></div></details><figcaption><strong>Read it this way:</strong> <span data-proof-caption="determinant-signed-area-collapse">Columns send the square edges into a parallelogram. Unit area becomes 2.25, while probability mass stays fixed and density falls. Swapping column order reverses orientation, not unsigned area or density. Aligned columns send the square onto a line: area vanishes and distinct inputs can share an output. That collapsed map has no unique inverse or ordinary two-dimensional density. <strong>Does reversing the column order change the floor area?</strong></span></figcaption>
</figure>
<p class="diagram-source">Original coordinate construction checked against <a href="https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-18-properties-of-determinants/">MIT OpenCourseWare 18.06</a>, the <a href="https://www.deeplearningbook.org/contents/linear_algebra.html">Deep Learning linear algebra chapter</a>, and the open <a href="https://math.libretexts.org/Bookshelves/Linear_Algebra/Interactive_Linear_Algebra_(Margalit_and_Rabinoff)/04%3A_Determinants/4.01%3A_Determinants-_Definition">Interactive Linear Algebra determinant text</a>.</p>

## Work the area

The columns give

$$
A=\begin{pmatrix}2&0.5\\0.5&1.25\end{pmatrix},\qquad
\det(A)=2(1.25)-0.5(0.5)=2.25.
$$

The four corners become $(0,0)$, $(2,0.5)$, $(0.5,1.25)$ and $(2.5,1.75)$.
The filled region has area $9/4$.

You can check that area by sliding each vertical slice downward by $x/4$.
The slide preserves every slice's height and width.
The arrows become $(2,0)$ and $(0.5,1.125)$.
The base is now horizontal: base $2$ times height $1.125$ gives $2.25$.

## Change of variables (probability)

Put uniform probability density $1$ on the original unit square.
The image has $9/4$ times the area, so its density must fall to $4/9$.
The total mass stays $(9/4)(4/9)=1$.

For this map, $Y=AX$ and

$$
A^{-1}=\frac49\begin{pmatrix}1.25&-0.5\\-0.5&2\end{pmatrix}.
$$

Thus $p_Y(y)=4/9$ inside the image and zero outside it.

More generally, let $Y=f(X)$ on a region where $f$ has a differentiable inverse.
Assume its Jacobian is nonsingular there. Then

$$
p_Y(y) = p_X(f^{-1}(y)) \cdot |\det J_{f^{-1}}(y)|
$$

Here $J_{f^{-1}}$ is the inverse map's Jacobian.
The absolute value keeps probability density nonnegative, including when the map reverses orientation.

Normalizing flows use maps whose Jacobian determinants are cheap to compute.
For a triangular Jacobian, multiply its diagonal entries.
If several regular inverse branches reach the same output, add their density contributions.
The single-inverse formula alone does not handle that case.

## Change the map

### Exchange the columns

**New task:** keep the filled region but exchange the two basis images.

**Picture and change:** the second panel exchanges the arrows. The ordered area becomes $-2.25$.
Its magnitude remains $2.25$, so the transformed density remains $4/9$.

**Check:** a negative density would expose a missing absolute value.

**Use when:** tracking orientation or transporting probability mass requires the correct sign convention.

### Double every coordinate

**New task:** replace $A$ with $2A$.

**Picture and change:** both edge lengths double. In two dimensions, area grows by $2^2=4$.
Thus $\det(2A)=9$, and the uniform density becomes $1/9$.
In three dimensions, doubling every coordinate multiplies volume by $2^3=8$.

**Check:** doubling only one column multiplies the determinant by two.
Doubling the whole matrix multiplies it by $2^n$ in $n$ dimensions.

**Use when:** every coordinate unit changes, rather than one basis image alone.

## Properties

For square matrices $A,B\in\mathbb R^{n\times n}$:

- $\det(AB)=\det(A)\det(B)$. Composing maps multiplies their signed volume scales.
- $\det(A^\top)=\det(A)$.
- $\det(A^{-1})=1/\det(A)$ when $A$ is invertible.
- $\det(cA)=c^n\det(A)$.
- $\det(A)=\prod_i\lambda_i$, counting multiplicity. Real matrices can have complex eigenvalues.
- For a triangular matrix, $\det(A)=\prod_i A_{ii}$.
- For an orthogonal matrix $Q$, $|\det(Q)|=1$. Rotations and reflections preserve volume magnitude.

In exact arithmetic, these statements are equivalent:

$$
\det(A)=0
\iff \text{dependent columns}
\iff \text{not invertible}
\iff \text{a zero eigenvalue}.
$$

The last panel uses $B=\begin{pmatrix}2&1\\1&0.5\end{pmatrix}$.
Its second column is half its first. The image lies on a line, and $\det(B)=0$.
There is no ordinary two-dimensional inverse-density formula for this collapsed map.

## Try a different transformation

A reflection is followed by axis scalings of three and two.
Find the determinant and the image density of a unit-density square.

<details>
<summary>Check your answer</summary>

The determinant is $-6$. The image density is $1/6$ on its support.
The reflection changes orientation. The scale magnitudes multiply the area by six.

</details>

<!-- learning:core-end -->

<div data-learning-extension>

## Computing determinants

| Method | Cost | When |
|--------|------|------|
| LU decomposition | $O(n^3)$ | General-purpose; standard library default |
| Triangular: product of diag | $O(n)$ | When matrix is already triangular |
| Eigendecomposition: $\prod \lambda_i$ | $O(n^3)$ | If you need eigenvalues anyway |
| **Log-determinant** for PD matrices | via Cholesky | When you only need $\log \det$ (e.g., Gaussian log-likelihood) |

For a general matrix, use a signed log-determinant.
Track row-pivot signs and the signs of the LU diagonal entries. Sum $\log|U_{ii}|$ for the log-magnitude.
For a positive-definite matrix with $A=LL^\top$, compute $\log\det(A)=2\sum_i\log L_{ii}$.
An explicit determinant can overflow or underflow.

## Common pitfalls

- **Using $\det$ as a proxy for matrix "size."** A nearly singular matrix can have huge entries but tiny $\det$.
- **Computing $\det$ for invertibility tests.** Numerically unstable; use rank or condition number instead.
- **Forgetting the absolute value in change-of-variables formulas.** Densities are non-negative; the Jacobian determinant can be negative.

</div>

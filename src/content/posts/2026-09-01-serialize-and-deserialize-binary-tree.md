---
title: "Serialize and Deserialize Binary Tree"
description: "Convert a tree to text and rebuild the same tree from that text."
date: "2026-09-01"
draft: false
tags: ["coding interview", "data structures"]
category: "questions"
roles: ["MLE", "RE", "AS"]
rounds: ["Coding", "ML implementation"]
difficulty: "Advanced"
priority: "Specialist"
aliases: []
prerequisites: []
---

> Convert a tree to text and rebuild the same tree from that text.

Start with the concrete trace below. It shows the state the algorithm must carry as it runs.

<!-- visual:serialize-and-deserialize-binary-tree-state -->
<figure class="learning-figure coding-visual-figure" aria-labelledby="serialize-and-deserialize-binary-tree-state-title"><p class="visual-kicker">Problem trace</p><p class="visual-title" id="serialize-and-deserialize-binary-tree-state-title">Serialize and Deserialize Binary Tree: Preorder plus explicit null markers preserves both node values and tree shape.</p><div class="coding-visual" data-coding-visual data-coding-mode="trace" data-coding-slug="serialize-and-deserialize-binary-tree" role="group" aria-label="Serialize and Deserialize Binary Tree: Preorder plus explicit null markers preserves both node values and tree shape."><div class="coding-visual-example"><span>Input and goal</span><strong>Convert a tree to text and rebuild the same tree from that text.</strong></div><div class="coding-trace" data-coding-trace><div class="coding-trace-frame" data-coding-frame="0" role="group" aria-label="Visit preorder"><div class="coding-trace-frame-heading"><span>Visit preorder</span><strong>Tree 1 with a right child 2 visits 1, null-left, 2.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node trace-tone-focus"><span>1</span><small>visit 1</small></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node trace-tone-state"><span>-</span><small>null</small></span><span class="coding-trace-tree-node"><span>2</span></span></div></div></div><div class="coding-trace-frame" data-coding-frame="1" hidden role="group" aria-label="Write markers"><div class="coding-trace-frame-heading"><span>Write markers</span><strong>Missing children become # tokens, so the stream is 1,#,2,#,#.</strong></div><div class="coding-trace-array" role="list" aria-label="Array state"><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">1</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">shape marker</span><span class="coding-trace-array-cell">#</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">2</span></span><span class="coding-trace-array-item trace-tone-state" role="listitem"><span class="coding-trace-array-mark">shape marker</span><span class="coding-trace-array-cell">#</span></span><span class="coding-trace-array-item" role="listitem"><span class="coding-trace-array-mark"></span><span class="coding-trace-array-cell">#</span></span></div></div><div class="coding-trace-frame" data-coding-frame="2" hidden role="group" aria-label="Read the same stream"><div class="coding-trace-frame-heading"><span>Read the same stream</span><strong>The decoder consumes tokens in the same preorder and rebuilds the shape.</strong></div><div class="coding-trace-tree" role="group" aria-label="Tree state"><div class="coding-trace-tree-level" data-level="0"><span class="coding-trace-tree-node trace-tone-output"><span>1</span><small>rebuilt</small></span></div><div class="coding-trace-tree-level" data-level="1"><span class="coding-trace-tree-node"><span>-</span></span><span class="coding-trace-tree-node"><span>2</span></span></div></div><div class="coding-trace-meta"><span><b>result</b>same tree</span></div></div><div class="coding-trace-controls" data-coding-controls hidden><div class="coding-trace-control-buttons"><button type="button" data-coding-previous disabled><span aria-hidden="true">&larr;</span><span>Previous</span></button><button type="button" data-coding-play><span aria-hidden="true">&#9654;</span><span data-coding-play-label>Play trace</span></button><button type="button" data-coding-next><span>Next</span><span aria-hidden="true">&rarr;</span></button></div><output data-coding-progress>Step 1 of 3</output></div><div class="coding-trace-timeline" data-coding-timeline hidden role="group" aria-label="Trace steps"><button type="button" data-coding-frame-button="0" aria-current="step"><span>1</span><strong>Visit preorder</strong></button><button type="button" data-coding-frame-button="1"><span>2</span><strong>Write markers</strong></button><button type="button" data-coding-frame-button="2"><span>3</span><strong>Read the same stream</strong></button></div><p class="coding-trace-status sr-only" data-coding-status aria-live="polite"></p></div><p class="coding-visual-invariant"><span>Why this works</span>Preorder plus explicit null markers preserves both node values and tree shape.</p></div><figcaption><strong>Read it this way:</strong> Tree 1 with a right child 2 visits 1, null-left, 2. Step through the frames to watch the state change. The last frame shows the answer or the stopping condition.</figcaption></figure>

**Pattern:** Preorder DFS with markers for missing children.

**Simple idea:** Preorder alone is not enough because shapes can differ. Save `#` for every
missing child. The decoder follows the same order.

```python
class TreeCodec:
   def serialize(self, root: TreeNode | None) -> str:
      values: list[str] = []

      def visit(node: TreeNode | None) -> None:
         if node is None:
            values.append("#")
            return
         values.append(str(node.val))
         visit(node.left)
         visit(node.right)

      visit(root)
      return ",".join(values)

   def deserialize(self, data: str) -> TreeNode | None:
      values = iter(data.split(","))

      def build() -> TreeNode | None:
         value = next(values)
         if value == "#":
            return None
         node = TreeNode(int(value))
         node.left = build()
         node.right = build()
         return node

      return build()
```

**Cost:** $O(n)$ time and $O(n)$ space.

The platform supplies `TreeNode` with `val`, `left`, and `right`; this snippet assumes that definition.
